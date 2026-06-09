"""HTTP service/client helpers for centralized RWR++ tool execution."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .schemas import ToolAction, ToolObservation, ToolObservationStatus
from .tools import ToolExecutionError, ToolExecutionResult


RWR_HPC_SERVICE_TOOL_NAMES = frozenset(
    {
        "rwr",
        "rwr_loe",
        "shortest_paths",
        "get_rank",
        "get_distance",
        "get_spearman",
        "get_pearson",
        "get_dot_similarity",
        "get_rank_vector_summary",
        "get_encoding_summary",
        "get_gene_layers",
        "get_nodes_by_layer",
        "get_layer_stats",
        "get_path_layer_counts",
        "get_component_summary",
        "get_seed_essentiality",
        "get_layer_ablation",
        "get_node_perturbation",
        "rwr_multiplex",
        "rwr_monoplex",
    }
)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[max(0, min(index, len(ordered) - 1))]


class RwrHpcServiceMetrics:
    """Small in-memory metrics accumulator for one RWR++ service process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_tool: dict[str, dict[str, Any]] = defaultdict(self._new_tool_metrics)
        self.started_at = time.time()

    @staticmethod
    def _new_tool_metrics() -> dict[str, Any]:
        return {
            "request_count": 0,
            "error_count": 0,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "cache_unknown_count": 0,
            "queue_wait_seconds_total": 0.0,
            "service_time_seconds_total": 0.0,
            "queue_wait_seconds_values": [],
            "service_time_seconds_values": [],
        }

    def record(
        self,
        *,
        tool_name: str,
        queue_wait_seconds: float,
        service_time_seconds: float,
        cache_hit: bool | None,
        error: bool,
    ) -> None:
        with self._lock:
            bucket = self._by_tool[tool_name]
            bucket["request_count"] += 1
            bucket["error_count"] += int(error)
            if cache_hit is True:
                bucket["cache_hit_count"] += 1
            elif cache_hit is False:
                bucket["cache_miss_count"] += 1
            else:
                bucket["cache_unknown_count"] += 1
            bucket["queue_wait_seconds_total"] += queue_wait_seconds
            bucket["service_time_seconds_total"] += service_time_seconds
            bucket["queue_wait_seconds_values"].append(queue_wait_seconds)
            bucket["service_time_seconds_values"].append(service_time_seconds)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            by_tool: dict[str, dict[str, Any]] = {}
            for tool_name, raw in self._by_tool.items():
                request_count = int(raw["request_count"])
                cache_known = int(raw["cache_hit_count"]) + int(raw["cache_miss_count"])
                by_tool[tool_name] = {
                    "request_count": request_count,
                    "error_count": int(raw["error_count"]),
                    "cache_hit_count": int(raw["cache_hit_count"]),
                    "cache_miss_count": int(raw["cache_miss_count"]),
                    "cache_unknown_count": int(raw["cache_unknown_count"]),
                    "cache_hit_rate": (
                        raw["cache_hit_count"] / cache_known if cache_known else None
                    ),
                    "queue_wait_seconds_mean": (
                        raw["queue_wait_seconds_total"] / request_count
                        if request_count
                        else None
                    ),
                    "queue_wait_seconds_p95": _percentile(raw["queue_wait_seconds_values"], 0.95),
                    "queue_wait_seconds_max": max(raw["queue_wait_seconds_values"], default=None),
                    "service_time_seconds_mean": (
                        raw["service_time_seconds_total"] / request_count
                        if request_count
                        else None
                    ),
                    "service_time_seconds_p95": _percentile(raw["service_time_seconds_values"], 0.95),
                    "service_time_seconds_max": max(raw["service_time_seconds_values"], default=None),
                }
            return {
                "started_at_unix": self.started_at,
                "uptime_seconds": time.time() - self.started_at,
                "total_request_count": sum(item["request_count"] for item in by_tool.values()),
                "total_error_count": sum(item["error_count"] for item in by_tool.values()),
                "tools": by_tool,
            }


class RwrHpcServiceState:
    """Serialize RWR++ tool execution and record queue/service metrics."""

    def __init__(self, executor: Callable[[ToolAction], ToolObservation]) -> None:
        self.executor = executor
        self.metrics = RwrHpcServiceMetrics()
        self._executor_lock = threading.Lock()

    def execute(self, tool_action: ToolAction) -> ToolObservation:
        if tool_action.tool_name not in RWR_HPC_SERVICE_TOOL_NAMES:
            return ToolObservation(
                status=ToolObservationStatus.INVALID,
                provenance={"tool_name": tool_action.tool_name, "service_rejected": True},
                call_id=tool_action.call_id,
                error=f"Unsupported RWR++ service tool: {tool_action.tool_name}",
            )

        wait_start = time.time()
        with self._executor_lock:
            service_start = time.time()
            queue_wait_seconds = service_start - wait_start
            observation = self.executor(tool_action)
            service_time_seconds = time.time() - service_start

        cache_hit = observation.provenance.get("cache_hit")
        self.metrics.record(
            tool_name=tool_action.tool_name,
            queue_wait_seconds=queue_wait_seconds,
            service_time_seconds=service_time_seconds,
            cache_hit=cache_hit if isinstance(cache_hit, bool) else None,
            error=observation.status in {ToolObservationStatus.INVALID, ToolObservationStatus.ERROR},
        )
        return observation


def make_rwr_hpc_handler(state: RwrHpcServiceState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "MentorRwrHpcService/1.0"

        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path == "/health":
                self._send_json(200, {"ok": True})
                return
            if self.path == "/metrics":
                self._send_json(200, state.metrics.snapshot())
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path != "/execute":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request body must be a JSON object")
                action_payload = payload.get("tool_action", payload)
                if not isinstance(action_payload, dict):
                    raise ValueError("tool_action must be a JSON object")
                observation = state.execute(ToolAction.from_dict(action_payload))
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, {"observation": observation.to_dict()})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


class RwrHpcServiceClient:
    """Client used by RuntimeEnvironment when RWR_HPC_SERVICE_URL is set."""

    def __init__(self, service_url: str, *, timeout_seconds: float = 600.0) -> None:
        self.service_url = service_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def run_tool(self, tool_action: ToolAction) -> ToolExecutionResult:
        request = urllib.request.Request(
            f"{self.service_url}/execute",
            data=json.dumps({"tool_action": tool_action.to_dict()}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ToolExecutionError(f"RWR++ service HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"RWR++ service request failed: {exc}") from exc

        observation_payload = payload.get("observation") if isinstance(payload, dict) else None
        if not isinstance(observation_payload, dict):
            raise ToolExecutionError("RWR++ service response did not contain an observation object.")
        observation = ToolObservation.from_dict(observation_payload)
        if observation.status in {ToolObservationStatus.INVALID, ToolObservationStatus.ERROR}:
            raise ToolExecutionError(observation.error or f"RWR++ service returned {observation.status.value}.")

        provenance = dict(observation.provenance)
        provenance["rwr_hpc_service_url"] = self.service_url
        return ToolExecutionResult(
            payload=observation.payload or {},
            provenance=provenance,
            is_empty=observation.status == ToolObservationStatus.EMPTY,
        )


def build_rwr_hpc_server(
    *,
    host: str,
    port: int,
    state: RwrHpcServiceState,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_rwr_hpc_handler(state))
