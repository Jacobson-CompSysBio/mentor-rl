#!/usr/bin/env python3
"""Run a lightweight RWR++ worker service around RuntimeEnvironment."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.environment import RuntimeEnvironment
from runtime.rwr_hpc_service import RwrHpcServiceState, build_rwr_hpc_server


def _write_metrics(path: Path | None, state: RwrHpcServiceState) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.metrics.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _start_metrics_writer(
    *,
    path: Path | None,
    state: RwrHpcServiceState,
    interval_seconds: float,
    stop_event: threading.Event,
) -> threading.Thread | None:
    if path is None or interval_seconds <= 0:
        return None

    def run() -> None:
        while not stop_event.wait(interval_seconds):
            _write_metrics(path, state)

    thread = threading.Thread(target=run, name="rwr-hpc-metrics-writer", daemon=True)
    thread.start()
    return thread


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve model-facing RWR++ tools over HTTP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=51021)
    parser.add_argument("--store-dir", type=str, default=None)
    parser.add_argument("--rwr-hpc-flist", type=str, required=True)
    parser.add_argument("--rwr-hpc-build-dir", type=str, required=True)
    parser.add_argument("--rwr-hpc-cache-dir", type=str, default=None)
    parser.add_argument("--rwr-hpc-scratch-root", type=str, default=None)
    parser.add_argument(
        "--rwr-hpc-edgelist-has-headers",
        dest="rwr_hpc_edgelist_has_headers",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--rwr-hpc-edgelist-no-headers",
        dest="rwr_hpc_edgelist_has_headers",
        action="store_false",
    )
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--metrics-interval-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    os.environ.pop("RWR_HPC_SERVICE_URL", None)
    environment = RuntimeEnvironment(
        store_dir=args.store_dir,
        rwr_hpc_flist=args.rwr_hpc_flist,
        rwr_hpc_build_dir=args.rwr_hpc_build_dir,
        rwr_hpc_cache_dir=args.rwr_hpc_cache_dir,
        rwr_hpc_scratch_root=args.rwr_hpc_scratch_root,
        rwr_hpc_no_edgelist_headers=not args.rwr_hpc_edgelist_has_headers,
        enable_rwr_hpc_apps=True,
        enable_rwr_hpc_structured_tools=True,
        require_rwr_hpc_structured_tools=True,
    )
    state = RwrHpcServiceState(lambda action: environment.execute(action))
    server = build_rwr_hpc_server(host=args.host, port=args.port, state=state)
    stop_event = threading.Event()
    _start_metrics_writer(
        path=args.metrics_path,
        state=state,
        interval_seconds=args.metrics_interval_seconds,
        stop_event=stop_event,
    )

    def request_shutdown(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}; shutting down RWR++ service.", flush=True)
        stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    print(f"RWR++ service listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop_event.set()
        server.server_close()
        _write_metrics(args.metrics_path, state)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
