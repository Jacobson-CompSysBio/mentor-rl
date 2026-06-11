"""Small disk-backed cache for structured RWR-HPC tool results."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = "rwr-hpc-cache-v1"
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def stable_json_hash(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA256 hash for a JSON-compatible payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return SHA256 for one file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_rwr_loe_cache_key(
    *,
    request_payload: dict[str, Any],
    network_flist_sha256: str,
    rwr_hpc_build_id: str,
) -> str:
    """Build the cache key for one logical rwr_loe request."""
    return stable_json_hash(
        {
            "tool_name": "rwr_loe",
            **request_payload,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": rwr_hpc_build_id,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        }
    )


def make_rwr_hpc_cache_key(
    *,
    tool_name: str,
    request_payload: dict[str, Any],
    network_flist_sha256: str,
    rwr_hpc_build_id: str,
) -> str:
    """Build a cache key for a structured RWR-HPC request."""
    return stable_json_hash(
        {
            "tool_name": tool_name,
            **request_payload,
            "network_flist_sha256": network_flist_sha256,
            "rwr_hpc_build_id": rwr_hpc_build_id,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        }
    )


def _validate_safe_segment(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"{name} contains unsafe path characters: {value!r}")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text using a temporary file and atomic rename.

    This matters on Frontier because many rollout workers may try to write cache
    entries at the same time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


class RwrHpcCache:
    def __init__(self, root_dir: str | Path = "data/runtime/rwr_hpc_cache") -> None:
        self.root_dir = Path(root_dir).resolve()

    def _entry_dir(self, tool_name: str, cache_key: str) -> Path:
        _validate_safe_segment("tool_name", tool_name)
        _validate_safe_segment("cache_key", cache_key)
        return self.root_dir / tool_name / cache_key

    def get(
        self,
        tool_name: str,
        cache_key: str,
    ) -> dict[str, Any] | None:
        entry_dir = self._entry_dir(tool_name, cache_key)
        if not entry_dir.exists():
            return None

        payload_path = entry_dir / "payload.json"
        provenance_path = entry_dir / "provenance.json"
        request_path = entry_dir / "request.json"

        if not payload_path.exists() or not provenance_path.exists():
            return None

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

        request = None
        if request_path.exists():
            request = json.loads(request_path.read_text(encoding="utf-8"))

        return {
            "payload": payload,
            "provenance": provenance,
            "request": request,
        }

    def put(
        self,
        tool_name: str,
        cache_key: str,
        *,
        request: dict[str, Any],
        payload: dict[str, Any],
        provenance: dict[str, Any],
        raw_stdout: str = "",
        raw_stderr: str = "",
    ) -> None:
        entry_dir = self._entry_dir(tool_name, cache_key)
        entry_dir.mkdir(parents=True, exist_ok=True)

        _atomic_write_text(
            entry_dir / "request.json",
            json.dumps(request, indent=2, sort_keys=True),
        )
        _atomic_write_text(
            entry_dir / "payload.json",
            json.dumps(payload, indent=2, sort_keys=True),
        )
        _atomic_write_text(
            entry_dir / "provenance.json",
            json.dumps(provenance, indent=2, sort_keys=True),
        )
        _atomic_write_text(entry_dir / "raw_stdout.txt", raw_stdout)
        _atomic_write_text(entry_dir / "raw_stderr.txt", raw_stderr)

        raw_outputs_dir = entry_dir / "raw_outputs"
        raw_outputs_dir.mkdir(exist_ok=True)
