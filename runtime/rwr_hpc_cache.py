"""Small disk-backed cache for structured RWR-HPC tool results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

class RwrHpcCache:
    def __init__(
        self,
        root_dir: str | Path,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()

    def get(
        self,
        tool_name: str,
        cache_key: str
    ) -> dict[str, dict[str, Any]] | None:
        entry_dir = self.root_dir / tool_name / cache_key
        if not entry_dir.exists():
            return None
        
        payload_path = entry_dir / "payload.json"
        provenance_path = entry_dir / "provenance.json"

        if not payload_path.exists() or not provenance_path.exists():
            return None
        
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

        return {
            "payload": payload,
            "provenance": provenance,
        }

    def put(
        self,
        tool_name: str,
        cache_key: str,
        payload: dict[str, Any],
        provenance: dict[str, Any]
    ) -> None:

        entry_dir = self.root_dir / tool_name / cache_key
        entry_dir.mkdir(parents=True, exist_ok=True)

        (entry_dir / "payload.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (entry_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    

