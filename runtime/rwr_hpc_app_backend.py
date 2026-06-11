from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# dataclass for app results, including stdout, stderr, return code, and any structured payload or provenance info
@dataclass(frozen=True)
class RwrHpcAppResult:
    """Result returned by a standalone RWR++ app call"""

    tool_name: str
    executable: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any]
    provenance: dict[str, Any]

# class to actually discover and run apps, given a build directory and manifest path
class RwrHpcAppBackend:
    """Subprocess-backed backend for full RWR++ app workflows.

    Exposes app-level RWR++ workflows without linking them into libmentor_runtime.so"""

    REQUIRED_APPS = {
        "rwr": {"rwr_wrapper", "rwr"},
        "rwr_loe": {"rwr_loe"},
        "rwr_ablation": {"rwr_ablation"},
        "rwr_perturbation": {"rwr_perturbation"},
        "grin": {"grin"},
        "shortest_paths": {"shortest_paths"},
        "clean_edge_list": {"clean_edge_list"},
    }

    OPTIONAL_APPS = {
        "gene_layer_map": {"gene_layer_map"},
        "disconnected_components": {"disconnected_components"},
    }

    EXPECTED_APPS = {
        **REQUIRED_APPS,
        **OPTIONAL_APPS,
    }

    def __init__(
        self,
        build_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:

        raw_build_dir = build_dir or os.environ.get("RWR_HPC_BUILD_DIR")
        if not raw_build_dir:
            raise ValueError("Set RWR_HPC_BUILD_DIR or pass build_dir.")
        
        self.build_dir = Path(raw_build_dir).resolve()
        if not self.build_dir.exists():
            raise FileNotFoundError(f"RWR-HPC build directory does not exist: {self.build_dir}")
        
        if manifest_path is None and build_dir is None:
            default_manifest_path = Path("data/runtime/rwr_hpc_apps.txt")
            manifest_path = default_manifest_path if default_manifest_path.exists() else None
        
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.apps = self._discover_apps()

    # get executables from manifest (.txt of paths)
    def _load_manifest_executables(self) -> list[Path]:
        if self.manifest_path is None:
            return []
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"RWR-HPC app manifest file not found: {self.manifest_path}")
        
        executables: list[Path] = []
        for raw_line in self.manifest_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            
            path = Path(line).resolve()
            if path.exists() and os.access(path, os.X_OK):
                executables.append(path)
        
        return executables
    
    # make sure files have exec perms
    def _scan_build_dir_executables(self) -> list[Path]:
        return [
            path
            for path in self.build_dir.rglob("*")
            if path.is_file() and os.access(path, os.X_OK)
        ]
    
    # match executables to apps and return a dict of app name to executable path
    def _discover_apps(self) -> dict[str, Path]:
        executables = self._load_manifest_executables()
        if not executables:
            executables = self._scan_build_dir_executables()
        
        apps: dict[str, Path] = {}
        for tool_name, possible_names in self.EXPECTED_APPS.items():
            matches = [path for path in executables if path.name in possible_names]
            if matches:
                apps[tool_name] = sorted(matches, key=lambda p: len(str(p)))[0]
        
        return apps

    # list available apps,
    def available_apps(self) -> dict[str, str]:
        return {name: str(path) for name, path in sorted(self.apps.items())}

    # list missing apps 
    def missing_apps(self) -> list[str]:
        return sorted(set(self.REQUIRED_APPS) - set(self.apps))

    def missing_optional_apps(self) -> list[str]:
        return sorted(set(self.OPTIONAL_APPS) - set(self.apps))

    # given an app name, return the executable path or raise if not found
    def require_app(self, tool_name: str) -> Path:
        if tool_name not in self.apps:
            available = ", ".join(sorted(self.apps)) or "none"
            missing = ", ".join(self.missing_apps()) or "none"
            raise KeyError(
                f"RWR++ app {tool_name!r} not found. "
                f"Available apps: {available}. Missing apps: {missing}."
            )

        return self.apps[tool_name]
    
    # run an app with args, capture stdout/stderr/returncode, and return an RwrHpcAppResult
    def run_app(
        self,
        tool_name: str,
        args: list[str],
        *,
        timeout_seconds: int = 3600,
        cwd: str | Path | None = None,
    ) -> RwrHpcAppResult:
        executable = self.require_app(tool_name)
        command = [str(executable), *args] 

        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            cwd=str(cwd) if cwd is not None else None
        )

        payload = {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }

        provenance = {
            "tool_name": tool_name,
            "implementation": "rwr_hpc_app",
            "executable": str(executable),
            "command": command,
            "build_dir": str(self.build_dir),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
        }

        return RwrHpcAppResult(
            tool_name=tool_name,
            executable=str(executable),
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            payload=payload,
            provenance=provenance,
        )
