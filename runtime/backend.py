"""Python wrapper around the compiled C++ multiplex runtime.

The rest of the repo should not need to know how the compiled backend is
loaded. This module hides the `ctypes` details and returns the same simple
result objects used by the Python reference tools.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any

from .tools import ToolExecutionError, ToolExecutionResult


def find_default_runtime_library(repo_root: str | Path | None = None) -> Path | None:
    """Look for the compiled runtime library in the default build locations."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    repo_root = Path(repo_root)

    candidates = [
        repo_root / "cpp_runtime" / "build" / "libmentor_runtime.so",
        repo_root / "cpp_runtime" / "build" / "mentor_runtime.so",
        repo_root / "cpp_runtime" / "libmentor_runtime.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class CompiledRuntimeBackend:
    """Thin Python wrapper for the compiled C++ graph runtime."""

    def __init__(
        self,
        *,
        store_dir: str,
        library_path: str | None = None,
    ) -> None:
        if library_path is None:
            default_library = find_default_runtime_library()
            if default_library is None:
                raise ToolExecutionError(
                    "Could not find the compiled runtime library. Build cpp_runtime first."
                )
            library_path = str(default_library)

        self.store_dir = str(store_dir)
        self.library_path = str(library_path)
        self._library = ctypes.CDLL(self.library_path)
        self._configure_signatures()

        handle = self._library.mentor_open_store(self.store_dir.encode("utf-8"))
        if not handle:
            raise ToolExecutionError(
                f"Could not open compiled runtime store at {self.store_dir}."
            )
        self._handle = handle

    def _configure_signatures(self) -> None:
        self._library.mentor_open_store.argtypes = [ctypes.c_char_p]
        self._library.mentor_open_store.restype = ctypes.c_void_p

        self._library.mentor_close_store.argtypes = [ctypes.c_void_p]
        self._library.mentor_close_store.restype = None

        self._library.mentor_store_summary.argtypes = [ctypes.c_void_p]
        self._library.mentor_store_summary.restype = ctypes.c_void_p

        self._library.mentor_get_neighbors.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._library.mentor_get_neighbors.restype = ctypes.c_void_p

        self._library.mentor_induce_subgraph.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._library.mentor_induce_subgraph.restype = ctypes.c_void_p

        self._library.mentor_shortest_path.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self._library.mentor_shortest_path.restype = ctypes.c_void_p

        self._library.mentor_rwr_monoplex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_ulonglong,
            ctypes.c_double,
        ]
        self._library.mentor_rwr_monoplex.restype = ctypes.c_void_p

        self._library.mentor_rwr_multiplex.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_ulonglong,
            ctypes.c_double,
        ]
        self._library.mentor_rwr_multiplex.restype = ctypes.c_void_p

        self._library.mentor_free_string.argtypes = [ctypes.c_void_p]
        self._library.mentor_free_string.restype = None

    def close(self) -> None:
        """Release the compiled store handle."""

        if getattr(self, "_handle", None):
            self._library.mentor_close_store(self._handle)
            self._handle = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def describe(self) -> dict[str, Any]:
        """Return a small store summary."""

        return self._call_payload(self._library.mentor_store_summary, self._handle)

    def get_neighbors(self, gene_id: str, *, layers: list[str] | None = None) -> ToolExecutionResult:
        """Run the compiled `get_neighbors` tool."""

        return self._call_tool_result(
            self._library.mentor_get_neighbors,
            self._handle,
            gene_id.encode("utf-8"),
            self._encode_csv(layers),
        )

    def induce_subgraph(
        self,
        gene_ids: list[str],
        *,
        layers: list[str] | None = None,
    ) -> ToolExecutionResult:
        """Run the compiled `induce_subgraph` tool."""

        return self._call_tool_result(
            self._library.mentor_induce_subgraph,
            self._handle,
            self._encode_csv(gene_ids),
            self._encode_csv(layers),
        )

    def shortest_path(
        self,
        source_gene_id: str,
        target_gene_id: str,
        *,
        layer_name: str | None = None,
    ) -> ToolExecutionResult:
        """Run the compiled `shortest_path` tool."""

        return self._call_tool_result(
            self._library.mentor_shortest_path,
            self._handle,
            source_gene_id.encode("utf-8"),
            target_gene_id.encode("utf-8"),
            (layer_name or "").encode("utf-8"),
        )

    def rwr_monoplex(
        self,
        seed_gene_ids: list[str],
        *,
        layer_name: str,
        top_k: int,
        restart_probability: float,
    ) -> ToolExecutionResult:
        """Run the compiled `rwr_monoplex` tool."""

        return self._call_tool_result(
            self._library.mentor_rwr_monoplex,
            self._handle,
            self._encode_csv(seed_gene_ids),
            layer_name.encode("utf-8"),
            int(top_k),
            float(restart_probability),
        )

    def rwr_multiplex(
        self,
        seed_gene_ids: list[str],
        *,
        top_k: int,
        restart_probability: float,
    ) -> ToolExecutionResult:
        """Run the compiled `rwr_multiplex` tool."""

        return self._call_tool_result(
            self._library.mentor_rwr_multiplex,
            self._handle,
            self._encode_csv(seed_gene_ids),
            int(top_k),
            float(restart_probability),
        )

    def _encode_csv(self, values: list[str] | None) -> bytes:
        if not values:
            return b""
        return ",".join(values).encode("utf-8")

    def _call_payload(self, function: Any, *args: Any) -> dict[str, Any]:
        response = self._call_json(function, *args)
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise ToolExecutionError("Compiled runtime returned a malformed payload.")
        return payload

    def _call_tool_result(self, function: Any, *args: Any) -> ToolExecutionResult:
        response = self._call_json(function, *args)
        payload = response.get("payload")
        provenance = response.get("provenance")
        if not isinstance(payload, dict) or not isinstance(provenance, dict):
            raise ToolExecutionError("Compiled runtime returned malformed tool output.")
        return ToolExecutionResult(
            payload=payload,
            provenance=provenance,
            is_empty=bool(response.get("is_empty", False)),
        )

    def _call_json(self, function: Any, *args: Any) -> dict[str, Any]:
        raw_pointer = function(*args)
        if not raw_pointer:
            raise ToolExecutionError("Compiled runtime returned a null response pointer.")

        try:
            raw_text = ctypes.string_at(raw_pointer).decode("utf-8")
        finally:
            self._library.mentor_free_string(raw_pointer)

        try:
            response = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise ToolExecutionError(
                f"Compiled runtime returned invalid JSON: {error}"
            ) from error

        if not response.get("ok", False):
            raise ToolExecutionError(response.get("error", "Compiled runtime call failed."))
        return response


__all__ = ["CompiledRuntimeBackend", "find_default_runtime_library"]
