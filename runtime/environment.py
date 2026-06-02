"""Simple environment wrapper for deterministic runtime tool execution.

This file gives the rest of the pipeline one small object to talk to.
The environment does four things:

1. load or accept the multiplex graph
2. expose the available genes and layers
3. validate tool calls against the current runtime state
4. return structured `ToolObservation` objects

The environment is intentionally small. It should be easy to inspect and easy
to swap out later if the runtime grows more complex.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Iterable

from utils.multiplex import Multiplex

from .backend import CompiledRuntimeBackend
from .rwr_hpc_app_backend import RwrHpcAppBackend
from .schemas import (
    RUNTIME_SCHEMA_VERSION,
    StructuredState,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
)
from .tools import (
    MultiplexIndex,
    ToolExecutionError,
    ToolExecutionResult,
    build_multiplex_index,
    enrich_gene_set,
    get_neighbors,
    induce_subgraph,
    load_enrichment_cache,
    load_mygene_cache,
    query_mygene,
    rwr_monoplex,
    rwr_multiplex,
    shortest_path,
)
from .validators import is_duplicate_tool_action, normalize_tool_action, validate_tool_action


class RuntimeEnvironment:
    """Validate and execute runtime tool calls against one multiplex graph."""

    def __init__(
        self,
        *,
        multiplex: Multiplex | None = None,
        multiplex_flist: str | None = None,
        store_dir: str | None = None,
        compiled_library_path: str | None = None,
        mygene_cache: dict[str, list[dict[str, Any]]] | None = None,
        mygene_cache_path: str | None = None,
        allow_network_mygene: bool = False,
        enrichment_cache: dict[str, dict[str, Any]] | None = None,
        enrichment_cache_path: str | None = None,
        allow_network_enrichment: bool = False,
        enrichment_background_gene_ids: Iterable[str] | None = None,
        rwr_hpc_build_dir: str | None = None,
        rwr_hpc_app_manifest_path: str | None = None,
        enable_rwr_hpc_apps: bool = True,
    ) -> None:
        if store_dir is None and multiplex is None and not multiplex_flist:
            raise ValueError(
                "Provide one of: 'store_dir', 'multiplex', or 'multiplex_flist'."
            )

        self.multiplex = None
        self.index = None
        self.compiled_backend = None
        self.graph_backend_kind = "python_reference"

        if store_dir is not None:
            self.compiled_backend = CompiledRuntimeBackend(
                store_dir=store_dir,
                library_path=compiled_library_path,
            )
            store_summary = self.compiled_backend.describe()
            self.available_gene_ids = set(store_summary["gene_ids"])
            self.available_layers = set(store_summary["layer_names"])
            self.graph_backend_kind = "compiled_cpp"
        else:
            if multiplex is None:
                multiplex = Multiplex(flist=multiplex_flist or "")
            self.multiplex = multiplex
            self.index = build_multiplex_index(multiplex)
            self.available_gene_ids = set(self.index.gene_ids)
            self.available_layers = set(self.index.layer_names)
        
        self.rwr_hpc_app_backend = None
        if enable_rwr_hpc_apps and (rwr_hpc_build_dir or os.environ.get("RWR_HPC_BUILD_DIR")):
            self.rwr_hpc_app_backend = RwrHpcAppBackend(
                build_dir=rwr_hpc_build_dir,
                manifest_path=rwr_hpc_app_manifest_path,
            )

        self.mygene_cache_path = mygene_cache_path
        self.allow_network_mygene = allow_network_mygene
        self.enrichment_cache_path = enrichment_cache_path
        self.allow_network_enrichment = allow_network_enrichment
        self._annotation_lock = threading.Lock()
        if mygene_cache is not None:
            self.mygene_cache = mygene_cache
        else:
            self.mygene_cache = load_mygene_cache(mygene_cache_path)
        if enrichment_cache is not None:
            self.enrichment_cache = enrichment_cache
        else:
            self.enrichment_cache = load_enrichment_cache(enrichment_cache_path)
        if enrichment_background_gene_ids is None:
            self.enrichment_background_gene_ids = sorted(self.available_gene_ids)
        else:
            self.enrichment_background_gene_ids = sorted(
                {str(gene_id) for gene_id in enrichment_background_gene_ids if str(gene_id)}
            )

    def describe(self) -> dict[str, Any]:
        """Return a small summary of the runtime resources."""

        return {
            "runtime_version": RUNTIME_SCHEMA_VERSION,
            "graph_backend_kind": self.graph_backend_kind,
            "layer_names": sorted(self.available_layers),
            "layer_count": len(self.available_layers),
            "gene_count": len(self.available_gene_ids),
            "mygene_cache_size": len(self.mygene_cache),
            "allow_network_mygene": self.allow_network_mygene,
            "enrichment_cache_size": len(self.enrichment_cache),
            "allow_network_enrichment": self.allow_network_enrichment,
            "enrichment_background_gene_count": len(self.enrichment_background_gene_ids),
            "rwr_hpc_apps_enabled": self.rwr_hpc_app_backend is not None,
            "rwr_hpc_apps_available": (
                sorted(self.rwr_hpc_app_backend.available_apps() if self.rwr_hpc_app_backend is not None else [])
            ),
            "rwr_hpc_apps_missing_required": (
                self.rwr_hpc_app_backend.missing_apps()
                if self.rwr_hpc_app_backend is not None else []
            ),
            "rwr_hpc_apps_missing_optional": (
                self.rwr_hpc_app_backend.missing_optional_apps()
                if self.rwr_hpc_app_backend is not None 
                else []
            ),
        }

    def execute(
        self,
        tool_action: ToolAction,
        *,
        state: StructuredState | None = None,
        prior_actions: Iterable[ToolAction] | None = None,
    ) -> ToolObservation:
        """Validate and execute one tool call."""

        tool_action = normalize_tool_action(tool_action)
        validation = validate_tool_action(
            tool_action,
            state=state,
            available_gene_ids=self.available_gene_ids,
            available_layers=self.available_layers,
        )
        if not validation.valid:
            return self._invalid_observation(tool_action, "; ".join(validation.errors), validation.errors)

        if prior_actions is not None and is_duplicate_tool_action(tool_action, prior_actions):
            return self._invalid_observation(
                tool_action,
                "Duplicate tool call matches an earlier action.",
                ["duplicate_tool_call"],
            )

        try:
            result = self._dispatch(tool_action)
        except ToolExecutionError as exc:
            return ToolObservation(
                status=ToolObservationStatus.ERROR,
                provenance=self._base_provenance(tool_action),
                call_id=tool_action.call_id,
                error=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive safety net
            return ToolObservation(
                status=ToolObservationStatus.ERROR,
                provenance=self._base_provenance(tool_action),
                call_id=tool_action.call_id,
                error=f"Unexpected runtime error: {exc}",
            )

        status = ToolObservationStatus.EMPTY if result.is_empty else ToolObservationStatus.SUCCESS
        provenance = self._merge_provenance(tool_action, result.provenance)
        return ToolObservation(
            status=status,
            provenance=provenance,
            call_id=tool_action.call_id,
            payload=result.payload,
        )

    def execute_tool_action(
        self,
        tool_action: ToolAction,
        *,
        state: StructuredState | None = None,
        prior_actions: Iterable[ToolAction] | None = None,
    ) -> ToolObservation:
        """Alias for `execute` to keep call sites readable."""

        return self.execute(tool_action, state=state, prior_actions=prior_actions)

    def _dispatch(self, tool_action: ToolAction) -> ToolExecutionResult:
        if tool_action.tool_name == "query_mygene":
            with self._annotation_lock:
                return query_mygene(
                    tool_action.arguments["query"],
                    fields=tool_action.arguments.get("fields"),
                    cache=self.mygene_cache,
                    cache_path=self.mygene_cache_path,
                    allow_network=self.allow_network_mygene,
                )

        if tool_action.tool_name == "enrich_gene_set":
            with self._annotation_lock:
                return enrich_gene_set(
                    tool_action.arguments["genes"],
                    background_gene_ids=self.enrichment_background_gene_ids,
                    sources=tool_action.arguments.get("sources"),
                    user_threshold=tool_action.arguments.get("user_threshold", 0.05),
                    top_k=tool_action.arguments.get("top_k", 10),
                    cache=self.enrichment_cache,
                    cache_path=self.enrichment_cache_path,
                    allow_network=self.allow_network_enrichment,
                )

        if tool_action.tool_name == "get_neighbors":
            if self.compiled_backend is not None:
                return self.compiled_backend.get_neighbors(
                    tool_action.arguments["gene"],
                    layers=tool_action.arguments.get("layers"),
                )
            return get_neighbors(
                self.index,
                tool_action.arguments["gene"],
                layers=tool_action.arguments.get("layers"),
            )

        if tool_action.tool_name == "induce_subgraph":
            if self.compiled_backend is not None:
                return self.compiled_backend.induce_subgraph(
                    tool_action.arguments["genes"],
                    layers=tool_action.arguments.get("layers"),
                )
            return induce_subgraph(
                self.index,
                tool_action.arguments["genes"],
                layers=tool_action.arguments.get("layers"),
            )

        if tool_action.tool_name == "shortest_path":
            if self.compiled_backend is not None:
                return self.compiled_backend.shortest_path(
                    tool_action.arguments["source"],
                    tool_action.arguments["target"],
                    layer_name=tool_action.arguments.get("layer"),
                )
            return shortest_path(
                self.index,
                tool_action.arguments["source"],
                tool_action.arguments["target"],
                layer=tool_action.arguments.get("layer"),
            )

        if tool_action.tool_name == "rwr_multiplex":
            if self.compiled_backend is not None:
                return self.compiled_backend.rwr_multiplex(
                    tool_action.arguments["seeds"],
                    top_k=tool_action.arguments.get("top_k", 20),
                    restart_probability=0.35,
                )
            return rwr_multiplex(
                self.index,
                tool_action.arguments["seeds"],
                top_k=tool_action.arguments.get("top_k", 20),
            )

        if tool_action.tool_name == "rwr_monoplex":
            if self.compiled_backend is not None:
                return self.compiled_backend.rwr_monoplex(
                    tool_action.arguments["seeds"],
                    layer_name=tool_action.arguments["layer"],
                    top_k=tool_action.arguments.get("top_k", 20),
                    restart_probability=0.35,
                )
            return rwr_monoplex(
                self.index,
                tool_action.arguments["seeds"],
                layer=tool_action.arguments["layer"],
                top_k=tool_action.arguments.get("top_k", 20),
            )
        
        if tool_action.tool_name == "rwr_hpc_app":
            if self.rwr_hpc_app_backend is None:
                raise ToolExecutionError("RWR-HPC app backend is not initialized. "
                                         "Set RWR_HPC_BUILD_DIR or pass rwr_hpc_build_dir.")
            app_name = tool_action.arguments.get("app") or arguments.get("app_name")
            app_args = tool_action.arguments.get("args", [])
            timeout_seconds = tool_action.arguments.get("timeout_seconds", 300)
            cwd = tool_action.arguments.get("cwd")
            allow_nonzero = tool_action.arguments.get("allow_nonzero", False)

            try:
                app_result = self.rwr_hpc_app_backend.run_app(
                    app_name,
                    app_args,
                    timeout_seconds=timeout_seconds,
                    cwd=cwd,
                )
            except KeyError as exc:
                raise ToolExecutionError(str(exc)) from exc
            
            if app_result.returncode != 0 and not allow_nonzero:
                text = app_result.stderr or app_result.stdout
                preview = text[:1000] if text else ""
                raise ToolExecutionError(
                    f"RWR-HPC app {app_name!r} failed with return code {app_result.returncode}. "
                    f"Output: {preview}"
                )
            
            return ToolExecutionResult(
                payload=app_result.payload,
                provenance=app_result.provenance,
                is_empty=not bool(app_result.stdout or app_result.stderr),
            )

        raise ToolExecutionError(f"Unsupported tool name: {tool_action.tool_name}.")

    def _base_provenance(self, tool_action: ToolAction) -> dict[str, Any]:
        return {
            "tool_name": tool_action.tool_name,
            "runtime_version": RUNTIME_SCHEMA_VERSION,
        }

    def _merge_provenance(
        self,
        tool_action: ToolAction,
        runtime_provenance: dict[str, Any],
    ) -> dict[str, Any]:
        merged = self._base_provenance(tool_action)
        merged.update(runtime_provenance)
        return merged

    def _invalid_observation(
        self,
        tool_action: ToolAction,
        error_message: str,
        validation_errors: list[str],
    ) -> ToolObservation:
        provenance = self._base_provenance(tool_action)
        provenance["validation_errors"] = list(validation_errors)
        return ToolObservation(
            status=ToolObservationStatus.INVALID,
            provenance=provenance,
            call_id=tool_action.call_id,
            error=error_message,
        )


__all__ = ["RuntimeEnvironment", "MultiplexIndex"]
