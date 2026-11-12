"""Acquire native S2.4 neighbor maps with complete support for each returned row.

Canonical neighbor order determines the bounded selection. Transport pages can
divide that selection, but never divide a row's selected-layer support.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

from .s2_multiplex import MultiplexRuntime, SelectionError
from .s2_neighbors import NeighborIndex, _selection
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ("multiplex_id", "gene_id", "layer_scope", "direction", "mode", "top_k", "cursor")
DIRECTIONS = ("incident", "in", "out", "undirected")


class NeighborLayersRuntime:
    """Serve the approved multiplex neighbor operation over a pinned native view."""

    def __init__(self, view: GraphView, *, page_size: int | None = None,
                 disclose_total: bool = True, overlap: int = 0,
                 empty_first_page: bool = False, empty_page_at: int | None = None,
                 index: NeighborIndex | None = None) -> None:
        if page_size is not None and (type(page_size) is not int or page_size < 1):
            raise ValueError("page_size must be a positive integer or null")
        if type(overlap) is not int or overlap < 0 or (overlap and (page_size is None or overlap >= page_size)):
            raise ValueError("overlap must be nonnegative and smaller than page_size")
        if type(disclose_total) is not bool or type(empty_first_page) is not bool:
            raise ValueError("transport switches must be booleans")
        if empty_page_at is not None and (type(empty_page_at) is not int or empty_page_at < 0):
            raise ValueError("empty_page_at must be a nonnegative page ordinal or null")
        if empty_first_page and empty_page_at not in (None, 0):
            raise ValueError("choose one empty-page position")
        if index is not None and index.view is not view:
            raise ValueError("a shared neighbor index must belong to this exact immutable view")
        self.view = view
        self.multiplex = MultiplexRuntime(view)
        self.index = index if index is not None else NeighborIndex(view)
        self.transport = {"page_size": page_size, "disclose_total": disclose_total,
                          "overlap": overlap, "empty_first_page": empty_first_page,
                          "empty_page_at": empty_page_at}
        self._empty_at = 0 if empty_first_page else empty_page_at
        self._schedules: dict[str, tuple[list[tuple[int, int]], dict[str | None, int]]] = {}

    def arguments(self, gene_id: str, layer_scope: Mapping[str, Any], direction: str,
                  mode: str = "complete", top_k: int | None = None,
                  cursor: str | None = None) -> dict[str, Any]:
        """Construct all seven explicit arguments without changing the supplied selection."""
        return {"multiplex_id": self.view.multiplex_id, "gene_id": gene_id,
                "layer_scope": deepcopy(layer_scope), "direction": direction,
                "mode": mode, "top_k": top_k, "cursor": cursor}

    def resolve_scope(self, layer_scope: Any, direction: str = "incident") -> tuple[str, ...]:
        """Expand the registered scope and enforce each selected layer's native direction."""
        selected = self.multiplex.resolve_scope(layer_scope)
        if direction not in DIRECTIONS:
            raise SelectionError("invalid_input", "DIRECTION_UNAVAILABLE")
        if direction in ("in", "out") and any(not self.view.layers[layer].directed for layer in selected):
            raise SelectionError("invalid_input", "DIRECTION_UNAVAILABLE")
        if direction == "undirected" and any(self.view.layers[layer].directed for layer in selected):
            raise SelectionError("invalid_input", "DIRECTION_UNAVAILABLE")
        return selected

    def _validate(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        """Reject malformed identity, scope, direction, selection, or cursor atomically."""
        if set(arguments) != set(ARGUMENT_KEYS):
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        if not arguments["multiplex_id"] or arguments["layer_scope"] is None:
            raise SelectionError("invalid_input", "MISSING_SCOPE")
        if arguments["multiplex_id"] != self.view.multiplex_id:
            raise SelectionError("not_found", "UNKNOWN_MULTIPLEX")
        gene = arguments["gene_id"]
        if not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene):
            raise SelectionError("invalid_input", "INVALID_GENE_INPUT")
        selected = self.resolve_scope(arguments["layer_scope"], arguments["direction"])
        _selection(arguments["mode"], arguments["top_k"])
        cursor = arguments["cursor"]
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise SelectionError("invalid_input", "INVALID_CURSOR")
        return selected

    def native_inputs(self, arguments: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
        """List the complete native collections needed for visible-list assembly."""
        selected = self._validate(arguments)
        return tuple({"layer_id": layer, "direction": direction}
                     for layer in selected
                     for direction in (self.index.directions(layer) if arguments["direction"] == "incident"
                                       else (arguments["direction"],)))

    def _neighbors(self, gene_id: str, selected: tuple[str, ...], direction: str) -> list[dict[str, Any]]:
        """Group every native connection before any bounded selection or page split."""
        by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for layer_id in selected:
            layer = self.view.layers[layer_id]
            directions = self.index.directions(layer_id) if direction == "incident" else (direction,)
            edges_by_neighbor: dict[str, dict[tuple[str, str], int | float]] = defaultdict(dict)
            for native_direction in directions:
                for neighbor, weight in self.index.neighbors(layer_id, gene_id, native_direction):
                    if native_direction == "undirected":
                        pair = tuple(sorted((gene_id, neighbor)))
                    elif native_direction == "in":
                        pair = (neighbor, gene_id)
                    else:
                        pair = (gene_id, neighbor)
                    edges_by_neighbor[neighbor][pair] = weight
            for neighbor, edges in edges_by_neighbor.items():
                by_gene[neighbor].append({"layer_id": layer_id, "directed": layer.directed,
                                          "edges": [{"gene_ids": list(pair), "weight": weight}
                                                    for pair, weight in edges.items()]})
        return [{"gene_id": gene, "supporting_layers": by_gene[gene],
                 "supporting_layer_count": len(by_gene[gene])} for gene in sorted(by_gene)]

    def _schedule(self, arguments: Mapping[str, Any], size: int) -> tuple[list[tuple[int, int]], dict[str | None, int]]:
        """Bind opaque cursors to the exact selection, graph version, and transport."""
        selection = {key: arguments[key] for key in ARGUMENT_KEYS if key != "cursor"}
        identity = json.dumps([self.view.graph_version, selection, self.transport], sort_keys=True, separators=(",", ":"))
        if identity not in self._schedules:
            pages = []
            offset = 0
            page_size = self.transport["page_size"] or max(size, 1)
            if size == 0:
                pages.append((0, 0))
            while offset < size:
                if len(pages) == self._empty_at:
                    pages.append((offset, offset))
                end = min(offset + page_size, size)
                pages.append((offset, end))
                if end == size:
                    break
                offset = end - self.transport["overlap"]
            cursors = {None: 0}
            for ordinal in range(1, len(pages)):
                token = "s24_" + hashlib.sha256((identity + ":" + str(ordinal)).encode()).hexdigest()
                cursors[token] = ordinal
            self._schedules[identity] = pages, cursors
        return self._schedules[identity]

    def get_neighbor_layers(self, multiplex_id: Any, gene_id: Any, layer_scope: Any,
                            direction: Any, mode: Any, top_k: Any, cursor: Any,
                            *, evidence_id: str) -> dict[str, Any]:
        """Acquire one map page with all selected-layer support for each returned neighbor."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, gene_id, layer_scope, direction, mode, top_k, cursor)))
        try:
            selected = self._validate(arguments)
            missing = gene_id not in self.view.multiplex_nodes
            neighbors = self._neighbors(gene_id, selected, direction) if not missing else []
            selected_size = len(neighbors) if mode == "complete" else min(top_k, len(neighbors))
            pages, cursors = self._schedule(arguments, selected_size)
            if cursor not in cursors or missing and cursor is not None:
                raise SelectionError("invalid_input", "INVALID_CURSOR")
        except SelectionError as error:
            return {"status": error.status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": error.issue_code, "arguments": arguments}}
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("a successful acquisition requires an evidence ID")
        scope = self.multiplex._scope(layer_scope, selected)
        layer_results = rows = count = absent = collection = None
        if not missing:
            ordinal = cursors[cursor]
            start, end = pages[ordinal]
            rows = neighbors[start:end]
            count = len(neighbors) if self.transport["disclose_total"] or not neighbors else None
            absent = [layer for layer in selected if gene_id not in self.view.layers[layer].nodes]
            layer_results = [{"layer_id": layer, "directed": self.view.layers[layer].directed,
                              "not_in_layer_gene_ids": [gene_id] if layer in absent else []} for layer in selected]
            next_cursor = next((token for token, page in cursors.items() if page == ordinal + 1), None)
            collection = {"mode": mode, "returned_count": len(rows), "total_count": count,
                          "complete": start == 0 and end == len(neighbors), "next_cursor": next_cursor}
        return {
            "status": "not_found" if missing else "resolved",
            "provenance": {"registry": [], "graph": [{
                "multiplex_id": self.view.multiplex_id, "graph_version": self.view.graph_version,
                "layer_scope": scope, "evidence_id": evidence_id,
            }]},
            "result": {"query_gene_id": gene_id, "multiplex_id": self.view.multiplex_id,
                       "layer_scope": scope, "direction": direction, "layer_results": layer_results,
                       "neighbors": rows, "unique_neighbor_count": count,
                       "missing_gene_ids": [gene_id] if missing else [], "layers_without_gene": absent,
                       "coverage": {"operation": "layer_neighbor_maps", "requested_layer_ids": list(selected),
                                    "checked_layer_ids": [] if missing else list(selected),
                                    "unchecked_layer_ids": list(selected) if missing else []},
                       "collection": collection},
        }
