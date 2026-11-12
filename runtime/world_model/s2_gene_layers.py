"""Acquire S2.5 layer memberships from the declared node inventories.

Membership does not depend on degree or edge direction. Transport pages divide
the selected positive layer list without changing its manifest order.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

from .s2_multiplex import MultiplexRuntime, SelectionError
from .s2_neighbors import _selection
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ("multiplex_id", "gene_id", "layer_scope", "mode", "top_k", "cursor")


class GeneLayersRuntime:
    """Serve scoped membership lists over one authoritative versioned graph view."""

    def __init__(self, view: GraphView, *, page_size: int | None = None,
                 disclose_total: bool = True, overlap: int = 0,
                 empty_page_at: int | None = None) -> None:
        if page_size is not None and (type(page_size) is not int or page_size < 1):
            raise ValueError("page_size must be a positive integer or null")
        if type(overlap) is not int or overlap < 0 or (overlap and (page_size is None or overlap >= page_size)):
            raise ValueError("overlap must be nonnegative and smaller than page_size")
        if type(disclose_total) is not bool:
            raise ValueError("disclose_total must be a boolean")
        if empty_page_at is not None and (type(empty_page_at) is not int or empty_page_at < 0):
            raise ValueError("empty_page_at must be a nonnegative page ordinal or null")
        self.view = view
        self.multiplex = MultiplexRuntime(view)
        self.transport = {"page_size": page_size, "disclose_total": disclose_total,
                          "overlap": overlap, "empty_page_at": empty_page_at}
        self._schedules: dict[str, tuple[list[tuple[int, int]], dict[str | None, int]]] = {}

    def arguments(self, gene_id: str, layer_scope: Mapping[str, Any], mode: str = "complete",
                  top_k: int | None = None, cursor: str | None = None) -> dict[str, Any]:
        """Construct all six explicit fields without changing the requested selection."""
        return {"multiplex_id": self.view.multiplex_id, "gene_id": gene_id,
                "layer_scope": deepcopy(layer_scope), "mode": mode, "top_k": top_k, "cursor": cursor}

    def resolve_scope(self, layer_scope: Any) -> tuple[str, ...]:
        """Expand the scope in manifest order and distinguish an unsupported family."""
        if (isinstance(layer_scope, dict) and set(layer_scope) == {"kind", "layer_ids", "layer_family"}
                and layer_scope["kind"] == "all" and layer_scope["layer_ids"] == []
                and layer_scope["layer_family"] is not None
                and layer_scope["layer_family"] not in self.multiplex.families):
            raise SelectionError("invalid_input", "INVALID_LAYER_FAMILY")
        return self.multiplex.resolve_scope(layer_scope)

    def _validate(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        """Reject invalid scope, identity, selection, or cursor without graph evidence."""
        if set(arguments) != set(ARGUMENT_KEYS):
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        if not arguments["multiplex_id"] or arguments["layer_scope"] is None:
            raise SelectionError("invalid_input", "MISSING_SCOPE")
        if arguments["multiplex_id"] != self.view.multiplex_id:
            raise SelectionError("not_found", "UNKNOWN_MULTIPLEX")
        gene_id = arguments["gene_id"]
        if not isinstance(gene_id, str) or not GENE_ID_PATTERN.fullmatch(gene_id):
            raise SelectionError("invalid_input", "INVALID_GENE_INPUT")
        selected = self.resolve_scope(arguments["layer_scope"])
        _selection(arguments["mode"], arguments["top_k"])
        cursor = arguments["cursor"]
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise SelectionError("invalid_input", "INVALID_CURSOR")
        return selected

    def _schedule(self, arguments: Mapping[str, Any], size: int) -> tuple[list[tuple[int, int]], dict[str | None, int]]:
        """Bind opaque cursors to the exact selection, version, and transport."""
        selection = {key: arguments[key] for key in ARGUMENT_KEYS if key != "cursor"}
        identity = json.dumps([self.view.graph_version, selection, self.transport], sort_keys=True, separators=(",", ":"))
        if identity not in self._schedules:
            pages, offset = [], 0
            page_size = self.transport["page_size"] or max(size, 1)
            if size == 0:
                pages.append((0, 0))
            while offset < size:
                if len(pages) == self.transport["empty_page_at"]:
                    pages.append((offset, offset))
                end = min(offset + page_size, size)
                pages.append((offset, end))
                if end == size:
                    break
                offset = end - self.transport["overlap"]
            cursors = {None: 0}
            for ordinal in range(1, len(pages)):
                cursors["s25_" + hashlib.sha256((identity + ":" + str(ordinal)).encode()).hexdigest()] = ordinal
            self._schedules[identity] = pages, cursors
        return self._schedules[identity]

    def get_gene_layers(self, multiplex_id: Any, gene_id: Any, layer_scope: Any,
                        mode: Any, top_k: Any, cursor: Any, *, evidence_id: str) -> dict[str, Any]:
        """Acquire one positive-membership page and independently report global presence."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, gene_id, layer_scope, mode, top_k, cursor)))
        try:
            selected = self._validate(arguments)
            present = gene_id in self.view.multiplex_nodes
            containing = [layer for layer in selected if gene_id in self.view.layers[layer].nodes]
            size = len(containing) if mode == "complete" else min(top_k, len(containing))
            pages, cursors = self._schedule(arguments, size)
            if cursor not in cursors:
                raise SelectionError("invalid_input", "INVALID_CURSOR")
        except SelectionError as error:
            return {"status": error.status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": error.issue_code, "arguments": arguments}}
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("a successful acquisition requires an evidence ID")
        scope = self.multiplex._scope(layer_scope, selected)
        ordinal = cursors[cursor]
        start, end = pages[ordinal]
        layers = containing[start:end]
        count = len(containing) if self.transport["disclose_total"] or not containing else None
        next_cursor = next((token for token, page in cursors.items() if page == ordinal + 1), None)
        return {
            "status": "resolved",
            "provenance": {"registry": [], "graph": [{
                "multiplex_id": self.view.multiplex_id, "graph_version": self.view.graph_version,
                "layer_scope": scope, "evidence_id": evidence_id,
            }]},
            "result": {"gene_id": gene_id, "multiplex_id": self.view.multiplex_id,
                       "layer_scope": scope, "present": present, "layer_ids": layers, "layer_count": count,
                       "coverage": {"operation": "layer_membership", "requested_layer_ids": list(selected),
                                    "checked_layer_ids": list(selected), "unchecked_layer_ids": []},
                       "collection": {"mode": mode, "returned_count": len(layers), "total_count": count,
                                      "complete": start == 0 and end == len(containing), "next_cursor": next_cursor}},
        }
