"""Classify finite S2.6 gene lists from authoritative node inventories.

Every successful batch covers its full input. Degree and direction do not
change node membership, and positive counts cover only the requested genes.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .s2_multiplex import SelectionError
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ("multiplex_id", "layer_id", "gene_ids")


class NodesByLayerRuntime:
    """Serve complete finite classifications over one versioned graph view."""

    def __init__(self, view: GraphView) -> None:
        self.view = view

    def arguments(self, layer_id: Any, gene_ids: Any) -> dict[str, Any]:
        """Copy all three explicit arguments without changing the supplied input."""
        return {"multiplex_id": self.view.multiplex_id, "layer_id": deepcopy(layer_id),
                "gene_ids": deepcopy(gene_ids)}

    def _validate(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        """Reject the whole batch or return its unique IDs in first-mention order."""
        multiplex_id, layer_id, gene_ids = (arguments.get(key) for key in ARGUMENT_KEYS)
        if not multiplex_id or not layer_id:
            raise SelectionError("invalid_input", "MISSING_SCOPE")
        if multiplex_id != self.view.multiplex_id:
            raise SelectionError("not_found", "UNKNOWN_MULTIPLEX")
        if not isinstance(layer_id, str) or layer_id not in self.view.layers:
            raise SelectionError("not_found", "UNKNOWN_LAYER")
        if gene_ids is None:
            raise SelectionError("invalid_input", "MISSING_GENES")
        if set(arguments) != set(ARGUMENT_KEYS) or not isinstance(gene_ids, list) or any(
            not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene)
            for gene in gene_ids
        ):
            raise SelectionError("invalid_input", "INVALID_GENE_LIST")
        return tuple(dict.fromkeys(gene_ids))

    def get_nodes_by_layer(self, multiplex_id: Any, layer_id: Any, gene_ids: Any,
                           *, evidence_id: str) -> dict[str, Any]:
        """Partition every supplied canonical ID and retain its exact scope evidence."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, layer_id, gene_ids)))
        try:
            requested = self._validate(arguments)
        except SelectionError as error:
            return {"status": error.status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": error.issue_code, "arguments": deepcopy(arguments)}}
        if requested and (not isinstance(evidence_id, str) or not evidence_id.strip()):
            raise ValueError("a successful acquisition requires an evidence ID")
        layer_nodes = self.view.layers[layer_id].nodes
        positive = sorted(gene for gene in requested if gene in layer_nodes)
        outside = sorted(gene for gene in requested if gene in self.view.multiplex_nodes and gene not in layer_nodes)
        missing = sorted(gene for gene in requested if gene not in self.view.multiplex_nodes)
        provenance = {"registry": [], "graph": []}
        if requested:
            provenance["graph"].append({
                "multiplex_id": self.view.multiplex_id, "graph_version": self.view.graph_version,
                "layer_scope": {"kind": "layers", "layer_ids": [layer_id], "layer_family": None},
                "evidence_id": evidence_id,
            })
        return {
            "status": "resolved", "provenance": provenance,
            "result": {
                "multiplex_id": self.view.multiplex_id, "layer_id": layer_id,
                "layer_gene_ids": positive, "layer_gene_count": len(positive),
                "not_in_layer_gene_ids": outside, "missing_gene_ids": missing,
                "coverage": {"operation": "layer_membership", "requested_gene_ids": list(requested),
                             "checked_gene_ids": list(requested), "unchecked_gene_ids": []},
                "collection": {"mode": "complete", "returned_count": len(positive),
                               "total_count": len(positive), "complete": True, "next_cursor": None},
            },
        }
