"""Native S2.3 neighbor indexing, explicit pagination, and visible-page proofs.

Transport settings are declared test/build schedules, not graph facts. The
collection helper uses only supplied observations; its index is never consulted
to fill unseen neighbors, totals, or cursor state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from .s2_multiplex import EvidenceError, SelectionError
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ("multiplex_id", "layer_id", "gene_id", "direction", "mode", "top_k", "cursor")


def _rank(row: Mapping[str, Any]) -> tuple[int | float, str]:
    """Order exact native weights descending, breaking ties by canonical ID."""
    return -row["weight"], row["gene_id"]


def _selection(mode: Any, top_k: Any) -> None:
    """Validate logical retrieval independently of transport page boundaries."""
    if mode == "top_k" and top_k is None:
        raise SelectionError("invalid_input", "MISSING_LIMIT")
    if mode not in ("complete", "top_k") or (
        mode == "complete" and top_k is not None
    ) or (mode == "top_k" and (type(top_k) is not int or top_k <= 0)):
        raise SelectionError("invalid_input", "INVALID_SELECTION")


def _ordinary(arguments: Mapping[str, Any], neighbors: list | None, missing: list,
              outside: list, collection: dict | None) -> dict[str, Any]:
    """Keep the approved ordinary neighbor-result field order."""
    return {"gene_id": arguments["gene_id"], "multiplex_id": arguments["multiplex_id"],
            "layer_id": arguments["layer_id"], "direction": arguments["direction"],
            "neighbors": neighbors, "missing_gene_ids": missing,
            "not_in_layer_gene_ids": outside, "collection": collection}


class NeighborIndex:
    """Index every native directional neighbor, retaining each self-loop once."""

    def __init__(self, view: GraphView) -> None:
        self.view = view
        indexed = {}
        for layer_id, layer in view.layers.items():
            directions = ("in", "out") if layer.directed else ("undirected",)
            lists = {direction: {gene: [] for gene in layer.nodes} for direction in directions}
            for (source, target), weight in layer.edges.items():
                if layer.directed:
                    lists["out"][source].append((target, weight))
                    lists["in"][target].append((source, weight))
                else:
                    lists["undirected"][source].append((target, weight))
                    if source != target:
                        lists["undirected"][target].append((source, weight))
            indexed[layer_id] = MappingProxyType({
                direction: MappingProxyType({gene: tuple(sorted(rows, key=lambda row: (-row[1], row[0])))
                                             for gene, rows in by_gene.items()})
                for direction, by_gene in lists.items()
            })
        self.layers = MappingProxyType(indexed)

    def directions(self, layer_id: str) -> tuple[str, ...]:
        """Return only native directions, with incoming before outgoing."""
        return tuple(self.layers[layer_id])

    def neighbors(self, layer_id: str, gene_id: str, direction: str) -> tuple[tuple[str, int | float], ...]:
        """Return the full native ranked list for builder sampling or acquisition."""
        return self.layers[layer_id][direction].get(gene_id, ())

    def histogram(self, layer_id: str, direction: str) -> dict[int, int]:
        """Count distinct-neighbor cardinalities among actual layer members only."""
        return dict(sorted(Counter(len(rows) for rows in self.layers[layer_id][direction].values()).items()))

    def members_by_count(self, layer_id: str, direction: str) -> dict[int, tuple[str, ...]]:
        """Expose genuine count strata without conflating absence with member zero."""
        groups: dict[int, list[str]] = defaultdict(list)
        for gene, rows in self.layers[layer_id][direction].items():
            groups[len(rows)].append(gene)
        return {count: tuple(sorted(genes)) for count, genes in sorted(groups.items())}


@dataclass(frozen=True)
class CollectionProof:
    """A proved result or exact next call, plus unchanged relevant acquisitions."""

    result: dict[str, Any] | None
    next_arguments: dict[str, Any] | None
    observations: tuple[Mapping[str, Any], ...]
    provenance: dict[str, list]


class NeighborRuntime:
    """Serve the approved neighbor operation over a verified native graph view."""

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
        self.index = index if index is not None else NeighborIndex(view)
        self.transport = {"page_size": page_size, "disclose_total": disclose_total,
                          "overlap": overlap, "empty_first_page": empty_first_page,
                          "empty_page_at": empty_page_at}
        self._empty_at = 0 if empty_first_page else empty_page_at
        self._schedules: dict[str, tuple[list[tuple[int, int]], dict[str | None, int]]] = {}

    def _validate(self, arguments: Mapping[str, Any]) -> None:
        """Check scope, identity, native direction, and explicit selection arguments."""
        if set(arguments) != set(ARGUMENT_KEYS):
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        if not arguments["multiplex_id"] or not arguments["layer_id"]:
            raise SelectionError("invalid_input", "MISSING_SCOPE")
        if arguments["multiplex_id"] != self.view.multiplex_id:
            raise SelectionError("not_found", "UNKNOWN_MULTIPLEX")
        layer_id = arguments["layer_id"]
        if not isinstance(layer_id, str) or layer_id not in self.view.layers:
            raise SelectionError("not_found", "UNKNOWN_LAYER")
        gene = arguments["gene_id"]
        if not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene):
            raise SelectionError("invalid_input", "INVALID_GENE_INPUT")
        allowed = ("in", "out") if self.view.layers[layer_id].directed else ("undirected",)
        if arguments["direction"] not in allowed:
            raise SelectionError("invalid_input", "DIRECTION_UNAVAILABLE")
        _selection(arguments["mode"], arguments["top_k"])
        if arguments["cursor"] is not None and (not isinstance(arguments["cursor"], str) or not arguments["cursor"]):
            raise SelectionError("invalid_input", "INVALID_CURSOR")

    def _schedule(self, arguments: Mapping[str, Any], size: int) -> tuple[list[tuple[int, int]], dict[str | None, int]]:
        """Bind deterministic opaque cursor tokens to the entire traversal and transport."""
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
                token = "s23_" + hashlib.sha256((identity + ":" + str(ordinal)).encode()).hexdigest()
                cursors[token] = ordinal
            self._schedules[identity] = pages, cursors
        return self._schedules[identity]

    def get_neighbors(self, multiplex_id: Any, layer_id: Any, gene_id: Any,
                      direction: Any, mode: Any, top_k: Any, cursor: Any,
                      *, evidence_id: str) -> dict[str, Any]:
        """Acquire one page without imposing a hidden logical neighbor cap."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, layer_id, gene_id, direction, mode, top_k, cursor)))
        try:
            self._validate(arguments)
            missing = gene_id not in self.view.multiplex_nodes
            outside = not missing and gene_id not in self.view.layers[layer_id].nodes
            neighbors = self.index.neighbors(layer_id, gene_id, direction) if not missing else ()
            selected_size = len(neighbors) if mode == "complete" else min(top_k, len(neighbors))
            pages, cursors = self._schedule(arguments, selected_size)
            if cursor not in cursors or missing and cursor is not None:
                raise SelectionError("invalid_input", "INVALID_CURSOR")
        except SelectionError as error:
            return {"status": error.status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": error.issue_code, "arguments": arguments}}
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("a successful acquisition requires an evidence ID")
        collection = None
        rows = None
        if not missing:
            ordinal = cursors[cursor]
            start, end = pages[ordinal]
            rows = [{"gene_id": gene, "weight": weight} for gene, weight in neighbors[start:end]]
            next_cursor = next((token for token, page in cursors.items() if page == ordinal + 1), None)
            collection = {"mode": mode, "returned_count": len(rows),
                          "total_count": len(neighbors) if self.transport["disclose_total"] or not neighbors else None,
                          "complete": start == 0 and end == len(neighbors), "next_cursor": next_cursor}
        return {
            "status": "not_found" if missing else "resolved",
            "provenance": {"registry": [], "graph": [{
                "multiplex_id": self.view.multiplex_id, "graph_version": self.view.graph_version,
                "layer_scope": {"kind": "layers", "layer_ids": [layer_id], "layer_family": None},
                "evidence_id": evidence_id,
            }]},
            "result": _ordinary(arguments, rows, [gene_id] if missing else [],
                                [gene_id] if outside else [], collection),
        }

    def collect(self, arguments: Mapping[str, Any],
                observations: Sequence[Mapping[str, Any]]) -> CollectionProof:
        """Prove the requested list or identify its next exact acquisition.

        Target arguments start at cursor:null. Relevant saved prefixes remain
        attributed even when a changed mode/bound requires a new traversal.
        Each saved selection must have a connected initial-cursor prefix. All
        consistent relevant acquisitions are retained, including redundant ones.
        """
        self._validate(arguments)
        if arguments["cursor"] is not None:
            raise EvidenceError("a collection request starts at the initial null cursor")
        groups: dict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
        relevant = []
        provenance: dict[str, list] = {"registry": [], "graph": []}
        identities = {}
        membership = None
        for observation in observations:
            if observation.get("operation") != "get_neighbors":
                continue
            source_args = observation.get("arguments", {})
            if any(source_args.get(key) != arguments[key] for key in ("multiplex_id", "layer_id", "gene_id", "direction")):
                continue
            records = observation.get("provenance", {}).get("graph", [])
            if not records or any(record.get("graph_version") != self.view.graph_version
                                  or record.get("multiplex_id") != self.view.multiplex_id for record in records):
                continue
            self._validate(source_args)
            source = observation["result"]
            if any(source.get(key) != source_args[key] for key in ("multiplex_id", "layer_id", "gene_id", "direction")):
                raise EvidenceError("saved page subjects differ from its acquisition arguments")
            missing, outside = source.get("missing_gene_ids"), source.get("not_in_layer_gene_ids")
            if missing not in ([], [arguments["gene_id"]]) or outside not in ([], [arguments["gene_id"]]) or missing and outside:
                raise EvidenceError("invalid saved neighbor membership lists")
            current_membership = (tuple(missing), tuple(outside))
            if membership is not None and membership != current_membership:
                raise EvidenceError("contradictory saved memberships")
            membership = current_membership
            for record in records:
                identity = record.get("evidence_id")
                if not isinstance(identity, str) or not identity:
                    raise EvidenceError("saved acquisition requires an evidence ID")
                if identity in identities and identities[identity] != observation:
                    raise EvidenceError("one evidence ID names contradictory acquisitions")
                identities[identity] = observation
            if observation not in relevant:
                relevant.append(observation)
            for channel in provenance:
                for record in observation["provenance"].get(channel, []):
                    if record not in provenance[channel]:
                        provenance[channel].append(record)
            if missing:
                if observation["status"] != "not_found" or source["neighbors"] is not None or source["collection"] is not None:
                    raise EvidenceError("missing multiplex membership must leave neighbors unavailable")
                continue
            if observation["status"] != "resolved":
                raise EvidenceError("a saved neighbor page must resolve its membership prerequisites")
            groups[(source_args["mode"], source_args["top_k"])].append(observation)
        used = tuple(relevant)
        if membership and membership[0]:
            return CollectionProof(_ordinary(arguments, None, list(membership[0]), [], None), None, used, provenance)
        summaries = {selection: _traversal(pages) for selection, pages in groups.items()}
        longest: list[dict[str, Any]] = []
        known_total = None
        full = False
        for summary in summaries.values():
            prefix, total, complete, _ = summary
            common = min(len(longest), len(prefix))
            if longest[:common] != prefix[:common]:
                raise EvidenceError("saved selections disagree about the ranked neighbor prefix")
            if len(prefix) > len(longest):
                longest = prefix
            if total is not None:
                if known_total is not None and known_total != total:
                    raise EvidenceError("inconsistent non-null neighbor totals")
                known_total = total
            full = full or complete
        if known_total is not None:
            if len(longest) > known_total:
                raise EvidenceError("observed distinct neighbors exceed the known total")
            full = full or len(longest) == known_total
        sufficient = bool(used) and (full or arguments["mode"] == "top_k" and len(longest) >= arguments["top_k"])
        if sufficient:
            if full:
                known_total = len(longest)
            rows = longest if arguments["mode"] == "complete" else longest[:arguments["top_k"]]
            result = _ordinary(arguments, deepcopy(rows), [], list(membership[1]), {
                "mode": arguments["mode"], "returned_count": len(rows), "total_count": known_total,
                "complete": full and len(rows) == known_total, "next_cursor": None,
            })
            return CollectionProof(result, None, used, provenance)
        next_arguments = dict(arguments)
        matching = summaries.get((arguments["mode"], arguments["top_k"]))
        if matching:
            next_arguments["cursor"] = matching[3]
            if next_arguments["cursor"] is None:
                raise EvidenceError("exhausted selection did not establish its requested result")
        return CollectionProof(None, next_arguments, used, provenance)


def _traversal(observations: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int | None, bool, str | None]:
    """Validate one visible initial-prefix chain without decoding opaque cursors."""
    by_cursor = {}
    for observation in observations:
        cursor = observation["arguments"]["cursor"]
        if cursor in by_cursor and by_cursor[cursor]["result"] != observation["result"]:
            raise EvidenceError("one cursor names contradictory pages")
        by_cursor[cursor] = observation
    if None not in by_cursor:
        raise EvidenceError("saved traversal lacks its initial null-cursor page")
    cursor = None
    visited = set()
    accumulated = []
    weights = {}
    known_total = None
    full = False
    while cursor in by_cursor:
        if cursor in visited:
            raise EvidenceError("saved cursor chain loops")
        visited.add(cursor)
        observation = by_cursor[cursor]
        arguments, result = observation["arguments"], observation["result"]
        collection = result.get("collection")
        rows = result.get("neighbors")
        if not isinstance(collection, dict) or not isinstance(rows, list):
            raise EvidenceError("resolved neighbor pages need a collection and row list")
        if (collection.get("mode") != arguments["mode"]
                or type(collection.get("returned_count")) is not int or collection["returned_count"] != len(rows)
                or type(collection.get("complete")) is not bool):
            raise EvidenceError("saved page collection metadata is inconsistent")
        total = collection.get("total_count")
        if total is not None and (type(total) is not int or total < 0):
            raise EvidenceError("saved total must be a nonnegative integer or null")
        if total is not None:
            if known_total is not None and total != known_total:
                raise EvidenceError("inconsistent non-null neighbor totals")
            known_total = total
            if collection["complete"] != (len(rows) == total):
                raise EvidenceError("page complete flag differs from its full-set coverage")
        row_ids = set()
        for row in rows:
            gene, weight = row.get("gene_id"), row.get("weight")
            if (not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene)
                    or gene in row_ids or type(weight) not in (int, float) or not math.isfinite(weight)):
                raise EvidenceError("invalid or duplicate native neighbor row")
            row_ids.add(gene)
        if rows != sorted(rows, key=_rank):
            raise EvidenceError("saved neighbor page violates weight/ID ordering")
        if result["not_in_layer_gene_ids"] and (rows or total != 0 or not collection["complete"]):
            raise EvidenceError("layer absence must supply a complete known empty list")
        for row in rows:
            gene, weight = row["gene_id"], row["weight"]
            if gene in weights:
                if weights[gene] != weight:
                    raise EvidenceError("one neighbor has conflicting weights")
                continue
            if accumulated and _rank(row) < _rank(accumulated[-1]):
                raise EvidenceError("saved pages violate global weight/ID ordering")
            weights[gene] = weight
            accumulated.append(dict(row))
        if known_total is not None and len(accumulated) > known_total:
            raise EvidenceError("observed distinct neighbors exceed the known total")
        cursor = collection.get("next_cursor")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise EvidenceError("saved continuation cursor must be nonempty or null")
        if collection["complete"]:
            if cursor is not None or len(accumulated) != len(rows):
                raise EvidenceError("a whole-set page conflicts with earlier or future pages")
            if known_total is not None and known_total != len(rows):
                raise EvidenceError("whole-set page contradicts its total")
            known_total, full = len(rows), True
        if cursor is None:
            break
    if len(visited) != len(by_cursor):
        raise EvidenceError("saved pages are disconnected from the initial cursor chain")
    mode, top_k = observations[0]["arguments"]["mode"], observations[0]["arguments"]["top_k"]
    if mode == "top_k" and len(accumulated) > top_k:
        raise EvidenceError("bounded traversal exceeds its requested K")
    if cursor is None and (mode == "complete" or len(accumulated) < top_k):
        if known_total is not None and known_total != len(accumulated):
            raise EvidenceError("exhausted traversal contradicts the full total")
        known_total, full = len(accumulated), True
    return accumulated, known_total, full, cursor
