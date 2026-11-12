"""Answer S2.2 multiplex edge queries and inventory attainable source counts.

The runtime wraps the unchanged S2.1 native view. Saved-evidence helpers derive
answers from supplied observations only; the inventory is builder-side data and
must never supply an unobserved count to a bounded or existence answer.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Any

from .contract import CONTRACT_PATH, load_contract, parse_layer_ref
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN, file_sha256


EDGE_QUERIES = ("any_direction", "source_to_target", "both_directions")
MODES = ("complete", "top_k", "exists")


@lru_cache(maxsize=8)
def _family_contract(contract_sha256: str) -> dict[str, Any]:
    """Reuse the immutable grammar while actual file bytes still match the cache key."""
    return load_contract()


class SelectionError(ValueError):
    """Carry an atomic request rejection without creating graph evidence."""

    def __init__(self, status: str, issue_code: str) -> None:
        self.status = status
        self.issue_code = issue_code
        super().__init__(issue_code)


class EvidenceError(ValueError):
    """Reject incompatible, contradictory, or insufficient saved observations."""


def _pair(gene_ids: Any) -> tuple[str, str]:
    """Validate the canonical distinct-pair prerequisite, preserving order."""
    if not isinstance(gene_ids, (list, tuple)) or len(gene_ids) != 2:
        raise SelectionError("invalid_input", "INVALID_PAIR")
    if any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene)
           for gene in gene_ids):
        raise SelectionError("invalid_input", "INVALID_GENE_INPUT")
    if gene_ids[0] == gene_ids[1]:
        raise SelectionError("invalid_input", "INVALID_PAIR")
    return tuple(gene_ids)


def _selection(edge_query: Any, mode: Any, top_k: Any) -> None:
    """Reject unknown predicates and invalid bounded-retrieval combinations."""
    if edge_query not in EDGE_QUERIES or mode not in MODES:
        raise SelectionError("invalid_input", "INVALID_SELECTION")
    if (mode == "top_k" and (type(top_k) is not int or top_k <= 0)) or (
        mode != "top_k" and top_k is not None
    ):
        raise SelectionError("invalid_input", "INVALID_SELECTION")


def _requested_pairs(pair: tuple[str, str], directed: bool, query: str) -> list[tuple[str, str]]:
    """Retain native roles and the reference-forward/reverse presentation order."""
    if not directed:
        return [tuple(sorted(pair))]
    return [pair] if query == "source_to_target" else [pair, pair[::-1]]


def _predicate(requested: Sequence[tuple[str, str]], checked: set[tuple[str, str]],
               positives: Mapping[tuple[str, str], Any], query: str) -> bool:
    """Settle a layer predicate only when the checked orientations prove it."""
    if query == "both_directions":
        if any(pair not in positives for pair in checked):
            return False
        if all(pair in positives for pair in requested):
            return True
    elif positives:
        return True
    elif all(pair in checked for pair in requested):
        return False
    raise EvidenceError("supplied orientations do not settle the layer predicate")


def _layer_row(layer_id: str, directed: bool, requested: Sequence[tuple[str, str]],
               checked: set[tuple[str, str]], positives: Mapping[tuple[str, str], Any],
               outside: Sequence[str], query: str) -> dict[str, Any]:
    """Serialize a proved layer predicate without treating unchecked arcs as absent."""
    return {
        "layer_id": layer_id,
        "directed": directed,
        "edge_exists": _predicate(requested, checked, positives, query),
        "edges": [{"gene_ids": list(pair), "weight": positives[pair]}
                  for pair in requested if pair in positives],
        "not_in_layer_gene_ids": sorted(outside),
        "coverage": {
            "operation": "edge_checks",
            "requested_pairs": [list(pair) for pair in requested],
            "checked_pairs": [list(pair) for pair in requested if pair in checked],
            "unchecked_pairs": [list(pair) for pair in requested if pair not in checked],
        },
    }


class MultiplexRuntime:
    """Execute native multiplex predicates over one verified, versioned view."""

    def __init__(self, view: GraphView) -> None:
        self.view = view
        contract = _family_contract(file_sha256(CONTRACT_PATH))
        self.families = tuple(contract["vocabularies"]["layer_family"])
        self.layer_families = MappingProxyType({
            layer: parse_layer_ref(contract, layer)["layer_family"]
            for layer in view.layer_order
        })

    def family_of(self, layer_id: str) -> str:
        """Return the shared S1 grammar's family for a registered layer."""
        return self.layer_families[layer_id]

    def resolve_scope(self, layer_scope: Any, edge_query: str = "any_direction") -> tuple[str, ...]:
        """Resolve a typed scope in manifest order, including valid empty families."""
        if layer_scope is None:
            raise SelectionError("invalid_input", "MISSING_SCOPE")
        if not isinstance(layer_scope, dict) or set(layer_scope) != {
            "kind", "layer_ids", "layer_family",
        }:
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        kind, layers, family = (layer_scope[key] for key in ("kind", "layer_ids", "layer_family"))
        if not isinstance(layers, list) or any(not isinstance(layer, str) for layer in layers):
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        if kind == "layers":
            if not layers or family is not None:
                raise SelectionError("invalid_input", "INVALID_SELECTION")
            if any(layer not in self.view.layers for layer in layers):
                raise SelectionError("not_found", "UNKNOWN_LAYER")
            selected = set(layers)
            result = tuple(layer for layer in self.view.layer_order if layer in selected)
        elif kind == "all":
            if layers or (family is not None and family not in self.families):
                raise SelectionError("invalid_input", "INVALID_SELECTION")
            result = tuple(layer for layer in self.view.layer_order
                           if family is None or self.family_of(layer) == family)
        else:
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        if edge_query not in EDGE_QUERIES:
            raise SelectionError("invalid_input", "INVALID_SELECTION")
        if edge_query != "any_direction" and any(not self.view.layers[layer].directed for layer in result):
            raise SelectionError("invalid_input", "DIRECTION_UNAVAILABLE")
        return result

    def _scope(self, scope: Mapping[str, Any], selected: Sequence[str]) -> dict[str, Any]:
        """Normalize only explicit list order and duplicates in an accepted scope."""
        return {"kind": scope["kind"],
                "layer_ids": list(selected) if scope["kind"] == "layers" else [],
                "layer_family": scope["layer_family"]}

    def get_edge_layers(self, multiplex_id: Any, gene_ids: Any, layer_scope: Any,
                        edge_query: Any, mode: Any, top_k: Any, *, evidence_id: str) -> dict[str, Any]:
        """Check native orientations atomically, stopping at the requested layer bound.

        A visited directed layer always checks its whole requested orientation
        set, even when one arc already settles the Boolean. Callers own global
        evidence-ID uniqueness and preserve acquisition records when reusing them.
        """
        arguments = {"multiplex_id": multiplex_id, "gene_ids": gene_ids,
                     "layer_scope": layer_scope, "edge_query": edge_query,
                     "mode": mode, "top_k": top_k}
        try:
            if not multiplex_id or layer_scope is None:
                raise SelectionError("invalid_input", "MISSING_SCOPE")
            if multiplex_id != self.view.multiplex_id:
                raise SelectionError("not_found", "UNKNOWN_MULTIPLEX")
            if not isinstance(gene_ids, list):
                raise SelectionError("invalid_input", "INVALID_PAIR")
            pair = _pair(gene_ids)
            _selection(edge_query, mode, top_k)
            selected = self.resolve_scope(layer_scope, edge_query)
        except SelectionError as error:
            return {"status": error.status, "provenance": {"registry": [], "graph": []},
                    "result": {"issue_code": error.issue_code, "arguments": arguments}}
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError("a successful acquisition requires an evidence ID")
        missing = sorted(set(pair) - self.view.multiplex_nodes)
        rows = []
        positives = 0
        if not missing:
            for layer_id in selected:
                layer = self.view.layers[layer_id]
                requested = _requested_pairs(pair, layer.directed, edge_query)
                edges = {p: layer.edges[p] for p in requested if p in layer.edges}
                outside = sorted(set(pair) - layer.nodes)
                row = _layer_row(layer_id, layer.directed, requested, set(requested),
                                 edges, outside, edge_query)
                rows.append(row)
                positives += row["edge_exists"]
                if (mode == "exists" and positives) or (mode == "top_k" and positives >= top_k):
                    break
        result = self.assemble_result(pair, layer_scope, edge_query, mode, top_k, rows, missing)
        checked = result["coverage"]["checked_layer_ids"]
        evidence_scope = result["layer_scope"]
        if missing:
            evidence_scope = {"kind": "all", "layer_ids": [], "layer_family": None}
        elif result["coverage"]["unchecked_layer_ids"]:
            evidence_scope = {"kind": "layers", "layer_ids": checked, "layer_family": None}
        return {
            "status": "not_found" if missing else "resolved",
            "provenance": {"registry": [], "graph": [{
                "multiplex_id": self.view.multiplex_id,
                "graph_version": self.view.graph_version,
                "layer_scope": evidence_scope,
                "evidence_id": evidence_id,
            }]},
            "result": result,
        }

    def _merge_rows(self, pair: tuple[str, str], selected: Sequence[str], query: str,
                    rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Validate and merge supplied layer proofs without reading source edges."""
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            layer_id = row.get("layer_id")
            if layer_id not in selected:
                raise EvidenceError("saved layer is outside the requested scope")
            grouped[layer_id].append(row)
        result = []
        for layer_id in selected:
            if layer_id not in grouped:
                continue
            directed = self.view.layers[layer_id].directed
            requested = _requested_pairs(pair, directed, query)
            domain = set(requested)
            combined: dict[tuple[str, str], Any] = {}
            combined_checked: set[tuple[str, str]] = set()
            outside = None
            for row in grouped[layer_id]:
                if type(row.get("directed")) is not bool or row["directed"] != directed:
                    raise EvidenceError("saved representation differs from the view")
                coverage = row.get("coverage", {})
                if coverage.get("operation") != "edge_checks":
                    raise EvidenceError("saved layer requires edge-check coverage")
                try:
                    stated = [tuple(p) for p in coverage["requested_pairs"]]
                    checked_list = [tuple(p) for p in coverage["checked_pairs"]]
                    unchecked_list = [tuple(p) for p in coverage["unchecked_pairs"]]
                except (KeyError, TypeError) as error:
                    raise EvidenceError("malformed saved orientation coverage") from error
                checked, unchecked = set(checked_list), set(unchecked_list)
                if (len(stated) != len(domain) or set(stated) != domain
                        or len(checked) != len(checked_list)
                        or len(unchecked) != len(unchecked_list)
                        or checked & unchecked or checked | unchecked != domain):
                    raise EvidenceError("saved orientation coverage differs from the request")
                membership = row.get("not_in_layer_gene_ids")
                if (not isinstance(membership, list) or any(gene not in pair for gene in membership)
                        or membership != sorted(set(membership))):
                    raise EvidenceError("invalid saved layer-membership list")
                if outside is not None and membership != outside:
                    raise EvidenceError("contradictory saved layer-membership observations")
                outside = membership
                edges: dict[tuple[str, str], Any] = {}
                for edge in row.get("edges", []):
                    p = tuple(edge["gene_ids"])
                    weight = edge["weight"]
                    if (p not in checked or p in edges or type(weight) not in (int, float)
                            or not math.isfinite(weight)):
                        raise EvidenceError("invalid saved native edge")
                    edges[p] = weight
                if outside and edges:
                    raise EvidenceError("an absent layer endpoint cannot have a positive edge")
                predicate = _predicate(requested, checked, edges, query)
                if type(row.get("edge_exists")) is not bool or row["edge_exists"] != predicate:
                    raise EvidenceError("saved layer predicate contradicts its observations")
                for p in checked & combined_checked:
                    if (p in edges) != (p in combined) or (p in edges and edges[p] != combined[p]):
                        raise EvidenceError("contradictory saved edge observations")
                combined_checked.update(checked)
                combined.update(edges)
            result.append(_layer_row(layer_id, directed, requested, combined_checked,
                                     combined, outside or [], query))
        return result

    def assemble_result(self, gene_ids: Sequence[str], layer_scope: Mapping[str, Any],
                        edge_query: str, mode: str, top_k: int | None,
                        layer_results: Iterable[Mapping[str, Any]],
                        missing_gene_ids: Sequence[str] = ()) -> dict[str, Any]:
        """Combine supplied proofs into the ordinary result, without new acquisition.

        The caller supplies the checked multiplex-membership conclusion and is
        responsible for matching evidence identity/version. Complete requests
        need all layer predicates; top-K requests need a checked prefix through
        K matches or exhaustion. Existence needs a witness or exhaustion.
        Bounded projections retain richer checked rows, while their returned
        supporting-ID list contains only the first K proven matches.
        """
        pair = _pair(gene_ids)
        _selection(edge_query, mode, top_k)
        selected = self.resolve_scope(layer_scope, edge_query)
        missing = sorted(set(missing_gene_ids))
        if any(gene not in pair for gene in missing):
            raise EvidenceError("missing endpoint is outside the reference pair")
        rows = self._merge_rows(pair, selected, edge_query, layer_results)
        if missing and rows:
            raise EvidenceError("missing multiplex membership cannot license edge predicates")
        checked = [row["layer_id"] for row in rows]
        unchecked = [layer for layer in selected if layer not in checked]
        all_supports = [row["layer_id"] for row in rows if row["edge_exists"]]
        supports = all_supports[:top_k] if mode == "top_k" else all_supports
        exhaustive = not unchecked
        if not missing:
            if mode == "complete" and not exhaustive:
                raise EvidenceError("complete retrieval requires every selected layer")
            if mode == "exists" and not supports and not exhaustive:
                raise EvidenceError("existence requires a witness or exhaustive negatives")
            if mode == "top_k":
                if not exhaustive and len(supports) < top_k:
                    raise EvidenceError("first-K retrieval requires K witnesses or exhaustion")
                if supports:
                    through_last = selected[:selected.index(supports[-1]) + 1]
                    if any(layer not in checked for layer in through_last):
                        raise EvidenceError("first-K retrieval requires every preceding layer")
        count = len(all_supports) if exhaustive and not missing else None
        return {
            "multiplex_id": self.view.multiplex_id,
            "gene_ids": list(pair),
            "layer_scope": self._scope(layer_scope, selected),
            "edge_query": edge_query,
            "mode": mode,
            "edge_exists_any_layer": None if missing else bool(supports),
            "layer_results": None if missing else rows,
            "supporting_layer_ids": None if missing else supports,
            "supporting_layer_count": count,
            "missing_gene_ids": missing,
            "coverage": {
                "operation": "layer_edge_checks",
                "requested_layer_ids": list(selected),
                "checked_layer_ids": checked,
                "unchecked_layer_ids": unchecked,
            },
            "collection": None if missing or mode == "exists" else {
                "mode": mode, "returned_count": len(supports), "total_count": count,
                "complete": exhaustive and len(supports) == count, "next_cursor": None,
            },
        }

    def layer_from_edge(self, observation_result: Mapping[str, Any], edge_query: str,
                        reference_pair: Sequence[str]) -> dict[str, Any]:
        """Project one S2.1 observation when it already proves the layer predicate.

        A positive arc proves any-direction; a negative proves not-both. The
        other orientation remains unchecked unless endpoint absence proves it.
        Insufficient evidence raises EvidenceError instead of reading the graph.
        """
        pair = _pair(reference_pair)
        source = observation_result
        layer_id = source.get("layer_id")
        self.resolve_scope({"kind": "layers", "layer_ids": [layer_id], "layer_family": None}, edge_query)
        layer = self.view.layers[layer_id]
        observed = tuple(source.get("gene_ids", []))
        if (source.get("multiplex_id") != self.view.multiplex_id
                or type(source.get("directed")) is not bool or source["directed"] != layer.directed
                or set(observed) != set(pair) or len(observed) != 2
                or source.get("missing_gene_ids") != []
                or type(source.get("edge_exists")) is not bool):
            raise EvidenceError("saved edge has incompatible scope or missing prerequisites")
        if not layer.directed and observed != tuple(sorted(pair)):
            raise EvidenceError("saved undirected pair is not canonical")
        requested = _requested_pairs(pair, layer.directed, edge_query)
        outside = source.get("not_in_layer_gene_ids")
        if (not isinstance(outside, list) or any(gene not in pair for gene in outside)
                or outside != sorted(set(outside))):
            raise EvidenceError("invalid saved layer-membership list")
        positive = source["edge_exists"]
        weight = source.get("weight")
        if (positive and (outside or type(weight) not in (int, float) or not math.isfinite(weight))
                or not positive and weight is not None):
            raise EvidenceError("saved edge value contradicts its membership or weight")
        if outside:
            checked, edges = set(requested), {}
        else:
            checked = {observed} & set(requested)
            edges = {observed: weight} if positive and observed in checked else {}
        return _layer_row(layer_id, layer.directed, requested, checked, edges, outside, edge_query)

    def requery_layer(self, layer_result: Mapping[str, Any], edge_query: str,
                      reference_pair: Sequence[str]) -> dict[str, Any]:
        """Derive another predicate from supplied native orientation observations.

        Only checked orientations can supply positive or negative edge facts.
        Explicit endpoint absence proves all requested orientations negative.
        The caller validates the acquisition's original predicate and version;
        this helper never reads source edges or endpoint inventories.
        """
        pair = _pair(reference_pair)
        layer_id = layer_result.get("layer_id")
        if not isinstance(layer_id, str) or layer_id not in self.view.layers:
            raise EvidenceError("saved layer is not registered in this view")
        layer = self.view.layers[layer_id]
        if (type(layer_result.get("directed")) is not bool
                or layer_result["directed"] != layer.directed
                or type(layer_result.get("edge_exists")) is not bool):
            raise EvidenceError("saved layer has an invalid representation or predicate")
        if edge_query not in EDGE_QUERIES or (not layer.directed and edge_query != "any_direction"):
            raise EvidenceError("requested predicate is unavailable for the saved representation")
        native = set(_requested_pairs(pair, layer.directed, "any_direction"))
        coverage = layer_result.get("coverage", {})
        if coverage.get("operation") != "edge_checks":
            raise EvidenceError("saved layer requires edge-check coverage")
        try:
            original = [tuple(p) for p in coverage["requested_pairs"]]
            checked_list = [tuple(p) for p in coverage["checked_pairs"]]
            unchecked_list = [tuple(p) for p in coverage["unchecked_pairs"]]
            original_domain, checked, unchecked = set(original), set(checked_list), set(unchecked_list)
        except (KeyError, TypeError) as error:
            raise EvidenceError("malformed saved orientation coverage") from error
        if (not original_domain or not original_domain <= native
                or len(original_domain) != len(original)
                or len(checked) != len(checked_list) or len(unchecked) != len(unchecked_list)
                or checked & unchecked or checked | unchecked != original_domain):
            raise EvidenceError("saved orientation coverage differs from the reference pair")
        outside = layer_result.get("not_in_layer_gene_ids")
        if (not isinstance(outside, list) or any(gene not in pair for gene in outside)
                or outside != sorted(set(outside))):
            raise EvidenceError("invalid saved layer-membership list")
        positives = {}
        if not isinstance(layer_result.get("edges"), list):
            raise EvidenceError("saved edges must be an explicit list")
        for edge in layer_result["edges"]:
            try:
                observed, weight = tuple(edge["gene_ids"]), edge["weight"]
            except (KeyError, TypeError) as error:
                raise EvidenceError("malformed saved native edge") from error
            if (observed not in checked or observed in positives
                    or type(weight) not in (int, float) or not math.isfinite(weight)):
                raise EvidenceError("invalid saved native edge")
            positives[observed] = weight
        if outside and (positives or layer_result["edge_exists"]):
            raise EvidenceError("an absent layer endpoint cannot have a positive edge")
        requested = _requested_pairs(pair, layer.directed, edge_query)
        domain = set(requested)
        checked = domain if outside else checked & domain
        positives = {p: weight for p, weight in positives.items() if p in domain}
        return _layer_row(layer_id, layer.directed, requested, checked, positives, outside, edge_query)


class CandidateInventory:
    """Index positive native pairs once, deriving exact subset-count frequencies.

    Any-direction and reciprocal predicates use unordered distinct multiplex
    pairs. Source-to-target uses ordered pairs. Zero counts include all known
    multiplex endpoints, including endpoints outside selected layers. No missing
    endpoint or self-loop enters this sampling domain.
    """

    def __init__(self, view: GraphView) -> None:
        self.view = view
        self.nodes = tuple(sorted(view.multiplex_nodes))
        self.layer_order = view.layer_order
        self._positions = {layer: index for index, layer in enumerate(self.layer_order)}
        self._width = len(self.layer_order)
        self._all_bits = (1 << self._width) - 1
        records: dict[tuple[str, str], int] = {}
        for index, layer in enumerate(view.layers.values()):
            forward_bit, reverse_bit = 1 << index, 1 << (index + self._width)
            for left, right in layer.edges:
                if left == right:
                    continue
                pair = (left, right) if left < right else (right, left)
                bits = (forward_bit | reverse_bit) if not layer.directed else (
                    forward_bit if left < right else reverse_bit
                )
                records[pair] = records.get(pair, 0) | bits
        self.pairs = tuple(sorted(records))
        self._records = MappingProxyType(records)
        bins: dict[tuple[int, int], array] = defaultdict(lambda: array("Q"))
        for index, pair in enumerate(self.pairs):
            packed = records[pair]
            bins[(packed & self._all_bits, packed >> self._width)].append(index)
        self.mask_bins = MappingProxyType(dict(sorted(bins.items())))
        self.mask_frequencies = MappingProxyType({masks: len(indices) for masks, indices in self.mask_bins.items()})
        self.pair_universe_size = len(self.nodes) * (len(self.nodes) - 1) // 2

    def _mask(self, layer_ids: Iterable[str], edge_query: str) -> int:
        """Validate a resolved layer subset without assuming family implies direction."""
        if edge_query not in EDGE_QUERIES:
            raise ValueError("unknown edge predicate")
        mask = 0
        for layer in layer_ids:
            if layer not in self._positions:
                raise ValueError("unknown layer in candidate scope")
            if edge_query != "any_direction" and not self.view.layers[layer].directed:
                raise ValueError("directed predicate includes an undirected layer")
            mask |= 1 << self._positions[layer]
        return mask

    @staticmethod
    def _support(forward: int, reverse: int, mask: int, query: str) -> int:
        """Count qualifying layers, never individual directed arcs."""
        bits = forward | reverse if query == "any_direction" else (
            forward & reverse if query == "both_directions" else forward
        )
        return (bits & mask).bit_count()

    def histogram(self, layer_ids: Iterable[str], edge_query: str = "any_direction") -> dict[int, int]:
        """Count the full pair universe by support using compact positive-mask bins."""
        mask = self._mask(layer_ids, edge_query)
        counts: Counter[int] = Counter()
        unrecorded = self.pair_universe_size - len(self.pairs)
        counts[0] = unrecorded * (2 if edge_query == "source_to_target" else 1)
        for (forward, reverse), count in self.mask_frequencies.items():
            counts[self._support(forward, reverse, mask, edge_query)] += count
            if edge_query == "source_to_target":
                counts[self._support(reverse, forward, mask, edge_query)] += count
        return {support: count for support, count in sorted(counts.items()) if count}

    def iter_candidates(self, layer_ids: Iterable[str], edge_query: str,
                        support_count: int) -> Iterable[tuple[str, str]]:
        """Yield deterministic positive-count candidates without enumerating zeros.

        Zero sampling uses query_support on distinct global node pairs. Source
        candidates preserve their tested order, including reverse-only support.
        """
        if type(support_count) is not int or support_count <= 0:
            raise ValueError("candidate iteration requires a positive support count")
        mask = self._mask(layer_ids, edge_query)
        for (forward, reverse), indices in self.mask_bins.items():
            include_forward = self._support(forward, reverse, mask, edge_query) == support_count
            include_reverse = edge_query == "source_to_target" and (
                self._support(reverse, forward, mask, edge_query) == support_count
            )
            if include_forward or include_reverse:
                for index in indices:
                    pair = self.pairs[index]
                    if include_forward:
                        yield pair
                    if include_reverse:
                        yield pair[::-1]

    def query_support(self, pair: Sequence[str], layer_ids: Iterable[str],
                      edge_query: str = "any_direction") -> int:
        """Read an exact builder-only support count for one valid ordered reference."""
        reference = _pair(pair)
        if any(gene not in self.view.multiplex_nodes for gene in reference):
            raise ValueError("missing multiplex endpoints are not zero-support candidates")
        mask = self._mask(layer_ids, edge_query)
        packed = self._records.get(tuple(sorted(reference)), 0)
        forward, reverse = packed & self._all_bits, packed >> self._width
        if reference[0] > reference[1]:
            forward, reverse = reverse, forward
        return self._support(forward, reverse, mask, edge_query)
