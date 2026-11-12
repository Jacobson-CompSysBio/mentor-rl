"""Count distinct neighbor unions and apply the declared S2.8 rank policy.

The index uses every authoritative selected-layer member. Query splits and
registry entries never filter the reference population or the neighbor sets.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any

from .s2_multiplex import MultiplexRuntime, SelectionError
from .s2_neighbors import NeighborIndex
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


ARGUMENT_KEYS = ('multiplex_id', 'layer_scope', 'gene_id', 'direction', 'request_kind', 'hub_policy')
DIRECTIONS = ('incident', 'in', 'out', 'undirected')
DEGREE_POLICY_ID = 'distinct_neighbor_union_v1'
POPULATION_POLICY_ID = 'selected_layer_node_union_v1'
HUB_POLICY_ID = 'degree_midrank_v1'
CAVEAT = 'high_degree_inflates_apparent_proximity'


def _rank_values(rank_support: Mapping[str, Any]) -> tuple[int, int, int]:
    """Validate exact integer support before any percentile calculation."""
    if not isinstance(rank_support, Mapping) or set(rank_support) != {
            'population_policy_id', 'lower_degree_count', 'equal_degree_count', 'population_count'}:
        raise ValueError('exact rank support has unexpected fields')
    if rank_support['population_policy_id'] != POPULATION_POLICY_ID:
        raise ValueError('unsupported population policy')
    lower, equal, size = (rank_support[key] for key in (
        'lower_degree_count', 'equal_degree_count', 'population_count'))
    if (any(type(value) is not int for value in (lower, equal, size))
            or lower < 0 or equal < 1 or size < 1 or lower + equal > size):
        raise ValueError('rank support must describe an included population member')
    return lower, equal, size


def percentile(rank_support: Mapping[str, Any]) -> float:
    """Display the exact average rank with six decimal places and half-up rounding."""
    lower, equal, size = _rank_values(rank_support)
    if size == 1:
        return 50.0
    with localcontext() as context:
        context.prec = max(40, len(str(size)) * 3)
        value = Decimal(100 * (2 * lower + equal - 1)) / Decimal(2 * (size - 1))
        return float(value.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))


def hub_like(rank_support: Mapping[str, Any], threshold: int) -> bool:
    """Compare exact integer support with an inclusive declared threshold."""
    lower, equal, size = _rank_values(rank_support)
    if type(threshold) is not int or not 0 <= threshold <= 100:
        raise ValueError('a hub threshold must be an integer from zero through one hundred')
    return 50 >= threshold if size == 1 else 100 * (2 * lower + equal - 1) >= threshold * 2 * (size - 1)


def final_result(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Add the exact policy label and its warning to a supported measurement."""
    result = deepcopy(dict(measurement))
    label = None
    if measurement['request_kind'] == 'degree_and_hub' and measurement['rank_support'] is not None:
        label = hub_like(measurement['rank_support'], measurement['hub_policy']['threshold_percentile'])
    result.update(hub_like=label, caveats=[CAVEAT] if label else [])
    return result


class DegreeIndex:
    """Cache source counts and histograms without retaining large union sets."""

    def __init__(self, view: GraphView, index: NeighborIndex | None = None) -> None:
        if index is not None and index.view is not view:
            raise ValueError('a shared native index must belong to this exact immutable view')
        self.view = view
        self.native_index = index if index is not None else NeighborIndex(view)
        self.global_nodes = view.multiplex_nodes
        self._populations: dict[tuple[str, ...], frozenset[str]] = {}
        self._counts: dict[tuple[tuple[str, ...], str], dict[str, int]] = {}
        self._histograms: dict[tuple[tuple[str, ...], str], tuple[tuple[int, int], ...]] = {}
        self._ranks: dict[tuple[tuple[str, ...], str], dict[int, tuple[int, int, int]]] = {}

    def population(self, layer_ids: Sequence[str]) -> frozenset[str]:
        """Return the full selected-layer node union, including directional zeros."""
        selected = tuple(layer_ids)
        if selected not in self._populations:
            self._populations[selected] = frozenset(gene for layer in selected for gene in self.view.layers[layer].nodes)
        return self._populations[selected]

    def neighbor_ids(self, layer_ids: Sequence[str], gene_id: str, direction: str) -> frozenset[str]:
        """Deduplicate native neighbors across layers and reciprocal directions."""
        return frozenset(neighbor for layer in layer_ids
                         for native in (self.native_index.directions(layer) if direction == 'incident' else (direction,))
                         for neighbor, _ in self.native_index.neighbors(layer, gene_id, native))

    def degree(self, layer_ids: Sequence[str], gene_id: str, direction: str) -> int:
        """Count a source union; a member outside selected layers has degree zero."""
        key = (tuple(layer_ids), direction)
        counts = self._counts.get(key)
        if counts is not None:
            return counts.get(gene_id, 0)
        return len(self.neighbor_ids(layer_ids, gene_id, direction))

    def histogram(self, layer_ids: Sequence[str], direction: str) -> list[dict[str, int]]:
        """Count every attainable degree among all selected-layer members."""
        key = (tuple(layer_ids), direction)
        if key not in self._histograms:
            counts = {gene: len(self.neighbor_ids(key[0], gene, direction)) for gene in self.population(key[0])}
            histogram = tuple(sorted(Counter(counts.values()).items()))
            self._counts[key] = counts
            self._histograms[key] = histogram
            lower, size = 0, len(counts)
            ranks = {}
            for degree, equal in histogram:
                ranks[degree] = (lower, equal, size)
                lower += equal
            self._ranks[key] = ranks
        return [{'degree': degree, 'node_count': count} for degree, count in self._histograms[key]]

    def rank_support(self, layer_ids: Sequence[str], gene_id: str, direction: str) -> dict[str, Any] | None:
        """Return the exact rank of a population member without a registry filter."""
        selected = tuple(layer_ids)
        if gene_id not in self.population(selected):
            return None
        self.histogram(selected, direction)
        key = (selected, direction)
        lower, equal, size = self._ranks[key][self._counts[key][gene_id]]
        return {'population_policy_id': POPULATION_POLICY_ID, 'lower_degree_count': lower,
                'equal_degree_count': equal, 'population_count': size}


class DegreeRuntime:
    """Acquire one exact degree and, when requested, its full-population rank."""

    def __init__(self, view: GraphView, *, index: DegreeIndex | None = None) -> None:
        if index is not None and index.view is not view:
            raise ValueError('a shared degree index must belong to this exact immutable view')
        self.view, self.index = view, index
        self.multiplex = MultiplexRuntime(view)

    def resolve_scope(self, layer_scope: Any, direction: str = 'incident') -> tuple[str, ...]:
        """Resolve typed scope from manifest metadata without source inventories."""
        if (isinstance(layer_scope, Mapping) and isinstance(layer_scope.get('layer_ids'), list)
                and all(isinstance(layer, str) for layer in layer_scope['layer_ids'])
                and len(layer_scope['layer_ids']) != len(set(layer_scope['layer_ids']))):
            raise SelectionError('invalid_input', 'INVALID_SCOPE')
        try:
            selected = self.multiplex.resolve_scope(layer_scope)
        except SelectionError as error:
            if error.issue_code == 'INVALID_SELECTION':
                raise SelectionError('invalid_input', 'INVALID_SCOPE') from error
            raise
        if (direction not in DIRECTIONS
                or direction in ('in', 'out') and any(not self.view.layers[layer].directed for layer in selected)
                or direction == 'undirected' and any(self.view.layers[layer].directed for layer in selected)):
            raise SelectionError('invalid_input', 'DIRECTION_UNAVAILABLE')
        return selected

    def validate_arguments(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        """Reject all malformed prerequisites before a source acquisition starts."""
        if set(arguments) != set(ARGUMENT_KEYS):
            raise SelectionError('invalid_input', 'INVALID_SCOPE')
        if not arguments['multiplex_id'] or arguments['layer_scope'] is None:
            raise SelectionError('invalid_input', 'MISSING_SCOPE')
        if arguments['multiplex_id'] != self.view.multiplex_id:
            raise SelectionError('not_found', 'UNKNOWN_MULTIPLEX')
        gene = arguments['gene_id']
        if gene is None or gene == '':
            raise SelectionError('invalid_input', 'MISSING_GENE')
        if not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene):
            raise SelectionError('invalid_input', 'INVALID_GENE_INPUT')
        selected = self.resolve_scope(arguments['layer_scope'], arguments['direction'])
        kind, policy = arguments['request_kind'], arguments['hub_policy']
        if kind == 'degree_only' and policy is None:
            return selected
        if (kind != 'degree_and_hub' or not isinstance(policy, Mapping)
                or set(policy) != {'policy_id', 'threshold_percentile'}
                or policy['policy_id'] != HUB_POLICY_ID or type(policy['threshold_percentile']) is not int
                or not 0 <= policy['threshold_percentile'] <= 100):
            raise SelectionError('invalid_input', 'INVALID_HUB_POLICY')
        return selected

    def get_degree(self, multiplex_id: Any, layer_scope: Any, gene_id: Any, direction: Any,
                   request_kind: Any, hub_policy: Any, *, evidence_id: str) -> dict[str, Any]:
        """Return exact source counts and policy support with an attributed scope."""
        arguments = dict(zip(ARGUMENT_KEYS, (multiplex_id, layer_scope, gene_id, direction, request_kind, hub_policy)))
        try:
            selected = self.validate_arguments(arguments)
        except SelectionError as error:
            return {'status': error.status, 'provenance': {'registry': [], 'graph': []},
                    'result': {'issue_code': error.issue_code, 'arguments': arguments}}
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise ValueError('a successful acquisition requires an evidence ID')
        if self.index is None:
            self.index = DegreeIndex(self.view)
        present = gene_id in self.index.global_nodes
        in_scope = gene_id in self.index.population(selected) if present else None
        rank = self.index.rank_support(selected, gene_id, direction) if in_scope and request_kind == 'degree_and_hub' else None
        result = {'gene_id': gene_id, 'multiplex_id': multiplex_id,
            'layer_scope': self.multiplex._scope(layer_scope, selected), 'resolved_layer_ids': list(selected),
            'direction': direction, 'request_kind': request_kind, 'degree_policy_id': DEGREE_POLICY_ID,
            'membership': {'present_in_multiplex': present, 'present_in_scope': in_scope},
            'degree': self.index.degree(selected, gene_id, direction) if present else None,
            'degree_percentile': percentile(rank) if rank is not None else None,
            'rank_support': rank, 'hub_policy': deepcopy(hub_policy)}
        return {'status': 'resolved' if present and (request_kind == 'degree_only' or in_scope) else 'not_found',
            'provenance': {'registry': [], 'graph': [{'multiplex_id': multiplex_id,
                'graph_version': self.view.graph_version, 'layer_scope': deepcopy(result['layer_scope']),
                'evidence_id': evidence_id}]}, 'result': result}
