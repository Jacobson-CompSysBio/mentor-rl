"""Derive degrees and exact policy ranks from visible attributed observations.

The planner uses manifest metadata only. Source inventories and adjacency stay
behind the native runtime and cannot fill a missing count or population rank.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any

from .s2_4_evidence import _native_inputs, _scope as _map_scope, _traversal as _map_traversal, _validate_rows
from .s2_5_evidence import _global_membership, _map_membership, _neighbor_membership
from .s2_5_evidence import _traversal as _membership_traversal
from .s2_6_evidence import _membership_page, _nodes_partition
from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import EvidenceError, SelectionError
from .s2_neighbors import _rank, _traversal as _native_traversal
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


DEGREE_POLICY = 'distinct_neighbor_union_v1'
POPULATION_POLICY = 'selected_layer_node_union_v1'


@dataclass(frozen=True)
class DegreeProof:
    """Supply a completed statistic, a prerequisite issue, or the exact next call."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    provenance: dict[str, list]
    issue_code: str | None = None
    coverage: None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _direction(view: GraphView, selected: Sequence[str], direction: str) -> str:
    """Use manifest evidence to recognize equivalent undirected operations."""
    return 'undirected' if direction == 'incident' and all(not view.layers[layer].directed for layer in selected) else direction


def _rank_support(value: Any) -> dict:
    """Check exact average-rank support without consulting a source population."""
    keys = {'population_policy_id', 'lower_degree_count', 'equal_degree_count', 'population_count'}
    if not isinstance(value, Mapping) or set(value) != keys or value['population_policy_id'] != POPULATION_POLICY:
        raise EvidenceError('saved exact rank uses an unsupported population policy')
    lower, equal, total = (value[key] for key in ('lower_degree_count', 'equal_degree_count', 'population_count'))
    if (any(type(number) is not int for number in (lower, equal, total))
            or lower < 0 or equal < 1 or total < 1 or lower + equal > total):
        raise EvidenceError('saved exact rank support has invalid integer counts')
    return dict(value)


def validate_histogram(view: GraphView, observation: Mapping[str, Any]) -> tuple[tuple[str, ...], dict]:
    """Validate the approved complete distribution format from visible fields."""
    from .s2_degrees import DegreeRuntime
    runtime = DegreeRuntime(view)
    source, result = observation['arguments'], observation['result']
    fields = {'multiplex_id', 'layer_scope', 'direction', 'degree_policy_id', 'population_policy_id'}
    if (set(source) != fields or not isinstance(result, Mapping)
            or set(result) != fields | {'resolved_layer_ids', 'population_count', 'degree_histogram', 'complete'}
            or observation['status'] != 'resolved' or source['multiplex_id'] != view.multiplex_id):
        raise EvidenceError('saved distribution has unexpected fields or acquisition status')
    try:
        selected = runtime.resolve_scope(source['layer_scope'], source['direction'])
    except SelectionError as error:
        raise EvidenceError('saved distribution uses invalid scope or direction') from error
    scope = runtime.multiplex._scope(source['layer_scope'], selected)
    if (result['multiplex_id'] != source['multiplex_id'] or result['layer_scope'] != scope
            or result['resolved_layer_ids'] != list(selected)
            or any(result[key] != source[key] for key in ('direction', 'degree_policy_id', 'population_policy_id'))
            or type(result['population_count']) is not int or result['population_count'] < 0
            or type(result['complete']) is not bool or not isinstance(result['degree_histogram'], list)):
        raise EvidenceError('saved distribution identity or population metadata is inconsistent')
    rows, previous, total = result['degree_histogram'], -1, 0
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {'degree', 'node_count'}
                or type(row['degree']) is not int or type(row['node_count']) is not int
                or row['degree'] <= previous or row['node_count'] <= 0):
            raise EvidenceError('saved distribution rows must have ascending unique degrees and positive counts')
        previous, total = row['degree'], total + row['node_count']
    if total > result['population_count'] or result['complete'] and total != result['population_count']:
        raise EvidenceError('saved distribution does not account for its declared population')
    if not selected and (result['population_count'] or rows):
        raise EvidenceError('an empty resolved scope must have an empty reference population')
    return selected, dict(result)


def _measurement(runtime, observation: Mapping[str, Any]) -> tuple[tuple[str, ...], dict]:
    """Check a saved native statistic while retaining incomplete rank evidence."""
    from .s2_degrees import percentile
    source, result = observation['arguments'], observation['result']
    try:
        selected = runtime.validate_arguments(source)
    except SelectionError as error:
        raise EvidenceError('saved degree acquisition has invalid arguments') from error
    fields = {'gene_id', 'multiplex_id', 'layer_scope', 'resolved_layer_ids', 'direction', 'request_kind',
              'degree_policy_id', 'membership', 'degree', 'degree_percentile', 'rank_support', 'hub_policy'}
    if not isinstance(result, Mapping) or set(result) != fields or observation['status'] not in ('resolved', 'not_found'):
        raise EvidenceError('saved degree measurement has unexpected fields or status')
    scope = runtime.multiplex._scope(source['layer_scope'], selected)
    if (any(result[key] != source[key] for key in ('gene_id', 'multiplex_id', 'direction', 'request_kind', 'hub_policy'))
            or result['layer_scope'] != scope or result['resolved_layer_ids'] != list(selected)):
        raise EvidenceError('saved degree measurement differs from its acquisition identity')
    membership = result['membership']
    if (not isinstance(membership, Mapping) or set(membership) != {'present_in_multiplex', 'present_in_scope'}
            or type(membership['present_in_multiplex']) is not bool
            or membership['present_in_scope'] is not None and type(membership['present_in_scope']) is not bool):
        raise EvidenceError('saved degree membership fields must be explicit booleans or null')
    degree, rank, display = result['degree'], result['rank_support'], result['degree_percentile']
    if not membership['present_in_multiplex']:
        if (membership['present_in_scope'] is not None or degree is not None or rank is not None
                or display is not None or observation['status'] != 'not_found'):
            raise EvidenceError('global absence must leave degree and rank unavailable')
    elif type(degree) is not int or degree < 0:
        raise EvidenceError('a present query requires a nonnegative integer degree')
    if membership['present_in_scope'] is False and degree != 0:
        raise EvidenceError('an excluded query cannot have selected-scope neighbors')
    if not selected and membership['present_in_multiplex'] and (degree != 0 or membership['present_in_scope'] is not False):
        raise EvidenceError('an empty selected scope requires known zero and false scoped membership')
    if rank is not None:
        if not isinstance(rank, Mapping) or not isinstance(rank.get('population_policy_id'), str):
            raise EvidenceError('saved rank support must declare its population policy')
        _rank_support({**rank, 'population_policy_id': POPULATION_POLICY})
        compatible_rank = rank['population_policy_id'] == POPULATION_POLICY
        if (membership['present_in_scope'] is not True or source['request_kind'] != 'degree_and_hub'
                or compatible_rank and display != percentile(rank)):
            raise EvidenceError('saved exact rank conflicts with membership, request kind, or display')
    if display is not None and (type(display) not in (int, float) or not math.isfinite(display) or not 0 <= display <= 100):
        raise EvidenceError('saved percentile display is not a finite percentile')
    if source['request_kind'] == 'degree_only' and (rank is not None or display is not None):
        raise EvidenceError('a degree-only acquisition cannot claim an unrequested rank')
    return selected, dict(result)


def _native_page(observation: Mapping[str, Any]) -> None:
    """Check a page even when its initial cursor page is not supplied."""
    source, result = observation['arguments'], observation['result']
    rows, collection = result['neighbors'], result['collection']
    if (set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
            or collection['mode'] != source['mode'] or type(collection['returned_count']) is not int
            or collection['returned_count'] != len(rows) or type(collection['complete']) is not bool
            or source['mode'] == 'top_k' and len(rows) > source['top_k']):
        raise EvidenceError('saved native page has invalid collection fields')
    total, cursor = collection['total_count'], collection['next_cursor']
    if (total is not None and (type(total) is not int or total < len(rows))
            or total is not None and collection['complete'] != (len(rows) == total)
            or cursor is not None and (not isinstance(cursor, str) or not cursor)
            or collection['complete'] and cursor is not None):
        raise EvidenceError('saved native page count or cursor is inconsistent')
    ids = []
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {'gene_id', 'weight'}
                or not isinstance(row['gene_id'], str) or not GENE_ID_PATTERN.fullmatch(row['gene_id'])
                or type(row['weight']) not in (int, float) or not math.isfinite(row['weight'])):
            raise EvidenceError('saved native page has invalid neighbor rows')
        ids.append(row['gene_id'])
    if len(set(ids)) != len(ids) or rows != sorted(rows, key=_rank):
        raise EvidenceError('saved native page has duplicate IDs or invalid native order')


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]) -> DegreeProof:
    """Prove the count and requested rank, or acquire the missing native statistic."""
    from .s2_degrees import DegreeRuntime, final_result, percentile
    runtime = DegreeRuntime(view)
    selected = runtime.validate_arguments(arguments)
    requested = frozenset(selected)
    query, direction = arguments['gene_id'], _direction(view, selected, arguments['direction'])
    memberships, nodes = GeneLayersRuntime(view), NodesByLayerRuntime(view)
    ids, compatible, global_values, scope_values = {}, [], [], []
    counts, ranks, distributions, positives = [], [], [], []
    native_groups, map_groups, member_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    degree_weights = {}

    def global_fact(present: bool, index: int) -> None:
        if global_values and global_values[0][0] != present:
            raise EvidenceError('saved observations contradict global membership')
        global_values.append((present, index))

    def scope_fact(layers: Sequence[str], present: bool, index: int) -> None:
        layers = frozenset(layers)
        if present and not layers:
            raise EvidenceError('an empty scope cannot contain a query')
        for other, value, _ in scope_values:
            if present and not value and layers <= other or not present and value and other <= layers:
                raise EvidenceError('saved observations contradict reference-population membership')
        scope_values.append((layers, present, index))

    def membership(present: bool, facts: Mapping[str, bool], index: int) -> None:
        global_fact(present, index)
        for layer, value in facts.items():
            scope_fact([layer], value, index)

    def count(layers: Sequence[str], orientation: str, degree: int, indices: Sequence[int], values=None) -> None:
        key = (tuple(layers), _direction(view, layers, orientation))
        operations = {(item['layer_id'], item['direction']) for item in _native_inputs(view, layers, orientation)}
        for previous_key, previous_degree, _, previous_values in counts:
            previous_operations = {(item['layer_id'], item['direction']) for item in
                                   _native_inputs(view, previous_key[0], previous_key[1])}
            if (operations <= previous_operations and degree > previous_degree
                    or previous_operations <= operations and previous_degree > degree):
                raise EvidenceError('saved subset and union degree counts contradict each other')
            if key == previous_key and previous_degree != degree:
                raise EvidenceError('saved observations contradict an exact degree')
            if key == previous_key and values is not None and previous_values is not None and values != previous_values:
                raise EvidenceError('saved complete lists disagree about their distinct neighbor IDs')
        counts.append((key, degree, tuple(indices), values))

    def weight(layer: str, pair: Sequence[str], value: int | float) -> None:
        key = (layer, *pair)
        if key in degree_weights and degree_weights[key] != value:
            raise EvidenceError('saved observations disagree about a native edge weight')
        degree_weights[key] = value

    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != {'operation', 'arguments', 'status', 'result', 'provenance'}:
            raise EvidenceError('saved observations require exact attributed partial-result fields')
        provenance = observation['provenance']
        if not isinstance(provenance, Mapping) or set(provenance) != {'registry', 'graph'}:
            raise EvidenceError('saved observations require registry and graph provenance channels')
        for records in provenance.values():
            if not isinstance(records, list):
                raise EvidenceError('saved provenance must contain lists of acquisition records')
            for record in records:
                identity = record.get('evidence_id') if isinstance(record, Mapping) else None
                if not isinstance(identity, str) or not identity.strip():
                    raise EvidenceError('every saved acquisition requires a nonempty evidence ID')
                if identity in ids and _canonical(ids[identity]) != _canonical(observation):
                    raise EvidenceError('one evidence ID names conflicting acquisitions')
                ids[identity] = observation
        operation, source = observation['operation'], observation['arguments']
        if operation == 'lookup_human_gene_symbol':
            continue
        if operation not in ('get_degree', 'degree_distribution', 'get_degree_histogram', 'get_neighbors',
                             'get_neighbor_layers', 'get_gene_layers', 'get_nodes_by_layer', 'check_multiplex_membership'):
            continue
        records = provenance['graph']
        if not records or any(record.get('graph_version') != view.graph_version
                              or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        if (len(records) != 1 or provenance['registry']
                or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}):
            raise EvidenceError('a graph partial requires one original acquisition record')
        if source.get('multiplex_id') != arguments['multiplex_id']:
            raise EvidenceError('saved graph arguments conflict with their provenance multiplex')
        if operation in ('degree_distribution', 'get_degree_histogram'):
            layers, result = validate_histogram(view, observation)
            scope = runtime.multiplex._scope(source['layer_scope'], layers)
            if (source['degree_policy_id'] == DEGREE_POLICY and source['population_policy_id'] == POPULATION_POLICY
                    and layers == selected and _direction(view, layers, source['direction']) == direction):
                distributions.append((result, index))
                compatible.append(index)
        else:
            subjects = source.get('gene_ids', []) if operation in ('get_nodes_by_layer', 'check_multiplex_membership') else [source.get('gene_id')]
            if query not in subjects:
                continue
            if operation == 'get_degree':
                layers, result = _measurement(runtime, observation)
                scope = runtime.multiplex._scope(source['layer_scope'], layers)
                member = result['membership']
                global_fact(member['present_in_multiplex'], index)
                if member['present_in_scope'] is not None:
                    scope_fact(layers, member['present_in_scope'], index)
                if result['degree_policy_id'] == DEGREE_POLICY and result['degree'] is not None:
                    count(layers, source['direction'], result['degree'], [index])
                    if (layers == selected and _direction(view, layers, source['direction']) == direction
                            and result['rank_support'] is not None
                            and result['rank_support']['population_policy_id'] == POPULATION_POLICY):
                        rank = result['rank_support']
                        if ranks and _canonical(ranks[0][0]) != _canonical(rank):
                            raise EvidenceError('saved exact rank records disagree')
                        ranks.append((rank, [index]))
            elif operation == 'get_neighbors':
                present, facts, scope = _neighbor_membership(memberships, observation)
                membership(present, facts, index)
                if present:
                    _native_page(observation)
                    layer, orientation = source['layer_id'], source['direction']
                    for row in observation['result']['neighbors']:
                        pair = ([query, row['gene_id']] if orientation == 'out' else
                                [row['gene_id'], query] if orientation == 'in' else sorted((query, row['gene_id'])))
                        weight(layer, pair, row['weight'])
                    positives.append(([(layer, orientation)], {row['gene_id'] for row in observation['result']['neighbors']}))
                    total = observation['result']['collection']['total_count']
                    if total is not None:
                        count([layer], orientation, total, [index])
                    native_groups[(layer, orientation, source['mode'], source['top_k'])].append((index, observation))
            elif operation == 'get_neighbor_layers':
                present, facts, scope = _map_membership(memberships, observation)
                membership(present, facts, index)
                layers = _map_scope(runtime.multiplex, source)
                if present:
                    _validate_rows(runtime.multiplex, source, layers, observation['result'])
                    collection = observation['result']['collection']
                    if (not isinstance(collection, Mapping)
                            or set(collection) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
                            or collection['mode'] != source['mode'] or type(collection['returned_count']) is not int
                            or collection['returned_count'] != len(observation['result']['neighbors'])
                            or type(collection['complete']) is not bool
                            or collection['total_count'] is not None and (type(collection['total_count']) is not int
                                or collection['total_count'] < collection['returned_count'])
                            or observation['result']['unique_neighbor_count'] != collection['total_count']
                            or collection['total_count'] is not None
                                and collection['complete'] != (collection['returned_count'] == collection['total_count'])
                            or collection['complete'] and collection['next_cursor'] is not None):
                        raise EvidenceError('saved map page has invalid collection fields')
                    for row in observation['result']['neighbors']:
                        for support in row['supporting_layers']:
                            for edge in support['edges']:
                                weight(support['layer_id'], edge['gene_ids'], edge['weight'])
                    operations = [(item['layer_id'], item['direction']) for item in
                                  _native_inputs(view, layers, source['direction'])]
                    positives.append((operations, {row['gene_id'] for row in observation['result']['neighbors']}))
                    total = observation['result']['collection']['total_count']
                    if total is not None:
                        count(layers, source['direction'], total, [index])
                    map_groups[(layers, source['direction'], source['mode'], source['top_k'])].append((index, observation))
            elif operation == 'get_gene_layers':
                layers = memberships._validate(source)
                result = _membership_page(memberships, observation, layers)
                scope = memberships.multiplex._scope(source['layer_scope'], layers)
                facts = ({layer: layer in result['layer_ids'] for layer in layers} if result['collection']['complete']
                         else dict.fromkeys(result['layer_ids'], True))
                membership(result['present'], facts, index)
                member_groups[(layers, source['mode'], source['top_k'])].append((index, observation))
            elif operation == 'get_nodes_by_layer':
                partition, scope = _nodes_partition(nodes, observation)
                category = partition[query]
                membership(category != 'missing_gene_ids', {} if category == 'missing_gene_ids' else
                           {source['layer_id']: category == 'layer_gene_ids'}, index)
            else:
                present, facts, scope = _global_membership(arguments, observation)
                membership(present, facts, index)
            compatible.append(index)
        if records[0]['layer_scope'] != scope:
            raise EvidenceError('saved provenance scope differs from its original acquisition')

    for (layers, _, _), pages in member_groups.items():
        if any(page['arguments']['cursor'] is None for _, page in pages):
            summary = _membership_traversal(memberships, [page for _, page in pages], layers)
            if summary['full']:
                for index, _ in pages:
                    membership(summary['present'], {layer: layer in summary['layers'] for layer in layers}, index)
    native_sets = {}
    for (layer, orientation, _, _), pages in native_groups.items():
        if not any(page['arguments']['cursor'] is None for _, page in pages):
            continue
        rows, total, full, cursor = _native_traversal([page for _, page in pages])
        indices = [index for index, _ in pages]
        if total is not None:
            count([layer], orientation, total, indices)
        if full and cursor is None:
            values = frozenset(row['gene_id'] for row in rows)
            count([layer], orientation, len(values), indices, values)
            key = (layer, orientation)
            if key in native_sets and native_sets[key][0] != values:
                raise EvidenceError('saved native selections disagree about the complete neighbor set')
            native_sets[key] = (values, indices)
    for (layers, orientation, _, _), pages in map_groups.items():
        if not any(page['arguments']['cursor'] is None for _, page in pages):
            continue
        rows, _, total, full, cursor = _map_traversal(runtime.multiplex, [page for _, page in pages], layers)
        indices = [index for index, _ in pages]
        if total is not None:
            count(layers, orientation, total, indices)
        if full and cursor is None:
            values = frozenset(row['gene_id'] for row in rows)
            count(layers, orientation, len(values), indices, values)

    present = global_values[0][0] if global_values else None
    positive = [(layers, value, index) for layers, value, index in scope_values if value and layers <= requested]
    negative = [(layers, value, index) for layers, value, index in scope_values if not value and requested <= layers]
    absent_layers = {layer for layers, value, _ in scope_values if not value for layer in layers}
    in_scope = True if positive else False if negative or not selected or requested <= absent_layers else None
    if present is False and any(value for _, value, _ in scope_values):
        raise EvidenceError('global absence conflicts with positive scoped membership')

    needed = _native_inputs(view, selected, arguments['direction'])
    values, indices, complete = set(), [], True
    for item in needed:
        key = (item['layer_id'], item['direction'])
        if key in native_sets:
            neighbors, providers = native_sets[key]
            values.update(neighbors)
            indices.extend(providers)
        elif item['layer_id'] in absent_layers:
            indices.extend(index for layers, value, index in scope_values if not value and item['layer_id'] in layers)
        else:
            complete = False
    if present is True and complete:
        count(selected, arguments['direction'], len(values), indices, frozenset(values))
    if present is True and in_scope is False:
        count(selected, arguments['direction'], 0, [index for _, index in global_values] + [index for _, _, index in scope_values])
    for (layers, orientation), degree_count, _, complete_values in counts:
        operations = {(item['layer_id'], item['direction']) for item in _native_inputs(view, layers, orientation)}
        visible = set().union(*(values for source_operations, values in positives if set(source_operations) <= operations))
        if degree_count < len(visible) or complete_values is not None and not visible <= complete_values:
            raise EvidenceError('an exact degree or complete set contradicts visible positive neighbors')
    matches = [entry for entry in counts if entry[0] == (tuple(selected), direction)]
    degree = matches[0][1] if matches else None

    for distribution, index in distributions:
        if not distribution['complete']:
            continue
        histogram = distribution['degree_histogram']
        if degree is not None and in_scope is True:
            equal = next((row['node_count'] for row in histogram if row['degree'] == degree), 0)
            if not equal:
                raise EvidenceError('known population member has no matching degree in a complete distribution')
            rank = {'population_policy_id': POPULATION_POLICY,
                    'lower_degree_count': sum(row['node_count'] for row in histogram if row['degree'] < degree),
                    'equal_degree_count': equal, 'population_count': distribution['population_count']}
            _rank_support(rank)
            if ranks and _canonical(ranks[0][0]) != _canonical(rank):
                raise EvidenceError('saved exact rank conflicts with the complete distribution')
            ranks.append((rank, [index]))
    if distributions:
        complete_distributions = [result for result, _ in distributions if result['complete']]
        if any((result['population_count'], result['degree_histogram']) !=
               (complete_distributions[0]['population_count'], complete_distributions[0]['degree_histogram'])
               for result in complete_distributions[1:]):
            raise EvidenceError('saved complete distributions disagree')

    used = []
    for index in sorted(set(compatible)):
        observation = observations[index]
        if observation not in used:
            used.append(observation)
    provenance = {'registry': [], 'graph': []}
    for observation in used:
        for channel in provenance:
            for record in observation['provenance'][channel]:
                if record not in provenance[channel]:
                    provenance[channel].append(deepcopy(record))

    issue = ('NO_MULTIPLEX_MEMBERSHIP' if present is False else
             'EMPTY_REFERENCE_POPULATION' if arguments['request_kind'] == 'degree_and_hub'
                and present is True and (not selected or any(result['complete'] and result['population_count'] == 0 for result, _ in distributions)) else
             'OUTSIDE_REFERENCE_POPULATION' if arguments['request_kind'] == 'degree_and_hub'
                and present is True and in_scope is False else None)
    if issue:
        return DegreeProof(None, None, None, tuple(used), provenance, issue)
    enough = present is True and degree is not None and (arguments['request_kind'] == 'degree_only' or bool(ranks))
    if enough:
        rank = deepcopy(ranks[0][0]) if arguments['request_kind'] == 'degree_and_hub' else None
        measurement = {'gene_id': query, 'multiplex_id': arguments['multiplex_id'],
            'layer_scope': runtime.multiplex._scope(arguments['layer_scope'], selected),
            'resolved_layer_ids': list(selected), 'direction': arguments['direction'],
            'request_kind': arguments['request_kind'], 'degree_policy_id': DEGREE_POLICY,
            'membership': {'present_in_multiplex': True, 'present_in_scope': in_scope},
            'degree': degree, 'degree_percentile': percentile(rank) if rank else None,
            'rank_support': rank, 'hub_policy': deepcopy(arguments['hub_policy'])}
        return DegreeProof(final_result(measurement), None, None, tuple(used), provenance)
    return DegreeProof(None, deepcopy(dict(arguments)), 'get_degree', tuple(used), provenance)
