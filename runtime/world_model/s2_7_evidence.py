"""Prove exact shared sets from visible neighbor and membership observations.

The planner reads manifest metadata only. It never reads source inventories or
adjacency, and it acquires each required native list through get_neighbors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from .s2_5_evidence import _global_membership, _map_membership, _neighbor_membership
from .s2_5_evidence import _traversal as _membership_traversal
from .s2_6_evidence import _membership_page, _nodes_partition, collect as _validate_memberships
from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import EvidenceError, SelectionError
from .s2_neighbors import ARGUMENT_KEYS, _selection, _traversal
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView
from .sources import GENE_ID_PATTERN


@dataclass(frozen=True)
class SharedProof:
    """Return an exact intersection, a prerequisite blocker, or the next call."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    coverage: dict
    provenance: dict[str, list]
    issue_code: str | None = None


def validate_arguments(view: GraphView, arguments: Mapping[str, Any]) -> None:
    """Validate two ordered roles and the registered native representation."""
    if not arguments.get('multiplex_id') or not arguments.get('layer_id'):
        raise SelectionError('invalid_input', 'MISSING_SCOPE')
    if arguments['multiplex_id'] != view.multiplex_id:
        raise SelectionError('not_found', 'UNKNOWN_MULTIPLEX')
    layer = arguments['layer_id']
    if not isinstance(layer, str) or layer not in view.layers:
        raise SelectionError('not_found', 'UNKNOWN_LAYER')
    queries = arguments.get('queries')
    if queries is None:
        raise SelectionError('invalid_input', 'MISSING_GENES')
    if (set(arguments) != {'multiplex_id', 'layer_id', 'queries'} or not isinstance(queries, list)
            or len(queries) != 2 or any(not isinstance(query, Mapping) or set(query) != {'gene_id', 'direction'}
            or not isinstance(query['gene_id'], str) or not GENE_ID_PATTERN.fullmatch(query['gene_id']) for query in queries)):
        raise SelectionError('invalid_input', 'INVALID_QUERY_PAIR')
    allowed = ('in', 'out', 'both') if view.layers[layer].directed else ('undirected',)
    if any(query['direction'] not in allowed for query in queries):
        raise SelectionError('invalid_input', 'DIRECTION_UNAVAILABLE')


def atomic_queries(arguments: Mapping[str, Any]) -> list[dict]:
    """Expand both roles in input order and remove identical primitive queries."""
    queries = []
    for role in arguments['queries']:
        for direction in ('in', 'out') if role['direction'] == 'both' else (role['direction'],):
            query = {'gene_id': role['gene_id'], 'direction': direction}
            if query not in queries:
                queries.append(query)
    return queries


def _coverage(queries: Sequence[Mapping], checked: Sequence[tuple]) -> dict:
    """Mark only complete native sets as checked."""
    return {'operation': 'neighbor_lists', 'requested_queries': deepcopy(list(queries)),
            'checked_queries': [dict(query) for query in queries if (query['gene_id'], query['direction']) in checked],
            'unchecked_queries': [dict(query) for query in queries if (query['gene_id'], query['direction']) not in checked]}


def collect(view: GraphView, arguments: Mapping[str, Any],
            observations: Sequence[Mapping[str, Any]]) -> SharedProof:
    """Prove the requested intersection or request the first required primitive.

    Complete-set containment can settle the result before another list ends.
    Both input genes still need explicit positive multiplex membership.
    """
    validate_arguments(view, arguments)
    queries, layer = atomic_queries(arguments), arguments['layer_id']
    genes = tuple(dict.fromkeys(query['gene_id'] for query in queries))
    memberships, nodes = GeneLayersRuntime(view), NodesByLayerRuntime(view)
    try:
        _validate_memberships(view, {'multiplex_id': arguments['multiplex_id'], 'layer_id': layer,
                                    'gene_ids': list(genes)}, observations)
    except SelectionError as error:
        raise EvidenceError('saved membership has invalid acquisition arguments') from error
    global_values, layer_values = {}, {}
    global_providers, layer_providers = defaultdict(list), defaultdict(list)
    compatible, neighbor_groups, membership_groups = [], defaultdict(list), defaultdict(list)

    def fact(gene: str, present: bool, facts: Mapping, indices: Sequence[int]) -> None:
        global_values[gene] = present
        global_providers[gene].append(tuple(indices))
        if layer in facts:
            layer_values[gene] = facts[layer]
            layer_providers[gene].append(tuple(indices))

    for index, observation in enumerate(observations):
        operation, source = observation.get('operation'), observation.get('arguments', {})
        if operation not in ('get_neighbors', 'get_gene_layers', 'get_nodes_by_layer',
                             'get_neighbor_layers', 'check_multiplex_membership'):
            continue
        if not isinstance(source, Mapping) or source.get('multiplex_id') != arguments['multiplex_id']:
            continue
        supplied = source.get('gene_ids', []) if operation in ('get_nodes_by_layer', 'check_multiplex_membership') else [source.get('gene_id')]
        if not isinstance(supplied, list) or not any(gene in genes for gene in supplied):
            continue
        records = observation.get('provenance', {}).get('graph', [])
        if not records or any(record.get('graph_version') != view.graph_version or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        compatible.append(index)
        if operation == 'get_nodes_by_layer':
            partition, _ = _nodes_partition(nodes, observation)
            for gene in genes:
                if gene in partition:
                    category = partition[gene]
                    fact(gene, category != 'missing_gene_ids',
                         {} if category == 'missing_gene_ids' else {source['layer_id']: category == 'layer_gene_ids'}, [index])
        elif operation == 'check_multiplex_membership':
            for gene in genes:
                if gene in supplied:
                    present, facts, _ = _global_membership({'multiplex_id': arguments['multiplex_id'], 'gene_id': gene}, observation)
                    fact(gene, present, facts, [index])
        elif operation == 'get_gene_layers':
            selected = memberships._validate(source)
            page = _membership_page(memberships, observation, selected)
            facts = {item: item in page['layer_ids'] for item in selected} if page['collection']['complete'] else dict.fromkeys(page['layer_ids'], True)
            fact(source['gene_id'], page['present'], facts, [index])
            key = (source['gene_id'], selected, source['mode'], source['top_k'],
                   json.dumps(source['layer_scope'], sort_keys=True, separators=(',', ':')))
            membership_groups[key].append((index, observation))
        else:
            present, facts, _ = (_neighbor_membership(memberships, observation) if operation == 'get_neighbors'
                                 else _map_membership(memberships, observation))
            fact(source['gene_id'], present, facts, [index])
            if operation == 'get_neighbors' and source['layer_id'] == layer:
                cursor = source['cursor']
                if cursor is not None and (not isinstance(cursor, str) or not cursor):
                    raise EvidenceError('saved neighbor cursor must be nonempty or null')
                _selection(source['mode'], source['top_k'])
                if present:
                    result = observation['result']
                    if (set(result['collection']) != {'mode', 'returned_count', 'total_count', 'complete', 'next_cursor'}
                            or any(not isinstance(row, Mapping) or set(row) != {'gene_id', 'weight'} for row in result['neighbors'])):
                        raise EvidenceError('saved native rows or collection have unexpected fields')
                    neighbor_groups[(source['gene_id'], source['direction'], source['mode'], source['top_k'])].append((index, observation))

    for (gene, selected, _, _, _), pages in membership_groups.items():
        if not any(page['arguments']['cursor'] is None for _, page in pages):
            continue
        summary = _membership_traversal(memberships, [page for _, page in pages], selected)
        facts = {item: item in summary['layers'] for item in selected} if summary['full'] else dict.fromkeys(summary['layers'], True)
        fact(gene, summary['present'], facts, [index for index, _ in pages])

    summaries, positives, full_sets = defaultdict(list), {}, {}
    for (gene, direction, mode, top_k), pages in neighbor_groups.items():
        rows, total, full, cursor = _traversal([page for _, page in pages])
        key, indices = (gene, direction), tuple(index for index, _ in pages)
        summary = {'rows': rows, 'total': total, 'full': full, 'cursor': cursor,
                   'mode': mode, 'top_k': top_k, 'indices': indices}
        for prior in summaries[key]:
            common = min(len(prior['rows']), len(rows))
            if prior['rows'][:common] != rows[:common]:
                raise EvidenceError('saved selections disagree about the ranked neighbor prefix')
            if total is not None and prior['total'] is not None and total != prior['total']:
                raise EvidenceError('inconsistent non-null neighbor totals')
            if prior['full'] and len(rows) > len(prior['rows']) or full and len(prior['rows']) > len(rows):
                raise EvidenceError('saved prefix exceeds a complete neighbor set')
        summaries[key].append(summary)
    for query in queries:
        key, gene = (query['gene_id'], query['direction']), query['gene_id']
        candidates = summaries[key]
        complete = [summary for summary in candidates if summary['full']]
        if complete:
            chosen = min(complete, key=lambda summary: (len(summary['indices']), summary['indices']))
            full_sets[key] = ({row['gene_id'] for row in chosen['rows']}, chosen['indices'])
        elif global_values.get(gene) is True and layer_values.get(gene) is False:
            full_sets[key] = (set(), min(layer_providers[gene], key=lambda indices: (len(indices), indices)))
        if candidates:
            chosen = max(candidates, key=lambda summary: (len(summary['rows']), summary['full'], -min(summary['indices'])))
            # A full list replaces a redundant bounded prefix in the final proof.
            if complete:
                chosen = min(complete, key=lambda summary: (len(summary['indices']), summary['indices']))
            positives[key] = ({row['gene_id'] for row in chosen['rows']}, chosen['indices'])
        elif key in full_sets:
            positives[key] = full_sets[key]
        else:
            positives[key] = (set(), ())
    coverage = _coverage(queries, full_sets)

    def finish(result: dict | None = None, pending: dict | None = None,
               indices: Sequence[int] | None = None, issue: str | None = None) -> SharedProof:
        used, provenance = [], {'registry': [], 'graph': []}
        for index in sorted(set(compatible if indices is None else indices)):
            observation = observations[index]
            if observation not in used:
                used.append(observation)
            for channel in provenance:
                for record in observation['provenance'][channel]:
                    if record not in provenance[channel]:
                        provenance[channel].append(deepcopy(record))
        return SharedProof(result, pending, 'get_neighbors' if pending is not None else None,
                           tuple(used), coverage, provenance, issue)

    if any(global_values.get(gene) is False for gene in genes):
        return finish(issue='NO_MULTIPLEX_MEMBERSHIP')

    roles = []
    for role in arguments['queries']:
        keys = [(role['gene_id'], direction) for direction in
                (('in', 'out') if role['direction'] == 'both' else (role['direction'],))]
        observed = set().union(*(positives[key][0] for key in keys))
        full = all(key in full_sets for key in keys)
        indices = {index for key in keys for index in (full_sets[key][1] if full else positives[key][1])}
        roles.append((observed, full, indices))
    shared, support = None, set()
    if roles[0][1] and roles[1][1]:
        shared = roles[0][0] & roles[1][0]
        support = roles[0][2] | roles[1][2]
    else:
        for first, second in ((0, 1), (1, 0)):
            if roles[first][1] and roles[first][0] <= roles[second][0]:
                shared = roles[first][0]
                support = roles[first][2] | (roles[second][2] if shared else set())
                break
    if shared is not None and all(global_values.get(gene) is True for gene in genes):
        # Keep the evidence for each reported full primitive and membership fact.
        support.update(index for _, indices in full_sets.values() for index in indices)
        for gene in genes:
            for providers in (global_providers[gene], layer_providers[gene] if gene in layer_values else []):
                if providers and not any(set(indices) <= support for indices in providers):
                    support.update(min(providers, key=lambda indices: (len(set(indices) - support), len(indices), indices)))
        shared_ids = sorted(shared)
        result = {'multiplex_id': arguments['multiplex_id'], 'layer_id': layer,
                  'queries': deepcopy(arguments['queries']), 'shared_neighbors': shared_ids,
                  'shared_neighbor_count': len(shared_ids),
                  'input_memberships': [{'gene_id': gene, 'present_in_multiplex': True,
                                         'present_in_layer': layer_values.get(gene)} for gene in genes],
                  'coverage': coverage,
                  'collection': {'mode': 'complete', 'returned_count': len(shared_ids), 'total_count': len(shared_ids),
                                 'complete': True, 'next_cursor': None}}
        return finish(result, indices=support)

    pending_queries = [query for query in queries if (query['gene_id'], query['direction']) not in full_sets]
    if shared is not None:
        pending_queries = [query for query in pending_queries if global_values.get(query['gene_id']) is not True]
    if not pending_queries:
        raise EvidenceError('complete native sets lack their global membership prerequisites')
    query = pending_queries[0]
    pending = dict(zip(ARGUMENT_KEYS, (arguments['multiplex_id'], layer, query['gene_id'], query['direction'], 'complete', None, None)))
    matching = [summary for summary in summaries[(query['gene_id'], query['direction'])] if summary['mode'] == 'complete']
    if matching:
        pending['cursor'] = matching[0]['cursor']
    return finish(pending=pending)
