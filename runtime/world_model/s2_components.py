"""Calculate exact unweighted components from checked native graph evidence.

The caller proves that the visible edges suffice for the requested partition.
This module never reads source inventories or changes the native edge records.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from .s2_multiplex import SelectionError
from .s2_nodes_by_layer import NodesByLayerRuntime
from .s2_sources import GraphView


CONNECTIVITY_KINDS = ('undirected', 'weak', 'strong')
ARGUMENT_KEYS = ('multiplex_id', 'layer_id', 'gene_ids', 'connectivity_kind')


def validate_connectivity(view: GraphView, layer_id: Any, connectivity_kind: Any) -> str:
    """Resolve an undirected default and reject unsupported native interpretations."""
    if not layer_id:
        raise SelectionError('invalid_input', 'MISSING_SCOPE')
    if not isinstance(layer_id, str) or layer_id not in view.layers:
        raise SelectionError('not_found', 'UNKNOWN_LAYER')
    directed = view.layers[layer_id].directed
    if connectivity_kind is None:
        if directed:
            raise SelectionError('invalid_input', 'CONNECTIVITY_KIND_REQUIRED')
        return 'undirected'
    if not isinstance(connectivity_kind, str) or connectivity_kind not in CONNECTIVITY_KINDS:
        raise SelectionError('invalid_input', 'INVALID_CONNECTIVITY_KIND')
    if directed == (connectivity_kind == 'undirected'):
        raise SelectionError('invalid_input', 'CONNECTIVITY_KIND_UNAVAILABLE')
    return connectivity_kind


def components(gene_ids: Sequence[str], edges: Iterable[Mapping[str, Any] | Sequence[str]], kind: str) -> list[list[str]]:
    """Partition selected members without weights, outside paths, or recursion limits."""
    if kind not in CONNECTIVITY_KINDS:
        raise ValueError('components requires an explicit supported connectivity kind')
    nodes = sorted(set(gene_ids))
    selected = set(nodes)
    pairs = (tuple(edge['gene_ids'] if isinstance(edge, Mapping) else edge) for edge in edges)
    if kind != 'strong':
        parents = {gene: gene for gene in nodes}
        sizes = dict.fromkeys(nodes, 1)

        def root(gene):
            while parents[gene] != gene:
                parents[gene] = parents[parents[gene]]
                gene = parents[gene]
            return gene

        for source, target in pairs:
            if source not in selected or target not in selected:
                continue
            left, right = root(source), root(target)
            if left != right:
                if sizes[left] < sizes[right]:
                    left, right = right, left
                parents[right] = left
                sizes[left] += sizes[right]
        groups = defaultdict(list)
        for gene in nodes:
            groups[root(gene)].append(gene)
        return sorted(groups.values(), key=lambda group: group[0])

    outgoing, incoming = {gene: [] for gene in nodes}, {gene: [] for gene in nodes}
    for source, target in pairs:
        if source in selected and target in selected:
            outgoing[source].append(target)
            incoming[target].append(source)
    visited, finish = set(), []
    for gene in nodes:
        if gene in visited:
            continue
        visited.add(gene)
        stack = [(gene, iter(outgoing[gene]))]
        while stack:
            node, neighbors = stack[-1]
            target = next(neighbors, None)
            if target is None:
                finish.append(node)
                stack.pop()
            elif target not in visited:
                visited.add(target)
                stack.append((target, iter(outgoing[target])))
    visited, groups = set(), []
    for gene in reversed(finish):
        if gene in visited:
            continue
        group, stack = [], [gene]
        visited.add(gene)
        while stack:
            node = stack.pop()
            group.append(node)
            for source in incoming[node]:
                if source not in visited:
                    visited.add(source)
                    stack.append(source)
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: group[0])


def component_result(view: GraphView, arguments: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ordinary result after the caller proves full partition sufficiency."""
    source = {key: arguments.get(key) for key in ('multiplex_id', 'layer_id', 'gene_ids')}
    requested = list(NodesByLayerRuntime(view)._validate(source))
    kind = validate_connectivity(view, arguments['layer_id'], arguments.get('connectivity_kind'))
    if set(arguments) != set(ARGUMENT_KEYS):
        raise SelectionError('invalid_input', 'INVALID_SELECTION')
    keys = ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids')
    partition = {key: sorted(set(graph[key])) for key in keys}
    classified = [gene for key in keys for gene in partition[key]]
    coverage = {'operation': 'layer_membership', 'requested_gene_ids': requested,
                'checked_gene_ids': requested, 'unchecked_gene_ids': []}
    if len(classified) != len(set(classified)) or set(classified) != set(requested):
        raise ValueError('component results require a complete disjoint membership partition')
    if graph.get('coverage') != coverage:
        raise ValueError('component results require complete ordered membership coverage')
    for key, value in (('multiplex_id', arguments['multiplex_id']), ('layer_id', arguments['layer_id']),
                       ('directed', view.layers[arguments['layer_id']].directed)):
        if key in graph and graph[key] != value:
            raise ValueError('component results require matching native graph metadata')
    groups = components(partition['layer_gene_ids'], graph.get('edges', ()), kind)
    count = len(groups)
    return {'multiplex_id': arguments['multiplex_id'], 'layer_id': arguments['layer_id'],
        'directed': view.layers[arguments['layer_id']].directed, 'connectivity_kind': kind,
        'layer_gene_ids': partition['layer_gene_ids'], 'layer_gene_count': len(partition['layer_gene_ids']),
        'not_in_layer_gene_ids': partition['not_in_layer_gene_ids'], 'missing_gene_ids': partition['missing_gene_ids'],
        'components': groups, 'component_count': count,
        'single_component': count == 1 and not partition['not_in_layer_gene_ids'] and not partition['missing_gene_ids'],
        'coverage': deepcopy(coverage), 'collection': {'mode': 'complete', 'returned_count': count,
            'total_count': count, 'complete': True, 'next_cursor': None}}
