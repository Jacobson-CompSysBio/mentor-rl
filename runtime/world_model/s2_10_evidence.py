"""Prove exact components from visible native graph and membership records.

The planner reads metadata only. Positive paths and settled negative domains
can prove a partition without retrieval of every native edge or self-loop.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations, product
import math
from typing import Any

from .s2_5_evidence import _global_membership, _map_membership, _neighbor_membership
from .s2_5_evidence import _traversal as _membership_traversal
from .s2_6_evidence import _coverage, _membership_page
from .s2_9_evidence import _canonical, _chain, _edge_key, _scope
from .s2_9_evidence import collect as _subgraph_collect
from .s2_components import component_result, components, validate_connectivity
from .s2_gene_layers import GeneLayersRuntime
from .s2_multiplex import EvidenceError
from .s2_neighbors import _rank
from .s2_sources import GraphView
from .s2_subgraphs import SubgraphRuntime
from .sources import GENE_ID_PATTERN


@dataclass(frozen=True)
class ComponentProof:
    """Return the exact partition, an explicit gate issue, or the next call."""

    result: dict | None
    next_arguments: dict | None
    next_tool: str | None
    observations: tuple[Mapping[str, Any], ...]
    provenance: dict[str, list]
    coverage: dict
    issue_code: str | None = None


def _edge_observation(view: GraphView, observation: Mapping) -> dict[str, str]:
    """Validate a saved S2.1 edge and its complete endpoint classifications."""
    source, result = observation['arguments'], observation['result']
    if (not isinstance(source, Mapping) or set(source) != {'multiplex_id', 'layer_id', 'gene_ids'}
            or source['multiplex_id'] != view.multiplex_id or not isinstance(source['layer_id'], str)
            or source['layer_id'] not in view.layers
            or not isinstance(source['gene_ids'], list) or len(source['gene_ids']) != 2
            or any(not isinstance(gene, str) or not GENE_ID_PATTERN.fullmatch(gene) for gene in source['gene_ids'])
            or len(set(source['gene_ids'])) != 2):
        raise EvidenceError('saved edge acquisition has invalid arguments')
    directed = view.layers[source['layer_id']].directed
    pair = _edge_key(source['gene_ids'], directed)
    fields = {'gene_ids', 'multiplex_id', 'layer_id', 'directed', 'edge_exists', 'weight',
              'missing_gene_ids', 'not_in_layer_gene_ids'}
    if (not isinstance(result, Mapping) or set(result) != fields
            or any(result[key] != source[key] for key in ('multiplex_id', 'layer_id'))
            or result['gene_ids'] != list(pair) or type(result['directed']) is not bool
            or result['directed'] != directed):
        raise EvidenceError('saved edge identity or native representation is inconsistent')
    categories = {}
    for category in ('missing_gene_ids', 'not_in_layer_gene_ids'):
        rows = result[category]
        if (not isinstance(rows, list) or any(gene not in pair for gene in rows)
                or rows != sorted(set(rows)) or any(gene in categories for gene in rows)):
            raise EvidenceError('saved edge exclusions must be sorted unique endpoint lists')
        categories.update(dict.fromkeys(rows, category))
    categories.update((gene, 'layer_gene_ids') for gene in pair if gene not in categories)
    exists, weight, missing = result['edge_exists'], result['weight'], result['missing_gene_ids']
    if (observation['status'] != ('not_found' if missing else 'resolved')
            or missing and exists is not None or not missing and type(exists) is not bool
            or exists is True and (result['not_in_layer_gene_ids'] or type(weight) not in (int, float) or not math.isfinite(weight))
            or exists is not True and weight is not None
            or result['not_in_layer_gene_ids'] and exists is True):
        raise EvidenceError('saved edge value contradicts membership, status, or weight')
    records = observation['provenance']['graph']
    if (len(records) != 1 or observation['provenance']['registry']
            or set(records[0]) != {'multiplex_id', 'graph_version', 'layer_scope', 'evidence_id'}
            or records[0]['layer_scope'] != _scope(source['layer_id'])):
        raise EvidenceError('saved edge must retain its original native acquisition scope')
    return categories


def _facts(view: GraphView, layer: str, observations: Sequence[Mapping]) -> tuple:
    """Extract validated facts and compact complete domains without source reads."""
    memberships = GeneLayersRuntime(view)
    global_values, layer_values, categories, positives, negatives = {}, {}, {}, {}, set()
    graph_groups, native_groups, member_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    edge_positions, full_domains, full_native = [], [], {}

    def membership(gene, present, facts):
        if gene in global_values and global_values[gene] != present:
            raise EvidenceError('saved observations contradict global membership')
        global_values[gene] = present
        for selected, value in facts.items():
            key = gene, selected
            if key in layer_values and layer_values[key] != value:
                raise EvidenceError('saved observations contradict layer membership')
            layer_values[key] = value
        if not present:
            categories[gene] = 'missing_gene_ids'
        elif layer in facts:
            categories[gene] = 'layer_gene_ids' if facts[layer] else 'not_in_layer_gene_ids'

    def positive(selected, pair, weight):
        pair = _edge_key(pair, view.layers[selected].directed)
        key = selected, pair
        if key in positives and positives[key] != weight:
            raise EvidenceError('saved observations disagree about one native edge weight')
        positives[key] = weight
        for gene in pair:
            membership(gene, True, {selected: True})

    for position, observation in enumerate(observations):
        records = observation['provenance']['graph']
        if not records or any(record.get('graph_version') != view.graph_version
                              or record.get('multiplex_id') != view.multiplex_id for record in records):
            continue
        operation, source, result = observation['operation'], observation['arguments'], observation['result']
        if operation in ('induce_subgraph', 'get_nodes_by_layer', 'get_edge'):
            partition = (_edge_observation(view, observation) if operation == 'get_edge' else
                         {gene: category for category in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids')
                          for gene in result[category]})
            for gene, category in partition.items():
                membership(gene, category != 'missing_gene_ids',
                           {} if category == 'missing_gene_ids' else {source['layer_id']: category == 'layer_gene_ids'})
            if operation == 'induce_subgraph':
                key = source['layer_id'], tuple(dict.fromkeys(source['gene_ids'])), source['mode'], source['top_k']
                graph_groups[key].append(observation)
                for edge in result['edges']:
                    positive(source['layer_id'], edge['gene_ids'], edge['weight'])
            elif operation == 'get_edge':
                edge_positions.append(position)
                if result['edge_exists']:
                    positive(source['layer_id'], result['gene_ids'], result['weight'])
                else:
                    negatives.add((source['layer_id'], _edge_key(result['gene_ids'], result['directed'])))
        elif operation == 'check_multiplex_membership':
            for gene in source['gene_ids']:
                present, facts, _ = _global_membership({'multiplex_id': view.multiplex_id, 'gene_id': gene}, observation)
                membership(gene, present, facts)
        elif operation == 'get_gene_layers':
            selected = memberships._validate(source)
            page = _membership_page(memberships, observation, selected)
            facts = {item: item in page['layer_ids'] for item in selected} if page['collection']['complete'] else dict.fromkeys(page['layer_ids'], True)
            membership(source['gene_id'], page['present'], facts)
            key = source['gene_id'], tuple(selected), source['mode'], source['top_k'], _canonical(source['layer_scope'])
            member_groups[key].append(observation)
        elif operation in ('get_neighbors', 'get_neighbor_layers'):
            present, facts, _ = (_neighbor_membership(memberships, observation) if operation == 'get_neighbors'
                                 else _map_membership(memberships, observation))
            membership(source['gene_id'], present, facts)
            if operation == 'get_neighbors' and present:
                key = source['layer_id'], source['gene_id'], source['direction'], source['mode'], source['top_k']
                native_groups[key].append(observation)
                for row in result['neighbors']:
                    pair = (row['gene_id'], source['gene_id']) if source['direction'] == 'in' else (source['gene_id'], row['gene_id'])
                    positive(source['layer_id'], pair, row['weight'])

    for (gene, selected, _, _, _), pages in member_groups.items():
        if any(page['arguments']['cursor'] is None for page in pages):
            summary = _membership_traversal(memberships, pages, selected)
            facts = {item: item in summary['layers'] for item in selected} if summary['full'] else dict.fromkeys(summary['layers'], True)
            membership(gene, summary['present'], facts)
    for (selected, _, _, _), pages in graph_groups.items():
        summary = _chain(pages, 'edges', lambda row: tuple(row['gene_ids']))
        if summary['full']:
            full_domains.append((selected, frozenset(pages[0]['result']['layer_gene_ids']),
                                 {tuple(row['gene_ids']): row['weight'] for row in summary['rows']}))
    for (selected, gene, direction, _, _), pages in native_groups.items():
        summary = _chain(pages, 'neighbors', _rank)
        if summary['full']:
            full_native[selected, gene, direction] = {row['gene_id']: row['weight'] for row in summary['rows']}
    for (gene, _), present in layer_values.items():
        if present and global_values.get(gene) is False:
            raise EvidenceError('global absence contradicts positive layer membership')
    for (selected, pair), weight in positives.items():
        if (selected, pair) in negatives:
            raise EvidenceError('a saved native edge is both positive and negative')
        for domain_layer, domain, edges in full_domains:
            if selected == domain_layer and set(pair) <= domain and edges.get(pair) != weight:
                raise EvidenceError('a positive edge conflicts with a complete induced projection')
        directed = view.layers[selected].directed
        for gene, direction, neighbor in ((pair[0], 'out', pair[1]), (pair[1], 'in', pair[0])) if directed else (
                (pair[0], 'undirected', pair[1]), (pair[1], 'undirected', pair[0])):
            values = full_native.get((selected, gene, direction))
            if values is not None and values.get(neighbor) != weight:
                raise EvidenceError('a positive edge conflicts with a complete native list')
    return categories, positives, negatives, full_domains, full_native, edge_positions


def collect(view: GraphView, arguments: Mapping[str, Any], observations: Sequence[Mapping[str, Any]],
            *, require_all_in_layer: bool = False, membership_first: bool = False) -> ComponentProof:
    """Prove a full partition or request the unchanged S2.9 graph acquisition.

    The minimum graph contains all visible positive edges. The maximum graph
    also contains every unsettled candidate. Equal partitions prove the answer.
    """
    native = {key: arguments.get(key) for key in ('multiplex_id', 'layer_id', 'gene_ids')}
    native.update(mode='complete', top_k=None, cursor=None)
    genes = SubgraphRuntime(view).validate_arguments(native)
    kind = validate_connectivity(view, native['layer_id'], arguments.get('connectivity_kind'))
    selected = {**dict(arguments), 'connectivity_kind': kind}
    base = _subgraph_collect(view, native, observations, require_all_in_layer=False)
    if type(require_all_in_layer) is not bool:
        raise ValueError('the all-members gate must be an explicit boolean')
    if type(membership_first) is not bool:
        raise ValueError('the acquisition policy must be an explicit boolean')
    if not genes:
        result = component_result(view, selected, base.result)
        return ComponentProof(result, None, None, (), base.provenance, base.coverage)
    layer, requested = native['layer_id'], frozenset(genes)
    categories, positives, negatives, full_domains, full_native, edge_positions = _facts(view, layer, observations)
    used = []
    for position, observation in enumerate(observations):
        relevant_edge = position in edge_positions and bool(requested.intersection(observation['arguments']['gene_ids']))
        source, result = observation['arguments'], observation['result']
        other_layer = source.get('layer_id') is not None and source['layer_id'] != layer
        global_absence = bool(requested.intersection(result.get('missing_gene_ids', [])))
        compatible = not other_layer or global_absence
        if compatible and (observation in base.observations or relevant_edge) and observation not in used:
            used.append(observation)
    provenance = {'registry': [], 'graph': []}
    for observation in used:
        for channel in provenance:
            for record in observation['provenance'][channel]:
                if record not in provenance[channel]:
                    provenance[channel].append(deepcopy(record))
    coverage = _coverage(genes, categories)

    def proof(result=None, pending=None, tool=None, issue=None):
        return ComponentProof(result, pending, tool, tuple(used), provenance, coverage, issue)

    def known(pair):
        return ((layer, pair) in positives or (layer, pair) in negatives
                or any(selected_layer == layer and set(pair) <= domain for selected_layer, domain, _ in full_domains)
                or (layer, pair[0], 'out' if view.layers[layer].directed else 'undirected') in full_native
                or (layer, pair[1], 'in' if view.layers[layer].directed else 'undirected') in full_native)

    full_membership = not coverage['unchecked_gene_ids']
    if membership_first and not full_membership and not require_all_in_layer:
        return proof(pending={'multiplex_id': native['multiplex_id'], 'layer_id': layer,
                              'gene_ids': list(genes)}, tool='get_nodes_by_layer')
    if require_all_in_layer:
        if not full_membership:
            return proof(pending={'multiplex_id': native['multiplex_id'], 'layer_id': layer,
                                  'gene_ids': list(genes)}, tool='get_nodes_by_layer')
        if any(categories[gene] != 'layer_gene_ids' for gene in genes):
            return proof(issue='MEMBERSHIP_PRECONDITION_UNSATISFIED')
    if not full_membership and len(coverage['unchecked_gene_ids']) == 1:
        candidates = product(genes, repeat=2) if view.layers[layer].directed else combinations(sorted(genes), 2)
        if all(pair[0] == pair[1] or known(pair) for pair in candidates):
            return proof(pending={'multiplex_id': native['multiplex_id'], 'layer_id': layer,
                                  'gene_ids': coverage['unchecked_gene_ids']}, tool='get_nodes_by_layer')
    if full_membership:
        members = sorted(gene for gene in genes if categories[gene] == 'layer_gene_ids')
        projected = frozenset(members)
        edges = [{'gene_ids': list(pair), 'weight': weight} for (selected_layer, pair), weight in sorted(positives.items())
                 if selected_layer == layer and set(pair) <= projected]
        lower = components(members, edges, kind)
        exact = len(lower) <= 1 or base.result is not None
        if not exact:
            def possible():
                yield from edges
                candidates = product(members, repeat=2) if view.layers[layer].directed else combinations(members, 2)
                for pair in candidates:
                    if pair[0] != pair[1] and not known(pair):
                        yield {'gene_ids': list(pair), 'weight': 0}

            exact = lower == components(members, possible(), kind)
        if exact:
            graph = {category: sorted(gene for gene in genes if categories[gene] == category)
                     for category in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids')}
            graph.update(edges=edges, coverage=coverage)
            return proof(result=component_result(view, selected, graph))
    return proof(pending=base.next_arguments, tool=base.next_tool)
