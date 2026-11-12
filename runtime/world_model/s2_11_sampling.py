"""Select full-layer path facts from a disclosed finite source domain.

Each selected source has a complete breadth-first search. Query endpoints
keep the fixed gene split; intermediate witness nodes can belong to any split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .s2_3_corpus import gene_group
from .s2_corpus import SPLITS, digest
from .s2_components import components
from .s2_paths import PathIndex, PathRuntime
from .s2_sources import GraphView


@dataclass(frozen=True)
class PathFact:
    """One ordered source pair before wording, evidence, or failure selection."""

    split: str
    layer_id: str
    gene_ids: tuple[str, str]
    direction: str
    category_counts: tuple[int, int, int]
    path_found: bool | None
    path_gene_ids: tuple[str, ...] | None
    hop_count: int | None
    tied_shortest_paths: int | None
    split_unit: str
    category: str = 'constructed'


@lru_cache(maxsize=32)
def _path_counts(index: PathIndex, layer_id: str, source: str) -> dict[str, int]:
    """Count every minimum-hop route in the complete distance DAG."""
    distances = index.distances(layer_id, source)
    counts = dict.fromkeys(distances, 0)
    if not distances:
        return counts
    counts[source] = 1
    direction = 'out' if index.view.layers[layer_id].directed else 'undirected'
    for gene in sorted(distances, key=lambda item: (distances[item], item)):
        for target, _ in index.native_index.neighbors(layer_id, gene, direction):
            if distances.get(target) == distances[gene] + 1:
                counts[target] += counts[gene]
    return counts


def fact_from_genes(view: GraphView, index: PathIndex, layer_id: str, gene_ids: Sequence[str],
                    seed: int = 20261028, *, split: str | None = None,
                    category: str = 'constructed') -> PathFact:
    """Measure native shortest paths while preserving both endpoint positions."""
    direction = 'source_to_target' if view.layers[layer_id].directed else 'undirected'
    arguments = PathRuntime(view).arguments(list(gene_ids), layer_id, direction)
    endpoints = PathRuntime(view).validate_arguments(arguments)
    groups = {gene_group(gene, seed)[1] for gene in endpoints}
    if len(groups) != 1 or split is not None and split not in groups:
        raise ValueError('both query endpoints must belong to the same fixed gene split')
    selected = set(endpoints)
    counts = (len(selected & view.layers[layer_id].nodes),
              len((selected & view.multiplex_nodes) - view.layers[layer_id].nodes),
              len(selected - view.multiplex_nodes))
    result = PathRuntime(view, index=index).shortest_path(**arguments, evidence_id='source_label')['result']
    ties = (None if result['path_found'] is None else 0 if not result['path_found'] else
            1 if result['hop_count'] <= 1 else
            _path_counts(index, layer_id, endpoints[0])[endpoints[1]])
    unit = digest({'seed': seed, 'gene_ids': sorted(selected)})
    return PathFact(next(iter(groups)), layer_id, endpoints, direction, counts,
        result['path_found'], None if result['path_gene_ids'] is None else tuple(result['path_gene_ids']),
        result['hop_count'], ties, unit, category)


def select_facts(view: GraphView, index: PathIndex, registry: Any, counts: Mapping[str, int],
                 seed: int, *, smoke: bool = False, source_window: int | None = None) -> tuple[list[PathFact], dict]:
    """Allocate equal quotas across observed hops and endpoint membership outcomes.

    The finite source window limits construction work only. Every search uses
    the full layer, and every same-split target contributes to its hop audit.
    """
    if set(counts) != set(SPLITS) or any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    window = (8 if smoke else 32) if source_window is None else source_window
    if type(window) is not int or window < 1:
        raise ValueError('source_window must be a positive integer')
    absent = set(registry.symbols_by_id) - view.multiplex_nodes
    genes = view.multiplex_nodes | absent
    groups = {gene: gene_group(gene, seed) for gene in genes}
    cells, domains, selected, features, omissions = {}, [], {}, [], []

    def order(pool):
        return sorted(pool, key=lambda gene: groups[gene][0])

    def add(layer, pair, category='constructed'):
        fact = fact_from_genes(view, index, layer, pair, seed, category=category)
        key = (fact.split, layer, tuple(pair))
        if key not in selected or selected[key].category == 'constructed' and category != 'constructed':
            selected[key] = fact
        return fact

    for layer_id, layer in view.layers.items():
        full_components = [] if layer.directed else components(sorted(layer.nodes), layer.edges, 'undirected')
        for split in SPLITS:
            members = order(gene for gene in layer.nodes if groups[gene][1] == split)
            outside = order(gene for gene in view.multiplex_nodes - layer.nodes if groups[gene][1] == split)
            missing = order(gene for gene in absent if groups[gene][1] == split)
            native_rows = {gene: index.native_index.neighbors(layer_id, gene,
                'out' if layer.directed else 'undirected') for gene in members}
            source_categories = {'all_members': members,
                'nonempty_outgoing': [gene for gene in members if any(target != gene for target, _ in native_rows[gene])]}
            if layer.directed:
                source_categories['internal_directed'] = [gene for gene in source_categories['nonempty_outgoing']
                    if any(source != gene for source, _ in index.native_index.neighbors(layer_id, gene, 'in'))]
            source_features = {
                'graph_only_query': [gene for gene in members if gene not in registry.symbols_by_id],
                'self_loop': [gene for gene in members if any(target == gene for target, _ in native_rows[gene])],
                'zero_weight': [gene for gene in members if any(weight == 0 for _, weight in native_rows[gene])]}
            sources = order({gene for pool in source_categories.values() for gene in pool[:window]}
                | {gene for pool in source_features.values() for gene in pool[:counts[split]]})
            histogram, tied_histogram, unreachable = Counter(), Counter(), 0
            candidates = defaultdict(list)
            feature_choices = {}
            for source in sources:
                distances, ties = index.distances(layer_id, source), _path_counts(index, layer_id, source)
                native = index.native_index.neighbors(layer_id, source, 'out' if layer.directed else 'undirected')
                for target in members:
                    hops = distances.get(target)
                    if hops is None:
                        unreachable += 1
                    else:
                        histogram[hops] += 1
                        if ties[target] > 1:
                            tied_histogram[hops] += 1
                    key = 'no_path' if hops is None else hops
                    candidates[key].append((source, target))
                    tags = []
                    if ties.get(target, 0) > 1:
                        tags.append('tied_shortest')
                    if source == target:
                        tags.append('same_node_loop' if any(gene == source for gene, _ in native) else 'same_node_no_loop')
                    if source not in registry.symbols_by_id or target not in registry.symbols_by_id:
                        tags.append('graph_only_query')
                    if source != target and not native:
                        tags.append('empty_outgoing' if layer.directed else 'full_layer_isolate')
                    if hops == 1 and any(gene == target and weight == 0 for gene, weight in native):
                        tags.append('zero_weight_path')
                    for feature in tags:
                        score = (len(native), digest([seed, layer_id, source, target]))
                        if feature not in feature_choices or score < feature_choices[feature][0]:
                            feature_choices[feature] = score, (source, target)
            for hops, pairs in candidates.items():
                ranked = sorted(pairs, key=lambda pair: digest([seed, layer_id, pair]))
                for pair in ranked[:counts[split]]:
                    add(layer_id, pair)
                cell = {'split': split, 'layer_id': layer_id, 'outcome': hops,
                        'candidate_pairs': len(pairs), 'requested': counts[split],
                        'selected': min(len(pairs), counts[split])}
                cells[(split, layer_id, str(hops))] = cell
                if len(pairs) < counts[split]:
                    omissions.append(cell)
            for feature, (_, pair) in sorted(feature_choices.items()):
                add(layer_id, pair, feature)
                features.append({'split': split, 'layer_id': layer_id, 'feature': feature, 'gene_ids': list(pair)})
            closed = []
            for group in full_components:
                if len(group) > 8:
                    continue
                subjects = order(set(group) & set(members))
                targets = [gene for gene in members if gene not in group]
                if subjects and targets:
                    closed.append((len(group), sum(len(index.native_index.neighbors(layer_id, gene, 'undirected')) for gene in group),
                                   groups[subjects[0]][0], (subjects[0], targets[0]), group))
            for _, _, _, pair, group in sorted(closed)[:counts[split]]:
                add(layer_id, pair, 'closed_component')
                features.append({'split': split, 'layer_id': layer_id, 'feature': 'closed_component',
                                 'gene_ids': list(pair), 'component_gene_ids': group,
                                 'construction_component_size_budget': 8})
            categories = {'member': members, 'outside_layer': outside, 'missing': missing}
            for left, first in categories.items():
                for right, second in categories.items():
                    if not first or not second:
                        omissions.append({'split': split, 'layer_id': layer_id, 'membership': [left, right],
                                          'reason': 'an endpoint category has no source candidate'})
                        continue
                    for ordinal in range(counts[split]):
                        pair = (first[ordinal % len(first)], second[(ordinal + (left == right)) % len(second)])
                        if left != 'member' or right != 'member':
                            add(layer_id, pair, f'{left}_{right}')
                    if left != 'member' and left == right:
                        add(layer_id, (first[0], first[0]), f'repeated_{left}')
            domains.append({'split': split, 'layer_id': layer_id, 'directed': layer.directed,
                'source_genes': sources, 'source_window': window, 'available_member_sources': len(members),
                'source_categories': {name: {'available': len(pool), 'window_selected': min(window, len(pool))}
                                      for name, pool in source_categories.items()},
                'feature_source_categories': {name: {'available': len(pool), 'selected': min(counts[split], len(pool))}
                                             for name, pool in source_features.items()},
                'same_split_targets_per_source': len(members), 'reachable_hop_histogram': dict(sorted(histogram.items())),
                'tied_hop_histogram': dict(sorted(tied_histogram.items())), 'unreachable_pairs': unreachable,
                'searched_pairs': len(sources) * len(members), 'full_layer_members': len(layer.nodes),
                'full_layer_edges': len(layer.edges)})
    # A reverse query has its own witness even in an undirected layer.
    for fact in list(selected.values()):
        add(fact.layer_id, tuple(reversed(fact.gene_ids)), 'matched_reverse')
    facts = sorted(selected.values(), key=lambda fact: (SPLITS.index(fact.split),
        view.layer_order.index(fact.layer_id), groups[fact.gene_ids[0]][0], fact.gene_ids[1]))
    return facts, {'construction_policy': 'finite_full_layer_bfs_sources_v1', 'source_window': window,
        'domains': domains, 'cells': list(cells.values()), 'feature_witnesses': features, 'omissions': omissions,
        'selected_facts': len(facts), 'fact_selection_sha256': digest([asdict(fact) for fact in facts]),
        'split_policy': 'Both query endpoints use the fixed S2.3 SHA256(seed,gene) bins; route intermediates are unrestricted.',
        'matched_reverse_queries': True, 'all_source_pairs_enumerated': False,
        'runtime_hop_cap': None, 'runtime_query_cap': None, 'edge_fact_disjoint': False, 'smoke': smoke,
        'remaining_domains': ['sources outside the finite construction window and their unmeasured hop strata',
            'additional eligible layer families, full-layer-isolate inventories, graph versions, broader identities, and production allocation']}
