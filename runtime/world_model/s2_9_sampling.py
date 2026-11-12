"""Select verified input sets and disclose the finite construction domain.

The sampler measures real induced graphs before equal quotas across observed
edge-count cells. It does not enumerate the combinatorial subset distribution.
Feature cases and source size limits remain explicit in the audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from heapq import nsmallest
from typing import Any, Mapping, Sequence

from .s2_3_corpus import gene_group
from .s2_corpus import SPLITS, digest
from .s2_sources import GraphView


@dataclass(frozen=True)
class SubgraphFact:
    """One unique canonical input set before ordering, previews, or wording."""

    split: str
    layer_id: str
    gene_ids: tuple[str, ...]
    category_counts: tuple[int, int, int]
    edge_count: int
    isolated_count: int
    loop_count: int
    reciprocal_pairs: int
    descending_arc_count: int
    split_unit: str
    category: str = 'constructed'


def fact_from_genes(view: GraphView, index: Any, layer_id: str, gene_ids: Sequence[str],
                    seed: int = 20261028, *, split: str | None = None,
                    category: str = 'constructed') -> SubgraphFact:
    """Measure complete source labels while retaining fixed query-subject splits."""
    from .s2_subgraphs import SubgraphRuntime
    arguments = {'multiplex_id': view.multiplex_id, 'layer_id': layer_id, 'gene_ids': list(gene_ids),
                 'mode': 'complete', 'top_k': None, 'cursor': None}
    genes = SubgraphRuntime(view, index=index).validate_arguments(arguments)
    splits = {gene_group(gene, seed)[1] for gene in genes}
    if len(splits) > 1 or split is not None and splits and split not in splits:
        raise ValueError('an induced input set cannot cross fixed query-subject splits')
    actual_split = next(iter(splits)) if splits else split or 'train'
    if actual_split not in SPLITS:
        raise ValueError('unknown subgraph split')
    partition = index.partition(layer_id, genes)
    edges = index.edges(layer_id, genes)
    pairs = {tuple(edge['gene_ids']) for edge in edges}
    incident = {gene for pair in pairs for gene in pair}
    directed = view.layers[layer_id].directed
    counts = tuple(len(partition[key]) for key in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids'))
    unit = digest({'seed': seed, 'split': actual_split, 'layer_id': layer_id, 'gene_ids': sorted(genes)})
    return SubgraphFact(actual_split, layer_id, tuple(genes), counts, len(edges),
        len(set(partition['layer_gene_ids']) - incident), sum(a == b for a, b in pairs),
        sum(a < b and (b, a) in pairs for a, b in pairs) if directed else 0,
        sum(a > b for a, b in pairs) if directed else 0, unit, category)


def _compositions(size: int) -> list[tuple[int, int, int]]:
    """Enumerate small triples or named larger membership compositions."""
    if size <= 3:
        return [(a, b, size - a - b) for a in range(size + 1) for b in range(size - a + 1)]
    third = size // 3
    return sorted({(size, 0, 0), (0, size, 0), (0, 0, size), (size - 1, 1, 0),
                   (size - 1, 0, 1), (0, size // 2, size - size // 2), (size - 2 * third, third, third)})


def select_facts(view: GraphView, index: Any, registry: Any, counts: Mapping[str, int], seed: int,
                 *, smoke: bool = False, input_sizes: Sequence[int] | None = None) -> tuple[list[SubgraphFact], dict]:
    """Select equal quotas from measured finite cells plus source feature cases.

    The smoke domain uses three deterministic windows per composition. Explicit
    quotas use sixteen windows. Shortfalls and all unselected sizes are recorded.
    No construction clips a requested graph or removes an input after selection.
    """
    if set(counts) != set(SPLITS) or any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    raw_sizes = input_sizes if input_sizes is not None else (0, 1, 2, 3, 8, 32, 128)
    if (isinstance(raw_sizes, (str, bytes)) or not isinstance(raw_sizes, Sequence)
            or not raw_sizes or any(type(size) is not int or size < 0 for size in raw_sizes)):
        raise ValueError('input sizes must be explicit nonnegative integers')
    sizes = sorted(set(raw_sizes))
    missing = set(registry.symbols_by_id) - view.multiplex_nodes
    if not missing:
        raise ValueError('the registry has no globally absent candidates')
    units = {gene: gene_group(gene, seed) for gene in view.multiplex_nodes | missing}
    facts, cells, omissions, features, domains = {}, [], [], [], []
    trials = 3 if smoke else 16

    def add(fact):
        facts.setdefault((fact.split, fact.layer_id, tuple(sorted(fact.gene_ids))), fact)

    def feature(layer_id, split, genes, category):
        fact = fact_from_genes(view, index, layer_id, genes, seed, split=split, category=category)
        add(fact)
        features.append({'layer_id': layer_id, 'split': split, 'category': category,
                         'gene_ids': list(fact.gene_ids), 'edge_count': fact.edge_count,
                         'loop_count': fact.loop_count, 'reciprocal_pairs': fact.reciprocal_pairs,
                         'isolated_count': fact.isolated_count})

    for layer_id, layer in view.layers.items():
        full_pools = (layer.nodes, view.multiplex_nodes - layer.nodes, missing)
        source_features = {split: defaultdict(list) for split in SPLITS}
        for (first, second), weight in layer.edges.items():
            split = units[first][1]
            if units[second][1] != split:
                continue
            source_features[split]['native_edge'].append((first, second))
            if first == second:
                source_features[split]['self_loop'].append((first,))
            elif layer.directed and first < second and (second, first) in layer.edges:
                source_features[split]['reciprocal'].append((first, second))
            if layer.directed and first > second:
                source_features[split]['descending_arc'].append((first, second))
            if weight == 0:
                source_features[split]['zero_weight'].append((first, second))
        for split in SPLITS:
            pools = [sorted((gene for gene in pool if units[gene][1] == split), key=lambda gene: units[gene][0])
                     for pool in full_pools]
            maximum = sum(map(len, pools))
            ranges, previous = [], -1
            for size in [*(size for size in sizes if size <= maximum), maximum + 1]:
                if size > previous + 1:
                    ranges.append([previous + 1, size - 1])
                previous = size
            domains.append({'layer_id': layer_id, 'split': split, 'category_populations': list(map(len, pools)),
                'maximum_unique_input_size': maximum, 'requested_input_sizes': sizes,
                'unselected_input_size_ranges': ranges,
                'native_feature_candidates': {key: len(values) for key, values in source_features[split].items()},
                'all_combinatorial_subsets_enumerated': False})
            observed = defaultdict(dict)
            for size in sizes:
                for composition in _compositions(size):
                    if any(amount > len(pool) for amount, pool in zip(composition, pools)):
                        omissions.append({'layer_id': layer_id, 'split': split, 'input_size': size,
                            'composition': composition, 'reason': 'insufficient distinct IDs in a source category'})
                        continue
                    for trial in range(trials):
                        genes = []
                        for category, (amount, pool) in enumerate(zip(composition, pools)):
                            if amount:
                                offset = int(digest([seed, layer_id, split, size, composition, category, trial])[:16], 16) % len(pool)
                                genes.extend(pool[(offset + ordinal) % len(pool)] for ordinal in range(amount))
                        fact = fact_from_genes(view, index, layer_id, genes, seed, split=split)
                        key = size, composition, fact.edge_count
                        observed[key][tuple(sorted(fact.gene_ids))] = fact
            for (size, composition, edge_count), choices in sorted(observed.items()):
                ordered = sorted(choices.values(), key=lambda fact: fact.split_unit)
                selected = ordered[:counts[split]]
                for fact in selected:
                    add(fact)
                cells.append({'layer_id': layer_id, 'split': split, 'input_size': size,
                    'composition': composition, 'edge_count': edge_count, 'observed_unique_sets': len(ordered),
                    'requested': counts[split], 'selected': len(selected)})
                if len(selected) < counts[split]:
                    omissions.append({**cells[-1], 'reason': 'quota exceeds the verified finite candidate cell'})
            for category, candidates in source_features[split].items():
                chosen = nsmallest(counts[split], candidates,
                    key=lambda genes: digest([seed, layer_id, split, category, genes]))
                for genes in chosen:
                    feature(layer_id, split, genes, category)
            graph_only = [gene for gene in pools[0] if gene not in registry.symbols_by_id]
            for gene in graph_only[:counts[split]]:
                feature(layer_id, split, [gene], 'graph_only_query')
            # Build connected sets from real neighborhoods and keep all selected edges.
            anchors = sorted(pools[0], key=lambda gene: (-sum(len(index.native_index.neighbors(layer_id, gene, side))
                for side in index.native_index.directions(layer_id)), units[gene][0]))[:max(1, counts[split])]
            for anchor in anchors:
                neighbors = sorted({neighbor for side in index.native_index.directions(layer_id)
                    for neighbor, _ in index.native_index.neighbors(layer_id, anchor, side)
                    if neighbor != anchor and units[neighbor][1] == split}, key=lambda gene: units[gene][0])
                for size in sizes:
                    if size >= 2 and len(neighbors) >= size - 1:
                        feature(layer_id, split, [anchor, *neighbors[:size - 1]], 'source_star')
            edge_pairs = [pair for pair in source_features[split]['native_edge'] if pair[0] != pair[1]]
            if edge_pairs:
                pair = min(edge_pairs, key=lambda genes: digest([seed, layer_id, split, 'selected_isolate', genes]))
                adjacent = {neighbor for gene in pair for side in index.native_index.directions(layer_id)
                            for neighbor, _ in index.native_index.neighbors(layer_id, gene, side)}
                isolated = next((gene for gene in pools[0] if gene not in pair and gene not in adjacent
                    and all(neighbor != gene for side in index.native_index.directions(layer_id)
                            for neighbor, _ in index.native_index.neighbors(layer_id, gene, side))), None)
                if isolated is not None:
                    feature(layer_id, split, [*pair, isolated], 'selected_set_isolate')
    selected_facts = sorted(facts.values(), key=lambda fact: (
        SPLITS.index(fact.split), view.layer_order.index(fact.layer_id), len(fact.gene_ids),
        fact.category_counts, fact.edge_count, fact.category, fact.split_unit))
    return selected_facts, {'construction_policy': 'observed_subgraph_sets_v1', 'domains': domains,
        'cells': cells, 'feature_cases': features, 'omissions': omissions, 'input_sizes': sizes,
        'candidate_windows_per_composition': trials, 'selected_facts': len(selected_facts),
        'fact_selection_sha256': digest([asdict(fact) for fact in selected_facts]),
        'conditioning': 'equal quotas per observed edge-count/membership/size/layer/split cell plus declared feature cases',
        'split_policy': 'all unique query subjects use one fixed S2.3 SHA256(seed,gene) split; empty sets use their assigned split',
        'all_combinatorial_subsets_enumerated': False, 'runtime_input_cap': None, 'runtime_edge_cap': None,
        'edge_fact_disjoint': False, 'smoke': smoke,
        'remaining_domains': ['all unselected input sizes and membership compositions',
            'the unenumerated full combinatorial distribution of induced subsets',
            'additional pinned graph versions, source-supported PEN/other layers, and explicit full-layer isolates']}
