"""Select verified component strata from a disclosed finite source domain.

The candidate pool reuses the S2.9 source constructions and adds bridge, chain,
and strong-component witnesses. It never enumerates all graph subsets or
changes a source graph to fill an unattained component-count bin.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from .s2_3_corpus import gene_group
from .s2_9_sampling import fact_from_genes as graph_fact, select_facts as select_graph_facts
from .s2_components import components, validate_connectivity
from .s2_corpus import SPLITS, digest
from .s2_sources import GraphView


@dataclass(frozen=True)
class ComponentFact:
    """One source input and connectivity kind before wording or evidence choice."""

    split: str
    layer_id: str
    gene_ids: tuple[str, ...]
    connectivity_kind: str
    category_counts: tuple[int, int, int]
    edge_count: int
    components: tuple[tuple[str, ...], ...]
    component_count: int
    component_sizes: tuple[int, ...]
    single_component: bool
    isolated_count: int
    loop_count: int
    reciprocal_pairs: int
    split_unit: str
    category: str = 'constructed'


def _from_graph(view, index, fact, kind):
    kind = validate_connectivity(view, fact.layer_id, kind)
    partition = index.partition(fact.layer_id, fact.gene_ids)
    groups = components(partition['layer_gene_ids'], index.edges(fact.layer_id, fact.gene_ids), kind)
    return ComponentFact(fact.split, fact.layer_id, fact.gene_ids, kind, fact.category_counts,
        fact.edge_count, tuple(tuple(group) for group in groups), len(groups), tuple(sorted(map(len, groups))),
        len(groups) == 1 and not any(fact.category_counts[1:]), fact.isolated_count,
        fact.loop_count, fact.reciprocal_pairs, fact.split_unit, fact.category)


def fact_from_genes(view: GraphView, index: Any, layer_id: str, gene_ids: Sequence[str],
                    connectivity_kind: str | None, seed: int = 20261028, *,
                    split: str | None = None, category: str = 'constructed') -> ComponentFact:
    """Measure an exact native partition while keeping the existing query split."""
    return _from_graph(view, index, graph_fact(view, index, layer_id, gene_ids, seed,
                                              split=split, category=category), connectivity_kind)


def _targeted(view, index, registry, seed, quota):
    """Search finite source neighborhoods for component-specific contrasts."""
    facts, records = {}, []
    absent = set(registry.symbols_by_id) - view.multiplex_nodes
    groups = {gene: gene_group(gene, seed) for gene in view.multiplex_nodes | absent}

    def add(layer, split, genes, category):
        base = graph_fact(view, index, layer, genes, seed, split=split, category=category)
        facts.setdefault((layer, tuple(sorted(base.gene_ids))), base)
        records.append({'layer_id': layer, 'split': split, 'category': category, 'gene_ids': list(base.gene_ids),
                        'edge_count': base.edge_count})

    for layer, native in view.layers.items():
        for split in SPLITS:
            pool = sorted((gene for gene in native.nodes if groups[gene][1] == split), key=lambda gene: groups[gene][0])
            missing = sorted((gene for gene in absent if groups[gene][1] == split), key=lambda gene: groups[gene][0])
            reciprocal = []
            for source, target in native.edges:
                if groups[source][1] == split and groups[target][1] == split and source < target:
                    if native.directed and (target, source) in native.edges:
                        reciprocal.append((source, target))
            reciprocal.sort(key=lambda pair: digest([seed, layer, split, 'reciprocal', pair]))
            if missing:
                pair = next(((source, target) for source in pool
                    for target, _ in index.native_index.neighbors(layer, source, 'out' if native.directed else 'undirected')
                    if source != target and groups[target][1] == split), None)
                if pair:
                    add(layer, split, [*pair, missing[0]], 'connected_projection_with_exclusion')
            bridge_found, chain_found, cycle_found = 0, 0, 0
            # The fixed neighborhood window is a declared construction budget.
            for middle in pool:
                sides = index.native_index.directions(layer)
                neighbors = sorted({gene for side in sides for gene, _ in index.native_index.neighbors(layer, middle, side)
                    if gene != middle and groups[gene][1] == split}, key=lambda gene: groups[gene][0])[:12]
                if bridge_found < quota:
                    for first, second in combinations(neighbors, 2):
                        if not index.edges(layer, [first, second]):
                            add(layer, split, [first, second], 'removed_bridge')
                            records[-1]['bridge_gene_id'] = middle
                            bridge_found += 1
                            break
                if native.directed and chain_found < quota:
                    incoming = [gene for gene, _ in index.native_index.neighbors(layer, middle, 'in')
                        if gene != middle and groups[gene][1] == split][:12]
                    outgoing = [gene for gene, _ in index.native_index.neighbors(layer, middle, 'out')
                        if gene != middle and groups[gene][1] == split][:12]
                    for first in incoming:
                        candidate = next((last for last in outgoing if last != first and len(components(
                            [first, middle, last], index.edges(layer, [first, middle, last]), 'strong')) == 3), None)
                        if candidate:
                            add(layer, split, [first, middle, candidate], 'directed_chain')
                            chain_found += 1
                            break
                if native.directed and cycle_found < quota:
                    outgoing = [gene for gene, _ in index.native_index.neighbors(layer, middle, 'out')
                        if gene != middle and groups[gene][1] == split][:12]
                    incoming = {gene for gene, _ in index.native_index.neighbors(layer, middle, 'in')
                        if gene != middle and groups[gene][1] == split}
                    for second in outgoing:
                        third = next((gene for gene, _ in index.native_index.neighbors(layer, second, 'out')
                            if gene != second and gene != middle and gene in incoming), None)
                        if third:
                            add(layer, split, [middle, second, third], 'strong_cycle')
                            cycle_found += 1
                            break
                if bridge_found >= quota and (not native.directed or chain_found >= quota and cycle_found >= quota):
                    break
            multi = 0
            for first, second in combinations(reciprocal[:48], 2):
                genes = [*first, *second]
                if len(set(genes)) != 4:
                    continue
                sizes = sorted(map(len, components(genes, index.edges(layer, genes), 'strong')))
                if sizes == [2, 2]:
                    add(layer, split, genes, 'two_nontrivial_strong_groups')
                    multi += 1
                    if multi >= quota:
                        break
            prefix_found, examined = 0, 0
            if native.directed:
                core = [gene for gene in pool if all(any(neighbor != gene and groups[neighbor][1] == split
                    for neighbor, _ in index.native_index.neighbors(layer, gene, side)) for side in ('in', 'out'))][:128]
                for size in (2, 3):
                    for genes in combinations(core, size):
                        examined += 1
                        edges = index.edges(layer, genes)
                        if len(edges) > size and len(components(genes, edges[:-1], 'strong')) == 1:
                            add(layer, split, genes, 'partial_strong_witness')
                            prefix_found += 1
                            if prefix_found >= quota:
                                break
                    if prefix_found >= quota:
                        break
            records.append({'layer_id': layer, 'split': split, 'category': 'target_search',
                'removed_bridge_witnesses': bridge_found, 'directed_chain_witnesses': chain_found,
                'strong_cycle_witnesses': cycle_found, 'two_nontrivial_strong_witnesses': multi,
                'partial_strong_witnesses': prefix_found, 'strong_motif_candidates_examined': examined,
                'strong_motif_vertex_window': 128,
                'neighbor_window': 12, 'reciprocal_pair_window': 48,
                'limit': 'counts describe the declared search; zero does not assert absence from all possible subsets'})
    return list(facts.values()), records


def select_facts(view: GraphView, index: Any, registry: Any, counts: Mapping[str, int], seed: int,
                 *, smoke: bool = False, input_sizes: Sequence[int] | None = None) -> tuple[list[ComponentFact], dict]:
    """Allocate equal quotas across measured component counts and size multisets.

    All weak/strong versions of a source set share its fixed subject split.
    Feature allocations and the unenumerated subset distribution stay explicit.
    """
    if set(counts) != set(SPLITS) or any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    candidate_quota = 3 if smoke else 16
    bases, inherited = select_graph_facts(view, index, registry, dict.fromkeys(SPLITS, candidate_quota),
        seed, smoke=smoke, input_sizes=input_sizes)
    extras, targets = _targeted(view, index, registry, seed, 1 if smoke else max(counts.values()))
    graph_candidates = {(base.split, base.layer_id, tuple(sorted(base.gene_ids))): base for base in [*bases, *extras]}
    cells, chosen, histograms, quota_shortfalls = defaultdict(list), {}, defaultdict(Counter), []
    for base in graph_candidates.values():
        kinds = ('weak', 'strong') if view.layers[base.layer_id].directed else ('undirected',)
        for kind in kinds:
            fact = _from_graph(view, index, base, kind)
            key = (fact.split, fact.layer_id, len(fact.gene_ids), fact.category_counts,
                   kind, fact.component_count, fact.component_sizes)
            cells[key].append(fact)
            histograms[(fact.split, fact.layer_id, kind)][fact.component_count] += 1
    records = []
    for key, values in sorted(cells.items()):
        split, layer, size, membership, kind, count, sizes = key
        ordered = sorted(values, key=lambda fact: fact.split_unit)
        for fact in ordered[:counts[split]]:
            chosen[(fact.split_unit, kind)] = fact
        record = {'split': split, 'layer_id': layer, 'input_size': size, 'category_counts': membership,
            'connectivity_kind': kind, 'component_count': count, 'component_sizes': sizes,
            'candidate_sets': len(values), 'requested': counts[split], 'selected': min(counts[split], len(values))}
        records.append(record)
        if len(values) < counts[split]:
            quota_shortfalls.append(record)
    feature_choices = {}
    all_facts = {}
    for values in cells.values():
        for fact in values:
            all_facts[(fact.split_unit, fact.connectivity_kind)] = fact
            if fact.category != 'constructed':
                key = fact.split, fact.layer_id, fact.connectivity_kind, fact.category
                score = len(fact.gene_ids), fact.edge_count, fact.split_unit
                if key not in feature_choices or score < feature_choices[key][0]:
                    feature_choices[key] = score, fact
    for _, fact in feature_choices.values():
        chosen[(fact.split_unit, fact.connectivity_kind)] = fact
    paired_additions = 0
    for fact in list(chosen.values()):
        if fact.connectivity_kind in ('weak', 'strong'):
            alternate = 'strong' if fact.connectivity_kind == 'weak' else 'weak'
            key = fact.split_unit, alternate
            if key not in chosen:
                chosen[key] = all_facts[key]
                paired_additions += 1
    facts = sorted(chosen.values(), key=lambda fact: (SPLITS.index(fact.split), view.layer_order.index(fact.layer_id),
        len(fact.gene_ids), fact.connectivity_kind, fact.category_counts, fact.component_count, fact.category, fact.split_unit))
    return facts, {'construction_policy': 'observed_component_sets_v1',
        'candidate_graph_policy': inherited['construction_policy'], 'candidate_graphs': len(graph_candidates),
        'candidate_quota_per_graph_cell': candidate_quota, 'domains': inherited['domains'],
        'input_sizes': inherited['input_sizes'], 'inherited_graph_allocation': inherited,
        'targeted_source_cases': targets, 'component_cells': records, 'quota_shortfalls': quota_shortfalls,
        'feature_representatives': len(feature_choices), 'matched_kind_additions': paired_additions,
        'component_count_histograms': [{'split': split, 'layer_id': layer, 'connectivity_kind': kind,
            'histogram': dict(sorted(histogram.items()))} for (split, layer, kind), histogram in sorted(histograms.items())],
        'selected_facts': len(facts), 'fact_selection_sha256': digest([asdict(fact) for fact in facts]),
        'conditioning': 'equal quotas per observed component-count and size-multiset cell, plus explicit native-structure witnesses',
        'split_policy': inherited['split_policy'], 'matched_directed_kinds': True,
        'all_combinatorial_subsets_enumerated': False, 'runtime_input_cap': None, 'runtime_component_cap': None,
        'edge_fact_disjoint': False, 'smoke': smoke,
        'remaining_domains': ['unselected source sizes and unenumerated induced subsets',
            'additional eligible source families and graph versions, cold identities, and production allocation']}
