"""Select real pair observations and family controls before any wording choice."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from .s2_12_sampling import _registry_state
from .s2_corpus import BUCKETS, SPLITS, _pair_group, digest
from .s2_corpus import select_facts as _edge_facts
from .s2_multiplex import MultiplexRuntime


@dataclass(frozen=True)
class PairFact:
    """One native pair and scope, independent of its request and observations."""

    split: str
    layer_id: str
    gene_ids: tuple[str, str]
    category: str
    split_unit: str
    layer_family: str
    directed: bool
    weight: int | float | None
    reverse_exists: bool | None
    registry_states: tuple[str, str]
    positive_layers: tuple[str, ...]


def fact_from_pair(view, index, registry, old_view, layer_id, gene_ids, seed=20261028):
    """Measure native source facts without a registry filter or an inferred edge."""
    pair = tuple(gene_ids)
    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError('a specificity fact requires two distinct canonical IDs')
    native = view.layers[layer_id]
    key = pair if native.directed else tuple(sorted(pair))
    missing = set(pair) - view.multiplex_nodes
    outside = set(pair) - native.nodes
    category = ('outside_multiplex' if missing else 'outside_layer' if outside else
                'recorded' if key in native.edges else 'no_edge')
    unit, split = _pair_group(pair, seed)
    return PairFact(split, layer_id, pair, category, unit, MultiplexRuntime(view).family_of(layer_id),
        native.directed, native.edges.get(key),
        (pair[::-1] in native.edges) if native.directed and not missing else None,
        tuple(_registry_state(gene, registry) for gene in pair),
        tuple(layer for layer in view.layer_order
              if (pair if view.layers[layer].directed else tuple(sorted(pair))) in view.layers[layer].edges))


def select_facts(view, index, registry, counts, seed=20261028, *, smoke=False, old_view=None):
    """Sample the S2.1 strata, then add available semantic and identity controls.

    Complete inventories determine the native pair counts. Nonedges and
    additional controls are sampled, rather than an exhaustive pair census.
    All queries on the same unordered pair use its existing S2.1 split.
    """
    if set(counts) != set(SPLITS) or any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError('counts require a positive integer for each split')
    baseline = _edge_facts(view, sorted(registry.symbols_by_id), counts, seed)
    facts, witnesses, gaps = {}, [], []

    def add(layer, pair, feature):
        fact = fact_from_pair(view, index, registry, old_view, layer, pair, seed)
        key = fact.layer_id, fact.gene_ids
        facts[key] = fact
        witnesses.append({'feature': feature, 'split': fact.split, 'layer_id': layer,
                          'gene_ids': list(pair), 'category': fact.category})
        return fact

    def cover(layer, pairs, feature):
        found = set()
        for pair in pairs:
            split = _pair_group(tuple(pair), seed)[1]
            if split not in found:
                add(layer, pair, feature)
                found.add(split)
            if len(found) == len(SPLITS):
                break
        for split in SPLITS:
            if split not in found:
                gaps.append({'feature': feature, 'layer_id': layer, 'split': split})

    for fact in baseline:
        add(fact.layer_id, fact.gene_ids, 'baseline:' + fact.bucket)
    state_pools = {}
    for gene in sorted(view.multiplex_nodes | set(registry.symbols_by_id)):
        state_pools.setdefault(_registry_state(gene, registry), []).append(gene)
    domains = []
    for layer_id in view.layer_order:
        native = view.layers[layer_id]
        eligible = [pair for pair in native.edges if pair[0] != pair[1]]
        cover(layer_id, sorted(eligible, key=lambda pair: (-native.edges[pair], pair)), 'high_weight')
        if native.directed:
            for reciprocal in (False, True):
                cover(layer_id, (pair for pair in sorted(eligible)
                    if (pair[::-1] in native.edges) == reciprocal),
                    'reciprocal' if reciprocal else 'one_way')
        members = sorted(native.nodes)
        for state in ('unique_symbol', 'null_symbol', 'ambiguous_symbol', 'graph_only'):
            anchors = state_pools.get(state, ())
            # One authoritative identity is enough to vary its partner and split.
            cover(layer_id, ((anchors[0], other) for other in members if anchors and anchors[0] != other),
                  'registry:' + state)
        nodes, edges = len(native.nodes), len(eligible)
        possible = nodes * (nodes - 1) if native.directed else nodes * (nodes - 1) // 2
        outside = len(view.multiplex_nodes - native.nodes)
        missing = len(set(registry.symbols_by_id) - view.multiplex_nodes)
        domains.append({'layer_id': layer_id, 'layer_family': MultiplexRuntime(view).family_of(layer_id),
            'directed': native.directed, 'source_nodes': nodes, 'native_source_records': len(native.edges),
            'eligible_distinct_endpoint_edges': edges, 'self_loops_excluded_from_pair_queries': len(native.edges) - edges,
            'possible_distinct_endpoint_pairs': possible, 'checked_member_nonedge_population': possible - edges,
            'one_layer_member_outside_layer_pairs': nodes * outside * (2 if native.directed else 1),
            'one_layer_member_registry_missing_pairs': nodes * missing * (2 if native.directed else 1)})
    bulk = [layer for layer in view.layer_order if MultiplexRuntime(view).family_of(layer) == 'bulk_pen']
    single = [layer for layer in view.layer_order if MultiplexRuntime(view).family_of(layer) == 'scpen']
    for first in bulk:
        for second in single:
            shared_nodes = view.layers[first].nodes & view.layers[second].nodes
            pairs = sorted(set(view.layers[first].edges) | set(view.layers[second].edges))
            for presence in ((True, True), (True, False), (False, True)):
                qualified = [pair for pair in pairs if pair[0] != pair[1] and set(pair) <= shared_nodes
                    and (pair in view.layers[first].edges, pair in view.layers[second].edges) == presence]
                for layer in (first, second):
                    cover(layer, qualified, f'pen_contrast:{int(presence[0])}{int(presence[1])}')
    for layer_id in view.layer_order:
        if MultiplexRuntime(view).family_of(layer_id) != 'ppi':
            continue

        def routes():
            for source in sorted(view.layers[layer_id].nodes):
                for middle, _ in index.native_index.neighbors(layer_id, source, 'undirected'):
                    if source == middle:
                        continue
                    for target, _ in index.native_index.neighbors(layer_id, middle, 'undirected'):
                        if target not in (source, middle) and tuple(sorted((source, target))) not in view.layers[layer_id].edges:
                            yield source, target

        cover(layer_id, routes(), 'two_edge_route_without_direct_record')
    ordered = sorted(facts.values(), key=lambda fact: (SPLITS.index(fact.split), view.layer_order.index(fact.layer_id),
                                                       BUCKETS.index(fact.category), fact.gene_ids))
    histogram = Counter((fact.split, fact.layer_id, fact.category) for fact in ordered)
    return ordered, {'construction_policy': 'native_pair_strata_and_observed_specificity_controls_v1',
        'selected_facts': len(ordered), 'baseline_facts': len(baseline), 'smoke': smoke,
        'fact_selection_sha256': digest([asdict(fact) for fact in ordered]), 'domains': domains,
        'feature_witnesses': witnesses, 'unattained_features': gaps,
        'selected_strata': [{'split': key[0], 'layer_id': key[1], 'category': key[2], 'count': count}
                            for key, count in sorted(histogram.items())],
        'source_endpoint_genes': len(view.multiplex_nodes), 'registry_genes': len(registry.symbols_by_id),
        'all_source_endpoints_retained': True, 'all_possible_pairs_enumerated': False, 'runtime_pair_cap': None,
        'query_pair_disjoint': True, 'gene_disjoint': False, 'edge_fact_disjoint': False,
        'split_policy': 'The fixed S2.1 SHA256(seed,sorted endpoint pair) bins retain every orientation and layer contrast in one split.',
        'taxonomy_only_families': ['other'], 'unqualified_sources': ['HumanNetV3:cocitation'],
        'remaining_domains': ['More qualified contexts and representations, including an actual other-family source.',
            'Additional pair, identity, and failure combinations beyond these sampled controls.',
            'Causal evidence, regulatory signs, expression measurements, and cross-source weight calibration.']}
