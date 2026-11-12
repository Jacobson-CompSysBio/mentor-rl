"""Select real degree strata without changing full reference populations.

The smoke recipe selects declared degree representatives. Explicit quotas
visit every attainable degree in each scope, direction, and query split.
Neither recipe filters the population used for ranks or drops graph-only IDs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from fractions import Fraction
from typing import Any, Mapping

from .s2_3_corpus import gene_group
from .s2_corpus import SPLITS, canonical_json, digest
from .s2_multiplex import MultiplexRuntime
from .s2_sources import GraphView


@dataclass(frozen=True)
class DegreeFact:
    """One source query before request type, policy, wording, or evidence."""

    split: str
    gene_id: str
    layer_scope: dict
    direction: str
    degree: int | None
    membership: dict
    rank_support: dict | None
    split_unit: str
    category: str = 'present'


def fact_from_arguments(view: GraphView, index: Any, gene_id: str, layer_scope: Mapping,
                        direction: str, seed: int = 20261028, category: str = 'present') -> DegreeFact:
    """Measure source labels while retaining the existing fixed gene split."""
    from .s2_degrees import DegreeRuntime
    arguments = {'multiplex_id': view.multiplex_id, 'layer_scope': dict(layer_scope),
                 'gene_id': gene_id, 'direction': direction, 'request_kind': 'degree_only', 'hub_policy': None}
    selected = DegreeRuntime(view, index=index).validate_arguments(arguments)
    population = index.population(selected)
    present, in_scope = gene_id in view.multiplex_nodes, gene_id in population
    degree = index.degree(selected, gene_id, direction) if present else None
    support = index.rank_support(selected, gene_id, direction) if present and in_scope else None
    unit, split = gene_group(gene_id, seed)
    return DegreeFact(split, gene_id, dict(layer_scope), direction, degree,
        {'present_in_multiplex': present, 'present_in_scope': in_scope if present else None}, support, unit, category)


def scope_schedule(view: GraphView) -> list[tuple[dict, str]]:
    """Visit every nonempty explicit layer subset, all layers, and each family."""
    runtime = MultiplexRuntime(view)
    scopes = [{'kind': 'layers', 'layer_ids': list(layers), 'layer_family': None}
              for size in range(1, len(view.layer_order) + 1) for layers in combinations(view.layer_order, size)]
    scopes.extend({'kind': 'all', 'layer_ids': [], 'layer_family': family}
                  for family in (None, *runtime.families))
    result = []
    for scope in scopes:
        selected = runtime.resolve_scope(scope)
        kinds = {view.layers[layer].directed for layer in selected}
        directions = (('incident', 'in', 'out') if kinds == {True} else
                      ('incident', 'undirected') if kinds == {False} else
                      ('incident', 'in', 'out', 'undirected') if not kinds else ('incident',))
        result.extend((scope, direction) for direction in directions)
    return result


def _rank_position(support: Mapping) -> tuple[int, int]:
    """Return the exact percentile numerator and denominator for sampling."""
    n = support['population_count']
    return (50, 1) if n == 1 else (
        100 * (2 * support['lower_degree_count'] + support['equal_degree_count'] - 1), 2 * (n - 1))


def select_facts(view: GraphView, index: Any, registry: Any, counts: Mapping[str, int],
                 seed: int, *, smoke: bool = False) -> tuple[list[DegreeFact], dict]:
    """Audit complete count domains before a reproducible query allocation.

    Smoke includes small degrees, a median stratum, the maximum, and attainable
    ranks on both sides of 95. All unselected degree strata remain explicit.
    """
    if set(counts) != set(SPLITS) or any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError('counts require positive integers for all splits')
    missing = sorted(set(registry.symbols_by_id) - view.multiplex_nodes,
                     key=lambda gene: gene_group(gene, seed)[0])
    if not missing:
        raise ValueError('the registry has no globally absent source candidates')
    facts, domains, cells, omissions, feature_cells = {}, [], [], [], []

    def add(gene, scope, direction, category='present'):
        fact = fact_from_arguments(view, index, gene, scope, direction, seed, category)
        facts.setdefault((gene, canonical_json(scope), direction), fact)

    for scope, direction in scope_schedule(view):
        selected = MultiplexRuntime(view).resolve_scope(scope)
        population = index.population(selected)
        histogram = index.histogram(selected, direction)
        pools = defaultdict(list)
        for gene in sorted(population, key=lambda gene: gene_group(gene, seed)[0]):
            pools[(gene_group(gene, seed)[1], index.degree(selected, gene, direction))].append(gene)
        domains.append({'layer_scope': scope, 'resolved_layer_ids': list(selected), 'direction': direction,
            'population_policy_id': 'selected_layer_node_union_v1', 'population_count': len(population),
            'degree_histogram': histogram, 'attainable_degree_values': len(histogram),
            'graph_ids_outside_registry': len(population - registry.symbols_by_id.keys())})
        for split in SPLITS:
            available = sorted(degree for (cell_split, degree) in pools if cell_split == split)
            chosen = set(available)
            if smoke and available:
                chosen = {*available[:4], available[len(available) // 2], available[-1]}
                ranked = []
                for degree in available:
                    support = index.rank_support(selected, pools[(split, degree)][0], direction)
                    numerator, denominator = _rank_position(support)
                    ranked.append((numerator - 95 * denominator, degree, denominator))
                below = [row for row in ranked if row[0] < 0]
                above = [row for row in ranked if row[0] >= 0]
                if below:
                    chosen.add(max(below, key=lambda row: Fraction(row[0], row[2]))[1])
                if above:
                    chosen.add(min(above, key=lambda row: Fraction(row[0], row[2]))[1])
            cells.append({'layer_scope': scope, 'direction': direction, 'split': split,
                'available_degree_values': available, 'selected_degree_values': sorted(chosen),
                'unselected_degree_values': sorted(set(available) - chosen),
                'query_count_by_degree': {degree: len(pools[(split, degree)]) for degree in available},
                'requested_per_degree': counts[split],
                'selected_per_degree': {degree: min(counts[split], len(pools[(split, degree)])) for degree in sorted(chosen)}})
            for degree in sorted(chosen):
                pool = pools[(split, degree)]
                sample = pool[:counts[split]]
                for gene in sample:
                    add(gene, scope, direction)
                if len(sample) < counts[split]:
                    omissions.append({'scope': scope, 'direction': direction, 'split': split, 'degree': degree,
                        'population': len(pool), 'requested': counts[split], 'selected': len(sample)})
            for category, pool in (
                    ('outside_scope', view.multiplex_nodes - population), ('outside_multiplex', missing)):
                subjects = sorted((gene for gene in pool if gene_group(gene, seed)[1] == split),
                                  key=lambda gene: gene_group(gene, seed)[0])
                for gene in subjects[:counts[split]]:
                    add(gene, scope, direction, category)
                feature_cells.append({'layer_scope': scope, 'direction': direction, 'split': split,
                    'category': category, 'candidate_genes': len(subjects), 'selected_genes': subjects[:counts[split]]})
            # Select overlap witnesses independently of degree representatives.
            features = defaultdict(list)
            for gene in (gene for (cell_split, _), values in pools.items() if cell_split == split for gene in values):
                sets = [{neighbor for neighbor, _ in index.native_index.neighbors(layer, gene, native)}
                        for layer in selected
                        for native in (index.native_index.directions(layer) if direction == 'incident' else (direction,))]
                if any(gene in values for values in sets):
                    features['self_loop'].append(gene)
                if sum(map(len, sets)) > len(set().union(*sets)):
                    features['overlap'].append(gene)
                if gene not in registry.symbols_by_id:
                    features['graph_only_query'].append(gene)
            for category, subjects in features.items():
                subjects.sort(key=lambda gene: (index.degree(selected, gene, direction), gene_group(gene, seed)[0]))
                for gene in subjects[:counts[split]]:
                    add(gene, scope, direction, category)
                feature_cells.append({'layer_scope': scope, 'direction': direction, 'split': split,
                    'category': category, 'candidate_genes': len(subjects), 'selected_genes': subjects[:counts[split]]})
    selected_facts = sorted(facts.values(), key=lambda fact: (
        SPLITS.index(fact.split), canonical_json(fact.layer_scope), fact.direction, fact.category, fact.split_unit))
    return selected_facts, {'domains': domains, 'cells': cells, 'feature_cells': feature_cells, 'omissions': omissions,
        'selected_facts': len(selected_facts), 'fact_selection_sha256': digest([asdict(fact) for fact in selected_facts]),
        'split_policy': 'fixed S2.3 SHA256(seed,gene) query split; every reference population retains all source members',
        'degree_policy_id': 'distinct_neighbor_union_v1', 'population_policy_id': 'selected_layer_node_union_v1',
        'runtime_degree_cap': None, 'registry_filter_applied_to_populations': False, 'smoke': smoke,
        'conditioning': 'equal explicit quotas per attainable degree/scope/direction/query-split cell plus recorded feature/prerequisite contrasts; smoke uses recorded representatives',
        'remaining_domains': ['complete production degree allocation beyond the declared smoke strata',
            'additional graph versions and source-supported PEN/other layers and explicit isolated nodes']}
