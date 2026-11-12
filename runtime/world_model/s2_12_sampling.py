"""Select singleton membership facts from the complete qualified source domain.

Observed layer profiles and fixed gene splits define the finite allocation.
All selected genes retain matched facts across every registered layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping

from .s2_2_identity import RegistryContext
from .s2_3_corpus import gene_group
from .s2_corpus import SPLITS, digest
from .s2_sources import GraphView
from .s2_subgraphs import SubgraphIndex


@dataclass(frozen=True)
class MembershipFact:
    """One gene and layer before a request, context, or failure is selected."""

    split: str
    gene_id: str
    layer_id: str
    category: str
    layer_ids: tuple[str, ...]
    old_present: bool
    registry_state: str
    outgoing_count: int | None
    incoming_count: int | None
    self_loop: bool
    split_unit: str


def _registry_state(gene: str, registry: RegistryContext) -> str:
    if gene not in registry.symbols_by_id:
        return 'graph_only'
    symbol = registry.symbols_by_id[gene]
    if symbol is None:
        return 'null_symbol'
    return 'unique_symbol' if registry.gene_ids_by_symbol[symbol.casefold()] == (gene,) else 'ambiguous_symbol'


def fact_from_gene(view: GraphView, index: SubgraphIndex, registry: RegistryContext,
                   old_view: GraphView, layer_id: str, gene_id: str,
                   seed: int = 20261028) -> MembershipFact:
    """Measure exact membership without a registry filter or an invented isolate."""
    unit, split = gene_group(gene_id, seed)
    layers = tuple(layer for layer in view.layer_order if gene_id in view.layers[layer].nodes)
    category = 'member' if layer_id in layers else 'outside_layer' if layers else 'outside_multiplex'
    native = view.layers[layer_id]
    outgoing = index.native_index.neighbors(layer_id, gene_id, 'out' if native.directed else 'undirected')
    incoming = index.native_index.neighbors(layer_id, gene_id, 'in') if native.directed else ()
    return MembershipFact(split, gene_id, layer_id, category, layers, gene_id in old_view.multiplex_nodes,
        _registry_state(gene_id, registry), len(outgoing) if layers else None,
        len(incoming) if native.directed and layers else None,
        any(target == gene_id for target, _ in outgoing), unit)


def select_facts(view: GraphView, index: SubgraphIndex, registry: RegistryContext,
                 counts: Mapping[str, int], seed: int,
                 *, smoke: bool = False, old_view: GraphView) -> tuple[list[MembershipFact], dict]:
    """Draw equal quotas from observed complete layer profiles, then add real features.

    The finite domain includes all source endpoints and pinned registry IDs.
    It does not enumerate every syntactically possible absent identifier.
    """
    if set(counts) != set(SPLITS) or any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError('counts require positive integers for all three splits')
    candidates = view.multiplex_nodes | set(registry.symbols_by_id)
    profiles, by_profile, selected, domains = {}, defaultdict(list), set(), []
    for gene in sorted(candidates):
        profile = tuple(layer for layer in view.layer_order if gene in view.layers[layer].nodes)
        profiles[gene] = profile
        by_profile[(gene_group(gene, seed)[1], profile)].append(gene)
    cells, omissions = [], []
    for (split, profile), genes in sorted(by_profile.items()):
        ordered = sorted(genes, key=lambda gene: gene_group(gene, seed)[0])
        chosen = ordered[:counts[split]]
        selected.update(chosen)
        record = {'split': split, 'layer_ids': list(profile), 'containing_layer_count': len(profile),
                  'population': len(genes), 'requested': counts[split], 'selected': len(chosen)}
        cells.append(record)
        if len(chosen) < counts[split]:
            omissions.append(record)
    features = {}
    for gene in sorted(candidates):
        split = gene_group(gene, seed)[1]
        state = _registry_state(gene, registry)
        for layer_id in view.layer_order:
            fact = fact_from_gene(view, index, registry, old_view, layer_id, gene, seed)
            tags = [f'registry:{state}']
            if fact.layer_ids and not fact.old_present:
                tags.append('new_view_only')
            if fact.category == 'member':
                if not fact.self_loop:
                    tags.append('edgeless_singleton')
                else:
                    tags.append('self_loop')
                if view.layers[layer_id].directed and fact.outgoing_count == 0:
                    tags.append('zero_outgoing_member')
                if view.layers[layer_id].directed and fact.incoming_count == 0:
                    tags.append('zero_incoming_member')
            for tag in tags:
                key = split, layer_id, fact.category, tag
                score = ((fact.outgoing_count or 0) + (fact.incoming_count or 0), fact.split_unit)
                if key not in features or score < features[key][0]:
                    features[key] = score, gene
    for _, gene in features.values():
        selected.add(gene)
    facts = [fact_from_gene(view, index, registry, old_view, layer, gene, seed)
             for gene in sorted(selected, key=lambda item: (SPLITS.index(gene_group(item, seed)[1]), gene_group(item, seed)[0]))
             for layer in view.layer_order]
    for layer_id in view.layer_order:
        for split in SPLITS:
            population = [gene for gene in candidates if gene_group(gene, seed)[1] == split]
            histogram = Counter('member' if layer_id in profiles[gene] else 'outside_layer'
                                if profiles[gene] else 'outside_multiplex' for gene in population)
            domains.append({'split': split, 'layer_id': layer_id, 'population': len(population),
                'membership_histogram': dict(sorted(histogram.items())), 'full_layer_members': len(view.layers[layer_id].nodes),
                'full_layer_edges': len(view.layers[layer_id].edges)})
    return facts, {'construction_policy': 'complete_singleton_layer_profiles_v1',
        'candidate_genes': len(candidates), 'source_endpoint_genes': len(view.multiplex_nodes),
        'registry_genes': len(registry.symbols_by_id), 'old_view_endpoint_genes': len(old_view.multiplex_nodes),
        'new_view_only_genes': len(view.multiplex_nodes - old_view.multiplex_nodes),
        'domains': domains, 'profile_cells': cells, 'quota_shortfalls': omissions,
        'feature_witnesses': [{'split': key[0], 'layer_id': key[1], 'category': key[2], 'feature': key[3], 'gene_id': gene}
                              for key, (_, gene) in sorted(features.items())],
        'selected_genes': len(selected), 'selected_facts': len(facts),
        'fact_selection_sha256': digest([asdict(fact) for fact in facts]),
        'split_policy': 'The fixed S2.3 SHA256(seed,gene) bins keep each gene and every layer/context contrast in one split.',
        'all_registered_layers_per_selected_gene': True, 'all_finite_membership_profiles_audited': True,
        'all_possible_absent_identifiers_enumerated': False, 'runtime_gene_cap': None,
        'edge_fact_disjoint': False, 'smoke': smoke,
        'remaining_domains': ['production allocation and further qualified tissues or cell types',
            'authoritative full-layer isolates, future graph versions, and wider identity workflows']}
