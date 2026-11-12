"""Check readable source-panel choices without imposing component-output caps."""

from dataclasses import replace

from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_10_export import (
    _early_exact_witness, _fact_key, _gate_panel, _semantic_panel, _weak_prefix_for_strong,
)
from runtime.world_model.s2_10_sampling import fact_from_genes
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


SEED = 20261028
PPI, BRAIN = 'HumanNetV3:string_ppi', 'TFs:brain'


def source_cases():
    """Use a triangle, a strong witness, an isolate, and an excluded input."""
    genes = [f'ENSG{number:011d}' for number in range(1, 100)]
    a, b, c, isolated, outside, missing = [gene for gene in genes if gene_group(gene, SEED)[1] == 'train'][:6]
    view = GraphView('unit_component_export', 'a' * 64, {
        PPI: LayerGraph(PPI, False, {(a, b): 0, (a, c): .2, (b, c): .3}, frozenset((a, b, c, isolated))),
        BRAIN: LayerGraph(BRAIN, True, {(a, b): .1, (a, c): .2, (b, c): .3,
                                       (c, a): .4, (c, b): .5, (outside, outside): .6},
                          frozenset((a, b, c, isolated, outside))),
    })
    index = SubgraphIndex(view)
    facts = [fact_from_genes(view, index, PPI, values, 'undirected', SEED, split='train')
             for values in ([], [a], [a, b, c], [a, isolated], [outside], [a, outside, missing])]
    facts.extend(fact_from_genes(view, index, BRAIN, values, kind, SEED, split='train')
                 for values in ([], [a], [a, b, c], [a, isolated], [outside])
                 for kind in ('weak', 'strong'))
    return view, index, facts


def test_proper_edge_prefix_can_prove_one_component_without_complete_edge_retrieval():
    view, index, facts = source_cases()
    witnesses = [fact for fact in facts if _early_exact_witness(view, index, fact)]
    assert {fact.connectivity_kind for fact in witnesses} == {'undirected', 'weak', 'strong'}
    assert all(fact.component_count == 1 and fact.edge_count > 1 for fact in witnesses)
    assert any(_weak_prefix_for_strong(view, index, fact) for fact in facts)
    assert all(fact.connectivity_kind == 'strong' for fact in facts
               if _weak_prefix_for_strong(view, index, fact))


def test_semantic_panel_keeps_each_kind_and_coarse_outcome_and_ignores_fact_order():
    view, index, facts = source_cases()
    panel = _semantic_panel(view, index, facts)
    selected = [fact for fact in facts if _fact_key(fact) in panel]
    coarse = lambda fact: 'zero' if not fact.component_count else 'one' if fact.component_count == 1 else 'multiple'
    assert {(fact.connectivity_kind, coarse(fact)) for fact in selected} == {
        (fact.connectivity_kind, coarse(fact)) for fact in facts}
    assert {tuple(bool(count) for count in fact.category_counts) for fact in selected} == {
        tuple(bool(count) for count in fact.category_counts) for fact in facts}
    assert _semantic_panel(view, index, list(reversed(facts))) == panel
    assert len(panel) < len(facts)


def test_panel_size_policy_does_not_remove_large_source_facts():
    view, index, facts = source_cases()
    large = replace(facts[2], gene_ids=tuple(f'ENSG{number:011d}' for number in range(300, 309)),
                    split_unit='large-source-fact')
    augmented = [*facts, large]
    assert _fact_key(large) not in _semantic_panel(view, index, augmented)
    assert augmented[-1] is large and len(large.gene_ids) == 9


def test_gate_panel_retains_empty_passing_and_failing_membership_cases():
    _, _, facts = source_cases()
    selected = _gate_panel(facts)
    outcomes = lambda fact: 'empty' if not fact.gene_ids else 'fails' if sum(fact.category_counts[1:]) else 'passes'
    assert {(fact.connectivity_kind, outcomes(fact)) for fact in selected} == {
        (fact.connectivity_kind, outcomes(fact)) for fact in facts}
    assert _gate_panel(list(reversed(facts))) == selected
