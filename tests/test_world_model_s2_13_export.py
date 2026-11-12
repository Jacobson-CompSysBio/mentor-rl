"""Check compact source coverage, separated task branches, and complete import pins."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_13_export import (
    _claim_questions, _fact_key, _feature_keys, _implementation_paths, _semantic_panel, build_corpus,
)
from runtime.world_model.s2_13_evidence import CLAIM_VOCABULARY, FAMILIES
from runtime.world_model.s2_13_sampling import fact_from_pair
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


A, B, C, D, MISSING, EXPANDED, NULL = [f'ENSG{number:011d}' for number in range(1, 8)]
PPI, COEX, BRAIN, CD4 = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells')
BULK, SC = 'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte'


def fixture():
    """Use exact source records with route, orientation, registry, and PEN controls."""
    edges = {PPI: {(A, B): .2, (B, C): .3, (B, NULL): .5}, COEX: {(C, D): .4},
        BRAIN: {(A, B): .7}, CD4: {(A, B): .8, (B, A): .9},
        BULK: {(A, D): .6, (A, EXPANDED): .2}, SC: {(A, D): .4, (C, D): .9}}
    layers = {name: LayerGraph(name, name in (BRAIN, CD4), records,
        frozenset(gene for pair in records for gene in pair)) for name, records in edges.items()}
    view = GraphView('unit_layer_specificity', 'a' * 64, layers)
    old = GraphView('unit_original', 'b' * 64, {key: value for key, value in layers.items() if key not in (BULK, SC)})
    index = SubgraphIndex(view)
    registry = RegistryContext('116', 'ensembl_homo_sapiens_116_unit',
        {gene: None if gene == NULL else gene for gene in (A, B, C, D, MISSING, NULL)})
    pairs = ((A, B), (A, C), (A, EXPANDED), (A, MISSING), (A, D), (B, A), (C, D), (NULL, B))
    facts = [replace(fact_from_pair(view, index, registry, old, layer, pair), split=split)
        for split in ('train', 'validation', 'test') for layer in layers for pair in pairs]
    return view, index, registry, old, facts


def test_fact_key_keeps_layer_and_native_orientation_separate():
    _, _, _, _, facts = fixture()
    original = facts[0]
    assert _fact_key(original) != _fact_key(replace(original, layer_id=BRAIN))
    assert _fact_key(original) != _fact_key(replace(original, gene_ids=original.gene_ids[::-1]))
    assert len(_fact_key(original)) == 3


def test_panel_preserves_every_attainable_source_feature():
    view, index, registry, old, facts = fixture()
    selected = _semantic_panel(view, index, facts, registry, old)
    expected = set().union(*(_feature_keys(view, index, fact, registry, old) for fact in facts))
    actual = set().union(*(_feature_keys(view, index, fact, registry, old)
                          for fact in facts if _fact_key(fact) in selected))
    assert actual == expected
    assert len(selected) < len(facts)
    assert {key[1] for key in actual} >= {'layer_category', 'family_category', 'native_orientation',
        'registry_state', 'pen_recorded_contexts', 'expanded_view_endpoint', 'two_edge_route_without_direct_record'}
    assert {key[-1] for key in actual if key[1] == 'layer_category'} == {
        'recorded', 'no_edge', 'outside_layer', 'outside_multiplex'}


def test_panel_selection_is_independent_of_candidate_iteration_order():
    view, index, registry, old, facts = fixture()
    assert _semantic_panel(view, index, facts, registry, old) == _semantic_panel(
        view, index, list(reversed(facts)), registry, old)


def test_route_feature_requires_a_missing_direct_record():
    view, index, registry, old, facts = fixture()
    route = next(fact for fact in facts if fact.layer_id == PPI and fact.gene_ids == (A, C))
    positive = next(fact for fact in facts if fact.layer_id == PPI and fact.gene_ids == (A, B))
    assert route.category == 'no_edge' and positive.category == 'recorded'
    assert any(key[1] == 'two_edge_route_without_direct_record'
               for key in _feature_keys(view, index, route, registry, old))
    assert not any(key[1] == 'two_edge_route_without_direct_record'
                   for key in _feature_keys(view, index, positive, registry, old))


def test_pen_local_records_and_expanded_nodes_are_distinct_features():
    view, index, registry, old, facts = fixture()
    both = next(fact for fact in facts if fact.layer_id == BULK and fact.gene_ids == (A, D))
    bulk_only = next(fact for fact in facts if fact.layer_id == BULK and fact.gene_ids == (A, EXPANDED))
    both_keys, bulk_keys = (_feature_keys(view, index, fact, registry, old) for fact in (both, bulk_only))
    assert (both.split, 'pen_recorded_contexts', (BULK, SC), 'recorded') in both_keys
    assert (bulk_only.split, 'pen_recorded_contexts', (BULK,), 'recorded') in bulk_keys
    assert (bulk_only.split, 'expanded_view_endpoint', 'recorded') in bulk_keys


def test_native_reverse_role_controls_do_not_merge_orientations():
    view, index, registry, old, facts = fixture()
    one_way = next(fact for fact in facts if fact.layer_id == BRAIN and fact.gene_ids == (A, B))
    reciprocal = next(fact for fact in facts if fact.layer_id == CD4 and fact.gene_ids == (A, B))
    assert (one_way.split, 'native_orientation', True, 'recorded', False) in _feature_keys(view, index, one_way)
    assert (reciprocal.split, 'native_orientation', True, 'recorded', True) in _feature_keys(view, index, reciprocal)


def test_task_matrix_keeps_biological_support_truth_and_graph_truth_separate():
    view, _, _, _, facts = fixture()
    for category in ('recorded', 'no_edge', 'outside_layer', 'outside_multiplex'):
        fact = next(fact for fact in facts if fact.category == category)
        tasks = _claim_questions(fact, view)
        for kind in ('edge_support', 'claim_support'):
            assert {task['claim'] for _, task in tasks if task['question_kind'] == kind} == set(CLAIM_VOCABULARY)
        truths = [task for _, task in tasks if task['question_kind'] == 'biological_truth']
        assert {task['claim'] for task in truths} == {'causal_effect', 'holds_in_other_contexts'}
        assert all('statement' not in task for _, task in tasks)
        graphs = [task for _, task in tasks if task['question_kind'] == 'graph_truth']
        assert bool(graphs) == (category != 'outside_multiplex')
        assert all(task['graph_claim'] == 'no_edge_recorded_in_scope' and task['graph_truth_scope'] == 'scope' for task in graphs)
        assert ('lookup', {'question_kind': 'claim_card'}) in tasks
        assert len({(variant, str(task)) for variant, task in tasks}) == len(tasks)


def test_supplied_mismatch_contexts_have_evidence_only_claim_checks():
    view, _, _, _, facts = fixture()
    tasks = _claim_questions(facts[0], view)
    selected = {variant for variant, task in tasks if task['question_kind'] == 'claim_card'}
    assert {'reverse_context', 'family_context', 'wrong_layer_context', 'same_family_context',
        'wrong_pair_context', 'stale', 'old_view_context', 'pen_contrast_context',
        'two_context_context', 'reciprocal_context', 'route_context'} <= selected


def test_all_local_imports_have_manifest_pins():
    root = Path(__file__).resolve().parents[1]
    paths, pending, seen = set(_implementation_paths()), ['s2_13_export', 's2_13_corpus', 's2_13_sampling', 's2_13_evidence', 's2_13_prompts'], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = root / f'runtime/world_model/{name}.py'
        assert path.is_file() and str(path.relative_to(root)) in paths
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                if (root / f'runtime/world_model/{node.module}.py').is_file():
                    pending.append(node.module)
    assert 'scripts/build_world_model_s2_13.py' in paths
    assert 'runtime/world_model_training.py' in paths
    assert not any('s2_13_sources.py' in path for path in paths)


@pytest.mark.parametrize('counts,smoke', [(None, False), ({'train': 1}, True), ({'train': 1}, False),
    ({'train': 1, 'validation': 0, 'test': 1}, False), ({'train': True, 'validation': 1, 'test': 1}, False)])
def test_budget_errors_do_not_create_artifacts(tmp_path, counts, smoke):
    with pytest.raises(ValueError):
        build_corpus(tmp_path, counts=counts, smoke=smoke)
    assert list(tmp_path.iterdir()) == []
