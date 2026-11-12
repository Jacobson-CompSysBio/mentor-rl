"""Check singleton source features, literal truth tasks, and import-closure pins."""

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_12_export import (
    _claim_questions, _fact_key, _feature_keys, _implementation_paths, _native_cost, _semantic_panel,
)
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


SEED = 20261028
PPI, BRAIN, COEX, BULK, SC = ('HumanNetV3:string_ppi', 'TFs:brain', 'HumanNetV3:coexpression',
    'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte')


@dataclass(frozen=True)
class Fact:
    """Supply only the public singleton fields that panel allocation requires."""
    gene_id: str
    layer_id: str
    split: str
    category: str
    registry_state: str | None = None


def source_cases():
    """Use each split with present, outside-layer, absent, and expanded-view nodes."""
    genes = [f'ENSG{number:011d}' for number in range(1, 400)]
    groups = {split: [gene for gene in genes if gene_group(gene, SEED)[1] == split][:5]
              for split in ('train', 'validation', 'test')}
    common, sink, outside, missing, expanded = ([groups[split][rank] for split in groups] for rank in range(5))
    layer = lambda name, directed, edges: LayerGraph(name, directed, edges,
        frozenset(gene for pair in edges for gene in pair))
    ppi = layer(PPI, False, {(a, a): .1 for a in common})
    brain = layer(BRAIN, True, {(a, b): .2 for a, b in zip(common, sink)})
    coex = layer(COEX, False, {(a, b): .3 for a, b in zip(common, outside)})
    bulk = layer(BULK, False, {(a, b): .4 for a, b in zip(common, expanded)})
    sc = layer(SC, False, {(a, b): .5 for a, b in zip(outside, expanded)})
    view = GraphView('unit_membership_export', 'a' * 64,
        {item.layer_id: item for item in (ppi, brain, coex, bulk, sc)})
    old = GraphView('unit_previous_export', 'b' * 64,
        {item.layer_id: item for item in (ppi, brain, coex)})
    facts = [Fact(gene, layer_id, split,
        'member' if gene in view.layers[layer_id].nodes else
        'outside_layer' if gene in view.multiplex_nodes else 'outside_multiplex')
        for split, ids in groups.items() for gene in ids for layer_id in view.layer_order]
    registry = SimpleNamespace(symbols_by_id={gene: None if gene in missing else gene
        for gene in genes if gene not in sink})
    return view, old, SubgraphIndex(view), facts, registry


def test_fact_identity_separates_layers_and_retains_singleton_split():
    _, _, _, facts, _ = source_cases()
    assert _fact_key(facts[0]) != _fact_key(facts[1])
    assert len(_fact_key(facts[0])) == 3


def test_semantic_panel_covers_all_attainable_source_features_and_categories():
    view, old, index, facts, registry = source_cases()
    selected = _semantic_panel(view, index, facts, registry, old)
    all_keys = set().union(*(_feature_keys(view, index, fact, registry, old) for fact in facts))
    kept_keys = set().union(*(_feature_keys(view, index, fact, registry, old)
                             for fact in facts if _fact_key(fact) in selected))
    assert kept_keys == all_keys
    assert {key[-1] for key in kept_keys if key[1] == 'layer_category'} == {
        'member', 'outside_layer', 'outside_multiplex'}
    assert {key[1] for key in kept_keys} >= {'graph_only_member', 'null_registry_symbol',
        'expanded_view_member', 'native', 'top_one_positive', 'top_one_omits_target'}
    assert any(key[1:] == ('native', True, 'empty') for key in kept_keys)


def test_ambiguous_registry_state_remains_a_distinct_panel_feature():
    view, old, index, facts, registry = source_cases()
    fact = replace(facts[0], registry_state='ambiguous_symbol')
    keys = _feature_keys(view, index, fact, registry, old)
    assert (fact.split, 'registry_state', 'ambiguous_symbol', fact.category) in keys


def test_empty_directed_incidence_has_zero_cost_without_changing_membership():
    view, _, index, facts, _ = source_cases()
    candidates = [fact for fact in facts if fact.layer_id == BRAIN and fact.category == 'member'
                  and not index.native_index.neighbors(BRAIN, fact.gene_id, 'out')]
    assert candidates and all(_native_cost(view, index, fact) == 0 for fact in candidates)
    assert all(fact.gene_id in view.layers[BRAIN].nodes for fact in candidates)


def test_claim_tasks_cover_vocabulary_exact_truth_scopes_and_explicit_gates():
    from runtime.world_model.s2_12_corpus import truth_statement
    view, _, _, facts, _ = source_cases()
    for fact in (next(fact for fact in facts if fact.category == category)
                 for category in ('member', 'outside_layer', 'outside_multiplex')):
        tasks = _claim_questions(fact, view, CLAIM_VOCABULARY)
        for kind in ('membership_support', 'claim_support'):
            assert {task['claim'] for _, task in tasks if task['question_kind'] == kind} == set(CLAIM_VOCABULARY)
        truths = [task for _, task in tasks if task['question_kind'] in ('membership_truth', 'truth')]
        expected_scopes = ({'multiplex', 'future_versions', 'expression'} if fact.category == 'outside_multiplex'
            else {'category', 'all_layers', 'layers', 'family', 'future_versions', 'expression'})
        assert {task['truth_scope'] for task in truths} >= expected_scopes
        for task in truths:
            assert task['statement'] == truth_statement(view, fact, task['claim'], task['truth_scope'],
                truth_layer_ids=task['truth_layer_ids'], truth_family=task['truth_family'])
        assert {task['require_membership'] for _, task in tasks if 'require_membership' in task} == {'layer', 'multiplex'}
        if fact.category == 'outside_multiplex':
            assert all(task['claim'] == 'gene_not_node_in_multiplex'
                       for task in truths if task['truth_scope'] == 'multiplex')
            assert not any(task['truth_scope'] in ('category', 'all_layers', 'family', 'layers') for task in truths)


def test_all_imported_local_build_modules_have_manifest_pins():
    root = Path(__file__).resolve().parents[1]
    paths = set(_implementation_paths())
    pending = ['s2_12_export', 's2_12_corpus', 's2_12_sampling', 's2_12_evidence', 's2_12_sources']
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = root / f'runtime/world_model/{name}.py'
        assert path.is_file(), path
        assert str(path.relative_to(root)) in paths
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                candidate = root / f'runtime/world_model/{node.module}.py'
                if candidate.is_file():
                    pending.append(node.module)
    assert 'scripts/build_world_model_s2_12.py' in paths
    assert 'runtime/world_model_training.py' in paths


@pytest.mark.parametrize('counts,smoke', [(None, False), ({'train': 1}, True),
    ({'train': 1}, False), ({'train': 1, 'validation': 0, 'test': 1}, False),
    ({'train': True, 'validation': 1, 'test': 1}, False)])
def test_export_refuses_an_implicit_or_invalid_budget(tmp_path, counts, smoke):
    from runtime.world_model.s2_12_export import build_corpus
    with pytest.raises(ValueError):
        build_corpus(tmp_path, counts=counts, smoke=smoke)
    assert list(tmp_path.iterdir()) == []
