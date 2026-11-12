"""Verify exact quota allocation, trace identity, and pinned source conditions."""

from copy import deepcopy
from itertools import product
import json

import pytest

from runtime.world_model.s2_audit_allocation import allocate_groups
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_audit_source import SourceFeatures, nearest_below_threshold
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex
from scripts.audit_world_model_s2_case_quotas import catalog_inputs, load_bindings, register_row_identity


def test_all_catalog_cases_have_one_explicit_resolvable_binding():
    cases, _, _ = catalog_inputs()
    bindings, _, _ = load_bindings(cases)
    assert len(cases) == len(bindings) == 399
    assert all(binding['requirements'] and binding['variants'] for binding in bindings.values())


def test_all_63_repaired_early_witnesses_match_the_declared_raw_variants():
    from pathlib import Path
    from runtime.world_model.s2_audit_cases_1_2 import CASE_BINDINGS, matches
    root = Path(__file__).resolve().parents[1]
    observed = set()
    for family in (1, 2):
        path = root / f'diagnostics/s2_generator_checks/final/witnesses/s2_{family}/train.jsonl'
        for line in path.read_text().splitlines():
            row = json.loads(line)
            case_id = row['metadata'].get('catalog_case_id')
            if case_id is None:
                continue
            assert row['metadata']['variant'] in CASE_BINDINGS[case_id]['variants']
            assert matches(case_id, row)
            observed.add(case_id)
    assert len(observed) == 63


def test_exact_flow_preserves_a_scarce_case_that_greedy_assignment_would_strand():
    assigned = allocate_groups({('A', 'B'): ['shared'], ('A',): ['only_a']}, {'A': 1, 'B': 1})
    assert assigned == {'A': ['only_a'], 'B': ['shared']}


def test_grouped_quota_allocation_never_counts_one_row_twice():
    groups = {('A', 'B'): ['r1', 'r2', 'r3'], ('A',): ['r4'], ('B', 'C'): ['r5']}
    assigned = allocate_groups(groups, {'A': 2, 'B': 2, 'C': 1})
    flattened = [row for rows in assigned.values() for row in rows]
    assert len(flattened) == len(set(flattened)) == 5
    assert {case: len(rows) for case, rows in assigned.items()} == {'A': 2, 'B': 2, 'C': 1}


def test_small_allocations_match_exhaustive_feasibility():
    neighborhoods = [('A',), ('B',), ('A', 'B')]
    for groups in product(neighborhoods, repeat=3):
        by_group = {}
        for i, group in enumerate(groups):
            by_group.setdefault(group, []).append(str(i))
        for quotas in ({'A': 1, 'B': 1}, {'A': 2, 'B': 1}, {'A': 0, 'B': 2}):
            optimum = 0
            for choices in product((None, 'A', 'B'), repeat=3):
                if any(choice is not None and choice not in group for choice, group in zip(choices, groups)):
                    continue
                if all(choices.count(case) <= count for case, count in quotas.items()):
                    optimum = max(optimum, sum(choice is not None for choice in choices))
            assigned = allocate_groups(by_group, quotas)
            assert sum(map(len, assigned.values())) == optimum


@pytest.mark.parametrize('groups,quotas', [({('A',): ['x'], ('A', 'B'): ['x']}, {'A': 1, 'B': 1}),
    ({('unknown',): ['x']}, {'A': 1}), ({('A', 'A'): ['x']}, {'A': 1}),
    ({('A',): ['x']}, {'A': -1})])
def test_invalid_allocation_inputs_fail_closed(groups, quotas):
    with pytest.raises(ValueError):
        allocate_groups(groups, quotas)


def trace(identity='ev_one'):
    """Supply visible evidence with an important acquisition-reuse relationship."""
    observation = {'operation': 'get_edge', 'provenance': {'graph': [{'evidence_id': identity}]}}
    return {'row_id': 'row', 'metadata': {'split': 'train'}, 'tools': [], 'messages': [
        {'role': 'harness', 'content': json.dumps({'observations': [observation]})},
        {'role': 'assistant', 'thinking': f'Keep {identity}.', 'content': {'evidence_id': identity, 'value': True}}]}


def test_cosmetic_ids_and_private_metadata_do_not_inflate_distinct_rows():
    first, renamed = trace(), trace('other_acquisition')
    renamed.update(row_id='another', metadata={'split': 'validation', 'variant': 'different_private_label'})
    assert model_trace_sha256(first) == model_trace_sha256(renamed)


def test_renaming_preserves_reuse_and_actual_model_content():
    original = trace()
    broken_reuse = deepcopy(original)
    broken_reuse['messages'][-1]['content']['evidence_id'] = 'another_acquisition'
    assert model_trace_sha256(original) != model_trace_sha256(broken_reuse)
    changed_answer = deepcopy(original)
    changed_answer['messages'][-1]['content']['value'] = False
    assert model_trace_sha256(original) != model_trace_sha256(changed_answer)


def test_duplicate_row_id_cannot_hide_a_split_or_private_identity_fault():
    original = trace()
    seen = {}
    fingerprint = model_trace_sha256(original)
    assert register_row_identity(seen, original, fingerprint, 'train')
    assert not register_row_identity(seen, original, fingerprint, 'train')
    with pytest.raises(ValueError):
        register_row_identity(seen, original, fingerprint, 'test')
    changed = deepcopy(original)
    changed['metadata']['variant'] = 'a different case'
    with pytest.raises(ValueError):
        register_row_identity(seen, changed, fingerprint, 'train')


def test_source_features_detect_a_cutoff_tie_that_is_hidden_beyond_top_one():
    a, b, c = (f'ENSG{value:011d}' for value in (1, 2, 3))
    layer = 'TFs:brain'
    view = GraphView('audit_fixture', 'a' * 64, {layer: LayerGraph(layer, True,
        {(a, b): .9, (a, c): .9, (a, a): .1}, frozenset((a, b, c)))})
    features = SourceFeatures(view, SubgraphIndex(view)).for_row({'metadata': {
        'question_family': 'S2.3', 'layer_id': layer, 'gene_id': a, 'direction': 'out', 'top_k': 1}})
    assert features['cutoff_tie'] is True
    assert features['native_self_loop'] is True
    assert features['native_neighbor_count'] == 3


def test_scope_only_degree_metadata_still_receives_a_native_loop_feature():
    a, b = (f'ENSG{value:011d}' for value in (1, 2))
    layer = 'TFs:brain'
    view = GraphView('audit_scope_fixture', 'a' * 64, {layer: LayerGraph(layer, True,
        {(a, a): .5, (a, b): 1}, frozenset((a, b)))})
    features = SourceFeatures(view, SubgraphIndex(view)).for_row({'metadata': {
        'question_family': 'S2.8', 'gene_id': a, 'direction': 'out', 'split': 'train',
        'layer_scope': {'kind': 'layers', 'layer_ids': [layer], 'layer_family': None}}})
    assert features['native_self_loop'] is True


def test_nearest_below_threshold_uses_full_population_and_fixed_split_attainability():
    scope = {'kind': 'all', 'layer_ids': [], 'layer_family': None}
    selection = {'domains': [{'layer_scope': scope, 'direction': 'incident',
        'degree_histogram': [{'degree': 0, 'node_count': 90}, {'degree': 1, 'node_count': 5},
                             {'degree': 2, 'node_count': 5}]}],
        'cells': [{'layer_scope': scope, 'direction': 'incident', 'split': 'train', 'available_degree_values': [0, 1, 2]},
                  {'layer_scope': scope, 'direction': 'incident', 'split': 'test', 'available_degree_values': [0, 2]}]}
    lookup = nearest_below_threshold(selection)
    key = json.dumps(scope, sort_keys=True), 'incident'
    assert lookup[(*key, 'train')] == 1
    assert lookup[(*key, 'test')] == 0
