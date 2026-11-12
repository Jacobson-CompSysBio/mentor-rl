"""Keep numbered-case audit bindings faithful to their actual evidence contrasts."""

from copy import deepcopy
import importlib
import json

import pytest

from runtime.world_model.s2_audit_cases_3_6 import CASE_BINDINGS, _RULES, matches
from runtime.world_model.s2_production_cases import load_catalog


@pytest.fixture(scope='module')
def fixtures():
    """Reuse existing explicit test graphs, never corpus source replacements."""
    result = {}
    for number in range(3, 7):
        module = importlib.import_module(f'test_world_model_s2_{number}_corpus')
        cls = getattr(module, f'WorldModelS2{number}CorpusTests')
        cls.setUpClass()
        result[number] = module, cls(methodName='runTest')
    return result


def test_every_reviewed_case_is_bound_and_keeps_known_task_framing():
    catalog = [case for case in load_catalog() if case.family in {'S2.3', 'S2.4', 'S2.5', 'S2.6'}]
    assert len(CASE_BINDINGS) == 100
    assert set(CASE_BINDINGS) == {case.case_id for case in catalog}
    for case in catalog:
        if case.framing_hint:
            assert _RULES[case.case_id][2] == case.framing_hint
        binding = CASE_BINDINGS[case.case_id]
        assert binding['requirements'] and binding['variants']
        assert binding['generator'].startswith('runtime.world_model.')


def test_specific_tie_case_requires_source_truth_while_generic_bound_can_overlap(fixtures):
    _, fixture = fixtures[3]
    tied = fixture._row(mode='top_k', top_k=1)
    assert not matches('S2.3.18', tied)
    assert matches('S2.3.3', tied)
    tied['_audit_source'] = {'cutoff_tie': True}
    assert matches('S2.3.18', tied)
    assert matches('S2.3.3', tied)
    untied = deepcopy(tied)
    untied['_audit_source']['cutoff_tie'] = False
    assert matches('S2.3.3', untied)
    assert not matches('S2.3.18', untied)


def test_present_zero_outdegree_is_not_layer_absence(fixtures):
    module, fixture = fixtures[3]
    sink = fixture._row(module.H, module.BRAIN, 'out')
    fixture._validate(sink)
    assert matches('S2.3.10', sink)
    assert not matches('S2.3.5', sink)
    absent = fixture._row(module.F, module.BRAIN, 'out')
    assert not matches('S2.3.10', absent)


def test_truncated_context_must_really_be_incomplete(fixtures):
    _, fixture = fixtures[3]
    partial = fixture._row(variant='bounded_context', framing='step')
    fixture._validate(partial)
    assert matches('S2.3.7', partial)
    changed = deepcopy(partial)
    context = json.loads(changed['messages'][1]['content'])
    context['observations'][0]['result']['collection']['complete'] = True
    changed['messages'][1]['content'] = json.dumps(context)
    assert not matches('S2.3.7', changed)


def test_illustration_scale_high_degree_is_not_inferred_from_any_bound(fixtures):
    module, fixture = fixtures[3]
    bounded = fixture._row(layer=module.BRAIN, direction='out', mode='top_k', top_k=1)
    bounded['_audit_source'] = {'native_neighbor_count': 730}
    assert not matches('S2.3.20', bounded)
    bounded['_audit_source']['native_neighbor_count'] = 731
    assert matches('S2.3.20', bounded)


def test_native_failures_require_reviewed_atomic_coverage_and_retained_context(fixtures):
    module, fixture = fixtures[4]
    layer = fixture._row(variant='native_terminal', framing='step')
    fixture._validate(layer)
    assert not matches('S2.4.14', layer)
    assert not matches('S2.4.20', layer)
    assert 'implementation_gap' in CASE_BINDINGS['S2.4.14']
    reviewed = deepcopy(layer)
    reviewed['messages'][-1]['content']['result']['coverage'] = {
        'operation': 'neighbor_collections',
        'requested_inputs': [{'layer_id': module.PPI, 'direction': 'undirected'},
                             {'layer_id': module.COEX, 'direction': 'undirected'}],
        'checked_inputs': [{'layer_id': module.PPI, 'direction': 'undirected'}],
        'unchecked_inputs': [{'layer_id': module.COEX, 'direction': 'undirected'}]}
    assert matches('S2.4.14', reviewed)
    assert not matches('S2.4.20', reviewed)
    direction = fixture._row(scope=module._scope(module.BRAIN, module.CD4),
                             variant='native_terminal', framing='step')
    fixture._validate(direction)
    assert not matches('S2.4.20', direction)
    assert not matches('S2.4.14', direction)


def test_complete_scope_normalization_and_mixed_support_use_actual_records(fixtures):
    module, fixture = fixtures[4]
    normalized = fixture._row(variant='scope_order')
    fixture._validate(normalized)
    assert matches('S2.4.1', normalized)
    mixed = fixture._row(scope=module._scope(module.PPI, module.BRAIN))
    fixture._validate(mixed)
    assert matches('S2.4.9', mixed)
    assert not matches('S2.4.25', mixed)


def test_membership_failure_cannot_substitute_explicit_subset_for_reviewed_family_context(fixtures):
    _, fixture = fixtures[5]
    failed = fixture._row(variant='partial_terminal', framing='step')
    fixture._validate(failed)
    assert not matches('S2.5.14', failed)
    assert 'implementation_gap' in CASE_BINDINGS['S2.5.14']
    family = fixture._row(variant='family_context', framing='step')
    fixture._validate(family)
    assert matches('S2.5.7', family)
    changed = deepcopy(family)
    context = json.loads(changed['messages'][1]['content'])
    context['observations'][0]['result']['layer_count'] = 0
    changed['messages'][1]['content'] = json.dumps(context)
    assert not matches('S2.5.7', changed)


def test_strongest_ranking_diagnostic_must_preserve_requested_bound(fixtures):
    module, fixture = fixtures[4]
    fact = module._fact(fixture.runtime, module.A, module._scope(module.PPI, module.COEX),
                        'incident', module.SEED)
    row = module.build_diagnostic(fixture.view, fact, 'direct', 0, fixture.bank,
                                  fixture.contract, ranking=True)
    fixture._validate(row)
    assert not matches('S2.4.22', row)
    assert 'implementation_gap' in CASE_BINDINGS['S2.4.22']
    preserved = deepcopy(row)
    preserved['messages'][-1]['content']['result']['request'].update(mode='top_k', top_k=2)
    assert matches('S2.4.22', preserved)


def test_all_layer_global_negatives_are_separate_task_frames(fixtures):
    module, fixture = fixtures[5]
    direct = fixture._row(module.MISSING)
    workflow = fixture._row(module.MISSING, framing='step')
    assert matches('S2.5.6', direct)
    assert not matches('S2.5.24', direct)
    assert matches('S2.5.24', workflow)
    assert not matches('S2.5.6', workflow)


def test_known_total_and_unknown_total_bounded_cases_do_not_merge(fixtures):
    _, fixture = fixtures[5]
    known = fixture._row(mode='top_k', top_k=1)
    unknown = fixture._row(mode='top_k', top_k=1, variant='unknown_total')
    fixture._validate(known)
    fixture._validate(unknown)
    assert matches('S2.5.11', known)
    assert matches('S2.5.17', unknown)
    relabeled = deepcopy(unknown)
    relabeled['metadata']['variant'] = 'lookup'
    assert not matches('S2.5.11', relabeled)


def test_three_category_global_reuse_is_not_all_positive_context(fixtures):
    _, fixture = fixtures[6]
    mixed = fixture._row(variant='global_context', framing='step')
    fixture._validate(mixed)
    assert matches('S2.6.22', mixed)
    assert not matches('S2.6.8', mixed)
    positive = fixture._row([fixture.e, fixture.a, fixture.isolated],
                             variant='global_context', framing='step')
    fixture._validate(positive)
    assert matches('S2.6.8', positive)
    assert not matches('S2.6.22', positive)


def test_cold_failure_and_preserved_partial_failure_are_separate(fixtures):
    _, fixture = fixtures[6]
    cold = fixture._row(variant='terminal')
    fixture._validate(cold)
    assert matches('S2.6.23', cold)
    assert not matches('S2.6.12', cold)
    partial = fixture._row(variant='partial_terminal', framing='step')
    fixture._validate(partial)
    assert matches('S2.6.12', partial)
    assert not matches('S2.6.23', partial)


def test_empty_input_is_not_a_missing_list(fixtures):
    _, fixture = fixtures[6]
    empty = fixture._row([], split='train')
    fixture._validate(empty)
    assert matches('S2.6.19', empty)
    assert not matches('S2.6.24', empty)
