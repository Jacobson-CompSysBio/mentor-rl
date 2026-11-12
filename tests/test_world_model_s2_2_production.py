"""Qualify source-backed S2.2 case bindings and adversarial evidence edits."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from runtime.world_model.s2_2_production import (
    CASE_BINDINGS, DIAGNOSTICS, build_case_rows, load_inputs, validate_case_row,
)


@pytest.fixture(scope='module')
def qualified():
    """Use pinned source files, preserving their real pair splits and weights."""
    view, contract, bank, registry = load_inputs()
    rows = {split: build_case_rows(view, contract, bank=bank, registry=registry, split=split)
            for split in ('train', 'validation', 'test')}
    return view, contract, bank, registry, rows


def _check(qualified, row):
    view, contract, bank, registry, _ = qualified
    validate_case_row(row, view, contract=contract, bank=bank, registry=registry)


def _case(qualified, case_id, split='train'):
    return next(row for row in qualified[-1][split] if row['metadata']['catalog_case_id'] == case_id)


def _aux(qualified, behavior):
    return next(row for row in qualified[-1]['train'] if row['metadata']['production_behavior'] == behavior)


def test_every_numbered_case_has_a_real_witness_in_each_unchanged_split(qualified):
    """Coverage is per reviewed case, not a count of loosely related variants."""
    all_ids, subjects = set(), {}
    for split, rows in qualified[-1].items():
        counts = Counter(row['metadata']['catalog_case_id'] for row in rows
                         if row['metadata']['catalog_case_id'] is not None)
        assert counts == dict.fromkeys(CASE_BINDINGS, 1)
        for row in rows:
            _check(qualified, row)
            assert row['row_id'] not in all_ids
            all_ids.add(row['row_id'])
            assert row['metadata']['split'] == split
        subjects[split] = {row['metadata']['split_unit'] for row in rows}
    assert not subjects['train'] & subjects['validation']
    assert not subjects['train'] & subjects['test']
    assert not subjects['validation'] & subjects['test']


def test_bounded_and_partial_bindings_exercise_required_remaining_work(qualified):
    bounded = _case(qualified, 'S2.2.9')['messages'][-1]['content']['result']
    assert bounded['collection']['returned_count'] == 1
    assert bounded['collection']['total_count'] is None
    assert bounded['supporting_layer_count'] is None
    assert bounded['coverage']['unchecked_layer_ids']
    for case_id in ('S2.2.6', 'S2.2.10', 'S2.2.15', 'S2.2.16', 'S2.2.17'):
        row = _case(qualified, case_id)
        saved = json.loads(row['messages'][1]['content'])['observations']
        assert len(saved) == 1
        calls = [m['tool_call'] for m in row['messages'] if 'tool_call' in m]
        assert calls
        saved_layers = set(saved[0]['result']['coverage']['checked_layer_ids'])
        assert not saved_layers & set(calls[0]['arguments']['layer_scope']['layer_ids'])
        if case_id in ('S2.2.16', 'S2.2.17'):
            assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_partial_and_failure_cases_retain_an_earlier_positive_layer(qualified, split):
    """A negative prefix cannot replace the reviewed retained positive evidence."""
    for case_id in ('S2.2.6', 'S2.2.15', 'S2.2.16', 'S2.2.17'):
        row = _case(qualified, case_id, split)
        saved = json.loads(row['messages'][1]['content'])['observations'][0]
        assert saved['result']['supporting_layer_count'] == 1
        assert saved['result']['layer_results'][0]['edge_exists'] is True
        if case_id in ('S2.2.15', 'S2.2.17'):
            answer = row['messages'][-1]['content']
            assert answer['result']['partial_results'][0] == saved
            assert saved['provenance']['graph'][0] in answer['provenance']['graph']


def test_one_native_witness_preserves_unknown_coverage_and_reciprocal_layer_count(qualified):
    for case_id, directed in (('S2.2.8', False), ('S2.2.22', True)):
        row = _case(qualified, case_id)
        saved = json.loads(row['messages'][1]['content'])['observations']
        assert len(saved) == 1 and saved[0]['operation'] == 'get_edge'
        assert saved[0]['result']['directed'] is directed
        assert not any('tool_call' in message for message in row['messages'])
        result = row['messages'][-1]['content']['result']
        assert result['edge_exists_any_layer'] is True
        assert result['supporting_layer_count'] is None
        assert result['coverage']['unchecked_layer_ids']
    reciprocal = _case(qualified, 'S2.2.14')['messages'][-1]['content']['result']
    assert reciprocal['supporting_layer_count'] == 1
    assert any(len(layer['edges']) == 2 for layer in reciprocal['layer_results'] if layer['edge_exists'])


@pytest.mark.parametrize('behavior,count', [('fresh_identity', 2), ('partial_identity', 1)])
def test_composed_identity_calls_precede_canonical_graph_calls_and_keep_all_acquisitions(qualified, behavior, count):
    row = _aux(qualified, behavior)
    calls = [message['tool_call'] for message in row['messages'] if 'tool_call' in message]
    assert [call['name'] for call in calls] == ['lookup_human_gene_symbol'] * count + ['get_edge_layers']
    assert calls[-1]['arguments']['gene_ids'] == row['metadata']['gene_ids']
    assert len(row['messages'][-1]['content']['provenance']['registry']) == 2
    assert row['metadata']['catalog_case_id'] is None


def test_a_canonical_graph_call_cannot_run_before_a_missing_symbol_is_resolved(qualified):
    row = deepcopy(_aux(qualified, 'fresh_identity'))
    row['messages'][3:5], row['messages'][7:9] = row['messages'][7:9], row['messages'][3:5]
    with pytest.raises(ValueError, match='preceded required identity'):
        _check(qualified, row)


def test_forged_registry_response_and_dropped_acquisition_are_rejected(qualified):
    row = deepcopy(_aux(qualified, 'fresh_identity'))
    row['messages'][4]['content']['result']['gene_id'] = row['metadata']['gene_ids'][1]
    with pytest.raises(ValueError):
        _check(qualified, row)
    row = deepcopy(_aux(qualified, 'partial_identity'))
    row['messages'][-1]['content']['provenance']['registry'].pop()
    with pytest.raises(ValueError, match='provenance'):
        _check(qualified, row)


@pytest.mark.parametrize('kind', DIAGNOSTICS)
def test_literal_prerequisites_preserve_rejections_without_graph_evidence(qualified, kind):
    row = _aux(qualified, kind)
    answer = row['messages'][-1]['content']
    assert answer['result']['request'] == row['metadata']['request']
    assert answer['continuation_decision'] == 'defer'
    assert answer['provenance'] == {'registry': [], 'graph': []}
    assert len([m for m in row['messages'] if 'tool_call' in m]) == (kind == 'tool_unknown_layer')
    changed = deepcopy(row)
    changed['messages'][-1]['content']['result']['request']['genes'] = []
    with pytest.raises(ValueError):
        _check(qualified, changed)


def test_case_relabeling_and_invented_exact_counts_do_not_pass_coverage(qualified):
    row = deepcopy(_case(qualified, 'S2.2.9'))
    row['messages'][-1]['content']['result']['supporting_layer_count'] = row['metadata']['support_count']
    with pytest.raises(ValueError):
        _check(qualified, row)
    row = deepcopy(_case(qualified, 'S2.2.1'))
    row['metadata']['catalog_case_id'] = 'S2.2.2'
    with pytest.raises(ValueError, match='binding'):
        _check(qualified, row)
    row = deepcopy(_aux(qualified, 'fresh_identity'))
    row['metadata']['catalog_case_id'] = 'S2.2.4'
    with pytest.raises(ValueError, match='auxiliary'):
        _check(qualified, row)


@pytest.mark.parametrize('source,target,variant', [
    ('S2.2.6', 'S2.2.10', 'bounded_context'),
    ('S2.2.10', 'S2.2.6', 'partial'),
])
def test_complete_subset_and_bounded_full_scope_are_not_interchangeable(qualified, source, target, variant):
    """Equal final outcomes cannot erase a reviewed acquisition distinction."""
    row = deepcopy(_case(qualified, source))
    row['metadata'].update(catalog_case_id=target, production_behavior=variant, variant=variant)
    with pytest.raises(ValueError, match='acquisition'):
        _check(qualified, row)


def test_reader_summary_is_not_an_exact_match_target(qualified):
    row = deepcopy(_case(qualified, 'S2.2.1'))
    row['messages'][-1]['content']['summary'] = 'This completed request has its scoped supporting layers.'
    _check(qualified, row)
