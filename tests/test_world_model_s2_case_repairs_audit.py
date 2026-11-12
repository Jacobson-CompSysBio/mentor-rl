"""Check repair qualification boundaries and the immutable baseline receipt."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import audit_world_model_s2_case_quotas as quotas
from scripts import audit_world_model_s2_case_repairs as audit
from scripts import build_world_model_s2_case_repairs as build
from runtime.world_model.s2_audit_identity import model_trace_sha256


@pytest.fixture
def candidate(monkeypatch):
    """Supply one complete synthetic wire with an independently controlled encoder."""
    row = {'row_id': 'synthetic_repaired_degree', 'messages': [
        {'role': 'system', 'content': 'Use supplied evidence and exact requested scope.'},
        {'role': 'harness', 'content': json.dumps({'active_graph': {'graph_version': 'a' * 64}, 'observations': []})},
        {'role': 'user', 'content': 'Give this gene degree.'},
        {'role': 'assistant', 'thinking': 'The compatible source record settles the requested count.',
         'content': {'status': 'resolved', 'provenance': {'registry': [], 'graph': []},
             'result': {'degree': 3}, 'continuation_decision': 'stop', 'blocked_reason': None,
             'summary': 'The requested degree is three.'}}],
        'tools': [{'type': 'function', 'function': {'name': 'get_degree',
            'description': 'Acquire the requested native distinct-neighbor count.',
            'parameters': {'type': 'object', 'properties': {}}}}],
        'metadata': {'catalog_case_id': 'S2.8.22', 'question_family': 'S2.8', 'variant': 'stale',
                     'framing': 'direct', 'split': 'test'}}
    case = SimpleNamespace(case_id='S2.8.22', family='S2.8', status='resolved',
        continuation_decision='stop', framing_hint='direct')
    features = SimpleNamespace(for_row=lambda value: {'selected_layer_count': 1})
    matched = []

    def matches(case_id, classified):
        matched.append((case_id, classified))
        return classified['_audit_source'] == {'selected_layer_count': 1}

    encoded = {'input_ids': [7] * 8192, 'completion_mask': [0] * 8189 + [1, 1, 1]}
    tokenizer = object()
    encoder_calls = []

    def encode(value, selected_tokenizer):
        encoder_calls.append((value, selected_tokenizer))
        return encoded

    monkeypatch.setattr(build, 'tokenize_trace', encode)
    return SimpleNamespace(row=row, case=case, expected={'result': {'degree': 0}},
        binding={'variants': ('stale',)}, matcher=SimpleNamespace(matches=matches),
        features=features, tokenizer=tokenizer, encoded=encoded, matched=matched,
        encoder_calls=encoder_calls)


def check(candidate):
    """Use the public qualification function with its actual identity hashing."""
    return build.check_case(candidate.row, candidate.case, candidate.expected, candidate.binding,
                            candidate.matcher, candidate.features, candidate.tokenizer)


def test_qualifies_the_whole_8192_token_wire_and_passes_actual_features(candidate):
    result = check(candidate)
    assert result == {'case_id': 'S2.8.22', 'row_id': candidate.row['row_id'], 'split': 'test',
        'tokens': 8192, 'assistant_tokens': 3, 'model_trace_sha256': model_trace_sha256(candidate.row)}
    assert candidate.encoder_calls == [(candidate.row, candidate.tokenizer)]
    assert candidate.matched[0][0] == candidate.case.case_id
    assert candidate.matched[0][1]['_audit_source'] == {'selected_layer_count': 1}
    assert '_audit_source' not in candidate.row


@pytest.mark.parametrize('mutation', [
    'case_id', 'family', 'variant', 'status', 'decision', 'framing', 'extra_result_key', 'missing_result_key',
])
def test_rejects_changes_to_approved_case_scope_and_envelope(candidate, mutation):
    meta = candidate.row['metadata']
    final = candidate.row['messages'][-1]['content']
    if mutation == 'case_id':
        meta['catalog_case_id'] = 'S2.8.26'
    elif mutation == 'family':
        meta['question_family'] = 'S2.9'
    elif mutation == 'variant':
        meta['variant'] = 'lookup'
    elif mutation == 'status':
        final['status'] = 'error'
    elif mutation == 'decision':
        final['continuation_decision'] = 'continue'
    elif mutation == 'framing':
        meta['framing'] = 'step'
    elif mutation == 'extra_result_key':
        final['result']['hub_like'] = True
    elif mutation == 'missing_result_key':
        final['result'].pop('degree')
    with pytest.raises(ValueError, match='approved case'):
        check(candidate)
    assert not candidate.encoder_calls


def test_rejects_a_false_frozen_predicate_even_when_the_envelope_matches(candidate):
    candidate.matcher.matches = lambda *_: False
    with pytest.raises(ValueError, match='frozen audit requirements'):
        check(candidate)
    assert not candidate.encoder_calls


@pytest.mark.parametrize(('length', 'mask'), [
    (8193, [0] * 8192 + [1]),
    (0, []),
    (3, [0, 1]),
    (3, [0, 0, 0]),
    (3, [0, 2, 1]),
])
def test_rejects_overlength_or_invalid_complete_trace_supervision(candidate, length, mask):
    candidate.encoded.update(input_ids=[7] * length, completion_mask=mask)
    with pytest.raises(ValueError, match='complete trace or supervision mask'):
        check(candidate)


def test_allows_overlap_only_among_repaired_case_targets():
    added = set()
    audit.check_new_identity('S2.8.22', ['S2.8.22', 'S2.8.26'], 'new_trace',
        {'historical_trace'}, added, {'S2.8.22', 'S2.8.26'})
    assert added == {'new_trace'}


@pytest.mark.parametrize('matching', [[], ['S2.8.26'], ['S2.8.22', 'S2.8.10']])
def test_repair_neighborhood_requires_intended_case_and_no_outside_target(matching):
    added = set()
    with pytest.raises(ValueError, match='intended case and only'):
        audit.check_new_identity('S2.8.22', matching, 'new_trace', set(), added,
                                 {'S2.8.22', 'S2.8.26'})
    assert not added


@pytest.mark.parametrize('source', ['historical', 'newly_added'])
def test_duplicate_model_traces_cannot_receive_new_credit(source):
    previous = {'duplicate'} if source == 'historical' else set()
    added = {'duplicate'} if source == 'newly_added' else set()
    before = deepcopy(added)
    with pytest.raises(ValueError, match='cannot reuse an old or already credited'):
        audit.check_new_identity('S2.8.22', ['S2.8.22'], 'duplicate', previous, added, {'S2.8.22'})
    assert added == before


@pytest.fixture
def frozen_receipt(tmp_path, monkeypatch):
    """Pin a tiny four-artifact historical audit beneath an isolated root."""
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(quotas, 'ROOT', tmp_path)
    report = {'case_entries': [{'case_id': 'S2.8.22'}], 'training_ready': False}
    payloads = {'authoritative_report': json.dumps(report).encode(),
        'case_matrix': b'case_id,eligible_test\nS2.8.22,0\n',
        'allocation': b'{"case_id":"S2.8.1"}\n', 'validated_witnesses': b'[]\n'}
    receipt = {}
    for field, payload in payloads.items():
        relative = f'historical/{field}.json'
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        receipt[field] = {'path': relative, 'sha256': hashlib.sha256(payload).hexdigest()}
    receipt_path = tmp_path / 'diagnostics/s2_case_quota_audit/summary.json'
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))
    return tmp_path, report, receipt, receipt_path


def test_frozen_audit_loads_only_the_receipt_bound_historical_artifacts(frozen_receipt):
    _, report, receipt, receipt_path = frozen_receipt
    loaded, pins = build.frozen_audit()
    assert loaded == report
    assert pins[0] == {'path': 'diagnostics/s2_case_quota_audit/summary.json',
                       'sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    assert pins[1:] == list(receipt.values())


@pytest.mark.parametrize('field', ['authoritative_report', 'case_matrix', 'allocation', 'validated_witnesses'])
def test_frozen_audit_rejects_changed_bytes_for_each_published_artifact(frozen_receipt, field):
    root, _, receipt, _ = frozen_receipt
    path = root / receipt[field]['path']
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match=f'published receipt: {field}'):
        build.frozen_audit()
