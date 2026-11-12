"""Check the mixed production wire, label mask, and structured evaluation target."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.world_model import s2_production_adapter as adapter
from runtime.world_model.contract import load_contract, tool_definitions
from runtime.world_model.s2_3_tokens import load_tokenizer
from runtime.world_model.s2_3_tokens import _model_messages, verification_clock
from runtime.world_model.s2_production_replay import LEGACY_INSTRUCTION, adapt_replay_row


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def row():
    """Use an actually source-qualified retry trace with supplied native evidence."""
    path = ROOT / 'data/world_model_v4/diagnostics/s2_thirteen_case_repairs_v2/s2_10/train.jsonl'
    return next(value for line in path.read_text().splitlines()
                if (value := json.loads(line))['metadata']['catalog_case_id'] == 'S2.10.28')


@pytest.fixture
def routes(row):
    """Bind the complete prompt and ordered tool definitions from the real row."""
    return {row['metadata']['question_family']: {
        'contract_version': row['metadata']['tool_contract_version'],
        'system_prompt_sha256': hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest(),
        'tools_sha256': adapter.tools_sha256(row['tools'])}}


@pytest.fixture(scope='module')
def tokenizer():
    """Load the same pinned local tokenizer used by production qualification."""
    pin = json.loads((ROOT / 'runtime/world_model/s2_production_v1_8k.json').read_text())['tokenizer']
    if not (ROOT / pin['path']).exists():
        pytest.skip('the pinned local production tokenizer is unavailable')
    return load_tokenizer(pin)


def legacy_row(stage='S1'):
    """Retain historical object/null evidence and an opaque old graph hash."""
    contract = load_contract()
    system = contract['runtime']['system_prompt']
    provenance = {'registry': {'registry': 'ensembl', 'release': '116', 'evidence_id': 'ev_old'}, 'graph': None}
    return {'row_id': 'legacy_source', 'messages': [
        {'role': 'system', 'content': system},
        {'role': 'harness', 'content': '{"legacy_graph_version":"old_coordinate_hash"}'},
        {'role': 'user', 'content': 'Resolve TP53.'},
        {'role': 'assistant', 'thinking': 'The authoritative registry mapping is required.',
         'tool_call': {'name': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': 'TP53'}}},
        {'role': 'tool', 'name': 'lookup_human_gene_symbol', 'content': {'status': 'resolved',
            'provenance': deepcopy(provenance), 'gene_id': 'ENSG00000141510', 'gene_symbol': 'TP53'}},
        {'role': 'assistant', 'thinking': 'The returned ID settles the requested identity.', 'content': {
            'status': 'resolved', 'provenance': deepcopy(provenance),
            'result': {'gene_id': 'ENSG00000141510', 'gene_symbol': 'TP53'},
            'continuation_decision': 'stop', 'blocked_reason': None, 'summary': 'TP53 resolves to its human ID.'}}],
        'tools': tool_definitions(contract), 'metadata': {'step': stage,
            'question_family': 'human_symbol_to_ensembl', 'source_trace': stage + '.1.1',
            'split_unit': 'old_subject', 'framing': 'direct', 'schema_version': contract['runtime']['row_schema_version'],
            'tool_contract_version': contract['runtime']['tool_contract_version'],
            'system_prompt_sha256': hashlib.sha256(system.encode()).hexdigest()}}


def test_scalar_arrow_roundtrip_preserves_order_identity_and_all_evidence(row, routes):
    original = deepcopy(row)
    scalar = adapter.flatten_row(row, expected_split='train', contract_routes=routes)
    assert list(scalar) == ['trace_json', 'metadata_json', 'question_family', 'record_id',
                            'fact_id', 'prompt_form_id', 'split']
    assert all(isinstance(value, str) for value in scalar.values())
    decoded = json.loads(scalar['trace_json'])
    assert decoded == row == original
    assert list(decoded) == list(row)
    assert list(decoded['messages'][-1]['content']) == list(row['messages'][-1]['content'])
    assert json.loads(scalar['metadata_json']) == row['metadata']
    assert scalar['record_id'] == row['row_id'] and scalar['fact_id'] == row['metadata']['split_unit']


@pytest.mark.parametrize('mutation', ['row_order', 'split', 'family', 'case', 'version', 'prompt',
    'prompt_hash', 'tools', 'route_tools', 'no_route', 'extra_user', 'response_name', 'supervision'])
def test_scalar_adapter_rejects_contract_or_wire_drift(row, routes, mutation):
    if mutation == 'row_order':
        row = {key: row[key] for key in ('metadata', 'row_id', 'messages', 'tools')}
    elif mutation == 'split':
        row['metadata']['split'] = 'test'
    elif mutation == 'family':
        row['metadata']['question_family'] = 'S2.99'
    elif mutation == 'case':
        row['metadata']['catalog_case_id'] = 'S2.9.28'
    elif mutation == 'version':
        row['metadata']['tool_contract_version'] += '_changed'
    elif mutation == 'prompt':
        row['messages'][0]['content'] += ' Change the evidence interpretation.'
    elif mutation == 'prompt_hash':
        row['metadata']['system_prompt_sha256'] = '0' * 64
    elif mutation == 'tools':
        row['tools'][0]['function']['description'] += ' Changed.'
    elif mutation == 'route_tools':
        routes['S2.10']['tools_sha256'] = '0' * 64
    elif mutation == 'no_route':
        routes.clear()
    elif mutation == 'extra_user':
        row['messages'].extend([{'role': 'user', 'content': 'Another task.'}, deepcopy(row['messages'][-1])])
    elif mutation == 'response_name':
        next(message for message in row['messages'] if message['role'] == 'tool')['name'] = 'invented_tool'
    else:
        row['metadata']['supervision'] = 'all_tokens'
    with pytest.raises(ValueError):
        adapter.flatten_row(row, expected_split='train', contract_routes=routes)


def test_tool_route_hash_retains_definition_order(row):
    ordered = adapter.tools_sha256(row['tools'])
    reordered = deepcopy(row['tools'])
    definition = reordered[0]['function']
    reordered[0]['function'] = dict(reversed(list(definition.items())))
    assert adapter.tools_sha256(reordered) != ordered


def test_actual_tokenization_supervises_both_retry_thoughts_and_final_but_masks_source(row, routes, tokenizer):
    original = deepcopy(row)
    encoded = adapter.tokenize_row(row, tokenizer, expected_split='train', contract_routes=routes)
    assert row == original
    assert len(encoded['input_ids']) == len(encoded['attention_mask']) == len(encoded['labels'])
    assert all(value == 1 for value in encoded['attention_mask'])
    assert 0 < sum(encoded['completion_mask']) < len(encoded['input_ids']) < 8192
    assert encoded['labels'] == [token if flag else -100 for token, flag in
                                 zip(encoded['input_ids'], encoded['completion_mask'])]
    supervised = tokenizer.decode([token for token, flag in zip(encoded['input_ids'], encoded['completion_mask']) if flag])
    masked = tokenizer.decode([token for token, flag in zip(encoded['input_ids'], encoded['completion_mask']) if not flag])
    for message in row['messages']:
        if message['role'] == 'assistant':
            assert message['thinking'] in supervised
        elif message['role'] == 'tool':
            assert message['content']['message'] not in supervised
            assert message['content']['message'] in masked
    assert 'Context (from the harness):' in masked
    assert '"observations":' in masked and '"observations":' not in supervised


@pytest.mark.parametrize(('length', 'mask', 'valid'), [
    (8192, [0] * 8191 + [1], True), (8193, [0] * 8192 + [1], False),
    (0, [], False), (3, [0, 1], False), (3, [0, 0, 0], False), (3, [0, 2, 1], False),
])
def test_complete_trace_budget_and_supervision_boundaries(row, routes, monkeypatch, length, mask, valid):
    monkeypatch.setattr(adapter, 'tokenize_trace', lambda *_: {'input_ids': [11] * length, 'completion_mask': mask})
    if valid:
        encoded = adapter.tokenize_row(row, object(), expected_split='train', contract_routes=routes)
        assert encoded['labels'][-1] == 11 and encoded['labels'][:-1] == [-100] * 8191
    else:
        with pytest.raises(ValueError, match='complete production trace'):
            adapter.tokenize_row(row, object(), expected_split='train', contract_routes=routes)


@pytest.mark.parametrize('stage', ['S0', 'S1'])
def test_legacy_rows_keep_original_contract_and_are_training_only(stage, tokenizer):
    original = legacy_row(stage)
    row = adapt_replay_row(original, stage)
    before = deepcopy(row)
    scalar = adapter.flatten_row(row, expected_split='train', contract_routes={})
    assert json.loads(scalar['trace_json']) == row == before
    assert row['messages'][1:] == original['messages'][1:]
    assert row['messages'][0]['content'] == LEGACY_INSTRUCTION + original['messages'][0]['content']
    assert row['messages'][-1]['content']['provenance']['graph'] is None
    assert 'old_coordinate_hash' in scalar['trace_json']
    encoded = adapter.tokenize_row(row, tokenizer, expected_split='train', contract_routes={})
    assert 'legacy S0/S1' in tokenizer.decode(encoded['input_ids'])
    for split in ('validation', 'test'):
        with pytest.raises(ValueError, match='only to the training split'):
            adapter.flatten_row(row, expected_split=split, contract_routes={})
    with pytest.raises(ValueError):
        adapter.flatten_row(original, expected_split='train', contract_routes={})


def test_legacy_declared_original_prompt_hash_cannot_hide_a_changed_legacy_contract():
    row = adapt_replay_row(legacy_row(), 'S1')
    row['messages'][0]['content'] += ' Change the original interpretation.'
    row['metadata']['system_prompt_sha256'] = hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest()
    with pytest.raises(ValueError, match='original contract identity'):
        adapter.flatten_row(row, expected_split='train', contract_routes={})


def test_evaluation_target_preserves_structured_values_and_calls_without_scoring_free_prose(row):
    original = deepcopy(row)
    target = adapter.evaluation_target(row)
    final = row['messages'][-1]['content']
    assert target['result'] == final['result']
    assert target['provenance'] == final['provenance']
    assert target['blocked_reason_required'] is True
    assert target['expected_tool_calls'] == [message['tool_call'] for message in row['messages'] if 'tool_call' in message]
    assert 'summary' not in target and 'blocked_reason' not in target and 'thinking' not in target
    row['messages'][-1]['content'].update(summary='A different readable summary.', blocked_reason='Different recovery wording.')
    row['messages'][-1]['thinking'] = 'A differently worded evidence explanation.'
    assert adapter.evaluation_target(row) == target
    target['result'].clear()
    assert original['messages'][-1]['content']['result'] == row['messages'][-1]['content']['result']


def test_stale_and_current_graph_versions_survive_arrow_projection_without_relabeling():
    path = ROOT / 'data/world_model_v4/diagnostics/s2_thirteen_case_repairs_v2/s2_8/train.jsonl'
    row = json.loads(path.read_text().splitlines()[0])
    routes = {'S2.8': {'contract_version': row['metadata']['tool_contract_version'],
        'system_prompt_sha256': row['metadata']['system_prompt_sha256'], 'tools_sha256': adapter.tools_sha256(row['tools'])}}
    scalar = adapter.flatten_row(row, expected_split='train', contract_routes=routes)
    decoded = json.loads(scalar['trace_json'])
    stale = json.loads(decoded['messages'][1]['content'])['observations'][0]['provenance']['graph'][0]['graph_version']
    fresh = next(message for message in decoded['messages'] if message['role'] == 'tool')['content']['provenance']['graph'][0]['graph_version']
    assert stale == '0' * 64 and fresh == row['metadata']['graph_version'] != stale


def test_actual_multiturn_legacy_history_roundtrips_and_only_current_turn_is_supervised(tokenizer):
    path = ROOT / 'data/world_model_v4/diagnostics/s2_production_replay_v1/s1/rows.jsonl'
    with path.open() as stream:
        row = next(value for value in map(json.loads, stream)
                   if sum(message['role'] == 'user' for message in value['messages']) > 1)
    assert row['metadata']['source_trace'] == 'S1.3.9'
    current_user = max(index for index, message in enumerate(row['messages']) if message['role'] == 'user')
    prior = next(message for message in row['messages'][:current_user] if message['role'] == 'assistant')
    assert 'thinking' not in prior
    before = deepcopy(row)
    scalar = adapter.flatten_row(row, expected_split='train', contract_routes={})
    assert json.loads(scalar['trace_json']) == row == before
    encoded = adapter.tokenize_row(row, tokenizer, expected_split='train', contract_routes={})
    messages = _model_messages(row)
    last_user = max(index for index, message in enumerate(messages) if message['role'] == 'user')
    base = deepcopy(messages[:last_user + 1])
    for message in base:
        if message['role'] == 'assistant':
            message.pop('thinking', None)
    prefix = tokenizer.apply_chat_template(base, tools=row['tools'], tokenize=True,
        add_generation_prompt=False, enable_thinking=True, reasoning_effort='low', strftime_now=verification_clock)
    assert encoded['input_ids'][:len(prefix)] == prefix
    assert encoded['completion_mask'][:len(prefix)] == [0] * len(prefix)
    supervised = tokenizer.decode([token for token, flag in zip(encoded['input_ids'], encoded['completion_mask']) if flag])
    masked = tokenizer.decode([token for token, flag in zip(encoded['input_ids'], encoded['completion_mask']) if not flag])
    assert row['messages'][-1]['thinking'] in supervised
    assert prior['content']['summary'] in masked
    assert prior['content']['summary'] not in supervised


@pytest.mark.parametrize('prior_thinking', ['', 'Historical private reasoning that must never be supervised.'])
def test_legacy_prior_calls_are_context_and_evaluation_targets_cover_only_the_current_turn(tokenizer, prior_thinking):
    original = legacy_row()
    original['messages'][3]['thinking'] = prior_thinking
    original['messages'][5]['thinking'] = prior_thinking
    current = deepcopy(original['messages'][3:])
    current[0]['thinking'] = 'Acquire the authoritative mapping for the current question.'
    current[-1]['thinking'] = 'The current result settles this question.'
    original['messages'].extend([{'role': 'user', 'content': 'Resolve TP53 again.'}, *current])
    row = adapt_replay_row(original, 'S1')
    scalar = adapter.flatten_row(row, expected_split='train', contract_routes={})
    assert json.loads(scalar['trace_json']) == row
    encoded = adapter.tokenize_row(row, tokenizer, expected_split='train', contract_routes={})
    supervised = tokenizer.decode([token for token, flag in zip(encoded['input_ids'], encoded['completion_mask']) if flag])
    if prior_thinking:
        assert prior_thinking not in tokenizer.decode(encoded['input_ids'])
    assert current[0]['thinking'] in supervised and current[-1]['thinking'] in supervised
    target = adapter.evaluation_target(row)
    assert target['expected_tool_calls'] == [current[0]['tool_call']]
    assert target['expected_tool_responses'] == [{'name': current[1]['name'], 'content': current[1]['content']}]
