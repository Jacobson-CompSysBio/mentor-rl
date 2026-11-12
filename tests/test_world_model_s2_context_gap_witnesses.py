"""Reject context witnesses that collapse the intended partial-proof behavior."""

from copy import deepcopy
import json

import pytest

from scripts.build_world_model_s2_context_gap_witnesses import check_witness


def row(family=9, framing='direct'):
    """Supply a structurally sufficient proof for independent export guard tests."""
    result = ({'edges': [{'gene_ids': ['A', 'A'], 'weight': .913}]} if family == 9 else
              {'connectivity_kind': 'strong', 'component_count': 1,
               'layer_gene_count': 3, 'single_component': True})
    return {'row_id': 'proof', 'metadata': {'question_family': f'S2.{family}',
        'split': 'test' if family == 9 else 'validation',
        'variant': 'positive_loop_context' if family == 9 else 'strong_witness_context',
        'gene_ids': ['A'] if family == 9 else ['A', 'B', 'C'], 'framing': framing},
        'messages': [{'role': 'system', 'content': 'Pinned system.'},
            {'role': 'harness', 'content': json.dumps({'observations': [
                {'result': {'collection': {'complete': False}}}]})},
            {'role': 'user', 'content': 'The requested local graph.'},
            {'role': 'assistant', 'thinking': 'The visible partial proves the result.', 'content': {
                'status': 'resolved', 'blocked_reason': None,
                'continuation_decision': 'continue' if framing == 'step' else 'stop', 'result': result}}]}


def check(item):
    """Use the approved family and split rather than trusting mutated metadata."""
    family = 9 if item['metadata']['question_family'] == 'S2.9' else 10
    return check_witness(item, family=family, split='test' if family == 9 else 'validation',
        variant='positive_loop_context' if family == 9 else 'strong_witness_context', tokenizer=None)


@pytest.fixture(autouse=True)
def encoded(monkeypatch):
    """Keep source and real-tokenizer checks in the independent integration run."""
    data = {'input_ids': [1] * 100, 'completion_mask': [0] * 90 + [1] * 10}
    monkeypatch.setattr('scripts.build_world_model_s2_context_gap_witnesses.tokenize_trace', lambda *_: data)
    return data


@pytest.mark.parametrize('family,framing', [(9, 'direct'), (9, 'step'), (10, 'direct'), (10, 'step')])
def test_both_partial_proofs_preserve_task_framing(family, framing):
    assert check(row(family, framing))['assistant_tokens'] == 10


@pytest.mark.parametrize('family', [9, 10])
def test_complete_context_cannot_substitute_for_the_missing_partial_cell(family):
    item = row(family)
    item['messages'][1]['content'] = json.dumps({'observations': [{'result': {'collection': {'complete': True}}}]})
    with pytest.raises(ValueError, match='incomplete'):
        check(item)


@pytest.mark.parametrize('change', ['split', 'variant', 'call', 'decision', 'loop', 'additional_subject'])
def test_relabeling_or_a_different_graph_proof_cannot_pass(change):
    item = row()
    if change == 'split':
        item['metadata']['split'] = 'train'
    elif change == 'variant':
        item['metadata']['variant'] = 'context'
    elif change == 'call':
        item['messages'][-1]['tool_call'] = {'name': 'induce_subgraph', 'arguments': {}}
    elif change == 'decision':
        item['messages'][-1]['content']['continuation_decision'] = 'continue'
    elif change == 'loop':
        item['messages'][-1]['content']['result']['edges'][0]['gene_ids'] = ['A', 'B']
    else:
        item['metadata']['gene_ids'].append('B')
    with pytest.raises(ValueError):
        check(item)


def test_weak_connectivity_or_a_trivial_singleton_cannot_replace_the_strong_witness():
    original = row(10)
    for field, value in [('connectivity_kind', 'weak'), ('component_count', 2),
                         ('layer_gene_count', 1), ('single_component', False)]:
        changed = deepcopy(original)
        changed['messages'][-1]['content']['result'][field] = value
        with pytest.raises(ValueError, match='strong'):
            check(changed)


def test_complete_wire_budget_and_supervision_are_enforced(encoded):
    encoded.update(input_ids=[1] * 8192, completion_mask=[0] * 8191 + [1])
    assert check(row())['tokens'] == 8192
    encoded['input_ids'].append(1)
    encoded['completion_mask'].append(1)
    with pytest.raises(ValueError, match='scope'):
        check(row())
    encoded.update(input_ids=[1], completion_mask=[0])
    with pytest.raises(ValueError, match='scope'):
        check(row())
