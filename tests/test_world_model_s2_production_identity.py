"""Prevent hidden earlier analysis from inflating production replay quotas."""

from copy import deepcopy

from runtime.world_model.s2_audit_identity import model_trace_sha256 as source_identity
from runtime.world_model.s2_production_identity import model_trace_sha256


def history():
    """Supply earlier context followed by the actual supervised current turn."""
    return {'row_id': 'original', 'tools': [], 'metadata': {}, 'messages': [
        {'role': 'system', 'content': 'fixture'},
        {'role': 'user', 'content': 'Earlier question'},
        {'role': 'assistant', 'thinking': 'Earlier private analysis', 'content': {'answer': 'Earlier answer'}},
        {'role': 'user', 'content': 'Current question'},
        {'role': 'assistant', 'thinking': 'Current supervised analysis', 'content': {'answer': 'Current answer'}}]}


def test_masked_historical_analysis_cannot_make_duplicate_replay_distinct():
    first = history()
    second = deepcopy(first)
    second['messages'][2]['thinking'] = 'A differently phrased earlier analysis'
    assert source_identity(first) != source_identity(second)
    assert model_trace_sha256(first) == model_trace_sha256(second)
    assert 'thinking' in first['messages'][2]


def test_current_analysis_and_visible_historical_answer_remain_part_of_identity():
    first = history()
    for index, field, value in [(4, 'thinking', 'Different current reasoning'),
                                 (2, 'content', {'answer': 'Different visible history'})]:
        second = deepcopy(first)
        second['messages'][index][field] = value
        assert model_trace_sha256(first) != model_trace_sha256(second)


def test_single_user_s2_source_fingerprints_remain_byte_equivalent():
    row = history()
    row['messages'] = [row['messages'][0], *row['messages'][3:]]
    assert model_trace_sha256(row) == source_identity(row)
