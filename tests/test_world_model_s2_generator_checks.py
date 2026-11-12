"""Keep generator qualification honest about pinned inputs and witness scope."""

from copy import deepcopy
import hashlib
import json

import pytest

from scripts.check_world_model_s2_generators import check_tokens, representative_rows


def row(identity, *, split='train', variant='lookup', framing='direct', **features):
    """Supply a candidate identity with independent semantic coverage features."""
    return {'row_id': identity, 'metadata': {'split': split, 'question_family': 'S2.3',
            'variant': variant, 'framing': framing, **features}}


def candidates(tmp_path, rows):
    """Write a pinned candidate file exactly as the filtering stage would."""
    path = tmp_path / 'train.jsonl'
    data = ''.join(json.dumps(item) + '\n' for item in rows).encode()
    path.write_bytes(data)
    return path, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'rows': len(rows)}


def test_representatives_cover_variants_framings_and_independent_features(tmp_path):
    rows = [row('a', direction='in'), row('b', direction='out'),
            row('c', variant='terminal'), row('d', framing='step'),
            row('e', question_kind='claim')]
    path, pin = candidates(tmp_path, rows)
    selected = representative_rows(path, pin, 'train', 'S2.3')
    assert {item['row_id'] for item in selected} == {'a', 'b', 'c', 'd', 'e'}


def test_representatives_prefer_smaller_witness_without_duplicate_rows(tmp_path):
    small = row('a')
    large = row('b')
    large['extra'] = 'Long supplied evidence. ' * 20
    path, pin = candidates(tmp_path, [large, small])
    assert representative_rows(path, pin, 'train', 'S2.3') == [small]


@pytest.mark.parametrize('change', ['hash', 'size', 'count', 'split', 'family', 'duplicate'])
def test_changed_pins_or_source_identity_cannot_pass(tmp_path, change):
    rows = [row('a')]
    if change == 'split':
        rows[0]['metadata']['split'] = 'test'
    elif change == 'family':
        rows[0]['metadata']['question_family'] = 'S2.4'
    elif change == 'duplicate':
        rows.append(deepcopy(rows[0]))
    path, pin = candidates(tmp_path, rows)
    if change == 'hash':
        pin['sha256'] = '0' * 64
    elif change == 'size':
        pin['bytes'] += 1
    elif change == 'count':
        pin['rows'] += 1
    with pytest.raises(ValueError):
        representative_rows(path, pin, 'train', 'S2.3')


def test_complete_trace_check_accepts_boundary_and_rejects_truncation_or_empty_mask(monkeypatch):
    encoded = {'input_ids': [1] * 8192, 'completion_mask': [0] * 8191 + [1]}
    monkeypatch.setattr('scripts.check_world_model_s2_generators.tokenize_trace', lambda *_: encoded)
    assert check_tokens({'row_id': 'boundary'}, None) == 8192
    encoded['input_ids'].append(1)
    encoded['completion_mask'].append(1)
    with pytest.raises(ValueError):
        check_tokens({'row_id': 'long'}, None)
    encoded.update(input_ids=[1], completion_mask=[0])
    with pytest.raises(ValueError):
        check_tokens({'row_id': 'unsupervised'}, None)
