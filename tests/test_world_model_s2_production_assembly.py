"""Reject incomplete or altered production quotas and inconsistent input pins."""

import hashlib
import json

import pytest

from scripts import assemble_world_model_s2_production as assembly
from runtime.world_model.s2_production import production_budget


def test_budget_recomputes_the_exact_mixture_after_explicit_exclusions():
    budget = production_budget(394)
    assert budget['s2'] == {'train': 201728, 'validation': 25216, 'test': 25216}
    assert budget['replay'] == {'S1': 37824, 'S0': 12608}
    assert budget['train'] == 252160
    assert budget['s2']['train'] / budget['train'] == .8
    assert budget['replay']['S1'] / budget['train'] == .15
    assert budget['replay']['S0'] / budget['train'] == .05


def test_pin_verification_rejects_conflicting_versions_and_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly, 'ROOT', tmp_path)
    path = tmp_path / 'native-proof.py'
    path.write_bytes(b'proof')
    pin = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert assembly.checked_pins([pin, pin]) == [pin]
    with pytest.raises(ValueError, match='inconsistent immutable input versions'):
        assembly.checked_pins([pin, {**pin, 'sha256': 'f' * 64}])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='assembly input changed'):
        assembly.checked_pins([pin])


@pytest.fixture
def shards(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly, 'ROOT', tmp_path)
    quotas = {'train': 2, 'validation': 1, 'test': 1}
    for split, quota in quotas.items():
        directory = tmp_path / 'shards/s2_1/S2.1.1' / split
        directory.mkdir(parents=True)
        payload = ('{}\n' * quota).encode()
        path = directory / 'rows.jsonl'
        path.write_bytes(payload)
        report = {'case_id': 'S2.1.1', 'split': split, 'quota': quota,
            'rows': [{'tokens': 8, 'assistant_tokens': 3} for _ in range(quota)], 'source_replay_checked': True,
            'frozen_predicate_checked': True, 'truncation_applied': False,
            'artifact': str(path.relative_to(tmp_path)), 'artifact_sha256': hashlib.sha256(payload).hexdigest()}
        (directory / 'report.json').write_text(json.dumps(report))
    return tmp_path / 'shards', quotas


def test_all_retained_case_split_cells_are_required(shards):
    root, quotas = shards
    assert len(assembly.read_shards(root, ['S2.1.1'], quotas)) == 3
    with pytest.raises(ValueError, match='quota is incomplete: S2.1.2/train'):
        assembly.read_shards(root, ['S2.1.1', 'S2.1.2'], quotas)


@pytest.mark.parametrize('fault', ['qualification_count', 'source_replay', 'predicate', 'truncation', 'payload', 'overlength', 'supervision'])
def test_qualification_or_changed_rows_cannot_be_published_as_production(shards, fault):
    root, quotas = shards
    directory = root / 's2_1/S2.1.1/train'
    path = directory / 'report.json'
    report = json.loads(path.read_text())
    if fault == 'qualification_count':
        report['quota'] = 1
    elif fault == 'source_replay':
        report['source_replay_checked'] = False
    elif fault == 'predicate':
        report['frozen_predicate_checked'] = False
    elif fault == 'truncation':
        report['truncation_applied'] = True
    elif fault == 'overlength':
        report['rows'][0]['tokens'] = 8193
    elif fault == 'supervision':
        report['rows'][0]['assistant_tokens'] = 0
    else:
        (directory / 'rows.jsonl').write_text('{}\n')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        assembly.read_shards(root, ['S2.1.1'], quotas)
