"""Check rare-loop wording capacity and safe resume of frozen old shards."""

from copy import deepcopy
from itertools import islice
import hashlib
import json
from pathlib import Path

import pytest

from runtime.world_model import s2_production_case_3_19 as production
from runtime.world_model import s2_3_corpus as neighbors
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_corpus import canonical_json
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex
from scripts import build_world_model_s2_production_v2 as builder


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def context():
    """Supply one actual incoming loop subject per unchanged split."""
    genes = [f'ENSG{number:011d}' for number in range(1, 81)]
    loops = [next(gene for gene in genes if gene_group(gene, 20261028)[1] == split)
             for split in ('train', 'validation', 'test')]
    edges = {(genes[number], genes[(number + 1) % len(genes)]): .5 for number in range(len(genes))}
    edges.update({(gene, gene): .25 for gene in loops})
    layer = 'TFs:cd4+t_cells'
    view = GraphView('loop_quota_fixture', 'e' * 64,
        {layer: LayerGraph(layer, True, edges, frozenset(genes))})
    return {'view': view, 'index': SubgraphIndex(view), 'seed': 20261028,
        'contract': neighbors.load_build_contract(),
        'bank': json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text()),
        'registry': RegistryContext('110', 'ensembl_homo_sapiens_110_loop_fixture',
                                   {gene: f'LOOP_{number}' for number, gene in enumerate(genes)})}


@pytest.mark.parametrize(('split', 'quota', 'templates'), [('train', 512, 16), ('validation', 64, 8), ('test', 64, 8)])
def test_single_native_loop_subject_reaches_its_quota_with_actual_visible_wording(context, split, quota, templates):
    original_contract = deepcopy(context['contract'])
    rows = list(production.production_candidates(production.CASE_ID, context, split=split))
    assert len(rows) == len({model_trace_sha256(row) for row in rows}) == quota
    assert len({row['metadata']['gene_id'] for row in rows}) == 1
    assert len({row['metadata']['wrapper_index'] for row in rows}) == (32 if split == 'train' else 8)
    assert len({row['metadata']['request_template_index'] for row in rows}) == templates
    assert context['contract'] == original_contract
    for row in rows[::8]:
        production.validate_row(row, context)
        final = row['messages'][-1]['content']
        gene = row['metadata']['gene_id']
        assert final['result']['collection']['complete'] is True
        assert sum(entry['gene_id'] == gene for entry in final['result']['neighbors']) == 1
        assert final['continuation_decision'] == 'stop'
        response = next(message['content'] for message in row['messages'] if message['role'] == 'tool')
        identity = response['provenance']['graph'][0]['evidence_id']
        assert row['row_id'] in identity
        assert final['provenance']['graph'][0]['evidence_id'] == identity
        assert row['metadata']['split'] == gene_group(gene, 20261028)[1] == split


def test_other_neighbor_cases_retain_previous_generator_rows_byte_for_byte(context):
    expected = list(islice(production.previous.production_candidates('S2.3.9', context, split='train'), 3))
    actual = list(islice(production.production_candidates('S2.3.9', context, split='train'), 3))
    assert canonical_json(actual) == canonical_json(expected)
    for row in actual:
        production.validate_row(row, context)


@pytest.fixture(scope='module')
def high_context(context):
    """Supply the exact sparse source capacities needed by the real large case."""
    genes = [f'ENSG{number:011d}' for number in range(1, 901)]
    hubs = [gene for split, count in (('train', 4), ('validation', 2), ('test', 2))
            for gene in [g for g in genes if gene_group(g, 20261028)[1] == split][:count]]
    edges = {(hub, target): .5 for hub in hubs for target in genes[:731]}
    layer = 'TFs:brain'
    view = GraphView('high_population_fixture', 'f' * 64,
        {layer: LayerGraph(layer, True, edges, frozenset(gene for pair in edges for gene in pair))})
    return {'view': view, 'index': SubgraphIndex(view), 'seed': 20261028,
        'contract': deepcopy(context['contract']), 'bank': context['bank'],
        'registry': RegistryContext('110', 'ensembl_homo_sapiens_110_high_fixture',
                                   {gene: f'HIGH_{number}' for number, gene in enumerate(genes)})}


@pytest.mark.parametrize(('split', 'quota'), [('train', 512), ('validation', 64), ('test', 64)])
def test_large_outgoing_case_uses_all_existing_request_and_wrapper_combinations(high_context, split, quota):
    rows = list(production.production_candidates('S2.3.20', high_context, split=split))
    assert len(rows) == len({model_trace_sha256(row) for row in rows}) == quota
    for row in rows[::8]:
        production.validate_row(row, high_context)
        collection = row['messages'][-1]['content']['result']['collection']
        assert collection['returned_count'] == 1 and collection['total_count'] == 731
        assert collection['complete'] is False
        assert row['metadata']['request_template_index'] in range(4)


def test_original_completed_large_case_train_rows_still_validate(high_context):
    row = next(production.previous.production_candidates('S2.3.20', high_context, split='train'))
    before = deepcopy(row)
    assert 'production_extension' not in row['metadata']
    production.validate_row(row, high_context)
    assert row == before


@pytest.mark.parametrize('mutation', ['request_index', 'wrapper', 'loop_weight', 'identity', 'complete'])
def test_modified_wording_or_native_loop_proof_is_rejected(context, mutation):
    row = next(production.production_candidates(production.CASE_ID, context, split='train'))
    if mutation == 'request_index':
        row['metadata']['request_template_index'] += 1
    elif mutation == 'wrapper':
        row['metadata']['wrapper_index'] += 1
    elif mutation == 'loop_weight':
        response = next(message['content'] for message in row['messages'] if message['role'] == 'tool')
        next(entry for entry in response['result']['neighbors']
             if entry['gene_id'] == row['metadata']['gene_id'])['weight'] = 999
    elif mutation == 'identity':
        row['row_id'] += '_cosmetic_relabel'
    else:
        row['messages'][-1]['content']['result']['collection']['complete'] = False
    with pytest.raises(ValueError):
        production.validate_row(row, context)


@pytest.fixture
def old_shard(tmp_path, monkeypatch):
    """Make a tiny immutable old receipt whose inputs remain independently pinned."""
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    source = tmp_path / 'old_implementation.py'
    source.write_text('original implementation\n')
    artifact = tmp_path / 'rows.jsonl'
    artifact.write_text('{"row_id":"original"}\n')
    report = {'case_id': 'S2.3.1', 'quota': 1, 'rows': [{'row_id': 'original'}],
        'input_pins': [{'path': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}],
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (tmp_path / 'report.json').write_text(json.dumps(report))
    return tmp_path, report, source, artifact


def test_completed_old_shard_survives_additional_current_code_pins(old_shard):
    path, report, _, _ = old_shard
    assert builder.verify_shard(path, [{'path': 'new_helper.py', 'sha256': 'f' * 64}], 1) == report


@pytest.mark.parametrize('tamper', ['input', 'artifact', 'quota'])
def test_resume_rejects_changed_old_inputs_artifact_or_quota(old_shard, tamper):
    path, _, source, artifact = old_shard
    if tamper == 'input':
        source.write_text('changed original implementation\n')
    elif tamper == 'artifact':
        artifact.write_text('{"row_id":"different"}\n')
    with pytest.raises(ValueError):
        builder.verify_shard(path, [], 2 if tamper == 'quota' else 1)


def test_builder_substitutions_are_local_and_restored_after_failure(monkeypatch):
    previous_generator = builder.original.generator_module
    previous_verify = builder.original.verify_shard

    def fail():
        assert builder.original.generator_module(3) is production
        assert builder.original.verify_shard is builder.verify_shard
        raise RuntimeError('fixture interruption')

    with pytest.raises(RuntimeError, match='fixture interruption'):
        builder._run(fail)
    assert builder.original.generator_module is previous_generator
    assert builder.original.verify_shard is previous_verify
