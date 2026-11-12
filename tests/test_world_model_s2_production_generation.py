"""Check native production facts, wording variation, and immutable shard reuse."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from runtime.world_model import s2_1_production as edges, s2_3_corpus as neighbors
from runtime.world_model import s2_production_cases_1_3 as generation
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_corpus import _pair_group
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex
from scripts.build_world_model_s2_production import verify_shard


@pytest.fixture
def native_context():
    """Supply a small independent native ring with held-out gene and pair splits."""
    genes = [f'ENSG{number:011d}' for number in range(1, 81)]
    pairs = {tuple(sorted((gene, genes[(i + 1) % len(genes)]))): .5 for i, gene in enumerate(genes)}
    view = GraphView('production_ring', 'b' * 64, {'HumanNetV3:string_ppi':
        LayerGraph('HumanNetV3:string_ppi', False, pairs, frozenset(genes))})
    contract = edges.load_build_contract()
    return {'view': view, 'index': SubgraphIndex(view), 'registry': contract['_registry'],
        'contract': contract, 'bank': contract['_bank'], 'seed': 20261028}


def test_edge_wording_changes_are_real_unique_traces_with_the_same_source_partition(native_context):
    context = native_context
    layer = context['view'].layer_order[0]
    pair = next(iter(context['view'].layers[layer].edges))
    unit, split = _pair_group(pair, context['seed'])
    fact = edges.CaseFact(layer, pair, split, unit)
    rows = [generation._edge_row('S2.1.1', fact, context, ordinal) for ordinal in range(8)]
    for row in rows:
        generation.validate_row(row, context)
        assert row['metadata']['split_unit'] == unit
        assert row['metadata']['gene_ids'] == list(pair)
    assert len({model_trace_sha256(row) for row in rows}) == 8
    assert len({row['messages'][2]['content'] for row in rows}) == 8


def test_native_source_change_cannot_validate_an_old_edge_trace(native_context):
    context = native_context
    fact = generation._edge_facts('S2.1.1', context, 'train')[0]
    row = generation._edge_row('S2.1.1', fact, context, 0)
    altered = deepcopy(row)
    altered['messages'][-1]['content']['result']['weight'] = 999
    with pytest.raises(ValueError):
        generation.validate_row(altered, context)


def test_neighbor_fact_pool_preserves_gene_splits_and_real_cutoff_ties(native_context):
    context = native_context
    context['contract'] = neighbors.load_build_contract()
    context['bank'] = load_s2_prompt_banks(context['contract']['prompt_bank'])
    for split in ('train', 'validation', 'test'):
        facts = generation._neighbor_facts('S2.3.18', context, split)
        assert facts
        for fact in facts:
            assert neighbors.gene_group(fact.gene_id, context['seed']) == (fact.split_unit, split)
        row = generation._neighbor_row('S2.3.18', facts[0], context, 0)
        generation.validate_row(row, context)
        assert row['messages'][-1]['content']['result']['collection']['complete'] is False


@pytest.mark.parametrize('changed', ['payload', 'quota', 'inputs'])
def test_completed_shards_reject_changed_rows_or_generation_contract(tmp_path, changed):
    payload = b'{"row_id":"source_bound"}\n'
    (tmp_path / 'rows.jsonl').write_bytes(payload)
    pins = [{'path': 'frozen.py', 'sha256': 'a' * 64}]
    report = {'quota': 1, 'input_pins': pins, 'artifact_sha256': hashlib.sha256(payload).hexdigest()}
    (tmp_path / 'report.json').write_text(json.dumps(report))
    assert verify_shard(tmp_path, pins, 1) == report
    if changed == 'payload':
        (tmp_path / 'rows.jsonl').write_bytes(payload + b'{}\n')
    elif changed == 'quota':
        report['quota'] = 2
        (tmp_path / 'report.json').write_text(json.dumps(report))
    else:
        report['input_pins'] = []
        (tmp_path / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_shard(tmp_path, pins, 1)
