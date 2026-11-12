"""Regress full-bank wrapper bindings for the bounded strongest diagnostic."""

from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

import pytest

from runtime.world_model import s2_production_case_4_22 as production
from runtime.world_model import s2_4_case_repairs as repair
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_corpus import canonical_json
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex
from scripts import build_world_model_s2_production_v2 as builder


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def context():
    """Use actual small undirected evidence in every fixed subject split."""
    genes = [f'ENSG{number:011d}' for number in range(1, 161)]
    edges = {tuple(sorted((gene, genes[(number + 1) % len(genes)]))): .5 for number, gene in enumerate(genes)}
    view = GraphView('strongest_bound_fixture', 'c' * 64, {
        layer: LayerGraph(layer, False, edges, frozenset(genes))
        for layer in ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression')})
    return {'view': view, 'index': SubgraphIndex(view), 'contract': repair.load_build_contract(), 'seed': 20261028,
        'bank': json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text()),
        'registry': RegistryContext('110', 'ensembl_homo_sapiens_110_strongest_fixture',
                                   {gene: f'BOUND_{number}' for number, gene in enumerate(genes)})}


@pytest.mark.parametrize(('split', 'quota'), [('train', 512), ('validation', 64), ('test', 64)])
def test_full_quota_has_independent_actual_frozen_wording_and_validates(context, split, quota):
    rows = list(islice(production.production_candidates(production.CASE_ID, context, split=split), quota))
    assert len(rows) == len({model_trace_sha256(row) for row in rows}) == quota
    assert len({row['metadata']['wrapper_index'] for row in rows}) == (32 if split == 'train' else 8)
    assert {row['metadata']['request_template_index'] for row in rows} == set(range(4))
    for row in rows:
        before = deepcopy(row)
        production.validate_row(row, context)
        assert row == before
        final = row['messages'][-1]['content']
        assert final['status'] == 'ambiguous' and final['continuation_decision'] == 'defer'
        assert final['result']['request']['mode'] == 'top_k'
        assert final['result']['request']['top_k'] == 2
        assert len(row['messages']) == 4


@pytest.mark.parametrize('mutation', ['wrapper', 'core', 'bound', 'new_call', 'adapter'])
def test_changed_full_bank_binding_or_bound_is_rejected(context, mutation):
    row = next(production.production_candidates(production.CASE_ID, context, split='train'))
    if mutation == 'wrapper':
        row['metadata']['wrapper_index'] = 7
    elif mutation == 'core':
        row['metadata']['request_template_index'] = 2
    elif mutation == 'bound':
        row['messages'][-1]['content']['result']['request']['top_k'] = 3
    elif mutation == 'adapter':
        row['metadata'].pop('production_extension')
    else:
        row['messages'].insert(3, {'role': 'assistant', 'thinking': 'Acquire a graph record.',
                                  'tool_call': {'name': 'get_neighbor_layers', 'arguments': {}}})
    with pytest.raises(ValueError):
        production.validate_row(row, context)


def test_other_four_to_six_cases_keep_the_frozen_production_generator(context):
    expected = list(islice(production.previous.production_candidates('S2.4.14', context, split='train'), 2))
    actual = list(islice(production.production_candidates('S2.4.14', context, split='train'), 2))
    assert canonical_json(actual) == canonical_json(expected)
    for row in actual:
        production.validate_row(row, context)


def test_v2_selects_only_the_two_isolated_family_extensions():
    assert builder.generator_module(3).__name__.endswith('s2_production_case_3_19')
    assert builder.generator_module(4) is production
    assert builder.generator_module(5).__name__.endswith('s2_production_cases_4_6')
