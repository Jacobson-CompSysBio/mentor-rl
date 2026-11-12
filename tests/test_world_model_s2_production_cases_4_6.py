"""Regress production-only source selection and independent frozen wording."""

from copy import deepcopy
from itertools import islice
import json
from pathlib import Path

import pytest

from runtime.world_model import s2_production_cases_4_6 as production
from runtime.world_model import s2_4_case_repairs, s2_5_corpus, s2_6_corpus
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_corpus import canonical_json
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def contexts():
    """Supply real small adjacency and endpoint inventories in every fixed split."""
    genes = [f'ENSG{index:011d}' for index in range(1, 161)]
    layers = {}
    for number, (layer, directed) in enumerate((('HumanNetV3:string_ppi', False),
            ('HumanNetV3:coexpression', False), ('TFs:brain', True), ('TFs:cd4+t_cells', True))):
        members = genes[number * 10:100 + number * 10]
        edges = {(members[index], members[(index + 1) % 80]): .5 for index in range(80)}
        if not directed:
            edges = {tuple(sorted(pair)): value for pair, value in edges.items()}
        if directed:
            edges.update({(members[0], gene): .25 for gene in members[80:]})
        endpoints = frozenset(gene for edge in edges for gene in edge)
        layers[layer] = LayerGraph(layer, directed, edges, endpoints)
    view = GraphView('production_unit', 'd' * 64, layers)
    registry = RegistryContext('110', 'ensembl_homo_sapiens_110_production_unit',
        {gene: f'UNIT_{index}' for index, gene in enumerate(genes)})
    bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
    index = SubgraphIndex(view)
    return {family: {'view': view, 'index': index, 'registry': registry, 'bank': bank,
                     'contract': module.load_build_contract(), 'seed': 20261028}
            for family, module in ((4, s2_4_case_repairs), (5, s2_5_corpus), (6, s2_6_corpus))}


@pytest.mark.parametrize('case_id', ['S2.6.19', 'S2.6.24'])
def test_empty_and_null_cases_reach_512_by_distinct_frozen_visible_wording(contexts, case_id):
    context = contexts[6]
    rows = list(production.production_candidates(case_id, context, split='train'))
    assert len(rows) == len({model_trace_sha256(row) for row in rows}) == 512
    assert {row['metadata']['wrapper_index'] for row in rows} == set(range(32))
    assert {row['metadata']['request_template_index'] for row in rows} == set(range(4))
    assert len({row['metadata']['layer_id'] for row in rows}) == 4
    for row in rows[::32]:
        production.validate_row(row, context)
    row = rows[0]
    assert row['metadata']['input_genes'] == ([] if case_id == 'S2.6.19' else None)
    assert not any('tool_call' in message for message in row['messages'])


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_unknown_layer_diagnostic_rewording_preserves_the_literal_rejected_scope(contexts, split):
    context = contexts[6]
    row = next(production.production_candidates('S2.6.18', context, split=split))
    production.validate_row(row, context)
    literal = row['messages'][-1]['content']['result']['request']
    assert literal['layer_id'] == 'TFs:missing_context'
    assert literal['layer_id'] in row['messages'][2]['content']
    assert row['metadata']['layer_id'] != literal['layer_id']


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_zero_outgoing_singleton_keeps_positive_inventory_evidence(contexts, split):
    context = contexts[6]
    row = next(production.production_candidates('S2.6.16', context, split=split))
    production.validate_row(row, context)
    assert row['metadata']['gene_ids'] == [row['metadata']['gene_id']]
    saved = json.loads(row['messages'][1]['content'])['observations'][0]
    assert saved['operation'] == 'get_neighbors'
    assert saved['arguments']['direction'] == 'out'
    assert saved['result']['neighbors'] == []
    assert saved['result']['not_in_layer_gene_ids'] == []
    assert not any('tool_call' in message for message in row['messages'])
    corrupt = deepcopy(row)
    corrupt['messages'][1]['content'] = canonical_json({'active_graph': json.loads(
        row['messages'][1]['content'])['active_graph'], 'observations': []})
    with pytest.raises(ValueError):
        production.validate_row(corrupt, context)


def test_complete_negative_and_incomplete_omission_follow_different_acquisition_rules(contexts):
    context = contexts[6]
    negative = next(production.production_candidates('S2.6.20', context, split='train'))
    positive = next(production.production_candidates('S2.6.21', context, split='train'))
    for row in (negative, positive):
        production.validate_row(row, context)
    assert not any('tool_call' in message for message in negative['messages'])
    assert any('tool_call' in message for message in positive['messages'])
    assert negative['messages'][-1]['content']['result']['layer_gene_ids'] == []
    assert positive['messages'][-1]['content']['result']['layer_gene_ids'] == positive['metadata']['gene_ids']


def test_global_negative_is_a_completed_all_layer_membership_answer(contexts):
    context = contexts[5]
    row = next(production.production_candidates('S2.5.20', context, split='train'))
    production.validate_row(row, context)
    final = row['messages'][-1]['content']
    assert final['status'] == 'resolved' and final['continuation_decision'] == 'stop'
    assert final['result']['present'] is False
    assert not any('tool_call' in message for message in row['messages'])


def test_independent_wrapper_rebinding_retains_two_native_failure_observations(contexts):
    context = contexts[4]
    rows = list(islice(production.production_candidates('S2.4.20', context, split='train'), 33))
    assert len({model_trace_sha256(row) for row in rows}) == 33
    for row in (rows[0], rows[1], rows[32]):
        production.validate_row(row, context)
        saved = json.loads(row['messages'][1]['content'])['observations']
        assert [item['arguments']['direction'] for item in saved] == ['undirected', 'in']
        assert row['messages'][-1]['content']['result']['partial_results'] == saved
        assert row['messages'][-1]['content']['result']['arguments']['direction'] == 'out'
    assert rows[0]['metadata']['wrapper_index'] == rows[32]['metadata']['wrapper_index'] == 0
    assert rows[0]['metadata']['request_template_index'] == 0
    assert rows[32]['metadata']['request_template_index'] == 1


def test_catalog_has_no_unsupported_family_or_split(contexts):
    assert len(production.CASE_IDS) == 75
    for case, split in (('S2.7.1', 'train'), ('S2.6.1', 'invented')):
        with pytest.raises(ValueError):
            next(production.production_candidates(case, contexts[6], split=split))
