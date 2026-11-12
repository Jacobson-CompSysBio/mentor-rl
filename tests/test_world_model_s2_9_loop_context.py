"""Check genuine later-page singleton proofs without inventing or clipping loops."""

from copy import deepcopy
import json

import pytest

from runtime.world_model import s2_9_loop_context as extension
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


LAYER = 'TFs:cd4+t_cells'


@pytest.fixture(scope='module')
def example():
    """Use a synthetic directed loop after one higher-weight native neighbor."""
    genes = [f'ENSG{number:011d}' for number in range(1, 100)]
    gene = next(item for item in genes if gene_group(item)[1] == 'test')
    others = [item for item in genes if item != gene][:3]
    edges = {(gene, others[0]): .9, (gene, gene): .5,
             (gene, others[1]): .4, (gene, others[2]): .1}
    view = GraphView('unit_partial_loop_page', '7' * 64,
                     {LAYER: LayerGraph(LAYER, True, edges, frozenset([gene, *others]))})
    index = SubgraphIndex(view)
    contract = extension.load_build_contract()
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    rows, report = extension.build_rows(view, index, bank, contract)
    return view, index, bank, contract, rows, report


def validate(row, example):
    """Run the independent native replay and visible-evidence proof validator."""
    view, index, bank, contract, _, _ = example
    extension.validate_row(row, view, index=index, bank=bank, contract=contract)


def test_later_original_page_proves_only_the_positive_singleton(example):
    view, _, _, _, rows, report = example
    assert report['selection']['split'] == 'test'
    assert report['selection']['loop_rank'] == 2
    assert report['selection']['native_neighbor_count'] == 4
    assert report['selection']['direction'] == 'out'
    for row in rows:
        validate(row, example)
        context = json.loads(row['messages'][1]['content'])
        observation = context['observations'][0]
        assert observation['arguments']['cursor'] is not None
        assert observation['result']['collection']['next_cursor'] is not None
        assert observation['result']['collection']['complete'] is False
        assert observation['result']['collection']['total_count'] == 4
        assert len(observation['result']['neighbors']) == 1
        assert not any('tool_call' in message for message in row['messages'])
        result = row['messages'][-1]['content']['result']
        gene = row['metadata']['gene_ids'][0]
        assert result['edges'] == [{'gene_ids': [gene, gene], 'weight': view.layers[LAYER].edges[gene, gene]}]
        assert result['collection']['complete'] is True
        assert result['layer_gene_count'] == 1
    assert [row['messages'][-1]['content']['continuation_decision'] for row in rows] == ['stop', 'continue']


@pytest.mark.parametrize('mutation', ['cursor', 'weight', 'complete', 'extra_member', 'page_ordinal', 'wrong_split', 'tool_call'])
def test_validator_rejects_fabricated_pages_or_a_different_proof_scope(example, mutation):
    row = deepcopy(example[4][0])
    context = json.loads(row['messages'][1]['content'])
    observation = context['observations'][0]
    if mutation == 'cursor':
        observation['arguments']['cursor'] = 'invented_cursor'
    elif mutation == 'weight':
        observation['result']['neighbors'][0]['weight'] = 100
    elif mutation == 'complete':
        observation['result']['collection']['complete'] = True
    elif mutation == 'extra_member':
        row['metadata']['gene_ids'].append(next(gene for gene in example[0].multiplex_nodes
                                                if gene != row['metadata']['gene_ids'][0]))
    elif mutation == 'page_ordinal':
        row['metadata']['native_page_ordinal'] -= 1
    elif mutation == 'wrong_split':
        row['metadata']['split'] = 'train'
    elif mutation == 'tool_call':
        row['messages'].insert(3, {'role': 'assistant', 'thinking': 'Unnecessary acquisition.',
                                  'tool_call': {'name': 'induce_subgraph', 'arguments': {}}})
        row['messages'].insert(4, {'role': 'tool', 'name': 'induce_subgraph', 'content': {}})
    row['messages'][1]['content'] = json.dumps(context)
    with pytest.raises(ValueError):
        validate(row, example)


def test_complete_incoming_list_cannot_replace_an_incomplete_loop_page(example):
    view, index, _, _, _, report = example
    incoming = [item for item in report['native_loop_direction_inventory'] if item['direction'] == 'in']
    assert len(incoming) == 1 and incoming[0]['native_neighbor_count'] == 1
    assert incoming[0]['has_later_page'] is False
    gene = report['selection']['gene_id']
    complete_only = GraphView('unit_complete_loop_only', '8' * 64,
                              {LAYER: LayerGraph(LAYER, True, {(gene, gene): .5}, frozenset([gene]))})
    with pytest.raises(VariantUnavailable, match='no native loop'):
        extension.select_fact(complete_only, SubgraphIndex(complete_only), split='test')


def test_summary_and_reasoning_wording_are_not_exact_match_targets(example):
    row = deepcopy(example[4][0])
    row['messages'][-1]['thinking'] = 'A visible positive loop settles the only selected self-pair.'
    row['messages'][-1]['content']['summary'] = 'The singleton native projection is complete.'
    validate(row, example)
