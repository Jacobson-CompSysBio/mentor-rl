"""Verify the exact unavailable-kind and retained-native retry combinations."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_10_case_repairs import CASE_IDS, _visible_plan, build_case_rows, validate_row
from runtime.world_model.s2_10_corpus import _arguments, build_diagnostic, load_build_contract
from runtime.world_model.s2_10_sampling import fact_from_genes
from runtime.world_model.s2_corpus import SPLITS
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


UNDIRECTED, DIRECTED = 'HumanNetV3:coexpression', 'TFs:brain'


@pytest.fixture(scope='module')
def repaired():
    """Use actual split hashing and independent native pairs in all three splits."""
    pools = {split: [] for split in SPLITS}
    for number in range(1, 1000):
        gene = f'ENSG{number:011d}'
        split = gene_group(gene)[1]
        if len(pools[split]) < 4:
            pools[split].append(gene)
    undirected, directed = {}, {}
    for a, b, c, d in pools.values():
        undirected[tuple(sorted((a, b)))] = 0
        undirected[tuple(sorted((c, d)))] = .7
        directed.update({(a, b): .1, (b, c): .2, (c, d): .3})
    nodes = frozenset(gene for pool in pools.values() for gene in pool)
    view = GraphView('repair_unit', 'f' * 64, {
        UNDIRECTED: LayerGraph(UNDIRECTED, False, undirected, nodes),
        DIRECTED: LayerGraph(DIRECTED, True, directed, nodes)})
    index = SubgraphIndex(view)
    contract = load_build_contract()
    bank = load_s2_prompt_banks(contract['prompt_bank'])
    rows = {split: build_case_rows(view, index, bank, contract, None, split=split)[0] for split in SPLITS}
    return view, index, bank, contract, rows


def verify(row, repaired):
    """Replay a candidate against the unchanged fixture and request banks."""
    view, index, bank, contract, _ = repaired
    validate_row(row, view, index=index, bank=bank, contract=contract)


@pytest.mark.parametrize('split', SPLITS)
def test_both_cases_are_deterministic_and_roundtrip_as_json(repaired, split):
    view, index, bank, contract, rows = repaired
    built, report = build_case_rows(view, index, bank, contract, None, split=split)
    assert built == rows[split]
    assert [row['metadata']['catalog_case_id'] for row in built] == list(CASE_IDS)
    assert report['candidate_native_pairs_examined'] == 2
    for row in built:
        assert all(gene_group(gene)[1] == split for gene in row['metadata']['gene_ids'])
        verify(json.loads(json.dumps(row)), repaired)


def test_local_rejection_preserves_undirected_kind_on_native_directed_layer(repaired):
    row = repaired[-1]['train'][0]
    assert row['metadata']['framing'] == 'direct'
    assert row['metadata']['layer_id'] == DIRECTED
    assert not any('tool_call' in message for message in row['messages'])
    final = row['messages'][-1]['content']
    assert final['status'] == 'invalid_input'
    assert final['result']['issue_code'] == 'CONNECTIVITY_KIND_UNAVAILABLE'
    assert final['result']['request']['connectivity_kind'] == 'undirected'
    assert final['result']['coverage'] is None
    assert final['result']['partial_results'] == []
    assert final['provenance'] == {'registry': [], 'graph': []}


def test_retry_exhaustion_retains_complete_lists_and_only_two_checked_memberships(repaired):
    row = repaired[-1]['train'][1]
    context = json.loads(row['messages'][1]['content'])
    saved = context['observations']
    calls = [message['tool_call'] for message in row['messages'] if 'tool_call' in message]
    assert row['metadata']['framing'] == 'step'
    assert 'acquisition_policy' not in context
    assert len(saved) == 2 and all(item['result']['collection']['complete'] for item in saved)
    assert calls[0] == calls[1]
    assert calls[0]['name'] == 'get_nodes_by_layer'
    assert calls[0]['arguments']['gene_ids'] == row['metadata']['gene_ids'][2:]
    final = row['messages'][-1]['content']
    assert final['result']['partial_results'] == saved
    assert final['result']['coverage']['checked_gene_ids'] == row['metadata']['gene_ids'][:2]
    assert final['result']['coverage']['unchecked_gene_ids'] == row['metadata']['gene_ids'][2:]
    assert 'components' not in final['result']
    assert len(final['provenance']['graph']) == 2


def test_visible_planner_has_no_access_to_source_inventory_or_adjacency(repaired):
    view, index, _, _, rows = repaired
    row = rows['test'][1]
    meta = row['metadata']
    fact = fact_from_genes(view, index, meta['layer_id'], meta['gene_ids'], 'undirected',
                           split='test', category=meta['category'])
    visible = SimpleNamespace(multiplex_id=view.multiplex_id, graph_version=view.graph_version,
        layer_order=view.layer_order, layers={layer: SimpleNamespace(directed=view.layers[layer].directed)
                                             for layer in view.layer_order})
    proof, used = _visible_plan(visible, _arguments(view, fact, 'undirected'),
                                json.loads(row['messages'][1]['content'])['observations'])
    assert proof.result is None and proof.next_tool == 'get_nodes_by_layer'
    assert proof.coverage['unchecked_gene_ids'] == meta['gene_ids'][2:]
    assert len(used) == 2


@pytest.mark.parametrize('mutation', ['weight', 'incomplete', 'drop_saved', 'checked', 'partial',
                                    'retry_budget', 'retry_arguments', 'provenance', 'split',
                                    'wording', 'case', 'extra', 'row_id'])
def test_mutations_of_source_proof_faults_and_wire_are_rejected(repaired, mutation):
    row = deepcopy(repaired[-1]['train'][1])
    context = json.loads(row['messages'][1]['content'])
    final = row['messages'][-1]['content']
    if mutation == 'weight':
        context['observations'][0]['result']['neighbors'][0]['weight'] = 99
    elif mutation == 'incomplete':
        context['observations'][0]['result']['collection']['complete'] = False
    elif mutation == 'drop_saved':
        context['observations'].pop()
    elif mutation == 'checked':
        final['result']['coverage']['checked_gene_ids'] = row['metadata']['gene_ids']
        final['result']['coverage']['unchecked_gene_ids'] = []
    elif mutation == 'partial':
        final['result']['partial_results'].pop()
    elif mutation == 'retry_budget':
        row['messages'][4]['content']['message'] = 'Backend failed. retries_remaining=0.'
    elif mutation == 'retry_arguments':
        row['messages'][5]['tool_call']['arguments']['gene_ids'] = row['metadata']['gene_ids'][:1]
    elif mutation == 'provenance':
        final['provenance']['graph'].pop()
    elif mutation == 'split':
        row['metadata']['split'] = 'test'
    elif mutation == 'wording':
        row['messages'][2]['content'] += ' And guess the missing membership.'
    elif mutation == 'case':
        row['metadata']['catalog_case_id'] = CASE_IDS[0]
    elif mutation == 'extra':
        row['messages'][3]['invented'] = True
    elif mutation == 'row_id':
        row['row_id'] += '_changed'
    row['messages'][1]['content'] = json.dumps(context)
    with pytest.raises(ValueError):
        verify(row, repaired)


def test_the_inverse_unavailable_kind_diagnostic_cannot_claim_case17(repaired):
    view, index, bank, contract, rows = repaired
    genes = rows['train'][1]['metadata']['gene_ids']
    fact = fact_from_genes(view, index, UNDIRECTED, genes, 'undirected',
                           category='directed_unavailable_kind')
    row = build_diagnostic(view, fact, 'direct', 0, bank, contract, kind='unavailable_kind')
    row['metadata'].update(repair='s2_10_case_repairs_v1', catalog_case_id=CASE_IDS[0])
    with pytest.raises(ValueError):
        verify(row, repaired)


def test_free_prose_is_not_exact_scored(repaired):
    row = deepcopy(repaired[-1]['validation'][1])
    row['messages'][-1]['content']['summary'] = 'The supported pair is retained; membership remains unresolved.'
    row['messages'][-1]['content']['blocked_reason'] = 'The membership backend must recover.'
    for message in row['messages']:
        if message['role'] == 'assistant':
            message['thinking'] = 'Visible evidence supports the retained partials and the deferred decision.'
    verify(row, repaired)
