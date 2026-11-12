"""Exercise the four repaired degree and projection evidence combinations."""

from copy import deepcopy
import json

import pytest

from runtime.world_model import s2_8_case_repairs as degrees
from runtime.world_model import s2_8_corpus, s2_9_corpus
from runtime.world_model import s2_9_case_repairs as graphs
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_audit_cases_7_10 import matches
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


@pytest.fixture(scope='module')
def cases():
    """Provide native wedges, separate edges, and two directed layers per split."""
    pool = [f'ENSG{number:011d}' for number in range(1, 400)]
    groups = {split: [gene for gene in pool if gene_group(gene)[1] == split][:5]
              for split in ('train', 'validation', 'test')}
    undirected, first_directed, second_directed = {}, {}, {}
    for a, b, c, d, e in groups.values():
        undirected.update({tuple(sorted((a, b))): .9, tuple(sorted((a, c))): .7,
                           tuple(sorted((d, e))): .4})
        first_directed.update({(a, b): .8, (b, c): .6, (c, a): .5, (d, e): .3, (e, d): .2})
        second_directed.update({(b, a): .8, (c, b): .6, (a, c): .5, (e, d): .3, (d, e): .2})
    nodes = frozenset(gene for group in groups.values() for gene in group)
    view = GraphView('unit_reviewed_case_repairs', '9' * 64, {
        'HumanNetV3:coexpression': LayerGraph('HumanNetV3:coexpression', False, undirected, nodes),
        'TFs:cd4+t_cells': LayerGraph('TFs:cd4+t_cells', True, first_directed, nodes),
        'TFs:brain': LayerGraph('TFs:brain', True, second_directed, nodes)})
    index = SubgraphIndex(view)
    configurations = {}
    rows = {}
    for module, legacy in ((degrees, s2_8_corpus), (graphs, s2_9_corpus)):
        contract = legacy.load_build_contract()
        bank = load_s2_prompt_banks(contract['prompt_bank'])
        configurations[module.REPAIR] = (module, bank, contract)
        for split in groups:
            selected, _ = module.build_case_rows(view, index, bank, contract, None, split=split)
            rows[split, module.REPAIR] = selected
    return view, index, configurations, rows


def validate(row, cases):
    """Replay through the repair's independent source and visible-proof checks."""
    view, index, configurations, _ = cases
    module, bank, contract = configurations[row['metadata']['repair']]
    module.validate_row(row, view, index=index, bank=bank, contract=contract)


def row_for(cases, case_id, split='test'):
    repair = degrees.REPAIR if case_id.startswith('S2.8.') else graphs.REPAIR
    return deepcopy(next(row for row in cases[3][split, repair]
                         if row['metadata']['catalog_case_id'] == case_id))


def test_every_fixed_split_has_all_four_exact_catalog_combinations(cases):
    for (split, _), rows in cases[3].items():
        for row in rows:
            validate(row, cases)
            meta = row['metadata']
            assert meta['split'] == split
            assert matches(meta['catalog_case_id'], row)
            assert meta['framing'] == ('direct' if meta['catalog_case_id'] in ('S2.8.22', 'S2.9.16') else 'step')


def test_saved_degree_combines_all_three_incompatibilities_in_one_record(cases):
    row = row_for(cases, 'S2.8.22')
    saved, = json.loads(row['messages'][1]['content'])['observations']
    meta = row['metadata']
    assert saved['operation'] == 'get_degree'
    assert saved['arguments']['layer_scope'] != meta['layer_scope']
    assert saved['arguments']['direction'] == 'in' and meta['direction'] == 'out'
    assert saved['provenance']['graph'][0]['graph_version'] != meta['graph_version']
    assert row['messages'][-1]['content']['provenance']['graph'] != saved['provenance']['graph']


def test_exhausted_rank_retains_complete_native_count_without_an_invented_rank(cases):
    row = row_for(cases, 'S2.8.26')
    saved, = json.loads(row['messages'][1]['content'])['observations']
    final = row['messages'][-1]['content']
    assert saved['operation'] == 'get_neighbors'
    assert saved['result']['collection']['complete'] is True
    assert len(saved['result']['neighbors']) == row['metadata']['degree']
    assert final['result']['partial_results'] == [saved]
    assert final['status'] == 'error' and final['continuation_decision'] == 'defer'
    assert 'rank_support' not in final['result'] and 'hub_like' not in final['result']


def test_matching_preview_needs_no_call_and_does_not_invent_a_total(cases):
    row = row_for(cases, 'S2.9.16')
    saved, = json.loads(row['messages'][1]['content'])['observations']
    result = row['messages'][-1]['content']['result']
    assert not any('tool_call' in message for message in row['messages'])
    assert saved['arguments']['top_k'] == row['metadata']['request']['top_k'] == 1
    assert saved['result']['collection']['total_count'] is None
    assert result['collection']['total_count'] is None
    assert result['collection']['complete'] is False
    assert result['edges'] == saved['result']['edges']


def test_exhausted_graph_retains_both_full_native_observations_and_unchecked_input(cases):
    row = row_for(cases, 'S2.9.21')
    saved = json.loads(row['messages'][1]['content'])['observations']
    result = row['messages'][-1]['content']['result']
    assert len(saved) == 2 and all(item['result']['collection']['complete'] for item in saved)
    assert result['partial_results'] == saved
    assert len(result['coverage']['checked_gene_ids']) == 2
    assert len(result['coverage']['unchecked_gene_ids']) == 1
    calls = [message['tool_call'] for message in row['messages'] if 'tool_call' in message]
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize(('case_id', 'mutation'), [
    ('S2.8.22', 'version'), ('S2.8.22', 'scope'), ('S2.8.22', 'direction'),
    ('S2.8.26', 'weight'), ('S2.8.26', 'discard_partial'), ('S2.8.26', 'retry'),
    ('S2.9.16', 'invent_total'), ('S2.9.16', 'different_bound'), ('S2.9.16', 'prefix_weight'),
    ('S2.9.21', 'discard_partial'), ('S2.9.21', 'complete_coverage'), ('S2.9.21', 'retry'),
    ('S2.9.21', 'native_prefix'), ('S2.9.21', 'cross_split'),
])
def test_validator_rejects_changed_source_evidence_selection_and_failure_semantics(cases, case_id, mutation):
    row = row_for(cases, case_id)
    context = json.loads(row['messages'][1]['content'])
    saved = context['observations'][0]
    if mutation == 'version':
        saved['provenance']['graph'][0]['graph_version'] = row['metadata']['graph_version']
    elif mutation == 'scope':
        saved['arguments']['layer_scope'] = deepcopy(row['metadata']['layer_scope'])
    elif mutation == 'direction':
        saved['arguments']['direction'] = 'out'
    elif mutation == 'weight':
        saved['result']['neighbors'][0]['weight'] = 123
    elif mutation == 'discard_partial':
        row['messages'][-1]['content']['result']['partial_results'].pop()
    elif mutation == 'retry':
        row['messages'][5]['tool_call']['arguments']['multiplex_id'] = 'different_scope'
    elif mutation == 'invent_total':
        saved['result']['collection']['total_count'] = row['metadata']['edge_count']
    elif mutation == 'different_bound':
        saved['arguments']['top_k'] = 2
    elif mutation == 'prefix_weight':
        saved['result']['edges'][0]['weight'] = 123
    elif mutation == 'complete_coverage':
        row['messages'][-1]['content']['result']['coverage']['unchecked_gene_ids'] = []
    elif mutation == 'native_prefix':
        saved['result']['collection']['complete'] = False
    elif mutation == 'cross_split':
        row['metadata']['split'] = 'train'
    row['messages'][1]['content'] = json.dumps(context)
    with pytest.raises(ValueError):
        validate(row, cases)


@pytest.mark.parametrize('case_id', [*degrees.CASE_IDS, *graphs.CASE_IDS])
def test_readable_summary_reasoning_and_blocker_wording_are_not_exact_scored(cases, case_id):
    row = row_for(cases, case_id)
    for message in row['messages']:
        if message['role'] == 'assistant':
            message['thinking'] = 'Follow the visible evidence and retain every supported original observation.'
    row['messages'][-1]['content']['summary'] = 'The visible evidence determines this outcome.'
    if row['messages'][-1]['content']['blocked_reason'] is not None:
        row['messages'][-1]['content']['blocked_reason'] = 'A required backend acquisition remains unavailable.'
    validate(row, cases)


def test_selection_and_trace_identity_are_reproducible(cases):
    view, index, configurations, original = cases
    for repair, (module, bank, contract) in configurations.items():
        rows, _ = module.build_case_rows(view, index, bank, contract, None, split='test')
        assert rows == original['test', repair]
