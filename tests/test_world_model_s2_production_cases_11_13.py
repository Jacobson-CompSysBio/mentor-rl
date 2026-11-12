"""Check independent wording capacity and exact native multi-context evidence."""

from copy import deepcopy

import pytest

from runtime.world_model import s2_production_cases_11_13 as production
from runtime.world_model import s2_13_corpus as legacy
from runtime.world_model.s1_prompt_banks import load_s2_prompt_banks
from runtime.world_model.s2_13_sampling import fact_from_pair
from runtime.world_model.s2_audit_identity import model_trace_sha256
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex
from types import SimpleNamespace


@pytest.fixture(scope='module')
def context():
    """Use two real native PEN records and separate directional TF observations."""
    first, second = 'ENSG00000000001', 'ENSG00000000002'
    layers = {
        'HumanNetV3:string_ppi': LayerGraph('HumanNetV3:string_ppi', False,
            {(first, second): .6}, frozenset((first, second))),
        'bulkPEN:test:tissue': LayerGraph('bulkPEN:test:tissue', False,
            {(first, second): .7}, frozenset((first, second))),
        'scPEN:test:region:cell': LayerGraph('scPEN:test:region:cell', False,
            {(first, second): .8}, frozenset((first, second))),
        'TFs:brain': LayerGraph('TFs:brain', True,
            {(second, first): .9, (first, first): .2}, frozenset((first, second))),
    }
    view = GraphView('production_qualification_fixture', 'f' * 64, layers)
    contract = legacy.load_build_contract()
    return {'view': view, 'index': SubgraphIndex(view), 'contract': contract,
        'bank': load_s2_prompt_banks(contract['prompt_bank']),
        'registry': SimpleNamespace(symbols_by_id={}, gene_ids_by_symbol={}),
        'old_view': view, 'seed': 20261028}


@pytest.mark.parametrize(('split', 'quota'), [('train', 512), ('validation', 64), ('test', 64)])
def test_pure_family_case_has_enough_actual_distinct_wordings(context, split, quota):
    rows = list(production.production_candidates('S2.13.1', context, split=split))
    assert len(rows) == quota
    assert len({model_trace_sha256(row) for row in rows}) == quota
    assert len({row['messages'][2]['content'] for row in rows}) == quota
    assert {row['metadata']['wrapper_index'] for row in rows} == set(range(32 if split == 'train' else 8))
    assert {row['metadata']['request_template_index'] for row in rows} == set(range(16 if split == 'train' else 8))
    for row in rows[:2]:
        production.validate_row(row, context)


def selected_row(context, case_id):
    """Construct the exact evidence-only task without relying on historical selections."""
    prepared = production.prepare_context(context, 13)
    fact = fact_from_pair(context['view'], context['index'], context['registry'], context['old_view'],
        'TFs:brain' if case_id in ('S2.13.12', 'S2.13.26') else 'bulkPEN:test:tissue',
        ('ENSG00000000001', 'ENSG00000000002'), context['seed'])
    from scripts.audit_world_model_s2_case_quotas import catalog_inputs
    case = next(c for c in catalog_inputs()[0] if c.case_id == case_id)
    framing = case.framing_hint or ('step' if case.continuation_decision == 'continue' else 'direct')
    return production._build(prepared, case_id, fact, framing, 2, 3)


@pytest.mark.parametrize('case_id', ['S2.13.18', 'S2.13.21'])
def test_two_pen_contexts_retain_native_weight_meanings_and_both_cards(context, case_id):
    row = selected_row(context, case_id)
    production.validate_row(row, context)
    assert len(row['messages']) == 4
    cards = row['messages'][-1]['content']['result']['claim_cards']
    assert len(cards) == 2
    assert {card['subject']['layer_family'] for card in cards} == {'bulk_pen', 'scpen'}


@pytest.mark.parametrize('mutation', ['weight', 'row_id', 'seed', 'provenance'])
def test_two_pen_extension_rejects_forged_evidence_and_identity(context, mutation):
    import json
    row = selected_row(context, 'S2.13.18')
    if mutation == 'weight':
        harness = json.loads(row['messages'][1]['content'])
        harness['observations'][0]['result']['weight'] = 999
        row['messages'][1]['content'] = json.dumps(harness)
    elif mutation == 'row_id':
        row['row_id'] = 'arbitrary'
    elif mutation == 'seed':
        row['metadata']['selection_seed'] += 1
    else:
        row['messages'][-1]['content']['provenance']['graph'][0]['evidence_id'] = 'arbitrary'
    with pytest.raises(ValueError):
        production.validate_row(row, context)


def test_summary_and_reasoning_are_not_scored_as_exact_text(context):
    row = selected_row(context, 'S2.13.18')
    row['messages'][-1]['content']['summary'] = 'Both native PEN observations remain distinct.'
    row['messages'][-1]['thinking'] = 'Use each supplied source only for its own context.'
    production.validate_row(row, context)


def test_reverse_evidence_returns_both_requested_orientation_cards(context):
    row = selected_row(context, 'S2.13.26')
    production.validate_row(row, context)
    cards = row['messages'][-1]['content']['result']['claim_cards']
    assert len(cards) == 2
    assert {card['claim_supported'] for card in cards} == {True, False}


def test_reverse_positive_requires_the_separate_negative_orientation_acquisition(context):
    row = selected_row(context, 'S2.13.12')
    production.validate_row(row, context)
    calls = [message['tool_call'] for message in row['messages'] if 'tool_call' in message]
    assert len(calls) == 1
    assert calls[0]['name'] == 'get_edge'
    assert calls[0]['arguments']['gene_ids'] == ['ENSG00000000001', 'ENSG00000000002']
    assert row['messages'][-1]['content']['result']['edge_exists'] is False


def test_production_request_overlay_leaves_legacy_bank_unchanged(context):
    original = deepcopy(context['contract']['_requests'])
    prepared = production.prepare_context(context, 13)
    assert context['contract']['_requests'] == original
    assert len(prepared['contract']['_requests']['train']['family_card']) == 16
    assert len(original['train']['family_card']) == 4


def test_scarce_loop_path_accepts_the_last_new_wording_without_semantic_changes(context):
    from runtime.world_model import s2_11_corpus
    from runtime.world_model.s2_11_sampling import fact_from_genes
    selected = {**context, 'contract': s2_11_corpus.load_build_contract()}
    prepared = production.prepare_context(selected, 11)
    fact = fact_from_genes(prepared['view'], prepared['index'], 'TFs:brain',
        ['ENSG00000000001', 'ENSG00000000001'], prepared['seed'])
    row = production._build(prepared, 'S2.11.15', fact, 'direct',
        len(prepared['bank'][fact.split]['direct']) - 1,
        len(prepared['contract']['_requests'][fact.split]['path']) - 1)
    production.validate_row(row, selected)
    assert row['messages'][-1]['content']['result']['hop_count'] == 0
    assert len(row['messages']) == 4


def test_new_incoming_loop_bank_retains_all_original_entries_and_bindings():
    import json
    from runtime.world_model.s2_3_prompts import validate_s2_3_requests
    from runtime.world_model.s2_corpus import ROOT
    old = json.loads((ROOT / 'runtime/world_model/s2_3_v1_requests.json').read_text())
    new = json.loads((ROOT / 'runtime/world_model/s2_production_3_requests_v1.json').read_text())
    validate_s2_3_requests(new)
    for split, count in (('train', 16), ('validation', 8), ('test', 8)):
        assert new[split]['complete']['in'][:4] == old[split]['complete']['in']
        assert len(new[split]['complete']['in']) == count
