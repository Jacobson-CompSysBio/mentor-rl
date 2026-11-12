"""Check ordered-pair preservation and source-backed evidence panel selection."""

from types import SimpleNamespace
import json

from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_11_export import (
    _claim_questions, _fact_key, _native_cost, _semantic_panel, _zero_weight_availability,
)
from runtime.world_model.s2_11_sampling import fact_from_genes
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY
from runtime.world_model import s2_11_corpus as corpus
from runtime.world_model.s2_paths import PathIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


SEED = 20261028
PPI, BRAIN, COEX = 'HumanNetV3:string_ppi', 'TFs:brain', 'HumanNetV3:coexpression'


def source_cases():
    """Use a diamond, a sink, a loop, and separate endpoint membership outcomes."""
    genes = [f'ENSG{number:011d}' for number in range(1, 200)]
    a, b, c, d, isolated, outside, missing, second_missing = [gene for gene in genes if gene_group(gene, SEED)[1] == 'train'][:8]
    view = GraphView('unit_path_export', 'a' * 64, {
        PPI: LayerGraph(PPI, False, {(a, a): .1, (a, b): .2, (a, c): .3, (b, d): 0, (c, d): .4},
                        frozenset((a, b, c, d, isolated))),
        BRAIN: LayerGraph(BRAIN, True, {(a, a): .1, (a, b): .2, (a, c): .3, (b, d): 0,
                                      (c, d): .4, (outside, outside): .5},
                          frozenset((a, b, c, d, isolated, outside))),
        COEX: LayerGraph(COEX, False, {(a, d): 0}, frozenset((a, d))),
    })
    index = PathIndex(view)
    pairs = [(a, a), (b, b), (a, d), (d, a), (b, d), (a, isolated), (isolated, a),
             (a, outside), (missing, a), (outside, missing), (outside, outside), (missing, missing),
             (missing, second_missing)]
    facts = [fact_from_genes(view, index, layer, pair, SEED, split='train')
             for layer in view.layers for pair in pairs]
    registry = SimpleNamespace(symbols_by_id={gene: gene for gene in (a, c, d, isolated, outside, missing)})
    return view, index, facts, registry


def test_fact_identity_keeps_reverse_paths_separate_from_the_shared_split_unit():
    _, _, facts, _ = source_cases()
    forward = facts[2]
    reverse = facts[3]
    assert forward.split_unit == reverse.split_unit
    assert _fact_key(forward) != _fact_key(reverse)
    assert forward.path_gene_ids != reverse.path_gene_ids


def test_semantic_panel_retains_attainable_direction_and_membership_outcomes():
    view, index, facts, registry = source_cases()
    panel = _semantic_panel(view, index, facts, registry)
    selected = [fact for fact in facts if _fact_key(fact) in panel]
    outcomes = lambda fact: (fact.direction, fact.path_found, fact.hop_count if fact.hop_count in (0, 1, 2) else None)
    assert {outcomes(fact) for fact in selected} == {outcomes(fact) for fact in facts}
    assert {tuple(bool(count) for count in fact.category_counts) for fact in selected} == {
        tuple(bool(count) for count in fact.category_counts) for fact in facts}
    assert any((fact.tied_shortest_paths or 0) > 1 for fact in selected)
    assert any(fact.gene_ids[0] == fact.gene_ids[1] and fact.category_counts[1] for fact in selected)
    assert any(fact.gene_ids[0] == fact.gene_ids[1] and fact.category_counts[2] for fact in selected)
    assert any(fact.category_counts[2] > 1 for fact in selected)
    assert _semantic_panel(view, index, list(reversed(facts)), registry) == panel


def test_closed_component_cost_counts_full_incidence_instead_of_only_endpoints():
    view, index, facts, _ = source_cases()
    disconnected = facts[5]
    assert disconnected.path_found is False and not any(disconnected.category_counts[1:])
    component = index.distances(PPI, disconnected.gene_ids[0])
    expected = sum(len(index.native_index.neighbors(PPI, gene, 'undirected')) for gene in component)
    assert _native_cost(view, index, disconnected) == expected
    assert expected > len(index.native_index.neighbors(PPI, disconnected.gene_ids[0], 'undirected'))


def test_actual_zero_weight_paths_are_counted_without_treating_weights_as_distances():
    view, _, facts, _ = source_cases()
    availability = _zero_weight_availability(view, SEED)
    assert all(item['native_zero_weight_edges'] == item['distinct_endpoint_zero_edges'] == 1
               and item['zero_weight_self_loops'] == 0
               and item['eligible_one_hop_edges_by_split'] == {'train': 1, 'validation': 0, 'test': 0}
               for item in availability.values())
    assert any(fact.hop_count == 1 for fact in facts)


def test_claim_tasks_span_the_closed_vocabulary_without_treating_unsupported_as_false():
    view, _, facts, _ = source_cases()
    rows = _claim_questions(facts[2], view, CLAIM_VOCABULARY)
    for kind in ('path_support', 'claim_support'):
        assert {task['claim'] for _, task in rows if task['question_kind'] == kind} == set(CLAIM_VOCABULARY)
    truths = [task for _, task in rows if task['question_kind'] in ('path_truth', 'truth')]
    assert truths and all(task['statement'] for task in truths)
    assert {task['truth_scope'] for task in truths} == {'scope', 'all_layers'}
    assert all(task['membership_cards'] is True for _, task in rows)


def test_zero_hops_do_not_create_an_unchecked_self_loop_truth_question():
    view, _, facts, _ = source_cases()
    zero = facts[0]
    assert zero.hop_count == 0
    rows = _claim_questions(zero, view, CLAIM_VOCABULARY)
    assert not any(task['question_kind'] in ('path_truth', 'truth')
                   and task['claim'] == 'no_edge_recorded_in_scope' for _, task in rows)
    assert any(task['question_kind'] == 'claim_support'
               and task['claim'] == 'no_edge_recorded_in_scope' for _, task in rows)


def test_missing_multiplex_endpoint_leaves_truth_unassessed():
    view, _, facts, _ = source_cases()
    missing = facts[8]
    assert missing.path_found is None
    rows = _claim_questions(missing, view, CLAIM_VOCABULARY)
    assert not any(task['question_kind'] in ('path_truth', 'truth') for _, task in rows)
    assert any(task['question_kind'] == 'claim_card' for _, task in rows)


def claim_row(view, index, fact, variant, question, claim, *, truth_scope=None):
    """Replay a selected source-panel task with its supplied evidence."""
    tasks = _claim_questions(fact, view, CLAIM_VOCABULARY)
    task = next(task for selected, task in tasks if selected == variant
        and task['question_kind'] == question and task.get('claim') == claim
        and task.get('truth_scope') == truth_scope)
    contract = corpus.load_build_contract()
    bank = json.loads((corpus.ROOT / contract['prompt_bank']['path']).read_text())
    row = corpus.build_row(view, fact, 'direct', variant, 0, bank, contract,
                           index=index, **task)
    corpus.validate_row(row, view, bank=bank, contract=contract, index=index)
    calls = [message['tool_call'] for message in row['messages'] if 'tool_call' in message]
    assert calls == []
    return row['messages'][-1]['content']


def test_negative_edge_and_arbitrary_route_teach_different_absence_permissions():
    view, index, facts, _ = source_cases()
    fact = facts[2]
    assert fact.hop_count == 2
    edge = claim_row(view, index, fact, 'edge_context', 'claim_support', 'no_edge_recorded_in_scope')
    route = claim_row(view, index, fact, 'route_context', 'claim_support', 'no_edge_recorded_in_scope')
    assert edge['result']['claim_cards'][0]['claim_supported'] is True
    assert route['result']['claim_cards'][0]['claim_supported'] is False
    assert 'path_found' not in edge['result'] and 'path_found' not in route['result']


def test_zero_hop_membership_task_does_not_assess_the_unchecked_loop():
    view, index, facts, _ = source_cases()
    fact = facts[1]
    answer = claim_row(view, index, fact, 'membership_context', 'path_support', 'no_edge_recorded_in_scope')
    assert answer['result']['hop_count'] == 0
    assert answer['result']['path_gene_ids'] == [fact.gene_ids[0]]
    assert answer['result']['claim_cards'][0]['claim_supported'] is False


def test_complete_two_hop_graph_proves_edge_absence_but_not_the_global_tie_winner():
    view, index, facts, _ = source_cases()
    fact = facts[2]
    answer = claim_row(view, index, fact, 'tie_route_context', 'claim_support', 'no_edge_recorded_in_scope')
    assert answer['result']['claim_cards'][0]['claim_supported'] is True
    contract = corpus.load_build_contract()
    bank = json.loads((corpus.ROOT / contract['prompt_bank']['path']).read_text())
    row = corpus.build_row(view, fact, 'direct', 'tie_route_context', 0,
                           bank, contract, index=index)
    corpus.validate_row(row, view, bank=bank, contract=contract, index=index)
    context = json.loads(row['messages'][1]['content'])
    supplied = context['observations']
    assert len(supplied) == 1 and supplied[0]['operation'] == 'induce_subgraph'
    assert supplied[0]['result']['collection']['complete'] is True
    assert len(supplied[0]['result']['layer_gene_ids']) == 3
    assert not any(set(edge['gene_ids']) == set(fact.gene_ids) for edge in supplied[0]['result']['edges'])
    calls = [message['tool_call']['name'] for message in row['messages'] if 'tool_call' in message]
    assert calls == ['shortest_path']


def test_reverse_minimum_length_supports_absence_without_a_reverse_tie_witness():
    view, index, facts, _ = source_cases()
    fact = facts[2]
    support = claim_row(view, index, fact, 'reversed_context', 'claim_support', 'no_edge_recorded_in_scope')
    truth = claim_row(view, index, fact, 'reversed_context', 'truth', 'no_edge_recorded_in_scope', truth_scope='scope')
    assert support['result']['claim_cards'][0]['claim_supported'] is True
    assert truth['result']['claim_cards'][0]['statement_true'] is True
    assert 'path_gene_ids' not in truth['result']


def test_a_path_in_another_layer_refutes_the_explicit_universal_statement():
    view, index, facts, _ = source_cases()
    answer = claim_row(view, index, facts[2], 'wrong_layer', 'truth',
                       'absence_holds_in_other_layers', truth_scope='all_layers')
    assert answer['result']['claim_cards'][0]['claim_supported'] is False
    assert answer['result']['claim_cards'][0]['statement_true'] is False
    assert answer['provenance']['graph'][0]['layer_scope']['layer_ids'] == [COEX]
