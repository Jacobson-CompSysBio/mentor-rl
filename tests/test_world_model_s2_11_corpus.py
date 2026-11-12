"""Check path traces, scoped calibration, exact failures, and source-bound validation."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.world_model import s2_11_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_11_sampling import fact_from_genes
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY
from runtime.world_model.s2_paths import ARGUMENT_KEYS, PathIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


def calls(row):
    """Read the ordered current-turn calls without the supplied evidence."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def context(row):
    """Read the exact active graph and supplied observations."""
    return json.loads(row['messages'][1]['content'])


def answer(row):
    """Read the complete final envelope."""
    return row['messages'][-1]['content']


class PathCorpusTests(unittest.TestCase):
    """Use native motifs that separate existence, minimum hops, ties, and truth."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 180)
                if gene_group(f'ENSG{number:011d}', SEED)[1] == 'train']
        (cls.a, cls.b, cls.c, cls.d, cls.x, cls.z, cls.w, cls.r, cls.s,
         cls.j, cls.k, cls.outside, cls.missing, cls.graph_only, cls.null_symbol) = pool[:15]
        a, b, c, d, x, z, w, r, s, j, k, outside, _, graph_only, null_symbol = pool[:15]
        ppi = {(a, a): .95, (a, b): 0, (b, x): .7, (x, z): .6,
               (a, c): .8, (c, d): .5, (d, z): .4,
               (a, w): .3, (w, r): .2, (r, s): .1, (s, z): .05,
               (j, k): .9, (graph_only, null_symbol): .25}
        cls.view = GraphView('unit_path_corpus', 'b' * 64, {
            PPI: LayerGraph(PPI, False, {tuple(sorted(pair)): weight for pair, weight in ppi.items()},
                            frozenset(gene for pair in ppi for gene in pair)),
            COEX: LayerGraph(COEX, False, {(a, outside): .5, (z, outside): .4}, frozenset((a, z, outside))),
            BRAIN: LayerGraph(BRAIN, True, {(a, a): .9, (a, b): 0, (b, x): .7, (x, z): .6,
                (a, c): .8, (c, d): .5, (d, z): .4, (j, k): .9},
                frozenset((a, b, c, d, x, z, j, k))),
            CD4: LayerGraph(CD4, True, {(outside, a): .1}, frozenset((outside, a))),
        })
        symbols = {gene: f'UNIT_PATH_CORPUS_{ordinal}' for ordinal, gene in enumerate(pool) if gene != graph_only}
        symbols[null_symbol] = None
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', symbols)
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contract = corpus.load_build_contract()

    def setUp(self):
        self.index = PathIndex(self.view)

    def fact(self, genes=None, layer=PPI):
        return fact_from_genes(self.view, self.index, layer,
                               [self.a, self.z] if genes is None else genes, SEED)

    def row(self, genes=None, layer=PPI, *, variant='lookup', framing='direct', question='path',
            claim=None, truth_scope=None, statement=None, membership=False, variant_index=0, registry=None):
        return corpus.build_row(self.view, self.fact(genes, layer), framing, variant, variant_index,
            self.bank, self.contract, index=self.index, registry=self.registry if registry is None else registry,
            question_kind=question, claim=claim, truth_scope=truth_scope, statement=statement,
            membership_cards=membership, seed=SEED)

    def validate(self, row, *, registry=None):
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
                            registry=self.registry if registry is None else registry, index=self.index)

    def diamond_row(self, *, question='path', claim=None):
        """Supply two shortest routes whose alternate induced graph proves two hops."""
        edges = {(self.a, self.b): .8, (self.b, self.z): .7,
                 (self.a, self.c): .6, (self.c, self.z): .5}
        view = GraphView('unit_tied_path_corpus', 'd' * 64, {
            PPI: LayerGraph(PPI, False, edges, frozenset((self.a, self.b, self.c, self.z)))})
        index = PathIndex(view)
        fact = fact_from_genes(view, index, PPI, [self.a, self.z], SEED)
        row = corpus.build_row(view, fact, 'direct', 'tie_route_context', 0, self.bank,
            self.contract, index=index, registry=self.registry, question_kind=question, claim=claim, seed=SEED)
        corpus.validate_row(row, view, bank=self.bank, contract=self.contract, registry=self.registry, index=index)
        return row

    def assert_valid(self, row):
        self.validate(row)
        self.assertEqual(tuple(answer(row)), ('status', 'provenance', 'result',
            'continuation_decision', 'blocked_reason', 'summary'))

    def test_ordinary_path_has_exact_four_argument_native_call(self):
        row = self.row()
        self.assert_valid(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(calls(row)[0]['name'], 'shortest_path')
        self.assertEqual(tuple(calls(row)[0]['arguments']), ARGUMENT_KEYS)
        result = answer(row)['result']
        self.assertEqual(result['gene_ids'], [self.a, self.z])
        self.assertEqual(result['path_gene_ids'], [self.a, self.b, self.x, self.z])
        self.assertEqual(result['hop_count'], 3)
        self.assertNotIn('claim_cards', result)
        self.assertNotIn('collection', result)

    def test_zero_weight_edge_is_a_unique_one_hop_path(self):
        row = self.row([self.a, self.b], variant='edge_context')
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(context(row)['observations'][0]['result']['weight'], 0)
        self.assertEqual(answer(row)['result']['path_gene_ids'], [self.a, self.b])
        self.assertEqual(answer(row)['result']['hop_count'], 1)

    def test_reversed_undirected_query_recomputes_its_canonical_tie(self):
        forward, reverse = self.row(), self.row(variant='reversed')
        self.assert_valid(reverse)
        self.assertEqual(answer(reverse)['result']['gene_ids'], [self.z, self.a])
        self.assertEqual(answer(reverse)['result']['path_gene_ids'], [self.z, self.d, self.c, self.a])
        self.assertNotEqual(answer(reverse)['result']['path_gene_ids'], list(reversed(answer(forward)['result']['path_gene_ids'])))
        self.assertEqual(forward['metadata']['split_unit'], reverse['metadata']['split_unit'])

    def test_directed_reverse_can_be_a_completed_negative(self):
        forward, reverse = self.row(layer=BRAIN), self.row([self.z, self.a], BRAIN)
        self.assert_valid(forward)
        self.assert_valid(reverse)
        self.assertTrue(answer(forward)['result']['path_found'])
        self.assertFalse(answer(reverse)['result']['path_found'])
        self.assertEqual(answer(reverse)['result']['direction'], 'source_to_target')
        self.assertEqual(answer(reverse)['continuation_decision'], 'stop')

    def test_layer_absence_completes_a_negative_but_global_absence_leaves_nulls(self):
        outside = self.row([self.outside, self.a], framing='step')
        missing = self.row([self.missing, self.outside], framing='step')
        self.assert_valid(outside)
        self.assert_valid(missing)
        self.assertEqual((answer(outside)['status'], answer(outside)['continuation_decision']), ('resolved', 'continue'))
        self.assertFalse(answer(outside)['result']['path_found'])
        self.assertEqual(answer(outside)['result']['path_gene_ids'], [])
        self.assertEqual(answer(outside)['result']['not_in_layer_gene_ids'], [self.outside])
        self.assertEqual((answer(missing)['status'], answer(missing)['continuation_decision']), ('not_found', 'defer'))
        for key in ('path_found', 'path_gene_ids', 'hop_count'):
            self.assertIsNone(answer(missing)['result'][key])
        self.assertEqual(answer(missing)['result']['missing_gene_ids'], [self.missing])
        self.assertEqual(answer(missing)['result']['not_in_layer_gene_ids'], [self.outside])

    def test_repeated_endpoints_keep_two_positions_and_need_only_membership(self):
        for gene in (self.a, self.b):
            row = self.row([gene, gene], variant='membership_context')
            self.assert_valid(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(answer(row)['result']['gene_ids'], [gene, gene])
            self.assertEqual(answer(row)['result']['path_gene_ids'], [gene])
            self.assertEqual(answer(row)['result']['hop_count'], 0)

    def test_matching_path_and_duplicate_context_complete_without_a_call(self):
        for variant in ('context', 'duplicate_context'):
            row = self.row(variant=variant, framing='step')
            self.assert_valid(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(len(answer(row)['provenance']['graph']), 1)
            self.assertEqual(answer(row)['continuation_decision'], 'continue')

    def test_negative_edge_selected_disconnection_and_routes_still_need_exact_path_acquisition(self):
        for variant in ('edge_context', 'subgraph_context', 'route_context', 'longer_route_context', 'prefix_context'):
            with self.subTest(variant=variant):
                row = self.row(variant=variant)
                self.assert_valid(row)
                self.assertEqual(len(calls(row)), 1)
                self.assertEqual(answer(row)['result']['hop_count'], 3)
        self.assertFalse(context(self.row(variant='edge_context'))['observations'][0]['result']['edge_exists'])
        self.assertEqual(context(self.row(variant='subgraph_context'))['observations'][0]['result']['edges'],
                         [{'gene_ids': [self.a, self.a], 'weight': .95}])

    def test_complete_empty_directed_outgoing_and_closed_component_prove_no_path(self):
        empty = self.row([self.z, self.a], BRAIN, variant='native_context')
        closed = self.row([self.j, self.a], variant='closed_component_context')
        for row in (empty, closed):
            self.assert_valid(row)
            self.assertEqual(calls(row), [])
            self.assertFalse(answer(row)['result']['path_found'])
        self.assertEqual(context(empty)['observations'][-1]['result']['neighbors'], [])
        self.assertEqual(len(context(closed)['observations']), 3)

    def test_stale_other_layer_other_pair_and_reverse_evidence_do_not_supply_the_current_witness(self):
        for variant, layer in (('stale', PPI), ('wrong_layer', PPI), ('distractor', PPI),
                               ('reversed_context', PPI), ('wrong_direction', BRAIN)):
            with self.subTest(variant=variant):
                row = self.row(layer=layer, variant=variant)
                self.assert_valid(row)
                self.assertEqual(len(calls(row)), 1)
                self.assertEqual(answer(row)['result']['gene_ids'], [self.a, self.z])
                self.assertEqual(answer(row)['result']['path_gene_ids'], [self.a, self.b, self.x, self.z])

    def test_all_28_declared_variants_have_real_fixture_coverage(self):
        for variant in corpus.VARIANTS:
            if variant == 'tie_route_context':
                self.assertEqual(self.diamond_row()['metadata']['variant'], variant)
                continue
            genes, layer = ([self.j, self.a], PPI) if variant == 'closed_component_context' else ([self.a, self.z], BRAIN if variant == 'wrong_direction' else PPI)
            with self.subTest(variant=variant):
                self.assert_valid(self.row(genes, layer, variant=variant))
        self.assertEqual(len(corpus.VARIANTS), 28)

    def test_complete_alternate_route_proves_minimum_hops_but_leaves_the_canonical_tie_unsettled(self):
        row = self.diamond_row()
        supplied = context(row)['observations']
        self.assertEqual(len(supplied), 1)
        self.assertEqual(supplied[0]['operation'], 'induce_subgraph')
        graph = supplied[0]['result']
        self.assertTrue(graph['collection']['complete'])
        self.assertEqual(graph['layer_gene_ids'], [self.a, self.c, self.z])
        self.assertEqual([edge['gene_ids'] for edge in graph['edges']], [[self.a, self.c], [self.c, self.z]])
        self.assertNotIn([self.a, self.z], [edge['gene_ids'] for edge in graph['edges']])
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(calls(row)[0]['name'], 'shortest_path')
        self.assertEqual(answer(row)['result']['hop_count'], 2)
        self.assertEqual(answer(row)['result']['path_gene_ids'], [self.a, self.b, self.z])

    def test_the_same_complete_route_supports_direct_edge_absence_without_a_new_call(self):
        row = self.diamond_row(question='claim_support', claim='no_edge_recorded_in_scope')
        self.assertEqual(calls(row), [])
        card = answer(row)['result']['claim_cards'][0]
        self.assertEqual(card['supported_claims'], ['no_edge_recorded_in_scope'])
        self.assertTrue(card['claim_supported'])
        self.assertNotIn('path_found', answer(row)['result'])

    def test_the_minimum_known_tie_contrast_requires_two_hops_and_multiple_shortest_routes(self):
        for genes in ([self.a, self.z], [self.a, self.x], [self.a, self.b]):
            with self.subTest(genes=genes), self.assertRaises(VariantUnavailable):
                self.row(genes, variant='tie_route_context')

    def test_fault_schedules_repeat_exact_calls_and_preserve_null_coverage(self):
        for prefix in ('', 'edge_', 'symbol_'):
            variants = ('terminal',) if prefix == 'symbol_' else ('terminal', 'recovered', 'exhausted')
            for outcome in variants:
                with self.subTest(prefix=prefix, outcome=outcome):
                    row = self.row(variant=prefix + outcome)
                    self.assert_valid(row)
                    trace_calls = calls(row)
                    self.assertEqual(len(trace_calls), 1 if outcome == 'terminal' else 2)
                    if len(trace_calls) == 2:
                        self.assertEqual(trace_calls[0], trace_calls[1])
                    if outcome == 'recovered':
                        self.assertEqual(answer(row)['status'], 'resolved')
                    else:
                        result = answer(row)['result']
                        self.assertEqual(answer(row)['status'], 'error')
                        self.assertIsNone(result['coverage'])
                        self.assertNotIn('path_found', result)
                        self.assertEqual(result['partial_results'], context(row)['observations'])

    def test_execution_limit_is_an_error_with_the_checked_edge_retained(self):
        row = self.row(variant='execution_limit')
        self.assert_valid(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(answer(row)['status'], 'error')
        self.assertEqual(answer(row)['continuation_decision'], 'defer')
        self.assertIsNone(answer(row)['result']['coverage'])
        self.assertEqual(answer(row)['result']['partial_results'][0]['operation'], 'get_edge')
        self.assertFalse(answer(row)['result']['partial_results'][0]['result']['edge_exists'])
        self.assertNotIn('path_found', answer(row)['result'])

    def test_claim_only_negative_edge_does_not_acquire_an_unrequested_path(self):
        row = self.row(variant='edge_context', question='claim_support', claim='no_path_recorded_in_scope')
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        result = answer(row)['result']
        self.assertEqual(set(result), {'multiplex_id', 'layer_id', 'directed', 'direction', 'gene_ids', 'claim_cards'})
        card = result['claim_cards'][0]
        self.assertEqual(card['supported_claims'], ['no_edge_recorded_in_scope'])
        self.assertFalse(card['claim_supported'])
        self.assertNotIn('statement_true', card)

    def test_arbitrary_multihop_route_supports_neither_absence_claim(self):
        row = self.row(variant='longer_route_context', question='claim_card')
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(answer(row)['result']['claim_cards'][0]['supported_claims'], [])

    def test_complete_shortest_multihop_path_supports_direct_edge_absence_only(self):
        row = self.row(variant='context', question='path_claims')
        self.assert_valid(row)
        card = answer(row)['result']['claim_cards'][0]
        self.assertEqual(card['supported_claims'], ['no_edge_recorded_in_scope'])
        self.assertEqual(card['unsupported_claims'], list(CLAIM_VOCABULARY[1:]))
        self.assertEqual(card['subject'], {'gene_ids': [self.a, self.z], 'multiplex_id': self.view.multiplex_id,
            'layer_id': PPI, 'direction': 'undirected'})
        self.assertNotIn('claim', card)
        self.assertNotIn('statement_true', card)

    def test_biological_support_verdict_is_not_a_truth_verdict(self):
        row = self.row([self.j, self.a], variant='context', question='path_support', claim='no_interaction_in_biology')
        self.assert_valid(row)
        card = answer(row)['result']['claim_cards'][0]
        self.assertEqual(card['supported_claims'], list(CLAIM_VOCABULARY[:2]))
        self.assertFalse(card['claim_supported'])
        self.assertNotIn('statement_true', card)
        self.assertEqual(set(card['supported_claims']) | set(card['unsupported_claims']), set(CLAIM_VOCABULARY))

    def test_layer_membership_cards_bind_only_the_affected_single_gene(self):
        row = self.row([self.outside, self.a], variant='membership_context', question='path_claims', membership=True)
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        pair, gene = answer(row)['result']['claim_cards']
        self.assertEqual(pair['subject']['gene_ids'], [self.outside, self.a])
        self.assertEqual(pair['supported_claims'], list(CLAIM_VOCABULARY[:2]))
        self.assertEqual(gene['subject']['gene_ids'], [self.outside])
        self.assertEqual(gene['supported_claims'], [*CLAIM_VOCABULARY[:2], 'gene_not_node_in_layer'])

    def test_explicit_missing_membership_has_a_separate_card_without_a_path_claim(self):
        row = self.row([self.missing, self.missing], variant='membership_context', question='claim_card', membership=True)
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        pair, gene = answer(row)['result']['claim_cards']
        self.assertEqual(pair['supported_claims'], [])
        self.assertEqual(gene['subject']['gene_ids'], [self.missing])
        self.assertEqual(gene['supported_claims'], ['gene_not_node_in_multiplex'])
        self.assertNotIn('path_found', answer(row)['result'])

    def test_zero_hops_do_not_assess_self_loop_absence(self):
        row = self.row([self.b, self.b], variant='membership_context', question='path_support', claim='no_edge_recorded_in_scope')
        self.assert_valid(row)
        self.assertEqual(answer(row)['result']['hop_count'], 0)
        self.assertFalse(answer(row)['result']['claim_cards'][0]['claim_supported'])
        self.assertEqual(answer(row)['result']['claim_cards'][0]['supported_claims'], [])

    def test_top_k_omission_is_scoped_to_the_shown_list(self):
        row = self.row(variant='prefix_context', question='claim_support', claim='not_in_shown_top_k')
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        card = answer(row)['result']['claim_cards'][0]
        self.assertTrue(card['claim_supported'])
        self.assertIn('not_in_shown_top_k', card['supported_claims'])
        self.assertNotIn('no_path_recorded_in_scope', card['supported_claims'])

    def test_explicit_scoped_truth_retains_the_exact_literal_proposition(self):
        row = self.row([self.j, self.a], variant='context', question='path_truth', claim='no_path_recorded_in_scope')
        self.assert_valid(row)
        statement = corpus.truth_statement(self.view, self.fact([self.j, self.a]), 'no_path_recorded_in_scope', 'scope')
        self.assertEqual(row['metadata']['statement'], statement)
        self.assertIn(statement, row['messages'][2]['content'])
        self.assertTrue(answer(row)['result']['claim_cards'][0]['statement_true'])
        positive = self.row(variant='context', question='path_truth', claim='no_path_recorded_in_scope')
        self.assert_valid(positive)
        self.assertFalse(answer(positive)['result']['claim_cards'][0]['statement_true'])

    def test_one_visible_path_refutes_universal_absence_without_supporting_other_layer_absence(self):
        row = self.row(variant='context', question='truth', claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assert_valid(row)
        card = answer(row)['result']['claim_cards'][0]
        self.assertFalse(card['claim_supported'])
        self.assertFalse(card['statement_true'])
        self.assertIn('any layer of ' + self.view.multiplex_id, row['metadata']['statement'])

    def test_one_local_negative_cannot_decide_universal_truth(self):
        row = self.row([self.j, self.a], variant='context', question='truth',
                       claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(answer(row)['status'], 'invalid_input')
        self.assertEqual(answer(row)['result']['issue_code'], 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertEqual(answer(row)['continuation_decision'], 'defer')
        self.assertEqual(answer(row)['result']['partial_results'], context(row)['observations'])
        self.assertNotIn('claim_cards', answer(row)['result'])

    def test_a_path_in_another_registered_layer_can_refute_the_exact_universal_statement(self):
        row = self.row(variant='wrong_layer', question='truth',
                       claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assert_valid(row)
        self.assertEqual(calls(row), [])
        card = answer(row)['result']['claim_cards'][0]
        self.assertFalse(card['claim_supported'])
        self.assertFalse(card['statement_true'])
        self.assertEqual(context(row)['observations'][0]['result']['layer_id'], COEX)
        self.assertEqual(card['subject']['layer_id'], PPI)
        self.assertEqual(answer(row)['provenance']['graph'][0]['layer_scope']['layer_ids'], [COEX])

    def test_truth_and_support_tasks_reject_swapped_or_unrequested_statements(self):
        with self.assertRaises(ValueError):
            self.row(variant='context', question='truth', claim='no_path_recorded_in_scope', statement='There is no biological interaction.')
        with self.assertRaises(ValueError):
            self.row(variant='context', question='claim_support', truth_scope='scope')
        with self.assertRaises(ValueError):
            self.row(variant='context', question='path_truth', claim='no_interaction_in_biology')
        with self.assertRaises(VariantUnavailable):
            self.row(question='claim_support')

    def test_repeated_symbols_keep_two_input_positions_but_one_identity_observation(self):
        row = self.row([self.a, self.a], variant='symbol_context')
        self.assert_valid(row)
        self.assertEqual(len(context(row)['observations']), 1)
        self.assertEqual(context(row)['observations'][0]['operation'], 'lookup_human_gene_symbol')
        self.assertEqual(row['metadata']['request']['genes'], [self.registry.symbols_by_id[self.a]] * 2)
        self.assertEqual(answer(row)['result']['gene_ids'], [self.a, self.a])

    def test_ambiguous_null_and_graph_only_symbols_are_not_invented(self):
        symbols = dict(self.registry.symbols_by_id)
        symbols[self.missing] = symbols[self.a]
        ambiguous = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', symbols)
        for genes, registry in (([self.a, self.z], ambiguous), ([self.null_symbol, self.z], self.registry),
                                 ([self.graph_only, self.z], self.registry)):
            with self.assertRaises(VariantUnavailable):
                self.row(genes, variant='symbol_context', registry=registry)
        self.assert_valid(self.row([self.graph_only, self.z]))

    def test_all_ten_diagnostics_preserve_raw_four_fields_and_exact_calls(self):
        expected = {'unknown_layer': ('not_found', 'UNKNOWN_LAYER'), 'tool_unknown_layer': ('not_found', 'UNKNOWN_LAYER'),
            'unknown_multiplex': ('not_found', 'UNKNOWN_MULTIPLEX'), 'missing_genes': ('invalid_input', 'INVALID_PAIR'),
            'missing_scope': ('invalid_input', 'MISSING_SCOPE'), 'invalid_pair': ('invalid_input', 'INVALID_PAIR'),
            'missing_direction': ('ambiguous', 'AMBIGUOUS_PATH_DIRECTION'), 'invalid_direction': ('invalid_input', 'INVALID_DIRECTION'),
            'unavailable_direction': ('invalid_input', 'DIRECTION_UNAVAILABLE'), 'unresolved_identity': ('not_found', 'UNRESOLVED_IDENTITY')}
        for kind in corpus.DIAGNOSTICS:
            with self.subTest(kind=kind):
                row = corpus.build_diagnostic(self.view, self.fact(), 'step', 0, self.bank, self.contract,
                                              kind=kind, registry=self.registry, seed=SEED)
                self.assert_valid(row)
                final = answer(row)
                self.assertEqual((final['status'], final['result']['issue_code']), expected[kind])
                self.assertEqual(set(final['result']['request']), {'genes', 'multiplex_id', 'layer_id', 'direction'})
                self.assertEqual(final['result']['request'], row['metadata']['request'])
                self.assertEqual(final['continuation_decision'], 'defer')
                self.assertEqual(len(calls(row)), int(kind in ('tool_unknown_layer', 'unresolved_identity')))
                self.assertNotIn('path_found', final['result'])
                if kind == 'tool_unknown_layer':
                    self.assertNotIn('layers', context(row)['active_graph'])
                    self.assertEqual(final['result']['rejected_arguments'], calls(row)[0]['arguments'])
                elif kind == 'unresolved_identity':
                    self.assertEqual(calls(row)[0]['name'], 'lookup_human_gene_symbol')
                    self.assertEqual(final['result']['partial_results'][0]['result']['candidate_gene_ids'], [])
                else:
                    self.assertIsNone(final['result']['rejected_arguments'])

    def test_summary_blocker_and_reasoning_prose_remain_free(self):
        for row in (self.row(), self.row(variant='edge_terminal')):
            answer(row)['summary'] = 'A different correct description.'
            if answer(row)['blocked_reason']:
                answer(row)['blocked_reason'] = 'The required service must recover.'
            for message in row['messages']:
                if message['role'] == 'assistant':
                    message['thinking'] = 'The visible evidence determines this structured action.'
            self.assert_valid(row)

    def test_planning_uses_no_hidden_inventory_or_adjacency(self):
        class MetadataView:
            multiplex_id = self.view.multiplex_id
            graph_version = self.view.graph_version
            layers = {layer: SimpleNamespace(directed=source.directed) for layer, source in self.view.layers.items()}
            layer_order = self.view.layer_order

            @property
            def multiplex_nodes(self):
                raise AssertionError('the planner cannot use the source inventory')

        original = corpus._plan
        def visible_only(view, arguments, observations, task):
            return original(MetadataView(), arguments, observations, task)
        with patch.object(corpus, '_plan', side_effect=visible_only):
            for variant, question in (('lookup', 'path'), ('route_context', 'path'),
                                      ('context', 'path_claims'), ('edge_context', 'claim_support')):
                self.assert_valid(self.row(variant=variant, question=question))

    def test_validator_rejects_wrong_calls_path_labels_and_provenance(self):
        original = self.row()
        mutations = [lambda row: calls(row)[0]['arguments']['gene_ids'].reverse(),
            lambda row: answer(row)['result'].update(hop_count=99),
            lambda row: answer(row)['result'].update(path_gene_ids=[self.a, self.z]),
            lambda row: answer(row)['provenance']['graph'][0].update(graph_version='0' * 64),
            lambda row: row['metadata'].update(tied_shortest_paths=99),
            lambda row: row['metadata'].update(path_found=False),
            lambda row: row['metadata'].update(variant='context')]
        for mutate in mutations:
            row = deepcopy(original)
            mutate(row)
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_validator_rejects_altered_native_weights_and_false_context_labels(self):
        original = self.row([self.a, self.b], variant='edge_context')
        row = deepcopy(original)
        supplied = context(row)
        supplied['observations'][0]['result']['weight'] = .123
        row['messages'][1]['content'] = json.dumps(supplied)
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='context')
        row['metadata']['variant'] = 'duplicate_context'
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_validator_binds_case_labels_source_category_tasks_and_row_identity(self):
        original = self.row()
        for mutate in (lambda row: row['metadata'].update(variant='reversed'),
                       lambda row: row['metadata'].update(category='closed_component'),
                       lambda row: row['metadata'].update(question_kind='path_claims'),
                       lambda row: row.update(row_id='s211_' + 'f' * 32)):
            changed = deepcopy(original)
            mutate(changed)
            with self.assertRaises(ValueError):
                self.validate(changed)
        diagnostic = corpus.build_diagnostic(self.view, self.fact(), 'direct', 0, self.bank,
                                              self.contract, kind='missing_genes', registry=self.registry)
        diagnostic['row_id'] = 's211_' + 'e' * 32
        with self.assertRaises(ValueError):
            self.validate(diagnostic)

    def test_validator_rejects_extra_envelope_fields_invented_claims_and_swapped_truth(self):
        row = self.row(variant='context', question='path_truth', claim='no_path_recorded_in_scope')
        for mutate in (lambda item: answer(item).update(statement_true=False),
                       lambda item: answer(item)['result'].update(extra=True),
                       lambda item: answer(item)['result']['claim_cards'][0].update(statement_true=True),
                       lambda item: item['metadata'].update(statement='A different proposition.'),
                       lambda item: answer(item)['result']['claim_cards'][0]['supported_claims'].append('invented_claim')):
            changed = deepcopy(row)
            mutate(changed)
            with self.assertRaises(ValueError):
                self.validate(changed)

    def test_validator_rejects_retry_after_exhaustion_or_without_an_allowance(self):
        row = self.row(variant='edge_recovered')
        response = next(message for message in row['messages'] if message['role'] == 'tool')
        response['content']['retries_remaining'] = 0
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='edge_exhausted')
        calls(row)[1]['arguments']['gene_ids'].reverse()
        with self.assertRaises(ValueError):
            self.validate(row)


if __name__ == '__main__':
    unittest.main()
