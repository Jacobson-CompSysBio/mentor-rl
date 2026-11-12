"""Check scoped family and pair traces, exact calls, and independent blockers."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_13_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import load_registry_context
from runtime.world_model.s2_13_evidence import CLAIM_VOCABULARY, FAMILY_CARDS, FAMILIES
from runtime.world_model.s2_13_sampling import fact_from_pair, select_facts
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY as GRAPH_CLAIMS
from test_world_model_s2_13_sampling import fixture, SEED, PPI, COEX, BRAIN, CD4, BULK, SC


ROOT = Path(__file__).resolve().parents[1]


def calls(row):
    """Read current-turn calls separately from the original harness context."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def context(row):
    """Read the unchanged supplied observations and qualified metadata."""
    return json.loads(row['messages'][1]['content'])


def answer(row):
    """Read the final structured answer without its free reasoning."""
    return row['messages'][-1]['content']


class SpecificityCorpusTests(unittest.TestCase):
    """Use real fixture edges, endpoint inventories, and closed S1 family meanings."""

    @classmethod
    def setUpClass(cls):
        cls.view, cls.index, cls.registry, cls.old, cls.named = fixture()
        cls.contract = corpus.load_build_contract()
        cls.bank = json.loads((ROOT / cls.contract['prompt_bank']['path']).read_text())
        cls.facts, _ = select_facts(cls.view, cls.index, cls.registry,
            dict.fromkeys(('train', 'validation', 'test'), 1), SEED, smoke=True, old_view=cls.old)
        cls.pair = cls.named[PPI, 'train', 'high']

    def fact(self, pair=None, layer=PPI, registry=None):
        """Measure the requested source pair before the question or context is chosen."""
        return fact_from_pair(self.view, self.index, registry or self.registry, self.old,
                             layer, self.pair if pair is None else pair, SEED)

    def row(self, pair=None, layer=PPI, variant='lookup', framing='direct', **task):
        """Build one trace through the real frozen request and wrapper banks."""
        return corpus.build_row(self.view, self.fact(pair, layer), framing, variant, 0,
            self.bank, self.contract, index=self.index, registry=self.registry,
            old_view=self.old, seed=SEED, **task)

    def validate(self, row, registry=None):
        """Replay all native records and the exact visible-evidence result."""
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
            index=self.index, registry=registry or self.registry, old_view=self.old)

    def card(self, row):
        """Read the primary biological card."""
        return answer(row)['result']['claim_cards'][0]

    def test_family_capability_is_closed_taxonomy_without_graph_calls(self):
        expected = {'ppi': ['physical_interaction_recorded'], 'coexpression': ['global_coexpression'],
            'tf_target': ['regulatory_link_recorded'], 'bulk_pen': ['context_predictive_expression'],
            'scpen': ['context_predictive_expression'], 'other': []}
        for family, supported in expected.items():
            for framing, decision in (('direct', 'stop'), ('step', 'continue')):
                row = corpus.build_family_row(self.view, family, 'train', framing, 0,
                                              self.bank, self.contract)
                self.validate(row)
                final, card = answer(row), self.card(row)
                self.assertFalse(calls(row))
                self.assertEqual(final['provenance'], {'registry': [], 'graph': []})
                self.assertEqual(final['continuation_decision'], decision)
                self.assertEqual(card['subject'], {'kind': 'family', 'layer_family': family})
                self.assertEqual(card['supported_claims'], supported)
                self.assertEqual(card['unsupported_claims'], [c for c in CLAIM_VOCABULARY if c not in supported])
                self.assertEqual(set(final['result']), {'layer_family', 'evidence_type', 'claim_cards'})
                self.assertEqual(final['result']['evidence_type'], supported[0] if supported else None)

    def test_generic_tf_capability_cannot_identify_regulator_target_roles(self):
        row = corpus.build_family_row(self.view, 'tf_target', 'test', 'direct', 1,
            self.bank, self.contract, question_kind='family_support', claim='regulation_direction')
        self.validate(row)
        self.assertFalse(self.card(row)['claim_supported'])
        pair = self.named[CD4, 'train', 'one_way']
        recorded = self.row(pair, CD4, question_kind='edge_support', claim='regulation_direction')
        self.validate(recorded)
        self.assertTrue(self.card(recorded)['claim_supported'])
        self.assertEqual(self.card(recorded)['supported_claims'], ['regulatory_link_recorded', 'regulation_direction'])

    def test_all_four_pair_categories_keep_required_operation_status(self):
        present = self.pair
        pairs = {'recorded': present, 'no_edge': self.named[PPI, 'train', 'route'],
            'outside_layer': (present[0], self.named['pen', 'train', '11'][0]),
            'outside_multiplex': (present[0], self.named['missing'])}
        for category, pair in pairs.items():
            for framing, decision in (('direct', 'stop'), ('step', 'continue')):
                row = self.row(pair, framing=framing)
                self.validate(row)
                final = answer(row)
                if category == 'outside_multiplex':
                    self.assertEqual((final['status'], final['continuation_decision']), ('not_found', 'defer'))
                    self.assertIsNone(final['result']['edge_exists'])
                    self.assertNotIn('claim_cards', final['result'])
                else:
                    self.assertEqual((final['status'], final['continuation_decision']), ('resolved', decision))
                    self.assertEqual(final['result']['edge_exists'], category == 'recorded')
                    self.assertEqual(self.card(row)['supported_claims'],
                                     ['physical_interaction_recorded'] if category == 'recorded' else [])

    def test_native_edge_call_retains_only_the_existing_three_arguments(self):
        row = self.row()
        self.validate(row)
        self.assertEqual(calls(row), [{'name': 'get_edge', 'arguments': {
            'multiplex_id': self.view.multiplex_id, 'layer_id': PPI, 'gene_ids': list(self.pair)}}])
        subject = self.card(row)['subject']
        self.assertEqual(subject, {'kind': 'pair', 'gene_ids': sorted(self.pair),
            'multiplex_id': self.view.multiplex_id, 'graph_version': self.view.graph_version,
            'layer_id': PPI, 'layer_family': 'ppi', 'directed': False})

    def test_every_declared_variant_has_an_attainable_source_fixture(self):
        reached = set()
        for variant in corpus.VARIANTS:
            for fact in self.facts:
                try:
                    row = corpus.build_row(self.view, fact, 'direct', variant, 0,
                        self.bank, self.contract, index=self.index, registry=self.registry, old_view=self.old)
                except VariantUnavailable:
                    continue
                self.validate(row)
                reached.add(variant)
                break
        self.assertEqual(reached, set(corpus.VARIANTS))
        self.assertEqual(len(reached), 30)

    def test_inapplicable_source_controls_do_not_invent_observations(self):
        for layer, variant in ((PPI, 'same_family_context'), (PPI, 'reciprocal_context'),
                               (CD4, 'reversed_request'), (PPI, 'pen_contrast_context')):
            pair = self.named[layer, 'train', 'high']
            with self.subTest(layer=layer, variant=variant), self.assertRaises(VariantUnavailable):
                self.row(pair, layer, variant)

    def test_matching_duplicate_and_undirected_reverse_contexts_avoid_calls(self):
        for variant in ('context', 'duplicate_context', 'reverse_context'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertFalse(calls(row))
            self.assertEqual(len(answer(row)['provenance']['graph']), 1)
            self.assertEqual(self.card(row)['supported_claims'], ['physical_interaction_recorded'])
        reversed_row = self.row(variant='reversed_request')
        self.validate(reversed_row)
        self.assertEqual(calls(reversed_row)[0]['arguments']['gene_ids'], list(reversed(self.pair)))
        self.assertEqual(self.card(reversed_row)['subject']['gene_ids'], sorted(self.pair))

    def test_pair_evidence_is_not_supplied_by_a_family_tag(self):
        row = self.row(variant='family_context', question_kind='claim_support', claim='physical_interaction_recorded')
        self.validate(row)
        self.assertFalse(calls(row))
        self.assertEqual(self.card(row)['supported_claims'], [])
        self.assertEqual(set(answer(row)['result']), {'claim_cards'})
        self.assertNotIn('edge_exists', answer(row)['result'])
        self.assertEqual(answer(row)['provenance'], {'registry': [], 'graph': []})

    def test_a_directed_reverse_record_does_not_answer_the_forward_subject(self):
        pair = self.named[CD4, 'train', 'reciprocal']
        assessment = self.row(pair, CD4, 'reverse_context', question_kind='claim_support', claim='regulation_direction')
        self.validate(assessment)
        self.assertFalse(calls(assessment))
        self.assertEqual(self.card(assessment)['subject']['gene_ids'], list(pair))
        self.assertEqual(self.card(assessment)['supported_claims'], [])
        acquisition = self.row(pair, CD4, 'reverse_context')
        self.validate(acquisition)
        self.assertEqual(len(calls(acquisition)), 1)
        self.assertEqual(calls(acquisition)[0]['arguments']['gene_ids'], list(pair))
        supplied = context(acquisition)['observations'][0]['provenance']['graph'][0]
        self.assertNotIn(supplied, answer(acquisition)['provenance']['graph'])

    def test_wrong_pair_layer_and_old_view_have_distinct_unobserved_targets(self):
        for variant in ('wrong_pair_context', 'wrong_layer_context', 'old_view_context', 'stale'):
            assessment = self.row(variant=variant, question_kind='claim_support', claim='physical_interaction_recorded')
            self.validate(assessment)
            self.assertFalse(calls(assessment))
            self.assertEqual(self.card(assessment)['supported_claims'], [])
            self.assertEqual(answer(assessment)['provenance'], context(assessment)['observations'][0]['provenance'])
            acquisition = self.row(variant=variant)
            self.validate(acquisition)
            self.assertEqual(len(calls(acquisition)), 1)
            self.assertNotIn(context(acquisition)['observations'][0]['provenance']['graph'][0],
                             answer(acquisition)['provenance']['graph'])
        old = context(self.row(variant='old_view_context'))['observations'][0]
        self.assertEqual(old['arguments']['multiplex_id'], self.old.multiplex_id)
        self.assertEqual(old['provenance']['graph'][0]['graph_version'], self.old.graph_version)
        self.assertEqual(old['result'], self.old.get_edge(**old['arguments'], evidence_id='check_old')['result'])

    def test_pen_comparison_retains_separate_local_cards_in_explicit_order(self):
        pair = self.named['pen', 'train', '10']
        row = self.row(pair, SC, 'pen_contrast_context', question_kind='edge_support', claim='context_predictive_expression')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        cards = answer(row)['result']['claim_cards']
        self.assertEqual([card['subject']['layer_id'] for card in cards], [BULK, SC])
        self.assertEqual([card['supported_claims'] for card in cards], [['context_predictive_expression'], []])
        self.assertEqual(len(answer(row)['provenance']['graph']), 2)
        self.assertEqual(context(row)['observations'][0]['result']['weight'], .4)
        self.assertFalse(answer(row)['result']['edge_exists'])

    def test_two_observed_contexts_do_not_support_further_contexts_or_weight_comparison(self):
        pair = self.named['pen', 'train', '11']
        row = self.row(pair, BULK, 'pen_contrast_context', question_kind='claim_support', claim='holds_in_other_contexts')
        self.validate(row)
        self.assertFalse(calls(row))
        cards = answer(row)['result']['claim_cards']
        self.assertEqual([card['subject']['layer_id'] for card in cards], [SC, BULK])
        self.assertEqual(cards[0]['supported_claims'], ['context_predictive_expression'])
        self.assertEqual(cards[1]['supported_claims'], [])
        full = self.row(pair, BULK, 'two_context_context', question_kind='claim_support', claim='holds_in_other_contexts')
        self.validate(full)
        self.assertFalse(calls(full))
        for card in answer(full)['result']['claim_cards']:
            self.assertFalse(card['claim_supported'])
            self.assertNotIn('statement_true', card)
        self.assertNotIn('comparison', answer(full)['result'])

    def test_reciprocal_observations_are_local_recorded_roles_without_feedback(self):
        pair = self.named[BRAIN, 'train', 'reciprocal']
        row = self.row(pair, BRAIN, 'reciprocal_context', question_kind='claim_support', claim='causal_effect')
        self.validate(row)
        self.assertFalse(calls(row))
        self.assertEqual([card['subject']['gene_ids'] for card in answer(row)['result']['claim_cards']],
                         [list(pair), list(reversed(pair))])
        for card in answer(row)['result']['claim_cards']:
            self.assertEqual(card['supported_claims'], ['regulatory_link_recorded', 'regulation_direction'])
            self.assertFalse(card['claim_supported'])

    def test_two_edge_ppi_route_does_not_create_a_direct_pair_record(self):
        pair = self.named[PPI, 'train', 'route']
        row = self.row(pair, PPI, 'route_context', question_kind='claim_support', claim='physical_interaction_recorded')
        self.validate(row)
        self.assertFalse(calls(row))
        self.assertEqual(len(context(row)['observations']), 2)
        self.assertEqual(self.card(row)['supported_claims'], [])
        self.assertEqual(len(answer(row)['provenance']['graph']), 2)

    def test_missing_prerequisite_record_can_complete_an_evidence_only_check(self):
        pair = self.pair[0], self.named['missing']
        row = self.row(pair, variant='context', question_kind='claim_support', claim='physical_interaction_recorded')
        self.validate(row)
        self.assertFalse(calls(row))
        self.assertEqual((answer(row)['status'], answer(row)['continuation_decision']), ('resolved', 'stop'))
        self.assertEqual(self.card(row)['supported_claims'], [])
        self.assertIsNone(context(row)['observations'][0]['result']['edge_exists'])
        self.assertNotIn('edge_exists', answer(row)['result'])

    def test_graph_truth_retains_its_separate_vocabulary_and_exact_literal(self):
        for pair, truth in ((self.pair, False), (self.named[PPI, 'train', 'route'], True)):
            row = self.row(pair, variant='context', question_kind='graph_truth', claim='physical_interaction_recorded')
            self.validate(row)
            self.assertFalse(calls(row))
            bio, graph = answer(row)['result']['claim_cards']
            self.assertEqual(bio['claim_vocabulary'], 's1_layer_v1')
            self.assertNotIn('statement_true', bio)
            self.assertEqual(graph['claim_vocabulary'], 's2_graph_observations_v1')
            self.assertEqual(graph['claim'], 'no_edge_recorded_in_scope')
            self.assertIs(graph['statement_true'], truth)
            self.assertEqual(graph['subject']['direction'], 'undirected')
            self.assertEqual(graph['supported_claims'] + graph['unsupported_claims'],
                             [claim for claim in GRAPH_CLAIMS if claim in graph['supported_claims']] +
                             [claim for claim in GRAPH_CLAIMS if claim not in graph['supported_claims']])
            self.assertIn(self.view.graph_version, row['metadata']['statement'])

    def test_biological_truth_remains_unknown_for_positive_and_negative_edges(self):
        for pair in (self.pair, self.named[PPI, 'train', 'route']):
            row = self.row(pair, variant='context', question_kind='biological_truth', claim='causal_effect')
            self.validate(row)
            self.assertEqual((answer(row)['status'], answer(row)['continuation_decision']), ('invalid_input', 'defer'))
            self.assertEqual(answer(row)['result']['issue_code'], 'UNSUPPORTED_TRUTH_SCOPE')
            self.assertEqual(answer(row)['result']['partial_results'], context(row)['observations'])
            self.assertNotIn('claim_cards', answer(row)['result'])
            self.assertFalse(calls(row))

    def test_cold_symbols_and_versioned_ids_resolve_both_identities_before_graph(self):
        for variant in ('symbol_lookup', 'symbol_case_lookup', 'id_lookup'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), 3)
            self.assertTrue(all(call['name'].startswith('lookup_human_') for call in calls(row)[:2]))
            self.assertEqual(calls(row)[2]['arguments']['gene_ids'], list(self.pair))
            self.assertEqual(len(answer(row)['provenance']['registry']), 2)
        raw = self.row(variant='symbol_case_lookup')['metadata']['request']['genes']
        self.assertTrue(all(value.startswith(' ') and value.endswith(' ') and value == value.lower() for value in raw))

    def test_supplied_mappings_and_null_symbols_avoid_redundant_lookups(self):
        for variant in ('symbol_context', 'id_context'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual([call['name'] for call in calls(row)], ['get_edge'])
            self.assertEqual(len(answer(row)['provenance']['registry']), 2)
        row = self.row(self.named['null_edge'], CD4, 'null_symbol_context')
        self.validate(row)
        self.assertEqual([call['name'] for call in calls(row)], ['get_edge'])
        self.assertIsNone(context(row)['observations'][0]['result']['gene_symbol'])
        self.assertTrue(context(row)['observations'][0]['arguments']['gene_id'].endswith('.4'))

    def test_normalized_supplied_mappings_retain_original_queries(self):
        for variant in ('symbol_context', 'id_context'):
            row = self.row(variant=variant)
            request = deepcopy(row['metadata']['request'])
            request['genes'] = [' ' + value.lower() + ' ' for value in request['genes']]
            task = {key: row['metadata'][key] for key in corpus.TASK_FIELDS}
            observations = context(row)['observations']
            proof, used = corpus._plan(self.view, request, observations, task)
            self.assertEqual(proof.next_tool, 'get_edge')
            self.assertEqual(proof.next_arguments['gene_ids'], list(self.pair))
            self.assertEqual(used, observations)

    def test_required_failure_keeps_null_coverage_and_successful_identity_partials(self):
        for variant, attempts, status in (('symbol_terminal', 1, 'error'),
                                          ('symbol_exhausted', 2, 'error'), ('symbol_recovered', 2, 'resolved')):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), attempts)
            self.assertTrue(all(call == calls(row)[0] for call in calls(row)))
            if status == 'error':
                result = answer(row)['result']
                self.assertEqual(set(result), {'failed_tool', 'error_code', 'arguments', 'partial_results', 'coverage'})
                self.assertIsNone(result['coverage'])
                self.assertEqual(result['partial_results'], context(row)['observations'])
                self.assertNotIn('claim_cards', result)
            self.assertEqual(answer(row)['status'], status)

    def test_failed_next_step_retains_original_completed_graph_partial(self):
        row = self.row(variant='partial_exhausted')
        self.validate(row)
        self.assertEqual(len(calls(row)), 2)
        self.assertEqual(calls(row)[0], calls(row)[1])
        self.assertEqual(answer(row)['result']['partial_results'], context(row)['observations'])
        self.assertEqual(answer(row)['provenance'], context(row)['observations'][0]['provenance'])
        self.assertIsNone(answer(row)['result']['coverage'])

    def test_all_diagnostics_keep_raw_four_fields_and_actual_rejected_calls(self):
        for kind in corpus.DIAGNOSTICS:
            registry = self.registry
            fact = self.fact()
            if kind in ('ambiguous_identity', 'unresolved_identity', 'invalid_identity'):
                registry = load_registry_context(self.contract['registry_index'])
                fact = self.fact(registry=registry)
            row = corpus.build_diagnostic(self.view, fact, 'direct', 0,
                self.bank, self.contract, kind=kind, registry=registry, seed=SEED)
            self.validate(row, registry=registry)
            result = answer(row)['result']
            self.assertEqual(set(result), {'issue_code', 'request', 'rejected_arguments', 'partial_results'})
            self.assertEqual(set(result['request']), {'genes', 'multiplex_id', 'layer_id', 'direction'})
            self.assertEqual(result['request'], row['metadata']['request'])
            self.assertEqual(answer(row)['continuation_decision'], 'defer')
            self.assertNotIn('claim_cards', result)
            if kind.startswith('tool_'):
                self.assertEqual(result['rejected_arguments'], calls(row)[0]['arguments'])
                self.assertEqual(answer(row)['provenance'], {'registry': [], 'graph': []})
            elif kind.endswith('identity'):
                self.assertEqual(len(calls(row)), 1)
                self.assertEqual(len(answer(row)['provenance']['registry']), 1)
            else:
                self.assertFalse(calls(row))
                self.assertIsNone(result['rejected_arguments'])

    def test_serialized_json_rows_remain_valid(self):
        for row in (self.row(), self.row(variant='symbol_context'), self.row(variant='partial_exhausted'),
                    corpus.build_family_row(self.view, 'other', 'validation', 'direct', 0, self.bank, self.contract)):
            self.validate(json.loads(json.dumps(row)))

    def test_free_reasoning_summary_and_nonempty_blocker_text_are_not_scored(self):
        for row in (self.row(), self.row(variant='partial_terminal')):
            for message in row['messages']:
                if message['role'] == 'assistant':
                    message['thinking'] = 'A separate valid explanation of the visible observations.'
            answer(row)['summary'] = 'An independently phrased scoped result.'
            if answer(row)['blocked_reason']:
                answer(row)['blocked_reason'] = 'The required next acquisition must recover.'
            self.validate(row)

    def test_validation_rejects_changed_calls_source_records_claims_and_subjects(self):
        original = self.row()
        mutations = [lambda row: calls(row)[0]['arguments'].update(direction='undirected'),
                     lambda row: row['messages'][4]['content']['result'].update(weight=.333),
                     lambda row: self.card(row).update(supported_claims=['causal_effect']),
                     lambda row: self.card(row)['subject'].update(graph_version='0' * 64),
                     lambda row: self.card(row).update(statement_true=False),
                     lambda row: row['metadata'].update(weight=.333),
                     lambda row: row.update(row_id='s213_changed')]
        for mutation in mutations:
            row = deepcopy(original)
            mutation(row)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(row)

    def test_validation_rejects_changed_truth_literal_and_hidden_source_completion(self):
        row = self.row(variant='context', question_kind='graph_truth', claim='physical_interaction_recorded')
        row['metadata']['statement'] = 'An unrelated biological proposition.'
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='family_context', question_kind='claim_support', claim='physical_interaction_recorded')
        card = self.card(row)
        card['supported_claims'] = ['physical_interaction_recorded']
        card['unsupported_claims'].remove('physical_interaction_recorded')
        card['claim_supported'] = True
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_zero_weight_and_reversed_directed_records_keep_native_meaning(self):
        row = self.row(self.named['zero'])
        self.validate(row)
        self.assertTrue(answer(row)['result']['edge_exists'])
        self.assertEqual(answer(row)['result']['weight'], 0)
        self.assertEqual(self.card(row)['supported_claims'], ['physical_interaction_recorded'])
        pair = self.named[CD4, 'train', 'reciprocal'][::-1]
        row = self.row(pair, CD4)
        self.validate(row)
        self.assertEqual(calls(row)[0]['arguments']['gene_ids'], list(pair))
        self.assertEqual(self.card(row)['subject']['gene_ids'], list(pair))
        self.assertEqual(answer(row)['result']['weight'], .53)

    def test_validation_rejects_changed_retry_allowance_or_identical_pair_reordering(self):
        row = self.row(variant='recovered')
        row['messages'][4]['content']['message'] = 'The edge backend is unavailable. retries_remaining=0'
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='recovered')
        calls(row)[1]['arguments']['gene_ids'].reverse()
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_validation_rejects_relabelled_source_and_redundant_native_acquisition(self):
        row = self.row(variant='old_view_context')
        payload = context(row)
        payload['observations'][0]['provenance']['graph'][0]['graph_version'] = self.view.graph_version
        row['messages'][1]['content'] = json.dumps(payload)
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='context')
        acquired = self.row()
        row['messages'][3:3] = deepcopy(acquired['messages'][3:5])
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_blocker_and_reasoning_presence_remain_required_without_word_scoring(self):
        row = self.row(variant='terminal')
        answer(row)['blocked_reason'] = ''
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row()
        row['messages'][-1]['thinking'] = ''
        with self.assertRaises(ValueError):
            self.validate(row)


if __name__ == '__main__':
    unittest.main()
