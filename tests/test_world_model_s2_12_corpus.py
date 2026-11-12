"""Check calibrated singleton traces, source replay, identities, and failure states."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_12_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext, load_registry_context
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_12_sampling import fact_from_gene
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028
PPI, COEX, BRAIN, CD4 = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression',
                         'TFs:brain', 'TFs:cd4+t_cells')
BULK, SC = 'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte'


def calls(row):
    """Read new calls separately from supplied observations."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def context(row):
    """Read original supplied observations and active graph metadata."""
    return json.loads(row['messages'][1]['content'])


def answer(row):
    """Read the final structured envelope."""
    return row['messages'][-1]['content']


class SingletonCorpusTests(unittest.TestCase):
    """Use native endpoint motifs rather than fabricated full-layer isolates."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 200)
                if gene_group(f'ENSG{number:011d}', SEED)[1] == 'train']
        (cls.a, cls.b, cls.c, cls.pen, cls.peer, cls.loop,
         cls.graph_only, cls.null_symbol, cls.missing) = pool[:9]
        a, b, c, pen, peer, loop, graph, null, missing = pool[:9]
        rows = {
            PPI: {(a, b): 0, (a, graph): .25}, COEX: {(a, c): .5, (b, c): .6},
            BRAIN: {(a, b): .7, (loop, loop): .9}, CD4: {(a, null): .4},
            BULK: {(a, pen): .5, (pen, peer): .6}, SC: {(a, pen): .3, (pen, peer): .2},
        }
        native = {}
        for layer, edges in rows.items():
            directed = layer in (BRAIN, CD4)
            if not directed:
                edges = {tuple(sorted(pair)): weight for pair, weight in edges.items()}
            native[layer] = LayerGraph(layer, directed, edges,
                frozenset(gene for pair in edges for gene in pair))
        cls.view = GraphView('unit_s212_six_files', '6' * 64, native)
        cls.old = GraphView('unit_s212_four_files', '4' * 64,
            {layer: graph for layer, graph in native.items() if layer not in (BULK, SC)})
        cls.registry = RegistryContext('116', 'ensembl_homo_sapiens_116_unit', {
            gene: None if gene == null else f'UNIT_MEMBER_{ordinal}'
            for ordinal, gene in enumerate(pool[:9]) if gene != graph})
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contract = corpus.load_build_contract()

    def setUp(self):
        self.index = SubgraphIndex(self.view)

    def fact(self, gene=None, layer=PPI, registry=None):
        """Measure one fixture fact under the explicit old and current inventories."""
        return fact_from_gene(self.view, self.index, registry or self.registry, self.old,
                              layer, self.a if gene is None else gene, SEED)

    def row(self, gene=None, layer=PPI, variant='lookup', framing='direct', **task):
        """Build one trace using the real frozen prompt banks and stage overlay."""
        return corpus.build_row(self.view, self.fact(gene, layer), framing, variant, 0,
            self.bank, self.contract, index=self.index, registry=self.registry,
            old_view=self.old, seed=SEED, **task)

    def validate(self, row):
        """Replay original acquisitions and exact structured fields."""
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
            index=self.index, registry=self.registry, old_view=self.old)

    def card(self, row):
        """Read the singleton subject's sole claim card."""
        return answer(row)['result']['claim_cards'][0]

    def test_three_categories_resolve_and_follow_the_requested_continuation(self):
        for gene, label in ((self.a, 'layer_gene_ids'), (self.c, 'not_in_layer_gene_ids'),
                            (self.missing, 'missing_gene_ids')):
            for framing, decision in (('direct', 'stop'), ('step', 'continue')):
                with self.subTest(gene=gene, framing=framing):
                    row = self.row(gene, framing=framing)
                    self.validate(row)
                    final, result = answer(row), answer(row)['result']
                    self.assertEqual((final['status'], final['continuation_decision']), ('resolved', decision))
                    self.assertIsNone(final['blocked_reason'])
                    self.assertEqual(result[label], [gene])
                    self.assertEqual(result['layer_gene_count'], int(gene == self.a))
                    self.assertEqual(result['coverage']['checked_gene_ids'], [gene])
                    self.assertEqual(result['coverage']['unchecked_gene_ids'], [])

    def test_node_call_has_only_the_three_existing_arguments(self):
        row = self.row()
        self.validate(row)
        self.assertEqual(calls(row), [{'name': 'get_nodes_by_layer', 'arguments': {
            'multiplex_id': self.view.multiplex_id, 'layer_id': PPI, 'gene_ids': [self.a]}}])
        card = self.card(row)
        self.assertEqual(card['subject'], {'gene_id': self.a, 'multiplex_id': self.view.multiplex_id, 'layer_id': PPI})
        self.assertEqual(card['supported_claims'], [])
        self.assertEqual(card['unsupported_claims'], list(CLAIM_VOCABULARY))
        self.assertNotIn('direction', card['subject'])

    def test_layer_and_global_negatives_have_distinct_closed_permissions(self):
        outside, missing = self.row(self.c), self.row(self.missing)
        self.validate(outside)
        self.validate(missing)
        self.assertEqual(self.card(outside)['supported_claims'],
                         [*CLAIM_VOCABULARY[:2], 'gene_not_node_in_layer'])
        self.assertEqual(self.card(missing)['supported_claims'], ['gene_not_node_in_multiplex'])

    def test_all_32_variants_have_attainable_fixture_coverage(self):
        subjects = {'prefix_omission_context': (self.a, SC), 'old_view_context': (self.pen, BULK),
                    'neighbor_target_context': (self.b, BRAIN), 'neighbor_context': (self.b, BRAIN),
                    'null_symbol_context': (self.null_symbol, CD4)}
        for variant in corpus.VARIANTS:
            with self.subTest(variant=variant):
                gene, layer = subjects.get(variant, (self.a, PPI))
                row = self.row(gene, layer, variant)
                self.validate(row)
                self.assertEqual(row['metadata']['variant'], variant)
        self.assertEqual(len(corpus.VARIANTS), 32)

    def test_inapplicable_contexts_do_not_invent_source_witnesses(self):
        cases = [(self.missing, PPI, 'prefix_context'), (self.loop, BRAIN, 'edgeless_context'),
                 (self.loop, BRAIN, 'edge_context'), (self.loop, BRAIN, 'neighbor_target_context'),
                 (self.graph_only, PPI, 'symbol_lookup'), (self.a, PPI, 'null_symbol_context')]
        for gene, layer, variant in cases:
            with self.subTest(variant=variant), self.assertRaises(VariantUnavailable):
                self.row(gene, layer, variant)

    def test_matching_superset_and_duplicate_contexts_need_no_new_call(self):
        for variant in ('context', 'duplicate_context', 'superset_context', 'layer_list_context', 'family_context'):
            with self.subTest(variant=variant):
                row = self.row(variant=variant)
                self.validate(row)
                self.assertEqual(calls(row), [])
                self.assertEqual(answer(row)['result']['coverage']['requested_gene_ids'], [self.a])
                self.assertEqual(len(answer(row)['provenance']['graph']), 1)

    def test_an_omitted_top_one_positive_needs_acquisition_but_a_returned_positive_does_not(self):
        omitted = self.row(self.a, SC, 'prefix_omission_context')
        shown = self.row(self.a, PPI, 'prefix_positive_context')
        for row in (omitted, shown):
            self.validate(row)
        self.assertEqual(len(calls(omitted)), 1)
        self.assertFalse(context(omitted)['observations'][0]['result']['collection']['complete'])
        self.assertEqual(calls(shown), [])
        self.assertEqual(answer(omitted)['result']['layer_gene_ids'], [self.a])

    def test_claim_only_bounded_omission_does_not_acquire_or_fabricate_membership(self):
        row = self.row(self.a, SC, 'prefix_omission_context', question_kind='claim_support',
                       claim='gene_not_node_in_layer')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(set(answer(row)['result']), {'gene_id', 'multiplex_id', 'layer_id', 'claim_cards'})
        self.assertEqual(self.card(row)['supported_claims'], [])
        self.assertFalse(self.card(row)['claim_supported'])

    def test_other_layer_and_family_do_not_settle_target_membership(self):
        for variant in ('wrong_layer_context', 'wrong_family_context', 'distractor_context', 'stale'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), 1)
        supplied_only = self.row(variant='wrong_family_context', question_kind='claim_card')
        self.validate(supplied_only)
        self.assertEqual(calls(supplied_only), [])
        self.assertNotIn('layer_gene_ids', answer(supplied_only)['result'])

    def test_old_view_cannot_transfer_even_when_the_target_source_file_is_unchanged(self):
        for gene, layer in ((self.a, PPI), (self.pen, BULK)):
            row = self.row(gene, layer, 'old_view_context')
            self.validate(row)
            saved = context(row)['observations'][0]
            self.assertEqual(saved['arguments']['multiplex_id'], self.old.multiplex_id)
            self.assertEqual(saved['provenance']['graph'][0]['graph_version'], self.old.graph_version)
            self.assertEqual(len(calls(row)), 1)
            self.assertEqual(answer(row)['result']['layer_gene_ids'], [gene])
            self.assertEqual(answer(row)['provenance']['graph'][0]['graph_version'], self.view.graph_version)
        changed = context(self.row(self.pen, BULK, 'old_view_context'))['observations'][0]
        self.assertEqual(changed['result']['missing_gene_ids'], [self.pen])

    def test_zero_outgoing_and_edgeless_selected_nodes_remain_members(self):
        zero = self.row(self.b, BRAIN, 'neighbor_context')
        empty = self.row(self.a, PPI, 'edgeless_context')
        for row in (zero, empty):
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(len(answer(row)['result']['layer_gene_ids']), 1)
            self.assertEqual(self.card(row)['supported_claims'], [])
        self.assertEqual(context(zero)['observations'][0]['result']['neighbors'], [])
        self.assertEqual(context(empty)['observations'][0]['result']['edges'], [])
        self.assertGreater(len(self.view.layers[PPI].edges), 0)

    def test_zero_weight_positive_edge_context_preserves_its_native_record(self):
        row = self.row(variant='edge_context')
        self.validate(row)
        saved = context(row)['observations'][0]
        self.assertTrue(saved['result']['edge_exists'])
        self.assertEqual(saved['result']['weight'], 0)
        self.assertEqual(calls(row), [])

    def test_category_truth_keeps_the_multiplex_membership_premise_explicit(self):
        for gene, truth in ((self.c, True), (self.a, False), (self.missing, False)):
            row = self.row(gene, question_kind='membership_truth', claim='gene_not_node_in_layer', truth_scope='category')
            self.validate(row)
            self.assertIs(self.card(row)['statement_true'], truth)
            self.assertIn('belongs to multiplex', row['metadata']['statement'])
        global_absence = self.row(self.missing, question_kind='membership_truth',
                                  claim='gene_not_node_in_multiplex', truth_scope='multiplex')
        self.validate(global_absence)
        self.assertTrue(self.card(global_absence)['claim_supported'])
        self.assertTrue(self.card(global_absence)['statement_true'])

    def test_positive_global_membership_refutes_absence_from_the_layer_union(self):
        row = self.row(self.c, variant='context', question_kind='truth',
                       claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertFalse(self.card(row)['statement_true'])
        self.assertFalse(self.card(row)['claim_supported'])
        self.assertEqual(context(row)['observations'][0]['result']['not_in_layer_gene_ids'], [self.c])

    def test_single_layer_family_truth_is_complete_but_other_named_layers_are_unknown(self):
        family = self.row(self.c, variant='context', question_kind='truth',
                          claim='absence_holds_in_other_layers', truth_scope='family', truth_family='ppi')
        unknown = self.row(self.c, variant='context', question_kind='truth',
                           claim='absence_holds_in_other_layers', truth_scope='layers', truth_layer_ids=[BULK, SC])
        for row in (family, unknown):
            self.validate(row)
            self.assertEqual(calls(row), [])
        self.assertTrue(self.card(family)['statement_true'])
        self.assertEqual(answer(unknown)['result']['issue_code'], 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertNotIn('claim_cards', answer(unknown)['result'])
        self.assertEqual(answer(unknown)['result']['partial_results'], context(unknown)['observations'])

    def test_expression_and_future_truth_remain_unknown_for_every_membership_category(self):
        for gene in (self.a, self.c, self.missing):
            for scope, claim in (('expression', 'gene_not_expressed_in_context'),
                                 ('future_versions', 'absence_holds_in_other_versions')):
                row = self.row(gene, variant='context', question_kind='truth', claim=claim, truth_scope=scope)
                self.validate(row)
                self.assertEqual((answer(row)['status'], answer(row)['continuation_decision']), ('invalid_input', 'defer'))
                self.assertEqual(answer(row)['result']['issue_code'], 'UNSUPPORTED_TRUTH_SCOPE')
                self.assertEqual(calls(row), [])

    def test_explicit_outer_gate_preserves_the_successful_ordinary_classification(self):
        for gene, gate, label in ((self.c, 'layer', 'not_in_layer_gene_ids'),
                                  (self.missing, 'multiplex', 'missing_gene_ids')):
            row = self.row(gene, variant='context', require_membership=gate)
            self.validate(row)
            final = answer(row)
            self.assertEqual((final['status'], final['continuation_decision']), ('not_found', 'defer'))
            self.assertEqual(final['result']['issue_code'], 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
            partial = final['result']['partial_results'][0]
            self.assertEqual(partial['status'], 'resolved')
            self.assertEqual(partial['result'][label], [gene])
            self.assertNotIn('failed_tool', final['result'])
            self.assertNotIn('claim_cards', final['result'])

    def test_retry_recovery_and_exhaustion_preserve_exact_arguments_and_unchecked_coverage(self):
        for prefix in ('', 'symbol_'):
            for outcome in ('terminal', 'recovered', 'exhausted'):
                row = self.row(self.c, variant=prefix + outcome, framing='step')
                self.validate(row)
                trace_calls = calls(row)
                self.assertEqual(len(trace_calls), 1 if outcome == 'terminal' else 2)
                if len(trace_calls) == 2:
                    self.assertEqual(trace_calls[0], trace_calls[1])
                final = answer(row)
                if outcome == 'recovered':
                    self.assertEqual((final['status'], final['continuation_decision']), ('resolved', 'continue'))
                    self.assertEqual(final['result']['not_in_layer_gene_ids'], [self.c])
                else:
                    self.assertEqual(final['status'], 'error')
                    self.assertEqual(final['result']['coverage'], {'operation': 'layer_membership',
                        'requested_gene_ids': [self.c], 'checked_gene_ids': [], 'unchecked_gene_ids': [self.c]})
                    self.assertNotIn('claim_cards', final['result'])
                    self.assertEqual(len(final['result']['partial_results']), int(bool(prefix)))

    def test_symbols_and_versioned_ids_resolve_before_graph_calls(self):
        for variant in ('symbol_lookup', 'symbol_case_lookup', 'id_lookup'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), 2)
            self.assertTrue(calls(row)[0]['name'].startswith('lookup_human_'))
            self.assertEqual(calls(row)[1]['name'], 'get_nodes_by_layer')
            self.assertEqual(calls(row)[1]['arguments']['gene_ids'], [self.a])
            self.assertEqual(len(answer(row)['provenance']['registry']), 1)
        literal = self.row(variant='symbol_case_lookup')['metadata']['request']['genes'][0]
        self.assertEqual(literal, ' ' + self.registry.symbols_by_id[self.a].lower() + ' ')

    def test_supplied_authoritative_mappings_avoid_repeated_identity_lookups(self):
        for variant in ('symbol_context', 'id_context'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual([call['name'] for call in calls(row)], ['get_nodes_by_layer'])
            self.assertEqual(len(answer(row)['provenance']['registry']), 1)
        row = self.row(self.null_symbol, CD4, 'null_symbol_context')
        self.validate(row)
        saved = context(row)['observations'][0]
        self.assertEqual(saved['arguments']['gene_id'], self.null_symbol + '.4')
        self.assertEqual(saved['result'], {'gene_id': self.null_symbol, 'gene_symbol': None})
        self.assertEqual(answer(row)['result']['layer_gene_ids'], [self.null_symbol])

    def test_local_diagnostics_preserve_literal_scope_and_wrong_singleton_shape(self):
        for kind in corpus.DIAGNOSTICS:
            if kind.endswith('identity'):
                continue
            row = corpus.build_diagnostic(self.view, self.fact(), 'direct', 0, self.bank,
                                           self.contract, kind=kind, registry=self.registry)
            self.validate(row)
            final = answer(row)
            self.assertEqual(final['continuation_decision'], 'defer')
            self.assertNotIn('claim_cards', final['result'])
            if kind.startswith('tool_'):
                self.assertEqual(len(calls(row)), 1)
                self.assertEqual(final['result']['rejected_arguments'], calls(row)[0]['arguments'])
            else:
                self.assertEqual(calls(row), [])
                self.assertIsNone(final['result']['rejected_arguments'])
            if kind == 'invalid_gene_list':
                self.assertEqual(final['result']['request']['genes'], [self.a, self.a])
                self.assertEqual(final['result']['issue_code'], 'INVALID_GENE_LIST')

    def test_validation_rejects_changed_source_calls_cards_labels_and_row_identity(self):
        mutations = [lambda row: row.__setitem__('row_id', 's212_changed'),
            lambda row: row['metadata'].__setitem__('old_present', False),
            lambda row: answer(row)['result'].__setitem__('node_in_layer', True),
            lambda row: calls(row)[0]['arguments'].__setitem__('gene_ids', [self.missing]),
            lambda row: row['messages'][-1].__setitem__('private_fact', True),
            lambda row: self.card(row)['supported_claims'].append('gene_not_expressed_in_context')]
        for mutate in mutations:
            row = self.row()
            mutate(row)
            with self.assertRaises((ValueError, AssertionError)):
                self.validate(row)
        for field, value in (('weight', 0.1), ('edge_exists', False)):
            row = self.row(variant='edge_context')
            saved = context(row)
            saved['observations'][0]['result'][field] = value
            row['messages'][1]['content'] = json.dumps(saved)
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_validation_rejects_relabelled_provenance_and_changed_literal_truth(self):
        row = self.row(variant='context')
        saved = context(row)
        saved['observations'][0]['provenance']['graph'][0]['evidence_id'] = 'invented_acquisition'
        row['messages'][1]['content'] = json.dumps(saved)
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(question_kind='membership_truth')
        row['metadata']['statement'] = 'An unrelated biological statement.'
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_validation_rejects_unallowed_retries_and_checked_failure_coverage(self):
        row = self.row(variant='recovered')
        failures = [message for message in row['messages']
                    if message['role'] == 'tool' and message['content']['status'] == 'error']
        failures[0]['content']['retries_remaining'] = 0
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='exhausted')
        calls(row)[1]['arguments']['gene_ids'] = [self.missing]
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='terminal')
        answer(row)['result']['coverage']['checked_gene_ids'] = [self.a]
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_validation_rejects_hidden_source_completion_and_redundant_acquisitions(self):
        row = self.row()
        del row['messages'][3:-1]
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row()
        row['messages'][-1:-1] = deepcopy(row['messages'][3:5])
        with self.assertRaises(ValueError):
            self.validate(row)
        row = self.row(variant='context')
        saved = context(row)
        saved['private_membership'] = True
        row['messages'][1]['content'] = json.dumps(saved)
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_a_graph_only_canonical_id_needs_no_registry_lookup(self):
        row = self.row(self.graph_only)
        self.validate(row)
        self.assertEqual([call['name'] for call in calls(row)], ['get_nodes_by_layer'])
        self.assertEqual(answer(row)['result']['layer_gene_ids'], [self.graph_only])
        self.assertEqual(answer(row)['provenance']['registry'], [])

    def test_lowercase_versioned_id_preserves_raw_lookup_and_null_symbol(self):
        raw = self.null_symbol.lower() + '.4'
        request = {'genes': [raw], 'multiplex_id': self.view.multiplex_id, 'layer_id': CD4}
        task = corpus.validate_task(self.view, self.fact(self.null_symbol, CD4))
        proof, _ = corpus._plan(self.view, request, [], task)
        self.assertEqual(proof.next_tool, 'lookup_human_ensembl_gene_id')
        self.assertEqual(proof.next_arguments, {'gene_id': raw})
        response = corpus._identity_output(self.registry, self.contract['registry_index'],
            proof.next_tool, raw, 'unit_lowercase_identity')
        identity = corpus._identity_record(proof.next_tool, proof.next_arguments, response)
        self.assertEqual(identity['arguments'], {'gene_id': raw})
        self.assertEqual(identity['result'], {'gene_id': self.null_symbol, 'gene_symbol': None})
        proof, _ = corpus._plan(self.view, request, [identity], task)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments['gene_ids'], [self.null_symbol])
        response = NodesByLayerRuntime(self.view).get_nodes_by_layer(**proof.next_arguments,
            evidence_id='unit_lowercase_membership')
        graph = {'operation': proof.next_tool, 'arguments': proof.next_arguments, **response}
        proof, retained = corpus._plan(self.view, request, [identity, graph], task)
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['layer_gene_ids'], [self.null_symbol])
        self.assertIsNone(retained[0]['result']['gene_symbol'])

    def test_supplied_symbol_mapping_reuses_case_and_whitespace_variants(self):
        symbol = self.registry.symbols_by_id[self.a]
        operation = 'lookup_human_gene_symbol'
        response = corpus._identity_output(self.registry, self.contract['registry_index'],
            operation, symbol, 'unit_supplied_symbol')
        identity = corpus._identity_record(operation, {'gene_symbol': symbol}, response)
        original = deepcopy(identity)
        request = {'genes': [' ' + symbol.lower() + ' '],
                   'multiplex_id': self.view.multiplex_id, 'layer_id': PPI}
        task = corpus.validate_task(self.view, self.fact())
        proof, retained = corpus._plan(self.view, request, [identity], task)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments['gene_ids'], [self.a])
        self.assertEqual(retained, [original])
        self.assertEqual(identity, original)
        self.assertEqual(request['genes'], [' ' + symbol.lower() + ' '])

    def test_supplied_versioned_id_mapping_reuses_lowercase_raw_input(self):
        canonical_raw = self.null_symbol + '.4'
        operation = 'lookup_human_ensembl_gene_id'
        response = corpus._identity_output(self.registry, self.contract['registry_index'],
            operation, canonical_raw, 'unit_supplied_versioned_id')
        identity = corpus._identity_record(operation, {'gene_id': canonical_raw}, response)
        original = deepcopy(identity)
        request = {'genes': [canonical_raw.lower()],
                   'multiplex_id': self.view.multiplex_id, 'layer_id': CD4}
        task = corpus.validate_task(self.view, self.fact(self.null_symbol, CD4))
        proof, retained = corpus._plan(self.view, request, [identity], task)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments['gene_ids'], [self.null_symbol])
        self.assertEqual(retained, [original])
        self.assertEqual(identity, original)
        self.assertIsNone(retained[0]['result']['gene_symbol'])

    def test_validation_allows_free_reasoning_summary_and_nonempty_blocker_prose(self):
        for row in (self.row(), self.row(variant='terminal')):
            for message in row['messages']:
                if message['role'] == 'assistant':
                    message['thinking'] = 'Use the visible evidence and preserve its exact scope.'
            answer(row)['summary'] = 'This response preserves the evidence and the requested scope.'
            if answer(row)['blocked_reason'] is not None:
                answer(row)['blocked_reason'] = 'The required graph service has not supplied this classification.'
            self.validate(row)

    def test_pinned_shox_ambiguity_does_not_choose_the_candidate_with_membership(self):
        registry = load_registry_context(self.contract['registry_index'])
        candidates = registry.gene_ids_by_symbol['shox']
        self.assertEqual(len(candidates), 2)
        native = dict(self.view.layers)
        edge = tuple(sorted((self.a, candidates[0])))
        native[PPI] = LayerGraph(PPI, False, {edge: .5}, frozenset(edge))
        view = GraphView('unit_ambiguous_singleton', 'a' * 64, native)
        index = SubgraphIndex(view)
        fact = fact_from_gene(view, index, registry, self.old, PPI, self.a, SEED)
        self.assertIn(candidates[0], view.multiplex_nodes)
        self.assertNotIn(candidates[1], view.multiplex_nodes)
        for kind, status in (('ambiguous_identity', 'ambiguous'), ('unresolved_identity', 'not_found'),
                             ('invalid_identity', 'invalid_input')):
            row = corpus.build_diagnostic(view, fact, 'direct', 0, self.bank, self.contract,
                                           kind=kind, registry=registry)
            corpus.validate_row(row, view, bank=self.bank, contract=self.contract,
                                index=index, registry=registry, old_view=self.old)
            self.assertEqual(answer(row)['status'], status)
            self.assertEqual(len(calls(row)), 1)
            self.assertTrue(calls(row)[0]['name'].startswith('lookup_human_'))
            if kind == 'ambiguous_identity':
                partial = answer(row)['result']['partial_results'][0]
                self.assertEqual(partial['result']['candidate_gene_ids'], list(candidates))
                self.assertEqual(partial['result']['candidate_details'][candidates[0]]['chromosome'], 'X')
                self.assertEqual(partial['result']['candidate_details'][candidates[1]]['chromosome'], 'Y')


if __name__ == '__main__':
    unittest.main()
