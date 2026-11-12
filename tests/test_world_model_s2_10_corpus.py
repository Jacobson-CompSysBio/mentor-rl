"""Check complete component rows from visible memberships, native graphs, and proofs."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from runtime.world_model import s2_10_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_10_sampling import fact_from_genes
from runtime.world_model.s2_components import components
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import ARGUMENT_KEYS, SubgraphIndex
from runtime.world_model.sources import file_sha256


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


def calls(row):
    """Read only the ordered current-turn calls."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def context(row):
    """Read the supplied active graph and attributed observations."""
    return json.loads(row['messages'][1]['content'])


def answer(row):
    """Read the final structured envelope."""
    return row['messages'][-1]['content']


class ComponentCorpusTests(unittest.TestCase):
    """Use source motifs that distinguish exact groups from provisional previews."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 180)
                if gene_group(f'ENSG{number:011d}', SEED)[1] == 'train']
        (cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.outside, cls.isolated,
         cls.graph_only, cls.missing, cls.x, cls.y, cls.z) = pool[:13]
        a, b, c, d, e, f, outside, isolated, graph_only, _, x, y, z = pool[:13]
        cls.view = GraphView('unit_component_corpus', 'c' * 64, {
            PPI: LayerGraph(PPI, False, {(a, a): .95, (a, b): .8, (a, c): .7, (b, c): 0,
                (c, d): .3, (e, graph_only): .2}, frozenset((a, b, c, d, e, f, isolated, graph_only))),
            COEX: LayerGraph(COEX, False, {(a, outside): .4, (d, outside): .6}, frozenset((outside, a, d))),
            BRAIN: LayerGraph(BRAIN, True, {(a, b): .8, (a, c): .7, (b, a): .6, (c, a): .5,
                (c, c): .9, (b, e): .4, (e, f): .3, (f, e): .2, (x, y): .1, (y, z): 0},
                frozenset((a, b, c, d, e, f, isolated, graph_only, x, y, z))),
            CD4: LayerGraph(CD4, True, {(outside, a): .1}, frozenset((outside, a))),
        })
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_COMPONENT_CORPUS_{ordinal}' for ordinal, gene in enumerate(pool) if gene != graph_only})
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contract = corpus.load_build_contract()

    def setUp(self):
        self.index = SubgraphIndex(self.view)

    def fact(self, genes=None, layer=PPI, kind=None):
        return fact_from_genes(self.view, self.index, layer,
            [self.a, self.b, self.c] if genes is None else genes, kind, SEED)

    def row(self, genes=None, layer=PPI, kind=None, *, variant='lookup', framing='direct',
            variant_index=0, question='partition', gate=False, membership_first=False, bank=None):
        return corpus.build_row(self.view, self.fact(genes, layer, kind), framing, variant, variant_index,
            bank or self.bank, self.contract, index=self.index, registry=self.registry,
            question_kind=question, require_all_in_layer=gate, membership_first=membership_first, seed=SEED)

    def validate(self, row):
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
                            registry=self.registry, index=self.index)

    def assert_source_result(self, row):
        self.validate(row)
        result = answer(row)['result']
        if answer(row)['status'] != 'resolved':
            return
        layer, selected = result['layer_id'], result['coverage']['requested_gene_ids']
        partition = self.index.partition(layer, selected)
        native = self.index.edges(layer, selected)
        self.assertEqual(result['components'], components(partition['layer_gene_ids'], native, result['connectivity_kind']))
        for key, value in partition.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['component_count'], len(result['components']))
        self.assertEqual(result['single_component'], len(result['components']) == 1
                         and not partition['not_in_layer_gene_ids'] and not partition['missing_gene_ids'])
        self.assertEqual(result['collection'], {'mode': 'complete', 'returned_count': result['component_count'],
            'total_count': result['component_count'], 'complete': True, 'next_cursor': None})
        self.assertNotIn('edges', result)
        self.assertNotIn('edge_count', result)

    def test_native_acquisition_keeps_six_arguments_and_ignores_weights_locally(self):
        row = self.row()
        self.assert_source_result(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(tuple(calls(row)[0]['arguments']), ARGUMENT_KEYS)
        self.assertEqual(calls(row)[0]['name'], 'induce_subgraph')
        self.assertNotIn('connectivity_kind', calls(row)[0]['arguments'])
        source = next(message['content']['result'] for message in row['messages'] if message['role'] == 'tool')
        self.assertIn({'gene_ids': [self.b, self.c], 'weight': 0}, source['edges'])
        self.assertEqual(answer(row)['result']['components'], [[self.a, self.b, self.c]])

    def test_boolean_question_still_returns_the_full_exact_partition(self):
        partition = self.row([self.a, self.b, self.isolated], question='partition')
        boolean = self.row([self.a, self.b, self.isolated], question='single_component')
        self.assert_source_result(partition)
        self.assert_source_result(boolean)
        self.assertEqual(answer(partition)['result'], answer(boolean)['result'])
        self.assertEqual(answer(boolean)['result']['components'], [[self.a, self.b], [self.isolated]])

    def test_connected_projection_with_exclusions_completes_with_false_boolean(self):
        for framing, decision in (('direct', 'stop'), ('step', 'continue')):
            row = self.row([self.a, self.b, self.outside, self.missing], framing=framing)
            self.assert_source_result(row)
            result = answer(row)['result']
            self.assertEqual(result['components'], [[self.a, self.b]])
            self.assertEqual(result['component_count'], 1)
            self.assertFalse(result['single_component'])
            self.assertEqual(result['not_in_layer_gene_ids'], [self.outside])
            self.assertEqual(result['missing_gene_ids'], [self.missing])
            self.assertEqual(answer(row)['continuation_decision'], decision)

    def test_all_excluded_inputs_complete_without_singleton_groups(self):
        row = self.row([self.outside, self.missing], variant='membership_context', framing='step')
        self.assert_source_result(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(answer(row)['result']['components'], [])
        self.assertEqual(answer(row)['result']['component_count'], 0)
        self.assertFalse(answer(row)['result']['single_component'])
        self.assertEqual(answer(row)['continuation_decision'], 'continue')

    def test_empty_input_has_no_acquisition_or_evidence(self):
        for variant in ('lookup', 'context', 'duplicate_context', 'membership_context', 'gate_context', 'native_context'):
            with self.subTest(variant=variant):
                row = self.row([], variant=variant)
                self.assert_source_result(row)
                self.assertEqual(calls(row), [])
                self.assertEqual(context(row)['observations'], [])
                self.assertEqual(answer(row)['provenance'], {'registry': [], 'graph': []})
                self.assertEqual(answer(row)['result']['components'], [])

    def test_singleton_membership_proves_one_group_with_and_without_a_loop(self):
        for gene in (self.a, self.isolated):
            row = self.row([gene], variant='membership_context')
            self.assert_source_result(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(answer(row)['result']['components'], [[gene]])
            self.assertTrue(answer(row)['result']['single_component'])
            self.assertEqual(context(row)['observations'][0]['operation'], 'get_nodes_by_layer')

    def test_visible_membership_first_policy_needs_no_singleton_graph_lookup(self):
        row = self.row([self.isolated], membership_first=True)
        self.assert_source_result(row)
        self.assertEqual([call['name'] for call in calls(row)], ['get_nodes_by_layer'])
        self.assertEqual(context(row)['acquisition_policy'], 'membership_first')

    def test_matched_weak_and_strong_chain_keeps_native_backend_arguments(self):
        genes = [self.x, self.y, self.z]
        weak, strong = self.row(genes, BRAIN, 'weak'), self.row(genes, BRAIN, 'strong')
        self.assert_source_result(weak)
        self.assert_source_result(strong)
        self.assertEqual(calls(weak)[0]['arguments'], calls(strong)[0]['arguments'])
        self.assertTrue(answer(weak)['result']['directed'])
        self.assertEqual(answer(weak)['result']['components'], [genes])
        self.assertEqual(answer(strong)['result']['components'], [[gene] for gene in genes])
        self.assertEqual(weak['metadata']['split_unit'], strong['metadata']['split_unit'])

    def test_two_nontrivial_sccs_have_a_one_way_connection(self):
        row = self.row([self.a, self.b, self.e, self.f], BRAIN, 'strong')
        self.assert_source_result(row)
        self.assertEqual(answer(row)['result']['components'], [[self.a, self.b], [self.e, self.f]])
        self.assertIn({'gene_ids': [self.b, self.e], 'weight': .4},
                      self.index.edges(BRAIN, [self.a, self.b, self.e, self.f]))

    def test_duplicate_and_reordered_inputs_keep_coverage_and_shared_source_unit(self):
        direct, reordered, duplicate = self.row(), self.row(variant='reordered'), self.row(variant='duplicate_inputs')
        self.assert_source_result(reordered)
        self.assert_source_result(duplicate)
        self.assertEqual(answer(direct)['result']['components'], answer(reordered)['result']['components'])
        self.assertEqual(answer(reordered)['result']['coverage']['requested_gene_ids'], [self.c, self.b, self.a])
        self.assertEqual(duplicate['metadata']['request']['genes'], [self.a, self.b, self.c, self.a, self.c])
        self.assertEqual(calls(duplicate)[0]['arguments']['gene_ids'], [self.a, self.b, self.c])
        self.assertEqual(direct['metadata']['split_unit'], reordered['metadata']['split_unit'])

    def test_complete_saved_graph_and_duplicate_context_require_no_new_call(self):
        for variant in ('context', 'duplicate_context'):
            row = self.row([self.a, self.b, self.c, self.isolated], variant=variant)
            self.assert_source_result(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(len(answer(row)['provenance']['graph']), 1)

    def test_complete_native_incoming_and_outgoing_records_keep_directed_groups(self):
        for kind in ('weak', 'strong'):
            for variant in ('native_context', 'native_in_context'):
                row = self.row([self.a, self.b, self.e, self.f], BRAIN, kind, variant=variant)
                self.assert_source_result(row)
                self.assertEqual(calls(row), [])
                self.assertTrue(answer(row)['result']['directed'])

    def test_membership_alone_leaves_a_multi_node_partition_unproved(self):
        row = self.row([self.a, self.b, self.isolated], variant='membership_context')
        self.assert_source_result(row)
        self.assertEqual([call['name'] for call in calls(row)], ['induce_subgraph'])

    def test_a_split_bounded_preview_triggers_complete_native_acquisition(self):
        row = self.row([self.a, self.b, self.c, self.isolated], variant='bounded_context')
        self.assert_source_result(row)
        self.assertEqual(len(calls(row)), 1)
        source = calls(row)[0]['arguments']
        self.assertEqual((source['mode'], source['top_k'], source['cursor']), ('complete', None, None))
        self.assertTrue(answer(row)['result']['collection']['complete'])
        self.assertFalse(context(row)['observations'][0]['result']['collection']['complete'])

    def test_partial_spanning_witness_proves_one_complete_group_without_new_calls(self):
        row = self.row(variant='spanning_context')
        self.assert_source_result(row)
        self.assertEqual(calls(row), [])
        original = context(row)['observations'][0]['result']
        self.assertFalse(original['collection']['complete'])
        self.assertLess(original['edge_count'], self.fact().edge_count)
        self.assertEqual(answer(row)['result']['components'], [[self.a, self.b, self.c]])
        self.assertTrue(answer(row)['result']['collection']['complete'])

    def test_partial_strong_witness_retains_native_arcs_and_partial_edge_collection(self):
        row = self.row([self.a, self.b, self.c], BRAIN, 'strong', variant='strong_witness_context')
        self.assert_source_result(row)
        self.assertEqual(calls(row), [])
        original = context(row)['observations'][0]['result']
        self.assertTrue(original['directed'])
        self.assertFalse(original['collection']['complete'])
        self.assertEqual(len(components(original['layer_gene_ids'], original['edges'], 'strong')), 1)
        self.assertTrue(answer(row)['result']['collection']['complete'])

    def test_weak_witness_cannot_supply_a_strong_partition(self):
        row = self.row([self.a, self.b, self.c], BRAIN, 'strong', variant='weak_witness_strong_context')
        self.assert_source_result(row)
        original = context(row)['observations'][0]['result']
        self.assertEqual(len(components(original['layer_gene_ids'], original['edges'], 'weak')), 1)
        self.assertGreater(len(components(original['layer_gene_ids'], original['edges'], 'strong')), 1)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(answer(row)['result']['components'], [[self.a, self.b, self.c]])

    def test_disconnected_partial_edges_do_not_establish_a_complete_split(self):
        row = self.row([self.a, self.b, self.c, self.isolated], variant='insufficient_context')
        self.assert_source_result(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertFalse(context(row)['observations'][0]['result']['collection']['complete'])
        self.assertEqual(answer(row)['result']['components'], [[self.a, self.b, self.c], [self.isolated]])

    def test_removal_of_a_saved_bridge_requires_partition_from_restricted_edges(self):
        row = self.row([self.a, self.d], COEX, variant='bridge_context')
        self.assert_source_result(row)
        self.assertEqual(calls(row), [])
        original = context(row)['observations'][0]['result']
        self.assertIn(self.outside, original['layer_gene_ids'])
        self.assertEqual(len(components(original['layer_gene_ids'], original['edges'], 'undirected')), 1)
        self.assertEqual(answer(row)['result']['components'], [[self.a], [self.d]])

    def test_visible_outside_route_does_not_join_the_requested_induced_groups(self):
        row = self.row([self.a, self.d], COEX, variant='outside_route_context')
        self.assert_source_result(row)
        observations = context(row)['observations']
        self.assertEqual([item['operation'] for item in observations], ['get_edge', 'get_edge'])
        self.assertTrue(all(item['result']['edge_exists'] for item in observations))
        self.assertEqual([call['name'] for call in calls(row)], ['induce_subgraph'])
        self.assertEqual(answer(row)['result']['components'], [[self.a], [self.d]])

    def test_omitted_undirected_kind_remains_null_in_the_raw_request(self):
        row = self.row(variant='unqualified')
        self.assert_source_result(row)
        self.assertIsNone(row['metadata']['request']['connectivity_kind'])
        self.assertEqual(answer(row)['result']['connectivity_kind'], 'undirected')
        with self.assertRaises(VariantUnavailable):
            self.row([self.a], BRAIN, 'strong', variant='unqualified')

    def test_separate_batches_cannot_settle_unseen_cross_edges(self):
        row = self.row([self.a, self.c], variant='disjoint_context')
        self.assert_source_result(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(calls(row)[0]['arguments']['gene_ids'], [self.a, self.c])

    def test_pages_keep_the_full_input_and_returned_cursor_until_separation_is_proved(self):
        genes = [self.a, self.b, self.c, self.isolated]
        row = self.row(genes, variant='paged', framing='step')
        self.assert_source_result(row)
        self.assertGreater(len(calls(row)), 1)
        previous = None
        for position, message in enumerate(row['messages']):
            if 'tool_call' not in message:
                continue
            source = message['tool_call']['arguments']
            self.assertEqual(source['gene_ids'], genes)
            self.assertEqual(source['cursor'], previous)
            previous = row['messages'][position + 1]['content']['result']['collection']['next_cursor']
        self.assertEqual(answer(row)['continuation_decision'], 'continue')

    def test_page_variants_keep_exact_source_partition_without_edge_output(self):
        for variant in ('paged', 'overlap', 'empty_page', 'unknown_total', 'prefix_context', 'final_page_only'):
            row = self.row([self.a, self.b, self.c, self.isolated], variant=variant)
            self.assert_source_result(row)
            self.assertEqual(answer(row)['result']['components'], [[self.a, self.b, self.c], [self.isolated]])
            if variant == 'final_page_only':
                self.assertIsNone(calls(row)[0]['arguments']['cursor'])

    def test_authorized_retry_repeats_identical_arguments_and_stops_after_success(self):
        row = self.row(variant='recovered')
        self.assert_source_result(row)
        self.assertEqual(len(calls(row)), 2)
        self.assertEqual(calls(row)[0], calls(row)[1])
        self.assertEqual(answer(row)['status'], 'resolved')

    def test_terminal_and_exhausted_failures_keep_unknown_partition(self):
        for variant in ('terminal', 'exhausted'):
            row = self.row(variant=variant)
            self.validate(row)
            result = answer(row)['result']
            self.assertEqual(answer(row)['status'], 'error')
            self.assertEqual(answer(row)['continuation_decision'], 'defer')
            self.assertNotIn('components', result)
            self.assertNotIn('component_count', result)
            self.assertEqual(result['coverage']['checked_gene_ids'], [])

    def test_later_page_failures_preserve_full_membership_without_provisional_groups(self):
        for variant in ('page_terminal', 'page_exhausted', 'page_recovered'):
            row = self.row([self.a, self.b, self.c, self.isolated], variant=variant)
            self.validate(row)
            if variant == 'page_recovered':
                self.assert_source_result(row)
                self.assertEqual(calls(row)[0], calls(row)[1])
                continue
            result = answer(row)['result']
            self.assertEqual(answer(row)['status'], 'error')
            self.assertEqual(result['coverage']['unchecked_gene_ids'], [])
            self.assertEqual(result['coverage']['checked_gene_ids'], [self.a, self.b, self.c, self.isolated])
            self.assertTrue(result['partial_results'])
            self.assertIsNotNone(result['arguments']['cursor'])
            self.assertNotIn('components', result)

    def test_explicit_all_members_gate_blocks_before_unnecessary_graph_acquisition(self):
        genes = [self.a, self.b, self.outside, self.missing]
        ordinary, gated = self.row(genes), self.row(genes, gate=True)
        self.assert_source_result(ordinary)
        self.validate(gated)
        self.assertEqual([call['name'] for call in calls(gated)], ['get_nodes_by_layer'])
        self.assertEqual(answer(gated)['status'], 'not_found')
        self.assertEqual(answer(gated)['result']['issue_code'], 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
        self.assertEqual(answer(gated)['continuation_decision'], 'defer')
        self.assertEqual(answer(gated)['result']['coverage']['unchecked_gene_ids'], [])

    def test_successful_membership_gate_can_continue_with_a_split_graph(self):
        row = self.row([self.a, self.b, self.isolated], gate=True, framing='step')
        self.assert_source_result(row)
        self.assertEqual([call['name'] for call in calls(row)], ['get_nodes_by_layer', 'induce_subgraph'])
        self.assertEqual(answer(row)['result']['component_count'], 2)
        self.assertEqual(answer(row)['continuation_decision'], 'continue')

    def test_membership_failure_policy_keeps_ordinary_projection_separate_from_gate(self):
        for variant in ('membership_terminal', 'membership_recovered', 'membership_exhausted'):
            row = self.row([self.a, self.b, self.outside, self.missing], variant=variant)
            self.validate(row)
            self.assertEqual(calls(row)[0]['name'], 'get_nodes_by_layer')
            self.assertFalse(row['metadata']['request']['require_all_in_layer'])
            self.assertEqual(context(row)['acquisition_policy'], 'membership_first')
            if variant == 'membership_recovered':
                self.assert_source_result(row)
                self.assertFalse(answer(row)['result']['single_component'])
            else:
                self.assertEqual(answer(row)['status'], 'error')
                self.assertEqual(answer(row)['result']['coverage']['checked_gene_ids'], [])

    def test_symbol_context_survives_terminal_graph_failure(self):
        for variant in ('symbol_context', 'symbol_terminal'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(answer(row)['provenance']['registry']), 3)
            self.assertTrue(all(call['name'] != 'lookup_human_gene_symbol' for call in calls(row)))
            if variant == 'symbol_terminal':
                self.assertEqual(answer(row)['status'], 'error')
                self.assertEqual(len(answer(row)['result']['partial_results']), 3)
                self.assertNotIn('components', answer(row)['result'])

    def test_graph_only_query_keeps_all_layer_members(self):
        row = self.row([self.e, self.graph_only])
        self.assert_source_result(row)
        self.assertEqual(answer(row)['result']['components'], [[self.e, self.graph_only]])
        self.assertEqual(answer(row)['provenance']['registry'], [])

    def test_wrong_scope_stale_and_distractor_evidence_require_a_current_partition(self):
        for variant in ('stale', 'wrong_layer', 'distractor'):
            row = self.row([self.a, self.b, self.c, self.isolated], variant=variant)
            self.assert_source_result(row)
            self.assertTrue(calls(row))

    def test_distractor_content_does_not_depend_on_wrapper_bank_size(self):
        bank = deepcopy(self.bank)
        bank['train']['direct'] = bank['train']['direct'][:3]
        first = self.row(variant='distractor', variant_index=17)
        second = self.row(variant='distractor', variant_index=17, bank=bank)
        left, right = context(first)['observations'][0], context(second)['observations'][0]
        self.assertEqual(left['arguments'], right['arguments'])
        self.assertEqual(left['result'], right['result'])

    def test_all_fixed_diagnostics_preserve_the_raw_request_and_defer(self):
        expected = {
            'unknown_layer': ('not_found', 'UNKNOWN_LAYER'),
            'unknown_multiplex': ('not_found', 'UNKNOWN_MULTIPLEX'),
            'missing_genes': ('invalid_input', 'MISSING_GENES'),
            'missing_scope': ('invalid_input', 'MISSING_SCOPE'),
            'invalid_list': ('invalid_input', 'INVALID_GENE_LIST'),
            'missing_kind': ('invalid_input', 'CONNECTIVITY_KIND_REQUIRED'),
            'invalid_kind': ('invalid_input', 'INVALID_CONNECTIVITY_KIND'),
            'unavailable_kind': ('invalid_input', 'CONNECTIVITY_KIND_UNAVAILABLE'),
            'unresolved_identity': ('not_found', 'UNRESOLVED_IDENTITY'),
            'tool_unknown_layer': ('not_found', 'UNKNOWN_LAYER'),
        }
        for kind in corpus.DIAGNOSTICS:
            row = corpus.build_diagnostic(self.view, self.fact(), 'direct', 0, self.bank, self.contract,
                kind=kind, registry=self.registry, seed=SEED)
            self.validate(row)
            result = answer(row)['result']
            self.assertEqual((answer(row)['status'], result['issue_code']), expected[kind])
            self.assertEqual(set(result), {'issue_code', 'request', 'rejected_arguments', 'partial_results', 'coverage'})
            self.assertEqual(result['request'], row['metadata']['request'])
            self.assertEqual(answer(row)['continuation_decision'], 'defer')
            self.assertEqual(answer(row)['provenance']['graph'], [])
            self.assertNotIn('components', result)
            if kind == 'tool_unknown_layer':
                self.assertEqual(result['rejected_arguments'], calls(row)[0]['arguments'])
            if kind == 'unresolved_identity':
                self.assertIsNone(result['coverage'])
                self.assertTrue(all(call['name'] != 'induce_subgraph' for call in calls(row)))

    def test_structured_and_private_source_metadata_tampering_are_rejected(self):
        original = self.row()
        mutations = [
            lambda row: answer(row)['result'].__setitem__('component_count', 2),
            lambda row: answer(row)['result'].__setitem__('component_count', 1.0),
            lambda row: answer(row)['result'].__setitem__('single_component', 1),
            lambda row: answer(row)['result'].__setitem__('components', [[self.a], [self.b, self.c]]),
            lambda row: answer(row)['result'].__setitem__('directed', True),
            lambda row: answer(row)['result']['coverage'].__setitem__('checked_gene_ids', []),
            lambda row: answer(row)['result']['collection'].__setitem__('returned_count', 3),
            lambda row: row['metadata'].__setitem__('component_count', 1.0),
            lambda row: row['metadata'].__setitem__('wrapper_index', -1),
            lambda row: row['metadata'].__setitem__('request_template_index', -1),
        ]
        for mutation in mutations:
            changed = deepcopy(original)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(changed)

    def test_supported_summary_and_blocker_prose_are_not_exact_matched(self):
        resolved = self.row()
        answer(resolved)['summary'] = 'The selected graph has one connected group.'
        self.validate(resolved)
        failed = self.row(variant='terminal')
        answer(failed)['summary'] = 'The required native graph lookup failed.'
        answer(failed)['blocked_reason'] = 'The backend must recover before the exact partition can be proved.'
        self.validate(failed)

    def test_explicit_kind_cannot_be_relabelled_as_an_unqualified_request(self):
        changed = deepcopy(self.row())
        self.assertEqual(changed['metadata']['request']['connectivity_kind'], 'undirected')
        changed['metadata']['variant'] = 'unqualified'
        with self.assertRaisesRegex(ValueError, 'visible connectivity choice'):
            self.validate(changed)

    def test_lookup_without_saved_evidence_cannot_be_relabelled_as_context(self):
        changed = deepcopy(self.row())
        self.assertEqual(context(changed)['observations'], [])
        changed['metadata']['variant'] = 'context'
        with self.assertRaisesRegex(ValueError, 'supplied evidence context'):
            self.validate(changed)

    def test_duplicate_context_requires_an_actual_repeated_observation(self):
        changed = deepcopy(self.row(variant='context'))
        changed['metadata']['variant'] = 'duplicate_context'
        with self.assertRaisesRegex(ValueError, 'repeated observation'):
            self.validate(changed)
        changed = deepcopy(self.row(variant='duplicate_context'))
        supplied = context(changed)
        self.assertGreater(len(supplied['observations']), 1)
        supplied['observations'] = supplied['observations'][:1]
        changed['messages'][1]['content'] = json.dumps(supplied)
        with self.assertRaisesRegex(ValueError, 'repeated observation'):
            self.validate(changed)

    def test_context_acquisition_executes_the_pinned_parent_contract_path(self):
        baseline = self.row(variant='context')
        parent_pin = self.contract['context_source_contract']
        parent = json.loads((ROOT / parent_pin['path']).read_text())
        parent['contract_version'] += '_unit_context_override'
        with TemporaryDirectory() as directory:
            parent_path = Path(directory) / 'parent.json'
            parent_path.write_text(json.dumps(parent))
            overlay = {key: deepcopy(value) for key, value in self.contract.items() if not key.startswith('_')}
            overlay['context_source_contract'] = {'path': str(parent_path), 'sha256': file_sha256(parent_path)}
            overlay_path = Path(directory) / 'overlay.json'
            overlay_path.write_text(json.dumps(overlay))
            contract = corpus.load_build_contract(overlay_path)
            with patch.object(corpus, '_graph_contract', wraps=corpus._graph_contract) as read_parent:
                changed = corpus.build_row(self.view, self.fact(), 'direct', 'context', 0,
                    self.bank, contract, index=self.index, registry=self.registry, seed=SEED)
            read_parent.assert_called_once_with(parent_path)
            corpus.validate_row(changed, self.view, bank=self.bank, contract=contract,
                                registry=self.registry, index=self.index)
        before, after = context(baseline)['observations'][0], context(changed)['observations'][0]
        self.assertEqual(before['arguments'], after['arguments'])
        self.assertEqual(before['status'], after['status'])
        self.assertEqual(before['result'], after['result'])
        self.assertNotEqual(before['provenance']['graph'][0]['evidence_id'], after['provenance']['graph'][0]['evidence_id'])
        self.assertEqual(answer(baseline)['result'], answer(changed)['result'])

    def test_source_weights_call_kind_cursor_and_provenance_tampering_are_rejected(self):
        changed = deepcopy(self.row(variant='context'))
        supplied = context(changed)
        supplied['observations'][0]['result']['edges'][0]['weight'] = 7.5
        changed['messages'][1]['content'] = json.dumps(supplied)
        with self.assertRaises(ValueError):
            self.validate(changed)
        for target in ('kind', 'cursor', 'provenance'):
            changed = deepcopy(self.row([self.a, self.b, self.c, self.isolated], variant='paged'))
            if target == 'kind':
                calls(changed)[0]['arguments']['connectivity_kind'] = 'undirected'
            elif target == 'cursor':
                calls(changed)[1]['arguments']['cursor'] = 'invented'
            else:
                answer(changed)['provenance']['graph'][0]['evidence_id'] = 'invented'
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.validate(changed)

    def test_every_declared_variant_has_an_applicable_source_fixture(self):
        candidates = [
            ([self.a, self.b, self.c], PPI, None),
            ([self.a, self.b, self.c, self.isolated], PPI, None),
            ([self.a, self.b, self.c], BRAIN, 'strong'),
            ([self.x, self.y, self.z], BRAIN, 'strong'),
            ([self.x, self.y, self.z], BRAIN, 'weak'),
            ([self.a, self.d], COEX, None),
            ([self.a], PPI, None),
            ([self.a, self.b, self.outside, self.missing], PPI, None),
        ]
        for variant in corpus.VARIANTS:
            failures = []
            for genes, layer, kind in candidates:
                try:
                    row = self.row(genes, layer, kind, variant=variant)
                except VariantUnavailable as error:
                    failures.append(str(error))
                    continue
                self.assert_source_result(row)
                break
            else:
                self.fail(f'{variant} has no applicable fixture: {failures}')


if __name__ == '__main__':
    unittest.main()
