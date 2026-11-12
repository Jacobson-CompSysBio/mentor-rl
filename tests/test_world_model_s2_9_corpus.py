"""Check induced corpus rows from visible memberships, edges, pages, and gates."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_9_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_9_sampling import fact_from_genes
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import ARGUMENT_KEYS, SubgraphIndex


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


def calls(row):
    """Read only current-turn calls."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def context(row):
    """Read supplied context without private source metadata."""
    return json.loads(row['messages'][1]['content'])


def answer(row):
    """Read the final structured envelope."""
    return row['messages'][-1]['content']


class SubgraphCorpusTests(unittest.TestCase):
    """Keep all queried and supplied-context subjects in the fixed train split."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 160)
                if gene_group(f'ENSG{number:011d}', SEED)[1] == 'train']
        cls.a, cls.b, cls.c, cls.d, cls.e, cls.outside, cls.isolated, cls.graph_only, cls.missing = pool[:9]
        a, b, c, d, e, outside, isolated, graph_only = pool[:8]
        cls.view = GraphView('unit_subgraph_corpus', '9' * 64, {
            PPI: LayerGraph(PPI, False, {(a, a): .95, (a, b): .8, (a, c): .7, (b, c): 0,
                (c, d): .3, (graph_only, e): .2}, frozenset((a, b, c, d, e, isolated, graph_only))),
            COEX: LayerGraph(COEX, False, {(outside, a): .4}, frozenset((outside, a))),
            BRAIN: LayerGraph(BRAIN, True, {(a, b): .8, (b, a): .6, (b, b): .9,
                (c, a): .5, (d, b): .4, (e, c): .3}, frozenset((a, b, c, d, e, isolated, graph_only))),
            CD4: LayerGraph(CD4, True, {(outside, a): .1}, frozenset((outside, a))),
        })
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_SUBGRAPH_CORPUS_{ordinal}' for ordinal, gene in enumerate(pool) if gene != graph_only})
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contract = corpus.load_build_contract()

    def setUp(self):
        self.index = SubgraphIndex(self.view)

    def fact(self, genes=None, layer=PPI):
        return fact_from_genes(self.view, self.index, layer,
            [self.a, self.b, self.c] if genes is None else genes, SEED)

    def row(self, genes=None, layer=PPI, *, variant='lookup', mode='complete', top_k=None,
            gate=False, framing='direct', variant_index=0, bank=None):
        return corpus.build_row(self.view, self.fact(genes, layer), framing, variant, variant_index,
            bank or self.bank, self.contract, index=self.index, registry=self.registry,
            mode=mode, top_k=top_k, require_all_in_layer=gate, seed=SEED)

    def validate(self, row):
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
                            registry=self.registry, index=self.index)

    def test_lookup_preserves_exact_six_arguments_and_native_weights(self):
        row = self.row()
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(tuple(calls(row)[0]['arguments']), ARGUMENT_KEYS)
        self.assertEqual(calls(row)[0]['name'], 'induce_subgraph')
        self.assertEqual(answer(row)['result']['edges'], [
            {'gene_ids': [self.a, self.a], 'weight': .95}, {'gene_ids': [self.a, self.b], 'weight': .8},
            {'gene_ids': [self.a, self.c], 'weight': .7}, {'gene_ids': [self.b, self.c], 'weight': 0}])

    def test_projection_retains_members_and_separates_both_exclusion_categories(self):
        row = self.row([self.a, self.outside, self.missing, self.isolated])
        self.validate(row)
        result = answer(row)['result']
        self.assertEqual(answer(row)['status'], 'resolved')
        self.assertEqual(result['layer_gene_ids'], [self.a, self.isolated])
        self.assertEqual(result['not_in_layer_gene_ids'], [self.outside])
        self.assertEqual(result['missing_gene_ids'], [self.missing])
        self.assertEqual(result['layer_gene_count'], 2)

    def test_selected_member_with_no_incident_edge_stays_in_the_result(self):
        row = self.row([self.a, self.b, self.isolated])
        self.validate(row)
        result = answer(row)['result']
        self.assertIn(self.isolated, result['layer_gene_ids'])
        self.assertTrue(all(self.isolated not in edge['gene_ids'] for edge in result['edges']))

    def test_all_excluded_projection_resolves_without_a_false_global_blocker(self):
        row = self.row([self.outside, self.missing])
        self.validate(row)
        self.assertEqual(answer(row)['status'], 'resolved')
        self.assertEqual(answer(row)['result']['edges'], [])
        self.assertEqual(answer(row)['result']['layer_gene_ids'], [])

    def test_empty_input_requires_no_graph_call_and_empty_provenance(self):
        for variant in ('lookup', 'context', 'duplicate_context', 'membership_context', 'gate_context', 'native_context'):
            row = self.row([], variant=variant)
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(answer(row)['provenance'], {'registry': [], 'graph': []})
            self.assertEqual(answer(row)['result']['edge_count'], 0)
            self.assertEqual(context(row)['observations'], [])

    def test_singleton_still_needs_membership_and_loop_evidence(self):
        for gene, edges in ((self.a, 1), (self.d, 0)):
            row = self.row([gene])
            self.validate(row)
            self.assertEqual(len(calls(row)), 1)
            self.assertEqual(answer(row)['result']['edge_count'], edges)

    def test_directed_graph_preserves_descending_source_order_and_reciprocals(self):
        row = self.row(layer=BRAIN)
        self.validate(row)
        edges = answer(row)['result']['edges']
        self.assertIn({'gene_ids': [self.b, self.a], 'weight': .6}, edges)
        self.assertIn({'gene_ids': [self.a, self.b], 'weight': .8}, edges)
        self.assertIn({'gene_ids': [self.c, self.a], 'weight': .5}, edges)
        self.assertEqual(edges, sorted(edges, key=lambda edge: edge['gene_ids']))

    def test_graph_only_query_survives_without_registry_evidence(self):
        row = self.row([self.e, self.graph_only])
        self.validate(row)
        self.assertIn(self.graph_only, answer(row)['result']['layer_gene_ids'])
        self.assertEqual(answer(row)['provenance']['registry'], [])

    def test_reordered_inputs_change_coverage_order_but_keep_the_graph(self):
        direct, reordered = self.row(), self.row(variant='reordered')
        self.validate(reordered)
        self.assertEqual(answer(direct)['result']['edges'], answer(reordered)['result']['edges'])
        self.assertEqual(answer(reordered)['result']['coverage']['requested_gene_ids'], [self.c, self.b, self.a])
        self.assertEqual(direct['metadata']['split_unit'], reordered['metadata']['split_unit'])

    def test_literal_duplicates_remain_visible_but_do_not_duplicate_acquisition_inputs(self):
        row = self.row(variant='duplicate_inputs')
        self.validate(row)
        self.assertEqual(row['metadata']['request']['genes'], [self.a, self.b, self.c, self.a, self.c])
        self.assertEqual(calls(row)[0]['arguments']['gene_ids'], [self.a, self.b, self.c])
        self.assertEqual(answer(row)['result']['layer_gene_count'], 3)

    def test_complete_context_and_identical_duplicate_context_need_no_new_call(self):
        for variant in ('context', 'duplicate_context'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(len(answer(row)['provenance']['graph']), 1)

    def test_complete_superset_context_restricts_both_edge_endpoints(self):
        row = self.row(variant='superset_context')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(answer(row)['result']['edges'], self.index.edges(PPI, [self.a, self.b, self.c]))
        self.assertGreater(len(context(row)['observations'][0]['arguments']['gene_ids']), 3)

    def test_proper_subset_and_disjoint_batches_do_not_fill_cross_pairs(self):
        for variant in ('subset_context', 'disjoint_context'):
            row = self.row([self.a, self.c], variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), 1)
            self.assertEqual(calls(row)[0]['arguments']['gene_ids'], [self.a, self.c])
            self.assertIn({'gene_ids': [self.a, self.c], 'weight': .7}, answer(row)['result']['edges'])

    def test_complete_membership_alone_does_not_prove_loops_or_edges(self):
        row = self.row(variant='membership_context')
        self.validate(row)
        self.assertEqual([call['name'] for call in calls(row)], ['induce_subgraph'])

    def test_classification_with_no_projected_members_settles_without_edge_lookup(self):
        row = self.row([self.outside, self.missing], variant='membership_context')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(answer(row)['result']['edges'], [])

    def test_complete_outgoing_or_incoming_context_can_supply_directed_graph(self):
        for variant in ('native_context', 'native_in_context'):
            row = self.row(layer=BRAIN, variant=variant)
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(answer(row)['result']['edges'], self.index.edges(BRAIN, [self.a, self.b, self.c]))

    def test_partial_native_context_leaves_last_input_loop_unproved(self):
        row = self.row(variant='partial_native_context')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)

    def test_partial_positive_loop_settles_the_only_singleton_candidate(self):
        row = self.row([self.a], variant='positive_loop_context')
        self.validate(row)
        self.assertEqual(calls(row), [])
        observation = context(row)['observations'][0]
        self.assertFalse(observation['result']['collection']['complete'])
        self.assertEqual(answer(row)['result']['edges'], [{'gene_ids': [self.a, self.a], 'weight': .95}])

    def test_explicit_preview_is_canonical_prefix_and_keeps_full_membership(self):
        row = self.row(mode='top_k', top_k=1)
        self.validate(row)
        result = answer(row)['result']
        self.assertEqual(result['edges'], [{'gene_ids': [self.a, self.a], 'weight': .95}])
        self.assertEqual(result['layer_gene_count'], 3)
        self.assertFalse(result['collection']['complete'])
        self.assertEqual(result['collection']['total_count'], 4)

    def test_preview_at_or_above_full_count_is_complete(self):
        for top_k in (4, 9):
            row = self.row(mode='top_k', top_k=top_k)
            self.validate(row)
            self.assertTrue(answer(row)['result']['collection']['complete'])

    def test_bounded_context_requires_a_new_complete_selection_at_null_cursor(self):
        row = self.row(variant='bounded_context')
        self.validate(row)
        self.assertEqual(calls(row)[0]['arguments']['mode'], 'complete')
        self.assertIsNone(calls(row)[0]['arguments']['top_k'])
        self.assertIsNone(calls(row)[0]['arguments']['cursor'])

    def test_matching_or_longer_preview_context_can_answer_a_shorter_preview(self):
        for variant in ('bounded_context', 'shorter_preview_context'):
            row = self.row(mode='top_k', top_k=1, variant=variant)
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(answer(row)['result']['edge_count'], 1)

    def test_paged_overlap_empty_pages_and_unknown_totals_recover_the_full_graph(self):
        for variant in ('paged', 'overlap', 'empty_page', 'unknown_total', 'prefix_context', 'final_page_only'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(answer(row)['result']['edges'], self.index.edges(PPI, [self.a, self.b, self.c]))
            self.assertEqual(answer(row)['result']['collection']['total_count'], 4)
            self.assertTrue(answer(row)['result']['collection']['complete'])
            if variant == 'final_page_only':
                self.assertIsNone(calls(row)[0]['arguments']['cursor'])

    def test_page_calls_keep_the_entire_ordered_input_and_exact_returned_cursor(self):
        row = self.row(variant='paged')
        self.validate(row)
        previous = None
        for position, message in enumerate(row['messages']):
            if 'tool_call' not in message:
                continue
            arguments = message['tool_call']['arguments']
            self.assertEqual(arguments['gene_ids'], [self.a, self.b, self.c])
            self.assertEqual(arguments['cursor'], previous)
            previous = row['messages'][position + 1]['content']['result']['collection']['next_cursor']

    def test_terminal_and_exhausted_errors_never_fabricate_an_ordinary_graph(self):
        for variant in ('terminal', 'exhausted'):
            row = self.row(variant=variant)
            self.validate(row)
            result = answer(row)['result']
            self.assertEqual(answer(row)['status'], 'error')
            self.assertEqual(answer(row)['continuation_decision'], 'defer')
            self.assertNotIn('edges', result)
            self.assertNotIn('layer_gene_ids', result)
            self.assertEqual(result['coverage']['checked_gene_ids'], [])

    def test_authorized_retry_repeats_identical_arguments_and_stops_after_success(self):
        row = self.row(variant='recovered')
        self.validate(row)
        self.assertEqual(calls(row)[0], calls(row)[1])
        self.assertEqual(len(calls(row)), 2)
        self.assertEqual(answer(row)['status'], 'resolved')

    def test_later_page_failure_preserves_partition_and_exact_partial_edges(self):
        for variant in ('page_terminal', 'page_exhausted', 'page_recovered'):
            row = self.row(variant=variant)
            self.validate(row)
            if variant == 'page_recovered':
                self.assertEqual(answer(row)['status'], 'resolved')
            else:
                result = answer(row)['result']
                self.assertEqual(answer(row)['status'], 'error')
                self.assertEqual(result['coverage']['checked_gene_ids'], [self.a, self.b, self.c])
                self.assertEqual(result['partial_results'][0]['result']['edge_count'], 2)
                self.assertIsNotNone(result['arguments']['cursor'])

    def test_explicit_gate_fails_before_graph_construction_but_default_projection_resolves(self):
        genes = [self.a, self.outside, self.missing]
        default, gated = self.row(genes), self.row(genes, variant='membership_gate')
        self.validate(default)
        self.validate(gated)
        self.assertEqual(answer(default)['status'], 'resolved')
        self.assertEqual([call['name'] for call in calls(gated)], ['get_nodes_by_layer'])
        self.assertEqual(answer(gated)['result']['issue_code'], 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
        self.assertEqual(answer(gated)['result']['coverage']['unchecked_gene_ids'], [])

    def test_passed_gate_still_requires_the_complete_edge_domain(self):
        row = self.row(variant='membership_gate')
        self.validate(row)
        self.assertEqual([call['name'] for call in calls(row)], ['get_nodes_by_layer', 'induce_subgraph'])
        self.assertEqual(answer(row)['status'], 'resolved')

    def test_supplied_gate_context_and_membership_failures_keep_their_distinct_paths(self):
        row = self.row([self.a, self.outside], variant='gate_context')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(answer(row)['continuation_decision'], 'defer')
        for variant in ('membership_terminal', 'membership_recovered', 'membership_exhausted'):
            failed = self.row(variant=variant)
            self.validate(failed)
            self.assertEqual(calls(failed)[0]['name'], 'get_nodes_by_layer')

    def test_symbol_context_resolves_before_graph_and_survives_terminal_failure(self):
        for variant in ('symbol_context', 'symbol_terminal'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(answer(row)['provenance']['registry']), 3)
            self.assertTrue(all(call['name'] != 'lookup_human_gene_symbol' for call in calls(row)))
            if variant == 'symbol_terminal':
                self.assertEqual(answer(row)['status'], 'error')
                self.assertEqual(len(answer(row)['result']['partial_results']), 3)
                self.assertEqual(answer(row)['result']['coverage']['checked_gene_ids'], [])

    def test_stale_wrong_layer_and_distractor_context_do_not_fill_the_graph(self):
        for variant in ('stale', 'wrong_layer', 'distractor'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertTrue(calls(row))
            self.assertEqual(answer(row)['result']['edges'], self.index.edges(PPI, [self.a, self.b, self.c]))

    def test_outer_framing_controls_completed_stop_or_continue(self):
        for framing, decision in (('direct', 'stop'), ('step', 'continue')):
            row = self.row([self.d, self.isolated], framing=framing)
            self.validate(row)
            self.assertEqual(answer(row)['result']['edges'], [])
            self.assertEqual(answer(row)['continuation_decision'], decision)

    def test_distractor_choice_does_not_depend_on_wrapper_bank_size(self):
        bank = deepcopy(self.bank)
        bank['train']['direct'] = bank['train']['direct'][:3]
        first, second = self.row(variant='distractor', variant_index=17), self.row(variant='distractor', variant_index=17, bank=bank)
        left, right = context(first)['observations'][0], context(second)['observations'][0]
        self.assertEqual(left['arguments'], right['arguments'])
        self.assertEqual(left['result'], right['result'])

    def test_all_fixed_diagnostics_preserve_raw_request_and_empty_graph_provenance(self):
        for kind in corpus.DIAGNOSTICS:
            row = corpus.build_diagnostic(self.view, self.fact(), 'direct', 0, self.bank, self.contract,
                kind=kind, registry=self.registry, seed=SEED)
            self.validate(row)
            result = answer(row)['result']
            self.assertEqual(result['request'], row['metadata']['request'])
            self.assertEqual(answer(row)['provenance']['graph'], [])
            self.assertNotIn('edges', result)
            self.assertEqual(answer(row)['continuation_decision'], 'defer')
            if kind == 'tool_unknown_layer':
                self.assertEqual(result['rejected_arguments'], calls(row)[0]['arguments'])
            if kind == 'unresolved_identity':
                self.assertTrue(all(call['name'] != 'induce_subgraph' for call in calls(row)))
                self.assertIsNone(result['coverage'])

    def test_structured_and_private_metadata_tampering_are_rejected(self):
        row = self.row()
        mutations = [
            lambda item: answer(item)['result'].__setitem__('edge_count', 0),
            lambda item: answer(item)['result']['edges'][0].__setitem__('weight', .3),
            lambda item: answer(item)['result'].__setitem__('directed', True),
            lambda item: answer(item)['result']['coverage'].__setitem__('checked_gene_ids', []),
            lambda item: item['metadata'].__setitem__('edge_count', 4.0),
            lambda item: item['metadata'].__setitem__('wrapper_index', -1),
            lambda item: item['metadata'].__setitem__('request_template_index', -1),
        ]
        for mutation in mutations:
            changed = deepcopy(row)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(changed)

    def test_source_context_provenance_and_cursor_tampering_are_rejected(self):
        row = self.row(variant='context')
        changed = deepcopy(row)
        supplied = context(changed)
        supplied['observations'][0]['result']['edges'][0]['weight'] = 7.5
        changed['messages'][1]['content'] = json.dumps(supplied)
        with self.assertRaises(ValueError):
            self.validate(changed)
        for target in ('cursor', 'provenance'):
            changed = deepcopy(self.row(variant='paged'))
            if target == 'cursor':
                calls(changed)[1]['arguments']['cursor'] = 'invented'
            else:
                answer(changed)['provenance']['graph'][0]['evidence_id'] = 'invented'
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.validate(changed)


if __name__ == '__main__':
    unittest.main()
