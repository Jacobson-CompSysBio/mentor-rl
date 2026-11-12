"""Check degree traces against visible counts, exact ranks, and source records."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_8_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_8_sampling import fact_from_arguments
from runtime.world_model.s2_degrees import ARGUMENT_KEYS, CAVEAT, DegreeIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


def scope(*layers, family=None):
    """Construct a typed explicit selection or exhaustive family selector."""
    return {'kind': 'layers' if layers else 'all', 'layer_ids': list(layers), 'layer_family': family}


def calls(row):
    """Read the current-turn calls without supplied observations."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def context(row):
    """Read the model-visible harness context."""
    return json.loads(row['messages'][1]['content'])


def answer(row):
    """Read the final structured answer without its free reasoning."""
    return row['messages'][-1]['content']


class DegreeCorpusTests(unittest.TestCase):
    """Use query subjects from the existing S2.3 train split."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 130)
                if gene_group(f'ENSG{number:011d}', SEED)[1] == 'train']
        cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.outside, cls.isolated, cls.graph_only, cls.missing = pool[:10]
        a, b, c, d, e, f, outside, isolated, graph_only = pool[:9]
        cls.view = GraphView('unit_degree_corpus_view', '8' * 64, {
            PPI: LayerGraph(PPI, False, {(a, a): .9, (a, b): .8, (a, c): .7, (a, d): 0,
                (b, c): .6, (b, d): .5}, frozenset((a, b, c, d, isolated))),
            COEX: LayerGraph(COEX, False, {(a, b): .4, (a, e): .3, (outside, f): .2},
                frozenset((a, b, e, outside, f))),
            BRAIN: LayerGraph(BRAIN, True, {(a, a): .9, (a, b): .8, (a, c): .7, (a, d): .6,
                (c, a): .5, (d, a): .4, (graph_only, a): .3, (b, e): .2},
                frozenset((a, b, c, d, e, isolated, graph_only))),
            CD4: LayerGraph(CD4, True, {(f, a): .1}, frozenset((f, a))),
        })
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_DEGREE_CORPUS_{ordinal}' for ordinal, gene in enumerate(pool) if gene != graph_only})
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contract = corpus.load_build_contract()

    def setUp(self):
        self.index = DegreeIndex(self.view)

    def fact(self, gene=None, selected=None, direction='undirected'):
        return fact_from_arguments(self.view, self.index, gene or self.a, selected or scope(PPI), direction, SEED)

    def row(self, gene=None, selected=None, direction='undirected', *, variant='lookup',
            kind='degree_and_hub', threshold=95, framing='direct', variant_index=0, bank=None):
        return corpus.build_row(self.view, self.fact(gene, selected, direction), framing, variant, variant_index,
            bank or self.bank, self.contract, index=self.index, registry=self.registry,
            request_kind=kind, threshold=threshold, seed=SEED)

    def validate(self, row):
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract,
                            registry=self.registry, index=self.index)

    def test_lookup_uses_exact_six_arguments_and_separates_count_from_rank(self):
        row = self.row(kind='degree_only')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(tuple(calls(row)[0]['arguments']), ARGUMENT_KEYS)
        self.assertEqual(calls(row)[0]['name'], 'get_degree')
        result = answer(row)['result']
        self.assertEqual(result['degree'], 4)
        for key in ('degree_percentile', 'rank_support', 'hub_policy', 'hub_like'):
            self.assertIsNone(result[key])
        self.assertEqual(result['caveats'], [])

    def test_hub_request_uses_exact_rank_and_only_the_policy_warning(self):
        row = self.row()
        self.validate(row)
        result = answer(row)['result']
        self.assertEqual(result['degree'], 4)
        self.assertEqual(result['degree_percentile'], 100)
        self.assertTrue(result['hub_like'])
        self.assertEqual(result['caveats'], [CAVEAT])

    def test_zero_member_degree_resolves_with_outer_stop_or_continue(self):
        for framing, decision in (('direct', 'stop'), ('step', 'continue')):
            row = self.row(self.isolated, framing=framing)
            self.validate(row)
            self.assertEqual(answer(row)['status'], 'resolved')
            self.assertEqual(answer(row)['continuation_decision'], decision)
            self.assertEqual(answer(row)['result']['degree'], 0)
            self.assertFalse(answer(row)['result']['hub_like'])

    def test_outside_layer_zero_resolves_count_but_blocks_required_rank(self):
        count = self.row(self.outside, kind='degree_only')
        hub = self.row(self.outside)
        self.validate(count)
        self.validate(hub)
        self.assertEqual(answer(count)['result']['degree'], 0)
        self.assertEqual(answer(count)['status'], 'resolved')
        result = answer(hub)['result']
        self.assertEqual(answer(hub)['status'], 'not_found')
        self.assertEqual(result['issue_code'], 'OUTSIDE_REFERENCE_POPULATION')
        self.assertEqual(result['partial_results'][0]['result']['degree'], 0)
        self.assertNotIn('hub_like', result)

    def test_global_absence_preserves_unavailable_degree(self):
        row = self.row(self.missing)
        self.validate(row)
        result = answer(row)['result']
        self.assertEqual(result['issue_code'], 'NO_MULTIPLEX_MEMBERSHIP')
        self.assertIsNone(result['partial_results'][0]['result']['degree'])
        self.assertEqual(answer(row)['continuation_decision'], 'defer')

    def test_absent_native_and_map_context_do_not_crash_or_become_zero(self):
        for variant in ('native_context', 'bounded_total', 'map_context', 'map_total'):
            with self.subTest(variant=variant):
                row = self.row(self.missing, variant=variant)
                self.validate(row)
                self.assertEqual(answer(row)['result']['issue_code'], 'NO_MULTIPLEX_MEMBERSHIP')
                self.assertNotIn('degree', answer(row)['result'])

    def test_empty_family_retains_zero_without_inventing_a_rank(self):
        for kind, status in (('degree_only', 'resolved'), ('degree_and_hub', 'not_found')):
            row = self.row(selected=scope(family='bulk_pen'), direction='incident', kind=kind)
            self.validate(row)
            self.assertEqual(answer(row)['status'], status)
            if kind == 'degree_only':
                self.assertEqual(answer(row)['result']['degree'], 0)
            else:
                self.assertEqual(answer(row)['result']['issue_code'], 'EMPTY_REFERENCE_POPULATION')
                self.assertEqual(answer(row)['result']['partial_results'][0]['result']['degree'], 0)

    def test_matching_context_answers_both_request_types_without_a_call(self):
        for kind in ('degree_only', 'degree_and_hub'):
            row = self.row(kind=kind, variant='context')
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(len(answer(row)['provenance']['graph']), 1)

    def test_supplied_degree_is_sufficient_only_for_count_only(self):
        for kind, expected_calls in (('degree_only', 0), ('degree_and_hub', 1)):
            row = self.row(kind=kind, variant='degree_context')
            self.validate(row)
            self.assertEqual(len(calls(row)), expected_calls)

    def test_complete_histogram_and_degree_prove_rank_without_a_call(self):
        row = self.row(variant='histogram_context')
        self.validate(row)
        self.assertEqual(calls(row), [])
        histogram = context(row)['observations'][1]['result']
        self.assertEqual(sum(item['node_count'] for item in histogram['degree_histogram']), 5)
        self.assertEqual(answer(row)['result']['rank_support']['population_count'], 5)
        reordered = self.row(selected=scope(COEX, PPI), variant='histogram_context')
        self.validate(reordered)
        distribution = context(reordered)['observations'][1]
        self.assertEqual(distribution['arguments']['layer_scope']['layer_ids'], [COEX, PPI])
        self.assertEqual(distribution['result']['layer_scope']['layer_ids'], [PPI, COEX])
        self.assertEqual(calls(reordered), [])

    def test_complete_native_list_proves_count_but_requires_hub_rank(self):
        for kind, expected_calls in (('degree_only', 0), ('degree_and_hub', 1)):
            row = self.row(kind=kind, variant='native_context')
            self.validate(row)
            self.assertEqual(len(calls(row)), expected_calls)
            self.assertEqual(answer(row)['result']['degree'], 4)

    def test_authoritative_total_proves_count_from_a_bounded_list(self):
        row = self.row(kind='degree_only', variant='bounded_total')
        self.validate(row)
        self.assertEqual(calls(row), [])
        observation = context(row)['observations'][0]
        self.assertEqual(len(observation['result']['neighbors']), 1)
        self.assertEqual(answer(row)['result']['degree'], 4)

    def test_authoritative_total_proves_count_from_a_partial_page(self):
        row = self.row(kind='degree_only', variant='prefix_total')
        self.validate(row)
        self.assertEqual(calls(row), [])
        page = context(row)['observations'][0]['result']
        self.assertFalse(page['collection']['complete'])
        self.assertIsNotNone(page['collection']['next_cursor'])
        self.assertEqual(answer(row)['result']['degree'], 4)

    def test_unknown_partial_total_requires_an_exact_degree_call(self):
        row = self.row(kind='degree_only', variant='unknown_total')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(calls(row)[0]['name'], 'get_degree')
        self.assertIsNone(context(row)['observations'][0]['result']['collection']['total_count'])

    def test_complete_pages_count_the_whole_set(self):
        row = self.row(kind='degree_only', variant='complete_pages')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertGreater(len(context(row)['observations']), 1)
        self.assertEqual(answer(row)['result']['degree'], 4)

    def test_a_final_page_without_the_prefix_is_not_a_complete_set(self):
        row = self.row(kind='degree_only', variant='final_page_only')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        observation = context(row)['observations'][0]
        self.assertIsNotNone(observation['arguments']['cursor'])
        self.assertIsNone(observation['result']['collection']['total_count'])

    def test_multiplex_map_and_authoritative_map_total_prove_the_union_count(self):
        for variant in ('map_context', 'map_total'):
            row = self.row(selected=scope(PPI, COEX), kind='degree_only', variant=variant)
            self.validate(row)
            self.assertEqual(calls(row), [])
            self.assertEqual(answer(row)['result']['degree'], 5)

    def test_separate_layer_counts_do_not_prove_the_union(self):
        row = self.row(selected=scope(PPI, COEX), kind='degree_only', variant='union_counts')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(answer(row)['result']['degree'], 5)
        self.assertEqual(sum(item['result']['degree'] for item in context(row)['observations']), 6)

    def test_separate_directions_do_not_prove_the_incident_union(self):
        row = self.row(selected=scope(BRAIN), direction='incident', kind='degree_only', variant='union_counts')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)
        self.assertEqual(answer(row)['result']['degree'], 5)

    def test_equivalent_selectors_and_undirected_forms_reuse_original_records(self):
        for variant in ('equivalent_scope', 'equivalent_direction'):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(calls(row), [])
            original = context(row)['observations'][0]
            self.assertEqual(answer(row)['provenance'], original['provenance'])
            self.assertEqual(answer(row)['result']['layer_scope'], scope(PPI))
            self.assertEqual(answer(row)['result']['direction'], 'undirected')

    def test_new_threshold_reuses_exact_support_and_preserves_the_original_policy(self):
        row = self.row(variant='new_threshold')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(context(row)['observations'][0]['result']['hub_policy']['threshold_percentile'], 95)
        self.assertEqual(answer(row)['result']['hub_policy']['threshold_percentile'], 100)
        self.assertTrue(answer(row)['result']['hub_like'])

    def test_threshold_zero_fifty_and_hundred_are_policy_inputs(self):
        for variant, threshold in (('threshold_zero', 0), ('threshold_fifty', 50), ('threshold_hundred', 100)):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(answer(row)['result']['hub_policy']['threshold_percentile'], threshold)
            self.assertTrue(answer(row)['result']['hub_like'])

    def test_graph_only_query_is_valid_without_a_symbol_mapping(self):
        row = self.row(self.graph_only, scope(BRAIN), direction='out')
        self.validate(row)
        self.assertTrue(answer(row)['result']['membership']['present_in_scope'])
        self.assertEqual(answer(row)['result']['rank_support']['population_count'], 7)
        self.assertEqual(answer(row)['provenance']['registry'], [])

    def test_symbol_context_preserves_the_literal_input_and_registry_evidence(self):
        row = self.row(variant='symbol_context')
        self.validate(row)
        self.assertEqual(row['metadata']['input_gene'], self.registry.symbols_by_id[self.a])
        self.assertEqual(len(answer(row)['provenance']['registry']), 1)

    def test_stale_scope_direction_and_distractor_context_need_the_requested_statistic(self):
        for selected, direction, variant in ((scope(PPI), 'undirected', 'stale'),
            (scope(PPI), 'undirected', 'wrong_scope'), (scope(BRAIN), 'out', 'wrong_direction'),
            (scope(PPI), 'undirected', 'distractor')):
            with self.subTest(variant=variant):
                row = self.row(selected=selected, direction=direction, variant=variant)
                self.validate(row)
                self.assertEqual(len(calls(row)), 1)

    def test_duplicate_context_deduplicates_evidence(self):
        row = self.row(variant='duplicate_context')
        self.validate(row)
        self.assertEqual(calls(row), [])
        self.assertEqual(len(context(row)['observations']), 2)
        self.assertEqual(len(answer(row)['provenance']['graph']), 1)

    def test_global_positive_does_not_supply_an_unseen_count(self):
        row = self.row(kind='degree_only', variant='global_context')
        self.validate(row)
        self.assertEqual(len(calls(row)), 1)

    def test_explicit_retry_recovers_with_identical_six_arguments(self):
        row = self.row(variant='recovered', framing='step')
        self.validate(row)
        self.assertEqual(len(calls(row)), 2)
        self.assertEqual(calls(row)[0], calls(row)[1])
        self.assertEqual(answer(row)['continuation_decision'], 'continue')

    def test_terminal_and_exhausted_failures_have_no_invented_measurement(self):
        for variant, expected_calls in (('terminal', 1), ('exhausted', 2)):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), expected_calls)
            self.assertEqual(answer(row)['status'], 'error')
            self.assertIsNone(answer(row)['result']['coverage'])
            self.assertEqual(answer(row)['result']['partial_results'], [])
            self.assertNotIn('degree', answer(row)['result'])

    def test_failed_rank_acquisition_retains_the_known_degree(self):
        for variant, expected_calls, status in (('rank_terminal', 1, 'error'),
                ('rank_recovered', 2, 'resolved'), ('rank_exhausted', 2, 'error')):
            row = self.row(variant=variant)
            self.validate(row)
            self.assertEqual(len(calls(row)), expected_calls)
            self.assertEqual(answer(row)['status'], status)
            if status == 'error':
                self.assertEqual(answer(row)['result']['partial_results'][0]['result']['degree'], 4)
                self.assertIsNone(answer(row)['result']['coverage'])

    def test_every_declared_variant_has_a_real_fixture_path(self):
        found = set()
        for variant in corpus.VARIANTS:
            for selected, direction in ((scope(PPI), 'undirected'), (scope(BRAIN), 'incident'),
                    (scope(BRAIN), 'out'), (scope(PPI, COEX), 'undirected')):
                try:
                    row = self.row(selected=selected, direction=direction, variant=variant)
                except VariantUnavailable:
                    continue
                self.validate(row)
                found.add(variant)
                break
        self.assertEqual(found, set(corpus.VARIANTS))

    def test_distractor_choice_does_not_depend_on_wrapper_bank_size(self):
        first = self.row(variant='distractor')
        changed = deepcopy(self.bank)
        changed['train']['direct'].append(changed['train']['direct'][0])
        second = self.row(variant='distractor', bank=changed)
        self.assertEqual(context(first)['observations'][0]['arguments'], context(second)['observations'][0]['arguments'])

    def test_all_local_diagnostics_preserve_raw_requests_and_make_no_call(self):
        for kind in corpus.DIAGNOSTICS:
            row = corpus.build_diagnostic(self.view, self.fact(), 'direct', 0, self.bank, self.contract, kind=kind, seed=SEED)
            self.validate(row)
            self.assertEqual(len(calls(row)), 1 if kind == 'tool_unknown_layer' else 0)
            result = answer(row)['result']
            self.assertEqual(answer(row)['continuation_decision'], 'defer')
            self.assertIsNone(result['coverage'])
            self.assertNotIn('degree', result)
            self.assertNotIn('hub_like', result)
            if kind == 'tool_unknown_layer':
                self.assertEqual(result['rejected_arguments'], calls(row)[0]['arguments'])
            else:
                self.assertIsNone(result['rejected_arguments'])

    def test_structured_result_and_source_metadata_tampering_are_rejected(self):
        row = self.row()
        mutations = [
            lambda item: item['messages'][-1]['content']['result'].__setitem__('degree', 0),
            lambda item: item['messages'][-1]['content']['result'].__setitem__('hub_like', False),
            lambda item: item['messages'][-1]['content']['result']['rank_support'].__setitem__('population_count', 4),
            lambda item: item['messages'][-1]['content'].__setitem__('status', 'not_found'),
            lambda item: item['messages'][-1]['content'].__setitem__('continuation_decision', 'continue'),
            lambda item: item['metadata'].__setitem__('degree', 4.0),
            lambda item: item['metadata'].__setitem__('wrapper_index', -1),
            lambda item: item['metadata'].__setitem__('request_template_index', -1),
        ]
        for mutation in mutations:
            changed = deepcopy(row)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(changed)
        blocked = self.row(variant='terminal')
        blocked['messages'][-1]['content']['blocked_reason'] = ' '
        with self.assertRaises(ValueError):
            self.validate(blocked)

    def test_source_weights_provenance_and_call_arguments_cannot_change(self):
        row = self.row(variant='native_context')
        changed = deepcopy(row)
        harness = context(changed)
        harness['observations'][0]['result']['neighbors'][0]['weight'] = 7.5
        changed['messages'][1]['content'] = json.dumps(harness)
        with self.assertRaises(ValueError):
            self.validate(changed)
        for path in ('call', 'provenance'):
            changed = deepcopy(self.row())
            if path == 'call':
                next(message for message in changed['messages'] if 'tool_call' in message)['tool_call']['arguments']['direction'] = 'incident'
            else:
                changed['messages'][-1]['content']['provenance']['graph'][0]['evidence_id'] = 'invented'
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.validate(changed)


if __name__ == '__main__':
    unittest.main()
