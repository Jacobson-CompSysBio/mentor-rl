"""Check shared-neighbor corpus traces against source sets and visible proof."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_7_corpus as corpus
from runtime.world_model.s2_2_corpus import VariantUnavailable
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_7_sampling import fact_from_queries
from runtime.world_model.s2_neighbors import ARGUMENT_KEYS, NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


def _calls(row):
    """Read current-turn calls without the supplied observations."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


def _context(row):
    """Read the exact model-visible harness data."""
    return json.loads(row['messages'][1]['content'])


def _answer(row):
    """Read the final structured answer without its free prose."""
    return row['messages'][-1]['content']


class WorldModelS27CorpusTests(unittest.TestCase):
    """Keep all query subjects in their existing S2.3 train split."""

    @classmethod
    def setUpClass(cls):
        """Create native source sets with a loop, a zero weight, and asymmetric directions."""
        pool = [f'ENSG{i:011d}' for i in range(1, 100)
                if gene_group(f'ENSG{i:011d}', SEED)[1] == 'train']
        cls.a, cls.b, cls.c, cls.d, cls.e, cls.f, cls.isolated, cls.graph_only, cls.missing = pool[:9]
        a, b, c, d, e, f, isolated, graph_only = pool[:8]
        cls.view = GraphView('unit_shared_corpus_view', 'f' * 64, {
            PPI: LayerGraph(PPI, False, {(a, a): .8, (a, b): .7, (a, c): .6, (a, d): 0,
                (b, c): .6, (b, d): 0, (c, e): .4}, frozenset((a, b, c, d, e, isolated))),
            COEX: LayerGraph(COEX, False, {(f, a): .2}, frozenset((a, f))),
            BRAIN: LayerGraph(BRAIN, True, {(a, c): .9, (a, d): .8, (b, c): .7,
                (c, a): .6, (c, b): .5, (d, b): .4, (b, e): .3, (e, a): .2,
                (b, graph_only): .1}, frozenset((a, b, c, d, e, isolated, graph_only))),
            CD4: LayerGraph(CD4, True, {(f, a): .2}, frozenset((a, f))),
        })
        cls.index = NeighborIndex(cls.view)
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_SHARED_{ordinal}' for ordinal, gene in enumerate(pool) if gene != graph_only})
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contract = corpus.load_build_contract()

    def _fact(self, genes=None, layer=PPI, directions=None):
        genes = genes or (self.a, self.b)
        directions = directions or ('undirected', 'undirected')
        return fact_from_queries(self.view, self.index, layer, tuple(zip(genes, directions)), SEED)

    def _row(self, genes=None, layer=PPI, *, directions=None, variant='lookup', framing='direct', variant_index=0, bank=None):
        return corpus.build_row(self.view, self._fact(genes, layer, directions), framing, variant, variant_index,
                                bank or self.bank, self.contract, registry=self.registry, index=self.index, seed=SEED)

    def _validate(self, row):
        corpus.validate_row(row, self.view, bank=self.bank, contract=self.contract, registry=self.registry, index=self.index)

    def test_complete_intersection_keeps_three_sorted_ids_and_exact_native_weights(self):
        row = self._row()
        self._validate(row)
        result = _answer(row)['result']
        self.assertEqual(result['shared_neighbors'], sorted((self.a, self.c, self.d)))
        self.assertEqual(result['shared_neighbor_count'], 3)
        self.assertEqual(result['collection'], {'mode': 'complete', 'returned_count': 3, 'total_count': 3,
                                               'complete': True, 'next_cursor': None})
        self.assertEqual([tuple(call['arguments']) for call in _calls(row)], [ARGUMENT_KEYS, ARGUMENT_KEYS])
        self.assertTrue(all(call['name'] == 'get_neighbors' for call in _calls(row)))
        pages = [message['content']['result'] for message in row['messages'] if message['role'] == 'tool']
        self.assertEqual(pages[0]['neighbors'][-1], {'gene_id': self.d, 'weight': 0})

    def test_all_nine_directed_role_pairs_preserve_their_meaning(self):
        expected = {('in', 'in'): [self.c], ('out', 'out'): [self.c], ('out', 'in'): [self.c, self.d],
                    ('in', 'out'): [self.c, self.e], ('both', 'in'): [self.c, self.d],
                    ('both', 'out'): [self.c, self.e], ('in', 'both'): [self.c, self.e],
                    ('out', 'both'): [self.c, self.d], ('both', 'both'): [self.c, self.d, self.e]}
        for directions, shared in expected.items():
            with self.subTest(directions=directions):
                row = self._row(layer=BRAIN, directions=directions)
                self._validate(row)
                self.assertEqual(_answer(row)['result']['shared_neighbors'], sorted(shared))
                self.assertTrue(all(call['arguments']['direction'] != 'both' for call in _calls(row)))

    def test_identical_roles_share_one_call_and_keep_both_query_roles(self):
        row = self._row((self.a, self.a))
        self._validate(row)
        result = _answer(row)['result']
        self.assertEqual(len(_calls(row)), 1)
        self.assertEqual(len(result['queries']), 2)
        self.assertEqual(result['queries'][0], result['queries'][1])
        self.assertEqual(len(result['input_memberships']), 1)
        self.assertEqual(len(result['coverage']['requested_queries']), 1)

    def test_same_gene_different_directions_require_distinct_native_sets(self):
        row = self._row((self.a, self.a), BRAIN, directions=('out', 'in'))
        self._validate(row)
        self.assertEqual(_answer(row)['result']['shared_neighbors'], [self.c])
        self.assertEqual([call['arguments']['direction'] for call in _calls(row)], ['out', 'in'])
        self.assertEqual(len(_answer(row)['result']['input_memberships']), 1)

    def test_self_loop_can_make_a_query_endpoint_shared(self):
        row = self._row()
        self._validate(row)
        self.assertIn(self.a, _answer(row)['result']['shared_neighbors'])
        self.assertNotIn(self.b, _answer(row)['result']['shared_neighbors'])

    def test_graph_only_returned_id_survives_without_a_registry_mapping(self):
        row = self._row((self.b, self.b), BRAIN, directions=('out', 'out'))
        self._validate(row)
        self.assertIn(self.graph_only, _answer(row)['result']['shared_neighbors'])
        self.assertEqual(_answer(row)['provenance']['registry'], [])

    def test_empty_intersection_follows_outer_stop_or_continue(self):
        for framing, decision in (('direct', 'stop'), ('step', 'continue')):
            row = self._row((self.isolated, self.a), framing=framing)
            self._validate(row)
            answer = _answer(row)
            self.assertEqual((answer['status'], answer['continuation_decision']), ('resolved', decision))
            self.assertEqual(answer['result']['shared_neighbors'], [])
            self.assertIsNone(answer['blocked_reason'])

    def test_empty_context_uses_global_positive_and_keeps_unobserved_layer_null(self):
        row = self._row((self.isolated, self.f), variant='empty_context')
        self._validate(row)
        self.assertEqual(_calls(row), [])
        result = _answer(row)['result']
        self.assertEqual(result['shared_neighbors'], [])
        self.assertEqual(result['input_memberships'][1], {'gene_id': self.f,
            'present_in_multiplex': True, 'present_in_layer': None})
        self.assertEqual(result['coverage']['unchecked_queries'], [result['queries'][1]])

    def test_global_absence_blocks_even_when_opposite_role_is_known_empty(self):
        row = self._row((self.isolated, self.missing), variant='empty_context')
        self._validate(row)
        answer, result = _answer(row), _answer(row)['result']
        self.assertEqual(_calls(row), [])
        self.assertEqual((answer['status'], answer['continuation_decision']), ('not_found', 'defer'))
        self.assertEqual(result['issue_code'], 'NO_MULTIPLEX_MEMBERSHIP')
        self.assertNotIn('shared_neighbors', result)
        self.assertNotIn('shared_neighbor_count', result)
        self.assertEqual(len(result['partial_results']), 2)
        self.assertEqual(len(result['coverage']['checked_queries']), 1)

    def test_global_missing_first_role_stops_before_unnecessary_second_lookup(self):
        row = self._row((self.missing, self.a))
        self._validate(row)
        self.assertEqual(len(_calls(row)), 1)
        self.assertEqual(_calls(row)[0]['arguments']['gene_id'], self.missing)
        self.assertEqual(_answer(row)['result']['coverage']['checked_queries'], [])
        self.assertEqual(len(_answer(row)['result']['partial_results']), 1)

    def test_one_context_list_replaces_only_its_call(self):
        for variant, expected in (('first_context', self.b), ('second_context', self.a)):
            row = self._row(variant=variant)
            self._validate(row)
            self.assertEqual(len(_calls(row)), 1)
            self.assertEqual(_calls(row)[0]['arguments']['gene_id'], expected)

    def test_two_complete_and_duplicate_contexts_answer_without_calls(self):
        for variant in ('context', 'duplicate_context'):
            row = self._row(variant=variant)
            self._validate(row)
            self.assertEqual(_calls(row), [])
            self.assertEqual(len(_answer(row)['provenance']['graph']), 2)

    def test_global_positive_context_does_not_supply_either_native_list(self):
        row = self._row(variant='global_context')
        self._validate(row)
        self.assertEqual(len(_calls(row)), 2)
        self.assertEqual(_answer(row)['result']['shared_neighbor_count'], 3)
        saved = _context(row)['observations'][0]
        self.assertNotIn(saved['provenance']['graph'][0], _answer(row)['provenance']['graph'])

    def test_global_negative_context_blocks_without_a_new_native_call(self):
        row = self._row((self.a, self.missing), variant='global_context')
        self._validate(row)
        self.assertEqual(_calls(row), [])
        self.assertEqual(_answer(row)['result']['issue_code'], 'NO_MULTIPLEX_MEMBERSHIP')
        self.assertEqual(_answer(row)['result']['coverage']['checked_queries'], [])

    def test_containment_context_is_exact_but_keeps_the_other_list_unchecked(self):
        row = self._row((self.e, self.a), variant='containment_context')
        self._validate(row)
        result = _answer(row)['result']
        self.assertEqual(_calls(row), [])
        self.assertEqual(result['shared_neighbors'], [self.c])
        self.assertEqual(result['coverage']['checked_queries'], [result['queries'][0]])
        self.assertEqual(result['coverage']['unchecked_queries'], [result['queries'][1]])

    def test_two_bounded_lists_with_a_common_id_still_need_complete_calls(self):
        row = self._row(variant='bounded_context')
        self._validate(row)
        saved = _context(row)['observations']
        self.assertEqual(saved[0]['result']['neighbors'][0]['gene_id'], self.a)
        self.assertEqual(saved[1]['result']['neighbors'][0]['gene_id'], self.a)
        self.assertEqual(len(_calls(row)), 2)
        self.assertTrue(all(call['arguments']['mode'] == 'complete' and call['arguments']['cursor'] is None for call in _calls(row)))
        self.assertEqual(_answer(row)['result']['shared_neighbor_count'], 3)
        self.assertTrue(all(record not in _answer(row)['provenance']['graph']
                            for observation in saved for record in observation['provenance']['graph']))

    def test_exact_complete_prefix_cursor_is_reused(self):
        row = self._row(variant='prefix_context')
        self._validate(row)
        saved = _context(row)['observations'][0]
        expected = dict(saved['arguments'], cursor=saved['result']['collection']['next_cursor'])
        self.assertEqual(_calls(row)[0]['arguments'], expected)
        self.assertIn(saved['provenance']['graph'][0], _answer(row)['provenance']['graph'])

    def test_overlap_empty_pages_and_unknown_totals_keep_the_full_intersection(self):
        for variant in ('paged', 'overlap', 'empty_page', 'unknown_total'):
            with self.subTest(variant=variant):
                row = self._row(variant=variant)
                self._validate(row)
                self.assertEqual(_answer(row)['result']['shared_neighbors'], sorted((self.a, self.c, self.d)))
                responses = [message['content']['result'] for message in row['messages'] if message['role'] == 'tool']
                if variant == 'unknown_total':
                    self.assertTrue(all(result['collection']['total_count'] is None for result in responses))
                if variant == 'empty_page':
                    self.assertTrue(any(result['neighbors'] == [] and result['collection']['next_cursor'] is not None for result in responses))

    def test_failures_keep_completed_lists_and_partial_pages_truthfully_unchecked(self):
        for variant, partial_count, checked_count in (('terminal', 0, 0), ('exhausted', 0, 0),
                ('second_terminal', 1, 1), ('second_exhausted', 1, 1), ('page_terminal', 1, 0), ('page_exhausted', 1, 0)):
            with self.subTest(variant=variant):
                row = self._row(variant=variant)
                self._validate(row)
                answer, result = _answer(row), _answer(row)['result']
                self.assertEqual((answer['status'], answer['continuation_decision']), ('error', 'defer'))
                self.assertEqual(len(result['partial_results']), partial_count)
                self.assertEqual(len(result['coverage']['checked_queries']), checked_count)
                self.assertEqual(result['arguments'], _calls(row)[-1]['arguments'])
                self.assertNotIn('shared_neighbors', result)
                self.assertNotIn('shared_neighbor_count', result)
                if 'exhausted' in variant:
                    self.assertEqual(_calls(row)[-1], _calls(row)[-2])
                if variant.startswith('page_'):
                    self.assertIsNotNone(result['arguments']['cursor'])

    def test_authorized_retries_repeat_all_arguments_and_recover(self):
        for variant in ('recovered', 'second_recovered', 'page_recovered'):
            with self.subTest(variant=variant):
                row = self._row(variant=variant, framing='step')
                self._validate(row)
                calls = _calls(row)
                self.assertEqual(calls[0], calls[1])
                self.assertEqual((_answer(row)['status'], _answer(row)['continuation_decision']), ('resolved', 'continue'))
                self.assertEqual(_answer(row)['result']['shared_neighbor_count'], 3)

    def test_symbol_context_keeps_duplicate_raw_roles_but_one_identity_observation(self):
        row = self._row((self.a, self.a), variant='symbol_context')
        self._validate(row)
        self.assertEqual(row['metadata']['input_genes'][0], row['metadata']['input_genes'][1])
        self.assertEqual(len(_context(row)['observations']), 1)
        self.assertEqual(len(_answer(row)['provenance']['registry']), 1)
        self.assertEqual(_calls(row)[0]['arguments']['gene_id'], self.a)
        self.assertIn(row['metadata']['input_genes'][0], row['messages'][2]['content'])

    def test_stale_wrong_layer_and_distractor_records_cannot_supply_the_requested_sets(self):
        for variant in ('stale', 'wrong_layer', 'distractor'):
            with self.subTest(variant=variant):
                row = self._row(variant=variant)
                self._validate(row)
                self.assertEqual(len(_calls(row)), 2)
                for saved in _context(row)['observations']:
                    self.assertNotIn(saved['provenance']['graph'][0], _answer(row)['provenance']['graph'])

    def test_wrong_direction_requires_the_requested_orientation(self):
        row = self._row(layer=BRAIN, directions=('out', 'in'), variant='wrong_direction')
        self._validate(row)
        self.assertEqual(_calls(row)[0]['arguments']['direction'], 'out')
        self.assertEqual(_answer(row)['result']['shared_neighbors'], sorted((self.c, self.d)))
        self.assertNotIn(_context(row)['observations'][0]['provenance']['graph'][0], _answer(row)['provenance']['graph'])

    def test_role_swap_preserves_bindings_and_the_pair_split_unit(self):
        baseline = self._row(layer=BRAIN, directions=('out', 'in'))
        swapped = self._row(layer=BRAIN, directions=('out', 'in'), variant='swapped')
        self._validate(baseline)
        self._validate(swapped)
        self.assertEqual(swapped['metadata']['queries'], list(reversed(baseline['metadata']['queries'])))
        self.assertEqual(swapped['metadata']['split_unit'], baseline['metadata']['split_unit'])
        self.assertEqual(_answer(swapped)['result']['shared_neighbors'], _answer(baseline)['result']['shared_neighbors'])
        self.assertEqual(_calls(swapped)[0]['arguments']['gene_id'], self.b)
        self.assertEqual(_calls(swapped)[0]['arguments']['direction'], 'in')

    def test_distractor_facts_do_not_depend_on_the_wording_bank_size(self):
        bank = deepcopy(self.bank)
        wording = len(bank['train']['direct'])
        before = self._row(variant='distractor', variant_index=wording)
        bank['train']['direct'].append('{request} Please finish this request here.')
        after = self._row(variant='distractor', variant_index=wording, bank=bank)
        self.assertNotEqual(before['messages'][2]['content'], after['messages'][2]['content'])
        first, second = _context(before)['observations'][0], _context(after)['observations'][0]
        self.assertEqual(first['arguments'], second['arguments'])
        self.assertEqual(first['result'], second['result'])

    def test_all_local_diagnostics_preserve_raw_json_and_have_no_calls(self):
        expected = {'unknown_layer': ('not_found', 'UNKNOWN_LAYER'), 'unknown_multiplex': ('not_found', 'UNKNOWN_MULTIPLEX'),
                    'missing_scope': ('invalid_input', 'MISSING_SCOPE'), 'missing_genes': ('invalid_input', 'MISSING_GENES'),
                    'invalid_pair': ('invalid_input', 'INVALID_QUERY_PAIR'), 'invalid_directions': ('invalid_input', 'DIRECTION_UNAVAILABLE'),
                    'ranking_required': ('ambiguous', 'SHARED_NEIGHBOR_RANKING_REQUIRED')}
        self.assertEqual(set(corpus.DIAGNOSTICS), set(expected))
        for kind, (status, issue) in expected.items():
            with self.subTest(kind=kind):
                row = corpus.build_diagnostic(self.view, self._fact(), 'step', 0, self.bank, self.contract, kind=kind)
                self._validate(row)
                answer, result = _answer(row), _answer(row)['result']
                self.assertEqual(_calls(row), [])
                self.assertEqual((answer['status'], answer['continuation_decision']), (status, 'defer'))
                self.assertEqual(result['issue_code'], issue)
                self.assertEqual(answer['provenance'], {'registry': [], 'graph': []})
                self.assertIsNone(result['rejected_arguments'])
                self.assertIsNone(result['coverage'])
                self.assertIn(json.dumps(result['request']['genes'], ensure_ascii=False), row['messages'][2]['content'])
                self.assertIn(json.dumps(result['request']['directions'], ensure_ascii=False), row['messages'][2]['content'])

    def test_validator_rejects_counts_status_role_cursor_provenance_and_oracle_changes(self):
        mutations = []
        row = self._row()
        _answer(row)['result']['shared_neighbor_count'] = 2
        mutations.append(row)
        row = self._row()
        _answer(row)['status'] = 'not_found'
        mutations.append(row)
        row = self._row(layer=BRAIN, directions=('out', 'in'))
        _answer(row)['result']['queries'][0]['direction'] = 'in'
        mutations.append(row)
        row = self._row(variant='prefix_context')
        _calls(row)[0]['arguments']['cursor'] = None
        mutations.append(row)
        row = self._row(variant='recovered')
        _calls(row)[1]['arguments']['top_k'] = 1
        mutations.append(row)
        row = self._row()
        _answer(row)['provenance']['graph'].pop()
        mutations.append(row)
        row = self._row()
        next(message for message in row['messages'] if message['role'] == 'tool')['content']['result']['neighbors'][0]['weight'] += .01
        mutations.append(row)
        row = self._row(variant='second_terminal')
        _answer(row)['result']['partial_results'] = []
        mutations.append(row)
        row = self._row()
        row['metadata']['shared_count'] = 1
        mutations.append(row)
        row = self._row()
        row['metadata']['shared_count'] = 3.0
        mutations.append(row)
        row = self._row()
        row['metadata']['neighbor_counts'] = [4.0, 3.0]
        mutations.append(row)
        row = self._row(variant_index=len(self.bank['train']['direct']) - 1)
        row['metadata']['wrapper_index'] = -1
        mutations.append(row)
        row = self._row()
        context = _context(row)
        context['private_shared_count'] = 3
        row['messages'][1]['content'] = json.dumps(context)
        mutations.append(row)
        for ordinal, row in enumerate(mutations):
            with self.subTest(mutation=ordinal), self.assertRaises(ValueError):
                self._validate(row)

    def test_final_summary_and_blocker_prose_remain_free(self):
        row = self._row()
        self.assertEqual(set(_context(row)), {'active_graph', 'observations'})
        self.assertNotIn('shared_count', row['messages'][1]['content'])
        _answer(row)['summary'] = 'The requested shared set is complete.'
        self._validate(row)
        row = self._row(variant='terminal')
        _answer(row)['blocked_reason'] = 'The neighbor backend must recover.'
        _answer(row)['summary'] = 'The requested shared set remains unavailable.'
        self._validate(row)


if __name__ == '__main__':
    unittest.main()
