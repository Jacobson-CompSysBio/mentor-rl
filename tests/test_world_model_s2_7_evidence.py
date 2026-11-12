"""Check visible shared-set proofs, native pages, and membership prerequisites."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.s2_7_evidence import atomic_queries, collect, validate_arguments
from runtime.world_model.s2_gene_layers import GeneLayersRuntime
from runtime.world_model.s2_multiplex import EvidenceError, SelectionError
from runtime.world_model.s2_neighbors import NeighborIndex, NeighborRuntime
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class SharedEvidenceTests(unittest.TestCase):
    """Use explicit inventories with a loop, a zero weight, and native directions."""

    def setUp(self):
        self.a, self.b, self.c, self.d, self.e, self.f, self.missing = [f'ENSG{i:011d}' for i in range(1, 8)]
        self.ppi, self.tf, self.other = 'HumanNetV3:string_ppi', 'TFs:brain', 'TFs:cd4+t_cells'
        self.view = GraphView('shared_test', 'shared_version', {
            self.ppi: LayerGraph(self.ppi, False, {(self.a, self.a): .8, (self.a, self.b): .7,
                (self.a, self.c): .6, (self.a, self.d): 0, (self.b, self.c): .6,
                (self.b, self.d): 0, (self.c, self.e): .4}, frozenset((self.a, self.b, self.c, self.d, self.e))),
            self.tf: LayerGraph(self.tf, True, {(self.a, self.c): .9, (self.a, self.d): .8,
                (self.b, self.c): .7, (self.c, self.a): .6, (self.c, self.b): .5},
                frozenset((self.a, self.b, self.c, self.d, self.e))),
            self.other: LayerGraph(self.other, True, {(self.f, self.a): .2}, frozenset((self.f, self.a))),
        })
        self.runtime = NeighborRuntime(self.view, index=NeighborIndex(self.view))

    def arguments(self, first=None, second=None, *, layer=None, directions=None):
        directions = directions or ('undirected', 'undirected')
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': layer or self.ppi,
                'queries': [{'gene_id': first or self.a, 'direction': directions[0]},
                            {'gene_id': second or self.b, 'direction': directions[1]}]}

    def native(self, gene, *, direction='undirected', layer=None, mode='complete', top_k=None,
               cursor=None, runtime=None, identity=None):
        args = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer or self.ppi,
                'gene_id': gene, 'direction': direction, 'mode': mode, 'top_k': top_k, 'cursor': cursor}
        return {'operation': 'get_neighbors', 'arguments': args,
                **(runtime or self.runtime).get_neighbors(**args, evidence_id=identity or f'ev_{gene}_{direction}')}

    def global_observation(self, genes, identity='ev_global'):
        return {'operation': 'check_multiplex_membership',
                'arguments': {'multiplex_id': self.view.multiplex_id, 'gene_ids': list(genes)},
                'status': 'resolved', 'result': {'multiplex_id': self.view.multiplex_id,
                    'present_gene_ids': [gene for gene in genes if gene in self.view.multiplex_nodes],
                    'absent_gene_ids': [gene for gene in genes if gene not in self.view.multiplex_nodes]},
                'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                    'graph_version': self.view.graph_version,
                    'layer_scope': {'kind': 'all', 'layer_ids': [], 'layer_family': None}, 'evidence_id': identity}]}}

    def finish(self, args, observations=(), runtime=None):
        observations = list(observations)
        for ordinal in range(100):
            proof = collect(self.view, args, observations)
            if proof.next_arguments is None:
                return proof
            call = proof.next_arguments
            observations.append({'operation': 'get_neighbors', 'arguments': call,
                **(runtime or self.runtime).get_neighbors(**call, evidence_id=f'ev_new_{ordinal}')})
        self.fail('the planner did not finish')

    def test_initial_call_and_roles_preserve_order(self):
        args = self.arguments(self.b, self.a)
        proof = collect(self.view, args, [])
        self.assertEqual(proof.next_tool, 'get_neighbors')
        self.assertEqual(proof.next_arguments, {'multiplex_id': self.view.multiplex_id,
            'layer_id': self.ppi, 'gene_id': self.b, 'direction': 'undirected',
            'mode': 'complete', 'top_k': None, 'cursor': None})
        self.assertEqual(proof.coverage['unchecked_queries'], args['queries'])

    def test_planner_never_reads_source_inventories_or_adjacency(self):
        class Hidden:
            def __contains__(self, value):
                raise AssertionError('hidden inventory read')

            def __iter__(self):
                raise AssertionError('hidden inventory read')

        saved = [self.native(self.a), self.native(self.b)]
        object.__setattr__(self.view, 'multiplex_nodes', Hidden())
        for layer in self.view.layers.values():
            object.__setattr__(layer, 'nodes', Hidden())
            object.__setattr__(layer, 'edges', Hidden())
        with patch.object(NeighborRuntime, 'get_neighbors', side_effect=AssertionError('hidden acquisition')):
            proof = collect(self.view, self.arguments(), saved)
        self.assertEqual(proof.result['shared_neighbors'], [self.a, self.c, self.d])

    def test_complete_lists_have_sorted_ids_and_retain_zero_weight_and_loop(self):
        saved = [self.native(self.a), self.native(self.b)]
        proof = collect(self.view, self.arguments(), saved)
        self.assertEqual(proof.result['shared_neighbors'], [self.a, self.c, self.d])
        self.assertEqual(proof.result['shared_neighbor_count'], 3)
        self.assertEqual(proof.observations, tuple(saved))
        self.assertEqual(proof.result['collection'], {'mode': 'complete', 'returned_count': 3,
            'total_count': 3, 'complete': True, 'next_cursor': None})
        self.assertEqual(saved[0]['result']['neighbors'][-1], {'gene_id': self.d, 'weight': 0})
        self.assertNotIn('weight', proof.result)

    def test_identical_roles_share_one_primitive(self):
        args = self.arguments(self.a, self.a)
        proof = self.finish(args)
        self.assertEqual(proof.result['queries'], args['queries'])
        self.assertEqual(len(proof.observations), 1)
        self.assertEqual(len(proof.coverage['checked_queries']), 1)
        self.assertEqual(proof.result['shared_neighbors'], [self.a, self.b, self.c, self.d])
        self.assertEqual(len(proof.result['input_memberships']), 1)

    def test_same_gene_different_directions_remain_distinct(self):
        args = self.arguments(self.a, self.a, layer=self.tf, directions=('in', 'out'))
        proof = self.finish(args)
        self.assertEqual(proof.result['shared_neighbors'], [self.c])
        self.assertEqual([item['arguments']['direction'] for item in proof.observations], ['in', 'out'])
        self.assertEqual(len(proof.result['input_memberships']), 1)

    def test_native_direction_combinations(self):
        for directions, expected in ((('out', 'out'), [self.c]), (('in', 'in'), [self.c]),
                                     (('out', 'in'), [self.c]), (('both', 'both'), [self.c])):
            with self.subTest(directions=directions):
                proof = self.finish(self.arguments(layer=self.tf, directions=directions))
                self.assertEqual(proof.result['shared_neighbors'], expected)
                self.assertTrue(all(item['arguments']['direction'] != 'both' for item in proof.observations))

    def test_exhaustive_directed_pairs_match_native_sets_with_unknown_total_pages(self):
        runtime = NeighborRuntime(self.view, page_size=1, disclose_total=False)
        roles = [(gene, direction) for gene in (self.a, self.b, self.d, self.f)
                 for direction in ('in', 'out', 'both')]
        for first in roles:
            for second in roles:
                with self.subTest(first=first, second=second):
                    args = self.arguments(first[0], second[0], layer=self.tf, directions=(first[1], second[1]))
                    proof = self.finish(args, runtime=runtime)
                    sets = []
                    for gene, direction in (first, second):
                        directions = ('in', 'out') if direction == 'both' else (direction,)
                        sets.append({neighbor for native_direction in directions
                                     for neighbor, _ in self.runtime.index.neighbors(self.tf, gene, native_direction)})
                    self.assertEqual(proof.result['shared_neighbors'], sorted(sets[0] & sets[1]))

    def test_both_expansion_has_first_role_order_and_deduplicates(self):
        args = self.arguments(self.a, self.a, layer=self.tf, directions=('both', 'out'))
        self.assertEqual(atomic_queries(args), [{'gene_id': self.a, 'direction': 'in'},
                                              {'gene_id': self.a, 'direction': 'out'}])

    def test_one_empty_orientation_does_not_empty_both_union(self):
        args = self.arguments(self.d, self.a, layer=self.tf, directions=('both', 'out'))
        empty = self.native(self.d, layer=self.tf, direction='out')
        global_members = self.global_observation([self.d, self.a])
        proof = collect(self.view, args, [empty, global_members])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['direction'], 'in')
        self.assertEqual(proof.next_arguments['gene_id'], self.d)

    def test_complete_empty_and_global_positive_avoid_second_list(self):
        args = self.arguments(self.d, self.f, layer=self.tf, directions=('out', 'out'))
        saved = [self.native(self.d, layer=self.tf, direction='out'), self.global_observation([self.f])]
        proof = collect(self.view, args, saved)
        self.assertEqual(proof.result['shared_neighbors'], [])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['input_memberships'][1]['present_in_layer'], None)
        self.assertEqual(proof.coverage['unchecked_queries'], [args['queries'][1]])
        self.assertEqual(proof.observations, tuple(saved))

    def test_empty_set_without_opposite_global_fact_requires_a_native_call(self):
        args = self.arguments(self.d, self.f, layer=self.tf, directions=('out', 'out'))
        proof = collect(self.view, args, [self.native(self.d, layer=self.tf, direction='out')])
        self.assertEqual(proof.next_arguments['gene_id'], self.f)
        self.assertIsNone(proof.result)

    def test_global_absence_blocks_even_opposite_set_empty(self):
        args = self.arguments(self.d, self.missing, layer=self.tf, directions=('out', 'out'))
        saved = [self.native(self.d, layer=self.tf, direction='out'), self.global_observation([self.missing])]
        proof = collect(self.view, args, saved)
        self.assertEqual(proof.issue_code, 'NO_MULTIPLEX_MEMBERSHIP')
        self.assertIsNone(proof.result)
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.observations, tuple(saved))
        self.assertEqual(proof.coverage['checked_queries'], [args['queries'][0]])

    def test_unavailable_native_response_blocks_and_is_unchecked(self):
        args = self.arguments(self.missing, self.a)
        saved = self.native(self.missing)
        proof = collect(self.view, args, [saved])
        self.assertEqual(proof.issue_code, 'NO_MULTIPLEX_MEMBERSHIP')
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.coverage['checked_queries'], [])

    def test_layer_absence_from_s26_derives_all_empty_directions(self):
        runtime = NodesByLayerRuntime(self.view)
        args = runtime.arguments(self.tf, [self.f, self.a])
        saved = {'operation': 'get_nodes_by_layer', 'arguments': args,
                 **runtime.get_nodes_by_layer(**args, evidence_id='ev_nodes')}
        target = self.arguments(self.f, self.a, layer=self.tf, directions=('both', 'both'))
        proof = collect(self.view, target, [saved])
        self.assertEqual(proof.result['shared_neighbors'], [])
        self.assertEqual(proof.coverage['checked_queries'], atomic_queries(target)[:2])
        self.assertEqual(proof.result['input_memberships'][0]['present_in_layer'], False)

    def test_s25_complete_memberships_supply_global_and_layer_absence(self):
        runtime = GeneLayersRuntime(self.view)
        scope = {'kind': 'all', 'layer_ids': [], 'layer_family': None}
        source = runtime.arguments(self.f, scope)
        saved = {'operation': 'get_gene_layers', 'arguments': source,
                 **runtime.get_gene_layers(**source, evidence_id='ev_layers')}
        target = self.arguments(self.f, self.a, layer=self.tf, directions=('out', 'out'))
        proof = collect(self.view, target, [saved, self.global_observation([self.a])])
        self.assertEqual(proof.result['shared_neighbors'], [])
        self.assertEqual(proof.coverage['checked_queries'], [target['queries'][0]])

    def test_incomplete_common_ids_are_only_a_lower_bound(self):
        saved = [self.native(self.a, mode='top_k', top_k=1),
                 self.native(self.b, mode='top_k', top_k=1)]
        proof = collect(self.view, self.arguments(), saved)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['gene_id'], self.a)
        self.assertEqual(proof.next_arguments['mode'], 'complete')
        self.assertIsNone(proof.next_arguments['cursor'])
        self.assertEqual(proof.coverage['checked_queries'], [])

    def test_disjoint_incomplete_lists_do_not_prove_an_empty_intersection(self):
        args = self.arguments(self.a, self.e)
        saved = [self.native(self.a, mode='top_k', top_k=1),
                 self.native(self.e, mode='top_k', top_k=1)]
        proof = collect(self.view, args, saved)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['gene_id'], self.a)
        self.assertEqual(self.finish(args, saved).result['shared_neighbors'], [self.c])

    def test_fresh_complete_evidence_replaces_redundant_bounded_context(self):
        saved = [self.native(self.a, mode='top_k', top_k=1, identity='ev_bound_a'),
                 self.native(self.b, mode='top_k', top_k=1, identity='ev_bound_b')]
        proof = self.finish(self.arguments(), saved)
        self.assertEqual(proof.result['shared_neighbors'], [self.a, self.c, self.d])
        self.assertEqual(len(proof.observations), 2)
        self.assertTrue(all(item['arguments']['mode'] == 'complete' for item in proof.observations))

    def test_complete_containment_in_bounded_positives_is_exact(self):
        args = self.arguments(self.e, self.a)
        saved = [self.native(self.e), self.native(self.a, mode='top_k', top_k=3)]
        proof = collect(self.view, args, saved)
        self.assertEqual(proof.result['shared_neighbors'], [self.c])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.coverage['checked_queries'], [args['queries'][0]])
        self.assertEqual(proof.coverage['unchecked_queries'], [args['queries'][1]])

    def test_partial_complete_containment_stops_before_unneeded_page(self):
        runtime = NeighborRuntime(self.view, page_size=3)
        args = self.arguments(self.e, self.a)
        saved = [self.native(self.e), self.native(self.a, runtime=runtime)]
        proof = collect(self.view, args, saved)
        self.assertEqual(proof.result['shared_neighbors'], [self.c])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.coverage['unchecked_queries'], [args['queries'][1]])

    def test_complete_cursor_resumes_exact_selection_and_retains_pages(self):
        runtime = NeighborRuntime(self.view, page_size=2, disclose_total=False, overlap=1, empty_page_at=1)
        first = self.native(self.a, runtime=runtime)
        proof = collect(self.view, self.arguments(), [first])
        expected = dict(first['arguments'], cursor=first['result']['collection']['next_cursor'])
        self.assertEqual(proof.next_arguments, expected)
        self.assertEqual(proof.coverage['checked_queries'], [])
        proof = self.finish(self.arguments(), [first], runtime)
        self.assertEqual(proof.result['shared_neighbors'], [self.a, self.c, self.d])
        self.assertGreater(len(proof.observations), 2)

    def test_count_reaches_known_total_but_pending_cursor_stays_unchecked(self):
        runtime = NeighborRuntime(self.view, page_size=2)
        first = self.native(self.a, runtime=runtime, identity='ev_p1')
        second = self.native(self.a, runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='ev_p2')
        second['result']['collection']['next_cursor'] = 'opaque_pending'
        proof = collect(self.view, self.arguments(), [first, second])
        self.assertEqual(proof.next_arguments['cursor'], 'opaque_pending')
        self.assertEqual(proof.coverage['checked_queries'], [])
        self.assertIsNone(proof.result)

    def test_pending_failure_partials_retain_global_and_successful_page(self):
        runtime = NeighborRuntime(self.view, page_size=1)
        saved = [self.global_observation([self.b]), self.native(self.a, runtime=runtime)]
        failed = {'operation': 'get_neighbors', 'arguments': {'multiplex_id': self.view.multiplex_id},
                  'status': 'error', 'result': {'error_code': 'UNAVAILABLE'},
                  'provenance': {'registry': [], 'graph': []}}
        proof = collect(self.view, self.arguments(), [*saved, failed])
        self.assertEqual(proof.observations, tuple(saved))
        self.assertEqual(proof.coverage['checked_queries'], [])

    def test_wrong_direction_supplies_membership_but_does_not_answer_role(self):
        args = self.arguments(layer=self.tf, directions=('out', 'out'))
        saved = self.native(self.a, layer=self.tf, direction='in')
        proof = collect(self.view, args, [saved])
        self.assertEqual(proof.next_arguments['gene_id'], self.a)
        self.assertEqual(proof.next_arguments['direction'], 'out')
        self.assertEqual(proof.coverage['checked_queries'], [])

    def test_wrong_version_and_other_query_do_not_supply_current_facts(self):
        wrong = self.native(self.a)
        wrong['provenance']['graph'][0]['graph_version'] = 'older'
        proof = collect(self.view, self.arguments(), [wrong, self.native(self.c)])
        self.assertEqual(proof.observations, ())
        self.assertEqual(proof.next_arguments['gene_id'], self.a)

    def test_duplicate_observations_keep_one_original_acquisition(self):
        saved = [self.native(self.a), self.native(self.b)]
        proof = collect(self.view, self.arguments(), [saved[0], saved[0], saved[1]])
        self.assertEqual(proof.observations, tuple(saved))
        self.assertEqual(len(proof.provenance['graph']), 2)

    def test_evidence_id_reuse_is_a_data_fault(self):
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments(), [self.native(self.a, identity='same'), self.native(self.b, identity='same')])

    def test_conflicting_membership_is_a_data_fault(self):
        saved = self.global_observation([self.a])
        saved['result']['present_gene_ids'] = []
        saved['result']['absent_gene_ids'] = [self.a]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments(), [self.native(self.a), saved])

    def test_conflicting_weights_totals_and_prefixes_are_data_faults(self):
        full = self.native(self.a)
        for field in ('weight', 'total', 'prefix'):
            with self.subTest(field=field):
                changed = self.native(self.a, mode='top_k', top_k=2, identity='changed')
                if field == 'weight':
                    changed['result']['neighbors'][0]['weight'] += .01
                elif field == 'total':
                    changed['result']['collection']['total_count'] += 1
                else:
                    changed['result']['neighbors'][1]['gene_id'] = self.e
                with self.assertRaises(EvidenceError):
                    collect(self.view, self.arguments(), [full, changed])

    def test_invalid_rows_and_collection_are_data_faults(self):
        mutations = [lambda page: page['result']['neighbors'][0].update(weight=float('nan')),
                     lambda page: page['result']['neighbors'][0].update(extra='invalid'),
                     lambda page: page['result']['neighbors'].reverse(),
                     lambda page: page['result']['collection'].update(extra='invalid'),
                     lambda page: page['result']['collection'].update(returned_count=True),
                     lambda page: page['provenance']['graph'][0].update(evidence_id=''),
                     lambda page: page['result'].update(gene_id=self.b)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                saved = self.native(self.a)
                mutate(saved)
                with self.assertRaises(EvidenceError):
                    collect(self.view, self.arguments(), [saved])

    def test_missing_initial_page_and_disconnected_cursor_are_data_faults(self):
        runtime = NeighborRuntime(self.view, page_size=1)
        first = self.native(self.a, runtime=runtime, identity='first')
        later = self.native(self.a, runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='later')
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments(), [later])
        later['arguments']['cursor'] = 'unrelated'
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments(), [first, later])

    def test_invalid_argument_diagnostics(self):
        invalid = [(dict(self.arguments(), multiplex_id=None), 'MISSING_SCOPE'),
                   (dict(self.arguments(), multiplex_id='other'), 'UNKNOWN_MULTIPLEX'),
                   (dict(self.arguments(), layer_id='unknown'), 'UNKNOWN_LAYER'),
                   (dict(self.arguments(), queries=None), 'MISSING_GENES'),
                   (dict(self.arguments(), queries=[]), 'INVALID_QUERY_PAIR'),
                   (dict(self.arguments(), queries=[self.arguments()['queries'][0]]), 'INVALID_QUERY_PAIR'),
                   (self.arguments(directions=('out', 'out')), 'DIRECTION_UNAVAILABLE'),
                   (self.arguments(layer=self.tf), 'DIRECTION_UNAVAILABLE')]
        for args, code in invalid:
            with self.subTest(code=code), self.assertRaises(SelectionError) as caught:
                validate_arguments(self.view, args)
            self.assertEqual(caught.exception.issue_code, code)


if __name__ == '__main__':
    unittest.main()
