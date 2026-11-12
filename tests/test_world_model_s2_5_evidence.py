"""Check visible-only membership proofs, collection boundaries, and conflicts."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.s2_5_evidence import collect
from runtime.world_model.s2_gene_layers import GeneLayersRuntime
from runtime.world_model.s2_multiplex import EvidenceError
from runtime.world_model.s2_neighbor_layers import NeighborLayersRuntime
from runtime.world_model.s2_neighbors import NeighborIndex, NeighborRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class MembershipEvidenceTests(unittest.TestCase):
    """Use explicit inventories that include a real fixture isolate."""

    def setUp(self):
        self.a, self.b, self.c, self.d, self.e, self.missing = [f'ENSG{i:011d}' for i in range(1, 7)]
        self.ppi, self.coex, self.brain, self.cd4 = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells')
        self.view = GraphView('test_memberships', 'version_memberships', {
            self.ppi: LayerGraph(self.ppi, False, {(self.a, self.b): .2}, frozenset((self.a, self.b, self.e))),
            self.coex: LayerGraph(self.coex, False, {(self.a, self.c): .3}, frozenset((self.a, self.c))),
            self.brain: LayerGraph(self.brain, True, {(self.b, self.a): .4}, frozenset((self.a, self.b))),
            self.cd4: LayerGraph(self.cd4, True, {(self.d, self.a): .5}, frozenset((self.a, self.d))),
        })
        self.runtime = GeneLayersRuntime(self.view)
        self.scope = {'kind': 'all', 'layer_ids': [], 'layer_family': None}

    def arguments(self, **changes):
        return self.runtime.arguments(self.a, self.scope) | changes

    def subset(self, *layers):
        return {'kind': 'layers', 'layer_ids': list(layers), 'layer_family': None}

    def observe(self, arguments, runtime=None, identity='ev_test'):
        runtime = runtime or self.runtime
        return {'operation': 'get_gene_layers', 'arguments': deepcopy(arguments),
                **runtime.get_gene_layers(**arguments, evidence_id=identity)}

    def global_observation(self, gene=None, present=True, identity='ev_global'):
        gene = gene or self.a
        return {'operation': 'check_multiplex_membership',
            'arguments': {'multiplex_id': self.view.multiplex_id, 'gene_ids': [gene]}, 'status': 'resolved',
            'result': {'multiplex_id': self.view.multiplex_id,
                       'present_gene_ids': [gene] if present else [], 'absent_gene_ids': [] if present else [gene]},
            'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                'graph_version': self.view.graph_version, 'layer_scope': self.scope, 'evidence_id': identity}]}}

    def traverse(self, runtime, arguments, observations=None):
        observations = list(observations or [])
        for ordinal in range(20):
            proof = collect(self.view, arguments, observations)
            if proof.result is not None:
                return observations, proof
            self.assertEqual(proof.next_tool, 'get_gene_layers')
            observations.append(self.observe(proof.next_arguments, runtime, f'ev_page_{ordinal}'))
        self.fail('the fixture traversal did not finish')

    def test_initial_request_uses_no_hidden_membership_or_adjacency(self):
        class HiddenInventory:
            def __contains__(self, value):
                raise AssertionError('the planner read hidden membership')

            def __iter__(self):
                raise AssertionError('the planner read hidden membership')

        object.__setattr__(self.view, 'multiplex_nodes', HiddenInventory())
        for layer in self.view.layers.values():
            object.__setattr__(layer, 'nodes', HiddenInventory())
            object.__setattr__(layer, 'edges', HiddenInventory())
        with patch.object(GeneLayersRuntime, 'get_gene_layers', side_effect=AssertionError('hidden oracle')):
            proof = collect(self.view, self.arguments(), [])
        self.assertEqual(proof.next_arguments, self.arguments())
        self.assertEqual(proof.coverage['unchecked_layer_ids'], list(self.view.layer_order))

    def test_complete_context_and_shorter_prefix_need_no_call(self):
        saved = self.observe(self.arguments())
        proof = collect(self.view, self.arguments(mode='top_k', top_k=2), [saved, saved])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['layer_ids'], [self.ppi, self.coex])
        self.assertEqual(proof.result['layer_count'], 4)
        self.assertFalse(proof.result['collection']['complete'])
        self.assertEqual(proof.observations, (saved,))
        bounded = self.observe(self.arguments(mode='top_k', top_k=3))
        proof = collect(self.view, self.arguments(mode='top_k', top_k=2), [bounded])
        self.assertEqual(proof.result['layer_ids'], [self.ppi, self.coex])
        self.assertEqual(proof.result['layer_count'], 4)

    def test_complete_projection_recomputes_scoped_counts_and_preserves_scope(self):
        saved = self.observe(self.arguments())
        family = {'kind': 'all', 'layer_ids': [], 'layer_family': 'tf_target'}
        proof = collect(self.view, self.arguments(layer_scope=family), [saved])
        self.assertEqual(proof.result['layer_ids'], [self.brain, self.cd4])
        self.assertEqual(proof.result['layer_count'], 2)
        self.assertEqual(proof.provenance, saved['provenance'])
        bounded = self.observe(self.arguments(mode='top_k', top_k=1))
        proof = collect(self.view, self.arguments(layer_scope=family), [bounded])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.arguments(layer_scope=family))

    def test_saved_family_fetches_only_missing_layers_then_regroups(self):
        family = {'kind': 'all', 'layer_ids': [], 'layer_family': 'tf_target'}
        saved = self.observe(self.arguments(layer_scope=family))
        proof = collect(self.view, self.arguments(), [saved])
        self.assertEqual(proof.next_arguments['layer_scope'], self.subset(self.ppi, self.coex))
        self.assertEqual(proof.coverage['checked_layer_ids'], [self.brain, self.cd4])
        pages, proof = self.traverse(self.runtime, self.arguments(), [saved])
        self.assertEqual(len(pages), 2)
        self.assertEqual(proof.result['layer_ids'], list(self.view.layer_order))
        self.assertEqual(proof.result['layer_count'], 4)
        self.assertEqual(proof.observations[0], saved)

    def test_overlapping_complete_scopes_deduplicate_the_visible_union(self):
        first = self.observe(self.arguments(layer_scope=self.subset(self.ppi, self.brain)), identity='ev_first')
        second = self.observe(self.arguments(layer_scope=self.subset(self.coex, self.brain, self.cd4)), identity='ev_second')
        proof = collect(self.view, self.arguments(), [first, second])
        self.assertEqual(proof.result['layer_ids'], list(self.view.layer_order))
        self.assertEqual(proof.result['layer_count'], 4)
        self.assertEqual(len(proof.provenance['graph']), 2)

    def test_complete_pages_accumulate_and_copy_cursors_exactly(self):
        runtime = GeneLayersRuntime(self.view, page_size=2)
        first = self.observe(self.arguments(), runtime)
        proof = collect(self.view, self.arguments(), [first])
        self.assertEqual(proof.next_arguments['cursor'], first['result']['collection']['next_cursor'])
        self.assertEqual(proof.coverage['checked_layer_ids'], list(self.view.layer_order))
        pages, proof = self.traverse(runtime, self.arguments(), [first])
        self.assertEqual(len(pages), 2)
        self.assertFalse(pages[-1]['result']['collection']['complete'])
        self.assertTrue(proof.result['collection']['complete'])
        self.assertEqual(proof.result['layer_ids'], list(self.view.layer_order))

    def test_overlap_and_empty_intermediate_pages_keep_unique_memberships(self):
        for runtime in (GeneLayersRuntime(self.view, page_size=2, overlap=1),
                        GeneLayersRuntime(self.view, page_size=2, empty_page_at=1)):
            pages, proof = self.traverse(runtime, self.arguments())
            self.assertGreaterEqual(len(pages), 3)
            self.assertEqual(proof.result['layer_count'], 4)
            self.assertEqual(proof.result['layer_ids'], list(self.view.layer_order))

    def test_unknown_total_remains_unknown_until_exhaustion(self):
        runtime = GeneLayersRuntime(self.view, page_size=2, disclose_total=False)
        for mode, k, total, complete in (('complete', None, 4, True), ('top_k', 2, None, False),
                                       ('top_k', 4, None, False), ('top_k', 6, 4, True)):
            _, proof = self.traverse(runtime, self.arguments(mode=mode, top_k=k))
            self.assertEqual(proof.result['layer_count'], total)
            self.assertEqual(proof.result['collection']['total_count'], total)
            self.assertIs(proof.result['collection']['complete'], complete)

    def test_exhausted_known_total_equal_k_proves_full_collection(self):
        runtime = GeneLayersRuntime(self.view, page_size=2)
        _, proof = self.traverse(runtime, self.arguments(mode='top_k', top_k=4))
        self.assertEqual(proof.result['layer_count'], 4)
        self.assertTrue(proof.result['collection']['complete'])

    def test_count_equality_does_not_end_a_pending_complete_traversal(self):
        runtime = GeneLayersRuntime(self.view, page_size=2)
        first = self.observe(self.arguments(), runtime)
        cursor = first['result']['collection']['next_cursor']
        second = self.observe(self.arguments(cursor=cursor), runtime, 'ev_second')
        second['result']['collection']['next_cursor'] = 'tail_empty'
        proof = collect(self.view, self.arguments(), [first, second])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['cursor'], 'tail_empty')
        tail = deepcopy(second)
        tail['arguments']['cursor'] = 'tail_empty'
        tail['provenance']['graph'][0]['evidence_id'] = 'ev_tail'
        tail['result']['layer_ids'] = []
        tail['result']['collection'].update(returned_count=0, next_cursor=None)
        proof = collect(self.view, self.arguments(), [first, second, tail])
        self.assertTrue(proof.result['collection']['complete'])

    def test_changed_mode_or_bound_starts_a_new_selection(self):
        runtime = GeneLayersRuntime(self.view, page_size=1)
        bounded = self.observe(self.arguments(mode='top_k', top_k=2), runtime)
        proof = collect(self.view, self.arguments(), [bounded])
        self.assertEqual(proof.next_arguments, self.arguments())
        proof = collect(self.view, self.arguments(mode='top_k', top_k=3), [bounded])
        self.assertEqual(proof.next_arguments, self.arguments(mode='top_k', top_k=3))

    def test_equivalent_scope_selector_keeps_its_own_cursor_chain(self):
        runtime = GeneLayersRuntime(self.view, page_size=2)
        saved = self.observe(self.arguments(), runtime)
        target = self.arguments(layer_scope=self.subset(*self.view.layer_order))
        proof = collect(self.view, target, [saved])
        self.assertEqual(proof.next_arguments, target)

    def test_global_positive_does_not_require_a_positive_selected_layer(self):
        saved = self.observe(self.arguments(gene_id=self.b, layer_scope=self.subset(self.coex)))
        proof = collect(self.view, saved['arguments'], [saved])
        self.assertTrue(proof.result['present'])
        self.assertEqual(proof.result['layer_ids'], [])
        self.assertEqual(proof.result['layer_count'], 0)

    def test_subscope_pages_complete_before_the_remaining_scope(self):
        runtime = GeneLayersRuntime(self.view, page_size=1)
        sub = self.arguments(layer_scope=self.subset(self.ppi, self.coex))
        first = self.observe(sub, runtime)
        proof = collect(self.view, self.arguments(), [first])
        self.assertEqual(proof.next_arguments['layer_scope'], sub['layer_scope'])
        self.assertEqual(proof.next_arguments['cursor'], first['result']['collection']['next_cursor'])
        pages, proof = self.traverse(runtime, self.arguments(), [first])
        self.assertEqual(proof.result['layer_count'], 4)
        self.assertEqual(len(pages), 4)

    def test_empty_selected_family_keeps_global_presence_independent(self):
        family = {'kind': 'all', 'layer_ids': [], 'layer_family': 'bulk_pen'}
        for gene, present in ((self.a, True), (self.missing, False)):
            target = self.arguments(gene_id=gene, layer_scope=family)
            saved = self.global_observation(gene, present)
            proof = collect(self.view, target, [saved])
            self.assertIs(proof.result['present'], present)
            self.assertEqual(proof.result['layer_ids'], [])
            self.assertEqual(proof.result['layer_count'], 0)
            self.assertEqual(proof.coverage['requested_layer_ids'], [])

    def test_global_absence_finishes_membership_without_layer_calls(self):
        target = self.arguments(gene_id=self.missing)
        saved = self.global_observation(self.missing, False)
        proof = collect(self.view, target, [saved])
        self.assertFalse(proof.result['present'])
        self.assertEqual(proof.result['layer_ids'], [])
        self.assertEqual(proof.result['layer_count'], 0)
        self.assertEqual(proof.coverage['checked_layer_ids'], list(self.view.layer_order))
        self.assertIsNone(proof.next_arguments)
        positive = self.global_observation()
        proof = collect(self.view, self.arguments(), [positive])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.coverage['checked_layer_ids'], [])

    def test_directional_zero_uses_explicit_membership_not_degree(self):
        native = NeighborRuntime(self.view, index=NeighborIndex(self.view))
        arguments = {'multiplex_id': self.view.multiplex_id, 'gene_id': self.a, 'layer_id': self.cd4,
                     'direction': 'out', 'mode': 'complete', 'top_k': None, 'cursor': None}
        saved = {'operation': 'get_neighbors', 'arguments': arguments,
                 **native.get_neighbors(**arguments, evidence_id='ev_native')}
        self.assertEqual(saved['result']['neighbors'], [])
        target = self.arguments(layer_scope=self.subset(self.cd4))
        proof = collect(self.view, target, [saved])
        self.assertTrue(proof.result['present'])
        self.assertEqual(proof.result['layer_ids'], [self.cd4])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_neighbor_layer_absence_is_a_scoped_negative_only(self):
        native = NeighborRuntime(self.view, index=NeighborIndex(self.view))
        arguments = {'multiplex_id': self.view.multiplex_id, 'gene_id': self.e, 'layer_id': self.cd4,
                     'direction': 'out', 'mode': 'complete', 'top_k': None, 'cursor': None}
        saved = {'operation': 'get_neighbors', 'arguments': arguments,
                 **native.get_neighbors(**arguments, evidence_id='ev_absent')}
        target = self.arguments(gene_id=self.e, layer_scope=self.subset(self.cd4))
        proof = collect(self.view, target, [saved])
        self.assertTrue(proof.result['present'])
        self.assertEqual(proof.result['layer_ids'], [])
        self.assertEqual(proof.result['layer_count'], 0)

    def test_bounded_map_has_explicit_memberships_for_all_selected_layers(self):
        runtime = NeighborLayersRuntime(self.view, index=NeighborIndex(self.view))
        arguments = runtime.arguments(self.a, self.scope, 'incident', 'top_k', 1)
        saved = {'operation': 'get_neighbor_layers', 'arguments': arguments,
                 **runtime.get_neighbor_layers(**arguments, evidence_id='ev_map')}
        proof = collect(self.view, self.arguments(), [saved])
        self.assertEqual(proof.result['layer_ids'], list(self.view.layer_order))
        self.assertEqual(proof.result['layer_count'], 4)

    def test_wrong_gene_and_version_are_excluded(self):
        saved = self.observe(self.arguments())
        stale = deepcopy(saved)
        stale['provenance']['graph'][0]['graph_version'] = 'old_version'
        wrong = self.observe(self.arguments(gene_id=self.b), identity='ev_wrong')
        proof = collect(self.view, self.arguments(), [stale, wrong])
        self.assertEqual(proof.observations, ())
        self.assertEqual(proof.next_arguments, self.arguments())

    def test_redundant_conflicts_are_rejected_before_a_sufficient_answer(self):
        saved = self.observe(self.arguments())
        narrow = self.observe(self.arguments(layer_scope=self.subset(self.ppi)), identity='ev_narrow')
        narrow['result']['layer_ids'] = []
        narrow['result']['layer_count'] = 0
        narrow['result']['collection'].update(returned_count=0, total_count=0)
        with self.assertRaisesRegex(EvidenceError, 'contradict layer'):
            collect(self.view, self.arguments(), [saved, narrow])
        negative = self.global_observation(self.a, False)
        with self.assertRaisesRegex(EvidenceError, 'global membership'):
            collect(self.view, self.arguments(), [saved, negative])
        bounded = self.observe(self.arguments(mode='top_k', top_k=1), identity='ev_bounded')
        bounded['result']['layer_count'] = bounded['result']['collection']['total_count'] = 3
        with self.assertRaisesRegex(EvidenceError, 'total'):
            collect(self.view, self.arguments(), [saved, bounded])

    def test_bad_evidence_ids_totals_cursor_chains_and_hidden_omissions_fail(self):
        saved = self.observe(self.arguments())
        conflicting = self.observe(self.arguments(layer_scope=self.subset(self.ppi)))
        with self.assertRaisesRegex(EvidenceError, 'evidence ID'):
            collect(self.view, self.arguments(), [saved, conflicting])
        runtime = GeneLayersRuntime(self.view, page_size=2)
        first = self.observe(self.arguments(), runtime)
        second = self.observe(self.arguments(cursor=first['result']['collection']['next_cursor']), runtime, 'ev_second')
        with self.assertRaisesRegex(EvidenceError, 'initial'):
            collect(self.view, self.arguments(), [second])
        bad = deepcopy(second)
        bad['result']['collection']['next_cursor'] = bad['arguments']['cursor']
        with self.assertRaisesRegex(EvidenceError, 'loops'):
            collect(self.view, self.arguments(), [first, bad])
        bad = self.observe(self.arguments(mode='top_k', top_k=1), identity='ev_prefix')
        bad['result']['layer_ids'] = [self.coex]
        with self.assertRaisesRegex(EvidenceError, 'earlier known positive'):
            collect(self.view, self.arguments(mode='top_k', top_k=1), [saved, bad])


if __name__ == '__main__':
    unittest.main()
