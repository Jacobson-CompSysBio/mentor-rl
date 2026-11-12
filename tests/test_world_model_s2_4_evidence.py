"""Check visible-only map proofs, scoped projection, and native partial coverage."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.s2_4_evidence import collect
from runtime.world_model.s2_multiplex import EvidenceError
from runtime.world_model.s2_neighbor_layers import NeighborLayersRuntime
from runtime.world_model.s2_neighbors import NeighborIndex, NeighborRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class S2MapEvidenceTests(unittest.TestCase):
    """Use asymmetric edges so incomplete direction evidence cannot pass."""

    def setUp(self):
        self.a, self.b, self.c, self.d, self.e, self.missing = [f'ENSG{i:011d}' for i in range(1, 7)]
        self.ppi, self.tf = 'HumanNetV3:string_ppi', 'TFs:brain'
        specs = {self.ppi: (False, {(self.a, self.b): 0.2, (self.a, self.c): 0.8, (self.c, self.d): 0.3}),
                 self.tf: (True, {(self.b, self.a): 0.4, (self.a, self.b): 0.6,
                                  (self.a, self.a): 1, (self.a, self.d): 0, (self.e, self.d): 0.7})}
        self.view = GraphView('test_maps', 'version_maps', {
            name: LayerGraph(name, directed, edges, frozenset(g for pair in edges for g in pair))
            for name, (directed, edges) in specs.items()})
        self.index = NeighborIndex(self.view)
        self.runtime = NeighborLayersRuntime(self.view, index=self.index)
        self.native = NeighborRuntime(self.view, index=self.index)
        self.scope = {'kind': 'all', 'layer_ids': [], 'layer_family': None}

    def arguments(self, **changes):
        return self.runtime.arguments(self.a, self.scope, 'incident') | changes

    def observe(self, runtime, arguments, identity='ev_test', operation='get_neighbor_layers'):
        return {'operation': operation, 'arguments': deepcopy(arguments),
                **getattr(runtime, operation)(**arguments, evidence_id=identity)}

    def native_arguments(self, layer, direction, gene=None):
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_id': gene or self.a,
                'direction': direction, 'mode': 'complete', 'top_k': None, 'cursor': None}

    def traverse(self, runtime, arguments, observations=None):
        observations = list(observations or [])
        for i in range(20):
            proof = collect(self.view, arguments, observations)
            if proof.result is not None:
                return observations, proof
            server = runtime if proof.next_tool == 'get_neighbor_layers' else self.native
            observations.append(self.observe(server, proof.next_arguments, f'ev_page_{i}', proof.next_tool))
        self.fail('fixture traversal did not finish')

    def test_initial_request_fetches_map_without_source_reads(self):
        with patch.object(self.index, 'neighbors', side_effect=AssertionError('hidden adjacency')):
            proof = collect(self.view, self.arguments(), [])
        self.assertEqual(proof.next_tool, 'get_neighbor_layers')
        self.assertEqual(proof.next_arguments, self.arguments())
        self.assertEqual(proof.coverage['unchecked_layer_ids'], [self.ppi, self.tf])

    def test_saved_complete_map_and_shorter_bound_need_no_call(self):
        saved = self.observe(self.runtime, self.arguments())
        with patch.object(self.index, 'neighbors', side_effect=AssertionError('hidden adjacency')):
            proof = collect(self.view, self.arguments(mode='top_k', top_k=2), [saved, saved])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(len(proof.result['neighbors']), 2)
        self.assertEqual(proof.result['unique_neighbor_count'], 4)
        self.assertEqual(proof.observations, (saved,))
        bounded = self.observe(self.runtime, self.arguments(mode='top_k', top_k=3))
        proof = collect(self.view, self.arguments(mode='top_k', top_k=2), [bounded])
        self.assertEqual(len(proof.result['neighbors']), 2)
        self.assertFalse(proof.result['collection']['complete'])

    def test_complete_scope_projection_recomputes_rows_and_support_counts(self):
        saved = self.observe(self.runtime, self.arguments())
        target = self.arguments(layer_scope={'kind': 'layers', 'layer_ids': [self.ppi], 'layer_family': None})
        proof = collect(self.view, target, [saved])
        self.assertEqual([row['gene_id'] for row in proof.result['neighbors']], [self.b, self.c])
        self.assertEqual(proof.result['unique_neighbor_count'], 2)
        self.assertTrue(all(row['supporting_layer_count'] == 1 for row in proof.result['neighbors']))
        self.assertEqual(proof.provenance, saved['provenance'])
        bounded = self.observe(self.runtime, self.arguments(mode='top_k', top_k=2))
        pending = collect(self.view, target, [bounded])
        self.assertEqual(pending.next_arguments, target)
        self.assertIsNone(pending.result)

    def test_unknown_total_and_overlapping_complete_pages_accumulate(self):
        paged = NeighborLayersRuntime(self.view, page_size=2, overlap=1, disclose_total=False, empty_page_at=1, index=self.index)
        observations, proof = self.traverse(paged, self.arguments())
        self.assertEqual([len(o['result']['neighbors']) for o in observations], [2, 0, 2, 2])
        self.assertEqual(proof.result['collection']['total_count'], 4)
        self.assertTrue(proof.result['collection']['complete'])
        self.assertEqual(len(proof.provenance['graph']), 4)
        self.assertTrue(all(o['result']['unique_neighbor_count'] is None for o in observations))

    def test_complete_request_follows_cursor_after_reaching_disclosed_total(self):
        runtime = NeighborLayersRuntime(self.view, page_size=2, index=self.index)
        first = self.observe(runtime, self.arguments(), 'ev_first')
        second_args = self.arguments(cursor=first['result']['collection']['next_cursor'])
        second = self.observe(runtime, second_args, 'ev_second')
        # The transport retains a final empty continuation after all rows.
        second['result']['collection']['next_cursor'] = 'opaque_empty_tail'
        prefix = [first, second]
        pending = collect(self.view, self.arguments(), prefix)
        self.assertIsNone(pending.result)
        self.assertEqual(pending.next_arguments, self.arguments(cursor='opaque_empty_tail'))
        tail = deepcopy(second)
        tail['arguments']['cursor'] = 'opaque_empty_tail'
        tail['provenance']['graph'][0]['evidence_id'] = 'ev_tail'
        tail['result']['neighbors'] = []
        tail['result']['collection'].update(returned_count=0, next_cursor=None)
        proof = collect(self.view, self.arguments(), [*prefix, tail])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['unique_neighbor_count'], 4)
        self.assertTrue(proof.result['collection']['complete'])
        self.assertEqual(len(proof.observations), 3)

    def test_native_complete_collection_follows_its_empty_tail(self):
        runtime = NeighborRuntime(self.view, page_size=1, index=self.index)
        target = self.native_arguments(self.ppi, 'undirected')
        first = self.observe(runtime, target, 'ev_native_first', 'get_neighbors')
        second_args = {**target, 'cursor': first['result']['collection']['next_cursor']}
        second = self.observe(runtime, second_args, 'ev_native_second', 'get_neighbors')
        second['result']['collection']['next_cursor'] = 'opaque_native_tail'
        pending = collect(self.view, self.arguments(), [first, second])
        self.assertIsNone(pending.result)
        self.assertEqual(pending.next_tool, 'get_neighbors')
        self.assertEqual(pending.next_arguments, {**target, 'cursor': 'opaque_native_tail'})
        self.assertIn({'layer_id': self.ppi, 'direction': 'undirected'}, pending.coverage['unchecked_inputs'])
        tail = deepcopy(second)
        tail['arguments']['cursor'] = 'opaque_native_tail'
        tail['provenance']['graph'][0]['evidence_id'] = 'ev_native_tail'
        tail['result']['neighbors'] = []
        tail['result']['collection'].update(returned_count=0, next_cursor=None)
        pending = collect(self.view, self.arguments(), [first, second, tail])
        self.assertEqual(pending.next_tool, 'get_neighbor_layers')
        self.assertEqual(pending.next_arguments['layer_scope']['layer_ids'], [self.tf])
        self.assertEqual(pending.coverage['checked_layer_ids'], [self.ppi])

    def test_exhausted_bounded_equality_uses_only_a_disclosed_total(self):
        target = self.arguments(mode='top_k', top_k=4)
        for disclose in (True, False):
            runtime = NeighborLayersRuntime(self.view, page_size=1, disclose_total=disclose, index=self.index)
            observations, proof = self.traverse(runtime, target)
            self.assertTrue(all(not observation['result']['collection']['complete'] for observation in observations))
            self.assertEqual(proof.result['collection']['complete'], disclose)
            self.assertEqual(proof.result['unique_neighbor_count'], 4 if disclose else None)
        native = NeighborRuntime(self.view, page_size=1, index=self.index)
        native_args = {**self.native_arguments(self.ppi, 'undirected'), 'mode': 'top_k', 'top_k': 2}
        first = self.observe(native, native_args, 'ev_bounded_first', 'get_neighbors')
        second = self.observe(native, {**native_args, 'cursor': first['result']['collection']['next_cursor']},
                              'ev_bounded_last', 'get_neighbors')
        pending = collect(self.view, self.arguments(), [first, second])
        self.assertEqual(pending.next_tool, 'get_neighbor_layers')
        self.assertEqual(pending.next_arguments['layer_scope']['layer_ids'], [self.tf])

    def test_unknown_total_bound_does_not_invent_exhaustion(self):
        runtime = NeighborLayersRuntime(self.view, page_size=1, disclose_total=False, index=self.index)
        for k, total, full in ((2, None, False), (4, None, False), (5, 4, True)):
            _, proof = self.traverse(runtime, self.arguments(mode='top_k', top_k=k))
            self.assertEqual(proof.result['unique_neighbor_count'], total)
            self.assertEqual(proof.result['collection']['complete'], full)

    def test_changed_selection_restarts_at_null_and_preserves_partial(self):
        runtime = NeighborLayersRuntime(self.view, page_size=1, index=self.index)
        saved = self.observe(runtime, self.arguments(mode='top_k', top_k=2))
        proof = collect(self.view, self.arguments(), [saved])
        self.assertEqual(proof.next_arguments, self.arguments())
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.coverage['checked_layer_ids'], [self.ppi, self.tf])
        same = collect(self.view, self.arguments(mode='top_k', top_k=2), [saved])
        self.assertEqual(same.next_arguments['cursor'], saved['result']['collection']['next_cursor'])

    def test_equivalent_scope_labels_never_share_selection_bound_cursors(self):
        runtime = NeighborLayersRuntime(self.view, page_size=1, index=self.index)
        saved = self.observe(runtime, self.arguments(), 'ev_all')
        target = self.arguments(layer_scope={'kind': 'layers', 'layer_ids': [self.ppi, self.tf], 'layer_family': None})
        pending = collect(self.view, target, [saved])
        self.assertEqual(pending.next_arguments, target)
        explicit = self.observe(runtime, target, 'ev_explicit')
        pending = collect(self.view, target, [saved, explicit])
        self.assertEqual(pending.next_arguments['cursor'], explicit['result']['collection']['next_cursor'])
        complete = self.observe(self.runtime, self.arguments(), 'ev_complete_all')
        complete_explicit = self.observe(self.runtime, target, 'ev_complete_explicit')
        proof = collect(self.view, target, [complete, complete_explicit])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(len(proof.observations), 2)

    def test_native_lists_regroup_full_support_and_deduplicate_self_loop(self):
        saved = [self.observe(self.native, self.native_arguments(layer, direction), f'ev_{i}', 'get_neighbors')
                 for i, (layer, direction) in enumerate(((self.ppi, 'undirected'), (self.tf, 'in'), (self.tf, 'out')))]
        proof = collect(self.view, self.arguments(), saved)
        expected = self.runtime.get_neighbor_layers(**self.arguments(), evidence_id='ev_expected')['result']
        self.assertEqual(proof.result, expected)
        loop = proof.result['neighbors'][0]
        self.assertEqual(loop['gene_id'], self.a)
        self.assertEqual(loop['supporting_layer_count'], 1)
        self.assertEqual(len(loop['supporting_layers'][0]['edges']), 1)
        reciprocal = next(row for row in proof.result['neighbors'] if row['gene_id'] == self.b)
        self.assertEqual(reciprocal['supporting_layer_count'], 2)
        self.assertEqual([edge['gene_ids'] for edge in reciprocal['supporting_layers'][1]['edges']],
                         [[self.b, self.a], [self.a, self.b]])

    def test_complete_native_layer_fetches_only_the_missing_whole_layer(self):
        saved = self.observe(self.native, self.native_arguments(self.ppi, 'undirected'), 'ev_saved_ppi', 'get_neighbors')
        pending = collect(self.view, self.arguments(), [saved])
        expected = self.arguments(layer_scope={'kind': 'layers', 'layer_ids': [self.tf], 'layer_family': None})
        self.assertEqual(pending.next_tool, 'get_neighbor_layers')
        self.assertEqual(pending.next_arguments, expected)
        self.assertEqual(pending.coverage, {'operation': 'layer_neighbor_maps',
            'requested_layer_ids': [self.ppi, self.tf], 'checked_layer_ids': [self.ppi], 'unchecked_layer_ids': [self.tf]})
        added = self.observe(self.runtime, pending.next_arguments, 'ev_added_tf')
        proof = collect(self.view, self.arguments(), [saved, added])
        expected_result = self.runtime.get_neighbor_layers(**self.arguments(), evidence_id='ev_expected')['result']
        self.assertEqual(proof.result, expected_result)
        self.assertEqual(proof.observations, (saved, added))
        self.assertEqual([record['evidence_id'] for record in proof.provenance['graph']], ['ev_saved_ppi', 'ev_added_tf'])

    def test_complete_subset_map_supplies_layers_without_repeated_acquisition(self):
        subset = self.arguments(layer_scope={'kind': 'layers', 'layer_ids': [self.ppi], 'layer_family': None})
        saved = self.observe(self.runtime, subset, 'ev_saved_subset')
        pending = collect(self.view, self.arguments(), [saved])
        self.assertEqual(pending.next_arguments['layer_scope']['layer_ids'], [self.tf])
        added = self.observe(self.runtime, pending.next_arguments, 'ev_added_tf')
        proof = collect(self.view, self.arguments(), [saved, added])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['unique_neighbor_count'], 4)
        self.assertEqual(proof.observations, (saved, added))
        self.assertEqual(proof.provenance['graph'][0]['layer_scope'], subset['layer_scope'])
        incoming = self.observe(self.native, self.native_arguments(self.tf, 'in'), 'ev_in', 'get_neighbors')
        pending = collect(self.view, self.arguments(), [saved, incoming])
        self.assertEqual(pending.next_tool, 'get_neighbors')
        self.assertEqual(pending.next_arguments, self.native_arguments(self.tf, 'out'))
        self.assertEqual(pending.coverage['checked_inputs'], [
            {'layer_id': self.ppi, 'direction': 'undirected'}, {'layer_id': self.tf, 'direction': 'in'}])

    def test_paged_missing_submap_preserves_native_context_and_map_coverage(self):
        saved = self.observe(self.native, self.native_arguments(self.ppi, 'undirected'), 'ev_saved_ppi', 'get_neighbors')
        runtime = NeighborLayersRuntime(self.view, page_size=1, disclose_total=False, index=self.index)
        observations = [saved]
        pending = collect(self.view, self.arguments(), observations)
        first = self.observe(runtime, pending.next_arguments, 'ev_first_tf')
        observations.append(first)
        pending = collect(self.view, self.arguments(), observations)
        self.assertEqual(pending.next_tool, 'get_neighbor_layers')
        self.assertEqual(pending.next_arguments['layer_scope']['layer_ids'], [self.tf])
        self.assertEqual(pending.next_arguments['cursor'], first['result']['collection']['next_cursor'])
        self.assertEqual(pending.coverage['operation'], 'layer_neighbor_maps')
        self.assertEqual(pending.coverage['checked_layer_ids'], [self.ppi, self.tf])
        self.assertEqual(pending.observations, (saved, first))
        observations, proof = self.traverse(runtime, self.arguments(), observations)
        self.assertEqual(proof.result['unique_neighbor_count'], 4)
        self.assertTrue(proof.result['collection']['complete'])
        self.assertEqual(proof.observations, tuple(observations))
        self.assertTrue(all(observation['arguments'].get('layer_scope', {}).get('layer_ids') == [self.tf]
                            for observation in observations[1:]))

    def test_sufficient_map_still_rejects_conflicting_subset_or_native_evidence(self):
        full = self.observe(self.runtime, self.arguments(), 'ev_full')
        subset_args = self.arguments(layer_scope={'kind': 'layers', 'layer_ids': [self.ppi], 'layer_family': None})
        subset = self.observe(self.runtime, subset_args, 'ev_subset')
        subset['result']['neighbors'][0]['supporting_layers'][0]['edges'][0]['weight'] += 0.1
        with self.assertRaisesRegex(EvidenceError, 'complete maps disagree'):
            collect(self.view, self.arguments(), [full, subset])
        native = self.observe(self.native, self.native_arguments(self.ppi, 'undirected'), 'ev_native', 'get_neighbors')
        native['result']['neighbors'][0]['weight'] += 0.1
        with self.assertRaisesRegex(EvidenceError, 'native collections contradict'):
            collect(self.view, self.arguments(), [full, native])

    def test_native_partial_requires_exact_missing_direction_and_coverage(self):
        saved = [self.observe(self.native, self.native_arguments(self.ppi, 'undirected'), 'ev_u', 'get_neighbors'),
                 self.observe(self.native, self.native_arguments(self.tf, 'in'), 'ev_i', 'get_neighbors')]
        proof = collect(self.view, self.arguments(), saved)
        self.assertEqual(proof.next_tool, 'get_neighbors')
        self.assertEqual(proof.next_arguments, self.native_arguments(self.tf, 'out'))
        self.assertEqual(proof.coverage, {'operation': 'neighbor_collections', 'requested_inputs': [
            {'layer_id': self.ppi, 'direction': 'undirected'}, {'layer_id': self.tf, 'direction': 'in'},
            {'layer_id': self.tf, 'direction': 'out'}], 'checked_inputs': [
            {'layer_id': self.ppi, 'direction': 'undirected'}, {'layer_id': self.tf, 'direction': 'in'}],
            'unchecked_inputs': [{'layer_id': self.tf, 'direction': 'out'}]})

    def test_native_layer_absence_supplies_both_empty_directions(self):
        args = self.arguments(gene_id=self.e)
        saved = [self.observe(self.native, self.native_arguments(self.ppi, 'undirected', self.e), 'ev_abs', 'get_neighbors'),
                 self.observe(self.native, self.native_arguments(self.tf, 'in', self.e), 'ev_i', 'get_neighbors')]
        proof = collect(self.view, args, saved)
        self.assertEqual(proof.next_arguments, self.native_arguments(self.tf, 'out', self.e))
        # Query c belongs to PPI but not TF; one TF observation supplies both sides.
        args = self.arguments(gene_id=self.c)
        saved = [self.observe(self.native, self.native_arguments(self.ppi, 'undirected', self.c), 'ev_u', 'get_neighbors'),
                 self.observe(self.native, self.native_arguments(self.tf, 'in', self.c), 'ev_abs', 'get_neighbors')]
        proof = collect(self.view, args, saved)
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['layers_without_gene'], [self.tf])
        self.assertEqual(len(proof.observations), 2)

    def test_global_absence_remains_unavailable(self):
        target = self.arguments(gene_id=self.missing)
        saved = self.observe(self.runtime, target)
        proof = collect(self.view, target, [saved])
        self.assertIsNone(proof.result['neighbors'])
        self.assertIsNone(proof.result['collection'])
        self.assertEqual(proof.result['coverage']['unchecked_layer_ids'], [self.ppi, self.tf])
        native = self.observe(self.native, self.native_arguments(self.ppi, 'undirected', self.missing), operation='get_neighbors')
        proof = collect(self.view, target, [native])
        self.assertIsNone(proof.result['neighbors'])
        self.assertIsNone(proof.next_arguments)

    def test_stale_and_other_gene_context_do_not_supply_the_map(self):
        saved = self.observe(self.runtime, self.arguments())
        saved['provenance']['graph'][0]['graph_version'] = 'stale'
        self.assertEqual(collect(self.view, self.arguments(), [saved]).next_arguments, self.arguments())
        other = self.observe(self.runtime, self.arguments(gene_id=self.b))
        self.assertEqual(collect(self.view, self.arguments(), [other]).next_arguments, self.arguments())

    def test_conflicts_reject_evidence_ids_totals_support_and_cursor_chains(self):
        saved = self.observe(self.runtime, self.arguments())
        other = self.observe(self.runtime, self.arguments(mode='top_k', top_k=2))
        with self.assertRaisesRegex(EvidenceError, 'evidence ID'):
            collect(self.view, self.arguments(), [saved, other])
        other['provenance']['graph'][0]['evidence_id'] = 'ev_other'
        other['result']['unique_neighbor_count'] = other['result']['collection']['total_count'] = 5
        with self.assertRaisesRegex(EvidenceError, 'totals'):
            collect(self.view, self.arguments(), [saved, other])
        runtime = NeighborLayersRuntime(self.view, page_size=2, overlap=1, index=self.index)
        pages, _ = self.traverse(runtime, self.arguments())
        bad = deepcopy(pages)
        bad[1]['result']['neighbors'][0]['supporting_layers'][0]['edges'][0]['weight'] += 0.1
        with self.assertRaisesRegex(EvidenceError, 'conflicting support'):
            collect(self.view, self.arguments(), bad)
        with self.assertRaisesRegex(EvidenceError, 'initial'):
            collect(self.view, self.arguments(), pages[1:])
        bad = deepcopy(pages)
        bad[-1]['result']['collection']['next_cursor'] = bad[1]['arguments']['cursor']
        with self.assertRaisesRegex(EvidenceError, 'loops'):
            collect(self.view, self.arguments(), bad)


if __name__ == '__main__':
    unittest.main()
