"""Check visible membership, pair proofs, edge pages, and source independence."""

from copy import deepcopy
import unittest

from runtime.world_model.s2_6_corpus import _global_observation
from runtime.world_model.s2_9_evidence import collect, validate_subgraph_observation
from runtime.world_model.s2_multiplex import EvidenceError
from runtime.world_model.s2_neighbors import NeighborRuntime
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphRuntime


A, B, C, D, E, F, OUTSIDE, MISSING = [f'ENSG{number:011d}' for number in range(1, 9)]
PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'


def observation(operation, arguments, response):
    """Keep the exact acquisition beside its attributed response."""
    return {'operation': operation, 'arguments': deepcopy(arguments), **deepcopy(response)}


class SubgraphEvidenceTests(unittest.TestCase):
    """Use loops, zero weights, reciprocals, exclusions, and selected-set isolates."""

    def setUp(self):
        self.view = GraphView('unit_subgraph_evidence', '9' * 64, {
            PPI: LayerGraph(PPI, False, {(A, A): .9, (A, B): .8, (B, C): .7, (C, D): 0, (D, F): .6},
                frozenset((A, B, C, D, E, F))),
            COEX: LayerGraph(COEX, False, {(A, OUTSIDE): .2}, frozenset((A, OUTSIDE))),
            BRAIN: LayerGraph(BRAIN, True, {(A, B): .9, (B, A): .8, (A, A): .7, (C, B): 0, (C, D): .5},
                frozenset((A, B, C, D, E))),
        })
        self.serial = 0

    def args(self, genes=(A, B, C), layer=PPI, mode='complete', top_k=None):
        return SubgraphRuntime(self.view).arguments(list(genes), layer, mode, top_k)

    def subgraph(self, genes=(A, B, C), layer=PPI, *, mode='complete', top_k=None, runtime=None, cursor=None):
        runtime = runtime or SubgraphRuntime(self.view)
        arguments = {**self.args(genes, layer, mode, top_k), 'cursor': cursor}
        self.serial += 1
        return observation('induce_subgraph', arguments,
            runtime.induce_subgraph(**arguments, evidence_id=f'sub_{self.serial}'))

    def native(self, gene, layer=PPI, direction='undirected', *, mode='complete', top_k=None, runtime=None, cursor=None):
        runtime = runtime or NeighborRuntime(self.view)
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_id': gene,
            'direction': direction, 'mode': mode, 'top_k': top_k, 'cursor': cursor}
        self.serial += 1
        return observation('get_neighbors', arguments,
            runtime.get_neighbors(**arguments, evidence_id=f'native_{self.serial}'))

    def nodes(self, genes, layer=PPI):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': list(genes)}
        self.serial += 1
        return observation('get_nodes_by_layer', arguments,
            NodesByLayerRuntime(self.view).get_nodes_by_layer(**arguments, evidence_id=f'nodes_{self.serial}'))

    def test_no_evidence_requests_full_ordered_input_not_membership_batches(self):
        arguments = self.args((C, A, OUTSIDE, MISSING))
        proof = collect(self.view, arguments, [])
        self.assertEqual(proof.next_tool, 'induce_subgraph')
        self.assertEqual(proof.next_arguments, arguments)
        self.assertEqual(proof.coverage['unchecked_gene_ids'], arguments['gene_ids'])

    def test_explicit_empty_is_complete_without_provenance(self):
        proof = collect(self.view, self.args(()), [])
        self.assertEqual(proof.result['edges'], [])
        self.assertEqual(proof.result['collection']['total_count'], 0)
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})
        self.assertIsNone(proof.next_arguments)

    def test_full_subgraph_preserves_weights_loops_and_isolates(self):
        saved = self.subgraph((C, A, B, E, OUTSIDE, MISSING))
        proof = collect(self.view, self.args((C, A, B, E, OUTSIDE, MISSING)), [saved])
        self.assertEqual(proof.result, saved['result'])
        self.assertIn(E, proof.result['layer_gene_ids'])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [OUTSIDE])
        self.assertEqual(proof.result['missing_gene_ids'], [MISSING])
        self.assertEqual(proof.observations, (saved,))

    def test_complete_superset_restricts_both_endpoints_and_reorders_coverage(self):
        saved = self.subgraph((A, B, C, D, E, OUTSIDE, MISSING))
        proof = collect(self.view, self.args((D, B, OUTSIDE, B)), [saved])
        self.assertEqual(proof.result['layer_gene_ids'], [B, D])
        self.assertEqual(proof.result['edges'], [])
        self.assertEqual(proof.result['coverage']['requested_gene_ids'], [D, B, OUTSIDE])

    def test_directed_graph_preserves_native_roles_and_reciprocals(self):
        saved = self.subgraph((B, A), BRAIN)
        proof = collect(self.view, self.args((B, A), BRAIN), [saved])
        self.assertEqual(proof.result['edges'], [
            {'gene_ids': [A, A], 'weight': .7}, {'gene_ids': [A, B], 'weight': .9},
            {'gene_ids': [B, A], 'weight': .8}])

    def test_full_outgoing_lists_are_sufficient_without_incoming_lists(self):
        observations = [self.native(gene, BRAIN, 'out') for gene in (A, B, C, D)]
        proof = collect(self.view, self.args((D, A, C, B), BRAIN), observations)
        self.assertEqual(proof.result['edge_count'], 5)
        self.assertEqual(proof.result['edges'][-2]['weight'], 0)

    def test_full_incoming_lists_are_equally_sufficient(self):
        observations = [self.native(gene, BRAIN, 'in') for gene in (A, B, C, D)]
        proof = collect(self.view, self.args((A, B, C, D), BRAIN), observations)
        self.assertEqual(proof.result['edge_count'], 5)

    def test_mixed_outgoing_incoming_and_loop_proofs_are_sufficient(self):
        observations = [self.native(A, BRAIN, 'out'), self.native(A, BRAIN, 'in'), self.subgraph((B,), BRAIN)]
        proof = collect(self.view, self.args((A, B), BRAIN), observations)
        self.assertEqual(proof.result['edge_count'], 3)
        self.assertIsNone(proof.next_arguments)

    def test_full_undirected_lists_write_each_edge_once(self):
        observations = [self.native(gene) for gene in (A, B, C)]
        proof = collect(self.view, self.args(), observations)
        self.assertEqual(proof.result['edge_count'], 3)
        self.assertEqual(proof.result['edges'][0], {'gene_ids': [A, A], 'weight': .9})

    def test_unqueried_member_loop_still_requires_evidence(self):
        observations = [self.native(A), self.native(B), self.nodes((A, B, C))]
        proof = collect(self.view, self.args(), observations)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [])
        self.assertEqual(proof.next_tool, 'induce_subgraph')

    def test_selected_set_singleton_without_loop_needs_negative_incidence(self):
        proof = collect(self.view, self.args((E,)), [self.nodes((E,))])
        self.assertIsNone(proof.result)
        proof = collect(self.view, self.args((E,)), [self.native(E)])
        self.assertEqual(proof.result['edge_count'], 0)

    def test_positive_partial_self_loop_settles_singleton(self):
        saved = self.native(A, mode='top_k', top_k=1)
        self.assertFalse(saved['result']['collection']['complete'])
        proof = collect(self.view, self.args((A,)), [saved])
        self.assertEqual(proof.result['edges'], [{'gene_ids': [A, A], 'weight': .9}])
        self.assertTrue(proof.result['collection']['complete'])

    def test_global_negative_neighbor_classifies_exclusion_without_edge_predicate(self):
        proof = collect(self.view, self.args((MISSING,)), [self.native(MISSING)])
        self.assertEqual(proof.result['missing_gene_ids'], [MISSING])
        self.assertEqual(proof.result['edges'], [])

    def test_all_excluded_partition_settles_empty_edge_domain(self):
        saved = self.nodes((OUTSIDE, MISSING))
        proof = collect(self.view, self.args((OUTSIDE, MISSING)), [saved])
        self.assertEqual(proof.result['edge_count'], 0)
        self.assertEqual(proof.result['layer_gene_ids'], [])

    def test_global_positive_alone_does_not_establish_layer_membership(self):
        saved = _global_observation(self.view, [A], 'global_positive')
        proof = collect(self.view, self.args((A,)), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [A])

    def test_preview_prefix_supports_same_and_shorter_requested_preview(self):
        saved = self.subgraph(mode='top_k', top_k=2)
        proof = collect(self.view, self.args(mode='top_k', top_k=1), [saved])
        self.assertEqual(proof.result['edge_count'], 1)
        self.assertEqual(proof.result['collection']['total_count'], 3)
        self.assertFalse(proof.result['collection']['complete'])

    def test_unknown_total_preview_remains_unknown(self):
        runtime = SubgraphRuntime(self.view, disclose_total=False)
        saved = self.subgraph(mode='top_k', top_k=1, runtime=runtime)
        proof = collect(self.view, self.args(mode='top_k', top_k=1), [saved])
        self.assertIsNone(proof.result['collection']['total_count'])
        self.assertFalse(proof.result['collection']['complete'])

    def test_bounded_context_restarts_complete_selection_at_null(self):
        saved = self.subgraph(mode='top_k', top_k=1)
        proof = collect(self.view, self.args(), [saved])
        self.assertEqual(proof.next_arguments, self.args())
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [])

    def test_disjoint_singleton_subgraphs_do_not_prove_cross_edges(self):
        observations = [self.subgraph((B,)), self.subgraph((C,))]
        proof = collect(self.view, self.args((B, C)), observations)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.args((B, C)))

    def test_edge_pages_preserve_membership_and_follow_exact_cursor(self):
        runtime = SubgraphRuntime(self.view, page_size=2)
        first = self.subgraph(runtime=runtime)
        cursor = first['result']['collection']['next_cursor']
        proof = collect(self.view, self.args(), [first])
        self.assertEqual(proof.next_arguments, {**self.args(), 'cursor': cursor})
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [])
        second = self.subgraph(runtime=runtime, cursor=cursor)
        proof = collect(self.view, self.args(), [first, second])
        self.assertEqual(proof.result['edge_count'], 3)
        self.assertEqual(len(proof.provenance['graph']), 2)

    def test_last_page_without_initial_page_does_not_prove_a_complete_graph(self):
        runtime = SubgraphRuntime(self.view, page_size=2, disclose_total=False)
        first = self.subgraph(runtime=runtime)
        last = self.subgraph(runtime=runtime, cursor=first['result']['collection']['next_cursor'])
        proof = collect(self.view, self.args(), [last])
        self.assertIsNone(proof.result)
        self.assertIsNone(proof.next_arguments['cursor'])

    def test_unknown_totals_are_counted_after_complete_exhaustion(self):
        runtime = SubgraphRuntime(self.view, page_size=2, disclose_total=False)
        first = self.subgraph(runtime=runtime)
        last = self.subgraph(runtime=runtime, cursor=first['result']['collection']['next_cursor'])
        proof = collect(self.view, self.args(), [first, last])
        self.assertEqual(proof.result['collection']['total_count'], 3)

    def test_overlapping_and_empty_pages_preserve_one_copy_per_edge(self):
        for transport in ({'page_size': 2, 'overlap': 1}, {'page_size': 2, 'empty_page_at': 1}):
            runtime = SubgraphRuntime(self.view, **transport)
            observations, cursor = [], None
            while True:
                page = self.subgraph(runtime=runtime, cursor=cursor)
                observations.append(page)
                cursor = page['result']['collection']['next_cursor']
                if cursor is None:
                    break
            proof = collect(self.view, self.args(), observations)
            self.assertEqual(proof.result['edge_count'], 3)
            self.assertEqual(proof.result['collection']['total_count'], 3)

    def test_a_changed_membership_partition_on_a_later_page_is_a_fault(self):
        runtime = SubgraphRuntime(self.view, page_size=2)
        first = self.subgraph(runtime=runtime)
        second = self.subgraph(runtime=runtime, cursor=first['result']['collection']['next_cursor'])
        second['result']['layer_gene_ids'].remove(A)
        second['result']['layer_gene_count'] -= 1
        second['result']['not_in_layer_gene_ids'] = [A]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_duplicate_context_deduplicates_original_observation(self):
        saved = self.subgraph()
        proof = collect(self.view, self.args(), [saved, deepcopy(saved)])
        self.assertEqual(proof.observations, (saved,))

    def test_stale_context_does_not_supply_current_membership_or_edges(self):
        saved = self.subgraph()
        saved['provenance']['graph'][0]['graph_version'] = '0' * 64
        proof = collect(self.view, self.args(), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, ())
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [A, B, C])

    def test_gate_acquires_classification_before_edges(self):
        arguments = self.args((A, OUTSIDE))
        proof = collect(self.view, arguments, [], require_all_in_layer=True)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments, {'multiplex_id': self.view.multiplex_id,
            'layer_id': PPI, 'gene_ids': [A, OUTSIDE]})

    def test_explicit_failed_gate_preserves_partition_and_defers_without_graph(self):
        saved = self.nodes((A, OUTSIDE))
        proof = collect(self.view, self.args((A, OUTSIDE)), [saved], require_all_in_layer=True)
        self.assertEqual(proof.issue_code, 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.observations, (saved,))

    def test_passing_gate_still_needs_edge_and_loop_proofs(self):
        saved = self.nodes((A, B))
        proof = collect(self.view, self.args((A, B)), [saved], require_all_in_layer=True)
        self.assertIsNone(proof.issue_code)
        self.assertEqual(proof.next_tool, 'induce_subgraph')

    def test_conflicting_weights_memberships_totals_and_evidence_ids_are_faults(self):
        saved = self.subgraph()
        variants = []
        changed = deepcopy(saved)
        changed['result']['edges'][0]['weight'] = .1
        variants.append([saved, changed])
        changed = self.subgraph()
        changed['result']['collection']['total_count'] = 7
        changed['result']['collection']['complete'] = False
        variants.append([saved, changed])
        changed = self.nodes((A,))
        changed['result']['layer_gene_ids'] = []
        changed['result']['layer_gene_count'] = 0
        changed['result']['not_in_layer_gene_ids'] = [A]
        changed['result']['collection'].update(returned_count=0, total_count=0)
        variants.append([saved, changed])
        for observations in variants:
            with self.subTest(observations=observations), self.assertRaises(EvidenceError):
                collect(self.view, self.args(), observations)

    def test_complete_native_absence_conflicts_with_an_observed_positive(self):
        saved = self.native(A)
        saved['result']['neighbors'] = [row for row in saved['result']['neighbors'] if row['gene_id'] != B]
        saved['result']['collection'].update(returned_count=1, total_count=1)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [saved, self.subgraph()])

    def test_native_selections_must_agree_on_totals_and_ranked_prefixes(self):
        full = self.native(A)
        wrong_total = self.native(A, mode='top_k', top_k=1)
        wrong_total['result']['collection']['total_count'] = 3
        wrong_prefix = self.native(A, mode='top_k', top_k=1)
        wrong_prefix['result']['neighbors'] = [{'gene_id': B, 'weight': .8}]
        for changed in (wrong_total, wrong_prefix):
            with self.subTest(changed=changed), self.assertRaises(EvidenceError):
                collect(self.view, self.args(), [full, changed])

    def test_detached_page_cannot_contradict_an_initial_prefix_or_full_total(self):
        runtime = SubgraphRuntime(self.view, page_size=1)
        first = self.subgraph(runtime=runtime)
        second = self.subgraph(runtime=runtime, cursor=first['result']['collection']['next_cursor'])
        third = self.subgraph(runtime=runtime, cursor=second['result']['collection']['next_cursor'])
        third['result']['edges'] = [{'gene_ids': [A, A], 'weight': .9}]
        # The first page becomes a later tuple while retaining an initial cursor.
        first['result']['edges'] = [{'gene_ids': [A, B], 'weight': .8}]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, third])

    def test_metadata_validation_does_not_read_sources(self):
        class NoInventories:
            def __init__(self, view):
                self.multiplex_id, self.graph_version = view.multiplex_id, view.graph_version
                self.layers = {layer: LayerGraph(layer, value.directed, {}, frozenset()) for layer, value in view.layers.items()}
                self.layer_order = view.layer_order

            @property
            def multiplex_nodes(self):
                raise AssertionError('the planner read the source inventory')

        metadata = NoInventories(self.view)
        proof = collect(metadata, self.args(), [self.subgraph()])
        self.assertEqual(proof.result['edge_count'], 3)


if __name__ == '__main__':
    unittest.main()
