"""Check source-blind finite-gene classification, reuse, and batch failures."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.s2_6_evidence import collect
from runtime.world_model.s2_gene_layers import GeneLayersRuntime
from runtime.world_model.s2_multiplex import EvidenceError, SelectionError
from runtime.world_model.s2_neighbor_layers import NeighborLayersRuntime
from runtime.world_model.s2_neighbors import NeighborIndex, NeighborRuntime
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class NodesEvidenceTests(unittest.TestCase):
    """Use explicit fixture inventories with isolates and a target-only node."""

    def setUp(self):
        self.a, self.b, self.c, self.d, self.e, self.missing = [f'ENSG{i:011d}' for i in range(1, 7)]
        self.ppi, self.coex, self.brain, self.cd4 = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells')
        self.view = GraphView('test_nodes', 'version_nodes', {
            self.ppi: LayerGraph(self.ppi, False, {(self.a, self.b): .2}, frozenset((self.a, self.b, self.e))),
            self.coex: LayerGraph(self.coex, False, {(self.a, self.c): .3}, frozenset((self.a, self.c))),
            self.brain: LayerGraph(self.brain, True, {(self.b, self.a): .4}, frozenset((self.a, self.b))),
            self.cd4: LayerGraph(self.cd4, True, {(self.d, self.a): .5}, frozenset((self.a, self.d))),
        })
        self.runtime = NodesByLayerRuntime(self.view)
        self.scope = {'kind': 'all', 'layer_ids': [], 'layer_family': None}

    def arguments(self, genes=None, layer=None, **changes):
        return self.runtime.arguments(layer or self.brain, [self.a, self.c, self.missing] if genes is None else genes) | changes

    def observe(self, arguments, identity='ev_nodes'):
        return {'operation': 'get_nodes_by_layer', 'arguments': deepcopy(arguments),
                **self.runtime.get_nodes_by_layer(**arguments, evidence_id=identity)}

    def global_observation(self, genes, identity='ev_global'):
        return {'operation': 'check_multiplex_membership',
                'arguments': {'multiplex_id': self.view.multiplex_id, 'gene_ids': list(genes)}, 'status': 'resolved',
                'result': {'multiplex_id': self.view.multiplex_id,
                           'present_gene_ids': sorted(gene for gene in genes if gene in self.view.multiplex_nodes),
                           'absent_gene_ids': sorted(gene for gene in genes if gene not in self.view.multiplex_nodes)},
                'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                    'graph_version': self.view.graph_version, 'layer_scope': self.scope, 'evidence_id': identity}]}}

    def gene_layers(self, gene, *, mode='complete', top_k=None, scope=None, runtime=None, cursor=None, identity='ev_layers'):
        runtime = runtime or GeneLayersRuntime(self.view)
        arguments = runtime.arguments(gene, scope or self.scope, mode, top_k, cursor)
        return {'operation': 'get_gene_layers', 'arguments': arguments,
                **runtime.get_gene_layers(**arguments, evidence_id=identity)}

    def native(self, gene, layer=None, identity='ev_native'):
        layer = layer or self.cd4
        runtime = NeighborRuntime(self.view, index=NeighborIndex(self.view))
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_id': gene,
                     'direction': 'out', 'mode': 'complete', 'top_k': None, 'cursor': None}
        return {'operation': 'get_neighbors', 'arguments': arguments,
                **runtime.get_neighbors(**arguments, evidence_id=identity)}

    def test_initial_request_reads_no_source_inventory_or_adjacency(self):
        class HiddenInventory:
            def __contains__(self, value):
                raise AssertionError('the planner read a source inventory')

            def __iter__(self):
                raise AssertionError('the planner read a source inventory')

        object.__setattr__(self.view, 'multiplex_nodes', HiddenInventory())
        for layer in self.view.layers.values():
            object.__setattr__(layer, 'nodes', HiddenInventory())
            object.__setattr__(layer, 'edges', HiddenInventory())
        arguments = self.arguments()
        with patch.object(NodesByLayerRuntime, 'get_nodes_by_layer', side_effect=AssertionError('hidden tool call')):
            proof = collect(self.view, arguments, [])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, arguments)
        self.assertEqual(proof.coverage['unchecked_gene_ids'], arguments['gene_ids'])
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_empty_domain_has_no_call_or_provenance(self):
        saved = self.observe(self.arguments())
        proof = collect(self.view, self.arguments([]), [saved])
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.observations, ())
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})
        self.assertEqual(proof.result['layer_gene_count'], 0)
        self.assertEqual(proof.result['coverage']['requested_gene_ids'], [])

    def test_empty_domain_still_requires_valid_scope(self):
        with self.assertRaises(SelectionError) as caught:
            collect(self.view, self.arguments([], layer_id='TFs:unknown'), [])
        self.assertEqual(caught.exception.issue_code, 'UNKNOWN_LAYER')

    def test_missing_null_and_invalid_genes_are_diagnostics(self):
        for invalid, code in ((None, 'MISSING_GENES'), ('CDC27', 'INVALID_GENE_LIST'), ([self.a, None], 'INVALID_GENE_LIST')):
            with self.subTest(invalid=invalid):
                with self.assertRaises(SelectionError) as caught:
                    collect(self.view, self.arguments(gene_ids=invalid), [])
                self.assertEqual(caught.exception.issue_code, code)
        arguments = self.arguments()
        del arguments['gene_ids']
        with self.assertRaises(SelectionError) as caught:
            collect(self.view, arguments, [])
        self.assertEqual(caught.exception.issue_code, 'MISSING_GENES')

    def test_default_has_no_implicit_batch_cap(self):
        genes = [f'ENSG{i:011d}' for i in range(1, 2001)]
        proof = collect(self.view, self.arguments(genes), [])
        self.assertEqual(proof.next_arguments['gene_ids'], genes)

    def test_batch_limit_requires_a_positive_integer(self):
        for value in (0, -1, True, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect(self.view, self.arguments(), [], batch_limit=value)

    def test_requested_input_deduplicates_in_first_mention_order(self):
        arguments = self.arguments([self.missing, self.a, self.c, self.a, self.missing])
        proof = collect(self.view, arguments, [])
        self.assertEqual(proof.next_arguments['gene_ids'], [self.missing, self.a, self.c])
        saved = self.observe(proof.next_arguments)
        proof = collect(self.view, arguments, [saved])
        self.assertEqual(proof.result['coverage']['requested_gene_ids'], [self.missing, self.a, self.c])
        self.assertEqual(proof.result['layer_gene_ids'], [self.a])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [self.c])
        self.assertEqual(proof.result['missing_gene_ids'], [self.missing])

    def test_larger_reversed_partition_projects_counts_and_original_evidence(self):
        saved = self.observe(self.arguments([self.e, self.missing, self.c, self.b, self.a]))
        proof = collect(self.view, self.arguments([self.c, self.a, self.missing]), [saved, saved])
        self.assertEqual(proof.result['layer_gene_count'], 1)
        self.assertEqual(proof.result['collection']['total_count'], 1)
        self.assertEqual(proof.result['coverage']['checked_gene_ids'], [self.c, self.a, self.missing])
        self.assertEqual(proof.provenance, saved['provenance'])
        self.assertEqual(proof.observations, (saved,))

    def test_overlapping_batches_merge_by_gene_in_request_order(self):
        first = self.observe(self.arguments([self.a, self.c]), 'ev_first')
        second = self.observe(self.arguments([self.missing, self.a]), 'ev_second')
        proof = collect(self.view, self.arguments([self.missing, self.c, self.a]), [first, second])
        self.assertEqual(proof.result['layer_gene_count'], 1)
        self.assertEqual(proof.result['coverage']['checked_gene_ids'], [self.missing, self.c, self.a])
        self.assertEqual(proof.observations, (first, second))

    def test_first_complete_record_can_supply_all_required_facts(self):
        saved = self.observe(self.arguments(), 'ev_saved')
        redundant = self.observe(self.arguments([self.a]), 'ev_redundant')
        proof = collect(self.view, self.arguments(), [saved, redundant])
        self.assertEqual(proof.observations, (saved,))

    def test_explicit_batches_retain_every_success_and_only_fetch_pending(self):
        arguments, saved = self.arguments([self.e, self.a, self.missing, self.c, self.d]), []
        for ordinal, expected in enumerate(([self.e, self.a], [self.missing, self.c], [self.d])):
            proof = collect(self.view, arguments, saved, batch_limit=2)
            self.assertEqual(proof.next_arguments['gene_ids'], expected)
            saved.append(self.observe(proof.next_arguments, f'ev_batch_{ordinal}'))
        proof = collect(self.view, arguments, saved, batch_limit=2)
        self.assertIsNotNone(proof.result)
        self.assertEqual(proof.observations, tuple(saved))
        self.assertEqual(proof.result['layer_gene_ids'], [self.a])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [self.c, self.d, self.e])
        self.assertEqual(proof.result['missing_gene_ids'], [self.missing])

    def test_failed_batch_remains_unchecked_with_original_successful_partial(self):
        arguments = self.arguments([self.c, self.a, self.missing])
        saved = self.observe(self.arguments([self.c]), 'ev_prior_negative')
        pending = collect(self.view, arguments, [saved], batch_limit=1)
        failed = {'operation': pending.next_tool, 'arguments': pending.next_arguments, 'status': 'error',
                  'result': {'error_code': 'BACKEND_UNAVAILABLE', 'retries_remaining': 0},
                  'provenance': {'registry': [], 'graph': []}}
        proof = collect(self.view, arguments, [saved, failed], batch_limit=1)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, pending.next_arguments)
        self.assertEqual(proof.coverage['checked_gene_ids'], [self.c])
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [self.a, self.missing])
        self.assertEqual(proof.observations, (saved,))

    def test_rejected_call_adds_no_evidence_or_checked_genes(self):
        rejected = self.observe(self.arguments(layer_id='TFs:unknown'))
        self.assertEqual(rejected['status'], 'not_found')
        proof = collect(self.view, self.arguments(), [rejected])
        self.assertEqual(proof.next_arguments, self.arguments())
        self.assertEqual(proof.observations, ())
        self.assertEqual(proof.coverage['checked_gene_ids'], [])

    def test_global_positive_does_not_classify_but_negative_does(self):
        saved = self.global_observation([self.missing, self.a])
        proof = collect(self.view, self.arguments([self.a, self.missing]), [saved])
        self.assertEqual(proof.next_arguments['gene_ids'], [self.a])
        self.assertEqual(proof.coverage['checked_gene_ids'], [self.missing])
        self.assertEqual(proof.observations, (saved,))
        positive = self.global_observation([self.a])
        proof = collect(self.view, self.arguments([self.a]), [positive])
        self.assertEqual(proof.observations, (positive,))

    def test_global_negative_returns_resolved_empty_positive_collection(self):
        saved = self.global_observation([self.missing])
        proof = collect(self.view, self.arguments([self.missing]), [saved])
        self.assertIsNotNone(proof.result)
        self.assertEqual(proof.result['missing_gene_ids'], [self.missing])
        self.assertEqual(proof.result['collection']['total_count'], 0)
        self.assertTrue(proof.result['collection']['complete'])

    def test_complete_gene_layers_distinguishes_layer_and_global_negatives(self):
        for gene, category in ((self.a, 'layer_gene_ids'), (self.c, 'not_in_layer_gene_ids'), (self.missing, 'missing_gene_ids')):
            with self.subTest(gene=gene):
                saved = self.gene_layers(gene)
                proof = collect(self.view, self.arguments([gene]), [saved])
                self.assertEqual(proof.result[category], [gene])
                self.assertEqual(proof.provenance, saved['provenance'])

    def test_bounded_returned_positive_is_usable_without_exhaustive_list(self):
        saved = self.gene_layers(self.a, mode='top_k', top_k=1)
        proof = collect(self.view, self.arguments([self.a], self.ppi), [saved])
        self.assertEqual(proof.result['layer_gene_ids'], [self.a])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_bounded_omission_and_checked_coverage_do_not_prove_absence(self):
        saved = self.gene_layers(self.a, mode='top_k', top_k=1)
        proof = collect(self.view, self.arguments([self.a]), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [self.a])
        self.assertEqual(proof.observations, (saved,))
        fresh = self.observe(proof.next_arguments, 'ev_fresh')
        proof = collect(self.view, self.arguments([self.a]), [saved, fresh])
        self.assertEqual(proof.provenance, fresh['provenance'])

    def test_pending_failure_retains_global_facts_without_layer_coverage(self):
        positive = self.global_observation([self.a], 'ev_global_positive')
        bounded = self.gene_layers(self.b, mode='top_k', top_k=1, identity='ev_bounded')
        arguments = self.arguments([self.a, self.b], self.cd4)
        proof = collect(self.view, arguments, [positive, bounded])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.coverage['checked_gene_ids'], [])
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [self.a, self.b])
        self.assertEqual(proof.observations, (positive, bounded))
        failed = {'operation': 'get_nodes_by_layer', 'arguments': proof.next_arguments, 'status': 'error',
                  'result': {'error_code': 'BACKEND_UNAVAILABLE'}, 'provenance': {'registry': [], 'graph': []}}
        pending = collect(self.view, arguments, [positive, bounded, failed])
        self.assertEqual(pending.observations, (positive, bounded))
        self.assertEqual(pending.coverage, proof.coverage)
        fresh = self.observe(proof.next_arguments, 'ev_final_partition')
        complete = collect(self.view, arguments, [positive, bounded, fresh])
        self.assertEqual(complete.observations, (fresh,))
        self.assertEqual(complete.provenance, fresh['provenance'])

    def test_complete_narrow_scope_cannot_classify_an_uncovered_layer(self):
        scope = {'kind': 'layers', 'layer_ids': [self.ppi], 'layer_family': None}
        saved = self.gene_layers(self.a, scope=scope)
        proof = collect(self.view, self.arguments([self.a]), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['gene_ids'], [self.a])

    def test_complete_gene_layer_pages_prove_negative_after_exhaustion(self):
        runtime = GeneLayersRuntime(self.view, page_size=1)
        first = self.gene_layers(self.b, runtime=runtime, identity='ev_first')
        arguments = self.arguments([self.b], self.cd4)
        proof = collect(self.view, arguments, [first])
        self.assertIsNone(proof.result)
        second = self.gene_layers(self.b, runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='ev_second')
        proof = collect(self.view, arguments, [first, second])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [self.b])
        self.assertEqual(proof.observations, (first, second))

    def test_zero_outgoing_neighbor_result_proves_positive_membership(self):
        saved = self.native(self.a)
        self.assertEqual(saved['result']['neighbors'], [])
        proof = collect(self.view, self.arguments([self.a], self.cd4), [saved])
        self.assertEqual(proof.result['layer_gene_ids'], [self.a])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_native_explicit_negative_fields_separate_both_categories(self):
        for gene, category in ((self.c, 'not_in_layer_gene_ids'), (self.missing, 'missing_gene_ids')):
            with self.subTest(gene=gene):
                saved = self.native(gene)
                proof = collect(self.view, self.arguments([gene], self.cd4), [saved])
                self.assertEqual(proof.result[category], [gene])

    def test_bounded_neighbor_map_memberships_are_explicit_for_every_layer(self):
        runtime = NeighborLayersRuntime(self.view, index=NeighborIndex(self.view))
        for gene, category in ((self.a, 'layer_gene_ids'), (self.c, 'not_in_layer_gene_ids'), (self.missing, 'missing_gene_ids')):
            with self.subTest(gene=gene):
                arguments = runtime.arguments(gene, self.scope, 'incident', 'top_k', 1)
                saved = {'operation': 'get_neighbor_layers', 'arguments': arguments,
                         **runtime.get_neighbor_layers(**arguments, evidence_id='ev_map')}
                proof = collect(self.view, self.arguments([gene]), [saved])
                self.assertEqual(proof.result[category], [gene])

    def test_wrong_version_graph_and_gene_context_do_not_fill_gaps(self):
        saved = self.observe(self.arguments())
        stale = deepcopy(saved)
        stale['provenance']['graph'][0]['graph_version'] = 'old'
        wrong_graph = deepcopy(saved)
        wrong_graph['arguments']['multiplex_id'] = 'other'
        wrong_gene = self.observe(self.arguments([self.d]), 'ev_other_gene')
        proof = collect(self.view, self.arguments(), [stale, wrong_graph, wrong_gene])
        self.assertEqual(proof.next_arguments, self.arguments())
        self.assertEqual(proof.observations, ())

    def test_reuse_reads_no_source_inventory_or_adjacency(self):
        observations = [self.native(self.a, self.brain), self.gene_layers(self.c), self.global_observation([self.missing])]

        class HiddenInventory:
            def __contains__(self, value):
                raise AssertionError('the planner read a source inventory')

            def __iter__(self):
                raise AssertionError('the planner read a source inventory')

        object.__setattr__(self.view, 'multiplex_nodes', HiddenInventory())
        for layer in self.view.layers.values():
            object.__setattr__(layer, 'nodes', HiddenInventory())
            object.__setattr__(layer, 'edges', HiddenInventory())
        proof = collect(self.view, self.arguments(), observations)
        self.assertEqual(proof.result['layer_gene_ids'], [self.a])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [self.c])
        self.assertEqual(proof.result['missing_gene_ids'], [self.missing])

    def test_other_layer_result_supplies_only_global_negative_for_requested_layer(self):
        saved = self.observe(self.arguments([self.c, self.missing], self.coex))
        proof = collect(self.view, self.arguments([self.c, self.missing]), [saved])
        self.assertEqual(proof.coverage['checked_gene_ids'], [self.missing])
        self.assertEqual(proof.next_arguments['gene_ids'], [self.c])

    def test_conflicting_redundant_records_fail_before_sufficient_answer(self):
        saved = self.observe(self.arguments(), 'ev_complete')
        conflicting = self.observe(self.arguments([self.a]), 'ev_conflict')
        conflicting['result'].update(layer_gene_ids=[], layer_gene_count=0, not_in_layer_gene_ids=[self.a])
        conflicting['result']['collection'].update(returned_count=0, total_count=0)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments(), [saved, conflicting])

    def test_global_conflict_fails_even_when_layer_answer_is_complete(self):
        saved = self.observe(self.arguments([self.a]), 'ev_complete')
        negative = self.global_observation([self.a])
        negative['result'].update(present_gene_ids=[], absent_gene_ids=[self.a])
        with self.assertRaisesRegex(EvidenceError, 'global membership'):
            collect(self.view, self.arguments([self.a]), [saved, negative])

    def test_same_evidence_id_cannot_name_different_acquisitions(self):
        first = self.observe(self.arguments(), 'ev_collision')
        second = self.observe(self.arguments([self.a]), 'ev_collision')
        with self.assertRaisesRegex(EvidenceError, 'evidence ID'):
            collect(self.view, self.arguments(), [first, second])

    def test_source_scope_and_evidence_id_must_match_acquisition(self):
        saved = self.observe(self.arguments())
        for field, value in (('evidence_id', ''), ('layer_scope', self.scope)):
            malformed = deepcopy(saved)
            malformed['provenance']['graph'][0][field] = value
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                collect(self.view, self.arguments(), [malformed])

    def test_success_requires_attributed_graph_evidence(self):
        saved = self.observe(self.arguments())
        saved['provenance']['graph'] = []
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments(), [saved])

    def test_incomplete_overlapping_unsorted_or_inconsistent_nodes_result_fails(self):
        saved = self.observe(self.arguments())
        mutations = (
            lambda row: row['result']['coverage'].update(unchecked_gene_ids=[self.a]),
            lambda row: row['result']['missing_gene_ids'].append(self.a),
            lambda row: row['result'].update(layer_gene_count=2),
            lambda row: row['result']['collection'].update(complete=False),
            lambda row: row['result']['collection'].update(total_count=True),
            lambda row: row['result']['not_in_layer_gene_ids'].append(self.c),
        )
        for mutate in mutations:
            malformed = deepcopy(saved)
            mutate(malformed)
            with self.subTest(result=malformed['result']), self.assertRaises(EvidenceError):
                collect(self.view, self.arguments(), [malformed])

    def test_complete_s25_conflict_and_impossible_bounded_total_fail(self):
        saved = self.observe(self.arguments([self.a]), 'ev_nodes')
        contradictory = self.gene_layers(self.a)
        contradictory['result']['layer_ids'].remove(self.brain)
        contradictory['result']['layer_count'] = 3
        contradictory['result']['collection'].update(returned_count=3, total_count=3)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.arguments([self.a]), [saved, contradictory])
        complete = self.gene_layers(self.a, identity='ev_all')
        bounded = self.gene_layers(self.a, mode='top_k', top_k=1, identity='ev_prefix')
        bounded['result']['layer_count'] = bounded['result']['collection']['total_count'] = 3
        with self.assertRaisesRegex(EvidenceError, 'total'):
            collect(self.view, self.arguments([self.a]), [complete, bounded])

    def test_detached_s25_page_supplies_a_positive_without_an_exhaustive_negative(self):
        runtime = GeneLayersRuntime(self.view, page_size=1)
        first = self.gene_layers(self.a, runtime=runtime, mode='top_k', top_k=2, identity='ev_first')
        second = self.gene_layers(self.a, runtime=runtime, mode='top_k', top_k=2,
                                  cursor=first['result']['collection']['next_cursor'], identity='ev_second')
        proof = collect(self.view, self.arguments([self.a], self.coex), [second])
        self.assertEqual(proof.result['layer_gene_ids'], [self.a])
        self.assertEqual(proof.observations, (second,))
        proof = collect(self.view, self.arguments([self.a], self.brain), [second])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, (second,))

    def test_s25_cursor_chain_fault_is_not_hidden_by_complete_nodes_answer(self):
        saved = self.observe(self.arguments([self.a]), 'ev_nodes')
        runtime = GeneLayersRuntime(self.view, page_size=1)
        first = self.gene_layers(self.a, runtime=runtime, identity='ev_first')
        second = self.gene_layers(self.a, runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='ev_second')
        second['result']['collection']['next_cursor'] = second['arguments']['cursor']
        with self.assertRaisesRegex(EvidenceError, 'loops'):
            collect(self.view, self.arguments([self.a]), [saved, first, second])


if __name__ == '__main__':
    unittest.main()
