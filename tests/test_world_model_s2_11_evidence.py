"""Check exact paths, native proof sufficiency, and scoped graph claims."""

from copy import deepcopy
from itertools import combinations, product
import unittest

from runtime.world_model.s2_11_evidence import calibrate, collect
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY, claim_card
from runtime.world_model.s2_multiplex import EvidenceError, SelectionError
from runtime.world_model.s2_neighbors import NeighborRuntime
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_paths import PathRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphRuntime


A, B, C, D, E, F, OUTSIDE, MISSING = [f'ENSG{number:011d}' for number in range(1, 9)]
PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'


class PathEvidenceTests(unittest.TestCase):
    """Exercise membership, ties, native scope, partial evidence, and claim limits."""

    def setUp(self):
        self.view = GraphView('unit_path_evidence', 'a' * 64, {
            PPI: LayerGraph(PPI, False, {(A, A): .9, (A, B): .8, (A, C): .7,
                (B, D): .6, (C, D): 0, (D, E): .5}, frozenset((A, B, C, D, E, F))),
            COEX: LayerGraph(COEX, False, {(A, OUTSIDE): .2}, frozenset((A, OUTSIDE))),
            BRAIN: LayerGraph(BRAIN, True, {(A, B): .8, (A, C): .7, (B, D): .6,
                (C, D): 0, (C, A): .5, (D, D): .4, (E, D): .3}, frozenset((A, B, C, D, E, F))),
        })
        self.serial = 0

    def args(self, genes=(A, D), layer=PPI):
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': layer,
                'gene_ids': list(genes), 'direction': 'source_to_target' if self.view.layers[layer].directed else 'undirected'}

    def saved(self, operation, arguments, response):
        return {'operation': operation, 'arguments': deepcopy(arguments), **deepcopy(response)}

    def path(self, genes=(A, D), layer=PPI):
        arguments = self.args(genes, layer)
        self.serial += 1
        return self.saved('shortest_path', arguments,
            PathRuntime(self.view).shortest_path(**arguments, evidence_id=f'path_{self.serial}'))

    def edge(self, pair, layer=PPI):
        arguments = {key: value for key, value in self.args(pair, layer).items() if key != 'direction'}
        self.serial += 1
        return self.saved('get_edge', arguments, self.view.get_edge(**arguments, evidence_id=f'edge_{self.serial}'))

    def nodes(self, genes, layer=PPI):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': list(genes)}
        self.serial += 1
        return self.saved('get_nodes_by_layer', arguments,
            NodesByLayerRuntime(self.view).get_nodes_by_layer(**arguments, evidence_id=f'nodes_{self.serial}'))

    def subgraph(self, genes, layer=PPI):
        runtime = SubgraphRuntime(self.view)
        arguments = runtime.arguments(list(genes), layer)
        self.serial += 1
        return self.saved('induce_subgraph', arguments,
            runtime.induce_subgraph(**arguments, evidence_id=f'sub_{self.serial}'))

    def native(self, gene, layer=PPI, direction='undirected', *, runtime=None):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_id': gene,
            'direction': direction, 'mode': 'complete', 'top_k': None, 'cursor': None}
        self.serial += 1
        return self.saved('get_neighbors', arguments,
            (runtime or NeighborRuntime(self.view)).get_neighbors(**arguments, evidence_id=f'native_{self.serial}'))

    def test_unknown_pair_requests_exact_full_layer_operation(self):
        arguments = self.args((D, A))
        proof = collect(self.view, arguments, [])
        self.assertEqual(proof.next_tool, 'shortest_path')
        self.assertEqual(proof.next_arguments, arguments)
        self.assertIsNone(proof.coverage)

    def test_complete_path_uses_only_its_original_acquisition(self):
        old, matching = self.edge((A, D)), self.path()
        proof = collect(self.view, self.args(), [old, matching])
        self.assertEqual(proof.result, matching['result'])
        self.assertEqual(proof.observations, (matching,))
        self.assertEqual(proof.provenance, matching['provenance'])

    def test_checked_edge_proves_one_hop_in_requested_undirected_order(self):
        saved = self.edge((A, B))
        proof = collect(self.view, self.args((B, A)), [saved])
        self.assertEqual(proof.result['path_gene_ids'], [B, A])
        self.assertEqual(proof.result['hop_count'], 1)

    def test_zero_weight_arc_proves_one_directed_hop(self):
        proof = collect(self.view, self.args((C, D), BRAIN), [self.edge((C, D), BRAIN)])
        self.assertEqual(proof.result['path_gene_ids'], [C, D])

    def test_membership_proves_zero_hops_without_a_loop_lookup(self):
        saved = self.nodes((B,))
        proof = collect(self.view, self.args((B, B)), [saved])
        self.assertEqual(proof.result['path_gene_ids'], [B])
        self.assertEqual(proof.result['hop_count'], 0)
        self.assertEqual(proof.observations, (saved,))

    def test_same_member_keeps_zero_hops_with_a_recorded_loop(self):
        proof = collect(self.view, self.args((A, A)), [self.subgraph((A,))])
        self.assertEqual(proof.result['path_gene_ids'], [A])
        self.assertEqual(proof.result['hop_count'], 0)

    def test_same_layer_exclusion_is_a_completed_negative(self):
        proof = collect(self.view, self.args((OUTSIDE, OUTSIDE)), [self.nodes((OUTSIDE,))])
        self.assertEqual(proof.status, 'resolved')
        self.assertFalse(proof.result['path_found'])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [OUTSIDE])

    def test_same_missing_endpoint_leaves_traversal_unevaluated(self):
        proof = collect(self.view, self.args((MISSING, MISSING)), [self.nodes((MISSING,))])
        self.assertEqual(proof.status, 'not_found')
        self.assertIsNone(proof.result['path_found'])
        self.assertIsNone(proof.result['path_gene_ids'])

    def test_both_membership_categories_survive_a_blocked_pair(self):
        proof = collect(self.view, self.args((OUTSIDE, MISSING)), [self.nodes((OUTSIDE, MISSING))])
        self.assertEqual(proof.status, 'not_found')
        self.assertEqual(proof.result['missing_gene_ids'], [MISSING])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [OUTSIDE])

    def test_one_missing_endpoint_does_not_classify_the_other(self):
        proof = collect(self.view, self.args((MISSING, OUTSIDE)), [self.nodes((MISSING,))])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['gene_ids'], [MISSING, OUTSIDE])

    def test_negative_edge_does_not_prove_no_path(self):
        saved = self.edge((A, D))
        proof = collect(self.view, self.args(), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, (saved,))

    def test_positive_multihop_route_does_not_prove_minimality(self):
        saved = [self.edge((A, B)), self.edge((B, D))]
        proof = collect(self.view, self.args(), saved)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, tuple(saved))

    def test_induced_shortest_candidate_does_not_prove_the_global_tie(self):
        proof = collect(self.view, self.args(), [self.subgraph((A, C, D))])
        self.assertIsNone(proof.result)

    def test_selected_disconnection_does_not_prove_full_layer_absence(self):
        proof = collect(self.view, self.args(), [self.subgraph((A, D))])
        self.assertIsNone(proof.result)

    def test_complete_bfs_levels_prove_the_canonical_tied_witness(self):
        saved = [self.native(gene) for gene in (A, B, C)]
        proof = collect(self.view, self.args(), saved)
        self.assertEqual(proof.result['path_gene_ids'], [A, B, D])
        self.assertEqual(proof.result['hop_count'], 2)

    def test_complete_component_proves_no_route_to_an_excluded_target(self):
        saved = [self.native(gene) for gene in (A, B, C, D, E)] + [self.nodes((F,))]
        proof = collect(self.view, self.args((A, F)), saved)
        self.assertFalse(proof.result['path_found'])
        self.assertEqual(proof.observations, tuple(saved))

    def test_closed_directed_outgoing_domain_proves_no_path(self):
        saved = [self.native(D, BRAIN, 'out'), self.nodes((A,), BRAIN)]
        proof = collect(self.view, self.args((D, A), BRAIN), saved)
        self.assertFalse(proof.result['path_found'])

    def test_closed_incoming_domain_can_prove_no_directed_path(self):
        saved = [self.native(gene, BRAIN, 'in') for gene in (A, C)] + [self.nodes((B,), BRAIN)]
        proof = collect(self.view, self.args((B, A), BRAIN), saved)
        self.assertFalse(proof.result['path_found'])

    def test_an_empty_partial_page_does_not_prove_an_empty_native_domain(self):
        saved = [self.native(A, runtime=NeighborRuntime(self.view, page_size=1, empty_page_at=0)), self.nodes((F,))]
        self.assertFalse(saved[0]['result']['collection']['complete'])
        self.assertEqual(saved[0]['result']['neighbors'], [])
        proof = collect(self.view, self.args((A, F)), saved)
        self.assertIsNone(proof.result)

    def test_reverse_directed_result_does_not_answer_the_new_order(self):
        proof = collect(self.view, self.args((D, A), BRAIN), [self.path((A, D), BRAIN)])
        self.assertIsNone(proof.result)

    def test_reverse_undirected_negative_can_transfer(self):
        saved = self.path((A, F))
        proof = collect(self.view, self.args((F, A)), [saved])
        self.assertFalse(proof.result['path_found'])
        self.assertEqual(proof.result['gene_ids'], [F, A])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_reverse_undirected_multihop_witness_needs_its_own_tie_proof(self):
        proof = collect(self.view, self.args((D, A)), [self.path((A, D))])
        self.assertIsNone(proof.result)

    def test_reversed_undirected_minimum_length_supports_direct_edge_absence(self):
        saved = self.path((A, D))
        proof = calibrate(self.view, self.args((D, A)), [saved], path_requested=False,
            claim='no_edge_recorded_in_scope')
        card = proof.result['claim_cards'][0]
        self.assertTrue(card['claim_supported'])
        self.assertEqual(card['supported_claims'], ['no_edge_recorded_in_scope'])
        self.assertNotIn('path_gene_ids', proof.result)
        self.assertIsNone(proof.next_tool)
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_reversed_undirected_minimum_length_settles_exact_edge_truth(self):
        saved = self.path((A, D))
        proof = calibrate(self.view, self.args((D, A)), [saved], path_requested=False,
            claim='no_edge_recorded_in_scope', truth_scope='scope')
        self.assertTrue(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_reversed_undirected_reachability_refutes_scoped_path_absence(self):
        proof = calibrate(self.view, self.args((D, A)), [self.path((A, D))], path_requested=False,
            claim='no_path_recorded_in_scope', truth_scope='scope')
        card = proof.result['claim_cards'][0]
        self.assertFalse(card['claim_supported'])
        self.assertFalse(card['statement_true'])

    def test_reversed_claim_proof_does_not_complete_an_exact_path_request(self):
        proof = calibrate(self.view, self.args((D, A)), [self.path((A, D))],
            claim='no_edge_recorded_in_scope')
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_tool, 'shortest_path')
        self.assertEqual(proof.next_arguments['gene_ids'], [D, A])

    def test_reverse_claims_do_not_copy_a_different_canonical_witness(self):
        view = GraphView('reverse_ties', 'd' * 64, {PPI: LayerGraph(PPI, False,
            {(A, B): 1, (B, F): 1, (D, F): 1, (A, C): 1, (C, E): 1, (D, E): 1},
            frozenset((A, B, C, D, E, F)))})
        runtime = PathRuntime(view)
        source = runtime.arguments([A, D], PPI, 'undirected')
        target = runtime.arguments([D, A], PPI, 'undirected')
        saved = self.saved('shortest_path', source, runtime.shortest_path(**source, evidence_id='reverse_tie'))
        self.assertEqual(saved['result']['path_gene_ids'], [A, B, F, D])
        self.assertEqual(runtime.shortest_path(**target, evidence_id='oracle')['result']['path_gene_ids'], [D, E, C, A])
        proof = calibrate(view, target, [saved], path_requested=False, claim='no_edge_recorded_in_scope')
        self.assertTrue(proof.result['claim_cards'][0]['claim_supported'])
        self.assertIsNone(collect(view, target, [saved]).result)

    def test_directed_reverse_minimum_does_not_support_opposite_edge_absence(self):
        proof = calibrate(self.view, self.args((D, A), BRAIN), [self.path((A, D), BRAIN)],
            path_requested=False, claim='no_edge_recorded_in_scope')
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_reversed_minimum_from_another_undirected_layer_does_not_transfer(self):
        view = GraphView('wrong_layer_reverse', 'e' * 64, {
            PPI: LayerGraph(PPI, False, {(A, D): 1}, frozenset((A, D))),
            COEX: LayerGraph(COEX, False, {(A, B): 1, (B, D): 1}, frozenset((A, B, D)))})
        runtime = PathRuntime(view)
        source = runtime.arguments([A, D], COEX, 'undirected')
        target = runtime.arguments([D, A], PPI, 'undirected')
        saved = self.saved('shortest_path', source, runtime.shortest_path(**source, evidence_id='other_layer'))
        proof = calibrate(view, target, [saved], path_requested=False, claim='no_edge_recorded_in_scope')
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_stale_reversed_minimum_does_not_support_current_edge_absence(self):
        saved = self.path((A, D))
        saved['provenance']['graph'][0]['graph_version'] = 'b' * 64
        proof = calibrate(self.view, self.args((D, A)), [saved], path_requested=False,
            claim='no_edge_recorded_in_scope')
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_another_layer_cannot_supply_the_requested_shortest_distance(self):
        proof = collect(self.view, self.args(), [self.path((A, OUTSIDE), COEX)])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.provenance['graph'], [])

    def test_another_version_cannot_supply_the_current_path(self):
        saved = self.path()
        saved['provenance']['graph'][0]['graph_version'] = 'b' * 64
        proof = collect(self.view, self.args(), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, ())

    def test_conflicting_evidence_ids_raise_a_harness_fault(self):
        first, second = self.path(), self.path((A, B))
        second['provenance'] = deepcopy(first['provenance'])
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_matching_exact_responses_must_agree(self):
        first, second = self.path(), self.path()
        second['result']['path_gene_ids'] = [A, C, D]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_visible_better_tie_contradicts_an_exact_backend_witness(self):
        saved = self.path()
        saved['result']['path_gene_ids'] = [A, C, D]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [saved, self.edge((A, B)), self.edge((B, D))])

    def test_route_and_checked_negative_edge_cannot_both_hold(self):
        saved = self.edge((A, B))
        saved['result'].update(edge_exists=False, weight=None)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [self.path(), saved])

    def test_backend_path_requires_exact_hops_and_order(self):
        for field, value in (('hop_count', True), ('hop_count', 3), ('gene_ids', [D, A]), ('path_gene_ids', [D, B, A])):
            with self.subTest(field=field, value=value):
                saved = self.path()
                saved['result'][field] = value
                with self.assertRaises(EvidenceError):
                    collect(self.view, self.args(), [saved])

    def test_same_node_negative_is_not_a_valid_exact_path_result(self):
        saved = self.path((A, A))
        saved['result'].update(path_found=False, path_gene_ids=[], hop_count=None)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args((A, A)), [saved])

    def test_direction_diagnostics_do_not_guess_an_order(self):
        arguments = self.args((A, D), BRAIN)
        arguments['direction'] = None
        with self.assertRaises(SelectionError) as caught:
            collect(self.view, arguments, [])
        self.assertEqual(caught.exception.issue_code, 'AMBIGUOUS_PATH_DIRECTION')

    def test_shortest_multihop_supports_only_scoped_edge_absence(self):
        proof = calibrate(self.view, self.args(), [self.path()])
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], ['no_edge_recorded_in_scope'])

    def test_complete_negative_does_not_establish_biological_noninteraction(self):
        proof = calibrate(self.view, self.args((A, F)), [self.path((A, F))], claim='no_interaction_in_biology')
        card = proof.result['claim_cards'][0]
        self.assertEqual(card['supported_claims'], list(CLAIM_VOCABULARY[:2]))
        self.assertFalse(card['claim_supported'])
        self.assertNotIn('statement_true', card)

    def test_claim_only_negative_edge_needs_no_path_search(self):
        proof = calibrate(self.view, self.args(), [self.edge((A, D))], path_requested=False,
                          claim='no_interaction_in_biology')
        self.assertIsNone(proof.next_tool)
        self.assertNotIn('path_found', proof.result)
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], ['no_edge_recorded_in_scope'])

    def test_arbitrary_route_does_not_support_direct_edge_absence(self):
        proof = calibrate(self.view, self.args(), [self.edge((A, B)), self.edge((B, D))],
            path_requested=False, claim='no_edge_recorded_in_scope')
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_arbitrary_route_can_refute_explicit_scoped_path_absence(self):
        proof = calibrate(self.view, self.args(), [self.edge((A, B)), self.edge((B, D))],
            path_requested=False, claim='no_path_recorded_in_scope', truth_scope='scope')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])

    def test_local_negative_does_not_decide_universal_absence(self):
        saved = self.path((A, F))
        proof = calibrate(self.view, self.args((A, F)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertEqual(proof.status, 'invalid_input')
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, (saved,))
        self.assertIsNone(proof.coverage)

    def test_visible_path_refutes_universal_absence_without_proving_other_layers(self):
        proof = calibrate(self.view, self.args((A, B)), [self.edge((A, B))], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        card = proof.result['claim_cards'][0]
        self.assertFalse(card['claim_supported'])
        self.assertFalse(card['statement_true'])
        self.assertEqual(card['supported_claims'], [])

    def test_another_registered_layer_can_refute_the_universal_statement(self):
        saved = self.path((A, OUTSIDE), COEX)
        proof = calibrate(self.view, self.args((A, OUTSIDE)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        card = proof.result['claim_cards'][0]
        self.assertFalse(card['statement_true'])
        self.assertFalse(card['claim_supported'])
        self.assertEqual(card['subject']['layer_id'], PPI)
        self.assertEqual(proof.provenance, saved['provenance'])
        self.assertEqual(proof.provenance['graph'][0]['layer_scope']['layer_ids'], [COEX])

    def test_an_undirected_counterexample_can_reverse_endpoint_order(self):
        saved = self.path((OUTSIDE, A), COEX)
        proof = calibrate(self.view, self.args((A, OUTSIDE)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.observations, (saved,))

    def test_a_reverse_directed_record_does_not_refute_the_requested_order(self):
        saved = self.path((E, D), BRAIN)
        proof = calibrate(self.view, self.args((D, E)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertIsNone(proof.result)

    def test_an_arbitrary_other_layer_route_can_refute_universal_absence(self):
        saved = [self.edge((A, B), BRAIN), self.edge((B, D), BRAIN)]
        proof = calibrate(self.view, self.args(), saved, path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.observations, tuple(saved))

    def test_same_node_membership_in_another_layer_refutes_universal_path_absence(self):
        saved = self.nodes((OUTSIDE,), COEX)
        proof = calibrate(self.view, self.args((OUTSIDE, OUTSIDE)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_path_and_universal_truth_keep_both_required_acquisitions(self):
        counterexample = self.path((A, OUTSIDE), COEX)
        current = self.path((A, OUTSIDE))
        proof = calibrate(self.view, self.args((A, OUTSIDE)), [counterexample, current],
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertFalse(proof.result['path_found'])
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.observations, (counterexample, current))

    def test_other_layer_negative_is_retained_when_universal_truth_is_unsettled(self):
        saved = self.path((A, F), BRAIN)
        self.assertFalse(saved['result']['path_found'])
        self.assertEqual(saved['result']['not_in_layer_gene_ids'], [])
        proof = calibrate(self.view, self.args((A, F)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_other_layer_counterexample_is_retained_while_local_path_is_pending(self):
        saved = self.path((A, B), BRAIN)
        arguments = self.args((A, B))
        proof = calibrate(self.view, arguments, [saved], claim='absence_holds_in_other_layers',
            truth_scope='all_layers')
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_tool, 'shortest_path')
        self.assertEqual(proof.next_arguments, arguments)
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_current_and_other_layer_negatives_keep_both_original_records(self):
        foreign, current = self.path((A, F), BRAIN), self.path((A, F))
        proof = calibrate(self.view, self.args((A, F)), [foreign, current],
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertEqual(proof.observations, (foreign, current))
        self.assertEqual(proof.provenance['graph'], [*foreign['provenance']['graph'], *current['provenance']['graph']])

    def test_foreign_negative_remains_attributed_after_a_counterexample_succeeds(self):
        foreign, current = self.path((A, D), COEX), self.path((A, D))
        proof = calibrate(self.view, self.args(), [foreign, current],
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.observations, (foreign, current))
        self.assertEqual(proof.provenance['graph'], [*foreign['provenance']['graph'], *current['provenance']['graph']])

    def test_undirected_reverse_negative_is_relevant_to_universal_truth(self):
        saved = self.path((F, A), COEX)
        proof = calibrate(self.view, self.args((A, F)), [saved], path_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_stale_and_foreign_multiplex_pairs_do_not_become_universal_partials(self):
        for foreign_multiplex in (False, True):
            with self.subTest(foreign_multiplex=foreign_multiplex):
                saved = self.path((A, F), BRAIN)
                if foreign_multiplex:
                    saved['arguments']['multiplex_id'] = 'another_multiplex'
                    saved['result']['multiplex_id'] = 'another_multiplex'
                    saved['provenance']['graph'][0]['multiplex_id'] = 'another_multiplex'
                else:
                    saved['provenance']['graph'][0]['graph_version'] = 'b' * 64
                proof = calibrate(self.view, self.args((A, F)), [saved], path_requested=False,
                    claim='absence_holds_in_other_layers', truth_scope='all_layers')
                self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
                self.assertEqual(proof.observations, ())
                self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_opposite_directed_pairs_do_not_become_other_layer_partials(self):
        for genes in ((D, E), (A, F)):
            with self.subTest(genes=genes):
                saved = self.path(tuple(reversed(genes)), BRAIN)
                proof = calibrate(self.view, self.args(genes), [saved], path_requested=False,
                    claim='absence_holds_in_other_layers', truth_scope='all_layers')
                self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
                self.assertEqual(proof.observations, ())
                self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_other_layer_partials_do_not_change_an_ordinary_path_task(self):
        saved = self.path((A, B), BRAIN)
        arguments = self.args((A, B))
        ordinary = collect(self.view, arguments, [saved])
        scoped = calibrate(self.view, arguments, [saved], claim='no_path_recorded_in_scope', truth_scope='scope')
        for proof in (ordinary, scoped):
            self.assertIsNone(proof.result)
            self.assertEqual(proof.next_arguments, arguments)
            self.assertEqual(proof.observations, ())
            self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_scoped_negative_has_an_explicit_true_verdict_only_when_requested(self):
        proof = calibrate(self.view, self.args((A, F)), [self.path((A, F))],
            claim='no_path_recorded_in_scope', truth_scope='scope')
        self.assertTrue(proof.result['claim_cards'][0]['statement_true'])

    def test_zero_hops_does_not_assess_a_self_loop(self):
        proof = calibrate(self.view, self.args((B, B)), [self.nodes((B,))], claim='no_edge_recorded_in_scope')
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_separate_complete_loop_evidence_can_establish_its_absence(self):
        proof = calibrate(self.view, self.args((B, B)), [self.subgraph((B,))], claim='no_edge_recorded_in_scope')
        self.assertTrue(proof.result['claim_cards'][0]['claim_supported'])

    def test_affected_gene_cards_keep_their_own_subjects(self):
        proof = calibrate(self.view, self.args((OUTSIDE, A)), [self.path((OUTSIDE, A))], membership_cards=True)
        pair, gene = proof.result['claim_cards']
        self.assertNotIn('gene_not_node_in_layer', pair['supported_claims'])
        self.assertEqual(gene['subject']['gene_ids'], [OUTSIDE])
        self.assertEqual(gene['supported_claims'], [*CLAIM_VOCABULARY[:2], 'gene_not_node_in_layer'])

    def test_missing_endpoint_can_have_a_requested_claim_only_membership_card(self):
        proof = calibrate(self.view, self.args((MISSING, A)), [self.nodes((MISSING, A))],
            path_requested=False, membership_cards=True)
        pair, gene = proof.result['claim_cards']
        self.assertEqual(pair['supported_claims'], [])
        self.assertEqual(gene['supported_claims'], ['gene_not_node_in_multiplex'])

    def test_path_request_with_a_missing_endpoint_omits_unassessed_cards(self):
        proof = calibrate(self.view, self.args((MISSING, A)), [self.path((MISSING, A))], membership_cards=True)
        self.assertEqual(proof.status, 'not_found')
        self.assertNotIn('claim_cards', proof.result)

    def test_cards_partition_the_closed_ordered_vocabulary(self):
        card = claim_card(self.args(), (A, D), ('no_path_recorded_in_scope', 'no_edge_recorded_in_scope'))
        self.assertEqual(card['supported_claims'], list(CLAIM_VOCABULARY[:2]))
        self.assertEqual(card['unsupported_claims'], list(CLAIM_VOCABULARY[2:]))
        with self.assertRaises(ValueError):
            claim_card(self.args(), (A, D), ('invented_claim',))

    def test_one_hop_path_refutes_exact_edge_absence(self):
        proof = calibrate(self.view, self.args((A, B)), [self.path((A, B))],
            claim='no_edge_recorded_in_scope', truth_scope='scope')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])

    def test_a_separate_positive_loop_refutes_exact_loop_absence(self):
        proof = calibrate(self.view, self.args((A, A)), [self.subgraph((A,))],
            claim='no_edge_recorded_in_scope', truth_scope='scope')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])

    def test_a_graph_path_cannot_refute_an_unrelated_biological_claim(self):
        with self.assertRaises(ValueError):
            calibrate(self.view, self.args((A, B)), [self.path((A, B))],
                claim='no_interaction_in_biology', truth_scope='all_layers')

    def test_unhashable_route_values_raise_an_evidence_fault(self):
        saved = self.path()
        saved['result']['path_gene_ids'] = [A, {}, D]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [saved])

    def test_path_absence_cannot_contradict_membership_from_another_layer(self):
        saved = self.path((OUTSIDE, A))
        saved['status'] = 'not_found'
        saved['result'].update(path_found=None, path_gene_ids=None, hop_count=None,
            missing_gene_ids=[OUTSIDE], not_in_layer_gene_ids=[])
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args((OUTSIDE, A)), [self.nodes((OUTSIDE,), COEX), saved])

    def test_global_absence_keeps_its_original_membership_acquisition(self):
        arguments = {'multiplex_id': self.view.multiplex_id, 'gene_ids': [MISSING]}
        saved = self.saved('check_multiplex_membership', arguments, {
            'status': 'resolved', 'result': {'multiplex_id': self.view.multiplex_id,
                'present_gene_ids': [], 'absent_gene_ids': [MISSING]},
            'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                'graph_version': self.view.graph_version,
                'layer_scope': {'kind': 'all', 'layer_ids': [], 'layer_family': None},
                'evidence_id': 'global_missing'}]}})
        proof = collect(self.view, self.args((MISSING, MISSING)), [saved])
        self.assertEqual(proof.status, 'not_found')
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_registry_observations_cannot_give_one_id_two_current_symbols(self):
        records = [self.saved('lookup_human_gene_symbol', {'gene_symbol': symbol}, {
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': symbol},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': symbol}], 'graph': []}})
            for symbol in ('FIRST', 'SECOND')]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), records)

    def test_resolved_unique_symbol_cannot_conflict_with_an_observed_other_id(self):
        symbol = self.saved('lookup_human_gene_symbol', {'gene_symbol': 'ONE'}, {
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': 'ONE'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'symbol'}], 'graph': []}})
        by_id = self.saved('lookup_human_ensembl_gene_id', {'gene_id': B}, {
            'status': 'resolved', 'result': {'gene_id': B, 'gene_symbol': 'ONE'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'id'}], 'graph': []}})
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [symbol, by_id])

    def test_symbol_case_and_outer_whitespace_preserve_the_raw_arguments(self):
        saved = self.saved('lookup_human_gene_symbol', {'gene_symbol': '  tp53  '}, {
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': 'TP53'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'raw_symbol'}], 'graph': []}})
        original = deepcopy(saved)
        proof = collect(self.view, self.args((A, A)), [saved, self.nodes((A,))])
        self.assertEqual(proof.result['path_gene_ids'], [A])
        self.assertEqual(saved, original)

    def test_conflicting_symbol_case_variants_still_raise_an_identity_fault(self):
        records = [self.saved('lookup_human_gene_symbol', {'gene_symbol': raw}, {
            'status': 'resolved', 'result': {'gene_id': gene, 'gene_symbol': 'TP53'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': str(number)}], 'graph': []}})
            for number, (raw, gene) in enumerate((('tp53', A), (' TP53 ', B)))]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), records)

    def test_an_actual_alias_is_not_a_current_symbol_case_variant(self):
        saved = self.saved('lookup_human_gene_symbol', {'gene_symbol': 'HER2'}, {
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': 'ERBB2'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'alias'}], 'graph': []}})
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [saved])

    def test_visible_versioned_id_keeps_s1_normalization_and_a_null_symbol(self):
        saved = self.saved('lookup_human_ensembl_gene_id', {'gene_id': f' {A.lower()}.12 '}, {
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': None},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'raw_id'}], 'graph': []}})
        original = deepcopy(saved)
        proof = collect(self.view, self.args((A, A)), [saved, self.nodes((A,))])
        self.assertEqual(proof.result['hop_count'], 0)
        self.assertEqual(saved, original)

    def test_an_unreviewed_id_suffix_cannot_have_a_resolved_canonical_result(self):
        saved = self.saved('lookup_human_ensembl_gene_id', {'gene_id': A + '.alias'}, {
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': 'ONE'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'suffix'}], 'graph': []}})
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [saved])

    def test_native_incidence_proofs_are_sound_over_all_three_node_graphs(self):
        nodes, checked = (A, B, C), 0
        for directed in (False, True):
            pairs = [(left, right) for left, right in product(nodes, repeat=2) if left != right] if directed else list(combinations(nodes, 2))
            for mask in range(1 << len(pairs)):
                selected = [pair for bit, pair in enumerate(pairs) if mask & (1 << bit)]
                view = GraphView('exhaustive_paths', 'c' * 64, {
                    PPI: LayerGraph(PPI, directed, {pair: 0 for pair in selected}, frozenset(nodes))})
                args = {'multiplex_id': view.multiplex_id, 'layer_id': PPI, 'gene_ids': [A, C],
                    'direction': 'source_to_target' if directed else 'undirected'}
                member_args = {key: value for key, value in args.items() if key != 'direction'}
                member = self.saved('get_nodes_by_layer', member_args,
                    NodesByLayerRuntime(view).get_nodes_by_layer(**member_args, evidence_id='members'))
                reach = {node: [] for node in nodes}
                for left, right in selected:
                    reach[left].append(right)
                    if not directed:
                        reach[right].append(left)
                routes = [[A]]
                while routes and not any(route[-1] == C for route in routes):
                    routes = [route + [neighbor] for route in routes for neighbor in reach[route[-1]] if neighbor not in route]
                expected = min((route for route in routes if route[-1] == C), default=[])
                for incoming in ((False, True) if directed else (False,)):
                    saved = []
                    for gene in nodes:
                        native_args = {'multiplex_id': view.multiplex_id, 'layer_id': PPI, 'gene_id': gene,
                            'direction': 'in' if incoming else 'out' if directed else 'undirected',
                            'mode': 'complete', 'top_k': None, 'cursor': None}
                        saved.append(self.saved('get_neighbors', native_args,
                            NeighborRuntime(view).get_neighbors(**native_args, evidence_id='native_' + gene)))
                    for subset in range(1 << len(nodes)):
                        observations = [member, *(record for bit, record in enumerate(saved) if subset & (1 << bit))]
                        proof = collect(view, args, observations)
                        if proof.result is not None:
                            self.assertEqual(proof.result['path_gene_ids'], expected,
                                (directed, mask, incoming, subset))
                        checked += 1
        self.assertEqual(checked, 1088)

    def test_planner_reads_no_inventory_or_native_graph(self):
        original = self.view
        class LayerMetadata:
            def __init__(self, layer):
                self.layer_id, self.directed = layer.layer_id, layer.directed
            @property
            def nodes(self):
                raise AssertionError('source node inventory access')
            @property
            def edges(self):
                raise AssertionError('source edge inventory access')
        class MetadataOnly:
            multiplex_id, graph_version = original.multiplex_id, original.graph_version
            layers = {name: LayerMetadata(layer) for name, layer in original.layers.items()}
            layer_order = tuple(layers)
            @property
            def multiplex_nodes(self):
                raise AssertionError('source multiplex inventory access')
        saved = [self.native(gene) for gene in (A, B, C)]
        proof = collect(MetadataOnly(), self.args(), saved)
        self.assertEqual(proof.result['path_gene_ids'], [A, B, D])


if __name__ == '__main__':
    unittest.main()
