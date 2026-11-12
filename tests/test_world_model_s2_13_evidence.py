"""Check family capability and pair support without private source graph access."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.contract import load_contract
from runtime.world_model.s2_13_evidence import (
    CLAIM_VOCABULARY, FAMILIES, FAMILY_CARDS, collect, family_result, pair_subject,
)
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY as GRAPH_CLAIMS
from runtime.world_model.s2_multiplex import EvidenceError, SelectionError
from runtime.world_model.s2_neighbors import NeighborRuntime
from runtime.world_model.s2_paths import PathRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphRuntime


A, B, C, D, MISSING = [f'ENSG{number:011d}' for number in range(1, 6)]
PPI, COEX, TF, CD4, BULK, SC, OTHER = (
    'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells',
    'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte', 'HumanNetV3:cocitation')


class HiddenLayer:
    """Expose only the qualified layer metadata used to interpret an observation."""

    def __init__(self, layer):
        self.layer_id, self.directed = layer.layer_id, layer.directed

    @property
    def nodes(self):
        raise AssertionError('hidden source inventory access')

    @property
    def edges(self):
        raise AssertionError('hidden source adjacency access')


class MetadataOnly:
    """Reject private inventory and adjacency reads in every planner test."""

    def __init__(self, view):
        self.multiplex_id, self.graph_version = view.multiplex_id, view.graph_version
        self.layers = {key: HiddenLayer(layer) for key, layer in view.layers.items()}
        self.layer_order = tuple(self.layers)

    @property
    def multiplex_nodes(self):
        raise AssertionError('hidden multiplex inventory access')


class LayerEvidenceTests(unittest.TestCase):
    """Keep positive record meaning distinct from general capability and truth."""

    def setUp(self):
        rows = {PPI: (False, {(A, B): 0, (B, C): .8}),
            COEX: (False, {(A, C): .7, (A, D): .6}),
            TF: (True, {(A, B): .6, (B, A): .4}),
            CD4: (True, {(A, C): 1}),
            BULK: (False, {(A, C): .5, (B, C): .9}),
            SC: (False, {(B, C): .7})}
        layers = {layer: LayerGraph(layer, directed, edges,
            frozenset(gene for pair in edges for gene in pair))
            for layer, (directed, edges) in rows.items()}
        self.source = GraphView('unit_layer_meaning', 'f' * 64, layers)
        self.view, self.serial = MetadataOnly(self.source), 0

    def args(self, pair=(A, B), layer=PPI):
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': list(pair)}

    def edge(self, pair=(A, B), layer=PPI, source=None):
        graph = source or self.source
        arguments = {'multiplex_id': graph.multiplex_id, 'layer_id': layer, 'gene_ids': list(pair)}
        self.serial += 1
        response = graph.get_edge(**arguments, evidence_id=f'edge_{self.serial}')
        return {'operation': 'get_edge', 'arguments': arguments, **response}

    def identity(self, gene=A, symbol='GENE_A', raw=None, operation='lookup_human_gene_symbol'):
        self.serial += 1
        field = 'gene_symbol' if operation.endswith('symbol') else 'gene_id'
        query = symbol if field == 'gene_symbol' else gene + '.4'
        return {'operation': operation, 'arguments': {field: raw if raw is not None else query},
            'status': 'resolved', 'result': {'gene_id': gene, 'gene_symbol': symbol},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116',
                'evidence_id': f'identity_{self.serial}'}], 'graph': []}}

    def path(self, pair=(A, B), layer=PPI):
        self.serial += 1
        arguments = {**self.args(pair, layer), 'direction': 'source_to_target'
            if self.source.layers[layer].directed else 'undirected'}
        return {'operation': 'shortest_path', 'arguments': arguments,
            **PathRuntime(self.source).shortest_path(**arguments, evidence_id=f'path_{self.serial}')}

    def assess(self, observations, pair=(A, B), layer=PPI, **kwargs):
        return collect(self.view, self.args(pair, layer), observations,
            acquisition_requested=False, **kwargs)

    def test_family_constants_match_the_unchanged_s1_contract(self):
        contract = load_contract()
        self.assertEqual(CLAIM_VOCABULARY, tuple(contract['vocabularies']['layer_claims']))
        self.assertEqual(FAMILIES, ('ppi', 'coexpression', 'tf_target', 'bulk_pen', 'scpen', 'other'))
        self.assertEqual(len(CLAIM_VOCABULARY), 8)
        self.assertEqual(set(FAMILY_CARDS), set(FAMILIES))

    def test_all_six_family_results_have_exact_shapes_and_closed_partitions(self):
        expected = {'ppi': ['physical_interaction_recorded'], 'coexpression': ['global_coexpression'],
            'tf_target': ['regulatory_link_recorded'], 'bulk_pen': ['context_predictive_expression'],
            'scpen': ['context_predictive_expression'], 'other': []}
        for family, supported in expected.items():
            with self.subTest(family=family):
                result = family_result(family)
                self.assertEqual(set(result), {'layer_family', 'evidence_type', 'claim_cards'})
                self.assertEqual(result['evidence_type'], supported[0] if supported else None)
                card = result['claim_cards'][0]
                self.assertEqual(card['subject'], {'kind': 'family', 'layer_family': family})
                self.assertEqual(card['supported_claims'], supported)
                self.assertEqual(card['unsupported_claims'], [name for name in CLAIM_VOCABULARY if name not in supported])
                self.assertEqual(set(card), {'subject', 'claim_vocabulary', 'supported_claims', 'unsupported_claims'})

    def test_generic_tf_family_does_not_identify_recorded_roles(self):
        card = family_result('tf_target', claim='regulation_direction')['claim_cards'][0]
        self.assertFalse(card['claim_supported'])
        self.assertIn('regulation_direction', card['unsupported_claims'])

    def test_family_other_needs_no_registered_layer_or_observation(self):
        self.assertNotIn(OTHER, self.view.layers)
        card = family_result('other', claim='cofunctional_association')['claim_cards'][0]
        self.assertEqual(card['supported_claims'], [])
        self.assertFalse(card['claim_supported'])

    def test_unknown_families_and_invented_claims_are_rejected(self):
        for family in ('PPI', 'activation', None):
            with self.assertRaises(ValueError):
                family_result(family)
        for claim in ('activation', 'repression', 'probability', 'signed_causal_confidence'):
            with self.assertRaises(ValueError):
                family_result('ppi', claim=claim)
            with self.assertRaises(ValueError):
                self.assess([], claim=claim)

    def test_family_results_do_not_share_mutable_lists(self):
        first = family_result('ppi')
        first['claim_cards'][0]['supported_claims'].clear()
        self.assertEqual(family_result('ppi')['claim_cards'][0]['supported_claims'], ['physical_interaction_recorded'])

    def test_required_missing_observation_requests_only_the_three_ordinary_arguments(self):
        with patch.object(GraphView, 'get_edge', side_effect=AssertionError('hidden tool execution')):
            proof = collect(self.view, self.args((B, A)), [])
        self.assertEqual(proof.next_arguments, self.args((B, A)))
        self.assertEqual(proof.next_tool, 'get_edge')
        self.assertIsNone(proof.result)
        self.assertIsNone(proof.coverage)
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_family_tag_does_not_supply_a_pair_observation(self):
        proof = self.assess([], claim='physical_interaction_recorded')
        self.assertEqual(set(proof.result), {'claim_cards'})
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])
        self.assertIsNone(proof.next_tool)
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_each_qualified_positive_record_has_only_its_native_family_meaning(self):
        cases = [(PPI, (A, B), ['physical_interaction_recorded']),
            (COEX, (A, C), ['global_coexpression']),
            (TF, (A, B), ['regulatory_link_recorded', 'regulation_direction']),
            (BULK, (A, C), ['context_predictive_expression']),
            (SC, (B, C), ['context_predictive_expression'])]
        for layer, pair, supported in cases:
            with self.subTest(layer=layer):
                proof = self.assess([self.edge(pair, layer)], pair, layer)
                card = proof.result['claim_cards'][0]
                self.assertEqual(card['supported_claims'], supported)
                self.assertEqual(card['unsupported_claims'], [name for name in CLAIM_VOCABULARY if name not in supported])
                self.assertNotIn('statement_true', card)

    def test_zero_weight_is_a_positive_record_and_is_retained_exactly(self):
        saved = self.edge()
        proof = collect(self.view, self.args((B, A)), [saved])
        self.assertIs(proof.result['edge_exists'], True)
        self.assertEqual(proof.result['weight'], 0)
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], ['physical_interaction_recorded'])

    def test_matching_negative_completes_acquisition_with_no_positive_biological_claim(self):
        saved = self.edge((A, C))
        proof = collect(self.view, self.args((A, C)), [saved], claim='physical_interaction_recorded')
        self.assertEqual(proof.status, 'resolved')
        self.assertFalse(proof.result['edge_exists'])
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])
        self.assertIsNone(proof.next_tool)
        self.assertNotIn('statement_true', proof.result['claim_cards'][0])

    def test_layer_exclusion_is_checked_negative_not_missing_multiplex(self):
        proof = collect(self.view, self.args((A, D)), [self.edge((A, D))])
        self.assertEqual(proof.status, 'resolved')
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [D])
        self.assertEqual(proof.result['missing_gene_ids'], [])
        self.assertFalse(proof.result['edge_exists'])
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])

    def test_required_multiplex_missing_result_retains_the_ordinary_shape_without_cards(self):
        saved = self.edge((A, MISSING))
        proof = collect(self.view, self.args((A, MISSING)), [saved])
        self.assertEqual(proof.status, 'not_found')
        self.assertEqual(proof.result, saved['result'])
        self.assertNotIn('claim_cards', proof.result)
        self.assertIsNone(proof.result['edge_exists'])
        self.assertIsNone(proof.next_tool)

    def test_supplied_missing_result_can_complete_an_evidence_only_assessment(self):
        saved = self.edge((A, MISSING))
        proof = self.assess([saved], (A, MISSING), claim='physical_interaction_recorded')
        self.assertEqual(proof.status, 'resolved')
        self.assertEqual(set(proof.result), {'claim_cards'})
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
        self.assertEqual(proof.observations, (saved,))
        self.assertIsNone(proof.next_tool)

    def test_undirected_subject_is_canonical_but_requested_tool_order_is_preserved(self):
        subject = pair_subject(self.view, self.args((B, A)))
        self.assertEqual(subject['gene_ids'], [A, B])
        self.assertFalse(subject['directed'])
        self.assertEqual(collect(self.view, self.args((B, A)), []).next_arguments['gene_ids'], [B, A])

    def test_forward_record_does_not_settle_required_reverse_edge(self):
        saved = self.edge((A, C), CD4)
        proof = collect(self.view, self.args((C, A), CD4), [saved])
        self.assertEqual(proof.next_arguments, self.args((C, A), CD4))
        self.assertEqual(proof.observations, ())
        self.assertIsNone(proof.result)

    def test_evidence_only_reverse_keeps_the_assessed_forward_record_without_a_call(self):
        saved = self.edge((A, C), CD4)
        proof = self.assess([saved], (C, A), CD4, claim='regulation_direction')
        self.assertEqual(proof.observations, (saved,))
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.result['claim_cards'][0]['subject']['gene_ids'], [C, A])
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])

    def test_reciprocal_records_have_distinct_roles_and_no_causal_feedback_claim(self):
        records = [self.edge((A, B), TF), self.edge((B, A), TF)]
        proof = self.assess(records, (A, B), TF, subjects=[self.args((A, B), TF), self.args((B, A), TF)], claim='causal_effect')
        self.assertEqual(len(proof.result['claim_cards']), 2)
        self.assertEqual([card['subject']['gene_ids'] for card in proof.result['claim_cards']], [[A, B], [B, A]])
        self.assertTrue(all(card['supported_claims'] == ['regulatory_link_recorded', 'regulation_direction']
                            and card['claim_supported'] is False for card in proof.result['claim_cards']))

    def test_undirected_tf_metadata_cannot_supply_regulator_roles(self):
        alternate = GraphView('qualified_undirected_tf', 'e' * 64,
            {TF: LayerGraph(TF, False, {(A, B): .6}, frozenset((A, B)))})
        saved = self.edge((B, A), TF, alternate)
        args = {'multiplex_id': alternate.multiplex_id, 'layer_id': TF, 'gene_ids': [B, A]}
        proof = collect(MetadataOnly(alternate), args, [saved], acquisition_requested=False, claim='regulation_direction')
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], ['regulatory_link_recorded'])
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_current_representation_mismatch_is_a_harness_fault(self):
        saved = self.edge((A, B), TF)
        saved['result']['directed'] = False
        with self.assertRaises(EvidenceError):
            self.assess([saved], (A, B), TF)

    def test_another_pair_record_does_not_support_the_target_pair(self):
        saved = self.edge()
        proof = self.assess([saved], (A, C), claim='physical_interaction_recorded')
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
        self.assertEqual(proof.observations, (saved,))
        self.assertNotIn('edge_exists', proof.result)

    def test_two_edge_route_does_not_create_direct_pair_support_or_a_negative(self):
        records = [self.edge((A, B)), self.edge((B, C))]
        proof = self.assess(records, (A, C), claim='physical_interaction_recorded')
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
        self.assertEqual(proof.observations, tuple(records))
        self.assertNotIn('edge_exists', proof.result)
        required = collect(self.view, self.args((A, C)), records)
        self.assertEqual(required.next_arguments, self.args((A, C)))

    def test_wrong_layer_record_does_not_settle_required_target_layer(self):
        saved = self.edge((A, C))
        proof = collect(self.view, self.args((A, C), COEX), [saved])
        self.assertEqual(proof.next_arguments, self.args((A, C), COEX))
        self.assertEqual(proof.observations, ())

    def test_stale_and_different_view_records_are_assessed_but_not_transferred(self):
        for different in (False, True):
            saved = self.edge()
            saved['provenance']['graph'][0]['graph_version'] = '0' * 64
            if different:
                saved['arguments']['multiplex_id'] = saved['result']['multiplex_id'] = 'old_four_file_view'
                saved['provenance']['graph'][0]['multiplex_id'] = 'old_four_file_view'
            proof = self.assess([saved])
            self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
            self.assertEqual(proof.observations, (saved,))
            needed = collect(self.view, self.args(), [saved])
            self.assertEqual(needed.next_arguments, self.args())
            self.assertEqual(needed.observations, ())

    def test_cross_context_cards_keep_requested_order_and_separate_meanings(self):
        bulk, single = self.edge((A, C), BULK), self.edge((A, C), SC)
        proof = collect(self.view, self.args((A, C), SC), [bulk, single],
            subjects=[self.args((A, C), BULK), self.args((A, C), SC)], claim='context_predictive_expression')
        self.assertEqual([card['subject']['layer_id'] for card in proof.result['claim_cards']], [BULK, SC])
        self.assertEqual([card['claim_supported'] for card in proof.result['claim_cards']], [True, False])
        self.assertEqual(proof.observations, (bulk, single))
        self.assertFalse(proof.result['edge_exists'])

    def test_two_positive_contexts_still_do_not_support_unexamined_contexts(self):
        records = [self.edge((B, C), BULK), self.edge((B, C), SC)]
        proof = self.assess(records, (B, C), BULK,
            subjects=[self.args((B, C), BULK), self.args((B, C), SC)], claim='holds_in_other_contexts')
        for card in proof.result['claim_cards']:
            self.assertEqual(card['supported_claims'], ['context_predictive_expression'])
            self.assertFalse(card['claim_supported'])

    def test_context_strength_and_high_weights_do_not_promote_causation(self):
        saved = self.edge((B, C), BULK)
        saved['result']['weight'] = 1e30
        proof = self.assess([saved], (B, C), BULK, claim='causal_effect')
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])
        self.assertEqual(proof.observations[0]['result']['weight'], 1e30)

    def test_biological_truth_remains_unknown_and_preserves_usable_evidence(self):
        saved = self.edge((A, C), BULK)
        proof = self.assess([saved], (A, C), BULK, claim='causal_effect', truth_kind='biological')
        self.assertEqual((proof.status, proof.issue_code), ('invalid_input', 'UNSUPPORTED_TRUTH_SCOPE'))
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, (saved,))
        self.assertIsNone(proof.next_tool)

    def test_exact_graph_truth_uses_its_separate_eleven_concept_schema(self):
        saved = self.edge()
        proof = self.assess([saved], claim='physical_interaction_recorded', truth_kind='graph')
        biological, graph = proof.result['claim_cards']
        self.assertEqual(biological['claim_vocabulary'], 's1_layer_v1')
        self.assertTrue(biological['claim_supported'])
        self.assertNotIn('statement_true', biological)
        self.assertEqual(graph['claim_vocabulary'], 's2_graph_observations_v1')
        self.assertEqual(graph['subject'], {**self.args(), 'direction': 'undirected'})
        self.assertFalse(graph['claim_supported'])
        self.assertFalse(graph['statement_true'])
        self.assertEqual(graph['unsupported_claims'], list(GRAPH_CLAIMS))

    def test_checked_edge_negative_can_settle_graph_truth_without_biological_truth(self):
        proof = self.assess([self.edge((A, C))], (A, C), truth_kind='graph')
        biological, graph = proof.result['claim_cards']
        self.assertEqual(biological['supported_claims'], [])
        self.assertEqual(graph['supported_claims'], ['no_edge_recorded_in_scope'])
        self.assertTrue(graph['statement_true'])

    def test_unknown_graph_truth_scope_does_not_acquire_to_strengthen_supplied_evidence(self):
        proof = self.assess([], truth_kind='graph')
        self.assertEqual((proof.status, proof.issue_code), ('invalid_input', 'UNSUPPORTED_TRUTH_SCOPE'))
        self.assertIsNone(proof.next_arguments)
        self.assertIsNone(proof.result)

    def test_duplicate_observations_and_undirected_subjects_keep_first_use_order(self):
        first, second = self.edge(), self.edge()
        proof = self.assess([first, deepcopy(first), second], subjects=[self.args((B, A)), self.args()])
        self.assertEqual(len(proof.result['claim_cards']), 1)
        self.assertEqual(proof.observations, (first, second))
        self.assertEqual(proof.provenance['graph'], [first['provenance']['graph'][0], second['provenance']['graph'][0]])

    def test_conflicting_values_and_reused_evidence_ids_are_harness_faults(self):
        first, second = self.edge(), self.edge()
        second['result']['weight'] = .2
        with self.assertRaises(EvidenceError):
            self.assess([first, second])
        second = self.edge((B, C))
        second['provenance']['graph'][0]['evidence_id'] = first['provenance']['graph'][0]['evidence_id']
        with self.assertRaises(EvidenceError):
            self.assess([first, second])

    def test_positive_and_negative_records_for_the_same_pair_conflict(self):
        first, second = self.edge(), self.edge()
        second['result'].update(edge_exists=False, weight=None)
        with self.assertRaises(EvidenceError):
            self.assess([first, second])

    def test_original_registry_records_and_optional_null_symbols_are_retained(self):
        records = [self.identity(raw=' gene_a '), self.identity(B, None, operation='lookup_human_ensembl_gene_id'), self.edge()]
        proof = self.assess(records)
        self.assertEqual(proof.observations, tuple(records))
        self.assertEqual(len(proof.provenance['registry']), 2)
        self.assertIsNone(proof.observations[1]['result']['gene_symbol'])

    def test_contradictory_current_symbol_mappings_are_rejected(self):
        first, second = self.identity(), self.identity(B)
        with self.assertRaises(EvidenceError):
            self.assess([first, second])

    def test_explicit_native_neighbors_and_induced_edges_support_their_exact_pairs(self):
        native_args = {'multiplex_id': self.source.multiplex_id, 'layer_id': CD4,
            'gene_id': A, 'direction': 'out', 'mode': 'complete', 'top_k': None, 'cursor': None}
        response = NeighborRuntime(self.source).get_neighbors(**native_args, evidence_id='native_positive')
        native = {'operation': 'get_neighbors', 'arguments': native_args, **response}
        card = self.assess([native], (A, C), CD4).result['claim_cards'][0]
        self.assertEqual(card['supported_claims'], ['regulatory_link_recorded', 'regulation_direction'])
        runtime = SubgraphRuntime(self.source)
        arguments = runtime.arguments([A, B], PPI)
        induced = {'operation': 'induce_subgraph', 'arguments': arguments,
            **runtime.induce_subgraph(**arguments, evidence_id='induced_positive')}
        self.assertEqual(self.assess([induced]).result['claim_cards'][0]['supported_claims'], ['physical_interaction_recorded'])

    def test_one_hop_path_proves_pair_meaning_but_not_the_unreturned_edge_weight(self):
        saved = self.path()
        proof = self.assess([saved], claim='physical_interaction_recorded')
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], ['physical_interaction_recorded'])
        self.assertTrue(proof.result['claim_cards'][0]['claim_supported'])
        self.assertEqual(set(proof.result), {'claim_cards'})
        required = collect(self.view, self.args(), [saved])
        self.assertIsNone(required.result)
        self.assertEqual(required.next_arguments, self.args())

    def test_longer_route_supports_its_consecutive_pairs_but_not_its_endpoint_pair(self):
        saved = self.path((A, C))
        self.assertEqual(saved['result']['hop_count'], 2)
        first = self.assess([saved], (A, B)).result['claim_cards'][0]
        second = self.assess([saved], (B, C)).result['claim_cards'][0]
        endpoints = self.assess([saved], (A, C)).result['claim_cards'][0]
        self.assertEqual(first['supported_claims'], ['physical_interaction_recorded'])
        self.assertEqual(second['supported_claims'], ['physical_interaction_recorded'])
        self.assertEqual(endpoints['supported_claims'], [])

    def test_directed_path_arc_supports_only_the_recorded_orientation(self):
        saved = self.path((A, C), CD4)
        forward = self.assess([saved], (A, C), CD4).result['claim_cards'][0]
        reverse = self.assess([saved], (C, A), CD4).result['claim_cards'][0]
        self.assertEqual(forward['supported_claims'], ['regulatory_link_recorded', 'regulation_direction'])
        self.assertEqual(reverse['supported_claims'], [])

    def test_failed_attempts_and_invalid_original_provenance_supply_no_evidence(self):
        saved = self.edge()
        saved['status'] = 'error'
        with self.assertRaises(EvidenceError):
            self.assess([saved])
        saved = self.edge()
        saved['provenance']['graph'][0]['multiplex_id'] = 'invented'
        with self.assertRaises(EvidenceError):
            self.assess([saved])

    def test_invalid_pair_scope_direction_and_task_selectors_do_not_guess(self):
        for args in (self.args((A, A)), self.args((A, MISSING)) | {'gene_ids': [A]},
                     self.args() | {'layer_id': 'HumanNetV3:unregistered'}):
            with self.assertRaises(SelectionError):
                collect(self.view, args, [])
        with self.assertRaises(SelectionError):
            self.assess([], graph_direction='source_to_target')
        with self.assertRaises(ValueError):
            self.assess([], truth_kind='biological')
        with self.assertRaises(ValueError):
            self.assess([], truth_kind='false_when_unsupported')
        with self.assertRaises(ValueError):
            self.assess([], subjects={'layer_id': PPI})


if __name__ == '__main__':
    unittest.main()
