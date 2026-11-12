"""Check singleton membership, evidence reuse, exact truth, and explicit gates."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from runtime.world_model.s2_12_evidence import collect
from runtime.world_model.s2_gene_layers import GeneLayersRuntime
from runtime.world_model.s2_graph_claims import CLAIM_VOCABULARY
from runtime.world_model.s2_multiplex import EvidenceError, SelectionError
from runtime.world_model.s2_neighbors import NeighborRuntime
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphRuntime


A, B, C, TARGET, ISOLATE, MISSING = [f'ENSG{number:011d}' for number in range(1, 7)]
PPI, COEX, TF, BULK, SC = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain',
    'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte')


class HiddenLayer:
    """Expose layer metadata and reject all hidden graph access."""

    def __init__(self, layer):
        self.layer_id, self.directed = layer.layer_id, layer.directed

    @property
    def nodes(self):
        raise AssertionError('hidden source inventory access')

    @property
    def edges(self):
        raise AssertionError('hidden source adjacency access')


class MetadataOnly:
    """Keep every proof test independent of source inventory access."""

    def __init__(self, view):
        self.multiplex_id, self.graph_version = view.multiplex_id, view.graph_version
        self.layer_order = view.layer_order
        self.layers = {key: HiddenLayer(value) for key, value in view.layers.items()}

    @property
    def multiplex_nodes(self):
        raise AssertionError('hidden multiplex inventory access')


class SingletonEvidenceTests(unittest.TestCase):
    """Separate full membership facts from bounded displays and biological claims."""

    def setUp(self):
        self.source = GraphView('unit_singleton', 'f' * 64, {
            PPI: LayerGraph(PPI, False, {(A, B): .8}, frozenset((A, B, ISOLATE))),
            COEX: LayerGraph(COEX, False, {(A, C): .7}, frozenset((A, C))),
            TF: LayerGraph(TF, True, {(A, TARGET): .6}, frozenset((A, TARGET))),
            BULK: LayerGraph(BULK, False, {(B, C): .5}, frozenset((B, C))),
            SC: LayerGraph(SC, False, {(A, C): .4}, frozenset((A, C))),
        })
        self.view = MetadataOnly(self.source)
        self.serial = 0

    def args(self, gene=A, layer=PPI):
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': [gene]}

    def saved(self, operation, arguments, response):
        return {'operation': operation, 'arguments': deepcopy(arguments), **deepcopy(response)}

    def nodes(self, gene=A, layer=PPI, genes=None):
        arguments = self.args(gene, layer)
        if genes is not None:
            arguments['gene_ids'] = list(genes)
        self.serial += 1
        return self.saved('get_nodes_by_layer', arguments,
            NodesByLayerRuntime(self.source).get_nodes_by_layer(**arguments, evidence_id=f'nodes_{self.serial}'))

    def layers(self, gene=A, family=None, mode='complete', top_k=None):
        arguments = GeneLayersRuntime(self.source).arguments(gene,
            {'kind': 'all', 'layer_ids': [], 'layer_family': family}, mode, top_k, None)
        self.serial += 1
        return self.saved('get_gene_layers', arguments,
            GeneLayersRuntime(self.source).get_gene_layers(**arguments, evidence_id=f'layers_{self.serial}'))

    def edge(self, pair=(A, B), layer=PPI):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': list(pair)}
        self.serial += 1
        return self.saved('get_edge', arguments, self.source.get_edge(**arguments, evidence_id=f'edge_{self.serial}'))

    def native(self, gene=TARGET, layer=TF, direction='out'):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_id': gene,
            'direction': direction, 'mode': 'complete', 'top_k': None, 'cursor': None}
        self.serial += 1
        return self.saved('get_neighbors', arguments,
            NeighborRuntime(self.source).get_neighbors(**arguments, evidence_id=f'native_{self.serial}'))

    def subgraph(self, gene=ISOLATE, layer=PPI):
        runtime = SubgraphRuntime(self.source)
        arguments = runtime.arguments([gene], layer)
        self.serial += 1
        return self.saved('induce_subgraph', arguments,
            runtime.induce_subgraph(**arguments, evidence_id=f'subgraph_{self.serial}'))

    def test_pending_classification_requests_only_the_unchanged_three_arguments(self):
        with patch.object(NodesByLayerRuntime, 'get_nodes_by_layer', side_effect=AssertionError('hidden tool execution')):
            proof = collect(self.view, self.args(), [])
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments, self.args())
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [A])
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_all_three_categories_resolve_with_exact_singleton_counts(self):
        for gene, category, supported in ((A, 'layer_gene_ids', []), (C, 'not_in_layer_gene_ids',
                [*CLAIM_VOCABULARY[:2], 'gene_not_node_in_layer']), (MISSING, 'missing_gene_ids', ['gene_not_node_in_multiplex'])):
            with self.subTest(category=category):
                saved = self.nodes(gene)
                proof = collect(self.view, self.args(gene), [saved], claim='gene_not_expressed_in_context')
                self.assertEqual(proof.status, 'resolved')
                self.assertIsNone(proof.next_tool)
                ordinary = {key: value for key, value in proof.result.items() if key != 'claim_cards'}
                self.assertEqual(ordinary, saved['result'])
                self.assertEqual(proof.result[category], [gene])
                self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], supported)
                self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])

    def test_cards_partition_the_shared_vocabulary_without_direction_or_unrequested_fields(self):
        card = collect(self.view, self.args(), [self.nodes()]).result['claim_cards'][0]
        self.assertEqual(card['subject'], {'gene_id': A, 'multiplex_id': self.view.multiplex_id, 'layer_id': PPI})
        self.assertEqual(card['unsupported_claims'], list(CLAIM_VOCABULARY))
        self.assertEqual(set(card), {'subject', 'claim_vocabulary', 'supported_claims', 'unsupported_claims'})

    def test_complete_superset_restricts_and_recalculates_coverage(self):
        saved = self.nodes(genes=(B, A, C, MISSING))
        proof = collect(self.view, self.args(C), [saved])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [C])
        self.assertEqual(proof.result['layer_gene_count'], 0)
        self.assertEqual(proof.result['coverage']['requested_gene_ids'], [C])
        self.assertEqual(proof.observations, (saved,))
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_complete_layer_list_proves_absence_without_a_new_call(self):
        saved = self.layers(A)
        proof = collect(self.view, self.args(A, BULK), [saved])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [A])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_matching_single_layer_family_is_complete(self):
        saved = self.layers(A, 'bulk_pen')
        proof = collect(self.view, self.args(A, BULK), [saved])
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [A])

    def test_another_family_global_positive_leaves_the_target_unknown(self):
        proof = collect(self.view, self.args(A, BULK), [self.layers(A, 'scpen')])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.args(A, BULK))

    def test_another_family_global_negative_settles_the_target(self):
        proof = collect(self.view, self.args(MISSING, BULK), [self.layers(MISSING, 'scpen')])
        self.assertEqual(proof.result['missing_gene_ids'], [MISSING])

    def test_top_one_omission_does_not_use_checked_coverage_as_absence(self):
        saved = self.layers(A, mode='top_k', top_k=1)
        self.assertEqual(saved['result']['coverage']['unchecked_layer_ids'], [])
        self.assertIsNone(collect(self.view, self.args(A, SC), [saved]).result)
        proof = collect(self.view, self.args(A, SC), [saved], classification_requested=False,
                        claim='gene_not_node_in_layer')
        self.assertIsNone(proof.next_tool)
        self.assertEqual(set(proof.result), {'gene_id', 'multiplex_id', 'layer_id', 'claim_cards'})
        self.assertFalse(proof.result['claim_cards'][0]['claim_supported'])
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])
        self.assertEqual(proof.provenance, saved['provenance'])

    def test_top_one_explicit_positive_proves_membership(self):
        saved = self.layers(A, mode='top_k', top_k=1)
        self.assertEqual(collect(self.view, self.args(), [saved]).result['layer_gene_ids'], [A])

    def test_positive_edge_and_native_neighbor_endpoint_prove_membership(self):
        for saved, gene in ((self.edge(), B), (self.native(A), TARGET)):
            layer = saved['arguments']['layer_id']
            with self.subTest(operation=saved['operation']):
                proof = collect(self.view, self.args(gene, layer), [saved])
                self.assertEqual(proof.result['layer_gene_ids'], [gene])
                self.assertEqual(proof.observations, (saved,))

    def test_target_only_node_with_no_outgoing_edges_remains_a_member(self):
        saved = self.native()
        self.assertEqual(saved['result']['neighbors'], [])
        proof = collect(self.view, self.args(TARGET, TF), [saved])
        self.assertEqual(proof.result['layer_gene_ids'], [TARGET])
        self.assertEqual(proof.result['claim_cards'][0]['supported_claims'], [])

    def test_edgeless_selected_member_remains_a_member(self):
        saved = self.subgraph()
        self.assertEqual(saved['result']['edges'], [])
        self.assertEqual(collect(self.view, self.args(ISOLATE), [saved]).result['layer_gene_ids'], [ISOLATE])

    def test_stale_version_does_not_supply_current_membership(self):
        saved = self.nodes(MISSING)
        saved['arguments']['gene_ids'] = [A]
        for key in ('requested_gene_ids', 'checked_gene_ids'):
            saved['result']['coverage'][key] = [A]
        saved['result']['missing_gene_ids'] = [A]
        saved['provenance']['graph'][0]['graph_version'] = 'old_version'
        self.assertIsNone(collect(self.view, self.args(), [saved]).result)

    def test_wrong_gene_multiplex_and_other_layer_do_not_transfer_membership(self):
        wrong_gene, wrong_multiplex, other_layer = self.nodes(B), self.nodes(A), self.nodes(A, SC)
        wrong_multiplex['provenance']['graph'][0]['multiplex_id'] = 'other_multiplex'
        for saved in (wrong_gene, wrong_multiplex, other_layer):
            with self.subTest(saved=saved):
                self.assertIsNone(collect(self.view, self.args(), [saved]).result)

    def test_complete_target_proof_omits_an_earlier_insufficient_other_layer_record(self):
        other, target = self.nodes(A, SC), self.nodes(A)
        proof = collect(self.view, self.args(), [other, target])
        self.assertEqual(proof.observations, (target,))

    def test_repeated_identical_evidence_is_deduplicated_and_first_provider_wins(self):
        first, second = self.nodes(), self.nodes()
        proof = collect(self.view, self.args(), [first, deepcopy(first), second])
        self.assertEqual(proof.observations, (first,))
        self.assertEqual(proof.provenance, first['provenance'])

    def test_conflicting_category_and_reused_identity_are_harness_faults(self):
        first, second = self.nodes(), self.nodes()
        second['result']['layer_gene_ids'] = []
        second['result']['not_in_layer_gene_ids'] = [A]
        second['result']['layer_gene_count'] = 0
        second['result']['collection'].update(returned_count=0, total_count=0)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])
        second = self.nodes(B)
        second['provenance']['graph'][0]['evidence_id'] = first['provenance']['graph'][0]['evidence_id']
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_exact_category_and_global_absence_truth_follow_distinct_partitions(self):
        for gene, claim, scope, verdict in ((C, 'gene_not_node_in_layer', 'category', True),
                (A, 'gene_not_node_in_layer', 'category', False),
                (MISSING, 'gene_not_node_in_multiplex', 'multiplex', True),
                (A, 'gene_not_node_in_multiplex', 'multiplex', False)):
            with self.subTest(gene=gene, scope=scope):
                proof = collect(self.view, self.args(gene), [self.nodes(gene)], claim=claim, truth_scope=scope)
                self.assertIs(proof.result['claim_cards'][0]['statement_true'], verdict)

    def test_literal_layer_nonmembership_is_not_false_for_a_globally_missing_gene(self):
        proof = collect(self.view, self.args(MISSING), [self.nodes(MISSING)],
            classification_requested=False, claim='absence_holds_in_other_layers',
            truth_scope='layers', truth_layer_ids=[PPI])
        self.assertTrue(proof.result['claim_cards'][0]['statement_true'])

    def test_single_family_negative_settles_its_exact_universal_truth(self):
        proof = collect(self.view, self.args(A, BULK), [self.nodes(A, BULK)],
            classification_requested=False, claim='absence_holds_in_other_layers',
            truth_scope='family', truth_family='bulk_pen')
        self.assertTrue(proof.result['claim_cards'][0]['statement_true'])

    def test_positive_multiplex_union_fact_refutes_all_layer_absence(self):
        saved = self.nodes(A, BULK)
        proof = collect(self.view, self.args(A, BULK), [saved], classification_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='all_layers')
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])

    def test_global_absence_entails_absence_in_all_registered_layers(self):
        proof = collect(self.view, self.args(MISSING), [self.nodes(MISSING)],
            classification_requested=False, claim='gene_not_node_in_multiplex', truth_scope='all_layers')
        self.assertTrue(proof.result['claim_cards'][0]['statement_true'])

    def test_foreign_layer_positive_refutes_wider_truth_without_a_target_classification(self):
        saved = self.nodes(A, SC)
        proof = collect(self.view, self.args(A, BULK), [saved], classification_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='layers', truth_layer_ids=[SC, TF])
        self.assertFalse(proof.result['claim_cards'][0]['statement_true'])
        self.assertEqual(proof.observations, (saved,))
        self.assertNotIn('layer_gene_ids', proof.result)

    def test_unknown_wider_truth_preserves_foreign_layer_negatives_without_a_verdict(self):
        saved = self.nodes(B, SC)
        proof = collect(self.view, self.args(B, BULK), [saved], classification_requested=False,
            claim='absence_holds_in_other_layers', truth_scope='layers', truth_layer_ids=[SC, TF])
        self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
        self.assertEqual(proof.status, 'invalid_input')
        self.assertIsNone(proof.result)
        self.assertIsNone(proof.next_arguments)
        self.assertEqual(proof.observations, (saved,))

    def test_membership_cannot_settle_expression_truth_in_either_direction_or_future_versions(self):
        for gene in (A, C, MISSING):
            for scope, claim in (('expression', 'gene_not_expressed_in_context'),
                                 ('future_versions', 'absence_holds_in_other_versions')):
                with self.subTest(gene=gene, scope=scope):
                    saved = self.nodes(gene)
                    proof = collect(self.view, self.args(gene), [saved], classification_requested=False,
                                    claim=claim, truth_scope=scope)
                    self.assertEqual(proof.issue_code, 'UNSUPPORTED_TRUTH_SCOPE')
                    self.assertEqual(proof.observations, (saved,))

    def test_explicit_outer_gate_preserves_success_and_omits_unassessed_cards(self):
        for gene, gate in ((MISSING, 'multiplex'), (C, 'layer')):
            with self.subTest(gene=gene, gate=gate):
                saved = self.nodes(gene)
                proof = collect(self.view, self.args(gene), [saved], require_membership=gate,
                                claim='gene_not_expressed_in_context')
                self.assertEqual(proof.status, 'not_found')
                self.assertEqual(proof.issue_code, 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
                self.assertEqual(proof.observations, (saved,))
                self.assertIsNone(proof.result)
        self.assertIsNotNone(collect(self.view, self.args(C), [self.nodes(C)], require_membership='multiplex').result)

    def test_failed_attempt_adds_no_evidence_and_leaves_the_singleton_unchecked(self):
        failed = self.saved('get_nodes_by_layer', self.args(), {'status': 'error',
            'result': {'error_code': 'BACKEND_UNAVAILABLE'}, 'provenance': {'registry': [], 'graph': []}})
        proof = collect(self.view, self.args(), [failed])
        self.assertEqual(proof.observations, ())
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [A])
        self.assertEqual(proof.next_arguments, self.args())

    def test_singleton_shape_and_scope_rejections_are_atomic(self):
        for genes, issue in ((None, 'MISSING_GENES'), ([], 'INVALID_GENE_LIST'),
                ([A, A], 'INVALID_GENE_LIST'), ([A, B], 'INVALID_GENE_LIST'), (A, 'INVALID_GENE_LIST')):
            with self.subTest(genes=genes):
                with self.assertRaises(SelectionError) as caught:
                    collect(self.view, {**self.args(), 'gene_ids': genes}, [])
                self.assertEqual(caught.exception.issue_code, issue)
        for changes, issue in (({'layer_id': None}, 'MISSING_SCOPE'), ({'layer_id': 'TFs:unknown'}, 'UNKNOWN_LAYER'),
                                ({'multiplex_id': 'unknown'}, 'UNKNOWN_MULTIPLEX')):
            with self.subTest(changes=changes):
                with self.assertRaises(SelectionError) as caught:
                    collect(self.view, {**self.args(), **changes}, [])
                self.assertEqual(caught.exception.issue_code, issue)

    def test_invalid_claim_truth_and_gate_choices_are_rejected(self):
        for options in ({'claim': 'invented'}, {'truth_scope': 'expression'}, {'classification_requested': 1},
                {'require_membership': True}, {'truth_scope': 'layers', 'claim': 'absence_holds_in_other_layers'},
                {'truth_scope': 'family', 'claim': 'absence_holds_in_other_layers'},
                {'truth_scope': 'expression', 'claim': 'gene_not_node_in_multiplex'},
                {'truth_scope': 'multiplex', 'claim': 'gene_not_expressed_in_context'},
                {'truth_scope': 'layers', 'truth_layer_ids': [PPI], 'claim': 'gene_not_node_in_multiplex'},
                {'truth_layer_ids': [PPI]}, {'truth_family': 'ppi'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                collect(self.view, self.args(), [], **options)

    def test_authoritative_null_symbol_id_does_not_block_membership(self):
        identity = {'operation': 'lookup_human_ensembl_gene_id', 'arguments': {'gene_id': f' {A.lower()}.4 '},
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': None},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'identity_null'}], 'graph': []}}
        saved = self.nodes()
        proof = collect(self.view, self.args(), [identity, saved])
        self.assertEqual(proof.result['layer_gene_ids'], [A])
        self.assertEqual(proof.observations, (saved,))

    def test_conflicting_current_registry_mappings_are_harness_faults(self):
        identity = {'operation': 'lookup_human_gene_symbol', 'arguments': {'gene_symbol': ' a '},
            'status': 'resolved', 'result': {'gene_id': A, 'gene_symbol': 'A'},
            'provenance': {'registry': [{'registry': 'ensembl', 'release': '116', 'evidence_id': 'identity_a'}], 'graph': []}}
        conflicting = deepcopy(identity)
        conflicting['arguments']['gene_symbol'] = 'A'
        conflicting['result']['gene_id'] = B
        conflicting['provenance']['registry'][0]['evidence_id'] = 'identity_b'
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [identity, conflicting])

    def test_global_positive_cannot_be_absent_from_the_complete_layer_union(self):
        observations = [self.nodes(MISSING, layer) for layer in self.view.layer_order]
        for observation in observations:
            observation['result']['missing_gene_ids'] = []
            observation['result']['not_in_layer_gene_ids'] = [MISSING]
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(MISSING), observations)
        full = self.layers(MISSING)
        full['result']['present'] = True
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(MISSING), [full])


if __name__ == '__main__':
    unittest.main()
