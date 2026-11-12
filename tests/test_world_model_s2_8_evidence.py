"""Check degree and hub proofs from exact model-visible evidence."""

from copy import deepcopy
import unittest

from runtime.world_model.s2_8_evidence import collect, validate_histogram
from runtime.world_model.s2_degrees import DegreeIndex, DegreeRuntime
from runtime.world_model.s2_multiplex import EvidenceError
from runtime.world_model.s2_neighbor_layers import NeighborLayersRuntime
from runtime.world_model.s2_neighbors import NeighborRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph


class DegreeEvidenceTests(unittest.TestCase):
    """Use loops, overlapping native sets, isolates, and graph-only neighbors."""

    def setUp(self):
        self.a, self.b, self.c, self.d, self.e, self.missing = [f'ENSG{i:011d}' for i in range(1, 7)]
        a, b, c, d, e = self.a, self.b, self.c, self.d, self.e
        self.ppi, self.coex, self.tf = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'
        self.view = GraphView('degree_evidence', 'f' * 64, {
            self.ppi: LayerGraph(self.ppi, False, {(a, a): .9, (a, b): .8, (a, c): .7, (b, c): 0}, frozenset((a, b, c, d))),
            self.coex: LayerGraph(self.coex, False, {(a, b): .6, (a, e): .5}, frozenset((a, b, e))),
            self.tf: LayerGraph(self.tf, True, {(a, a): .8, (a, b): .7, (b, a): .6, (c, a): .5}, frozenset((a, b, c, d))),
        })
        self.index = DegreeIndex(self.view)
        self.runtime = DegreeRuntime(self.view, index=self.index)
        self.neighbors = NeighborRuntime(self.view, index=self.index.native_index)

    def scope(self, layers=None, *, family=None, all_layers=False):
        return {'kind': 'all' if family or all_layers else 'layers',
                'layer_ids': [] if family or all_layers else list(layers or [self.ppi]), 'layer_family': family}

    def args(self, *, gene=None, scope=None, direction='undirected', hub=False, threshold=95):
        return {'multiplex_id': self.view.multiplex_id, 'layer_scope': scope or self.scope(),
                'gene_id': gene or self.a, 'direction': direction,
                'request_kind': 'degree_and_hub' if hub else 'degree_only',
                'hub_policy': {'policy_id': 'degree_midrank_v1', 'threshold_percentile': threshold} if hub else None}

    def degree(self, args=None, identity='ev_degree'):
        args = args or self.args()
        return {'operation': 'get_degree', 'arguments': deepcopy(args),
                **self.runtime.get_degree(**args, evidence_id=identity)}

    def native(self, *, gene=None, layer=None, direction='undirected', mode='complete', top_k=None,
               cursor=None, runtime=None, identity='ev_native'):
        args = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer or self.ppi,
                'gene_id': gene or self.a, 'direction': direction, 'mode': mode, 'top_k': top_k, 'cursor': cursor}
        return {'operation': 'get_neighbors', 'arguments': args,
                **(runtime or self.neighbors).get_neighbors(**args, evidence_id=identity)}

    def global_record(self, gene=None, present=True, identity='ev_global'):
        gene = gene or self.a
        scope = self.scope(all_layers=True)
        return {'operation': 'check_multiplex_membership',
            'arguments': {'multiplex_id': self.view.multiplex_id, 'gene_ids': [gene]}, 'status': 'resolved',
            'result': {'multiplex_id': self.view.multiplex_id,
                       'present_gene_ids': [gene] if present else [], 'absent_gene_ids': [] if present else [gene]},
            'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                'graph_version': self.view.graph_version, 'layer_scope': scope, 'evidence_id': identity}]}}

    def histogram(self, args=None, identity='ev_distribution'):
        args = args or self.args(hub=True)
        selected = self.runtime.validate_arguments(args)
        scope = self.runtime.multiplex._scope(args['layer_scope'], selected)
        source = {key: deepcopy(args[key]) for key in ('multiplex_id', 'layer_scope', 'direction')}
        source.update(degree_policy_id='distinct_neighbor_union_v1', population_policy_id='selected_layer_node_union_v1')
        result = {**deepcopy(source), 'layer_scope': scope, 'resolved_layer_ids': list(selected),
                  'population_count': len(self.index.population(selected)),
                  'degree_histogram': self.index.histogram(selected, args['direction']), 'complete': True}
        return {'operation': 'degree_distribution', 'arguments': source, 'status': 'resolved', 'result': result,
            'provenance': {'registry': [], 'graph': [{'multiplex_id': self.view.multiplex_id,
                'graph_version': self.view.graph_version, 'layer_scope': scope, 'evidence_id': identity}]}}

    def test_missing_evidence_requests_exact_six_arguments(self):
        args = self.args(hub=True)
        proof = collect(self.view, args, [])
        self.assertEqual(proof.next_arguments, args)
        self.assertEqual(proof.next_tool, 'get_degree')
        self.assertIsNone(proof.coverage)

    def test_source_blind_planner_does_not_read_any_source_nodes_or_edges(self):
        class Hidden:
            def __iter__(self):
                raise AssertionError('hidden source read')

            def __contains__(self, key):
                raise AssertionError('hidden source read')

        view = GraphView(self.view.multiplex_id, self.view.graph_version, {
            layer: LayerGraph(layer, graph.directed, {}, frozenset()) for layer, graph in self.view.layers.items()})
        for graph in view.layers.values():
            object.__setattr__(graph, 'edges', Hidden())
            object.__setattr__(graph, 'nodes', Hidden())
        proof = collect(view, self.args(), [self.native()])
        self.assertEqual(proof.result['degree'], 3)

    def test_matching_measurement_resolves_degree_and_hub(self):
        args = self.args(hub=True)
        proof = collect(self.view, args, [self.degree(args)])
        self.assertEqual(proof.result['degree'], 3)
        self.assertTrue(proof.result['hub_like'])
        self.assertEqual(proof.result['caveats'], ['high_degree_inflates_apparent_proximity'])

    def test_exact_ranks_recompare_a_new_threshold(self):
        old = self.args(gene=self.b, hub=True, threshold=0)
        proof = collect(self.view, self.args(gene=self.b, hub=True, threshold=100), [self.degree(old)])
        self.assertFalse(proof.result['hub_like'])
        self.assertEqual(proof.result['hub_policy']['threshold_percentile'], 100)
        self.assertIsNone(proof.next_arguments)

    def test_degree_only_does_not_require_or_retain_rank_fields(self):
        proof = collect(self.view, self.args(), [self.degree(self.args(hub=True))])
        for key in ('degree_percentile', 'rank_support', 'hub_policy', 'hub_like'):
            self.assertIsNone(proof.result[key])
        self.assertEqual(proof.result['caveats'], [])

    def test_complete_native_set_counts_a_loop_once(self):
        proof = collect(self.view, self.args(), [self.native()])
        self.assertEqual(proof.result['degree'], 3)
        self.assertTrue(proof.result['membership']['present_in_scope'])

    def test_complete_native_neighbors_do_not_supply_population_rank(self):
        proof = collect(self.view, self.args(hub=True), [self.native()])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_tool, 'get_degree')
        self.assertEqual(len(proof.observations), 1)

    def test_bounded_authoritative_total_supplies_exact_degree(self):
        proof = collect(self.view, self.args(), [self.native(mode='top_k', top_k=1)])
        self.assertEqual(proof.result['degree'], 3)
        self.assertIsNone(proof.next_arguments)

    def test_unknown_total_prefix_requests_statistic_instead_of_unneeded_pages(self):
        runtime = NeighborRuntime(self.view, index=self.index.native_index, page_size=1, disclose_total=False)
        proof = collect(self.view, self.args(), [self.native(runtime=runtime)])
        self.assertEqual(proof.next_arguments, self.args())

    def test_detached_final_page_never_becomes_a_complete_neighbor_set(self):
        runtime = NeighborRuntime(self.view, index=self.index.native_index, page_size=2, disclose_total=False)
        first = self.native(runtime=runtime)
        last = self.native(runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='ev_last')
        proof = collect(self.view, self.args(), [last])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.args())

    def test_detached_page_with_authoritative_total_can_answer_count(self):
        runtime = NeighborRuntime(self.view, index=self.index.native_index, page_size=2)
        first = self.native(runtime=runtime)
        last = self.native(runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='ev_last')
        self.assertEqual(collect(self.view, self.args(), [last]).result['degree'], 3)

    def test_complete_unknown_total_traversal_answers_exact_count(self):
        runtime = NeighborRuntime(self.view, index=self.index.native_index, page_size=2, disclose_total=False)
        first = self.native(runtime=runtime)
        last = self.native(runtime=runtime, cursor=first['result']['collection']['next_cursor'], identity='ev_last')
        self.assertEqual(collect(self.view, self.args(), [first, last]).result['degree'], 3)

    def test_incident_union_deduplicates_reciprocal_and_self_neighbors(self):
        args = self.args(scope=self.scope([self.tf]), direction='incident')
        records = [self.native(layer=self.tf, direction=direction, identity=f'ev_{direction}') for direction in ('in', 'out')]
        self.assertEqual(collect(self.view, args, records).result['degree'], 3)

    def test_separate_directional_totals_cannot_establish_incident_union(self):
        args = self.args(scope=self.scope([self.tf]), direction='incident')
        records = [self.native(layer=self.tf, direction=direction, mode='top_k', top_k=1, identity=f'ev_{direction}')
                   for direction in ('in', 'out')]
        proof = collect(self.view, args, records)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, args)

    def test_cross_layer_sets_form_a_distinct_union(self):
        args = self.args(scope=self.scope([self.ppi, self.coex]))
        records = [self.native(layer=layer, identity=f'ev_{layer}') for layer in (self.ppi, self.coex)]
        self.assertEqual(collect(self.view, args, records).result['degree'], 4)

    def test_complete_map_or_its_authoritative_total_proves_union_degree(self):
        runtime = NeighborLayersRuntime(self.view, index=self.index.native_index)
        args = self.args(scope=self.scope(all_layers=True), direction='incident')
        source = {'multiplex_id': self.view.multiplex_id, 'gene_id': self.a,
                  'layer_scope': self.scope(all_layers=True), 'direction': 'incident',
                  'mode': 'top_k', 'top_k': 1, 'cursor': None}
        record = {'operation': 'get_neighbor_layers', 'arguments': source,
                  **runtime.get_neighbor_layers(**source, evidence_id='ev_map')}
        self.assertEqual(collect(self.view, args, [record]).result['degree'], 4)

    def test_equivalent_selectors_and_undirected_incident_forms_reuse_measurement(self):
        old = self.args(scope=self.scope([self.ppi]))
        args = self.args(scope=self.scope(family='ppi'), direction='incident')
        proof = collect(self.view, args, [self.degree(old)])
        self.assertEqual(proof.result['degree'], 3)
        self.assertEqual(proof.result['layer_scope'], args['layer_scope'])
        self.assertEqual(proof.observations[0]['arguments'], old)

    def test_empty_family_requires_global_membership_even_for_degree_only(self):
        args = self.args(scope=self.scope(family='bulk_pen'), direction='incident')
        self.assertEqual(collect(self.view, args, []).next_arguments, args)
        proof = collect(self.view, args, [self.global_record()])
        self.assertEqual(proof.result['degree'], 0)
        self.assertFalse(proof.result['membership']['present_in_scope'])

    def test_empty_family_hub_is_unavailable_and_retains_global_evidence(self):
        args = self.args(scope=self.scope(family='bulk_pen'), direction='incident', hub=True)
        proof = collect(self.view, args, [self.global_record()])
        self.assertEqual(proof.issue_code, 'EMPTY_REFERENCE_POPULATION')
        self.assertIsNone(proof.result)
        self.assertEqual(len(proof.observations), 1)

    def test_outside_layer_zero_resolves_count_but_defers_hub(self):
        record = self.native(gene=self.e)
        self.assertEqual(collect(self.view, self.args(gene=self.e), [record]).result['degree'], 0)
        proof = collect(self.view, self.args(gene=self.e, hub=True), [record])
        self.assertEqual(proof.issue_code, 'OUTSIDE_REFERENCE_POPULATION')

    def test_global_absence_is_not_a_measured_zero(self):
        args = self.args(gene=self.missing)
        proof = collect(self.view, args, [self.native(gene=self.missing)])
        self.assertEqual(proof.issue_code, 'NO_MULTIPLEX_MEMBERSHIP')
        self.assertIsNone(proof.result)

    def test_histogram_and_known_query_membership_supply_exact_rank(self):
        args = self.args(hub=True)
        proof = collect(self.view, args, [self.native(), self.histogram(args)])
        self.assertEqual(proof.result['rank_support'], self.index.rank_support([self.ppi], self.a, 'undirected'))
        self.assertTrue(proof.result['hub_like'])
        self.assertEqual(len(proof.observations), 2)

    def test_histogram_and_global_membership_do_not_establish_narrow_population_membership(self):
        measurement = self.degree()
        measurement['result']['membership']['present_in_scope'] = None
        args = self.args(hub=True)
        proof = collect(self.view, args, [measurement, self.global_record(), self.histogram(args)])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, args)

    def test_incomplete_histogram_cannot_supply_exact_rank(self):
        distribution = self.histogram()
        distribution['result']['degree_histogram'] = distribution['result']['degree_histogram'][:1]
        distribution['result']['complete'] = False
        proof = collect(self.view, self.args(hub=True), [self.native(), distribution])
        self.assertIsNone(proof.result)

    def test_rounded_percentile_without_support_requires_rank_acquisition(self):
        record = self.degree(self.args(hub=True))
        record['result']['rank_support'] = None
        record['result']['degree_percentile'] = 95.0
        self.assertEqual(collect(self.view, self.args(hub=True), [record]).next_arguments, self.args(hub=True))

    def test_histogram_rows_and_population_counts_are_strict(self):
        for mutation in ('sum', 'duplicate', 'negative', 'boolean'):
            record = self.histogram()
            if mutation == 'sum':
                record['result']['population_count'] += 1
            elif mutation == 'duplicate':
                record['result']['degree_histogram'].append(record['result']['degree_histogram'][-1])
            elif mutation == 'negative':
                record['result']['degree_histogram'][0]['degree'] = -1
            else:
                record['result']['degree_histogram'][0]['node_count'] = True
            with self.subTest(mutation=mutation), self.assertRaises(EvidenceError):
                validate_histogram(self.view, record)

    def test_conflicting_counts_are_a_fault_even_when_complete_measurement_exists(self):
        first, second = self.degree(), self.degree(identity='ev_other')
        second['result']['degree'] += 1
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_conflicting_native_weights_are_a_fault(self):
        first, second = self.native(), self.native(identity='ev_other')
        second['result']['neighbors'][0]['weight'] += .01
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_conflicting_rank_and_distribution_are_a_fault(self):
        args = self.args(gene=self.b, hub=True)
        record = self.histogram(args)
        record['result']['degree_histogram'][0]['node_count'] += 1
        record['result']['degree_histogram'][1]['node_count'] -= 1
        with self.assertRaises(EvidenceError):
            collect(self.view, args, [self.degree(args), record])

    def test_reused_evidence_ids_cannot_name_different_acquisitions(self):
        first, second = self.degree(), self.degree(self.args(gene=self.b))
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [first, second])

    def test_duplicate_observations_do_not_duplicate_provenance(self):
        record = self.degree()
        proof = collect(self.view, self.args(), [record, deepcopy(record)])
        self.assertEqual(len(proof.observations), 1)
        self.assertEqual(len(proof.provenance['graph']), 1)

    def test_stale_records_never_supply_a_current_count(self):
        record = self.degree()
        record['provenance']['graph'][0]['graph_version'] = '0' * 64
        proof = collect(self.view, self.args(), [record])
        self.assertEqual(proof.next_arguments, self.args())
        self.assertEqual(proof.observations, ())

    def test_wrong_scope_or_direction_cannot_supply_requested_count(self):
        records = [self.degree(self.args(scope=self.scope([self.tf]), direction='in'))]
        proof = collect(self.view, self.args(scope=self.scope([self.tf]), direction='incident'), records)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_tool, 'get_degree')

    def test_exact_union_count_cannot_be_smaller_than_a_known_native_subset(self):
        args = self.args(scope=self.scope([self.tf]), direction='incident')
        union = self.degree(args)
        union['result']['degree'] = 1
        native = self.native(layer=self.tf, direction='in')
        with self.assertRaises(EvidenceError):
            collect(self.view, args, [native, union])

    def test_unknown_total_positive_prefix_still_constrains_a_later_exact_count(self):
        runtime = NeighborRuntime(self.view, index=self.index.native_index, page_size=2, disclose_total=False)
        prefix = self.native(runtime=runtime)
        measurement = self.degree()
        measurement['result']['degree'] = 1
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args(), [prefix, measurement])

    def test_incompatible_degree_policy_cannot_supply_the_requested_count(self):
        record = self.degree()
        record['result']['degree_policy_id'] = 'incidence_count_v0'
        proof = collect(self.view, self.args(), [record])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.args())

    def test_incompatible_population_policy_leaves_a_compatible_degree_usable(self):
        record = self.degree(self.args(hub=True))
        record['result']['rank_support']['population_policy_id'] = 'positive_degree_only_v0'
        self.assertEqual(collect(self.view, self.args(), [record]).result['degree'], 3)
        proof = collect(self.view, self.args(hub=True), [record])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.args(hub=True))

    def test_incompatible_distribution_policy_cannot_supply_exact_rank(self):
        record = self.histogram()
        for item in ('arguments', 'result'):
            record[item]['population_policy_id'] = 'positive_degree_only_v0'
        proof = collect(self.view, self.args(hub=True), [self.native(), record])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments, self.args(hub=True))


if __name__ == '__main__':
    unittest.main()
