"""Check source degree unions and exact policy comparisons without registry caps."""

from copy import deepcopy
import unittest

from runtime.world_model.s2_degrees import (
    ARGUMENT_KEYS, CAVEAT, DegreeIndex, DegreeRuntime, final_result, hub_like, percentile,
)
from runtime.world_model.s2_multiplex import SelectionError
from runtime.world_model.s2_sources import GraphView, LayerGraph


PPI, COEX, TF = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'
A, B, C, D, E, MISSING = [f'ENSG{number:011d}' for number in range(1, 7)]


def scope(*layers, family=None):
    """Construct one explicit scope or one exhaustive family selector."""
    return {'kind': 'layers' if layers else 'all', 'layer_ids': list(layers), 'layer_family': family}


def rank(lower, equal, size):
    """Construct exact rank support under the pinned population policy."""
    return {'population_policy_id': 'selected_layer_node_union_v1', 'lower_degree_count': lower,
            'equal_degree_count': equal, 'population_count': size}


class DegreeRuntimeTests(unittest.TestCase):
    """Use loops, reciprocal edges, a cross-layer overlap, and authoritative isolates."""

    def setUp(self):
        self.view = GraphView('unit_degree_view', 'd' * 64, {
            PPI: LayerGraph(PPI, False, {(A, A): .5, (A, B): 0, (B, C): .3}, frozenset((A, B, C, D))),
            COEX: LayerGraph(COEX, False, {(A, B): .8, (A, E): .4}, frozenset((A, B, E))),
            TF: LayerGraph(TF, True, {(A, B): .9, (B, A): .8, (A, A): .7, (C, A): .6}, frozenset((A, B, C, D))),
        })
        self.index = DegreeIndex(self.view)
        self.runtime = DegreeRuntime(self.view, index=self.index)

    def call(self, gene=A, selected=None, direction='undirected', kind='degree_and_hub', threshold=95):
        return self.runtime.get_degree(self.view.multiplex_id, selected or scope(PPI), gene, direction, kind,
            None if kind == 'degree_only' else {'policy_id': 'degree_midrank_v1', 'threshold_percentile': threshold},
            evidence_id='ev_unit_degree')

    def test_undirected_self_loop_is_one_neighbor_and_zero_weight_is_retained(self):
        self.assertEqual(self.index.neighbor_ids((PPI,), A, 'undirected'), frozenset((A, B)))
        result = self.call()['result']
        self.assertEqual(result['degree'], 2)
        self.assertEqual(result['rank_support'], rank(2, 2, 4))

    def test_cross_layer_union_deduplicates_a_shared_neighbor(self):
        response = self.call(selected=scope(PPI, COEX), kind='degree_only')
        self.assertEqual(response['result']['degree'], 3)
        self.assertNotEqual(response['result']['degree'], self.index.degree((PPI,), A, 'undirected')
                            + self.index.degree((COEX,), A, 'undirected'))

    def test_incident_union_deduplicates_reciprocal_and_loop_edges(self):
        self.assertEqual(self.index.degree((TF,), A, 'in'), 3)
        self.assertEqual(self.index.degree((TF,), A, 'out'), 2)
        self.assertEqual(self.call(selected=scope(TF), direction='incident')['result']['degree'], 3)

    def test_mixed_scope_uses_incident_union(self):
        response = self.call(selected=scope(PPI, TF), direction='incident', kind='degree_only')
        self.assertEqual(response['result']['degree'], 3)

    def test_histogram_contains_authoritative_directional_zeros(self):
        self.assertEqual(self.index.histogram((TF,), 'out'), [
            {'degree': 0, 'node_count': 1}, {'degree': 1, 'node_count': 2}, {'degree': 2, 'node_count': 1}])
        self.assertEqual(self.call(D, scope(TF), direction='out')['result']['rank_support'], rank(0, 1, 4))

    def test_outside_layer_degree_only_resolves_zero(self):
        response = self.call(E, scope(PPI), kind='degree_only')
        self.assertEqual(response['status'], 'resolved')
        self.assertEqual(response['result']['degree'], 0)
        self.assertEqual(response['result']['membership'], {'present_in_multiplex': True, 'present_in_scope': False})

    def test_outside_population_hub_request_retains_zero_without_rank(self):
        response = self.call(E, scope(PPI))
        self.assertEqual(response['status'], 'not_found')
        self.assertEqual(response['result']['degree'], 0)
        self.assertIsNone(response['result']['rank_support'])

    def test_global_absence_has_unavailable_count(self):
        response = self.call(MISSING)
        self.assertEqual(response['status'], 'not_found')
        self.assertIsNone(response['result']['degree'])
        self.assertEqual(response['result']['membership'], {'present_in_multiplex': False, 'present_in_scope': None})

    def test_empty_family_degree_only_resolves_and_hub_request_is_unavailable(self):
        empty = scope(family='bulk_pen')
        count = self.call(selected=empty, direction='incident', kind='degree_only')
        hub = self.call(selected=empty, direction='incident')
        self.assertEqual(count['status'], 'resolved')
        self.assertEqual(count['result']['degree'], 0)
        self.assertEqual(hub['status'], 'not_found')
        self.assertEqual(hub['result']['resolved_layer_ids'], [])

    def test_equivalent_selectors_have_equal_source_statistics(self):
        explicit = self.call(selected=scope(PPI))['result']
        family = self.call(selected=scope(family='ppi'))['result']
        for key in ('resolved_layer_ids', 'degree', 'degree_percentile', 'rank_support'):
            self.assertEqual(explicit[key], family[key])

    def test_count_only_has_no_hub_statistics_or_warning(self):
        result = final_result(self.call(kind='degree_only')['result'])
        for key in ('degree_percentile', 'rank_support', 'hub_policy', 'hub_like'):
            self.assertIsNone(result[key])
        self.assertEqual(result['caveats'], [])

    def test_true_policy_label_has_only_the_declared_warning(self):
        result = final_result(self.call(threshold=0)['result'])
        self.assertTrue(result['hub_like'])
        self.assertEqual(result['caveats'], [CAVEAT])

    def test_false_policy_label_has_no_warning(self):
        result = final_result(self.call(threshold=100)['result'])
        self.assertFalse(result['hub_like'])
        self.assertEqual(result['caveats'], [])

    def test_runtime_returns_exact_measurement_and_provenance_fields(self):
        response = self.call()
        self.assertEqual(tuple(response['result']), ('gene_id', 'multiplex_id', 'layer_scope', 'resolved_layer_ids',
            'direction', 'request_kind', 'degree_policy_id', 'membership', 'degree', 'degree_percentile',
            'rank_support', 'hub_policy'))
        self.assertEqual(response['provenance']['graph'][0], {'multiplex_id': self.view.multiplex_id,
            'graph_version': self.view.graph_version, 'layer_scope': scope(PPI), 'evidence_id': 'ev_unit_degree'})

    def test_scope_and_direction_rejections_have_no_graph_provenance(self):
        for selected, direction, code in ((scope(TF), 'undirected', 'DIRECTION_UNAVAILABLE'),
            (scope(PPI), 'out', 'DIRECTION_UNAVAILABLE'),
            ({'kind': 'layers', 'layer_ids': [], 'layer_family': None}, 'incident', 'INVALID_SCOPE'),
            (scope(PPI, PPI), 'incident', 'INVALID_SCOPE'),
            (scope('TFs:missing'), 'incident', 'UNKNOWN_LAYER'),
            ({'kind': 'layers', 'layer_ids': [{}], 'layer_family': None}, 'incident', 'INVALID_SCOPE')):
            with self.subTest(selected=selected, direction=direction):
                response = self.call(selected=selected, direction=direction)
                self.assertEqual(response['result']['issue_code'], code)
                self.assertEqual(response['provenance'], {'registry': [], 'graph': []})

    def test_invalid_thresholds_are_not_coerced(self):
        for threshold in (True, False, 95.0, '95', -1, 101, None):
            with self.subTest(threshold=threshold):
                response = self.call(threshold=threshold)
                self.assertEqual(response['result']['issue_code'], 'INVALID_HUB_POLICY')

    def test_all_six_arguments_are_required(self):
        arguments = dict(zip(ARGUMENT_KEYS, (self.view.multiplex_id, scope(PPI), A,
            'undirected', 'degree_only', None)))
        self.runtime.validate_arguments(arguments)
        for name in ARGUMENT_KEYS:
            with self.subTest(name=name), self.assertRaises(SelectionError):
                self.runtime.validate_arguments({key: value for key, value in arguments.items() if key != name})

    def test_validation_does_not_construct_a_source_index(self):
        runtime = DegreeRuntime(self.view)
        runtime.resolve_scope(scope(TF))
        runtime.validate_arguments(dict(zip(ARGUMENT_KEYS, (self.view.multiplex_id, scope(PPI), A,
            'undirected', 'degree_only', None))))
        self.assertIsNone(runtime.index)

    def test_explicit_scope_normalizes_manifest_order(self):
        response = self.call(selected=scope(COEX, PPI), kind='degree_only')
        self.assertEqual(response['result']['layer_scope'], scope(PPI, COEX))

    def test_cached_histograms_return_independent_rows(self):
        first = self.index.histogram((PPI,), 'undirected')
        first[0]['node_count'] = 999
        self.assertNotEqual(first, self.index.histogram((PPI,), 'undirected'))


class DegreePolicyTests(unittest.TestCase):
    """Check attainable ties, singleton policy, and exact threshold comparisons."""

    def test_unique_minimum_and_maximum(self):
        self.assertEqual(percentile(rank(0, 1, 17)), 0)
        self.assertEqual(percentile(rank(16, 1, 17)), 100)

    def test_all_tied_and_singleton_populations_are_neutral(self):
        for support in (rank(0, 12, 12), rank(0, 1, 1)):
            self.assertEqual(percentile(support), 50)
            self.assertTrue(hub_like(support, 50))
            self.assertFalse(hub_like(support, 51))

    def test_six_place_half_up_rounding(self):
        self.assertEqual(percentile(rank(1, 1, 7)), 16.666667)
        self.assertEqual(percentile(rank(1, 1, 257)), .390625)

    def test_exact_comparison_does_not_use_the_display(self):
        support = rank(189999999, 1, 200000001)
        self.assertEqual(percentile(support), 95)
        self.assertFalse(hub_like(support, 95))

    def test_exact_equality_qualifies(self):
        self.assertTrue(hub_like(rank(19, 1, 21), 95))

    def test_malformed_rank_support_is_rejected(self):
        for support in (rank(0, 0, 1), rank(-1, 1, 2), rank(1, 2, 2), rank(0, True, 2),
            rank(0, 1, 0), {'population_policy_id': 'wrong', 'lower_degree_count': 0,
                          'equal_degree_count': 1, 'population_count': 1}):
            with self.subTest(support=support), self.assertRaises(ValueError):
                percentile(support)


if __name__ == '__main__':
    unittest.main()
