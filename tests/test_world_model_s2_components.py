"""Check exact local partitions, native direction, and complete membership results."""

from copy import deepcopy
import itertools
import random
import unittest

from runtime.world_model.s2_components import component_result, components, validate_connectivity
from runtime.world_model.s2_multiplex import SelectionError
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphRuntime


A, B, C, D, ISOLATED, OUTSIDE, MISSING = [f'ENSG{number:011d}' for number in range(1, 8)]
PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'


def view():
    """Supply selected isolates, an outside route, and two nontrivial strong groups."""
    return GraphView('unit_components', 'a' * 64, {
        PPI: LayerGraph(PPI, False, {(A, B): 0, (B, C): .4, (C, C): .2,
                                    (A, OUTSIDE): .6, (D, OUTSIDE): .5},
                        frozenset((A, B, C, D, ISOLATED, OUTSIDE))),
        COEX: LayerGraph(COEX, False, {(MISSING, MISSING): .1}, frozenset((MISSING,))),
        BRAIN: LayerGraph(BRAIN, True, {(A, B): .8, (B, A): .3, (B, C): .5,
                                       (C, D): 0, (D, C): .9, (A, A): .1},
                          frozenset((A, B, C, D, ISOLATED))),
    })


def rows(pairs):
    """Give each native pair a weight that does not affect connectivity."""
    return [{'gene_ids': list(pair), 'weight': number % 3 / 2} for number, pair in enumerate(pairs)]


class ComponentAlgorithmTests(unittest.TestCase):
    """Check source-independent algorithms with explicit partition expectations."""

    def test_empty_and_zero_length_singleton_reachability(self):
        for kind in ('undirected', 'weak', 'strong'):
            with self.subTest(kind=kind):
                self.assertEqual(components([], [], kind), [])
                self.assertEqual(components([A], [], kind), [[A]])

    def test_isolates_remain_members_and_groups_sort_by_first_id(self):
        self.assertEqual(components([D, C, B, A], rows([(D, B)]), 'undirected'), [[A], [B, D], [C]])

    def test_unweighted_zero_and_negative_edges_join_members(self):
        edges = [{'gene_ids': [A, B], 'weight': 0}, {'gene_ids': [B, C], 'weight': -1}]
        self.assertEqual(components([C, B, A], edges, 'undirected'), [[A, B, C]])

    def test_self_loops_and_reciprocal_arcs_never_duplicate_groups(self):
        edges = rows([(A, A), (A, B), (B, A), (B, B), (C, C)])
        for kind in ('undirected', 'weak', 'strong'):
            with self.subTest(kind=kind):
                self.assertEqual(components([A, B, C, A], edges, kind), [[A, B], [C]])

    def test_outside_routes_cannot_join_selected_members(self):
        edges = rows([(A, OUTSIDE), (OUTSIDE, B), (B, C)])
        for kind in ('undirected', 'weak'):
            with self.subTest(kind=kind):
                self.assertEqual(components([A, B, C], edges, kind), [[A], [B, C]])

    def test_directed_chain_has_one_weak_and_three_strong_groups(self):
        edges = rows([(C, B), (B, A)])
        self.assertEqual(components([A, B, C], edges, 'weak'), [[A, B, C]])
        self.assertEqual(components([A, B, C], edges, 'strong'), [[A], [B], [C]])

    def test_one_way_bridge_keeps_two_nontrivial_strong_groups(self):
        edges = rows([(A, B), (B, A), (B, C), (C, D), (D, C)])
        self.assertEqual(components([D, C, B, A, ISOLATED], edges, 'strong'), [[A, B], [C, D], [ISOLATED]])
        self.assertEqual(components([D, C, B, A, ISOLATED], edges, 'weak'), [[A, B, C, D], [ISOLATED]])

    def test_removing_a_bridge_requires_a_new_partition(self):
        edges = rows([(A, B), (B, C)])
        self.assertEqual(components([A, B, C], edges, 'undirected'), [[A, B, C]])
        self.assertEqual(components([A, C], edges, 'undirected'), [[A], [C]])

    def test_order_duplicates_and_edge_iterators_do_not_change_partition(self):
        pairs = [(A, B), (B, C), (C, A)]
        for kind in ('undirected', 'weak', 'strong'):
            with self.subTest(kind=kind):
                self.assertEqual(components([C, A, B, C], iter(rows(reversed(pairs))), kind), [[A, B, C]])

    def test_pair_generators_support_possible_edge_proofs_without_row_arrays(self):
        for kind in ('undirected', 'weak', 'strong'):
            with self.subTest(kind=kind):
                pairs = ((source, target) for source in (A, B, C) for target in (A, B, C))
                self.assertEqual(components([A, B, C], pairs, kind), [[A, B, C]])

    def test_native_edge_records_and_weights_are_unchanged(self):
        edges = rows([(B, A), (A, C), (C, B)])
        before = deepcopy(edges)
        components([A, B, C], edges, 'weak')
        components([A, B, C], edges, 'strong')
        self.assertEqual(edges, before)

    def test_long_directed_chain_and_cycle_have_no_recursion_cap(self):
        genes = [f'ENSG{number:011d}' for number in range(5000)]
        edges = rows(zip(genes, genes[1:]))
        self.assertEqual(components(genes, edges, 'strong'), [[gene] for gene in genes])
        self.assertEqual(components(genes, edges, 'weak'), [genes])
        edges.append({'gene_ids': [genes[-1], genes[0]], 'weight': 0})
        self.assertEqual(components(genes, edges, 'strong'), [genes])

    def test_strong_groups_match_independent_mutual_reachability(self):
        randomizer = random.Random(20261029)
        genes = [A, B, C, D, ISOLATED]
        for ordinal in range(100):
            pairs = [pair for pair in itertools.product(genes, repeat=2) if randomizer.random() < .23]
            reachable = {}
            for source in genes:
                reached = {source}
                for _ in genes:
                    reached |= {right for left, right in pairs if left in reached}
                reachable[source] = reached
            expected, remaining = [], set(genes)
            while remaining:
                source = min(remaining)
                group = sorted(target for target in remaining
                               if target in reachable[source] and source in reachable[target])
                remaining.difference_update(group)
                expected.append(group)
            with self.subTest(ordinal=ordinal):
                self.assertEqual(components(genes, rows(pairs), 'strong'), expected)

    def test_kind_must_be_explicit_for_the_pure_algorithm(self):
        for kind in (None, '', 'directed', True, [], {}):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                components([A], [], kind)


class ConnectivityValidationTests(unittest.TestCase):
    """Use layer metadata only and retain the reviewed local error codes."""

    def test_undirected_default_and_explicit_eligible_kinds(self):
        self.assertEqual(validate_connectivity(view(), PPI, None), 'undirected')
        self.assertEqual(validate_connectivity(view(), PPI, 'undirected'), 'undirected')
        for kind in ('weak', 'strong'):
            self.assertEqual(validate_connectivity(view(), BRAIN, kind), kind)

    def test_directed_kind_is_required_even_for_a_singleton(self):
        with self.assertRaises(SelectionError) as raised:
            validate_connectivity(view(), BRAIN, None)
        self.assertEqual((raised.exception.status, raised.exception.issue_code),
                         ('invalid_input', 'CONNECTIVITY_KIND_REQUIRED'))

    def test_unknown_values_are_invalid_without_coercion(self):
        for kind in ('Weak', '', 'directed', True, 0, [], {}):
            with self.subTest(kind=kind), self.assertRaises(SelectionError) as raised:
                validate_connectivity(view(), PPI, kind)
            self.assertEqual(raised.exception.issue_code, 'INVALID_CONNECTIVITY_KIND')

    def test_known_incompatible_values_are_unavailable(self):
        for layer, kind in ((PPI, 'weak'), (PPI, 'strong'), (BRAIN, 'undirected')):
            with self.subTest(layer=layer, kind=kind), self.assertRaises(SelectionError) as raised:
                validate_connectivity(view(), layer, kind)
            self.assertEqual(raised.exception.issue_code, 'CONNECTIVITY_KIND_UNAVAILABLE')

    def test_invalid_scope_is_separate_from_invalid_connectivity(self):
        for layer, status, issue in ((None, 'invalid_input', 'MISSING_SCOPE'),
                                     ('unknown', 'not_found', 'UNKNOWN_LAYER'),
                                     ([], 'invalid_input', 'MISSING_SCOPE')):
            with self.subTest(layer=layer), self.assertRaises(SelectionError) as raised:
                validate_connectivity(view(), layer, 'bad')
            self.assertEqual((raised.exception.status, raised.exception.issue_code), (status, issue))

    def test_metadata_validation_never_reads_inventory(self):
        class MetadataView:
            layers = {PPI: type('Layer', (), {'directed': False})()}

            @property
            def multiplex_nodes(self):
                raise AssertionError('kind validation must not read source inventories')

        self.assertEqual(validate_connectivity(MetadataView(), PPI, None), 'undirected')


class ComponentResultTests(unittest.TestCase):
    """Keep exact projected groups separate from all-input membership and edge counts."""

    def setUp(self):
        self.view = view()
        self.runtime = SubgraphRuntime(self.view)

    def result(self, genes, layer=PPI, kind=None):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer,
                     'gene_ids': list(genes), 'connectivity_kind': kind}
        graph = self.runtime.induce_subgraph(**self.runtime.arguments(list(genes), layer), evidence_id='unit')['result']
        return component_result(self.view, arguments, graph)

    def test_group_collection_counts_groups_instead_of_edges_or_genes(self):
        result = self.result([A, B, C, D, ISOLATED])
        self.assertEqual(result['components'], [[A, B, C], [D], [ISOLATED]])
        self.assertEqual(result['collection'], {'mode': 'complete', 'returned_count': 3,
            'total_count': 3, 'complete': True, 'next_cursor': None})
        self.assertEqual(result['component_count'], 3)
        self.assertEqual(result['layer_gene_count'], 5)
        self.assertNotIn('edges', result)
        self.assertNotIn('edge_count', result)

    def test_connected_projection_with_exclusions_has_false_all_input_boolean(self):
        global_absent = 'ENSG99999999999'
        result = self.result([C, MISSING, global_absent, A, B, C])
        self.assertEqual(result['components'], [[A, B, C]])
        self.assertEqual(result['component_count'], 1)
        self.assertFalse(result['single_component'])
        self.assertEqual(result['not_in_layer_gene_ids'], [MISSING])
        self.assertEqual(result['missing_gene_ids'], [global_absent])
        self.assertEqual(result['coverage']['requested_gene_ids'], [C, MISSING, global_absent, A, B])
        self.assertEqual(result['coverage']['checked_gene_ids'], [C, MISSING, global_absent, A, B])

    def test_empty_projection_all_excluded_and_singleton_results(self):
        for genes in ([], [MISSING, 'ENSG99999999999']):
            with self.subTest(genes=genes):
                result = self.result(genes)
                self.assertEqual(result['components'], [])
                self.assertEqual(result['component_count'], 0)
                self.assertIs(result['single_component'], False)
        self.assertEqual(self.result([ISOLATED])['components'], [[ISOLATED]])
        self.assertIs(self.result([ISOLATED])['single_component'], True)

    def test_weak_result_preserves_native_directedness(self):
        result = self.result([A, B, C, D], BRAIN, 'weak')
        self.assertTrue(result['directed'])
        self.assertEqual(result['connectivity_kind'], 'weak')
        self.assertEqual(result['components'], [[A, B, C, D]])

    def test_multiple_nontrivial_strong_components(self):
        self.assertEqual(self.result([D, C, B, A], BRAIN, 'strong')['components'], [[A, B], [C, D]])

    def test_result_does_not_mutate_graph_evidence(self):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': PPI,
                     'gene_ids': [C, A, B], 'connectivity_kind': None}
        graph = self.runtime.induce_subgraph(**self.runtime.arguments([C, A, B], PPI), evidence_id='unit')['result']
        before = deepcopy(graph)
        result = component_result(self.view, arguments, graph)
        result['coverage']['requested_gene_ids'].append(D)
        result['components'][0].append(D)
        self.assertEqual(graph, before)

    def test_result_accepts_proved_partial_witness_without_claiming_complete_edges(self):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': PPI,
                     'gene_ids': [A, B, C], 'connectivity_kind': 'undirected'}
        graph = self.runtime.induce_subgraph(**self.runtime.arguments([A, B, C], PPI), evidence_id='unit')['result']
        graph['edges'] = rows([(A, B), (B, C)])
        graph['collection']['complete'] = False
        self.assertTrue(component_result(self.view, arguments, graph)['collection']['complete'])
        self.assertFalse(graph['collection']['complete'])

    def test_component_result_uses_supplied_membership_without_inventory_access(self):
        class MetadataView:
            multiplex_id = 'unit_metadata_only'
            layers = {BRAIN: type('Layer', (), {'directed': True})()}

            @property
            def multiplex_nodes(self):
                raise AssertionError('local component answers must not read hidden source inventories')

        arguments = {'multiplex_id': 'unit_metadata_only', 'layer_id': BRAIN,
                     'gene_ids': [A], 'connectivity_kind': 'strong'}
        graph = {'layer_gene_ids': [A], 'not_in_layer_gene_ids': [], 'missing_gene_ids': [],
                 'coverage': {'operation': 'layer_membership', 'requested_gene_ids': [A],
                              'checked_gene_ids': [A], 'unchecked_gene_ids': []}}
        self.assertEqual(component_result(MetadataView(), arguments, graph)['components'], [[A]])

    def test_result_rejects_incomplete_membership_and_scope_mismatch(self):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': PPI,
                     'gene_ids': [A, B], 'connectivity_kind': None}
        graph = self.runtime.induce_subgraph(**self.runtime.arguments([A, B], PPI), evidence_id='unit')['result']
        changes = [{'layer_gene_ids': [A]}, {'missing_gene_ids': [A]}, {'directed': True},
                   {'layer_id': BRAIN}, {'coverage': {**graph['coverage'], 'unchecked_gene_ids': [B]}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                component_result(self.view, arguments, {**graph, **change})


if __name__ == '__main__':
    unittest.main()
