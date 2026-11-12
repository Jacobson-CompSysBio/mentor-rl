"""Check native shortest routes, canonical ties, membership, and exact diagnostics."""

from itertools import combinations, permutations
import unittest

from runtime.world_model.s2_multiplex import SelectionError
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_paths import ARGUMENT_KEYS, PathIndex, PathRuntime, path_result
from runtime.world_model.s2_sources import GraphView, LayerGraph


A, B, C, D, X, Z, ISOLATED, OUTSIDE, MISSING = [f'ENSG{number:011d}' for number in range(1, 10)]
PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'


def view():
    """Supply equal-hop routes, native arcs, a loop, and explicit fixture isolates."""
    return GraphView('unit_path_view', '8' * 64, {
        PPI: LayerGraph(PPI, False, {
            (A, A): .9, (A, B): 0, (B, X): .1, (X, Z): .2,
            (A, C): .8, (C, D): .7, (D, Z): .6}, frozenset((A, B, C, D, X, Z, ISOLATED))),
        COEX: LayerGraph(COEX, False, {(A, OUTSIDE): .1}, frozenset((A, OUTSIDE))),
        BRAIN: LayerGraph(BRAIN, True, {
            (Z, B): .9, (B, A): .2, (Z, C): 0, (C, A): .1, (A, A): .3},
            frozenset((A, B, C, Z, ISOLATED))),
    })


def independent_path(genes, pairs, source, target, directed):
    """Enumerate all simple routes, then apply length and full-sequence order."""
    if source == target:
        return [source]
    candidates = []
    for size in range(len(genes) - 1):
        for middle in permutations(sorted(set(genes) - {source, target}), size):
            route = (source, *middle, target)
            if all((left, right) in pairs or not directed and (right, left) in pairs
                   for left, right in zip(route, route[1:])):
                candidates.append(route)
        if candidates:
            return list(min(candidates))
    return []


class PathRuntimeTests(unittest.TestCase):
    """Keep endpoint order and distinguish a negative result from blocked traversal."""

    def setUp(self):
        self.view = view()
        self.index = PathIndex(self.view)
        self.runtime = PathRuntime(self.view, index=self.index)

    def call(self, genes=(A, Z), layer=PPI, direction='undirected', *, runtime=None, **changes):
        runtime = runtime or self.runtime
        arguments = runtime.arguments(list(genes), layer, direction)
        arguments.update(changes)
        return runtime.shortest_path(**arguments, evidence_id='unit_path')

    def test_exact_four_arguments_retain_order_and_repeated_positions(self):
        args = self.runtime.arguments([Z, A], PPI, 'undirected')
        self.assertEqual(tuple(args), ARGUMENT_KEYS)
        self.assertEqual(self.runtime.validate_arguments(args), (Z, A))
        self.assertEqual(self.runtime.validate_arguments({**args, 'gene_ids': [A, A]}), (A, A))

    def test_argument_copy_preserves_original_input(self):
        raw = [A, B]
        args = self.runtime.arguments(raw, PPI, 'undirected')
        raw.reverse()
        self.assertEqual(args['gene_ids'], [A, B])

    def test_final_shape_is_exact_and_has_no_collection_or_edge_fields(self):
        result = self.call()['result']
        self.assertEqual(tuple(result), ('multiplex_id', 'layer_id', 'directed', 'direction',
            'gene_ids', 'path_found', 'path_gene_ids', 'hop_count', 'missing_gene_ids', 'not_in_layer_gene_ids'))
        self.assertEqual(result, {'multiplex_id': self.view.multiplex_id, 'layer_id': PPI,
            'directed': False, 'direction': 'undirected', 'gene_ids': [A, Z], 'path_found': True,
            'path_gene_ids': [A, B, X, Z], 'hop_count': 3, 'missing_gene_ids': [], 'not_in_layer_gene_ids': []})

    def test_tie_rule_uses_complete_id_sequence_not_weights(self):
        self.assertEqual(self.call()['result']['path_gene_ids'], [A, B, X, Z])

    def test_reversed_undirected_query_recomputes_canonical_tie(self):
        forward = self.call()['result']
        reverse = self.call([Z, A])['result']
        self.assertEqual(reverse['gene_ids'], [Z, A])
        self.assertEqual(reverse['path_gene_ids'], [Z, D, C, A])
        self.assertNotEqual(reverse['path_gene_ids'], list(reversed(forward['path_gene_ids'])))

    def test_zero_weight_is_a_usable_native_edge(self):
        result = self.call([A, B])['result']
        self.assertEqual(result['path_gene_ids'], [A, B])
        self.assertEqual(result['hop_count'], 1)

    def test_recorded_loop_does_not_replace_zero_hops(self):
        result = self.call([A, A])['result']
        self.assertEqual(result['path_gene_ids'], [A])
        self.assertEqual(result['hop_count'], 0)

    def test_zero_hops_needs_no_recorded_loop(self):
        for gene in (B, ISOLATED):
            result = self.call([gene, gene])['result']
            self.assertTrue(result['path_found'])
            self.assertEqual(result['path_gene_ids'], [gene])
            self.assertEqual(result['hop_count'], 0)

    def test_zero_hops_need_no_adjacency_index(self):
        runtime = PathRuntime(self.view)
        self.assertEqual(self.call([A, A], runtime=runtime)['result']['path_gene_ids'], [A])
        self.assertIsNone(runtime.index)

    def test_directed_path_keeps_native_orientation(self):
        result = self.call([Z, A], BRAIN, 'source_to_target')['result']
        self.assertEqual(result['gene_ids'], [Z, A])
        self.assertEqual(result['path_gene_ids'], [Z, B, A])
        self.assertEqual(result['hop_count'], 2)

    def test_reverse_directed_path_is_a_separate_negative_operation(self):
        result = self.call([A, Z], BRAIN, 'source_to_target')['result']
        self.assertFalse(result['path_found'])
        self.assertEqual(result['path_gene_ids'], [])
        self.assertIsNone(result['hop_count'])

    def test_disconnected_present_endpoints_are_a_completed_negative(self):
        response = self.call([A, ISOLATED])
        self.assertEqual(response['status'], 'resolved')
        self.assertFalse(response['result']['path_found'])
        self.assertEqual(response['result']['path_gene_ids'], [])
        self.assertIsNone(response['result']['hop_count'])

    def test_layer_absence_resolves_negative_without_search(self):
        runtime = PathRuntime(self.view)
        response = self.call([OUTSIDE, A], runtime=runtime)
        self.assertEqual(response['status'], 'resolved')
        self.assertEqual(response['result']['not_in_layer_gene_ids'], [OUTSIDE])
        self.assertFalse(response['result']['path_found'])
        self.assertIsNone(runtime.index)

    def test_missing_endpoint_leaves_all_path_fields_null(self):
        response = self.call([MISSING, A])
        self.assertEqual(response['status'], 'not_found')
        for field in ('path_found', 'path_gene_ids', 'hop_count'):
            self.assertIsNone(response['result'][field])
        self.assertEqual(response['result']['missing_gene_ids'], [MISSING])

    def test_both_memberships_are_checked_when_one_endpoint_is_missing(self):
        result = self.call([MISSING, OUTSIDE])['result']
        self.assertEqual(result['missing_gene_ids'], [MISSING])
        self.assertEqual(result['not_in_layer_gene_ids'], [OUTSIDE])

    def test_repeated_excluded_and_missing_endpoints_keep_two_positions(self):
        for gene, category, status in ((OUTSIDE, 'not_in_layer_gene_ids', 'resolved'),
                                       (MISSING, 'missing_gene_ids', 'not_found')):
            response = self.call([gene, gene])
            self.assertEqual(response['status'], status)
            self.assertEqual(response['result']['gene_ids'], [gene, gene])
            self.assertEqual(response['result'][category], [gene])

    def test_success_provenance_binds_native_scope_version_and_acquisition(self):
        self.assertEqual(self.call()['provenance'], {'registry': [], 'graph': [{
            'multiplex_id': self.view.multiplex_id, 'graph_version': self.view.graph_version,
            'layer_scope': {'kind': 'layers', 'layer_ids': [PPI], 'layer_family': None}, 'evidence_id': 'unit_path'}]})

    def test_invalid_endpoint_lists_have_exact_pair_diagnostic(self):
        for genes in (None, [], [A], [A, B, C], 'A', [A, None], [A, 'bad'], [A, 3]):
            with self.subTest(genes=genes):
                response = self.runtime.shortest_path(self.view.multiplex_id, PPI, genes, 'undirected', evidence_id='bad')
                self.assertEqual(response['status'], 'invalid_input')
                self.assertEqual(response['result']['issue_code'], 'INVALID_PAIR')
                self.assertEqual(response['result']['arguments']['gene_ids'], genes)
                self.assertEqual(response['provenance'], {'registry': [], 'graph': []})

    def test_direction_diagnostics_distinguish_missing_invalid_and_unavailable(self):
        cases = [(BRAIN, None, 'ambiguous', 'AMBIGUOUS_PATH_DIRECTION'),
                 (PPI, None, 'invalid_input', 'INVALID_DIRECTION'),
                 (PPI, 'weak', 'invalid_input', 'INVALID_DIRECTION'),
                 (PPI, {}, 'invalid_input', 'INVALID_DIRECTION'),
                 (PPI, 'source_to_target', 'invalid_input', 'DIRECTION_UNAVAILABLE'),
                 (BRAIN, 'undirected', 'invalid_input', 'DIRECTION_UNAVAILABLE')]
        for layer, direction, status, code in cases:
            response = self.call(layer=layer, direction=direction)
            self.assertEqual((response['status'], response['result']['issue_code']), (status, code))

    def test_scope_diagnostics_keep_rejected_arguments(self):
        for changes, code in (({'layer_id': None}, 'MISSING_SCOPE'), ({'multiplex_id': None}, 'MISSING_SCOPE'),
                              ({'layer_id': 'unknown'}, 'UNKNOWN_LAYER'), ({'multiplex_id': 'unknown'}, 'UNKNOWN_MULTIPLEX')):
            response = self.call(**changes)
            self.assertEqual(response['result']['issue_code'], code)
            self.assertEqual(response['result']['arguments'], {**self.runtime.arguments([A, Z], PPI, 'undirected'), **changes})

    def test_extra_selection_keys_are_rejected(self):
        with self.assertRaises(SelectionError) as caught:
            self.runtime.validate_arguments({**self.runtime.arguments([A, B], PPI, 'undirected'), 'top_k': 1})
        self.assertEqual(caught.exception.issue_code, 'INVALID_SELECTION')

    def test_validation_and_result_helper_need_no_hidden_inventories(self):
        class MetadataView:
            multiplex_id = 'metadata_only'
            layers = {PPI: type('Layer', (), {'directed': False})()}

            @property
            def multiplex_nodes(self):
                raise AssertionError('result helpers must use supplied membership only')

        metadata = MetadataView()
        args = PathRuntime(metadata).arguments([A, A], PPI, 'undirected')
        self.assertEqual(PathRuntime(metadata).validate_arguments(args), (A, A))
        self.assertEqual(path_result(metadata, args, [], [], [A])['hop_count'], 0)

    def test_result_helper_rejects_contradictory_categories_and_routes(self):
        args = self.runtime.arguments([A, B], PPI, 'undirected')
        for missing, outside, route in (([A], [A], None), ([MISSING], [], None),
                                        ([A], [], []), ([], [], None), ([], [A], [A, B]),
                                        ([], [], [B, A]), ([], [], [A, C, C, B])):
            with self.subTest(missing=missing, outside=outside, route=route), self.assertRaises(ValueError):
                path_result(self.view, args, missing, outside, route)
        with self.assertRaises(ValueError):
            path_result(self.view, {**args, 'gene_ids': [A, A]}, [], [], [])

    def test_runtime_reuses_matching_native_index_and_rejects_foreign_views(self):
        native = NeighborIndex(self.view)
        self.assertIs(PathIndex(self.view, native).native_index, native)
        with self.assertRaises(ValueError):
            PathIndex(view(), native)
        with self.assertRaises(ValueError):
            PathRuntime(view(), index=self.index)

    def test_success_requires_an_evidence_id(self):
        for evidence_id in ('', ' ', None):
            with self.assertRaises(ValueError):
                self.runtime.shortest_path(**self.runtime.arguments([A, B], PPI, 'undirected'), evidence_id=evidence_id)

    def test_full_distance_domain_and_unreachable_source(self):
        self.assertEqual(self.index.distances(BRAIN, Z), {Z: 0, B: 1, C: 1, A: 2})
        self.assertEqual(self.index.distances(PPI, MISSING), {})
        self.assertEqual(self.index.path(PPI, MISSING, MISSING), [])

    def test_zero_and_one_hop_paths_need_no_full_search_tree(self):
        class DirectIndex(PathIndex):
            def _tree(self, layer_id, source):
                raise AssertionError('a direct native edge already proves one exact hop')

        index = DirectIndex(self.view)
        self.assertEqual(index.path(PPI, A, A), [A])
        self.assertEqual(index.path(PPI, A, B), [A, B])
        self.assertEqual(index.path(BRAIN, Z, B), [Z, B])
        self.assertEqual(index.path(PPI, MISSING, A), [])

    def test_long_native_chain_has_no_hop_or_recursion_limit(self):
        genes = [f'ENSG{number:011d}' for number in range(10000, 15000)]
        layer = LayerGraph(BRAIN, True, dict.fromkeys(zip(genes, genes[1:]), 0), frozenset(genes))
        source = GraphView('long_path', '4' * 64, {BRAIN: layer})
        runtime = PathRuntime(source)
        result = runtime.shortest_path(**runtime.arguments([genes[0], genes[-1]], BRAIN, 'source_to_target'), evidence_id='long')['result']
        self.assertEqual(result['path_gene_ids'], genes)
        self.assertEqual(result['hop_count'], len(genes) - 1)

    def test_exhaustive_small_graphs_match_independent_simple_route_enumeration(self):
        for directed, genes in ((True, (A, B, C)), (False, (A, B, C, D))):
            pairs = list(permutations(genes, 2) if directed else combinations(genes, 2))
            for mask in range(1 << len(pairs)):
                edges = {pair: position % 2 for position, pair in enumerate(pairs) if mask & (1 << position)}
                source = GraphView('small_path', '6' * 64, {PPI: LayerGraph(PPI, directed, edges, frozenset(genes))})
                index = PathIndex(source)
                for left in genes:
                    for right in genes:
                        self.assertEqual(index.path(PPI, left, right), independent_path(genes, edges, left, right, directed),
                                         (directed, mask, left, right))


if __name__ == '__main__':
    unittest.main()
