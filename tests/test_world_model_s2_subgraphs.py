"""Check native induced projections, complete partitions, and exact edge pages."""

import unittest

from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import ARGUMENT_KEYS, SubgraphIndex, SubgraphRuntime


A, B, C, ISOLATED, OUTSIDE, MISSING = [f'ENSG{number:011d}' for number in range(1, 7)]
PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'


def view(version='9' * 64):
    """Supply loops, reciprocal arcs, zero weight, and one fixture isolate."""
    return GraphView('unit_subgraph_view', version, {
        PPI: LayerGraph(PPI, False, {(A, A): .2, (A, B): .1, (A, C): .9, (B, C): 0},
                        frozenset((A, B, C, ISOLATED))),
        COEX: LayerGraph(COEX, False, {(OUTSIDE, A): .5}, frozenset((A, OUTSIDE))),
        BRAIN: LayerGraph(BRAIN, True, {(A, B): .7, (B, A): .4, (C, A): .3, (B, B): .8},
                          frozenset((A, B, C, ISOLATED))),
    })


class SubgraphRuntimeTests(unittest.TestCase):
    """Keep membership coverage separate from native edge completeness."""

    def setUp(self):
        self.view = view()
        self.index = SubgraphIndex(self.view)
        self.runtime = SubgraphRuntime(self.view, index=self.index)

    def call(self, genes=(C, A, B), layer=PPI, *, mode='complete', top_k=None, cursor=None, runtime=None):
        runtime = runtime or self.runtime
        return runtime.induce_subgraph(**runtime.arguments(list(genes), layer, mode, top_k, cursor), evidence_id='unit_subgraph')

    def test_exact_six_arguments_keep_original_input_order(self):
        arguments = self.runtime.arguments([C, A, C, B], PPI)
        self.assertEqual(tuple(arguments), ARGUMENT_KEYS)
        self.assertEqual(arguments['gene_ids'], [C, A, C, B])
        self.assertEqual(self.runtime.validate_arguments(arguments), (C, A, B))

    def test_membership_partition_uses_full_inventories(self):
        result = self.call([MISSING, OUTSIDE, ISOLATED, C, A, B])['result']
        self.assertEqual(result['layer_gene_ids'], [A, B, C, ISOLATED])
        self.assertEqual(result['layer_gene_count'], 4)
        self.assertEqual(result['not_in_layer_gene_ids'], [OUTSIDE])
        self.assertEqual(result['missing_gene_ids'], [MISSING])
        self.assertEqual(result['coverage'], {'operation': 'layer_membership',
            'requested_gene_ids': [MISSING, OUTSIDE, ISOLATED, C, A, B],
            'checked_gene_ids': [MISSING, OUTSIDE, ISOLATED, C, A, B], 'unchecked_gene_ids': []})

    def test_undirected_edge_order_is_lexicographic_not_weight_rank(self):
        result = self.call()['result']
        self.assertEqual(result['edges'], [
            {'gene_ids': [A, A], 'weight': .2}, {'gene_ids': [A, B], 'weight': .1},
            {'gene_ids': [A, C], 'weight': .9}, {'gene_ids': [B, C], 'weight': 0}])
        self.assertEqual(result['edge_count'], 4)
        self.assertFalse(result['directed'])

    def test_directed_edges_keep_native_source_target_and_reciprocal_weights(self):
        result = self.call([C, B, A], BRAIN)['result']
        self.assertTrue(result['directed'])
        self.assertEqual(result['edges'], [
            {'gene_ids': [A, B], 'weight': .7}, {'gene_ids': [B, A], 'weight': .4},
            {'gene_ids': [B, B], 'weight': .8}, {'gene_ids': [C, A], 'weight': .3}])

    def test_singleton_self_loop_occurs_once(self):
        result = self.call([B], BRAIN)['result']
        self.assertEqual(result['edges'], [{'gene_ids': [B, B], 'weight': .8}])
        self.assertEqual(result['edge_count'], 1)

    def test_selected_set_isolate_stays_in_the_projected_nodes(self):
        result = self.call([ISOLATED, A])['result']
        self.assertEqual(result['layer_gene_ids'], [A, ISOLATED])
        self.assertEqual(result['edges'], [{'gene_ids': [A, A], 'weight': .2}])

    def test_nonempty_members_can_have_zero_induced_edges(self):
        result = self.call([C, ISOLATED])['result']
        self.assertEqual(result['layer_gene_count'], 2)
        self.assertEqual(result['edges'], [])
        self.assertEqual(result['collection']['total_count'], 0)

    def test_all_excluded_inputs_resolve_empty_projection(self):
        response = self.call([OUTSIDE, MISSING])
        self.assertEqual(response['status'], 'resolved')
        self.assertEqual(response['result']['layer_gene_ids'], [])
        self.assertEqual(response['result']['not_in_layer_gene_ids'], [OUTSIDE])
        self.assertEqual(response['result']['missing_gene_ids'], [MISSING])
        self.assertTrue(response['result']['collection']['complete'])

    def test_explicit_empty_input_needs_only_manifest_metadata(self):
        class MetadataView:
            multiplex_id = 'unit_metadata_only'
            graph_version = '8' * 64
            layers = {PPI: type('Layer', (), {'directed': False})()}

            @property
            def multiplex_nodes(self):
                raise AssertionError('empty input must not consult inventories')

        runtime = SubgraphRuntime(MetadataView())
        response = self.call([], runtime=runtime)
        self.assertEqual(response['status'], 'resolved')
        self.assertEqual(response['provenance'], {'registry': [], 'graph': []})
        self.assertEqual(response['result']['coverage']['checked_gene_ids'], [])
        self.assertIsNone(runtime.index)

    def test_duplicate_inputs_are_classified_once_in_first_mention_order(self):
        result = self.call([C, A, C, MISSING, A, MISSING])['result']
        self.assertEqual(result['coverage']['requested_gene_ids'], [C, A, MISSING])
        self.assertEqual(result['layer_gene_ids'], [A, C])
        self.assertEqual(result['missing_gene_ids'], [MISSING])

    def test_projection_does_not_filter_registry_independent_ids(self):
        self.assertEqual(self.call([C])['result']['layer_gene_ids'], [C])

    def test_subgraphs_on_disjoint_batches_do_not_include_cross_batch_edges(self):
        singles = [self.call([gene], BRAIN)['result']['edges'] for gene in (A, C)]
        self.assertEqual(singles, [[], []])
        self.assertEqual(self.call([A, C], BRAIN)['result']['edges'], [{'gene_ids': [C, A], 'weight': .3}])

    def test_top_k_is_canonical_edge_preview(self):
        result = self.call(mode='top_k', top_k=1)['result']
        self.assertEqual(result['edges'], [{'gene_ids': [A, A], 'weight': .2}])
        self.assertEqual(result['collection'], {'mode': 'top_k', 'returned_count': 1,
            'total_count': 4, 'complete': False, 'next_cursor': None})

    def test_top_k_at_or_above_total_can_complete_the_graph(self):
        for top_k in (4, 9):
            with self.subTest(top_k=top_k):
                result = self.call(mode='top_k', top_k=top_k)['result']
                self.assertEqual(result['edge_count'], 4)
                self.assertTrue(result['collection']['complete'])

    def test_pages_keep_full_partition_and_last_page_is_not_full_graph(self):
        runtime = SubgraphRuntime(self.view, index=self.index, page_size=2)
        first = self.call([C, OUTSIDE, A, B, MISSING], runtime=runtime)['result']
        last = self.call([C, OUTSIDE, A, B, MISSING], runtime=runtime,
                         cursor=first['collection']['next_cursor'])['result']
        self.assertEqual(first['coverage'], last['coverage'])
        for key in ('layer_gene_ids', 'not_in_layer_gene_ids', 'missing_gene_ids'):
            self.assertEqual(first[key], last[key])
        self.assertEqual(first['edge_count'], last['edge_count'])
        self.assertFalse(last['collection']['complete'])
        self.assertIsNone(last['collection']['next_cursor'])

    def test_unknown_totals_remain_null_on_nonempty_pages(self):
        runtime = SubgraphRuntime(self.view, index=self.index, page_size=2, disclose_total=False)
        first = self.call(runtime=runtime)['result']
        last = self.call(runtime=runtime, cursor=first['collection']['next_cursor'])['result']
        self.assertIsNone(first['collection']['total_count'])
        self.assertIsNone(last['collection']['total_count'])
        self.assertEqual(self.call([ISOLATED], runtime=runtime)['result']['collection']['total_count'], 0)

    def test_empty_intermediate_page_has_a_real_continuation(self):
        runtime = SubgraphRuntime(self.view, index=self.index, page_size=2, empty_page_at=1)
        first = self.call(runtime=runtime)['result']
        empty = self.call(runtime=runtime, cursor=first['collection']['next_cursor'])['result']
        self.assertEqual(empty['edges'], [])
        self.assertFalse(empty['collection']['complete'])
        self.assertIsNotNone(empty['collection']['next_cursor'])
        last = self.call(runtime=runtime, cursor=empty['collection']['next_cursor'])['result']
        self.assertEqual(last['edges'], self.call()['result']['edges'][2:])

    def test_overlapping_pages_preserve_identical_native_rows(self):
        runtime = SubgraphRuntime(self.view, index=self.index, page_size=3, overlap=1)
        first = self.call(runtime=runtime)['result']
        second = self.call(runtime=runtime, cursor=first['collection']['next_cursor'])['result']
        self.assertEqual(first['edges'][-1], second['edges'][0])
        self.assertEqual(len(first['edges']) + len(second['edges']) - 1, 4)

    def test_cursor_is_bound_to_input_order_layer_mode_bound_version_and_transport(self):
        runtime = SubgraphRuntime(self.view, index=self.index, page_size=1)
        first = self.call(runtime=runtime)['result']
        cursor = first['collection']['next_cursor']
        attempts = [dict(genes=[A, B, C]), dict(layer=BRAIN), dict(mode='top_k', top_k=2)]
        for change in attempts:
            with self.subTest(change=change):
                response = self.call(runtime=runtime, cursor=cursor, **change)
                self.assertEqual(response['result']['issue_code'], 'INVALID_CURSOR')
                self.assertEqual(response['provenance']['graph'], [])
        for other in (SubgraphRuntime(view('7' * 64), page_size=1),
                      SubgraphRuntime(self.view, index=self.index, page_size=2)):
            self.assertEqual(self.call(runtime=other, cursor=cursor)['result']['issue_code'], 'INVALID_CURSOR')

    def test_validation_is_source_blind_and_runtime_index_is_lazy(self):
        class MetadataView:
            multiplex_id = 'unit_metadata_only'
            graph_version = '8' * 64
            layers = {PPI: type('Layer', (), {'directed': False})()}

            @property
            def multiplex_nodes(self):
                raise AssertionError('validation must not consult inventories')

        runtime = SubgraphRuntime(MetadataView())
        self.assertEqual(runtime.validate_arguments(runtime.arguments([A], PPI)), (A,))
        self.assertIsNone(runtime.index)

    def test_invalid_scope_and_input_are_atomic_and_unattributed(self):
        changes = [({'multiplex_id': None}, 'MISSING_SCOPE'), ({'layer_id': None}, 'MISSING_SCOPE'),
            ({'multiplex_id': 'unknown'}, 'UNKNOWN_MULTIPLEX'), ({'layer_id': 'unknown'}, 'UNKNOWN_LAYER'),
            ({'gene_ids': None}, 'MISSING_GENES'), ({'gene_ids': A}, 'INVALID_GENE_LIST'),
            ({'gene_ids': [A, {}]}, 'INVALID_GENE_LIST'), ({'gene_ids': ['bad']}, 'INVALID_GENE_LIST')]
        for change, code in changes:
            with self.subTest(change=change):
                arguments = {**self.runtime.arguments([A], PPI), **change}
                response = self.runtime.induce_subgraph(**arguments, evidence_id='unused')
                self.assertEqual(response['result'], {'issue_code': code, 'arguments': arguments})
                self.assertEqual(response['provenance'], {'registry': [], 'graph': []})

    def test_invalid_mode_and_bounds_are_rejected_without_coercion(self):
        for mode, top_k in (('bad', None), ('complete', 1), ('top_k', None), ('top_k', 0),
                            ('top_k', -1), ('top_k', True), ('top_k', 1.0)):
            with self.subTest(mode=mode, top_k=top_k):
                self.assertEqual(self.call(mode=mode, top_k=top_k)['result']['issue_code'], 'INVALID_SELECTION')

    def test_invalid_cursor_shapes_and_unknown_tokens_are_rejected(self):
        for cursor in ('', 0, True, [], {}, 'unknown'):
            with self.subTest(cursor=cursor):
                self.assertEqual(self.call(cursor=cursor)['result']['issue_code'], 'INVALID_CURSOR')

    def test_input_copy_and_result_copy_do_not_mutate_the_source(self):
        genes = [A, B]
        arguments = self.runtime.arguments(genes, PPI)
        genes.append(C)
        self.assertEqual(arguments['gene_ids'], [A, B])
        result = self.call()['result']
        result['edges'][0]['weight'] = 55
        self.assertEqual(self.call()['result']['edges'][0]['weight'], .2)

    def test_successful_nonempty_acquisition_requires_an_evidence_id(self):
        with self.assertRaises(ValueError):
            self.runtime.induce_subgraph(**self.runtime.arguments([A], PPI), evidence_id=' ')

    def test_transport_settings_and_shared_indexes_are_checked(self):
        for kwargs in ({'page_size': 0}, {'page_size': True}, {'disclose_total': 1}, {'overlap': -1},
                       {'overlap': 1}, {'page_size': 2, 'overlap': 2}, {'empty_page_at': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SubgraphRuntime(self.view, **kwargs)
        other = view()
        with self.assertRaises(ValueError):
            SubgraphIndex(other, self.index.native_index)
        with self.assertRaises(ValueError):
            SubgraphRuntime(other, index=self.index)


if __name__ == '__main__':
    unittest.main()
