"""Check exact component proofs, native evidence, and source independence."""

from copy import deepcopy
from itertools import combinations, product
import unittest

from runtime.world_model.s2_10_evidence import collect
from runtime.world_model.s2_multiplex import EvidenceError, SelectionError
from runtime.world_model.s2_neighbors import NeighborRuntime
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphRuntime


A, B, C, D, E, F, OUTSIDE, MISSING = [f'ENSG{number:011d}' for number in range(1, 9)]
PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'


def observation(operation, arguments, response):
    """Keep the acquisition beside its unchanged attributed response."""
    return {'operation': operation, 'arguments': deepcopy(arguments), **deepcopy(response)}


def naive_partition(genes, pairs, kind):
    """Compute small independent reference partitions by reachability closure."""
    reach = {(gene, gene) for gene in genes} | set(pairs)
    if kind != 'strong':
        reach |= {(right, left) for left, right in pairs}
    for bridge in genes:
        reach |= {(left, right) for left in genes for right in genes
                  if (left, bridge) in reach and (bridge, right) in reach}
    groups, remaining = [], set(genes)
    while remaining:
        first = min(remaining)
        group = sorted(gene for gene in remaining if (first, gene) in reach and (gene, first) in reach)
        groups.append(group)
        remaining.difference_update(group)
    return groups


class ComponentEvidenceTests(unittest.TestCase):
    """Use native weights, exclusions, loops, directed arcs, and partial pages."""

    def setUp(self):
        self.view = GraphView('unit_component_evidence', 'a' * 64, {
            PPI: LayerGraph(PPI, False, {(A, A): .9, (A, B): .8, (B, C): .7,
                (B, D): .5, (C, D): 0, (D, F): .6}, frozenset((A, B, C, D, E, F))),
            COEX: LayerGraph(COEX, False, {(A, OUTSIDE): .2}, frozenset((A, OUTSIDE))),
            BRAIN: LayerGraph(BRAIN, True, {(A, B): .9, (B, A): .8, (A, A): .7,
                (C, B): 0, (C, D): .5}, frozenset((A, B, C, D, E))),
        })
        self.serial = 0

    def args(self, genes=(A, B, C), layer=PPI, kind=None):
        return {'multiplex_id': self.view.multiplex_id, 'layer_id': layer,
                'gene_ids': list(genes), 'connectivity_kind': kind}

    def subgraph(self, genes=(A, B, C), layer=PPI, *, mode='complete', top_k=None, runtime=None, cursor=None):
        runtime = runtime or SubgraphRuntime(self.view)
        arguments = {**runtime.arguments(list(genes), layer, mode, top_k), 'cursor': cursor}
        self.serial += 1
        return observation('induce_subgraph', arguments,
            runtime.induce_subgraph(**arguments, evidence_id=f'sub_{self.serial}'))

    def native(self, gene, layer=PPI, direction='undirected', *, mode='complete', top_k=None):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_id': gene,
            'direction': direction, 'mode': mode, 'top_k': top_k, 'cursor': None}
        self.serial += 1
        return observation('get_neighbors', arguments,
            NeighborRuntime(self.view).get_neighbors(**arguments, evidence_id=f'native_{self.serial}'))

    def nodes(self, genes, layer=PPI):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': list(genes)}
        self.serial += 1
        return observation('get_nodes_by_layer', arguments,
            NodesByLayerRuntime(self.view).get_nodes_by_layer(**arguments, evidence_id=f'nodes_{self.serial}'))

    def edge(self, pair, layer=PPI):
        arguments = {'multiplex_id': self.view.multiplex_id, 'layer_id': layer, 'gene_ids': list(pair)}
        self.serial += 1
        return observation('get_edge', arguments, self.view.get_edge(**arguments, evidence_id=f'edge_{self.serial}'))

    def test_default_requests_full_native_graph_with_first_mention_order(self):
        proof = collect(self.view, self.args((C, A, B, A)), [])
        self.assertEqual(proof.next_tool, 'induce_subgraph')
        self.assertEqual(proof.next_arguments, SubgraphRuntime(self.view).arguments([C, A, B], PPI))
        self.assertNotIn('connectivity_kind', proof.next_arguments)

    def test_explicit_empty_has_zero_groups_no_calls_and_empty_provenance(self):
        proof = collect(self.view, self.args(()), [])
        self.assertEqual(proof.result['components'], [])
        self.assertFalse(proof.result['single_component'])
        self.assertIsNone(proof.next_tool)
        self.assertEqual(proof.provenance, {'registry': [], 'graph': []})

    def test_directed_empty_still_requires_declared_connectivity(self):
        with self.assertRaises(SelectionError) as caught:
            collect(self.view, self.args((), BRAIN), [])
        self.assertEqual(caught.exception.issue_code, 'CONNECTIVITY_KIND_REQUIRED')

    def test_singleton_membership_needs_no_loop_proof_for_each_kind(self):
        for layer, kind in ((PPI, None), (BRAIN, 'weak'), (BRAIN, 'strong')):
            with self.subTest(kind=kind):
                saved = self.nodes((E,), layer)
                proof = collect(self.view, self.args((E,), layer, kind), [saved])
                self.assertEqual(proof.result['components'], [[E]])
                self.assertTrue(proof.result['single_component'])
                self.assertEqual(proof.observations, (saved,))

    def test_new_singleton_acquires_membership_only(self):
        proof = collect(self.view, self.args((E,)), [])
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments['gene_ids'], [E])

    def test_all_excluded_membership_proves_empty_projection(self):
        saved = self.nodes((OUTSIDE, MISSING))
        proof = collect(self.view, self.args((OUTSIDE, MISSING)), [saved])
        self.assertEqual(proof.result['component_count'], 0)
        self.assertEqual(proof.result['not_in_layer_gene_ids'], [OUTSIDE])
        self.assertEqual(proof.result['missing_gene_ids'], [MISSING])

    def test_connected_projection_with_exclusions_has_false_all_input_boolean(self):
        saved = self.subgraph((A, B, C, OUTSIDE, MISSING))
        proof = collect(self.view, self.args((A, B, C, OUTSIDE, MISSING)), [saved])
        self.assertEqual(proof.result['components'], [[A, B, C]])
        self.assertFalse(proof.result['single_component'])

    def test_one_exclusion_does_not_supply_remaining_partition(self):
        proof = collect(self.view, self.args((A, B, C, OUTSIDE)), [self.nodes((OUTSIDE,))])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [A, B, C])

    def test_spanning_partial_prefix_proves_one_undirected_component(self):
        saved = self.subgraph((A, B, C, D), mode='top_k', top_k=4)
        self.assertFalse(saved['result']['collection']['complete'])
        original = deepcopy(saved)
        proof = collect(self.view, self.args((D, C, B, A)), [saved])
        self.assertEqual(proof.result['components'], [[A, B, C, D]])
        self.assertEqual(proof.result['collection']['total_count'], 1)
        self.assertEqual(saved, original)
        self.assertNotIn('edges', proof.result)

    def test_disconnected_preview_requires_fresh_complete_graph(self):
        saved = self.subgraph((A, B, C, D), mode='top_k', top_k=2)
        proof = collect(self.view, self.args((A, B, C, D)), [saved])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_arguments['mode'], 'complete')
        self.assertIsNone(proof.next_arguments['cursor'])

    def test_weak_witness_does_not_prove_strong_partition(self):
        saved = self.subgraph((A, B, C), BRAIN, mode='top_k', top_k=2)
        weak = collect(self.view, self.args((A, B, C), BRAIN, 'weak'), [saved])
        strong = collect(self.view, self.args((A, B, C), BRAIN, 'strong'), [saved])
        self.assertIsNone(weak.result)
        self.assertIsNone(strong.result)
        arcs = [self.edge((A, B), BRAIN), self.edge((C, B), BRAIN)]
        weak = collect(self.view, self.args((A, B, C), BRAIN, 'weak'), arcs)
        strong = collect(self.view, self.args((A, B, C), BRAIN, 'strong'), arcs)
        self.assertEqual(weak.result['components'], [[A, B, C]])
        self.assertIsNone(strong.result)

    def test_strong_positive_witness_can_finish_without_unknown_arcs_or_loops(self):
        saved = [self.edge((A, B), BRAIN), self.edge((B, A), BRAIN)]
        proof = collect(self.view, self.args((B, A), BRAIN, 'strong'), saved)
        self.assertEqual(proof.result['components'], [[A, B]])
        self.assertEqual(proof.result['coverage']['requested_gene_ids'], [B, A])

    def test_complete_native_graph_reuses_arcs_for_both_directed_kinds(self):
        saved = self.subgraph((A, B, C, D), BRAIN)
        weak = collect(self.view, self.args((A, B, C, D), BRAIN, 'weak'), [saved])
        strong = collect(self.view, self.args((A, B, C, D), BRAIN, 'strong'), [saved])
        self.assertEqual(weak.result['components'], [[A, B, C, D]])
        self.assertEqual(strong.result['components'], [[A, B], [C], [D]])

    def test_restricting_a_complete_graph_recomputes_after_bridge_deletion(self):
        saved = self.subgraph((A, B, C))
        proof = collect(self.view, self.args((C, A)), [saved])
        self.assertEqual(proof.result['components'], [[A], [C]])

    def test_outside_route_supplies_membership_but_does_not_join_selected_nodes(self):
        saved = [self.edge((A, B)), self.edge((B, C))]
        proof = collect(self.view, self.args((A, C)), saved)
        self.assertIsNone(proof.result)
        fresh = self.subgraph((A, C))
        proof = collect(self.view, self.args((A, C)), [*saved, fresh])
        self.assertEqual(proof.result['components'], [[A], [C]])
        self.assertEqual(proof.observations, (*saved, fresh))

    def test_two_separate_complete_batches_do_not_prove_cross_batch_absence(self):
        saved = [self.subgraph((A,)), self.subgraph((C,))]
        self.assertIsNone(collect(self.view, self.args((A, C)), saved).result)

    def test_negative_domains_can_prove_multiple_groups_without_complete_loops(self):
        saved = [self.native(A), self.native(E), self.nodes((A, B, E))]
        proof = collect(self.view, self.args((E, B, A)), saved)
        self.assertEqual(proof.result['components'], [[A, B], [E]])

    def test_outgoing_and_incoming_lists_preserve_native_direction(self):
        for direction in ('out', 'in'):
            saved = [self.native(gene, BRAIN, direction) for gene in (A, B, C, D)]
            proof = collect(self.view, self.args((A, B, C, D), BRAIN, 'strong'), saved)
            self.assertEqual(proof.result['components'], [[A, B], [C], [D]])

    def test_loop_and_zero_weight_have_the_declared_unweighted_meaning(self):
        loop = self.subgraph((A,))
        self.assertEqual(collect(self.view, self.args((A,)), [loop]).result['components'], [[A]])
        zero = self.edge((C, D))
        self.assertEqual(zero['result']['weight'], 0)
        self.assertEqual(collect(self.view, self.args((C, D)), [zero]).result['components'], [[C, D]])

    def test_pagination_stops_once_positive_edges_span_every_member(self):
        runtime = SubgraphRuntime(self.view, page_size=4)
        first = self.subgraph((A, B, C, D), runtime=runtime)
        self.assertIsNotNone(first['result']['collection']['next_cursor'])
        proof = collect(self.view, self.args((A, B, C, D)), [first])
        self.assertEqual(proof.result['components'], [[A, B, C, D]])
        self.assertIsNone(proof.next_arguments)
        self.assertFalse(first['result']['collection']['complete'])

    def test_required_next_page_keeps_exact_order_mode_and_cursor(self):
        runtime = SubgraphRuntime(self.view, page_size=2)
        first = self.subgraph((D, C, B, A), runtime=runtime)
        proof = collect(self.view, self.args((D, C, B, A)), [first])
        self.assertEqual(proof.next_arguments, {**first['arguments'], 'cursor': first['result']['collection']['next_cursor']})
        self.assertEqual(proof.coverage['unchecked_gene_ids'], [])

    def test_complete_graph_domain_uses_no_quadratic_candidate_scan(self):
        saved = self.subgraph((A, C, E))
        self.assertEqual(collect(self.view, self.args((A, C, E)), [saved]).result['component_count'], 3)

    def test_membership_only_remainder_keeps_full_outer_coverage(self):
        saved = [self.native(A), self.native(B), self.native(C), self.native(D), self.native(E), self.native(F)]
        proof = collect(self.view, self.args((A, B, C, D, E, F, MISSING)), saved)
        self.assertIsNone(proof.result)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments['gene_ids'], [MISSING])
        self.assertEqual(proof.coverage['checked_gene_ids'], [A, B, C, D, E, F])

    def test_membership_first_policy_selects_classification_without_invented_facts(self):
        proof = collect(self.view, self.args((A, B, MISSING)), [], membership_first=True)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.coverage['checked_gene_ids'], [])
        saved = self.nodes((A, B, MISSING))
        proof = collect(self.view, self.args((A, B, MISSING)), [saved], membership_first=True)
        self.assertEqual(proof.next_tool, 'induce_subgraph')

    def test_explicit_gate_precedes_graph_acquisition_and_preserves_failure_partition(self):
        arguments = self.args((A, OUTSIDE, A))
        proof = collect(self.view, arguments, [], require_all_in_layer=True)
        self.assertEqual(proof.next_tool, 'get_nodes_by_layer')
        self.assertEqual(proof.next_arguments['gene_ids'], [A, OUTSIDE])
        saved = self.nodes((A, OUTSIDE))
        proof = collect(self.view, arguments, [saved], require_all_in_layer=True)
        self.assertEqual(proof.issue_code, 'MEMBERSHIP_PRECONDITION_UNSATISFIED')
        self.assertEqual(proof.observations, (saved,))
        self.assertIsNone(proof.next_tool)

    def test_gate_and_acquisition_policy_require_explicit_booleans(self):
        for kwargs in ({'require_all_in_layer': 1}, {'membership_first': 'yes'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collect(self.view, self.args(), [], **kwargs)

    def test_duplicate_and_stale_observations_keep_original_evidence_identity(self):
        saved = self.subgraph()
        proof = collect(self.view, self.args(), [saved, deepcopy(saved)])
        self.assertEqual(proof.observations, (saved,))
        stale = deepcopy(saved)
        stale['provenance']['graph'][0]['graph_version'] = '0' * 64
        proof = collect(self.view, self.args(), [stale])
        self.assertIsNone(proof.result)
        self.assertEqual(proof.observations, ())

    def test_other_layer_graph_is_validated_but_not_attributed_to_current_partition(self):
        other = self.subgraph((A, B, C), COEX)
        current = self.subgraph()
        proof = collect(self.view, self.args(), [other, current])
        self.assertEqual(proof.observations, (current,))

    def test_conflicts_are_rejected_even_after_a_connected_witness_is_known(self):
        connected = [self.edge((A, B)), self.edge((B, C))]
        wrong = self.edge((A, B))
        wrong['result']['weight'] = .1
        negative = self.edge((A, B))
        negative['result'].update(edge_exists=False, weight=None)
        reused = deepcopy(connected[0])
        reused['result']['weight'] = .2
        for extra in (wrong, negative, reused):
            with self.subTest(extra=extra), self.assertRaises(EvidenceError):
                collect(self.view, self.args(), [*connected, extra])

    def test_complete_native_absence_conflicts_with_a_later_positive(self):
        saved = self.native(A)
        saved['result']['neighbors'] = [row for row in saved['result']['neighbors'] if row['gene_id'] != B]
        count = len(saved['result']['neighbors'])
        saved['result']['collection'].update(returned_count=count, total_count=count)
        with self.assertRaises(EvidenceError):
            collect(self.view, self.args((A, B)), [saved, self.edge((A, B))])

    def test_direct_edge_exclusions_and_global_absence_are_classifications(self):
        excluded = self.edge((OUTSIDE, MISSING))
        proof = collect(self.view, self.args((OUTSIDE, MISSING)), [excluded])
        self.assertEqual(proof.result['components'], [])
        self.assertEqual(proof.result['missing_gene_ids'], [MISSING])

    def test_malformed_direct_edge_schema_and_membership_are_faults(self):
        variants = []
        missing = self.edge((A, B))
        missing['result']['missing_gene_ids'] = [A]
        variants.append(missing)
        malformed = self.edge((A, B))
        malformed['result']['gene_ids'].reverse()
        variants.append(malformed)
        for saved in variants:
            with self.subTest(saved=saved), self.assertRaises(EvidenceError):
                collect(self.view, self.args(), [saved])

    def test_metadata_sentinels_prevent_source_inventory_and_adjacency_access(self):
        class LayerMetadata:
            def __init__(self, layer):
                self.layer_id, self.directed = layer.layer_id, layer.directed

            @property
            def nodes(self):
                raise AssertionError('inventory read')

            @property
            def edges(self):
                raise AssertionError('adjacency read')

        class MetadataOnly:
            def __init__(self, view):
                self.multiplex_id, self.graph_version = view.multiplex_id, view.graph_version
                self.layers = {name: LayerMetadata(layer) for name, layer in view.layers.items()}
                self.layer_order = view.layer_order

            @property
            def multiplex_nodes(self):
                raise AssertionError('global inventory read')

        saved = self.subgraph((A, B, C, D), mode='top_k', top_k=4)
        proof = collect(MetadataOnly(self.view), self.args((A, B, C, D)), [saved])
        self.assertEqual(proof.result['component_count'], 1)

    def test_all_three_node_partial_domains_match_every_possible_completion(self):
        genes = (A, B, C)
        for kind, layer in (('undirected', PPI), ('weak', BRAIN), ('strong', BRAIN)):
            candidates = list(combinations(genes, 2)) if kind == 'undirected' else [pair for pair in product(genes, repeat=2) if pair[0] != pair[1]]
            for states in product(('unknown', 'absent', 'present'), repeat=len(candidates)):
                positive = [pair for pair, state in zip(candidates, states) if state == 'present']
                unknown = [pair for pair, state in zip(candidates, states) if state == 'unknown']
                view = GraphView('partial_domain', 'b' * 64, {layer: LayerGraph(layer, kind != 'undirected',
                    dict.fromkeys(positive, 0), frozenset(genes))})
                arguments = {'multiplex_id': view.multiplex_id, 'layer_id': layer, 'gene_ids': list(genes), 'connectivity_kind': kind}
                node_arguments = {key: arguments[key] for key in ('multiplex_id', 'layer_id', 'gene_ids')}
                saved = [observation('get_nodes_by_layer', node_arguments,
                    NodesByLayerRuntime(view).get_nodes_by_layer(**node_arguments, evidence_id='members'))]
                for index, (pair, state) in enumerate(zip(candidates, states)):
                    if state != 'unknown':
                        edge_arguments = {**node_arguments, 'gene_ids': list(pair)}
                        saved.append(observation('get_edge', edge_arguments,
                            view.get_edge(**edge_arguments, evidence_id=f'edge_{index}')))
                completions = set()
                for included in product((False, True), repeat=len(unknown)):
                    pairs = [*positive, *(pair for pair, keep in zip(unknown, included) if keep)]
                    completions.add(tuple(tuple(group) for group in naive_partition(genes, pairs, kind)))
                proof = collect(view, arguments, saved)
                with self.subTest(kind=kind, states=states):
                    self.assertEqual(proof.result is not None, len(completions) == 1)
                    if proof.result is not None:
                        self.assertEqual(tuple(tuple(group) for group in proof.result['components']), next(iter(completions)))


if __name__ == '__main__':
    unittest.main()
