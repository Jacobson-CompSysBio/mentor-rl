"""Check finite full-layer path samples, real hop strata, and fixed query splits."""

from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_corpus import SPLITS, digest
from runtime.world_model.s2_paths import PathIndex, PathRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_11_sampling import fact_from_genes, select_facts


PPI, COEX, BRAIN = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain'
SEED = 20261028


def fixture():
    """Place query subjects in fixed splits and route intermediates in other splits."""
    pools = {split: [] for split in SPLITS}
    for number in range(1000, 2000):
        gene = f'ENSG{number:011d}'
        pools[gene_group(gene, SEED)[1]].append(gene)
    pools = {split: sorted(genes, key=lambda gene: gene_group(gene, SEED)[0])
             for split, genes in pools.items()}
    a, b, c, d, outside, missing, end, direct = pools['train'][:8]
    first, second, val_outside, val_missing = pools['validation'][:4]
    middle, test_outside, test_missing = pools['test'][:3]
    ppi_edges = {(a, a): .5, (a, first): .8, (first, b): .7,
                 (a, second): .6, (second, b): .4, (b, middle): .3,
                 (middle, end): .2, (a, direct): 0, (c, d): .9}
    brain_edges = {(a, a): .5, (a, first): .8, (first, b): .7,
                   (a, second): .6, (second, b): .4, (b, middle): .3,
                   (middle, end): .2, (c, d): .9}
    layers = {
        PPI: LayerGraph(PPI, False, {tuple(sorted(pair)): weight for pair, weight in ppi_edges.items()},
                        frozenset(gene for pair in ppi_edges for gene in pair)),
        COEX: LayerGraph(COEX, False, {tuple(sorted((outside, val_outside))): .1,
                                     tuple(sorted((test_outside, val_outside))): .2},
                         frozenset((outside, val_outside, test_outside))),
        BRAIN: LayerGraph(BRAIN, True, brain_edges,
                          frozenset(gene for pair in brain_edges for gene in pair)),
    }
    view = GraphView('unit_path_sampling', '5' * 64, layers)
    registry = SimpleNamespace(symbols_by_id={gene: f'SYMBOL_{gene}' for gene in
        (set(view.multiplex_nodes) - {c}) | {missing, val_missing, test_missing}})
    names = dict(zip(('a', 'b', 'c', 'd', 'outside', 'missing', 'end', 'direct',
                      'first', 'second', 'middle'),
                    (a, b, c, d, outside, missing, end, direct, first, second, middle)))
    return view, PathIndex(view), registry, names


class PathSamplingTests(unittest.TestCase):
    """Use only attainable path outcomes and preserve each endpoint's fixed split."""

    def setUp(self):
        self.view, self.index, self.registry, self.genes = fixture()
        self.counts = dict.fromkeys(SPLITS, 1)

    def fact(self, left, right, layer=PPI, **kwargs):
        return fact_from_genes(self.view, self.index, layer,
                               [self.genes[left], self.genes[right]], SEED, **kwargs)

    def select(self, **kwargs):
        return select_facts(self.view, self.index, self.registry, self.counts,
                            SEED, smoke=True, source_window=1, **kwargs)

    def test_native_ties_have_exact_count_and_canonical_witness(self):
        for layer in (PPI, BRAIN):
            fact = self.fact('a', 'b', layer)
            self.assertEqual(fact.hop_count, 2)
            self.assertEqual(fact.tied_shortest_paths, 2)
            self.assertEqual(fact.path_gene_ids, (self.genes['a'],
                min(self.genes['first'], self.genes['second']), self.genes['b']))

    def test_route_intermediates_can_belong_to_held_out_splits(self):
        fact = self.fact('a', 'end')
        self.assertEqual(fact.split, 'train')
        self.assertEqual(fact.hop_count, 4)
        self.assertEqual({gene_group(gene, SEED)[1] for gene in fact.gene_ids}, {'train'})
        self.assertIn('validation', {gene_group(gene, SEED)[1] for gene in fact.path_gene_ids})
        self.assertIn('test', {gene_group(gene, SEED)[1] for gene in fact.path_gene_ids})

    def test_same_node_fact_keeps_two_positions_and_one_zero_hop_witness(self):
        fact = self.fact('a', 'a')
        self.assertEqual(fact.gene_ids, (self.genes['a'], self.genes['a']))
        self.assertEqual(fact.category_counts, (1, 0, 0))
        self.assertEqual(fact.path_gene_ids, (self.genes['a'],))
        self.assertEqual((fact.hop_count, fact.tied_shortest_paths), (0, 1))

    def test_zero_and_one_hop_tie_counts_need_no_full_distance_dag(self):
        with patch('runtime.world_model.s2_11_sampling._path_counts', side_effect=AssertionError('unneeded full DAG')):
            self.assertEqual(self.fact('a', 'a').tied_shortest_paths, 1)
            fact = self.fact('a', 'direct')
            self.assertEqual((fact.hop_count, fact.tied_shortest_paths), (1, 1))

    def test_native_zero_weight_edge_remains_a_one_hop_path(self):
        fact = self.fact('a', 'direct')
        self.assertTrue(fact.path_found)
        self.assertEqual(fact.path_gene_ids, (self.genes['a'], self.genes['direct']))

    def test_directed_reverse_has_its_own_outcome_and_shared_split_unit(self):
        forward, reverse = self.fact('a', 'b', BRAIN), self.fact('b', 'a', BRAIN)
        self.assertTrue(forward.path_found)
        self.assertFalse(reverse.path_found)
        self.assertEqual(reverse.path_gene_ids, ())
        self.assertEqual(reverse.tied_shortest_paths, 0)
        self.assertEqual(forward.split_unit, reverse.split_unit)
        self.assertNotEqual(forward.gene_ids, reverse.gene_ids)

    def test_complete_layer_disconnection_has_no_invented_distance(self):
        fact = self.fact('a', 'c')
        self.assertFalse(fact.path_found)
        self.assertIsNone(fact.hop_count)
        self.assertEqual(fact.path_gene_ids, ())
        self.assertEqual(fact.tied_shortest_paths, 0)

    def test_layer_and_global_exclusions_remain_separate(self):
        fact = self.fact('missing', 'outside')
        self.assertEqual(fact.category_counts, (0, 1, 1))
        self.assertIsNone(fact.path_found)
        self.assertIsNone(fact.path_gene_ids)
        self.assertIsNone(fact.hop_count)
        self.assertIsNone(fact.tied_shortest_paths)

    def test_repeated_layer_absence_is_negative_and_repeated_global_absence_is_unevaluated(self):
        outside, missing = self.fact('outside', 'outside'), self.fact('missing', 'missing')
        self.assertEqual(outside.category_counts, (0, 1, 0))
        self.assertFalse(outside.path_found)
        self.assertEqual(missing.category_counts, (0, 0, 1))
        self.assertIsNone(missing.path_found)
        self.assertEqual(outside.gene_ids[0], outside.gene_ids[1])

    def test_query_endpoints_cannot_cross_fixed_splits(self):
        with self.assertRaises(ValueError):
            self.fact('a', 'first')
        with self.assertRaises(ValueError):
            self.fact('a', 'b', split='validation')

    def test_pair_shape_still_allows_repeated_positions(self):
        self.assertEqual(len(self.fact('a', 'a').gene_ids), 2)
        for pair in ([], [self.genes['a']], [self.genes['a']] * 3):
            with self.assertRaises(ValueError):
                fact_from_genes(self.view, self.index, PPI, pair, SEED)

    def test_selection_is_deterministic_with_an_exact_fact_digest(self):
        first, audit = self.select()
        second, repeated = self.select()
        self.assertEqual(first, second)
        self.assertEqual(audit, repeated)
        self.assertEqual(audit['fact_selection_sha256'], digest([asdict(fact) for fact in first]))

    def test_every_selected_reverse_pair_exists_in_the_same_layer_and_split(self):
        facts, audit = self.select()
        by_pair = {(fact.layer_id, fact.gene_ids): fact for fact in facts}
        for fact in facts:
            reverse = by_pair[fact.layer_id, tuple(reversed(fact.gene_ids))]
            self.assertEqual(fact.split, reverse.split)
            self.assertEqual(fact.split_unit, reverse.split_unit)
            self.assertEqual({gene_group(gene, SEED)[1] for gene in fact.gene_ids}, {fact.split})
        self.assertTrue(audit['matched_reverse_queries'])

    def test_finite_window_still_measures_full_layer_routes_and_all_same_split_targets(self):
        facts, audit = self.select()
        domain = next(item for item in audit['domains'] if item['split'] == 'train' and item['layer_id'] == PPI)
        self.assertEqual(domain['source_window'], 1)
        self.assertIn(self.genes['a'], domain['source_genes'])
        self.assertNotIn(self.genes['end'], domain['source_genes'])
        self.assertIn(4, domain['reachable_hop_histogram'])
        self.assertNotIn(3, domain['reachable_hop_histogram'])
        self.assertEqual(domain['searched_pairs'], len(domain['source_genes']) * domain['same_split_targets_per_source'])
        self.assertEqual(sum(domain['reachable_hop_histogram'].values()) + domain['unreachable_pairs'], domain['searched_pairs'])
        self.assertTrue(any(fact.layer_id == PPI and fact.hop_count == 4 for fact in facts))
        self.assertFalse(audit['all_source_pairs_enumerated'])
        self.assertIsNone(audit['runtime_hop_cap'])
        self.assertIsNone(audit['runtime_query_cap'])

    def test_each_observed_hop_cell_uses_its_real_candidate_count(self):
        _, audit = self.select()
        for cell in audit['cells']:
            self.assertGreater(cell['candidate_pairs'], 0)
            self.assertEqual(cell['selected'], min(cell['candidate_pairs'], cell['requested']))
        train = [cell for cell in audit['cells'] if cell['split'] == 'train' and cell['layer_id'] == PPI]
        self.assertEqual({cell['outcome'] for cell in train}, {0, 1, 2, 4, 'no_path'})

    def test_targeted_features_keep_graph_only_ids_loops_zero_weights_and_closed_components(self):
        facts, audit = self.select()
        features = {row['feature'] for row in audit['feature_witnesses']}
        self.assertTrue({'graph_only_query', 'same_node_loop', 'zero_weight_path', 'closed_component'} <= features)
        self.assertTrue(any(self.genes['c'] in fact.gene_ids for fact in facts))
        for row in audit['feature_witnesses']:
            if row['feature'] == 'closed_component':
                self.assertLessEqual(len(row['component_gene_ids']), row['construction_component_size_budget'])
                self.assertIn(row['gene_ids'][0], row['component_gene_ids'])
                self.assertNotIn(row['gene_ids'][1], row['component_gene_ids'])

    def test_all_selected_facts_match_the_complete_native_operation(self):
        facts, _ = self.select()
        runtime = PathRuntime(self.view, index=self.index)
        for fact in facts:
            result = runtime.shortest_path(**runtime.arguments(list(fact.gene_ids), fact.layer_id, fact.direction),
                                           evidence_id='sampling_audit')['result']
            self.assertEqual(fact.path_found, result['path_found'])
            self.assertEqual(fact.hop_count, result['hop_count'])
            self.assertEqual(fact.path_gene_ids, None if result['path_gene_ids'] is None else tuple(result['path_gene_ids']))

    def test_sampling_reads_no_wrapper_bank(self):
        expected, _ = self.select()
        with patch('runtime.world_model.s1_prompt_banks.load_s2_prompt_banks',
                   side_effect=AssertionError('source choices must not read a wrapper bank')):
            actual, _ = self.select()
        self.assertEqual(expected, actual)

    def test_sampling_preserves_all_source_and_registry_inputs(self):
        before = [(layer_id, tuple(layer.edges.items()), layer.nodes) for layer_id, layer in self.view.layers.items()]
        registry = dict(self.registry.symbols_by_id)
        self.select()
        self.assertEqual(before, [(layer_id, tuple(layer.edges.items()), layer.nodes) for layer_id, layer in self.view.layers.items()])
        self.assertEqual(registry, self.registry.symbols_by_id)

    def test_invalid_quotas_and_source_windows_are_rejected(self):
        for counts in ({}, {'train': 1}, dict.fromkeys(SPLITS, 0), dict.fromkeys(SPLITS, True), dict.fromkeys(SPLITS, 1.5)):
            with self.assertRaises(ValueError):
                select_facts(self.view, self.index, self.registry, counts, SEED, smoke=True)
        for window in (0, -1, True, 1.5, '2'):
            with self.assertRaises(ValueError):
                select_facts(self.view, self.index, self.registry, self.counts, SEED, smoke=True, source_window=window)


if __name__ == '__main__':
    unittest.main()
