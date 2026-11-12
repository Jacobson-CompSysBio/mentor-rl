"""Check finite induced-set allocation without a false combinatorial claim."""

from dataclasses import asdict
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_9_sampling import fact_from_genes, select_facts
from runtime.world_model.s2_corpus import SPLITS
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


class SubgraphSamplingTests(unittest.TestCase):
    """Keep genuine native motifs in every existing query split."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 301)]
        cls.groups = {split: [gene for gene in pool if gene_group(gene, SEED)[1] == split] for split in SPLITS}
        ppi, coex, brain, cd4 = {}, {}, {}, {}
        ppi_nodes, coex_nodes, brain_nodes, cd4_nodes = set(), set(), set(), set()
        for split in SPLITS:
            a, b, c, isolated, outside, other = cls.groups[split][:6]
            ppi.update({(a, a): .9, tuple(sorted((a, b))): .8, tuple(sorted((b, c))): 0})
            coex[tuple(sorted((outside, other)))] = .4
            brain.update({(a, b): .7, (b, a): .6, (c, a): .5, (b, b): .3})
            cd4[(outside, a)] = .2
            ppi_nodes.update((a, b, c, isolated))
            coex_nodes.update((outside, other))
            brain_nodes.update((a, b, c, isolated))
            cd4_nodes.update((outside, a))
        cls.graph_only = cls.groups['train'][2]
        cls.view = GraphView('unit_subgraph_sampling', '9' * 64, {
            PPI: LayerGraph(PPI, False, ppi, frozenset(ppi_nodes)),
            COEX: LayerGraph(COEX, False, coex, frozenset(coex_nodes)),
            BRAIN: LayerGraph(BRAIN, True, brain, frozenset(brain_nodes)),
            CD4: LayerGraph(CD4, True, cd4, frozenset(cd4_nodes)),
        })
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_SUBGRAPH_SAMPLE_{position}' for position, gene in enumerate(pool) if gene != cls.graph_only})

    def setUp(self):
        self.index = SubgraphIndex(self.view)

    def select(self, *, counts=None, smoke=True, sizes=(0, 1, 2, 3, 8), registry=None):
        return select_facts(self.view, self.index, registry or self.registry,
            counts or dict.fromkeys(SPLITS, 1), SEED, smoke=smoke, input_sizes=sizes)

    def test_facts_keep_first_mentions_and_order_independent_split_units(self):
        a, b, c = self.groups['train'][:3]
        first = fact_from_genes(self.view, self.index, PPI, [c, a, c, b], SEED)
        second = fact_from_genes(self.view, self.index, PPI, [a, b, c], SEED)
        self.assertEqual(first.gene_ids, (c, a, b))
        self.assertEqual(first.split_unit, second.split_unit)
        self.assertEqual(first.category_counts, (3, 0, 0))
        self.assertEqual(first.edge_count, 3)

    def test_fact_statistics_keep_native_loops_reciprocals_and_descending_arcs(self):
        a, b, c = self.groups['train'][:3]
        fact = fact_from_genes(self.view, self.index, BRAIN, [a, b, c], SEED)
        self.assertEqual(fact.edge_count, 4)
        self.assertEqual(fact.loop_count, 1)
        self.assertEqual(fact.reciprocal_pairs, 1)
        self.assertEqual(fact.descending_arc_count, sum(x > y for x, y in self.view.layers[BRAIN].edges
            if x in (a, b, c) and y in (a, b, c)))

    def test_selected_isolate_is_distinct_from_excluded_inputs(self):
        a, b, _, isolated, outside = self.groups['train'][:5]
        missing = next(gene for gene in self.groups['train'] if gene not in self.view.multiplex_nodes)
        fact = fact_from_genes(self.view, self.index, PPI, [a, b, isolated, outside, missing], SEED)
        self.assertEqual(fact.category_counts, (3, 1, 1))
        self.assertEqual(fact.isolated_count, 1)
        self.assertEqual(fact.edge_count, 2)

    def test_query_subjects_cannot_cross_existing_splits(self):
        with self.assertRaises(ValueError):
            fact_from_genes(self.view, self.index, PPI, [self.groups['train'][0], self.groups['test'][0]], SEED)

    def test_empty_sets_have_explicit_assigned_splits(self):
        for split in SPLITS:
            fact = fact_from_genes(self.view, self.index, PPI, [], SEED, split=split)
            self.assertEqual(fact.split, split)
            self.assertEqual(fact.category_counts, (0, 0, 0))
            self.assertEqual(fact.edge_count, 0)
        with self.assertRaises(ValueError):
            fact_from_genes(self.view, self.index, PPI, [], SEED, split='unknown')

    def test_every_selected_gene_retains_its_fixed_query_split(self):
        facts, _ = self.select()
        for fact in facts:
            self.assertTrue(all(gene_group(gene, SEED)[1] == fact.split for gene in fact.gene_ids))

    def test_registry_does_not_filter_real_projection_members_or_edges(self):
        a, b, graph_only = self.groups['train'][:3]
        fact = fact_from_genes(self.view, self.index, PPI, [a, b, graph_only], SEED)
        self.assertEqual(fact.category_counts, (3, 0, 0))
        self.assertEqual(fact.edge_count, 3)
        facts, audit = self.select()
        self.assertTrue(any(self.graph_only in item.gene_ids for item in facts))
        self.assertTrue(any(item['category'] == 'graph_only_query' and self.graph_only in item['gene_ids']
                            for item in audit['feature_cases']))

    def test_source_features_cover_real_native_edges_and_selected_isolates(self):
        _, audit = self.select()
        categories = {item['category'] for item in audit['feature_cases']}
        self.assertTrue({'native_edge', 'self_loop', 'reciprocal', 'descending_arc', 'zero_weight',
                         'selected_set_isolate', 'source_star'} <= categories)
        for feature in audit['feature_cases']:
            fact = fact_from_genes(self.view, self.index, feature['layer_id'], feature['gene_ids'], SEED)
            self.assertEqual(feature['edge_count'], fact.edge_count)
            self.assertEqual(feature['isolated_count'], fact.isolated_count)

    def test_all_observed_cells_report_real_edge_counts_and_quota_limits(self):
        _, audit = self.select(counts=dict.fromkeys(SPLITS, 5))
        for cell in audit['cells']:
            self.assertLessEqual(cell['selected'], cell['observed_unique_sets'])
            self.assertLessEqual(cell['selected'], cell['requested'])
        self.assertTrue(any('observed_unique_sets' in item for item in audit['omissions']))

    def test_finite_domain_audit_does_not_claim_all_combinatorial_subsets(self):
        _, audit = self.select(sizes=(0, 1, 3))
        self.assertFalse(audit['all_combinatorial_subsets_enumerated'])
        self.assertEqual(audit['candidate_windows_per_composition'], 3)
        self.assertTrue(all(not item['all_combinatorial_subsets_enumerated'] for item in audit['domains']))
        self.assertTrue(any([2, 2] in item['unselected_input_size_ranges'] for item in audit['domains']))

    def test_explicit_quota_recipe_visits_more_recorded_windows(self):
        _, audit = self.select(smoke=False)
        self.assertEqual(audit['candidate_windows_per_composition'], 16)
        self.assertFalse(audit['all_combinatorial_subsets_enumerated'])

    def test_default_sizes_keep_128_without_a_runtime_cap(self):
        _, audit = select_facts(self.view, self.index, self.registry, dict.fromkeys(SPLITS, 1), SEED, smoke=True)
        self.assertIn(128, audit['input_sizes'])
        self.assertIsNone(audit['runtime_input_cap'])
        self.assertIsNone(audit['runtime_edge_cap'])
        self.assertTrue(audit['omissions'])

    def test_selection_and_its_audit_are_deterministic(self):
        first, first_audit = self.select()
        second, second_audit = self.select()
        self.assertEqual([asdict(item) for item in first], [asdict(item) for item in second])
        self.assertEqual(first_audit, second_audit)

    def test_invalid_quotas_and_sizes_are_rejected(self):
        for counts in ({'train': 1}, dict.fromkeys(SPLITS, 0), dict.fromkeys(SPLITS, True)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.select(counts=counts)
        for sizes in ((), (-1,), (1.0,), (0, False), (1, True), (1, 'two'), ({},)):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                self.select(sizes=sizes)


if __name__ == '__main__':
    unittest.main()
