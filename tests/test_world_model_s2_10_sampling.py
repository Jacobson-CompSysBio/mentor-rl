"""Check measured component strata, fixed subjects, and declared source limits."""

from dataclasses import asdict
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_10_sampling import fact_from_genes, select_facts
from runtime.world_model.s2_components import components
from runtime.world_model.s2_corpus import SPLITS
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


class ComponentSamplingTests(unittest.TestCase):
    """Use explicit native motifs in each existing query split."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 601)]
        cls.groups = {split: [gene for gene in pool if gene_group(gene, SEED)[1] == split] for split in SPLITS}
        edges = {layer: {} for layer in (PPI, COEX, BRAIN, CD4)}
        nodes = {layer: set() for layer in edges}
        for split in SPLITS:
            a, b, c, d, isolated, outside, other, x, y, z, p, q, r = cls.groups[split][:13]
            edges[PPI].update({(a, a): .8, tuple(sorted((a, b))): .4, tuple(sorted((b, c))): 0})
            nodes[PPI].update((a, b, c, d, isolated))
            edges[COEX][tuple(sorted((outside, other)))] = .3
            nodes[COEX].update((outside, other))
            edges[BRAIN].update({(a, b): .8, (b, a): .4, (b, c): .3, (c, d): 0, (d, c): .6,
                (a, a): .9, (x, y): .2, (y, z): .5, (p, q): .4, (q, r): .3, (r, p): .7, (r, r): .6})
            nodes[BRAIN].update((a, b, c, d, isolated, x, y, z, p, q, r))
            edges[CD4][(outside, a)] = .2
            nodes[CD4].update((outside, a))
        cls.graph_only = cls.groups['train'][2]
        cls.view = GraphView('unit_component_sampling', 'b' * 64, {
            layer: LayerGraph(layer, layer in (BRAIN, CD4), edges[layer], frozenset(nodes[layer])) for layer in edges})
        cls.index = SubgraphIndex(cls.view)
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_COMPONENT_SAMPLE_{position}' for position, gene in enumerate(pool) if gene != cls.graph_only})
        cls.facts, cls.audit = cls.select()

    @classmethod
    def select(cls, *, counts=None, smoke=True, sizes=(0, 1, 2, 3, 8)):
        return select_facts(cls.view, cls.index, cls.registry, counts or dict.fromkeys(SPLITS, 1),
                            SEED, smoke=smoke, input_sizes=sizes)

    def test_order_and_duplicates_share_one_source_unit(self):
        a, b, c = self.groups['train'][:3]
        first = fact_from_genes(self.view, self.index, PPI, [c, a, c, b], None, SEED)
        second = fact_from_genes(self.view, self.index, PPI, [a, b, c], 'undirected', SEED)
        self.assertEqual(first.gene_ids, (c, a, b))
        self.assertEqual(first.split_unit, second.split_unit)
        self.assertEqual(first.components, ((a, b, c),))
        self.assertEqual(first.component_sizes, (3,))

    def test_native_directed_graph_supports_matched_weak_and_strong_counts(self):
        a, b, c, d = self.groups['train'][:4]
        weak = fact_from_genes(self.view, self.index, BRAIN, [d, c, b, a], 'weak', SEED)
        strong = fact_from_genes(self.view, self.index, BRAIN, [d, c, b, a], 'strong', SEED)
        self.assertEqual(weak.component_count, 1)
        self.assertEqual(strong.components, ((a, b), (c, d)))
        self.assertEqual(strong.component_sizes, (2, 2))
        self.assertEqual(weak.split_unit, strong.split_unit)
        self.assertEqual(weak.edge_count, strong.edge_count)
        self.assertEqual(strong.reciprocal_pairs, 2)
        self.assertEqual(strong.loop_count, 1)

    def test_exclusions_change_all_input_boolean_without_adding_components(self):
        a, b = self.groups['train'][:2]
        outside = self.groups['train'][5]
        missing = next(gene for gene in self.groups['train'] if gene not in self.view.multiplex_nodes)
        fact = fact_from_genes(self.view, self.index, PPI, [a, b, outside, missing], None, SEED)
        self.assertEqual(fact.category_counts, (2, 1, 1))
        self.assertEqual(fact.components, ((a, b),))
        self.assertEqual(fact.component_count, 1)
        self.assertFalse(fact.single_component)

    def test_singleton_without_loop_is_one_strong_group(self):
        isolated = self.groups['train'][4]
        fact = fact_from_genes(self.view, self.index, BRAIN, [isolated], 'strong', SEED)
        self.assertEqual(fact.components, ((isolated,),))
        self.assertEqual(fact.loop_count, 0)
        self.assertEqual(fact.edge_count, 0)
        self.assertTrue(fact.single_component)

    def test_empty_projection_counts_zero_and_requires_an_assigned_valid_split(self):
        for split in SPLITS:
            fact = fact_from_genes(self.view, self.index, PPI, [], None, SEED, split=split)
            self.assertEqual((fact.split, fact.component_count, fact.component_sizes), (split, 0, ()))
            self.assertFalse(fact.single_component)
        with self.assertRaises(ValueError):
            fact_from_genes(self.view, self.index, PPI, [], None, SEED, split='unknown')

    def test_source_queries_do_not_cross_existing_subject_splits(self):
        with self.assertRaises(ValueError):
            fact_from_genes(self.view, self.index, PPI,
                            [self.groups['train'][0], self.groups['test'][0]], None, SEED)
        for fact in self.facts:
            self.assertTrue(all(gene_group(gene, SEED)[1] == fact.split for gene in fact.gene_ids))

    def test_all_selected_directed_sets_keep_both_kinds_in_the_same_split(self):
        grouped = {}
        for fact in self.facts:
            grouped.setdefault((fact.split, fact.split_unit), set()).add(fact.connectivity_kind)
        for fact in self.facts:
            if self.view.layers[fact.layer_id].directed:
                self.assertEqual(grouped[(fact.split, fact.split_unit)], {'weak', 'strong'})
        self.assertTrue(self.audit['matched_directed_kinds'])

    def test_source_only_identity_does_not_filter_membership(self):
        a, b, graph_only = self.groups['train'][:3]
        self.assertNotIn(graph_only, self.registry.symbols_by_id)
        fact = fact_from_genes(self.view, self.index, PPI, [a, b, graph_only], None, SEED)
        self.assertEqual(fact.category_counts, (3, 0, 0))
        self.assertEqual(fact.components, ((a, b, graph_only),))
        self.assertTrue(any(graph_only in fact.gene_ids for fact in self.facts))

    def test_targeted_structures_come_from_the_native_fixture(self):
        records = self.audit['targeted_source_cases']
        categories = {record['category'] for record in records}
        self.assertTrue({'removed_bridge', 'directed_chain', 'strong_cycle',
                         'two_nontrivial_strong_groups', 'connected_projection_with_exclusion'} <= categories)
        for record in records:
            if 'gene_ids' not in record:
                continue
            layer, genes = record['layer_id'], record['gene_ids']
            native = self.index.edges(layer, genes)
            self.assertEqual(record['edge_count'], len(native))
            if record['category'] == 'two_nontrivial_strong_groups':
                self.assertEqual(sorted(map(len, components(genes, native, 'strong'))), [2, 2])
            if record['category'] == 'removed_bridge':
                self.assertEqual(native, [])
                bridge = record['bridge_gene_id']
                self.assertNotIn(bridge, genes)
                edges = self.index.edges(layer, [*genes, bridge])
                kind = 'weak' if self.view.layers[layer].directed else 'undirected'
                self.assertEqual(len(components([*genes, bridge], edges, kind)), 1)

    def test_component_strata_use_measured_groups_and_report_quota_shortfalls(self):
        _, audit = self.select(counts=dict.fromkeys(SPLITS, 5))
        self.assertTrue(audit['quota_shortfalls'])
        for cell in audit['component_cells']:
            self.assertEqual(cell['selected'], min(cell['requested'], cell['candidate_sets']))
            self.assertEqual(cell['component_count'], len(cell['component_sizes']))
            self.assertEqual(sum(cell['component_sizes']), cell['category_counts'][0])
            self.assertEqual(list(cell['component_sizes']), sorted(cell['component_sizes']))

    def test_partial_strong_search_uses_real_same_split_proper_prefixes(self):
        records = self.audit['targeted_source_cases']
        witnesses = [record for record in records if record['category'] == 'partial_strong_witness']
        self.assertEqual({record['split'] for record in witnesses if record['layer_id'] == BRAIN}, set(SPLITS))
        for record in witnesses:
            genes = record['gene_ids']
            self.assertTrue(all(gene_group(gene, SEED)[1] == record['split'] for gene in genes))
            native = self.index.edges(record['layer_id'], genes)
            self.assertGreater(len(native), len(genes))
            self.assertEqual(len(components(genes, native[:-1], 'strong')), 1)
        searches = [record for record in records if record['category'] == 'target_search']
        self.assertTrue(all(record['strong_motif_vertex_window'] == 128 for record in searches))
        self.assertTrue(any(record['strong_motif_candidates_examined'] > 0 for record in searches))

    def test_every_fact_matches_full_native_edges_and_partition(self):
        for fact in self.facts:
            partition = self.index.partition(fact.layer_id, fact.gene_ids)
            native = self.index.edges(fact.layer_id, fact.gene_ids)
            measured = components(partition['layer_gene_ids'], native, fact.connectivity_kind)
            self.assertEqual([list(group) for group in fact.components], measured)
            self.assertEqual(fact.component_count, len(measured))
            self.assertEqual(fact.edge_count, len(native))
            self.assertEqual(fact.single_component, len(measured) == 1 and not any(fact.category_counts[1:]))

    def test_finite_domain_does_not_claim_all_subsets_or_runtime_caps(self):
        self.assertFalse(self.audit['all_combinatorial_subsets_enumerated'])
        self.assertFalse(self.audit['edge_fact_disjoint'])
        self.assertIsNone(self.audit['runtime_input_cap'])
        self.assertIsNone(self.audit['runtime_component_cap'])
        self.assertTrue(self.audit['remaining_domains'])
        self.assertTrue(all(not domain['all_combinatorial_subsets_enumerated'] for domain in self.audit['domains']))

    def test_default_sizes_keep_128_and_explicit_recipe_uses_more_windows(self):
        _, default_audit = select_facts(self.view, self.index, self.registry, dict.fromkeys(SPLITS, 1), SEED, smoke=True)
        _, explicit_audit = self.select(smoke=False)
        self.assertIn(128, default_audit['input_sizes'])
        self.assertEqual(self.audit['candidate_quota_per_graph_cell'], 3)
        self.assertEqual(explicit_audit['candidate_quota_per_graph_cell'], 16)

    def test_content_and_audit_are_deterministic_without_any_wrapper_input(self):
        facts, audit = self.select()
        self.assertEqual([asdict(fact) for fact in facts], [asdict(fact) for fact in self.facts])
        self.assertEqual(audit, self.audit)

    def test_invalid_quotas_and_size_domains_are_rejected(self):
        for counts in ({'train': 1}, dict.fromkeys(SPLITS, 0), dict.fromkeys(SPLITS, True)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.select(counts=counts)
        for sizes in ((), (-1,), (True,), (1.0,), (1, 'two')):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                self.select(sizes=sizes)


if __name__ == '__main__':
    unittest.main()
