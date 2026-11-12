"""Check degree allocation against unfiltered full source populations."""

from copy import deepcopy
from itertools import combinations
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_8_sampling import fact_from_arguments, scope_schedule, select_facts
from runtime.world_model.s2_corpus import SPLITS, canonical_json
from runtime.world_model.s2_degrees import DegreeIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


SEED = 20261028
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


def scope(*layers, family=None):
    """Construct a typed explicit or exhaustive scope."""
    return {'kind': 'layers' if layers else 'all', 'layer_ids': list(layers), 'layer_family': family}


class DegreeSamplingTests(unittest.TestCase):
    """Retain source members in all query splits and one graph-only member."""

    @classmethod
    def setUpClass(cls):
        pool = [f'ENSG{number:011d}' for number in range(1, 101)]
        groups = {split: [gene for gene in pool if gene_group(gene, SEED)[1] == split] for split in SPLITS}
        cls.a, cls.b, cls.c = (groups[split][0] for split in SPLITS)
        cls.graph_only, cls.isolated, cls.other = groups['train'][1:4]
        a, b, c, graph_only, isolated, other = cls.a, cls.b, cls.c, cls.graph_only, cls.isolated, cls.other
        cls.view = GraphView('unit_degree_sampling_view', '8' * 64, {
            PPI: LayerGraph(PPI, False, {(a, a): .9, (a, b): .8, (a, c): .7, (b, c): .6,
                (c, graph_only): .5}, frozenset((a, b, c, graph_only, isolated))),
            COEX: LayerGraph(COEX, False, {(a, b): .4, (b, other): .3}, frozenset((a, b, other))),
            BRAIN: LayerGraph(BRAIN, True, {(a, a): .3, (a, b): .2, (b, a): .1,
                (c, graph_only): 0}, frozenset((a, b, c, graph_only, isolated))),
            CD4: LayerGraph(CD4, True, {(other, c): .5}, frozenset((other, c))),
        })
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: f'UNIT_DEGREE_SAMPLE_{ordinal}' for ordinal, gene in enumerate(pool) if gene != graph_only})

    def setUp(self):
        self.index = DegreeIndex(self.view)

    def select(self, *, smoke=False, counts=None, registry=None):
        return select_facts(self.view, self.index, registry or self.registry,
                            counts or dict.fromkeys(SPLITS, 1), SEED, smoke=smoke)

    def test_schedule_visits_every_nonempty_layer_subset(self):
        schedule = scope_schedule(self.view)
        explicit = {tuple(selected['layer_ids']) for selected, _ in schedule if selected['kind'] == 'layers'}
        expected = {layers for size in range(1, 5) for layers in combinations(self.view.layer_order, size)}
        self.assertEqual(explicit, expected)

    def test_schedule_matches_native_directions_and_empty_family_semantics(self):
        schedule = scope_schedule(self.view)
        self.assertEqual({side for selected, side in schedule if selected == scope(PPI)}, {'incident', 'undirected'})
        self.assertEqual({side for selected, side in schedule if selected == scope(BRAIN)}, {'incident', 'in', 'out'})
        self.assertEqual({side for selected, side in schedule if selected == scope(PPI, BRAIN)}, {'incident'})
        self.assertIn((scope(family='bulk_pen'), 'incident'), schedule)

    def test_fact_rank_uses_members_from_all_query_splits(self):
        fact = fact_from_arguments(self.view, self.index, self.a, scope(PPI), 'undirected', SEED)
        self.assertEqual(fact.split, 'train')
        self.assertEqual(fact.rank_support['population_count'], 5)
        self.assertGreater(fact.rank_support['population_count'], sum(
            gene_group(gene, SEED)[1] == fact.split for gene in self.view.layers[PPI].nodes))

    def test_graph_only_query_is_counted_and_can_be_selected(self):
        fact = fact_from_arguments(self.view, self.index, self.graph_only, scope(PPI), 'undirected', SEED)
        self.assertEqual(fact.degree, 1)
        self.assertTrue(fact.membership['present_in_scope'])
        facts, audit = self.select()
        self.assertTrue(any(item.gene_id == self.graph_only for item in facts))
        domain = next(item for item in audit['domains'] if item['layer_scope'] == scope(PPI) and item['direction'] == 'undirected')
        self.assertEqual(domain['graph_ids_outside_registry'], 1)

    def test_histogram_and_ranks_do_not_change_with_registry_filter(self):
        reduced = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', {
            gene: symbol for gene, symbol in self.registry.symbols_by_id.items() if gene != self.b})
        _, first = self.select()
        _, second = self.select(registry=reduced)
        self.assertEqual([(item['population_count'], item['degree_histogram']) for item in first['domains']],
                         [(item['population_count'], item['degree_histogram']) for item in second['domains']])

    def test_explicit_quotas_visit_every_attainable_degree_cell(self):
        _, audit = self.select()
        for cell in audit['cells']:
            self.assertEqual(cell['selected_degree_values'], cell['available_degree_values'])
            self.assertEqual(cell['unselected_degree_values'], [])

    def test_histogram_includes_directional_zeros_and_full_membership(self):
        _, audit = self.select()
        domain = next(item for item in audit['domains'] if item['layer_scope'] == scope(BRAIN) and item['direction'] == 'out')
        self.assertEqual(domain['population_count'], 5)
        self.assertEqual(sum(row['node_count'] for row in domain['degree_histogram']), 5)
        self.assertEqual(domain['degree_histogram'][0], {'degree': 0, 'node_count': 2})

    def test_smoke_keeps_each_split_maximum_and_records_omitted_degrees(self):
        facts, audit = self.select(smoke=True)
        for cell in audit['cells']:
            available = cell['available_degree_values']
            if available:
                self.assertIn(max(available), cell['selected_degree_values'])
                self.assertTrue(set(cell['selected_degree_values']) <= set(available))
        self.assertIsNone(audit['runtime_degree_cap'])
        self.assertFalse(audit['registry_filter_applied_to_populations'])

    def test_source_selection_is_deterministic(self):
        first, first_audit = self.select(smoke=True)
        second, second_audit = self.select(smoke=True)
        self.assertEqual(first, second)
        self.assertEqual(first_audit, second_audit)

    def test_absence_and_empty_family_are_not_population_members(self):
        missing = next(gene for gene in self.registry.symbols_by_id if gene not in self.view.multiplex_nodes)
        absent = fact_from_arguments(self.view, self.index, missing, scope(PPI), 'undirected', SEED)
        empty = fact_from_arguments(self.view, self.index, self.a, scope(family='bulk_pen'), 'incident', SEED)
        self.assertIsNone(absent.degree)
        self.assertIsNone(absent.membership['present_in_scope'])
        self.assertEqual(empty.degree, 0)
        self.assertFalse(empty.membership['present_in_scope'])
        self.assertIsNone(empty.rank_support)

    def test_oversubscribed_cells_report_the_actual_population(self):
        facts, audit = self.select(counts=dict.fromkeys(SPLITS, 20))
        self.assertTrue(audit['omissions'])
        for omission in audit['omissions']:
            self.assertEqual(omission['requested'], 20)
            self.assertEqual(omission['selected'], omission['population'])

    def test_invalid_quotas_are_rejected(self):
        for counts in ({'train': 1}, dict.fromkeys(SPLITS, 0), dict.fromkeys(SPLITS, True)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.select(counts=counts)


if __name__ == '__main__':
    unittest.main()
