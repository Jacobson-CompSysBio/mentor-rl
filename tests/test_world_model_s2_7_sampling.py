"""Check real-set pair audits, fixed subject splits, and uncapped extrema."""

from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations, product
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_7_sampling import fact_from_queries, neighbor_ids, select_facts
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph


SEED = 20261028
SPLITS = ('train', 'validation', 'test')
PPI, TF = 'HumanNetV3:string_ppi', 'TFs:brain'


class SharedSamplingTests(unittest.TestCase):
    """Compare sparse source audits with a small exhaustive pair calculation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Construct native layers and a larger identity-only registry in every split."""
        pools = defaultdict(list)
        for number in range(1, 900):
            gene = f'ENSG{number:011d}'
            pools[gene_group(gene, SEED)[1]].append(gene)
        cls.pools = {split: tuple(pools[split][:24]) for split in SPLITS}
        ppi, tf, symbols = {}, {}, {}
        for split, pool in cls.pools.items():
            a, b, c, d, e, f, hub = pool[:7]
            for first, second in ((a, c), (a, d), (b, c), (b, d), (c, d), (e, f)):
                ppi[tuple(sorted((first, second)))] = .5
            for other in pool[7:16]:
                ppi[tuple(sorted((hub, other)))] = .2
            tf.update({(a, c): .8, (a, d): .7, (b, c): .4, (e, a): .9,
                       (f, b): .3, (hub, a): .1, (d, d): .6})
            symbols.update({gene: f'UNIT_{gene}' for gene in pool})
            symbols[pool[15]] = None
            symbols.pop(hub)
        def layer(name, edges, directed):
            return LayerGraph(name, directed, edges, frozenset(g for edge in edges for g in edge))
        cls.view = GraphView('unit_shared_sampling', '7' * 64,
                             {PPI: layer(PPI, ppi, False), TF: layer(TF, tf, True)})
        cls.index = NeighborIndex(cls.view)
        cls.registry = RegistryContext('110', 'ensembl_homo_sapiens_110_unit', symbols)
        cls.counts = dict.fromkeys(SPLITS, 2)
        cls.facts, cls.audit = select_facts(cls.view, cls.index, cls.registry, cls.counts, SEED, smoke=True)

    def test_role_order_changes_queries_but_preserves_the_split_unit(self):
        a, b = self.pools['train'][:2]
        first = fact_from_queries(self.view, self.index, PPI, ((a, 'undirected'), (b, 'undirected')))
        second = fact_from_queries(self.view, self.index, PPI, first.queries[::-1])
        self.assertEqual(first.split_unit, second.split_unit)
        self.assertEqual(first.shared_count, 2)
        self.assertEqual(second.queries, first.queries[::-1])
        repeated = fact_from_queries(self.view, self.index, PPI, ((a, 'undirected'),) * 2)
        self.assertEqual(repeated.queries, ((a, 'undirected'),) * 2)

    def test_global_absence_stays_unavailable_and_layer_absence_is_empty(self):
        a, _, _, _, _, _, _, absent = self.pools['train'][:8]
        missing = self.pools['train'][-1]
        unknown = fact_from_queries(self.view, self.index, TF, ((missing, 'in'), (a, 'in')))
        self.assertIsNone(unknown.neighbor_counts[0])
        self.assertIsNone(unknown.shared_count)
        empty = fact_from_queries(self.view, self.index, TF, ((absent, 'out'), (a, 'out')))
        self.assertEqual(empty.neighbor_counts, (0, 2))
        self.assertEqual(empty.shared_count, 0)

    def test_both_uses_set_union_and_preserves_a_self_loop_once(self):
        a, _, _, loop = self.pools['train'][:4]
        expected = ({g for g, _ in self.index.neighbors(TF, a, 'in')}
                    | {g for g, _ in self.index.neighbors(TF, a, 'out')})
        self.assertEqual(neighbor_ids(self.index, TF, a, 'both'), expected)
        fact = fact_from_queries(self.view, self.index, TF, ((loop, 'both'),) * 2)
        self.assertIn(loop, neighbor_ids(self.index, TF, loop, 'both'))
        self.assertEqual(fact.shared_count, len(neighbor_ids(self.index, TF, loop, 'both')))

    def test_cross_split_and_invalid_direction_are_rejected(self):
        a, b = self.pools['train'][0], self.pools['test'][0]
        with self.assertRaisesRegex(ValueError, 'cannot cross'):
            fact_from_queries(self.view, self.index, PPI, ((a, 'undirected'), (b, 'undirected')))
        with self.assertRaisesRegex(ValueError, 'available directions'):
            fact_from_queries(self.view, self.index, PPI, ((a, 'out'), (a, 'out')))
        with self.assertRaisesRegex(ValueError, 'two roles'):
            fact_from_queries(self.view, self.index, PPI, ((a, 'undirected'),))

    def test_each_declared_pair_histogram_matches_exhaustive_source_sets(self):
        eligible = [g for g, symbol in self.registry.symbols_by_id.items() if symbol is not None]
        for domain in self.audit['domains']:
            layer = domain['layer_id']
            directions = domain['directions']
            pools = [[g for g in eligible if 1 <= len(neighbor_ids(self.index, layer, g, direction)) <= 5]
                     for direction in directions]
            pairs = product(*pools) if domain['category'] == 'mixed_out_in' else combinations(sorted(pools[0]), 2)
            histogram, split_histogram, cells = Counter(), {s: Counter() for s in SPLITS}, Counter()
            for first, second in pairs:
                count = len(neighbor_ids(self.index, layer, first, directions[0])
                            & neighbor_ids(self.index, layer, second, directions[1]))
                histogram[count] += 1
                split = gene_group(first, SEED)[1]
                if gene_group(second, SEED)[1] == split:
                    split_histogram[split][count] += 1
                    edges = self.view.layers[layer].edges
                    adjacent = ((first, second) in edges or (second, first) in edges)
                    cells[(split, count, adjacent)] += 1
            self.assertEqual({k: v for k, v in domain['histogram'].items() if v}, dict(sorted(histogram.items())))
            self.assertEqual(domain['candidate_pairs'], sum(histogram.values()))
            for cell in domain['cells']:
                key = (cell['split'], cell['shared_count'], cell['adjacent'])
                self.assertEqual(cell['population'], cells[key], (layer, directions, key))
                self.assertEqual(cell['selected'], min(self.counts[cell['split']], cells[key]))
            if 'split_histograms' in domain:
                for split in SPLITS:
                    self.assertEqual({k: v for k, v in domain['split_histograms'][split].items() if v},
                                     dict(sorted(split_histogram[split].items())))

    def test_all_selected_subjects_keep_their_existing_split(self):
        for fact in self.facts:
            self.assertEqual({gene_group(g, SEED)[1] for g, _ in fact.queries}, {fact.split})
            truth = fact_from_queries(self.view, self.index, fact.layer_id, fact.queries, category=fact.category)
            self.assertEqual(fact, truth)
        self.assertFalse(self.audit['edge_fact_disjoint'])
        self.assertIsNone(self.audit['runtime_degree_cap'])

    def test_repeated_source_extrema_exceed_the_readable_degree_range(self):
        for split in SPLITS:
            hub = self.pools[split][6]
            selected = [fact for fact in self.facts if fact.layer_id == PPI and fact.queries == ((hub, 'undirected'),) * 2]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].shared_count, 9)
            self.assertNotIn(hub, self.registry.symbols_by_id)
        self.assertTrue(any(f.category == 'outside_multiplex' for f in self.facts))
        self.assertTrue(any(f.category == 'outside_layer' for f in self.facts))
        self.assertTrue(any(f.category == 'zero_direction' for f in self.facts))
        self.assertTrue(any(f.category == 'same_gene_mixed_directions' for f in self.facts))

    def test_selection_is_deterministic_and_source_histograms_do_not_depend_on_quota(self):
        facts, audit = select_facts(self.view, self.index, self.registry, self.counts, SEED, smoke=True)
        self.assertEqual([asdict(f) for f in facts], [asdict(f) for f in self.facts])
        self.assertEqual(audit, self.audit)
        _, smaller = select_facts(self.view, self.index, self.registry, dict.fromkeys(SPLITS, 1), SEED)
        self.assertEqual([d['histogram'] for d in smaller['domains']], [d['histogram'] for d in self.audit['domains']])

    def test_invalid_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            select_facts(self.view, self.index, self.registry, {'train': 0}, SEED)


if __name__ == '__main__':
    unittest.main()
