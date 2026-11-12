"""Check native pair strata, qualified contrasts, and fixed pair splits."""

from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_corpus import BUCKETS, SPLITS, _pair_group, digest
from runtime.world_model.s2_corpus import select_facts as select_edges
from runtime.world_model.s2_13_sampling import fact_from_pair, select_facts
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


PPI, COEX, BRAIN, CD4 = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression',
                        'TFs:brain', 'TFs:cd4+t_cells')
BULK, SC = 'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte'
SEED = 20261028


@lru_cache(maxsize=1)
def fixture():
    """Provide genuine endpoint inventories and known pair controls in every split."""
    number = 1000
    rows = {layer: {} for layer in (PPI, COEX, BRAIN, CD4, BULK, SC)}
    named, all_genes = {}, set()

    def gene():
        nonlocal number
        result = f'ENSG{number:011d}'
        number += 1
        all_genes.add(result)
        return result

    def pair(split):
        for _ in range(10000):
            result = gene(), gene()
            if _pair_group(result, SEED)[1] == split:
                return result
        raise AssertionError('could not place a fixture pair')

    def edge(layer, values, weight):
        key = tuple(values) if layer in (BRAIN, CD4) else tuple(sorted(values))
        rows[layer][key] = weight

    for layer in rows:
        for split in SPLITS:
            for slot in range(3):
                selected = pair(split)
                edge(layer, selected, .1 + slot / 10)
            high = pair(split)
            edge(layer, high, .99)
            named[layer, split, 'high'] = high
    for layer in (BRAIN, CD4):
        for split in SPLITS:
            forward, reciprocal = pair(split), pair(split)
            edge(layer, forward, .51)
            edge(layer, reciprocal, .52)
            edge(layer, reciprocal[::-1], .53)
            named[layer, split, 'one_way'] = forward
            named[layer, split, 'reciprocal'] = reciprocal
    for split in SPLITS:
        for presence in ('11', '10', '01'):
            selected = pair(split)
            for layer, present in ((BULK, presence[0]), (SC, presence[1])):
                if present == '1':
                    edge(layer, selected, .4 if layer == BULK else .5)
                else:
                    edge(layer, (selected[0], gene()), .2)
                    edge(layer, (selected[1], gene()), .3)
            named['pen', split, presence] = selected
        endpoints, middle = pair(split), gene()
        edge(PPI, (endpoints[0], middle), .6)
        edge(PPI, (middle, endpoints[1]), .7)
        named[PPI, split, 'route'] = endpoints
        named[PPI, split, 'middle'] = middle
    zero, null_edge, ambiguous_edge, graph_only_edge = (pair('train') for _ in range(4))
    edge(PPI, zero, 0)
    edge(CD4, null_edge, .42)
    edge(COEX, ambiguous_edge, .43)
    edge(PPI, graph_only_edge, .44)
    loop = gene()
    edge(PPI, (loop, loop), 2)
    missing = gene()
    duplicate = gene()
    symbols = {item: 'SYMBOL_' + item for item in sorted(all_genes)}
    symbols[null_edge[0]] = None
    symbols[ambiguous_edge[0]] = symbols[duplicate] = 'AMBIGUOUS_FIXTURE'
    del symbols[graph_only_edge[0]]
    graphs = {layer: LayerGraph(layer, layer in (BRAIN, CD4), edges,
        frozenset(item for values in edges for item in values)) for layer, edges in rows.items()}
    view = GraphView('unit_s213_six', '6' * 64, graphs)
    old = GraphView('unit_s213_four', '4' * 64,
                    {layer: graphs[layer] for layer in (PPI, COEX, BRAIN, CD4)})
    registry = RegistryContext('116', 'ensembl_homo_sapiens_116_unit', symbols)
    named.update(zero=zero, null_edge=null_edge, ambiguous_edge=ambiguous_edge,
                 graph_only_edge=graph_only_edge, missing=missing, loop=loop)
    return view, SubgraphIndex(view), registry, old, named


class PairSamplingTests(unittest.TestCase):
    """Check source choices without a request bank or a wrapper bank."""

    @classmethod
    def setUpClass(cls):
        cls.view, cls.index, cls.registry, cls.old, cls.named = fixture()
        cls.counts = dict.fromkeys(SPLITS, 1)
        cls.facts, cls.audit = select_facts(cls.view, cls.index, cls.registry,
            cls.counts, SEED, smoke=True, old_view=cls.old)

    def fact(self, layer, pair):
        """Measure one explicit native pair under the immutable source view."""
        return fact_from_pair(self.view, self.index, self.registry, self.old, layer, pair, SEED)

    def test_four_categories_keep_registry_and_graph_membership_separate(self):
        present = self.named[PPI, 'train', 'high']
        no_edge = self.named[PPI, 'train', 'route']
        outside = self.named['pen', 'train', '11'][0]
        self.assertEqual(self.fact(PPI, present).category, 'recorded')
        self.assertEqual(self.fact(PPI, no_edge).category, 'no_edge')
        self.assertEqual(self.fact(PPI, (present[0], outside)).category, 'outside_layer')
        self.assertEqual(self.fact(PPI, (present[0], self.named['missing'])).category, 'outside_multiplex')
        graph_only = self.fact(PPI, self.named['graph_only_edge'])
        self.assertEqual(graph_only.category, 'recorded')
        self.assertIn('graph_only', graph_only.registry_states)

    def test_direction_preserves_order_while_pair_split_ignores_order_and_layer(self):
        pair = self.named[CD4, 'test', 'one_way']
        forward, reverse = self.fact(CD4, pair), self.fact(CD4, pair[::-1])
        other = self.fact(PPI, pair)
        self.assertEqual(forward.gene_ids, pair)
        self.assertEqual(reverse.gene_ids, pair[::-1])
        self.assertEqual((forward.category, reverse.category), ('recorded', 'no_edge'))
        self.assertFalse(forward.reverse_exists)
        self.assertTrue(reverse.reverse_exists)
        self.assertEqual((forward.split_unit, forward.split), _pair_group(pair, SEED))
        self.assertEqual((reverse.split_unit, reverse.split), (forward.split_unit, forward.split))
        self.assertEqual((other.split_unit, other.split), (forward.split_unit, forward.split))
        self.assertIsNone(other.reverse_exists)

    def test_reciprocal_edges_keep_two_weights_and_native_order(self):
        pair = self.named[BRAIN, 'validation', 'reciprocal']
        forward, reverse = self.fact(BRAIN, pair), self.fact(BRAIN, pair[::-1])
        self.assertEqual((forward.weight, reverse.weight), (.52, .53))
        self.assertTrue(forward.reverse_exists)
        self.assertTrue(reverse.reverse_exists)
        self.assertEqual((forward.category, reverse.category), ('recorded', 'recorded'))

    def test_zero_weight_is_a_positive_record_without_rescaling(self):
        fact = self.fact(PPI, self.named['zero'])
        self.assertEqual((fact.category, fact.weight), ('recorded', 0))
        self.assertIn(PPI, fact.positive_layers)
        reverse = self.fact(PPI, self.named['zero'][::-1])
        self.assertEqual((reverse.category, reverse.weight), ('recorded', 0))

    def test_positive_layers_are_exact_native_orientations(self):
        for presence in ('11', '10', '01'):
            pair = self.named['pen', 'validation', presence]
            fact = self.fact(BULK, pair)
            self.assertEqual(fact.positive_layers, tuple(layer for layer, value in
                ((BULK, presence[0]), (SC, presence[1])) if value == '1'))
            self.assertTrue(set(pair) <= self.view.layers[BULK].nodes & self.view.layers[SC].nodes)
        pair = self.named[CD4, 'train', 'one_way']
        self.assertIn(CD4, self.fact(PPI, pair).positive_layers)
        self.assertNotIn(CD4, self.fact(PPI, pair[::-1]).positive_layers)

    def test_optional_null_and_ambiguous_symbols_do_not_remove_native_pairs(self):
        null = self.fact(CD4, self.named['null_edge'])
        ambiguous = self.fact(COEX, self.named['ambiguous_edge'])
        self.assertEqual(null.registry_states[0], 'null_symbol')
        self.assertEqual(ambiguous.registry_states[0], 'ambiguous_symbol')
        self.assertEqual((null.category, ambiguous.category), ('recorded', 'recorded'))
        self.assertEqual(len(self.registry.gene_ids_by_symbol['ambiguous_fixture']), 2)

    def test_source_allocation_is_deterministic_and_has_exact_fact_digest(self):
        repeated, audit = select_facts(self.view, self.index, self.registry,
            self.counts, SEED, smoke=True, old_view=self.old)
        self.assertEqual((self.facts, self.audit), (repeated, audit))
        self.assertEqual(self.audit['fact_selection_sha256'], digest([asdict(fact) for fact in self.facts]))
        self.assertEqual(len(self.facts), len({(fact.layer_id, fact.gene_ids) for fact in self.facts}))

    def test_original_pair_baselines_and_every_real_stratum_are_retained(self):
        baseline = select_edges(self.view, sorted(self.registry.symbols_by_id), self.counts, SEED)
        self.assertEqual(self.audit['baseline_facts'], len(baseline))
        self.assertEqual(len(baseline), 6 * 4 * 3)
        selected = {(fact.layer_id, fact.gene_ids): fact for fact in self.facts}
        for fact in baseline:
            self.assertIn((fact.layer_id, fact.gene_ids), selected)
            self.assertEqual(selected[fact.layer_id, fact.gene_ids].category, fact.bucket)
        self.assertEqual({(fact.split, fact.layer_id, fact.category) for fact in self.facts},
            {(split, layer, bucket) for split in SPLITS for layer in self.view.layers for bucket in BUCKETS})

    def test_source_domain_counts_exclude_self_pairs_and_count_native_orientations(self):
        self.assertEqual(len(self.audit['domains']), 6)
        for domain in self.audit['domains']:
            graph = self.view.layers[domain['layer_id']]
            eligible = sum(left != right for left, right in graph.edges)
            possible = len(graph.nodes) * (len(graph.nodes) - 1) // (1 if graph.directed else 2)
            self.assertEqual(domain['possible_distinct_endpoint_pairs'], possible)
            self.assertEqual(domain['native_source_records'], len(graph.edges))
            self.assertEqual(domain['eligible_distinct_endpoint_edges'], eligible)
            self.assertEqual(domain['checked_member_nonedge_population'], possible - eligible)
            self.assertEqual(domain['self_loops_excluded_from_pair_queries'], len(graph.edges) - eligible)
        self.assertTrue(all(first != second for fact in self.facts for first, second in [fact.gene_ids]))

    def test_native_feature_witnesses_are_real_and_high_weights_stay_within_layer(self):
        selected = {(fact.layer_id, fact.gene_ids): fact for fact in self.facts}
        features = set()
        for witness in self.audit['feature_witnesses']:
            fact = selected[witness['layer_id'], tuple(witness['gene_ids'])]
            self.assertEqual((fact.split, fact.category), (witness['split'], witness['category']))
            feature = witness['feature']
            features.add(feature)
            if feature == 'high_weight':
                self.assertEqual(fact.weight, .99)
            elif feature in ('reciprocal', 'one_way'):
                self.assertEqual(fact.category, 'recorded')
                self.assertIs(fact.reverse_exists, feature == 'reciprocal')
            elif feature.startswith('registry:'):
                self.assertIn(feature.removeprefix('registry:'), fact.registry_states)
            elif feature == 'two_edge_route_without_direct_record':
                graph = self.view.layers[fact.layer_id]
                self.assertNotIn(tuple(sorted(fact.gene_ids)), graph.edges)
                a, b = fact.gene_ids
                self.assertTrue({gene for gene, _ in self.index.native_index.neighbors(fact.layer_id, a, 'undirected')}
                    & {gene for gene, _ in self.index.native_index.neighbors(fact.layer_id, b, 'undirected')})
        self.assertTrue({'high_weight', 'one_way', 'reciprocal', 'two_edge_route_without_direct_record',
            'registry:unique_symbol', 'registry:null_symbol', 'registry:ambiguous_symbol', 'registry:graph_only',
            'pen_contrast:11', 'pen_contrast:10', 'pen_contrast:01'} <= features)
        self.assertFalse(self.audit['unattained_features'])

    def test_pen_contrasts_match_both_known_memberships_and_keep_one_split(self):
        witnesses = defaultdict(list)
        for item in self.audit['feature_witnesses']:
            if item['feature'].startswith('pen_contrast:'):
                witnesses[item['feature'], item['split']].append(item)
                pair = item['gene_ids']
                self.assertTrue(set(pair) <= self.view.layers[BULK].nodes & self.view.layers[SC].nodes)
        self.assertEqual(set(witnesses), {(f'pen_contrast:{bits}', split)
            for bits in ('11', '10', '01') for split in SPLITS})
        for records in witnesses.values():
            self.assertEqual({item['layer_id'] for item in records}, {BULK, SC})
            self.assertEqual(len({tuple(item['gene_ids']) for item in records}), 1)

    def test_all_orientations_and_scopes_of_a_selected_pair_keep_one_split(self):
        splits = defaultdict(set)
        for fact in self.facts:
            splits[tuple(sorted(fact.gene_ids))].add(fact.split)
            self.assertEqual((fact.split_unit, fact.split), _pair_group(fact.gene_ids, SEED))
        self.assertTrue(all(len(values) == 1 for values in splits.values()))
        self.assertTrue(self.audit['query_pair_disjoint'])
        self.assertFalse(self.audit['gene_disjoint'])
        self.assertFalse(self.audit['edge_fact_disjoint'])

    def test_explicit_coverage_gaps_do_not_claim_exhaustive_other_sources(self):
        self.assertFalse(self.audit['all_possible_pairs_enumerated'])
        self.assertIsNone(self.audit['runtime_pair_cap'])
        self.assertTrue(self.audit['all_source_endpoints_retained'])
        self.assertEqual(self.audit['taxonomy_only_families'], ['other'])
        self.assertEqual(self.audit['unqualified_sources'], ['HumanNetV3:cocitation'])
        self.assertNotIn('HumanNetV3:cocitation', {fact.layer_id for fact in self.facts})

    def test_source_choices_do_not_change_with_verification_label(self):
        facts, audit = select_facts(self.view, self.index, self.registry,
            self.counts, SEED, smoke=False, old_view=self.old)
        self.assertEqual(facts, self.facts)
        self.assertEqual(audit['fact_selection_sha256'], self.audit['fact_selection_sha256'])
        self.assertNotEqual(audit['smoke'], self.audit['smoke'])

    def test_invalid_quotas_and_pair_shapes_are_rejected(self):
        for counts in ({}, {'train': 1, 'validation': 1}, dict.fromkeys(SPLITS, 0),
                       dict.fromkeys(SPLITS, True), dict.fromkeys(SPLITS, 1.2)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                select_facts(self.view, self.index, self.registry, counts, SEED, old_view=self.old)
        gene = self.named['zero'][0]
        for pair in ([], [gene], [gene, gene], [gene, gene, gene]):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                self.fact(PPI, pair)

    def test_separate_python_hash_seeds_produce_identical_fact_and_audit_bytes(self):
        script = '''
import json,sys
from dataclasses import asdict
sys.path.insert(0, 'tests')
from test_world_model_s2_13_sampling import fixture,SEED
from runtime.world_model.s2_13_sampling import select_facts
from runtime.world_model.s2_corpus import SPLITS
view,index,registry,old,_=fixture()
facts,audit=select_facts(view,index,registry,dict.fromkeys(SPLITS,1),SEED,smoke=True,old_view=old)
print(json.dumps({'facts':[asdict(f) for f in facts], 'audit':audit},separators=(',',':')))
'''
        outputs = []
        for seed in ('1', '2'):
            environment = dict(os.environ, PYTHONHASHSEED=seed, USE_TORCH='0', USE_TF='0', USE_FLAX='0')
            outputs.append(subprocess.check_output([sys.executable, '-c', script],
                cwd=Path(__file__).resolve().parents[1], env=environment))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
