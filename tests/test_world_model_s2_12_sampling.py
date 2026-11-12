"""Check finite membership profiles, fixed splits, and real source witnesses."""

from collections import Counter, defaultdict
from dataclasses import asdict
from types import MappingProxyType
import unittest
from unittest.mock import patch

from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_3_corpus import gene_group
from runtime.world_model.s2_12_sampling import fact_from_gene, select_facts
from runtime.world_model.s2_corpus import SPLITS, digest
from runtime.world_model.s2_nodes_by_layer import NodesByLayerRuntime
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


PPI, COEX, BRAIN, CD4 = ('HumanNetV3:string_ppi', 'HumanNetV3:coexpression',
                         'TFs:brain', 'TFs:cd4+t_cells')
BULK, SC = 'bulkPEN:brain:amygdala', 'scPEN:brain:extended-amygdala:astrocyte'
SEED = 20261028


def fixture():
    """Create genuine endpoint inventories and identity contrasts in every split."""
    roles = ('source', 'target', 'loop', 'bulk_a', 'bulk_b', 'sc_a', 'sc_b',
             'shared_pen', 'graph_only', 'null_symbol', 'ambiguous_a', 'ambiguous_b',
             'registry_only', 'ppi_peer', 'coex_peer', 'all_layers', 'extra_ppi', 'extra_missing')
    pools = {split: [] for split in SPLITS}
    for number in range(1000, 3000):
        gene = f'ENSG{number:011d}'
        pools[gene_group(gene, SEED)[1]].append(gene)
    names = {split: dict(zip(roles, genes[:len(roles)])) for split, genes in pools.items()}
    edges = {layer: {} for layer in (PPI, COEX, BRAIN, CD4, BULK, SC)}
    symbols = {}
    for split, genes in names.items():
        for role, gene in genes.items():
            if role != 'graph_only':
                symbols[gene] = None if role == 'null_symbol' else (
                    f'DUP_{split}' if role.startswith('ambiguous_') else f'SYMBOL_{gene}')
        pairs = {
            PPI: [('source', 'target'), ('source', 'graph_only'), ('all_layers', 'ppi_peer'),
                  ('extra_ppi', 'ppi_peer')],
            COEX: [('target', 'coex_peer'), ('ambiguous_a', 'coex_peer'), ('all_layers', 'coex_peer')],
            BRAIN: [('source', 'target'), ('loop', 'loop'), ('all_layers', 'target')],
            CD4: [('source', 'null_symbol'), ('all_layers', 'null_symbol')],
            BULK: [('bulk_a', 'bulk_b'), ('shared_pen', 'bulk_a'), ('all_layers', 'bulk_b')],
            SC: [('sc_a', 'sc_b'), ('shared_pen', 'sc_a'), ('all_layers', 'sc_b')],
        }
        for layer, rows in pairs.items():
            for position, (left, right) in enumerate(rows):
                pair = genes[left], genes[right]
                if layer not in (BRAIN, CD4):
                    pair = tuple(sorted(pair))
                edges[layer][pair] = position / 10
    native = {layer: LayerGraph(layer, layer in (BRAIN, CD4), rows,
              frozenset(gene for pair in rows for gene in pair)) for layer, rows in edges.items()}
    view = GraphView('unit_s212_six_files', '6' * 64, native)
    old = GraphView('unit_s212_four_files', '4' * 64,
                    {layer: graph for layer, graph in native.items() if layer not in (BULK, SC)})
    registry = RegistryContext('116', 'ensembl_homo_sapiens_116_unit', symbols)
    return view, SubgraphIndex(view), registry, old, names


class MembershipSamplingTests(unittest.TestCase):
    """Check source choices independently of prompts and request schedules."""

    def setUp(self):
        self.view, self.index, self.registry, self.old, self.names = fixture()
        self.counts = dict.fromkeys(SPLITS, 1)

    def select(self, **kwargs):
        """Use explicit quotas and the separately pinned old node universe."""
        return select_facts(self.view, self.index, self.registry, self.counts,
                            SEED, old_view=self.old, **kwargs)

    def fact(self, role, layer=BRAIN, split='train'):
        """Measure one named source fact without a request or wrapper bank."""
        return fact_from_gene(self.view, self.index, self.registry, self.old,
                              layer, self.names[split][role], SEED)

    def test_three_categories_use_endpoint_sets_and_not_registry_presence(self):
        self.assertEqual(self.fact('source').category, 'member')
        self.assertEqual(self.fact('bulk_a').category, 'outside_layer')
        missing = self.fact('registry_only')
        self.assertEqual(missing.category, 'outside_multiplex')
        self.assertIsNone(missing.outgoing_count)
        self.assertIsNone(missing.incoming_count)
        graph_only = self.fact('graph_only', PPI)
        self.assertEqual((graph_only.category, graph_only.registry_state), ('member', 'graph_only'))

    def test_identity_features_preserve_null_and_ambiguous_symbols(self):
        self.assertEqual(self.fact('null_symbol', CD4).registry_state, 'null_symbol')
        self.assertEqual(self.fact('ambiguous_a', COEX).registry_state, 'ambiguous_symbol')
        self.assertEqual(self.fact('ambiguous_b', COEX).category, 'outside_multiplex')
        self.assertEqual(self.fact('ambiguous_b', COEX).registry_state, 'ambiguous_symbol')
        self.assertEqual(self.fact('source').registry_state, 'unique_symbol')

    def test_old_global_negative_changes_only_under_the_new_node_union(self):
        gene = self.names['train']['bulk_a']
        current = self.fact('bulk_a', BULK)
        self.assertEqual(current.category, 'member')
        self.assertFalse(current.old_present)
        self.assertNotIn(gene, self.old.multiplex_nodes)
        old_fact = fact_from_gene(self.old, SubgraphIndex(self.old), self.registry,
                                  self.old, PPI, gene, SEED)
        self.assertEqual(old_fact.category, 'outside_multiplex')
        self.assertEqual(old_fact.layer_ids, ())
        self.assertEqual((old_fact.split, old_fact.split_unit), (current.split, current.split_unit))
        self.assertTrue(self.fact('source').old_present)

    def test_zero_outgoing_and_selected_isolation_preserve_real_membership(self):
        target = self.fact('target')
        self.assertEqual((target.category, target.outgoing_count, target.incoming_count), ('member', 0, 2))
        source = self.fact('source')
        self.assertEqual((source.category, source.outgoing_count, source.incoming_count), ('member', 1, 0))
        self.assertFalse(target.self_loop)
        self.assertEqual(self.index.edges(BRAIN, [target.gene_id]), [])
        loop = self.fact('loop')
        self.assertEqual((loop.category, loop.outgoing_count, loop.incoming_count), ('member', 1, 1))
        self.assertTrue(loop.self_loop)
        self.assertEqual(self.index.edges(BRAIN, [loop.gene_id]),
                         [{'gene_ids': [loop.gene_id, loop.gene_id], 'weight': 0.1}])
        for layer in self.view.layers.values():
            self.assertEqual(layer.nodes, frozenset(gene for pair in layer.edges for gene in pair))

    def test_selection_is_deterministic_and_has_an_exact_fact_digest(self):
        facts, audit = self.select()
        repeated, repeated_audit = self.select()
        self.assertEqual((facts, audit), (repeated, repeated_audit))
        self.assertEqual(audit['fact_selection_sha256'], digest([asdict(fact) for fact in facts]))
        self.assertEqual(len({(fact.gene_id, fact.layer_id) for fact in facts}), len(facts))

    def test_every_selected_gene_has_all_six_layer_counterparts_in_one_split(self):
        facts, audit = self.select()
        by_gene = defaultdict(list)
        for fact in facts:
            by_gene[fact.gene_id].append(fact)
            self.assertEqual((fact.split_unit, fact.split), gene_group(fact.gene_id, SEED))
        for counterparts in by_gene.values():
            self.assertEqual(tuple(fact.layer_id for fact in counterparts), self.view.layer_order)
            self.assertEqual(len({fact.split for fact in counterparts}), 1)
            self.assertEqual(len({fact.split_unit for fact in counterparts}), 1)
            self.assertEqual(len({fact.layer_ids for fact in counterparts}), 1)
        self.assertEqual((audit['selected_genes'], audit['selected_facts']),
                         (len(by_gene), len(by_gene) * len(self.view.layers)))

    def test_profile_quotas_cover_only_attainable_complete_profiles(self):
        _, audit = self.select()
        candidates = self.view.multiplex_nodes | set(self.registry.symbols_by_id)
        expected = Counter((gene_group(gene, SEED)[1], tuple(layer for layer in self.view.layer_order
                            if gene in self.view.layers[layer].nodes)) for gene in candidates)
        observed = {(cell['split'], tuple(cell['layer_ids'])): cell for cell in audit['profile_cells']}
        self.assertEqual(set(observed), set(expected))
        for key, population in expected.items():
            cell = observed[key]
            self.assertEqual(cell['population'], population)
            self.assertEqual(cell['containing_layer_count'], len(key[1]))
            self.assertEqual(cell['requested'], self.counts[key[0]])
            self.assertEqual(cell['selected'], min(population, self.counts[key[0]]))
        self.assertTrue(any(not profile for _, profile in expected))
        self.assertTrue(any(len(profile) == 6 for _, profile in expected))
        self.assertTrue(audit['all_finite_membership_profiles_audited'])
        self.assertFalse(audit['all_possible_absent_identifiers_enumerated'])

    def test_domain_histograms_include_registry_negatives_and_graph_only_members(self):
        facts, audit = self.select()
        candidates = self.view.multiplex_nodes | set(self.registry.symbols_by_id)
        self.assertEqual(audit['candidate_genes'], len(candidates))
        self.assertEqual(audit['source_endpoint_genes'], len(self.view.multiplex_nodes))
        self.assertEqual(audit['registry_genes'], len(self.registry.symbols_by_id))
        self.assertEqual(audit['new_view_only_genes'], len(self.view.multiplex_nodes - self.old.multiplex_nodes))
        self.assertEqual(len(audit['domains']), len(SPLITS) * len(self.view.layers))
        for domain in audit['domains']:
            genes = [gene for gene in candidates if gene_group(gene, SEED)[1] == domain['split']]
            nodes = self.view.layers[domain['layer_id']].nodes
            expected = Counter('member' if gene in nodes else 'outside_layer'
                               if gene in self.view.multiplex_nodes else 'outside_multiplex' for gene in genes)
            self.assertEqual(domain['membership_histogram'], dict(expected))
            self.assertEqual(domain['population'], len(genes))
            self.assertEqual(domain['full_layer_members'], len(nodes))
            self.assertEqual(domain['full_layer_edges'], len(self.view.layers[domain['layer_id']].edges))
        self.assertTrue(all(fact.gene_id in candidates for fact in facts))
        self.assertTrue(any(fact.registry_state == 'graph_only' for fact in facts))
        self.assertNotIn('ENSG99999999999', {fact.gene_id for fact in facts})

    def test_feature_witnesses_are_selected_and_match_native_source_facts(self):
        facts, audit = self.select()
        by_key = {(fact.layer_id, fact.gene_id): fact for fact in facts}
        features = {row['feature'] for row in audit['feature_witnesses']}
        self.assertTrue({'registry:graph_only', 'registry:null_symbol', 'registry:ambiguous_symbol',
                         'registry:unique_symbol', 'new_view_only', 'self_loop', 'edgeless_singleton',
                         'zero_outgoing_member', 'zero_incoming_member'} <= features)
        for row in audit['feature_witnesses']:
            fact = by_key[row['layer_id'], row['gene_id']]
            self.assertEqual((fact.split, fact.category), (row['split'], row['category']))
            tag = row['feature']
            if tag.startswith('registry:'):
                self.assertEqual(fact.registry_state, tag.split(':', 1)[1])
            elif tag == 'new_view_only':
                self.assertTrue(fact.layer_ids and not fact.old_present)
            else:
                self.assertEqual(fact.category, 'member')
                if tag in ('self_loop', 'edgeless_singleton'):
                    edges = self.index.edges(fact.layer_id, [fact.gene_id])
                    self.assertEqual(bool(edges), tag == 'self_loop')
                elif tag == 'zero_outgoing_member':
                    self.assertTrue(self.view.layers[fact.layer_id].directed)
                    self.assertEqual(fact.outgoing_count, 0)
                    self.assertGreater(fact.incoming_count, 0)
                elif tag == 'zero_incoming_member':
                    self.assertTrue(self.view.layers[fact.layer_id].directed)
                    self.assertEqual(fact.incoming_count, 0)
                    self.assertGreater(fact.outgoing_count, 0)

    def test_every_selected_category_matches_the_native_membership_operation(self):
        facts, _ = self.select()
        runtime = NodesByLayerRuntime(self.view)
        partition = {'member': 'layer_gene_ids', 'outside_layer': 'not_in_layer_gene_ids',
                     'outside_multiplex': 'missing_gene_ids'}
        for fact in facts:
            response = runtime.get_nodes_by_layer(self.view.multiplex_id, fact.layer_id,
                                                  [fact.gene_id], evidence_id='source_audit')
            self.assertEqual(response['status'], 'resolved')
            self.assertEqual(response['result'][partition[fact.category]], [fact.gene_id])
            self.assertEqual(response['result']['layer_gene_count'], int(fact.category == 'member'))

    def test_shortfalls_are_reported_without_synthetic_profile_members(self):
        self.counts = dict.fromkeys(SPLITS, 100)
        facts, audit = self.select()
        self.assertEqual(audit['quota_shortfalls'], audit['profile_cells'])
        self.assertEqual({fact.gene_id for fact in facts},
                         self.view.multiplex_nodes | set(self.registry.symbols_by_id))
        self.assertTrue(all(cell['selected'] == cell['population'] < cell['requested']
                            for cell in audit['quota_shortfalls']))

    def test_source_selection_reads_no_wrapper_bank_and_smoke_changes_no_facts(self):
        expected, audit = self.select()
        with patch('runtime.world_model.s1_prompt_banks.load_s2_prompt_banks',
                   side_effect=AssertionError('source choices must not read wrappers')):
            actual, smoke_audit = self.select(smoke=True)
        self.assertEqual(expected, actual)
        self.assertEqual(audit['fact_selection_sha256'], smoke_audit['fact_selection_sha256'])
        self.assertTrue(smoke_audit['smoke'])

    def test_sampling_preserves_graph_registry_and_quota_inputs(self):
        before = [(view.multiplex_id, tuple((name, tuple(layer.edges.items()), layer.nodes)
                  for name, layer in view.layers.items())) for view in (self.view, self.old)]
        symbols, candidates = dict(self.registry.symbols_by_id), dict(self.registry.gene_ids_by_symbol)
        counts = dict(self.counts)
        self.counts = MappingProxyType(self.counts)
        self.select()
        after = [(view.multiplex_id, tuple((name, tuple(layer.edges.items()), layer.nodes)
                 for name, layer in view.layers.items())) for view in (self.view, self.old)]
        self.assertEqual(before, after)
        self.assertEqual((symbols, candidates), (dict(self.registry.symbols_by_id), dict(self.registry.gene_ids_by_symbol)))
        self.assertEqual(counts, self.counts)

    def test_invalid_profile_quotas_are_rejected(self):
        invalid = ({}, {'train': 1}, {**self.counts, 'extra': 1},
                   *[dict.fromkeys(SPLITS, value) for value in (0, -1, True, 1.5, '1', None)])
        for counts in invalid:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                select_facts(self.view, self.index, self.registry, counts, SEED, old_view=self.old)


if __name__ == '__main__':
    unittest.main()
