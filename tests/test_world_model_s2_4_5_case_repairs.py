"""Regress the exact native/family context combinations found by the S2 audit."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model import s2_4_case_repairs as maps, s2_5_case_repairs as memberships
from runtime.world_model.s2_audit_cases_3_6 import matches
from runtime.world_model.s2_corpus import canonical_json
from runtime.world_model.s2_neighbors import NeighborIndex
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_subgraphs import SubgraphIndex


ROOT = Path(__file__).resolve().parents[1]
PPI, COEX, BRAIN, CD4 = 'HumanNetV3:string_ppi', 'HumanNetV3:coexpression', 'TFs:brain', 'TFs:cd4+t_cells'


class CaseRepairTests(unittest.TestCase):
    """Use independent small rings with positive evidence in every fixed split."""

    @classmethod
    def setUpClass(cls):
        genes = [f'ENSG{i:011d}' for i in range(1, 81)]
        edges = {(gene, genes[(number + 1) % len(genes)]): .25 for number, gene in enumerate(genes)}
        undirected = {tuple(sorted(pair)): weight for pair, weight in edges.items()}
        cls.view = GraphView('case_repair_unit', 'e' * 64, {
            layer: LayerGraph(layer, directed, edges if directed else undirected, frozenset(genes))
            for layer, directed in ((PPI, False), (COEX, False), (BRAIN, True), (CD4, True))})
        cls.index = NeighborIndex(cls.view)
        cls.bank = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
        cls.contracts = {maps: maps.load_build_contract(), memberships: memberships.load_build_contract()}
        cls.rows = {}
        for module in (maps, memberships):
            for split in ('train', 'validation', 'test'):
                rows, _ = module.build_case_rows(cls.view, cls.index, cls.bank, cls.contracts[module], split=split)
                cls.rows[module, split] = rows

    def validate(self, module, row):
        module.validate_row(row, self.view, index=self.index, bank=self.bank, contract=self.contracts[module])

    def test_all_seven_cases_are_source_bound_in_every_original_split(self):
        for (module, split), rows in self.rows.items():
            self.assertEqual([row['metadata']['catalog_case_id'] for row in rows], list(module.CASE_IDS))
            for row in rows:
                with self.subTest(case=row['metadata']['catalog_case_id'], split=split):
                    self.validate(module, row)
                    self.assertTrue(matches(row['metadata']['catalog_case_id'], row))
                    self.assertEqual(row['metadata']['split'], split)

    def test_aggregate_subgraph_index_preserves_native_index_rows_byte_for_byte(self):
        aggregate_index = SubgraphIndex(self.view)
        for (module, split), original in self.rows.items():
            with self.subTest(module=module.__name__, split=split):
                rebuilt, _ = module.build_case_rows(self.view, aggregate_index, self.bank,
                                                    self.contracts[module], split=split)
                self.assertEqual(canonical_json(original), canonical_json(rebuilt))
                for row in original:
                    module.validate_row(row, self.view, index=aggregate_index, bank=self.bank,
                                        contract=self.contracts[module])

    def test_native_failures_keep_atomic_inputs_and_only_acquire_missing_work(self):
        for row in self.rows[maps, 'train'][:3]:
            result = row['messages'][-1]['content']['result']
            checked, unchecked = result['coverage']['checked_inputs'], result['coverage']['unchecked_inputs']
            self.assertTrue(checked)
            self.assertTrue(unchecked)
            self.assertEqual(result['coverage']['requested_inputs'], checked + unchecked)
            case = row['metadata']['catalog_case_id']
            if case == 'S2.4.20':
                self.assertEqual(result['failed_tool'], 'get_neighbors')
                self.assertEqual(result['arguments']['direction'], 'out')
                self.assertEqual([entry['direction'] for entry in checked], ['undirected', 'in'])
            else:
                self.assertEqual(result['failed_tool'], 'get_neighbor_layers')
                self.assertEqual(len(result['arguments']['layer_scope']['layer_ids']), 1)
        row = deepcopy(self.rows[maps, 'train'][0])
        coverage = row['messages'][-1]['content']['result']['coverage']
        coverage['checked_inputs'].append(coverage['unchecked_inputs'].pop())
        with self.assertRaises(ValueError):
            self.validate(maps, row)

    def test_missing_original_observation_or_changed_native_weight_is_rejected(self):
        for mutate in ('drop', 'weight', 'version'):
            row = deepcopy(self.rows[maps, 'train'][2])
            context = json.loads(row['messages'][1]['content'])
            if mutate == 'drop':
                context['observations'].pop()
            elif mutate == 'weight':
                context['observations'][0]['result']['neighbors'][0]['weight'] = 999
            else:
                context['observations'][0]['provenance']['graph'][0]['graph_version'] = '0' * 64
            row['messages'][1]['content'] = canonical_json(context)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.validate(maps, row)

    def test_bounded_strongest_case_preserves_two_and_avoids_calls(self):
        original = self.rows[maps, 'test'][3]
        request = original['messages'][-1]['content']['result']['request']
        self.assertEqual((request['mode'], request['top_k']), ('top_k', 2))
        self.assertIn('2', original['messages'][2]['content'])
        self.assertFalse(any('tool_call' in message for message in original['messages']))
        for field, value in (('mode', None), ('top_k', None), ('top_k', 1)):
            row = deepcopy(original)
            row['messages'][-1]['content']['result']['request'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.validate(maps, row)

    def test_bounded_ranking_wording_respects_every_existing_held_out_bank(self):
        report = maps.audit_request_wording(self.contracts[maps], self.bank)
        self.assertLess(report['maximum_cross_split_jaccard'], .6)
        self.assertEqual(report['shared_cross_split_5grams'], 0)
        self.assertEqual(len(report['reference_request_pins']), 15)

    def test_family_failure_retry_and_exhaustion_preserve_scope_and_provenance(self):
        for row in self.rows[memberships, 'validation']:
            context = json.loads(row['messages'][1]['content'])['observations']
            saved = context[0]
            self.assertEqual(saved['arguments']['layer_scope']['layer_family'], 'tf_target')
            self.assertTrue(saved['result']['collection']['complete'])
            self.assertGreater(saved['result']['layer_count'], 0)
            final = row['messages'][-1]['content']
            self.assertIn(saved['provenance']['graph'][0], final['provenance']['graph'])
            for message in row['messages']:
                if 'tool_call' in message:
                    self.assertEqual(message['tool_call']['arguments']['layer_scope']['layer_ids'], [PPI, COEX])
        row = deepcopy(self.rows[memberships, 'validation'][0])
        context = json.loads(row['messages'][1]['content'])
        context['observations'][0]['arguments']['layer_scope'] = {
            'kind': 'layers', 'layer_ids': [BRAIN, CD4], 'layer_family': None}
        row['messages'][1]['content'] = canonical_json(context)
        with self.assertRaises(ValueError):
            self.validate(memberships, row)

    def test_case_relabeling_retry_changes_and_split_changes_are_rejected(self):
        for module, original, mutation in (
            (maps, self.rows[maps, 'train'][0], 'case'),
            (memberships, self.rows[memberships, 'train'][0], 'case'),
            (memberships, self.rows[memberships, 'train'][1], 'retry'),
            (memberships, self.rows[memberships, 'train'][2], 'split'),
        ):
            row = deepcopy(original)
            if mutation == 'case':
                row['metadata']['catalog_case_id'] = module.CASE_IDS[1]
            elif mutation == 'retry':
                call = next(m['tool_call'] for m in row['messages'][5:] if 'tool_call' in m)
                call['arguments']['cursor'] = 'invented'
            else:
                row['metadata']['split'] = 'test'
            with self.subTest(module=module.__name__, mutation=mutation), self.assertRaises(ValueError):
                self.validate(module, row)

    def test_free_summary_and_blocker_wording_remain_unscored(self):
        for module in (maps, memberships):
            row = deepcopy(self.rows[module, 'train'][0])
            row['messages'][-1]['content'].update(summary='Earlier checked evidence remains supported.',
                                                  blocked_reason='Backend recovery is needed.')
            self.validate(module, row)

    def test_forged_row_identity_and_top_level_private_data_are_rejected(self):
        for module in (maps, memberships):
            row = deepcopy(self.rows[module, 'train'][0])
            row['row_id'] += '_changed'
            with self.assertRaises(ValueError):
                self.validate(module, row)
            row = deepcopy(self.rows[module, 'train'][0])
            row['hidden_answer'] = True
            with self.assertRaises(ValueError):
                self.validate(module, row)

    def test_proof_decisions_can_run_without_source_edges_or_node_inventories(self):
        row = self.rows[maps, 'train'][2]
        context = json.loads(row['messages'][1]['content'])['observations']
        meta = row['metadata']
        arguments = {'multiplex_id': self.view.multiplex_id, 'gene_id': meta['gene_id'],
            'layer_scope': meta['layer_scope'], 'direction': 'incident', 'mode': 'complete', 'top_k': None, 'cursor': None}
        visible = maps._visible(self.view)
        self.assertFalse(hasattr(visible.layers[PPI], 'nodes'))
        self.assertFalse(hasattr(visible.layers[PPI], 'edges'))
        self.assertEqual(maps._native_coverage(visible, arguments, context),
                         row['messages'][-1]['content']['result']['coverage'])


if __name__ == '__main__':
    unittest.main()
