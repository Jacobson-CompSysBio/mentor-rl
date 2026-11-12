"""Check complete S2.1 generator behaviors against readable native fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from runtime.world_model.s2_1_production import (
    CASE_BINDINGS, build_case_rows, validate_case_row,
)
from runtime.world_model.s2_2_identity import RegistryContext
from runtime.world_model.s2_sources import GraphView, LayerGraph
from runtime.world_model.s2_corpus import _pair_group


ROOT = Path(__file__).resolve().parents[1]


def gene(number):
    """Construct fixture identifiers only; these are never production inputs."""
    return f'ENSG{number:011d}'


def fixture():
    """Supply many tiny components, one-way forks, sinks, and reciprocal pairs."""
    undirected, directed = {}, {}
    for start in range(1, 181, 3):
        a, b, c = map(gene, (start, start + 1, start + 2))
        undirected[(a, b)] = .9
        undirected[(a, c)] = .4
    for start in range(61, 601, 3):
        a, b, c = map(gene, (start, start + 1, start + 2))
        directed[(c, a)] = .8
        directed[(c, b)] = .3
    for start in range(61, 180):
        a, b = gene(start), gene(start + 1)
        directed[(a, b)] = .7
        directed[(b, a)] = .2
    ppi, tf = 'HumanNetV3:string_ppi', 'TFs:brain'
    layers = {ppi: LayerGraph(ppi, False, undirected,
                             frozenset(g for pair in undirected for g in pair)),
              tf: LayerGraph(tf, True, directed,
                            frozenset(g for pair in directed for g in pair))}
    view = GraphView('test-only-s21', 'a' * 64, layers)
    registry = RegistryContext('116', 'ensembl_homo_sapiens_116_fixture',
                               {gene(i): f'GENE{i}' for i in range(1, 801)})
    contract = json.loads((ROOT / 'runtime/world_model/s2_1_production_v1.json').read_text())
    contract['_bank'] = json.loads((ROOT / 'runtime/world_model/s2_v1_prompt_banks.json').read_text())
    contract['_requests'] = json.loads((ROOT / 'runtime/world_model/s2_1_production_v1_requests.json').read_text())
    contract['_registry'] = registry
    return view, registry, contract


def calls(row):
    """Return model calls in order, excluding supplied observations."""
    return [message['tool_call'] for message in row['messages'] if 'tool_call' in message]


class CompleteEdgeGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Generate all reviewed behaviors in every fixed split once."""
        cls.view, cls.registry, cls.contract = fixture()
        cls.rows = {split: {row['metadata']['catalog_case_id']: row for row in
                           build_case_rows(cls.view, cls.contract, split=split)}
                    for split in ('train', 'validation', 'test')}

    def test_all_41_cases_are_bound_and_source_backed_in_every_split(self):
        """No numbered case can disappear under generator variant counting."""
        for split, by_case in self.rows.items():
            self.assertEqual(set(by_case), {f'S2.1.{i}' for i in range(1, 42)})
            for case, row in by_case.items():
                validate_case_row(row, self.view, contract=self.contract)
                self.assertEqual(row['metadata']['split'], split)
                self.assertEqual(row['metadata']['framing'], CASE_BINDINGS[case]['framing'])

    def test_list_evidence_requires_membership_and_respects_truncation(self):
        """Complete negatives need membership; truncated positives remain usable."""
        rows = self.rows['train']
        for number in (8, 10, 11, 14, 24, 25, 26, 35):
            self.assertEqual(calls(rows[f'S2.1.{number}']), [], number)
        for number in (9, 12, 13, 20, 23, 27):
            self.assertEqual([c['name'] for c in calls(rows[f'S2.1.{number}'])], ['get_edge'], number)
        for number in (9, 27):
            context = json.loads(rows[f'S2.1.{number}']['messages'][1]['content'])['observations'][0]
            self.assertFalse(context['result']['collection']['complete'])
            target = rows[f'S2.1.{number}']['metadata']['gene_ids'][1]
            self.assertNotIn(target, [item['gene_id'] for item in context['result']['neighbors']])
        zero = json.loads(rows['S2.1.26']['messages'][1]['content'])['observations'][0]['result']
        self.assertEqual(zero['neighbors'], [])
        self.assertEqual(zero['missing_gene_ids'] + zero['not_in_layer_gene_ids'], [])

    def test_pair_classification_and_early_stop_have_exact_ordered_coverage(self):
        """Reference order, reciprocity, native weights, and unknowns stay distinct."""
        rows = self.rows['test']
        expected = {31: 'forward_only', 32: 'reverse_only', 33: 'both',
                    34: 'neither', 35: 'forward_only', 37: 'undirected', 41: 'forward_only'}
        for number, direction in expected.items():
            result = rows[f'S2.1.{number}']['messages'][-1]['content']['result']
            self.assertEqual(result['directionality'], direction)
            self.assertEqual(result['coverage']['unchecked_pairs'], [])
            layer = self.view.layers[result['layer_id']]
            for edge in result['edge_results']:
                pair = tuple(edge['gene_ids'])
                self.assertEqual(edge['weight'], layer.edges.get(pair))
        for number in (38, 39):
            result = rows[f'S2.1.{number}']['messages'][-1]['content']['result']
            self.assertEqual(len(calls(rows[f'S2.1.{number}'])), 1)
            self.assertEqual(len(result['coverage']['unchecked_pairs']), 1)
            self.assertIsNone(result['directionality'])
        missing = rows['S2.1.40']['messages'][-1]['content']
        self.assertEqual(missing['status'], 'not_found')
        self.assertEqual(missing['result']['coverage']['checked_pairs'], [])
        self.assertIsNone(missing['result']['any_edge_exists'])

    def test_identity_and_failures_preserve_original_observations_and_retries(self):
        """Successful registry/list/forward evidence survives required failure."""
        rows = self.rows['validation']
        self.assertEqual([c['name'] for c in calls(rows['S2.1.16'])],
                         ['lookup_human_gene_symbol', 'lookup_human_gene_symbol', 'get_edge'])
        for number, count in ((15, 2), (18, 0), (19, 1), (36, 1)):
            answer = rows[f'S2.1.{number}']['messages'][-1]['content']
            self.assertEqual((answer['status'], answer['continuation_decision']), ('error', 'defer'))
            self.assertEqual(len(answer['result']['partial_results']), count)
        for number in (17, 18, 41):
            chosen = calls(rows[f'S2.1.{number}'])
            self.assertEqual(chosen[-1], chosen[-2])

    def test_saved_distractor_pairs_keep_their_fixed_source_split(self):
        """A distractor cannot expose a differently assigned held-out pair."""
        for split, rows in self.rows.items():
            row = rows['S2.1.14']
            observations = json.loads(row['messages'][1]['content'])['observations']
            self.assertEqual(len(observations), 2)
            self.assertNotEqual(observations[0]['arguments']['gene_ids'], row['metadata']['gene_ids'])
            for observation in observations:
                self.assertEqual(_pair_group(observation['arguments']['gene_ids'],
                                             row['metadata']['selection_seed'])[1], split)

    def test_validator_rejects_fabricated_evidence_coverage_and_case_relabeling(self):
        """Independent source replay and behavior binding reject plausible corruption."""
        mutations = []
        row = deepcopy(self.rows['train']['S2.1.8'])
        harness = json.loads(row['messages'][1]['content'])
        harness['observations'][0]['result']['neighbors'][0]['weight'] = 1000
        row['messages'][1]['content'] = json.dumps(harness)
        mutations.append(row)
        row = deepcopy(self.rows['train']['S2.1.38'])
        row['messages'][-1]['content']['result']['coverage']['unchecked_pairs'] = []
        mutations.append(row)
        row = deepcopy(self.rows['train']['S2.1.31'])
        row['metadata']['catalog_case_id'] = 'S2.1.32'
        mutations.append(row)
        row = deepcopy(self.rows['train']['S2.1.41'])
        calls(row)[0]['arguments']['gene_ids'].reverse()
        mutations.append(row)
        row = deepcopy(self.rows['train']['S2.1.1'])
        row['messages'][1]['role'] = 'tool'
        mutations.append(row)
        for row in mutations:
            with self.assertRaises(ValueError):
                validate_case_row(row, self.view, contract=self.contract)

