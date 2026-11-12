"""Prevent missing S2 behavior and unfilled quotas from passing as 8k exclusions."""

from collections import Counter
from copy import deepcopy
import json
import unittest

from runtime.world_model.s2_production_cases import (
    CatalogCase, coverage_report, load_catalog, parse_catalog, semantic_signature,
)


def row(row_id='first', split='train', variant='lookup', framing='direct'):
    """Use the production metadata and final-envelope fields of a small edge row."""
    return {'row_id': row_id, 'metadata': {'question_family': 'S2.1.single_edge',
        'variant': variant, 'bucket': 'recorded', 'split': split, 'framing': framing,
        'gene_ids': ['ENSG00000101856', 'ENSG00000180383'], 'wrapper_index': 0},
        'messages': [{'role': 'assistant', 'thinking': 'Free reasoning.', 'content': {
            'status': 'resolved', 'provenance': {'registry': [], 'graph': []},
            'result': {'edge_exists': True, 'directed': False, 'weight': 0.6},
            'continuation_decision': 'continue' if framing == 'step' else 'stop',
            'blocked_reason': None, 'summary': 'Free summary.'}}]}


def case(case_id='S2.1.1'):
    """Supply one approved case without a guessed generator binding."""
    return CatalogCase(case_id, 'S2.1', 'A reviewed case', 1, 'resolved', 'stop', 'direct')


class ProductionCaseTests(unittest.TestCase):
    """Check the real design inventory and fail-closed production coverage."""

    def test_current_catalog_keeps_all_399_numbered_cases_and_titles(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog), 399)
        self.assertEqual(Counter(item.family for item in catalog),
            {f'S2.{number}': count for number, count in enumerate(
                (41, 22, 25, 25, 25, 25, 29, 31, 30, 34, 36, 36, 40), 1)})
        self.assertEqual(catalog[0].title, 'Edge present (direct)')
        self.assertEqual(catalog[-1].case_id, 'S2.13.40')
        self.assertEqual(catalog[-1].framing_hint, 'step')
        self.assertTrue(all(item.framing_hint is None for item in catalog
                            if item.continuation_decision == 'defer'))

    def test_catalog_rejects_missing_answers_and_duplicate_ids(self):
        valid = ('### S2.1.1: A\nFinal answer:\n```json\n'
                 '{"status":"resolved","continuation_decision":"stop"}\n```\n')
        self.assertEqual(parse_catalog(valid)[0].framing_hint, 'direct')
        for text in ('### S2.1.1: Missing\n', valid + valid):
            with self.assertRaises(ValueError):
                parse_catalog(text)

    def test_signature_ignores_fact_identity_weights_wording_and_split(self):
        original = row()
        changed = deepcopy(original)
        changed['row_id'] = 'different'
        changed['metadata'].update(gene_ids=['other', 'identities'], split='test', wrapper_index=19)
        changed['messages'][-1]['thinking'] = 'Different wording.'
        changed['messages'][-1]['content']['result']['weight'] = 0.9
        changed['messages'][-1]['content']['summary'] = 'Another summary.'
        self.assertEqual(semantic_signature(original), semantic_signature(changed))

    def test_signature_preserves_context_framing_and_native_direction(self):
        baseline = semantic_signature(row())
        self.assertNotEqual(baseline, semantic_signature(row(variant='context')))
        self.assertNotEqual(baseline, semantic_signature(row(framing='step')))
        directed = row()
        directed['messages'][-1]['content']['result']['directed'] = True
        self.assertNotEqual(baseline, semantic_signature(directed))

    def test_8192_boundary_is_eligible_and_full_three_split_quota_is_required(self):
        rows = [row(split, split) for split in ('train', 'validation', 'test')]
        signature = semantic_signature(rows[0])
        report = coverage_report([case()], {'S2.1.1': [signature]},
            [(item, 8192) for item in rows], quotas=dict.fromkeys(('train', 'validation', 'test'), 1))
        self.assertTrue(report['case_coverage_ready'])
        self.assertFalse(report['training_ready'])
        entry = report['case_entries'][0]
        self.assertFalse(entry['excluded_for_length'])
        self.assertEqual(entry['maximum_observed_tokens'], 8192)
        missing_test = coverage_report([case()], {'S2.1.1': [signature]},
            [(item, 8192) for item in rows[:2]], quotas=dict.fromkeys(('train', 'validation', 'test'), 1))
        self.assertFalse(missing_test['case_coverage_ready'])
        self.assertEqual(missing_test['case_entries'][0]['quota_shortfalls']['test'], 1)

    def test_unmapped_and_implementation_gaps_are_never_length_exclusions(self):
        report = coverage_report([case('S2.1.31')], {}, [(row(), 9000)])
        entry = report['case_entries'][0]
        self.assertEqual(entry['coverage_status'], 'unmapped')
        self.assertFalse(entry['excluded_for_length'])
        self.assertIn('implementation_gap', entry)
        self.assertFalse(report['case_coverage_ready'])
        self.assertEqual(report['blocking_case_ids'], ['S2.1.31'])

    def test_absent_mapped_candidates_do_not_count_as_length_exclusions(self):
        report = coverage_report([case()], {'S2.1.1': [semantic_signature(row())]}, [])
        self.assertEqual(report['case_entries'][0]['coverage_status'], 'unavailable')
        self.assertFalse(report['case_entries'][0]['excluded_for_length'])
        self.assertFalse(report['case_coverage_ready'])

    def test_overlength_requires_all_observed_candidates_to_exceed_limit(self):
        signature = semantic_signature(row())
        all_long = coverage_report([case()], {'S2.1.1': [signature]}, [(row(), 8193)])
        self.assertEqual(all_long['case_entries'][0]['coverage_status'], 'overlength')
        self.assertFalse(all_long['case_entries'][0]['excluded_for_length'])
        self.assertEqual(all_long['blocking_case_ids'], ['S2.1.1'])
        self.assertFalse(all_long['case_coverage_ready'])  # An entirely empty curriculum cannot pass.
        explicitly_excluded = coverage_report([case()], {'S2.1.1': [signature]}, [(row(), 8193)],
                                             length_excluded_case_ids=['S2.1.1'])
        self.assertTrue(explicitly_excluded['case_entries'][0]['excluded_for_length'])
        self.assertFalse(explicitly_excluded['case_coverage_ready'])
        partly_short = coverage_report([case()], {'S2.1.1': [signature]},
            [(row(), 8193), (row('shorter'), 8192)])
        self.assertEqual(partly_short['case_entries'][0]['coverage_status'], 'supported')
        self.assertFalse(partly_short['case_entries'][0]['excluded_for_length'])
        self.assertFalse(partly_short['case_coverage_ready'])  # Only one eligible example, below the quota.

    def test_observed_length_gap_blocks_readiness_until_explicitly_excluded(self):
        fitting = [row(split, split) for split in ('train', 'validation', 'test')]
        overlength = row('long', variant='context')
        bindings = {'S2.1.1': [semantic_signature(fitting[0])],
                    'S2.1.2': [semantic_signature(overlength)]}
        observations = [(item, 8192) for item in fitting] + [(overlength, 8193)]
        quotas = dict.fromkeys(('train', 'validation', 'test'), 1)
        blocked = coverage_report([case(), case('S2.1.2')], bindings, observations, quotas=quotas)
        self.assertFalse(blocked['case_coverage_ready'])
        self.assertEqual(blocked['blocking_case_ids'], ['S2.1.2'])
        checked = coverage_report([case(), case('S2.1.2')], bindings, observations, quotas=quotas,
                                  length_excluded_case_ids={'S2.1.2'})
        self.assertTrue(checked['case_coverage_ready'])
        self.assertEqual(checked['length_excluded_case_ids'], ['S2.1.2'])

    def test_length_exclusion_rejects_unknown_unmapped_unavailable_and_fitting_cases(self):
        signature = semantic_signature(row())
        attempts = [({}, [], ['S2.1.2']), ({}, [], ['S2.1.1']),
                    ({'S2.1.1': [signature]}, [], ['S2.1.1']),
                    ({'S2.1.1': [signature]}, [(row(), 8192)], ['S2.1.1'])]
        for mappings, observations, exclusions in attempts:
            with self.subTest(mappings=mappings, observations=observations):
                with self.assertRaises(ValueError):
                    coverage_report([case()], mappings, observations,
                                    length_excluded_case_ids=exclusions)

    def test_shared_signatures_cannot_double_count_rows_toward_two_case_quotas(self):
        rows = [row(split, split) for split in ('train', 'validation', 'test')]
        signature = semantic_signature(rows[0])
        report = coverage_report([case(), case('S2.1.2')],
            {'S2.1.1': [signature], 'S2.1.2': [signature]},
            [(item, 4000) for item in rows], quotas=dict.fromkeys(('train', 'validation', 'test'), 1))
        self.assertTrue(report['ambiguous_generator_mappings'])
        self.assertFalse(report['case_coverage_ready'])

    def test_duplicate_row_ids_unknown_cases_and_invalid_tokens_fail(self):
        signature = semantic_signature(row())
        with self.assertRaises(ValueError):
            coverage_report([case()], {'S2.1.1': [signature]}, [(row(), 100), (row(), 100)])
        with self.assertRaises(ValueError):
            coverage_report([case()], {'S2.1.2': [signature]}, [])
        with self.assertRaises(ValueError):
            coverage_report([case()], {}, [(row(), True)])

    def test_bindings_cannot_change_reviewed_family_or_completion_outcome(self):
        workflow = row(framing='step')
        foreign = row()
        foreign['metadata']['question_family'] = 'S2.2'
        for candidate in (workflow, foreign):
            with self.assertRaises(ValueError):
                coverage_report([case()], {'S2.1.1': [semantic_signature(candidate)]}, [])

    def test_bindings_keep_known_framing_and_canonical_signature_json(self):
        wrong_framing = row()
        wrong_framing['metadata']['framing'] = 'step'
        with self.assertRaises(ValueError):
            coverage_report([case()], {'S2.1.1': [semantic_signature(wrong_framing)]}, [])
        signature = json.dumps(json.loads(semantic_signature(row())), indent=2)
        with self.assertRaises(ValueError):
            coverage_report([case()], {'S2.1.1': [signature]}, [])


if __name__ == '__main__':
    unittest.main()
