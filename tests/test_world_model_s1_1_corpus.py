"""Focused tests for the canonical S1.1 corpus builder."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_1_corpus import (
    GeneFact,
    form_counts,
    largest_remainder_counts,
    select_scenarios,
    split_row_counts,
)


class WorldModelS11AllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()
        self.curriculum = self.contract["curriculum"]
        self.settings = self.curriculum["families"]["S1.1"]

    def test_split_and_form_allocations_are_exact(self) -> None:
        rows = split_row_counts(
            self.settings["train_rows"],
            self.curriculum["split_fractions"],
        )
        self.assertEqual(
            rows, {"train": 10000, "validation": 1250, "test": 1250}
        )
        expected = {
            "train": {
                1: 1200,
                2: 1600,
                3: 1200,
                4: 1200,
                5: 800,
                6: 800,
                7: 800,
                8: 400,
                9: 1000,
                10: 700,
                11: 300,
            },
            "validation": {
                1: 150,
                2: 200,
                3: 150,
                4: 150,
                5: 100,
                6: 100,
                7: 100,
                8: 50,
                9: 125,
                10: 88,
                11: 37,
            },
            "test": {
                1: 150,
                2: 200,
                3: 150,
                4: 150,
                5: 100,
                6: 100,
                7: 100,
                8: 50,
                9: 125,
                10: 88,
                11: 37,
            },
        }
        actual = {
            split: form_counts(total, self.settings, self.curriculum)
            for split, total in rows.items()
        }
        self.assertEqual(actual, expected)
        for split, counts in actual.items():
            tool_rows = sum(
                counts[form] for form in (1, 2, 3, 5, 6, 8, 9, 10, 11)
            )
            self.assertEqual(counts[8] * 20, tool_rows, split)

    def test_largest_remainder_ties_follow_contract_order(self) -> None:
        counts = largest_remainder_counts(
            1250, self.settings["shares"]
        )
        self.assertEqual(counts["membership_absent"], 88)
        self.assertEqual(counts["unknown_multiplex"], 37)
        self.assertEqual(sum(counts.values()), 1250)

    def test_failures_retain_prompt_identical_successes(self) -> None:
        pools = {}
        counts = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            facts = [
                GeneFact(
                    fact_id=f"{split}_human_{index}",
                    split_units=(f"{split}_unit_{index}",),
                    gene_symbol=f"H{split_index}{index}",
                    gene_id=f"ENSG{split_index * 100 + index:011d}",
                )
                for index in range(12)
            ]
            historical = [
                GeneFact(
                    fact_id=f"{split}_old_{index}",
                    split_units=(f"{split}_old_unit_{index}",),
                    gene_symbol=f"OLD{split_index}{index}",
                    gene_id=None,
                )
                for index in range(4)
            ]
            mouse = [
                GeneFact(
                    fact_id=f"{split}_mouse_{index}",
                    split_units=(f"{split}_mouse_unit_{index}",),
                    gene_symbol=f"Mouse{split_index}{index}",
                    gene_id=f"ENSMUSG{split_index * 100 + index:011d}",
                    species="mouse",
                )
                for index in range(4)
            ]
            pools[split] = {
                "current": facts,
                "membership_present": facts[:6],
                "membership_absent": facts[6:],
                "historical": historical,
                "mouse": mouse,
            }
            counts[split] = {
                1: 4,
                2: 4,
                3: 4,
                4: 4,
                5: 4,
                6: 4,
                7: 4,
                8: 2,
                9: 4,
                10: 4,
                11: 4,
            }
        selected = select_scenarios(
            pools, counts, seed=1729, case_rate=0.1
        )
        for split, scenarios in selected.items():
            successes = {
                scenario.scenario_id: scenario
                for scenario in scenarios
                if scenario.form == 1
            }
            failures = [item for item in scenarios if item.form == 8]
            self.assertEqual(len(failures), 2)
            for failure in failures:
                success = successes[failure.paired_success_id]
                self.assertEqual(failure.input_value, success.input_value)
                self.assertEqual(failure.template_index, success.template_index)
                self.assertEqual(failure.framing, "direct")


class WorldModelS11GeneratedCorpusTests(unittest.TestCase):
    def test_generated_manifest_reconciles_when_present(self) -> None:
        root = Path("data/world_model_v3/corpora/s1/s1_1")
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            self.skipTest(f"generated corpus is absent: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["section"], "S1.1")
        self.assertEqual(
            manifest["row_counts"],
            {"train": 10000, "validation": 1250, "test": 1250},
        )
        for split in ("train", "validation", "test"):
            path = root / f"{split}.jsonl"
            with path.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(rows), manifest["row_counts"][split])
            self.assertEqual(
                {row["metadata"]["source_trace"] for row in rows},
                {f"S1.1.{form}" for form in range(1, 12)},
            )


if __name__ == "__main__":
    unittest.main()
