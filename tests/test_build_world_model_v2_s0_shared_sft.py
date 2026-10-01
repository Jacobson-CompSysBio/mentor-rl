"""Test the version 7 S0 shared corpus builder."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from runtime.world_model_s0_shared import build_s0_shared_trace
from runtime.world_model_s0_tools import SYMBOL_TOOL_NAME
from scripts.build_world_model_v2_s0_shared_sft import (
    DATASET_ID,
    answer_key_row,
    assign_not_found_facts,
    build_components,
    build_explanation_audit,
    load_previous_symbol_pool,
    make_not_found_symbol_facts,
    public_question_row,
    select_shared_train_facts,
)


class S0SharedSourceTests(unittest.TestCase):
    """Test the pinned previous-symbol source."""

    def test_previous_symbol_pool_excludes_current_symbols(self) -> None:
        """Keep only previous symbols that the registry does not contain."""

        text = "symbol\tprev_symbol\nNEW1\tOLD1|CURRENT\nNEW2\tOLD2\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hgnc.tsv"
            path.write_text(text, encoding="utf-8")
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

            symbols = load_previous_symbol_pool(
                path,
                expected_sha256=digest,
                current_symbols={"CURRENT"},
            )

        self.assertEqual(symbols, ["OLD1", "OLD2"])

    def test_absent_symbol_splits_are_disjoint(self) -> None:
        """Assign each absent symbol to one split only."""

        assignments = assign_not_found_facts(
            [f"OLD{index}" for index in range(12)],
            {
                "assignment_seed": 7,
                "train_rows": 4,
                "validation_rows": 3,
                "test_rows": 2,
            },
        )
        split_symbols = {
            name: {fact["input"]["gene_symbol"] for fact in facts}
            for name, facts in assignments.items()
        }

        self.assertFalse(split_symbols["train"] & split_symbols["validation"])
        self.assertFalse(split_symbols["train"] & split_symbols["test"])
        self.assertFalse(
            split_symbols["validation"] & split_symbols["test"]
        )
        self.assertTrue(
            all(
                fact["answer"]["status"] == "not_found"
                for facts in assignments.values()
                for fact in facts
            )
        )


class S0SharedSelectionTests(unittest.TestCase):
    """Test the named and unnamed row allocation."""

    def test_selection_keeps_each_configured_registry_pool(self) -> None:
        """Select symbols, named IDs, and unnamed IDs separately."""

        components = build_components(
            {
                "ENSG00000000001": ["SHARED"],
                "ENSG00000000002": ["SHARED"],
                "ENSG00000000003": ["NAMED3"],
                "ENSG00000000004": ["NAMED4"],
                "ENSG00000000005": [],
                "ENSG00000000006": [],
            }
        )
        selected = select_shared_train_facts(
            components,
            {},
            {
                "assignment_seed": 11,
                "registry_symbol_rows": 2,
                "named_gene_id_rows": 1,
                "unnamed_gene_id_rows": 1,
                "excluded_fact_ids": [],
            },
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            sum(fact["family"] == "human_ambiguous_symbol" for fact in selected),
            1,
        )
        selected_ids = [
            fact
            for fact in selected
            if fact["family"] == "human_ensembl_to_symbol"
        ]
        self.assertEqual(
            sum(bool(fact["answer"]["gene_symbols"]) for fact in selected_ids),
            1,
        )
        self.assertEqual(
            sum(not fact["answer"]["gene_symbols"] for fact in selected_ids),
            1,
        )


class S0SharedArtifactTests(unittest.TestCase):
    """Test answer-free questions, private keys, and the audit."""

    @staticmethod
    def _trace(record_id: str, split: str):
        """Return one small shared trace."""

        return build_s0_shared_trace(
            record_id=record_id,
            question="Resolve TP53 for the next analysis step.",
            question_family="human_symbol_to_ensembl",
            tool_name=SYMBOL_TOOL_NAME,
            arguments={"gene_symbol": "TP53"},
            tool_result={
                "status": "resolved",
                "gene_id": "ENSG00000141510",
                "gene_symbol": "TP53",
            },
            split=split,
            provenance={
                "fact_id": "fact_tp53",
                "fact_group_id": "group_tp53",
                "fact_role": "train" if split == "train" else "unseen",
                "prompt_form_id": (
                    "train" if split == "train" else "validation"
                ),
                "rendering_index": 0,
            },
            identifier_registry_id="sha256:" + "0" * 64,
        )

    def test_public_question_has_no_target_decision(self) -> None:
        """Remove the user-input label and all supervised messages."""

        trace = self._trace("trace_public", "val")
        fact = make_not_found_symbol_facts(["OLD1"])[0]
        fact["input"] = {"gene_symbol": "TP53"}
        question = public_question_row(trace, fact, split="test")

        self.assertNotIn("requires_user_input", question["metadata"])
        self.assertNotIn("messages", question)
        self.assertNotIn("answer", question)
        self.assertEqual(question["split"], "test")

    def test_answer_key_keeps_both_explanations(self) -> None:
        """Keep exact call and final targets in the private key."""

        trace = self._trace("trace_key", "train")
        fact = {
            "fact_id": "fact_tp53",
            "fact_group_id": "group_tp53",
            "family": "human_symbol_to_ensembl",
        }
        answer = answer_key_row(trace, fact)

        self.assertEqual(answer["expected_status"], "resolved")
        self.assertEqual(answer["expected_continuation_decision"], "continue")
        self.assertTrue(answer["expected_assistant_explanation"])
        self.assertTrue(answer["expected_final_explanation"])
        self.assertTrue(answer["expected_final_answer"].startswith("{"))

    def test_audit_has_all_templates_and_source_flags(self) -> None:
        """Write the required source flags and deterministic samples."""

        train = self._trace("trace_train", "train")
        validation = self._trace("trace_validation", "val")
        audit = build_explanation_audit(
            [train],
            [validation],
            config={"sample_size_per_prompt_form": 1, "sample_seed": 17},
        )

        self.assertEqual(audit["dataset_id"], DATASET_ID)
        self.assertFalse(audit["model_text_created"])
        self.assertFalse(audit["model_sample_used"])
        self.assertEqual(audit["template_count"], 24)
        self.assertEqual(len(audit["samples"]["train"]), 1)
        self.assertEqual(len(audit["samples"]["validation"]), 1)


if __name__ == "__main__":
    unittest.main()
