"""Test the S0 full and debug exposure contracts."""

from __future__ import annotations

import unittest

from runtime.world_model_training import (
    build_training_exposure_manifest,
    consumed_training_index_plan,
    s0_exposure_scope,
)


SHA256 = "0" * 64


def exposure_args(*, total_steps: int, run_scope: str) -> dict:
    """Return one small valid exposure plan."""

    row_count = 4
    plan = consumed_training_index_plan(
        row_count,
        total_steps=total_steps,
        num_train_epochs=1,
        replica_count=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        seed=7,
        preserve_order=False,
    )
    return {
        "run_id": "test-run",
        "method_id": "test-method",
        "run_config_sha256": SHA256,
        "corpus_manifest_sha256": SHA256,
        "train_sha256": SHA256,
        "tokenizer_arm_manifest_sha256": SHA256,
        "tokenizer_manifest_sha256": SHA256,
        "record_ids": [f"record-{index}" for index in range(row_count)],
        "fact_ids": [f"fact-{index}" for index in range(row_count)],
        "question_families": [
            "human_symbol_to_ensembl",
            "human_ensembl_to_symbol",
            "human_ambiguous_symbol",
            "human_symbol_to_ensembl",
        ],
        "prompt_form_ids": ["train"] * row_count,
        "consumed_indices": list(plan["logical_indices"]),
        "distributed_padding_indices": list(
            plan["distributed_padding_indices"]
        ),
        "seed": 7,
        "total_steps": total_steps,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "data_parallel_size": 1,
        "preserve_order": False,
        "padding_policy": plan["padding_policy"],
        "exposure_scope": s0_exposure_scope(run_scope),
        "status": "planned",
    }


class TrainingExposureTest(unittest.TestCase):
    """Check the full and bounded exposure rules."""

    def test_debug_plan_records_partial_exposure(self) -> None:
        """A debug plan accepts and reports a bounded row subset."""

        manifest = build_training_exposure_manifest(
            **exposure_args(
                total_steps=1,
                run_scope="debug_qualification",
            )
        )
        self.assertEqual(
            manifest["schema_version"],
            "mentor-rl-s0-training-exposure-v2",
        )
        self.assertEqual(
            manifest["exposure_contract"]["scope"],
            "bounded_debug_subset",
        )
        self.assertFalse(
            manifest["logical_exposure"][
                "all_eligible_train_rows_exposed"
            ]
        )

    def test_full_plan_requires_all_eligible_rows(self) -> None:
        """A full plan rejects a partial eligible row set."""

        with self.assertRaisesRegex(
            ValueError,
            "must expose every eligible train row",
        ):
            build_training_exposure_manifest(
                **exposure_args(
                    total_steps=1,
                    run_scope="qualification",
                )
            )

    def test_full_plan_reports_complete_exposure(self) -> None:
        """A full plan reports all eligible rows after one epoch."""

        manifest = build_training_exposure_manifest(
            **exposure_args(
                total_steps=4,
                run_scope="matched_matrix",
            )
        )
        self.assertTrue(
            manifest["logical_exposure"][
                "all_eligible_train_rows_exposed"
            ]
        )
        self.assertTrue(
            manifest["exposure_contract"][
                "all_eligible_train_rows_required"
            ]
        )


if __name__ == "__main__":
    unittest.main()
