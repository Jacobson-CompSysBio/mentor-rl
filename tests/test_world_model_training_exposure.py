"""Test unrestricted S0 train exposure receipts."""

from __future__ import annotations

import unittest

from runtime.world_model_training import (
    build_training_exposure_manifest,
    consumed_training_index_plan,
    s0_exposure_scope,
)


SHA256 = "ignored-hash-value"


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
    """Check that exposure receipts do not restrict data length."""

    def test_debug_plan_records_partial_exposure(self) -> None:
        """A debug plan accepts and reports a row subset."""

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
            "unrestricted",
        )
        self.assertFalse(
            manifest["logical_exposure"][
                "all_eligible_train_rows_exposed"
            ]
        )

    def test_full_scope_accepts_partial_exposure(self) -> None:
        """A qualification plan accepts a partial row set."""

        arguments = exposure_args(
            total_steps=1,
            run_scope="qualification",
        )
        arguments["consumed_indices"] = [0]
        manifest = build_training_exposure_manifest(**arguments)
        self.assertFalse(
            manifest["logical_exposure"][
                "all_eligible_train_rows_exposed"
            ]
        )
        self.assertFalse(
            manifest["exposure_contract"][
                "all_eligible_train_rows_required"
            ]
        )

    def test_scope_reports_complete_exposure(self) -> None:
        """A plan can report all rows without a requirement."""

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
        self.assertFalse(
            manifest["exposure_contract"][
                "all_eligible_train_rows_required"
            ]
        )


if __name__ == "__main__":
    unittest.main()
