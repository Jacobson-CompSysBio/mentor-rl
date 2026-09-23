"""Test unrestricted S0 train exposure receipts."""

from __future__ import annotations

import unittest

from runtime.world_model_training import (
    build_training_exposure_manifest,
    consumed_training_index_plan,
    derive_optimizer_step_schedule,
    s0_exposure_corpus_identity,
    s0_exposure_scope,
)


SHA256 = "ignored-hash-value"


class OptimizerStepScheduleTest(unittest.TestCase):
    """Check the step calculation for complete epochs."""

    def test_single_replica_schedule(self) -> None:
        """One-row batches give one update for each row."""

        schedule = derive_optimizer_step_schedule(
            100,
            num_train_epochs=500,
            replica_count=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
        )

        self.assertEqual(schedule["updates_per_epoch"], 100)
        self.assertEqual(schedule["total_steps"], 50_000)

    def test_distributed_accumulation_schedule(self) -> None:
        """Replicas and accumulation reduce the update count."""

        schedule = derive_optimizer_step_schedule(
            100,
            num_train_epochs=500,
            replica_count=16,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
        )

        self.assertEqual(schedule["global_batch_size"], 32)
        self.assertEqual(schedule["updates_per_epoch"], 4)
        self.assertEqual(schedule["total_steps"], 2_000)

    def test_v6_complete_epoch_schedules(self) -> None:
        """Derived steps match the v6 train schedules."""

        cases = (
            (1, 1, 2, 2_048),
            (5, 16, 1, 1_280),
            (10, 16, 2, 1_280),
        )
        for epochs, replicas, accumulation, expected_steps in cases:
            with self.subTest(epochs=epochs, accumulation=accumulation):
                schedule = derive_optimizer_step_schedule(
                    4_096,
                    num_train_epochs=epochs,
                    replica_count=replicas,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=accumulation,
                )
                self.assertEqual(schedule["total_steps"], expected_steps)


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

    def test_qualification_records_partial_exposure(self) -> None:
        """A qualification plan accepts and reports a row subset."""

        manifest = build_training_exposure_manifest(
            **exposure_args(
                total_steps=1,
                run_scope="qualification",
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

    def test_qualification_reports_complete_exposure(self) -> None:
        """A qualification can report all rows without a requirement."""

        manifest = build_training_exposure_manifest(
            **exposure_args(
                total_steps=4,
                run_scope="qualification",
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

    def test_production_scope_requires_complete_exposure(self) -> None:
        """A production plan rejects a partial eligible row set."""

        with self.assertRaisesRegex(
            ValueError,
            "must expose every eligible train row",
        ):
            build_training_exposure_manifest(
                **exposure_args(
                    total_steps=1,
                    run_scope="production",
                )
            )

    def test_production_scope_reports_complete_exposure(self) -> None:
        """A production plan requires and reports all eligible rows."""

        manifest = build_training_exposure_manifest(
            **exposure_args(
                total_steps=4,
                run_scope="production",
            )
        )

        self.assertEqual(
            manifest["exposure_contract"]["scope"],
            "all_eligible_train_rows",
        )
        self.assertTrue(
            manifest["exposure_contract"][
                "all_eligible_train_rows_required"
            ]
        )
        self.assertTrue(
            manifest["logical_exposure"][
                "all_eligible_train_rows_exposed"
            ]
        )

    def test_trajectory_receipt_uses_the_v6_corpus_identity(self) -> None:
        """A trajectory receipt records its v6 data and prompt identities."""

        arguments = exposure_args(
            total_steps=4,
            run_scope="production",
        )
        arguments.update(
            s0_exposure_corpus_identity("s0_tool_trajectory_v1")
        )
        manifest = build_training_exposure_manifest(**arguments)

        self.assertEqual(
            manifest["corpus"]["dataset_id"],
            "world_model_v2_s0_human_identifier_trajectories_v6",
        )
        self.assertEqual(
            manifest["corpus"]["system_prompt_sha256"],
            "064a8fc8f103d9dc021ebbe124614342f4a52fd8123a4e4ba1b8bf88588e6dbf",
        )


if __name__ == "__main__":
    unittest.main()
