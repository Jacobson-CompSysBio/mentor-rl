"""Test S0 train exposure receipts."""

from __future__ import annotations

import unittest

from runtime.world_model.contract import load_contract, system_prompt_sha256
from runtime.world_model_training import (
    S0_V4_DATASET_ID,
    S0_V4_PRIORITY_LOSS_CONTRACT,
    build_training_exposure_manifest,
    consumed_training_index_plan,
    derive_optimizer_step_schedule,
    question_families_for_dataset,
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

    def test_v4_sweep_schedule(self) -> None:
        """Derived steps match the v4 sweep: 512 updates per epoch."""

        schedule = derive_optimizer_step_schedule(
            16_384,
            num_train_epochs=3,
            replica_count=16,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
        )

        self.assertEqual(schedule["global_batch_size"], 32)
        self.assertEqual(schedule["updates_per_epoch"], 512)
        self.assertEqual(schedule["total_steps"], 1_536)


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
        **s0_exposure_corpus_identity("multiplex_sft_v3", S0_V4_DATASET_ID),
    }


class TrainingExposureTest(unittest.TestCase):
    """Check that a receipt records the full exposure of the train rows."""

    def test_only_the_production_run_scope_is_supported(self) -> None:
        self.assertEqual(
            s0_exposure_scope("production"), "all_eligible_train_rows"
        )
        with self.assertRaisesRegex(ValueError, "run scope is invalid"):
            s0_exposure_scope("qualification")

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

    def test_receipt_records_the_v4_corpus_identity(self) -> None:
        """A receipt records the v4 dataset and the contract system prompt."""

        manifest = build_training_exposure_manifest(
            **exposure_args(total_steps=4, run_scope="production")
        )

        self.assertEqual(manifest["corpus"]["dataset_id"], S0_V4_DATASET_ID)
        self.assertEqual(
            manifest["corpus"]["system_prompt_sha256"],
            system_prompt_sha256(load_contract()),
        )

    def test_receipt_rejects_another_system_prompt(self) -> None:
        arguments = exposure_args(total_steps=4, run_scope="production")
        arguments["system_prompt_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "system prompt identity"):
            build_training_exposure_manifest(**arguments)

    def test_retired_datasets_are_rejected(self) -> None:
        for dataset_id in (
            "world_model_v2_s0_human_identifier_trajectories_v6",
            "world_model_v3_s0_human_identifiers",
        ):
            with self.subTest(dataset_id=dataset_id):
                with self.assertRaisesRegex(ValueError, "not supported"):
                    question_families_for_dataset(dataset_id)
                with self.assertRaisesRegex(ValueError, "not supported"):
                    s0_exposure_corpus_identity("multiplex_sft_v3", dataset_id)

    def test_unknown_loss_contract_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss contracts"):
            s0_exposure_corpus_identity(
                "s0_tool_trajectory_v1", S0_V4_DATASET_ID
            )

    def test_v4_dataset_accepts_uniform_and_priority_loss_receipts(self) -> None:
        identities = [
            s0_exposure_corpus_identity(
                loss_contract,
                S0_V4_DATASET_ID,
            )
            for loss_contract in (
                "multiplex_sft_v3",
                S0_V4_PRIORITY_LOSS_CONTRACT,
            )
        ]

        self.assertEqual(identities[0], identities[1])
        self.assertEqual(
            identities[0]["dataset_id"], S0_V4_DATASET_ID
        )


if __name__ == "__main__":
    unittest.main()
