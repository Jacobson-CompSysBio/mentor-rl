"""Verify immutable partial adapters and paired checkpoint diagnostic accounting."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from runtime.world_model_checkpoint_diagnostics import capture_validation_snapshot
from runtime.world_model_training import consumed_training_index_plan
from scripts.diagnose_world_model_checkpoints import (
    FIELDS, SEGMENTS, checked_record_order, correlations, paired_transitions, prepare, summarized_loss,
    s0_structured_accuracy, validation_scoring_contract,
)
from scripts.run_world_model_v2_s0_exact_generation import S0GenerationError, validate_checkpoint
from scripts.world_model_s1_eval_common import read_json


def exposure_args() -> dict:
    """Return a production plan whose middle checkpoint has partial exposure."""
    plan = consumed_training_index_plan(4, total_steps=4, num_train_epochs=1,
        replica_count=1, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        seed=7, preserve_order=False)
    return {
        "run_id": "run", "method_id": "method",
        **{name: "a" * 64 for name in ("run_config_sha256", "corpus_manifest_sha256", "train_sha256", "tokenizer_arm_manifest_sha256", "tokenizer_manifest_sha256")},
        "record_ids": [f"row-{index}" for index in range(4)],
        "fact_ids": [f"fact-{index}" for index in range(4)],
        "question_families": ["human_symbol_to_ensembl"] * 4,
        "prompt_form_ids": ["train"] * 4,
        "consumed_indices": list(plan["logical_indices"]),
        "distributed_padding_indices": [], "seed": 7, "total_steps": 4,
        "num_train_epochs": 1, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1, "data_parallel_size": 1,
        "preserve_order": False, "padding_policy": plan["padding_policy"],
        "exposure_scope": "all_eligible_train_rows",
    }


def write_checkpoint(path: Path) -> None:
    """Write a small completed inference input without allocating model weights."""
    path.mkdir()
    manifest = {
        "format": "mentor-rl-s0-tp-lora-v1", "rank_shards": 1,
        "identity": {"run_id": "run", "method_id": "method", **{name: "a" * 64 for name in (
            "run_config_sha256", "corpus_manifest_sha256", "train_sha256", "tokenizer_arm_manifest_sha256", "tokenizer_manifest_sha256", "model_identity_sha256")}},
    }
    (path / "tp_adapter_manifest.json").write_text(json.dumps(manifest))
    (path / "training_state.json").write_text(json.dumps({"global_step": 2}))
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"fake-adapter-bytes")
    (path / "optimizer-rank-00000.pt").write_bytes(b"not-for-inference")
    (path / "CHECKPOINT_COMPLETE").write_text("global_step=2\n")


def loss_row(record_id: str, nll_sum: float, count: int) -> dict:
    """Make additive statistics with deliberately unequal token counts."""
    values = {"nll_sum": nll_sum, "token_count": count, "correct_tokens": count}
    return {"record_id": record_id, "segments": {name: dict(values) for name in SEGMENTS},
            "fields": {name: dict(values) for name in FIELDS.values()}}


class CheckpointDiagnosticTests(unittest.TestCase):
    """Check data exposure, snapshot guards, and diagnostic-only measurements."""

    def test_atomic_snapshot_records_only_consumed_prefix(self):
        """Partial training cannot claim full exposure or pass the final validator."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-2"
            write_checkpoint(checkpoint)
            snapshot = capture_validation_snapshot(checkpoint, root, 2, exposure_args())
            self.assertEqual({p.name for p in snapshot.iterdir()}, {
                "adapter_config.json", "adapter_model.safetensors", "tp_adapter_manifest.json",
                "training_exposure.json", "snapshot_metadata.json", "SNAPSHOT_COMPLETE"})
            exposure = read_json(snapshot / "training_exposure.json")
            self.assertEqual(exposure["status"], "snapshot")
            self.assertEqual(exposure["logical_exposure"]["unique_record_count"], 2)
            self.assertFalse(exposure["logical_exposure"]["all_eligible_train_rows_exposed"])
            self.assertFalse(exposure["exposure_contract"]["satisfied"])
            self.assertTrue(all(p.stat().st_mode & 0o222 == 0 for p in snapshot.iterdir()))
            kwargs = dict(method_id="method", corpus_manifest_sha256="a" * 64,
                          tokenizer_manifest_sha256="a" * 64, model_identity_sha256="a" * 64)
            artifact, run_id = validate_checkpoint(snapshot, snapshot_validation=True, **kwargs)
            self.assertEqual(artifact["snapshot_step"], 2)
            self.assertEqual(run_id, "run")
            with self.assertRaisesRegex(S0GenerationError, "checkpoint contract"):
                validate_checkpoint(snapshot, **kwargs)
            with self.assertRaises(FileExistsError):
                capture_validation_snapshot(checkpoint, root, 2, exposure_args())

    def test_snapshot_rejects_incomplete_or_different_step(self):
        """A live writer or mismatched step must never publish an adapter."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-2"
            write_checkpoint(checkpoint)
            with self.assertRaisesRegex(ValueError, "steps disagree"):
                capture_validation_snapshot(checkpoint, root, 1, exposure_args())
            (checkpoint / "CHECKPOINT_COMPLETE").unlink()
            with self.assertRaisesRegex(ValueError, "complete checkpoint"):
                capture_validation_snapshot(checkpoint, root, 2, exposure_args())

    def test_snapshot_detects_adapter_corruption(self):
        """The reusable snapshot validator verifies actual copied bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-2"
            write_checkpoint(checkpoint)
            snapshot = capture_validation_snapshot(checkpoint, root, 2, exposure_args())
            adapter = snapshot / "adapter_model.safetensors"
            adapter.chmod(0o644)
            adapter.write_bytes(b"different")
            with self.assertRaisesRegex(S0GenerationError, "completed snapshot file changed"):
                validate_checkpoint(snapshot, method_id="method", corpus_manifest_sha256="a" * 64,
                    tokenizer_manifest_sha256="a" * 64, model_identity_sha256="a" * 64, snapshot_validation=True)

    def test_weighted_loss_and_empty_categories(self):
        """Unequal rows aggregate by tokens, and absent categories have null means."""
        result = summarized_loss([loss_row("one", 1., 1), loss_row("nine", 27., 9)])
        self.assertEqual(result["segments"]["reasoning"]["nll"], 2.8)
        self.assertEqual(result["segments"]["reasoning"]["token_count"], 10)
        self.assertIsNone(summarized_loss([])["fields"]["status"]["nll"])
        with self.assertRaisesRegex(ValueError, "Invalid additive"):
            summarized_loss([loss_row("broken", float("nan"), 1)])

    def test_correlation_uses_checkpoint_means_and_reports_sample_count(self):
        """Small within-run trends are explicit and constant metrics stay undefined."""
        checkpoints = []
        for index in range(3):
            measurement = {"loss": summarized_loss([loss_row("one", 3. - index, 1)]),
                           "accuracy": {name: .5 + .2 * index for name in ("structured", "result", "continuation", "tool", "json")},
                           "loss_by": {"family": {}, "source_trace": {}}}
            checkpoints.append({"panels": {"s1": measurement, "s0": deepcopy(measurement)}})
        rows = correlations(checkpoints)
        self.assertEqual(rows[0]["sample_count"], 3)
        self.assertAlmostEqual(rows[0]["pearson"], -1.)
        for checkpoint in checkpoints:
            checkpoint["panels"]["s1"]["accuracy"]["structured"] = 1.
        self.assertIsNone(correlations(checkpoints)[0]["pearson"])

    def test_paired_transitions_join_record_identity(self):
        """Shuffled row order cannot change the fix/regression counts."""
        rows = []
        for step, states in ((400, (("b", True), ("a", False))), (800, (("a", True), ("b", False)))):
            for rid, success in states:
                rows.append({"record_id": rid, "checkpoint": {"step": step},
                    "generated_accuracy": {name: success for name in ("structured", "result", "continuation", "tool", "json")}})
        transitions = paired_transitions(rows)
        self.assertEqual(transitions[0]["counts"], {"0->1": 1, "1->0": 1})
        with self.assertRaisesRegex(ValueError, "identical validation records"):
            paired_transitions(rows[:-1])

    def test_prediction_order_is_checked_before_native_scoring(self):
        """Scorers cannot silently attach the next question's ID to a wrong prediction."""
        checked_record_order([{"record_id": "a"}, {"record_id": "b"}], ["a", "b"])
        for rows in ([{"record_id": "b"}, {"record_id": "a"}], [{"record_id": "a"}], [{"record_id": "a"}, {"record_id": "a"}]):
            with self.assertRaisesRegex(ValueError, "coverage or order"):
                checked_record_order(rows, ["a", "b"])

    def test_validation_name_classification_keeps_original_contract(self):
        """Only diagnostic classification uses validation names; gates stay frozen."""
        contract = {"curriculum": {"families": {"S1.8": {
            "name_variants": {"test": {"m": ["t"]}, "validation": {"m": ["v"]}},
            "shared_aliases": {"test": ["t"], "validation": ["v"]}}}}}
        original = deepcopy(contract)
        diagnostic = validation_scoring_contract(contract)
        self.assertEqual(contract, original)
        self.assertEqual(diagnostic["curriculum"]["families"]["S1.8"]["shared_aliases"]["test"], ["v"])

    def test_s0_structured_score_preserves_provenance_and_ignores_prose_wording(self):
        """Correct results alone cannot hide false evidence or absent blocked reasons."""
        provenance = {"registry": {"registry": "ensembl", "release": "116", "evidence_id": "ev_reg_000001"}, "graph": None}
        expected = {"status": "ambiguous", "provenance": provenance, "result": {"gene_id": None},
                    "continuation_decision": "defer", "blocked_reason": "Ask which gene.", "summary": "Several possibilities."}
        generated = {**expected, "blocked_reason": "Please identify the intended candidate.", "summary": "This name is ambiguous."}
        row = {"structured_json_valid": True, "valid_final_answer": True, "payload_exact": True,
               "assistant_final": json.dumps(generated), "expected_final_answer": json.dumps(expected),
               "generated_tool_payload": {"provenance": provenance}}
        self.assertTrue(s0_structured_accuracy(row))
        generated["provenance"] = {"registry": {**provenance["registry"], "release": "115"}, "graph": None}
        row["assistant_final"] = json.dumps(generated)
        self.assertFalse(s0_structured_accuracy(row))
        generated["provenance"] = provenance
        generated["blocked_reason"] = None
        row["assistant_final"] = json.dumps(generated)
        self.assertFalse(s0_structured_accuracy(row))

    def test_preparation_requires_real_completed_checkpoint(self):
        """The post-training diagnostic cannot accept a future-checkpoint stub."""
        root = Path(__file__).resolve().parents[1]
        config = root / "config/world_model_s1_120b_checkpoint_diagnostics_v9_32n.json"
        future = root / "data/world_model_v4/checkpoints/s1_v9_120b_32n/oss120b-plain-base-tokenizer-lora-r32/slurm-999999999"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(S0GenerationError):
                prepare(config, future, Path(temporary) / "output")


if __name__ == "__main__":
    unittest.main()
