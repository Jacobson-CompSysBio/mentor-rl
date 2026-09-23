"""Test the v6 full trajectory evaluation contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts/validate_world_model_v2_s0_eval_contract.py"
RUN_CONFIG = (
    REPO_ROOT
    / "config/"
    "world_model_v2_s0_120b_full_registry_tool_trajectory_"
    "test_v6_64n_bs4_sans_3_long_rows.json"
)
METHOD_ID = "oss120b-plain-base-tokenizer-lora-r32"


def _stable_sha256(value: object) -> str:
    """Return one stable JSON identity."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_checkpoint(path: Path, *, loss_contract: str) -> None:
    """Write one mock v6 120B checkpoint contract."""

    contract_dir = path / "run_contract"
    contract_dir.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"test")
    adapter = {
        "format": "mentor-rl-s0-tp-lora-v1",
        "objective": {"loss_contract": loss_contract},
        "identity": {
            "run_id": "oss120b-v6-train-run",
            "method_id": METHOD_ID,
            "model_identity_sha256": (
                "905e86330e48859268f5e51fd569acf619e6030e1508b37e6a82e61dcf0c7889"
            ),
            "corpus_manifest_sha256": (
                "27343d0e58d09f42b039c15debaddb23a08eabbb9cd6ab93700c5dff1c3d21fe"
            ),
            "train_sha256": (
                "5d1b96e3d6f0b093d76d55ca960af7f0686cdbeed20b7d80bfba7718bb58922d"
            ),
            "tokenizer_manifest_sha256": (
                "245163b5189a64afb8e5b061585f90d4c15bcf94599b68307e9ff008880fe3c0"
            ),
        },
    }
    (path / "tp_adapter_manifest.json").write_text(
        json.dumps(adapter),
        encoding="utf-8",
    )
    exposure = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": "complete",
        "method_id": METHOD_ID,
        "logical_exposure": {},
        "exposure_contract": {"satisfied": True},
    }
    exposure["manifest_sha256"] = _stable_sha256(exposure)
    (contract_dir / "training_exposure.json").write_text(
        json.dumps(exposure),
        encoding="utf-8",
    )


def _run_validator(checkpoint: Path) -> subprocess.CompletedProcess[str]:
    """Run the v6 evaluation contract validator."""

    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(REPO_ROOT),
            str(RUN_CONFIG),
            METHOD_ID,
            str(checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class V6EvaluationContractTest(unittest.TestCase):
    """Check the v6 full trajectory evaluation contract."""

    def test_validator_exports_the_v6_trajectory_gates(self) -> None:
        """The validator exports each required trajectory gate."""

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            _write_checkpoint(
                checkpoint,
                loss_contract="s0_tool_trajectory_v1",
            )
            result = _run_validator(checkpoint)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "S0_EVALUATION_CONTRACT=full_registry_tool_trajectory_v1",
            result.stdout,
        )
        self.assertIn("S0_TEST_ROWS=85283", result.stdout)
        self.assertIn("S0_EVAL_NUM_NODES=64", result.stdout)
        self.assertIn("S0_MAX_NEW_TOKENS=384", result.stdout)
        self.assertIn("S0_MAX_TOTAL_TOKENS=1536", result.stdout)
        for name in (
            "S0_MINIMUM_VALID_TOOL_TRAJECTORY",
            "S0_MINIMUM_REASONING_PRESENT",
            "S0_MINIMUM_FAMILY_FINAL_ANSWER_ACCURACY",
        ):
            self.assertIn(f"{name}=0.99", result.stdout)
        self.assertIn("S0_MINIMUM_EXACT_REASONING=0.0", result.stdout)

    def test_validator_rejects_an_unknown_loss_contract(self) -> None:
        """The trajectory test rejects an unknown loss contract."""

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            _write_checkpoint(
                checkpoint,
                loss_contract="unknown_loss_contract",
            )
            result = _run_validator(checkpoint)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "The checkpoint loss contract changed",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
