"""Test the flexible S0 train contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts/validate_world_model_v2_s0_train_contract.py"
BASE_CONFIG = REPO_ROOT / "config/world_model_v2_s0_20b_qualification_v4.json"
METHOD_ID = "oss20b-fully-atomic-identifiers-lora-r32"


def run_validator(
    tmp_path: Path,
    *,
    max_steps: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the validator with a 1,000-row contract."""

    root = tmp_path / "repo"
    config_path = root / "config/run.json"
    evaluator_path = (
        root / "data/world_model_v2/eval/s0_human_identifiers_v4/manifest.json"
    )
    config_path.parent.mkdir(parents=True)
    evaluator_path.parent.mkdir(parents=True)

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["run_scope"] = "qualification"
    config["corpus"]["train_rows"] = 1_000
    config["corpus"]["manifest_sha256"] = "ignored-manifest-value"
    config["corpus"]["train_sha256"] = "ignored-train-value"
    config["corpus"]["validation_sha256"] = "ignored-validation-value"
    for method in config["methods"]:
        method["arm_manifest_sha256"] = "ignored-arm-value"
        method["tokenizer_manifest_sha256"] = "ignored-tokenizer-value"
    config["run_settings"]["num_train_epochs"] = 3
    if max_steps is not None:
        config["run_settings"]["max_steps"] = max_steps
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluator = {
        "validation": {
            "questions_path": "ignored/questions.jsonl",
            "questions_sha256": "ignored-question-value",
            "row_count": 1,
            "answer_key_path": "data/world_model_v2/eval/answers.jsonl",
            "answer_key_sha256": "ignored-answer-value",
        }
    }
    evaluator_path.write_text(
        json.dumps(evaluator, indent=2) + "\n",
        encoding="utf-8",
    )

    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(root),
            str(config_path),
            METHOD_ID,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validator_calculates_steps_from_epochs(
    tmp_path: Path,
) -> None:
    """The validator calculates steps for a smaller dataset."""

    result = run_validator(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "S0_RUN_SCOPE=qualification" in result.stdout
    assert "S0_TRAIN_ROWS=1000" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=32" in result.stdout
    assert "TRAIN_TOTAL_STEPS=96" in result.stdout
    assert "TRAIN_MAX_STEPS" not in result.stdout


@pytest.mark.parametrize("max_steps", [-1, 96])
def test_validator_rejects_max_steps(
    tmp_path: Path,
    max_steps: int,
) -> None:
    """The validator rejects the removed step parameter."""

    result = run_validator(tmp_path, max_steps=max_steps)

    assert result.returncode != 0
    assert "max_steps is not supported" in result.stderr
