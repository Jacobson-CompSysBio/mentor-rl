"""Test the S0 train contract validator."""

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
    max_steps: int,
) -> subprocess.CompletedProcess[str]:
    """Run the validator with one isolated contract root."""

    root = tmp_path / "repo"
    config_path = root / "config/run.json"
    evaluator_path = (
        root / "data/world_model_v2/eval/s0_human_identifiers_v4/manifest.json"
    )
    config_path.parent.mkdir(parents=True)
    evaluator_path.parent.mkdir(parents=True)

    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    config["run_scope"] = "debug_qualification"
    config["run_settings"]["num_train_epochs"] = 3
    config["run_settings"]["max_steps"] = max_steps
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    corpus = config["corpus"]
    evaluator = {
        "validation": {
            "questions_path": f"{corpus['root']}/{corpus['validation_file']}",
            "questions_sha256": corpus["validation_sha256"],
            "row_count": corpus["validation_rows"],
            "answer_key_path": "data/world_model_v2/eval/validation_answers.jsonl",
            "answer_key_sha256": "0" * 64,
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


def test_validator_allows_steps_below_source_exposure(tmp_path: Path) -> None:
    """The validator accepts a step cap that uses a train subset."""

    result = run_validator(tmp_path, max_steps=7)

    assert result.returncode == 0, result.stderr
    assert "TRAIN_NUM_EPOCHS=3" in result.stdout
    assert "TRAIN_MAX_STEPS=7" in result.stdout


def test_validator_allows_automatic_steps(tmp_path: Path) -> None:
    """The validator accepts the trainer's automatic step mode."""

    result = run_validator(tmp_path, max_steps=-1)

    assert result.returncode == 0, result.stderr
    assert "TRAIN_NUM_EPOCHS=3" in result.stdout
    assert "TRAIN_MAX_STEPS=-1" in result.stdout


@pytest.mark.parametrize("max_steps", [0, -2])
def test_validator_rejects_invalid_steps(
    tmp_path: Path,
    max_steps: int,
) -> None:
    """The validator rejects an invalid step cap."""

    result = run_validator(tmp_path, max_steps=max_steps)

    assert result.returncode != 0
    assert "max_steps must be -1 or a positive integer" in result.stderr
