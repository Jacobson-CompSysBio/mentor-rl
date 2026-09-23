"""Test the canonical S0 train contract."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts/validate_world_model_v2_s0_train_contract.py"
QUALIFICATION_CONFIG = (
    REPO_ROOT
    / "config/world_model_v2_s0_20b_tool_trajectory_qualification_v6.json"
)
TRAINING_CONFIG = (
    REPO_ROOT
    / "config/world_model_v2_s0_120b_tool_trajectory_training_v6.json"
)
QUALIFICATION_METHOD = "oss20b-plain-base-tokenizer-lora-r32"
TRAINING_METHOD = "oss120b-plain-base-tokenizer-lora-r32"


def run_validator(
    config: Path,
    method_id: str,
    *,
    root: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run the train validator."""

    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(root),
            str(config),
            method_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("config", "method_id", "model_id", "run_scope"),
    [
        (
            QUALIFICATION_CONFIG,
            QUALIFICATION_METHOD,
            "gpt-oss-20b-bf16",
            "qualification",
        ),
        (
            TRAINING_CONFIG,
            TRAINING_METHOD,
            "gpt-oss-120b-bf16",
            "production",
        ),
    ],
)
def test_validator_accepts_the_v6_trajectory_contracts(
    config: Path,
    method_id: str,
    model_id: str,
    run_scope: str,
) -> None:
    """Accept each canonical trajectory train contract."""

    result = run_validator(config, method_id)

    assert result.returncode == 0, result.stderr
    assert f"MODEL_ID={model_id}" in result.stdout
    assert f"S0_RUN_SCOPE={run_scope}" in result.stdout
    assert "S0_TRAIN_ROWS=4096" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=128" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1280" in result.stdout
    assert "LOSS_CONTRACT=s0_tool_trajectory_v1" in result.stdout


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "max_steps",
            96,
            "max_steps is not supported",
        ),
        (
            "run_scope",
            "unknown",
            "run_scope must be one of",
        ),
    ],
)
def test_validator_rejects_removed_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject one value outside the canonical contract."""

    root = tmp_path / "repo"
    config_path = root / "config/run.json"
    config_path.parent.mkdir(parents=True)
    config = json.loads(
        QUALIFICATION_CONFIG.read_text(encoding="utf-8")
    )
    if field == "max_steps":
        config["run_settings"][field] = value
    else:
        config[field] = value
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_validator(
        config_path,
        QUALIFICATION_METHOD,
        root=root,
    )

    assert result.returncode != 0
    assert message in result.stderr
