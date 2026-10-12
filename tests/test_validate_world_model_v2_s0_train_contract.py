"""Test the S0 v4 train contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts/validate_world_model_v2_s0_train_contract.py"
TRAIN_LAUNCHER = REPO_ROOT / "world_model_v2_s0.slurm"
V4_TRAINING_CONFIGS = [
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r32_uniform.json",
        "oss120b-plain-base-tokenizer-lora-r32",
        32,
        "multiplex_sft_v3",
    ),
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r32_priority.json",
        "oss120b-plain-base-tokenizer-lora-r32",
        32,
        "multiplex_sft_v4_priority",
    ),
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r128_uniform.json",
        "oss120b-plain-base-tokenizer-lora-r128",
        128,
        "multiplex_sft_v3",
    ),
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r128_priority.json",
        "oss120b-plain-base-tokenizer-lora-r128",
        128,
        "multiplex_sft_v4_priority",
    ),
]
ARM_IDS = ("r32-uniform", "r32-priority", "r128-uniform", "r128-priority")
requires_v4_corpus = pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s0/manifest.json").is_file(),
    reason="The v4 corpus is not built",
)


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


@requires_v4_corpus
@pytest.mark.parametrize(
    ("config", "method_id", "rank", "loss_contract"),
    V4_TRAINING_CONFIGS,
    ids=ARM_IDS,
)
def test_validator_accepts_each_v4_lora_and_loss_arm(
    config: Path,
    method_id: str,
    rank: int,
    loss_contract: str,
) -> None:
    result = run_validator(config, method_id)

    assert result.returncode == 0, result.stderr
    assert "S0_DATASET_ID=world_model_v4_s0_human_identifiers" in result.stdout
    assert "S0_TRAIN_ROWS=16384" in result.stdout
    assert "S0_VALIDATION_ROWS=622" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=512" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1536" in result.stdout
    assert "MAX_LENGTH=4096" in result.stdout
    assert f"LORA_R={rank}" in result.stdout
    assert f"LORA_ALPHA={2 * rank}" in result.stdout
    assert f"LOSS_CONTRACT={loss_contract}" in result.stdout
    assert "SNAPSHOT" not in result.stdout
    assert "INIT_ADAPTER" not in result.stdout


def _changed_config(tmp_path: Path, change) -> Path:
    """Write one changed copy of the r32 priority config below a fake root."""

    config = json.loads(V4_TRAINING_CONFIGS[1][0].read_text(encoding="utf-8"))
    change(config)
    config_path = tmp_path / "repo/config/changed.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config_path


def _set(section: str, key: str, value: object):
    def change(config: dict) -> None:
        if section:
            config[section][key] = value
        else:
            config[key] = value

    return change


# Each check runs before the validator reads any corpus file.
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            _set("", "schema_version", "mentor-rl-world-model-s0-120b-training-v3"),
            "The train config schema is not supported",
        ),
        (_set("run_settings", "max_steps", 96), "training schedule changed"),
        (_set("run_settings", "snapshot_steps", 0), "training schedule changed"),
        (_set("run_settings", "learning_rate", 0.0002), "training schedule changed"),
        (_set("", "run_scope", "qualification"), "run identity changed"),
        (_set("", "training_id", "renamed"), "run identity changed"),
        (_set("trainer", "lora_alpha", 63), "rank/alpha contract changed"),
        (_set("trainer", "token_loss_weights", None), "priority loss weights changed"),
        (_set("trainer", "loss_contract", "s0_tool_trajectory_v1"), "loss contract changed"),
        (_set("corpus", "train_rows", 4096), "optimizer-step schedule changed"),
        (_set("validation_gate", "snapshot_steps", 0), "validation gate changed"),
    ],
    ids=(
        "schema",
        "max_steps",
        "snapshot_steps",
        "learning_rate",
        "run_scope",
        "training_id",
        "lora_alpha",
        "priority_weights",
        "loss_contract",
        "train_rows",
        "validation_gate",
    ),
)
def test_validator_rejects_v4_drift(
    tmp_path: Path,
    change,
    message: str,
) -> None:
    config_path = _changed_config(tmp_path, change)

    result = run_validator(
        config_path,
        V4_TRAINING_CONFIGS[1][1],
        root=tmp_path / "repo",
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_validator_rejects_a_method_for_another_rank(tmp_path: Path) -> None:
    config_path = _changed_config(tmp_path, lambda config: None)

    result = run_validator(
        config_path,
        "oss120b-plain-base-tokenizer-lora-r128",
        root=tmp_path / "repo",
    )

    assert result.returncode != 0
    assert "S0_METHOD_ID must select one method" in result.stderr


@requires_v4_corpus
@pytest.mark.parametrize(
    ("config", "method_id", "rank", "loss_contract"),
    V4_TRAINING_CONFIGS[:2],
    ids=ARM_IDS[:2],
)
def test_slurm_launcher_resolves_v4_train_contracts(
    config: Path,
    method_id: str,
    rank: int,
    loss_contract: str,
) -> None:
    """Resolve the v4 train configs through the launcher entry point."""

    del rank, loss_contract
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_")
    }
    environment.update(
        {
            "CONTRACT_ONLY": "1",
            "JOB_MODE": "train",
            "RUN_CONFIG": str(config),
            "S0_METHOD_ID": method_id,
        }
    )
    result = subprocess.run(
        ["bash", str(TRAIN_LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Run: {method_id}-slurm-JOB-ID" in result.stdout
    assert "The selected inputs and config are valid" in result.stdout
