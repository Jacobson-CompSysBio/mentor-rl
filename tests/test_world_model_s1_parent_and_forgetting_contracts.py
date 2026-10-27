"""Test the S0 parent checkpoint for S1 and the S0 forgetting test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.validate_world_model_v2_s0_train_contract as train_contract


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_VALIDATOR = REPO_ROOT / "scripts/validate_world_model_v2_s0_eval_contract.py"
FORGETTING_CONFIG = (
    REPO_ROOT / "config/world_model_s1_120b_s0_forgetting_test_512n.json"
)
S0_FINAL_CONFIG = (
    REPO_ROOT / "config/world_model_s0_120b_test_v4_final_512n.json"
)
S1_TRAINING_CONFIG = REPO_ROOT / "config/world_model_s1_120b_training_v2.json"
REAL_PARENT = (
    REPO_ROOT
    / "data/world_model_v4/checkpoints/s0_120b/r32_uniform/slurm-5631883"
)
METHOD_ID = "oss120b-plain-base-tokenizer-lora-r32"
MODEL_IDENTITY = (
    "905e86330e48859268f5e51fd569acf619e6030e1508b37e6a82e61dcf0c7889"
)
TOKENIZER_MANIFEST = (
    "245163b5189a64afb8e5b061585f90d4c15bcf94599b68307e9ff008880fe3c0"
)
PARENT_RUN_ID = "oss120b-plain-base-tokenizer-lora-r32-slurm-5631883"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _complete_exposure(run_id: str) -> dict:
    """Return one content-addressed exposure receipt for a finished run."""

    exposure = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": "complete",
        "run_id": run_id,
        "method_id": METHOD_ID,
        "logical_exposure": {"all_eligible_train_rows_exposed": True},
        "exposure_contract": {"satisfied": True},
        "schedule": {"completed_global_step": 1536, "total_steps": 1536},
    }
    exposure["manifest_sha256"] = train_contract.stable_sha256(exposure)
    return exposure


def _fake_checkpoint(
    directory: Path,
    *,
    run_id: str,
    corpus_manifest_sha256: str,
    train_sha256: str,
    run_config: dict | None = None,
    initialization: dict | None = None,
) -> Path:
    """Write the files that the contract checkers read from one checkpoint."""

    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(b"adapter " + run_id.encode())
    _write_json(directory / "adapter_config.json", {"r": 32})
    run_config_sha256 = "0" * 64
    if run_config is not None:
        _write_json(directory / "run_contract/run_config.json", run_config)
        run_config_sha256 = _sha256(directory / "run_contract/run_config.json")
    manifest = {
        "format": "mentor-rl-s0-tp-lora-v1",
        "identity": {
            "run_id": run_id,
            "method_id": METHOD_ID,
            "model_identity_sha256": MODEL_IDENTITY,
            "tokenizer_manifest_sha256": TOKENIZER_MANIFEST,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "train_sha256": train_sha256,
            "run_config_sha256": run_config_sha256,
        },
        "lora": {"r": 32, "alpha": 64, "dropout": 0.0},
        "objective": {"loss_contract": "multiplex_sft_v3"},
    }
    if initialization is not None:
        manifest["initialization"] = initialization
    _write_json(directory / "tp_adapter_manifest.json", manifest)
    _write_json(
        directory / "run_contract/training_exposure.json",
        _complete_exposure(run_id),
    )
    return directory


def _fake_parent(directory: Path) -> Path:
    return _fake_checkpoint(
        directory,
        run_id=PARENT_RUN_ID,
        corpus_manifest_sha256="1" * 64,
        train_sha256="2" * 64,
        run_config={"schema_version": train_contract.S0_V4_TRAINING_SCHEMA},
    )


def _initialization(checkpoint: Path, root: Path, **changes: object) -> dict:
    initialization = {
        "parent": "s0_checkpoint",
        "checkpoint_path": str(checkpoint.relative_to(root)),
        "run_id": PARENT_RUN_ID,
        "adapter_sha256": _sha256(checkpoint / "adapter_model.safetensors"),
        "lora_rank": 32,
        "loss_contract": "multiplex_sft_v3",
    }
    initialization.update(changes)
    return initialization


def _validate_parent(root: Path, initialization: dict) -> dict[str, str]:
    return train_contract.validate_parent_checkpoint(
        root,
        initialization,
        model_identity_sha256=MODEL_IDENTITY,
        tokenizer_manifest_sha256=TOKENIZER_MANIFEST,
        lora_rank=32,
        lora_alpha=64,
    )


def test_parent_checkpoint_exports_the_adapter_to_load(tmp_path: Path) -> None:
    parent = _fake_parent(tmp_path / "parent")

    values = _validate_parent(tmp_path, _initialization(parent, tmp_path))

    assert values == {
        "INIT_ADAPTER_DIR": str(parent.resolve()),
        "INIT_ADAPTER_SHA256": _sha256(parent / "adapter_model.safetensors"),
        "INIT_ADAPTER_RUN_ID": PARENT_RUN_ID,
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"adapter_sha256": "f" * 64}, "parent adapter identity changed"),
        ({"lora_rank": 128}, "parent LoRA rank differs"),
        ({"run_id": "another-run"}, "parent checkpoint contract changed"),
        (
            {"loss_contract": "multiplex_sft_v4_priority"},
            "parent checkpoint contract changed",
        ),
        ({"parent": "base_model"}, "must be s0_checkpoint"),
    ],
)
def test_parent_checkpoint_rejects_a_changed_pin(
    tmp_path: Path, changes: dict, message: str
) -> None:
    parent = _fake_parent(tmp_path / "parent")

    with pytest.raises(SystemExit, match=message):
        _validate_parent(tmp_path, _initialization(parent, tmp_path, **changes))


def test_parent_checkpoint_must_be_a_complete_run(tmp_path: Path) -> None:
    parent = _fake_parent(tmp_path / "parent")
    exposure = _complete_exposure(PARENT_RUN_ID)
    exposure.pop("manifest_sha256")
    exposure["schedule"] = {"completed_global_step": 512, "total_steps": 1536}
    exposure["manifest_sha256"] = train_contract.stable_sha256(exposure)
    _write_json(parent / "run_contract/training_exposure.json", exposure)

    with pytest.raises(SystemExit, match="parent S0 run is not complete"):
        _validate_parent(tmp_path, _initialization(parent, tmp_path))


def test_parent_checkpoint_must_be_an_s0_v4_run(tmp_path: Path) -> None:
    parent = _fake_checkpoint(
        tmp_path / "parent",
        run_id=PARENT_RUN_ID,
        corpus_manifest_sha256="1" * 64,
        train_sha256="2" * 64,
        run_config={"schema_version": train_contract.S1_PRODUCTION_SCHEMA},
    )

    with pytest.raises(SystemExit, match="not an S0 v4 training run"):
        _validate_parent(tmp_path, _initialization(parent, tmp_path))


@pytest.mark.skipif(not REAL_PARENT.is_dir(), reason="the S0 v4 run is absent")
def test_completed_s0_v4_run_is_a_valid_parent() -> None:
    initialization = _initialization(REAL_PARENT, REPO_ROOT)

    values = _validate_parent(REPO_ROOT, initialization)

    assert values["INIT_ADAPTER_RUN_ID"] == PARENT_RUN_ID


def _run_eval_validator(
    config: Path, checkpoint: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(EVAL_VALIDATOR),
            str(REPO_ROOT),
            str(config),
            METHOD_ID,
            str(checkpoint),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _fake_s1_checkpoint(
    directory: Path, *, initialization: dict | None,
    training_config: Path = S1_TRAINING_CONFIG,
) -> Path:
    corpus = json.loads(training_config.read_text(encoding="utf-8"))["corpus"]
    return _fake_checkpoint(
        directory,
        run_id="oss120b-plain-base-tokenizer-lora-r32-slurm-1",
        corpus_manifest_sha256=corpus["manifest_sha256"],
        train_sha256=corpus["train_sha256"],
        initialization=initialization,
    )


@pytest.mark.parametrize("version", ["v2", "v8"])
def test_forgetting_test_accepts_an_s1_checkpoint_from_s0(tmp_path: Path, version: str) -> None:
    """Check the original and v8 configs using explicit temporary checkpoint fixtures."""
    training = S1_TRAINING_CONFIG if version == "v2" else (
        REPO_ROOT / "config/world_model_s1_120b_training_v8_32n.json"
    )
    config = FORGETTING_CONFIG if version == "v2" else (
        REPO_ROOT / "config/world_model_s1_120b_s0_forgetting_test_v8_512n.json"
    )
    checkpoint = _fake_s1_checkpoint(
        tmp_path / "s1",
        initialization={"parent": "s0_checkpoint", "run_id": PARENT_RUN_ID},
        training_config=training,
    )

    result = _run_eval_validator(config, checkpoint)

    assert result.returncode == 0, result.stderr
    assert "S0_WANDB_METRIC_PREFIX=s0_forgetting" in result.stdout
    assert "S0_EVAL_NUM_NODES=512" in result.stdout
    assert "S0_TEST_ROWS=120766" in result.stdout


def test_v8_forgetting_rejects_an_older_corpus_checkpoint(tmp_path: Path) -> None:
    """The v8 gate must not evaluate an older S1 adapter by mistake."""
    checkpoint = _fake_s1_checkpoint(
        tmp_path / "old_s1",
        initialization={"parent": "s0_checkpoint", "run_id": PARENT_RUN_ID},
    )
    result = _run_eval_validator(
        REPO_ROOT / "config/world_model_s1_120b_s0_forgetting_test_v8_512n.json",
        checkpoint,
    )
    assert result.returncode != 0
    assert "checkpoint contract changed" in result.stderr


def test_forgetting_test_rejects_an_s1_checkpoint_from_base(tmp_path: Path) -> None:
    checkpoint = _fake_s1_checkpoint(tmp_path / "s1", initialization=None)

    result = _run_eval_validator(FORGETTING_CONFIG, checkpoint)

    assert result.returncode != 0
    assert "started from S0" in result.stderr


@pytest.mark.skipif(not REAL_PARENT.is_dir(), reason="the S0 v4 run is absent")
def test_forgetting_test_rejects_an_s0_checkpoint() -> None:
    result = _run_eval_validator(FORGETTING_CONFIG, REAL_PARENT)

    assert result.returncode != 0
    assert "checkpoint contract changed" in result.stderr


@pytest.mark.skipif(not REAL_PARENT.is_dir(), reason="the S0 v4 run is absent")
def test_s0_test_keeps_the_test_metric_prefix() -> None:
    result = _run_eval_validator(S0_FINAL_CONFIG, REAL_PARENT)

    assert result.returncode == 0, result.stderr
    assert "S0_WANDB_METRIC_PREFIX=test" in result.stdout
