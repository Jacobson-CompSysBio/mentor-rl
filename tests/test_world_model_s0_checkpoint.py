"""Test the checkpoint check of the S0 test generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_world_model_v2_s0_exact_generation import (
    S0GenerationError,
    stable_sha256,
    validate_checkpoint,
)


METHOD_ID = "oss120b-plain-base-tokenizer-lora-r32"


def write_json(path: Path, value: dict) -> None:
    """Write one JSON object."""

    path.write_text(json.dumps(value), encoding="utf-8")


def write_checkpoint(path: Path, *, exposure_status: str) -> None:
    """Write the files that the generator checks in one final checkpoint."""

    identity = {
        "run_id": "train-run",
        "method_id": METHOD_ID,
        "corpus_manifest_sha256": "corpus-value",
        "tokenizer_manifest_sha256": "tokenizer-value",
        "model_identity_sha256": "model-value",
    }
    write_json(
        path / "tp_adapter_manifest.json",
        {"format": "mentor-rl-s0-tp-lora-v1", "identity": identity},
    )
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")

    contract_dir = path / "run_contract"
    contract_dir.mkdir()
    exposure = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": exposure_status,
        "method_id": METHOD_ID,
        "exposure_contract": {
            "scope": "all_eligible_train_rows",
            "all_eligible_train_rows_required": True,
            "satisfied": True,
        },
        "logical_exposure": {
            "all_eligible_train_rows_exposed": True,
            "unique_record_count": 16_384,
        },
    }
    exposure["manifest_sha256"] = stable_sha256(exposure)
    write_json(contract_dir / "training_exposure.json", exposure)


def check(path: Path):
    return validate_checkpoint(
        path,
        method_id=METHOD_ID,
        corpus_manifest_sha256="corpus-value",
        tokenizer_manifest_sha256="tokenizer-value",
        model_identity_sha256="model-value",
    )


def test_test_path_accepts_a_complete_final_checkpoint(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, exposure_status="complete")

    artifact, run_id = check(tmp_path)

    assert run_id == "train-run"
    assert len(artifact["checkpoint_identity_sha256"]) == 64
    assert set(artifact["files"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
        "run_contract/training_exposure.json",
    }


def test_test_path_rejects_an_unfinished_run(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, exposure_status="planned")

    with pytest.raises(S0GenerationError, match="one complete checkpoint"):
        check(tmp_path)
