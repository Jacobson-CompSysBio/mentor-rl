"""Test flexible S0 checkpoint validation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_world_model_v2_s0_exact_generation import (
    stable_sha256,
    validate_checkpoint,
)


def write_json(path: Path, value: dict) -> None:
    """Write one JSON object."""

    path.write_text(json.dumps(value), encoding="utf-8")


def test_test_path_accepts_partial_train_exposure(tmp_path: Path) -> None:
    """The test path accepts a complete partial-data checkpoint."""

    method_id = "oss20b-plain-base-tokenizer-lora-r32"
    identity = {
        "run_id": "subset-run",
        "method_id": method_id,
        "corpus_manifest_sha256": "ignored-corpus-value",
        "tokenizer_manifest_sha256": "ignored-tokenizer-value",
        "model_identity_sha256": "model-value",
    }
    write_json(
        tmp_path / "tp_adapter_manifest.json",
        {"format": "mentor-rl-s0-tp-lora-v1", "identity": identity},
    )
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"adapter")

    contract_dir = tmp_path / "run_contract"
    contract_dir.mkdir()
    exposure = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": "complete",
        "method_id": method_id,
        "exposure_contract": {
            "scope": "all_eligible_train_rows",
            "all_eligible_train_rows_required": False,
            "satisfied": True,
        },
        "logical_exposure": {
            "all_eligible_train_rows_exposed": False,
            "unique_record_count": 1_000,
        },
    }
    exposure["manifest_sha256"] = stable_sha256(exposure)
    write_json(contract_dir / "training_exposure.json", exposure)

    artifact, run_id = validate_checkpoint(
        tmp_path,
        method_id=method_id,
        corpus_manifest_sha256="ignored-corpus-value",
        tokenizer_manifest_sha256="ignored-tokenizer-value",
        model_identity_sha256="model-value",
    )

    assert run_id == "subset-run"
    assert len(artifact["checkpoint_identity_sha256"]) == 64
