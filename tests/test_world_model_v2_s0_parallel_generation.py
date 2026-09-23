"""Test deterministic S0 prediction shards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_world_model_v2_s0_generation_bundle import (
    BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
    FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
)
from scripts.run_world_model_v2_s0_exact_generation import (
    INFERENCE_DEVICE_MAP,
    SHARD_PREDICTION_SCHEMA_VERSION,
    SHARD_STRATEGY,
    S0GenerationError,
    merge_prediction_shards,
    select_shard_rows,
    sha256_file,
    stable_sha256,
    write_json,
    write_jsonl,
)


def _write_bundle(path: Path, row_count: int = 10) -> list[dict]:
    """Write one small answer-free tool bundle."""

    rows = [
        {
            "record_id": f"record-{index:02d}",
            "system": "Use the tool.",
            "question": f"Question {index}",
            "input": {"value": index},
            "metadata": {"question_family": "test"},
            "provenance": {"source": "test"},
            "split": "test",
            "tools": [],
        }
        for index in range(row_count)
    ]
    questions_path = path / "questions.jsonl"
    write_jsonl(questions_path, rows)
    record_ids = [row["record_id"] for row in rows]
    manifest = {
        "schema_version": BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
        "evaluation_contract": FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "identifier_registry": {
            "path": (
                "data/world_model_v2/sources/"
                "ensembl_116_human_gene_ids_and_symbols.json"
            ),
            "id": (
                "sha256:"
                "e67a4a1d8839c7f8ea568c3040c18941"
                "a189f65afeed66dfeedb03c6e305c4ab"
            ),
            "sha256": (
                "e67a4a1d8839c7f8ea568c3040c18941"
                "a189f65afeed66dfeedb03c6e305c4ab"
            ),
        },
        "record_count": len(rows),
        "record_ids_sha256": stable_sha256(record_ids),
        "questions_sha256": sha256_file(questions_path),
        "reads_private_answer_keys": False,
    }
    manifest["bundle_sha256"] = stable_sha256(manifest)
    write_json(path / "manifest.json", manifest)
    return rows


def _write_shards(
    path: Path,
    rows: list[dict],
    *,
    shard_count: int,
) -> None:
    """Write one valid prediction shard set."""

    generation_config = {"do_sample": False, "batch_size": 4}
    common = {
        "evaluation_contract": FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "method_id": "test-method",
        "train_run_id": "test-run",
        "base_model_identity_sha256": "b" * 64,
        "checkpoint_identity_sha256": "c" * 64,
        "tokenizer_manifest_sha256": "d" * 64,
        "generation_bundle_sha256": "",
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False,
    }
    bundle = json.loads(
        (path.parent / "bundle/manifest.json").read_text(encoding="utf-8")
    )
    common["generation_bundle_sha256"] = bundle["bundle_sha256"]
    for shard_index in range(shard_count):
        shard_dir = path / f"shard-{shard_index:05d}"
        selected = select_shard_rows(
            rows,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        predictions = [
            {
                "record_id": row["record_id"],
                "encoded_prediction": row["record_id"],
                "raw_generation": row["record_id"],
                "prompt_tokens": 1,
                "generated_tokens": 1,
                "disallowed_special_tokens": [],
            }
            for row in selected
        ]
        predictions_path = shard_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        manifest = {
            "schema_version": SHARD_PREDICTION_SCHEMA_VERSION,
            **common,
            "record_count": len(predictions),
            "full_record_count": len(rows),
            "shard_index": shard_index,
            "shard_count": shard_count,
            "shard_strategy": SHARD_STRATEGY,
            "predictions_sha256": sha256_file(predictions_path),
            "elapsed_seconds": float(shard_index + 1),
        }
        manifest["manifest_sha256"] = stable_sha256(manifest)
        write_json(shard_dir / "generation_manifest.json", manifest)


def test_merge_restores_the_original_record_order(tmp_path: Path) -> None:
    """Restore the bundle order from deterministic strided shards."""

    bundle_dir = tmp_path / "bundle"
    rows = _write_bundle(bundle_dir)
    shards_dir = tmp_path / "shards"
    _write_shards(shards_dir, rows, shard_count=3)

    result = merge_prediction_shards(
        bundle_root=bundle_dir,
        shards_root=shards_dir,
        output_dir=tmp_path / "merged",
        expected_shard_count=3,
    )
    predictions = [
        json.loads(line)
        for line in (tmp_path / "merged/predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert [row["record_id"] for row in predictions] == [
        row["record_id"] for row in rows
    ]
    assert result["manifest"]["record_count"] == len(rows)
    assert result["manifest"]["parallelism"]["shard_count"] == 3
    assert result["manifest"]["elapsed_seconds"] == 3.0


def test_merge_accepts_one_explicit_prediction_shard(tmp_path: Path) -> None:
    """Merge one shard from the launcher path."""

    bundle_dir = tmp_path / "bundle"
    rows = _write_bundle(bundle_dir)
    shards_dir = tmp_path / "shards"
    _write_shards(shards_dir, rows, shard_count=1)

    result = merge_prediction_shards(
        bundle_root=bundle_dir,
        shards_root=shards_dir,
        output_dir=tmp_path / "merged",
        expected_shard_count=1,
    )

    assert result["manifest"]["record_count"] == len(rows)
    assert result["manifest"]["parallelism"]["shard_count"] == 1


def test_inference_reserves_gpu_zero_memory() -> None:
    """Use the placement policy that keeps more free memory on GPU zero."""

    assert INFERENCE_DEVICE_MAP == "balanced_low_0"


def test_merge_rejects_one_missing_shard(tmp_path: Path) -> None:
    """Reject a shard set that lacks one expected directory."""

    bundle_dir = tmp_path / "bundle"
    rows = _write_bundle(bundle_dir)
    shards_dir = tmp_path / "shards"
    _write_shards(shards_dir, rows, shard_count=3)
    missing = shards_dir / "shard-00002"
    for path in missing.iterdir():
        path.unlink()
    missing.rmdir()

    with pytest.raises(S0GenerationError, match="shard set changed"):
        merge_prediction_shards(
            bundle_root=bundle_dir,
            shards_root=shards_dir,
            output_dir=tmp_path / "merged",
            expected_shard_count=3,
        )


def test_merge_rejects_the_wrong_shard_records(tmp_path: Path) -> None:
    """Reject a shard that contains a record from another shard."""

    bundle_dir = tmp_path / "bundle"
    rows = _write_bundle(bundle_dir)
    shards_dir = tmp_path / "shards"
    _write_shards(shards_dir, rows, shard_count=3)
    shard_dir = shards_dir / "shard-00000"
    predictions_path = shard_dir / "predictions.jsonl"
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    predictions[0]["record_id"] = rows[1]["record_id"]
    write_jsonl(predictions_path, predictions)
    manifest = json.loads(
        (shard_dir / "generation_manifest.json").read_text(encoding="utf-8")
    )
    manifest["predictions_sha256"] = sha256_file(predictions_path)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(shard_dir / "generation_manifest.json", manifest)

    with pytest.raises(S0GenerationError, match="wrong records"):
        merge_prediction_shards(
            bundle_root=bundle_dir,
            shards_root=shards_dir,
            output_dir=tmp_path / "merged",
            expected_shard_count=3,
        )
