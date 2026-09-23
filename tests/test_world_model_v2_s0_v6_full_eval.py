"""Test the v6 full-registry evaluator and public bundle."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.build_world_model_v2_s0_full_tool_eval as full_module
import scripts.build_world_model_v2_s0_generation_bundle as bundle_module
import scripts.build_world_model_v2_s0_tool_sft as tool_builder
from scripts.build_world_model_v2_s0_full_tool_eval import (
    EVALUATOR_SCHEMA_VERSION,
    FULL_DATASET_ID,
    FULL_EVALUATION_CONTRACT,
    build_full_evaluator,
)
from scripts.build_world_model_v2_s0_generation_bundle import (
    BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
    build_generation_bundle,
)


def _write_json(path: Path, value: object) -> None:
    """Write one stable JSON test file."""

    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read one JSON Lines test file."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _build_tiny_v6_panel(tmp_path: Path, monkeypatch) -> tuple[Path, dict]:
    """Build one small full-registry trajectory panel."""

    monkeypatch.setattr(tool_builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(full_module, "REPO_ROOT", tmp_path)
    registry_path = tmp_path / "registry.json"
    registry = {
        "schema_version": "mentor-rl-world-model-s0-ensembl-registry-v1",
        "source": {"release": 116},
        "counts": {"excluded_gene_ids_without_name": 0},
        "gene_symbols_by_id": {
            "ENSG00000000001": ["ONE", "SHARED"],
            "ENSG00000000002": ["SHARED", "TWO"],
        },
    }
    _write_json(registry_path, registry)
    registry_sha256 = full_module.sha256_file(registry_path)

    source_path = tmp_path / "source.json"
    source = {
        "schema_version": "mentor-rl-world-model-s0-tool-build-v6",
        "dataset_id": "world_model_v2_s0_human_identifier_trajectories_v6",
        "identifier_registry": {
            "path": "registry.json",
            "expected_sha256": registry_sha256,
        },
        "prompt_forms": {
            "test": {
                "human_symbol_to_ensembl": "Resolve symbol {gene_symbol}.",
                "human_ensembl_to_symbol": "Resolve ID {gene_id}.",
                "human_ambiguous_symbol": "Resolve symbol {gene_symbol}.",
            }
        },
    }
    _write_json(source_path, source)

    config_path = tmp_path / "full.json"
    config = {
        "schema_version": (
            "mentor-rl-world-model-s0-full-tool-trajectory-eval-build-v6"
        ),
        "evaluation_id": "tiny-v6-full-registry",
        "source_build_config": {
            "path": "source.json",
            "sha256": full_module.sha256_file(source_path),
        },
        "outputs": {"evaluator_dir": "evaluator"},
    }
    _write_json(config_path, config)
    manifest = build_full_evaluator(config_path)
    return tmp_path / "evaluator/manifest.json", manifest


def test_v6_full_evaluator_has_complete_trajectory_answers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Add the expected reason and final answer to each v6 answer row."""

    evaluator_path, manifest = _build_tiny_v6_panel(tmp_path, monkeypatch)
    answers = _read_jsonl(evaluator_path.parent / "test_answer_key.jsonl")
    questions = _read_jsonl(evaluator_path.parent / "test_questions.jsonl")

    assert manifest["schema_version"] == EVALUATOR_SCHEMA_VERSION
    assert manifest["dataset_id"] == FULL_DATASET_ID
    assert manifest["evaluation_contract"] == (
        FULL_EVALUATION_CONTRACT
    )
    assert manifest["test"]["row_count"] == 5
    assert all("expected_assistant_thinking" in row for row in answers)
    assert all("expected_final_answer" in row for row in answers)
    assert all("assistant_thinking" not in row for row in questions)
    assert all("assistant_final" not in row for row in questions)


def test_v6_full_generation_bundle_is_answer_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Keep all private v6 trajectory targets out of the public bundle."""

    evaluator_path, evaluator = _build_tiny_v6_panel(tmp_path, monkeypatch)
    monkeypatch.setattr(bundle_module, "REPO_ROOT", tmp_path)
    bundle = build_generation_bundle(
        evaluator_manifest_path=evaluator_path,
        evaluator_manifest_sha256=full_module.sha256_file(evaluator_path),
        output_dir=tmp_path / "bundle",
    )
    questions = _read_jsonl(tmp_path / "bundle/questions.jsonl")

    assert bundle["schema_version"] == (
        BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION
    )
    assert bundle["dataset_id"] == FULL_DATASET_ID
    assert bundle["evaluation_contract"] == (
        FULL_EVALUATION_CONTRACT
    )
    assert bundle["record_count"] == evaluator["test"]["row_count"]
    assert bundle["reads_private_answer_keys"] is False
    assert bundle["identifier_registry"] == evaluator["identifier_registry"]
    assert all("expected_assistant_thinking" not in row for row in questions)
    assert all("expected_final_answer" not in row for row in questions)
    assert all("expected_tool_payload" not in row for row in questions)
