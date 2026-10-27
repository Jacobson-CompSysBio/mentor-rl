"""Regression tests for the native S1 evaluation setup."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import (
    S1_OVERLAY_V3_SCHEMA_VERSION,
    S1_OVERLAY_V4_SCHEMA_VERSION,
)
from scripts.build_world_model_s1_generation_bundle import (
    S1_OVERLAY_SCHEMAS as GENERATION_OVERLAY_SCHEMAS,
    build_generation_bundle,
)
from scripts.validate_world_model_s1_eval_contract import (
    EXPECTED_TRANSCRIPT_SHAPES,
    PROMOTION_MAXIMUM_RATE_LIMITS,
    PROMOTION_RATE_GATE_NAMES,
    REQUIRED_BOOLEAN_GATE_NAMES,
    S1_OVERLAY_SCHEMAS as EVAL_OVERLAY_SCHEMAS,
    _model_metadata_identity,
    _validate_corpus,
    _validate_panel,
    _validate_promotion_gate,
    _validate_tiered_gate,
    _validate_tiered_v3_gate,
    _validate_tokenizer,
    _validate_training_link,
    validate_and_export,
)
from scripts.world_model_s1_eval_common import (
    QUESTION_KEYS,
    S1EvaluationError,
    S1_FAMILIES,
    failure_injection_required,
    message_shape,
    model_messages,
    read_json,
    read_jsonl,
    sha256_file,
    stable_sha256,
    validate_answer_row,
    validate_question,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = REPO_ROOT / "data/world_model_v3/eval/s1_smoke"
FULL_ROOT = REPO_ROOT / "data/world_model_v3/eval/s1"
V2_FULL_ROOT = REPO_ROOT / "data/world_model_v4/eval/s1"
V4_FULL_ROOT = REPO_ROOT / "data/world_model_v4/eval/s1_v4"
SOURCE_INDEX = (
    REPO_ROOT
    / "data/world_model_v3/source_indexes/"
    "world_model_source_index_"
    "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
    ".json.gz"
)
TEST_CONFIG_PATH = REPO_ROOT / "config/world_model_s1_120b_test_v1.json"
TRAIN_CONFIG_PATH = REPO_ROOT / "config/world_model_s1_120b_training_v1.json"
METHOD_ID = "oss120b-plain-base-tokenizer-lora-r32"


def _smoke_panel() -> tuple[list[dict], dict[str, dict]]:
    contract = load_contract()
    questions = [
        validate_question(row, contract)
        for row in read_jsonl(SMOKE_ROOT / "test_questions.jsonl")
    ]
    by_id = {row["record_id"]: row for row in questions}
    answers = {
        row["record_id"]: validate_answer_row(
            row,
            by_id[row["record_id"]],
            contract,
        )
        for row in read_jsonl(SMOKE_ROOT / "test_answer_key.jsonl")
    }
    return questions, answers


def test_smoke_generation_bundle_is_native_and_answer_free(tmp_path: Path) -> None:
    """Bundle all native question types without copying private suffixes."""

    evaluator_path = SMOKE_ROOT / "manifest.json"
    bundle = build_generation_bundle(
        evaluator_manifest_path=evaluator_path,
        evaluator_manifest_sha256=sha256_file(evaluator_path),
        source_index_path=SOURCE_INDEX,
        source_index_sha256=(
            "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
        ),
        output_dir=tmp_path / "bundle",
    )
    rows = read_jsonl(tmp_path / "bundle/questions.jsonl")

    assert bundle["record_count"] == len(rows) == 111
    assert set(bundle["family_counts"]) == set(S1_FAMILIES)
    assert bundle["book_mode_counts"] == {
        "closed_book": 49,
        "tool_call": 62,
    }
    assert bundle["failure_row_count"] == 4
    assert bundle["native_transcript"] is True
    assert bundle["reads_private_answer_keys"] is False
    assert all(set(row) == set(QUESTION_KEYS) for row in rows)
    assert not (tmp_path / "bundle/test_answer_key.jsonl").exists()
    assert "expected_messages" not in (
        tmp_path / "bundle/questions.jsonl"
    ).read_text(encoding="utf-8")


def test_v2_generation_bundle_accepts_the_pinned_s1_overlay(tmp_path: Path) -> None:
    """Keep the answer-free builder aligned with the tiered v2 preflight."""

    evaluator_path = V2_FULL_ROOT / "manifest.json"
    bundle = build_generation_bundle(
        evaluator_manifest_path=evaluator_path,
        evaluator_manifest_sha256=sha256_file(evaluator_path),
        source_index_path=SOURCE_INDEX,
        source_index_sha256=(
            "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
        ),
        output_dir=tmp_path / "bundle",
    )

    assert bundle["record_count"] == 6875
    assert bundle["contract"]["s1_overlay"] == {
        "path": "runtime/world_model/s1_v2.yaml",
        "schema_version": "mentor-rl-world-model-s1-overlay-v1",
        "sha256": (
            "15441cf1c81358a490c69c6f3aec51a12f6c016556a695d84eeee9bac7024999"
        ),
    }


def test_v4_eval_consumers_use_the_v3_overlay_schema() -> None:
    """Keep eval preflight and generation on the v4 normalization overlay."""

    overlay_path = Path("runtime/world_model/s1_v4.yaml")
    assert GENERATION_OVERLAY_SCHEMAS[overlay_path] == (
        S1_OVERLAY_V3_SCHEMA_VERSION
    )
    assert EVAL_OVERLAY_SCHEMAS[overlay_path.as_posix()] == (
        S1_OVERLAY_V3_SCHEMA_VERSION
    )


def test_v5_eval_consumers_use_the_v4_overlay_schema() -> None:
    """Accept the finalized v5 overlay in both evaluation consumers."""

    overlay_path = Path("runtime/world_model/s1_v5.yaml")
    assert GENERATION_OVERLAY_SCHEMAS[overlay_path] == (
        S1_OVERLAY_V4_SCHEMA_VERSION
    )
    assert EVAL_OVERLAY_SCHEMAS[overlay_path.as_posix()] == (
        S1_OVERLAY_V4_SCHEMA_VERSION
    )


def test_v5_generation_bundle_preserves_the_frozen_panel(tmp_path: Path) -> None:
    evaluator_path = REPO_ROOT / "data/world_model_v4/eval/s1_v5/manifest.json"
    bundle = build_generation_bundle(
        evaluator_manifest_path=evaluator_path,
        evaluator_manifest_sha256=sha256_file(evaluator_path),
        source_index_path=SOURCE_INDEX,
        source_index_sha256=(
            "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
        ),
        output_dir=tmp_path / "bundle",
    )
    assert bundle["record_count"] == 6875
    assert bundle["test_panel_id"] == (
        "eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904"
    )
    assert bundle["contract"]["s1_overlay"] == {
        "path": "runtime/world_model/s1_v5.yaml",
        "schema_version": S1_OVERLAY_V4_SCHEMA_VERSION,
        "sha256": sha256_file(REPO_ROOT / "runtime/world_model/s1_v5.yaml"),
    }


def test_v5_eval_config_pins_the_new_data_and_preserves_generation() -> None:
    """Check evaluation inputs before the new checkpoint exists."""

    config = read_json(
        REPO_ROOT / "config/world_model_s1_120b_32n_test_v5_512n.json"
    )
    baseline = read_json(
        REPO_ROOT / "config/world_model_s1_120b_32n_test_v4_512n.json"
    )
    training_path = REPO_ROOT / config["training"]["config"]
    training = read_json(training_path)
    assert config["training"]["config_sha256"] == sha256_file(training_path)
    assert config["training"]["training_id"] == training["training_id"]
    assert config["training"]["checkpoint_root"] == (
        training["methods"][0]["output_root"]
    )
    for key in ("generation", "gate", "model", "tracking", "source_index"):
        assert config[key] == baseline[key]
    corpus = _validate_corpus(REPO_ROOT, config["corpus"])
    assert corpus[-1] == 55276
    panel = _validate_panel(
        repo_root=REPO_ROOT,
        panel_config=config["test_panel"],
        corpus_manifest_sha256=corpus[2],
        contract=load_contract(),
        allow_s1_overlay=True,
    )
    assert panel[-1] == 6875
    _validate_tokenizer(
        repo_root=REPO_ROOT,
        method=config["methods"][0],
        training_method=training["methods"][0],
    )


def test_v4_generation_bundle_accepts_the_normalized_corpus(
    tmp_path: Path,
) -> None:
    evaluator_path = V4_FULL_ROOT / "manifest.json"
    bundle = build_generation_bundle(
        evaluator_manifest_path=evaluator_path,
        evaluator_manifest_sha256=sha256_file(evaluator_path),
        source_index_path=SOURCE_INDEX,
        source_index_sha256=(
            "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
        ),
        output_dir=tmp_path / "bundle",
    )

    assert bundle["record_count"] == 6875
    assert bundle["contract"]["s1_overlay"] == {
        "path": "runtime/world_model/s1_v4.yaml",
        "schema_version": S1_OVERLAY_V3_SCHEMA_VERSION,
        "sha256": (
            "63774e6fe7f31460fb03142820e80c7245a0d80d87402ec6dc5bc38d897311ca"
        ),
    }


def test_native_prefix_conversion_matches_s1_training_semantics() -> None:
    """Preserve history, mask old thinking, and fold harness context."""

    questions, _ = _smoke_panel()
    contract = load_contract()
    answer_keys = list(contract["answer"]["keys"])
    multiturn = next(
        row
        for row in questions
        if sum(message["role"] == "user" for message in row["messages"]) == 2
    )
    prior_answer = next(
        message
        for message in multiturn["messages"]
        if message["role"] == "assistant"
    )
    assert list(prior_answer["content"]) == answer_keys

    rendered = model_messages(multiturn)
    assert [message["role"] for message in rendered] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "thinking" not in rendered[2]
    parsed_pairs = json.loads(
        rendered[2]["content"],
        object_pairs_hook=lambda pairs: pairs,
    )
    assert [key for key, _ in parsed_pairs] == answer_keys

    harness = next(
        row
        for row in questions
        if any(message["role"] == "harness" for message in row["messages"])
    )
    harness_rendered = model_messages(harness)
    assert "harness" not in [message["role"] for message in harness_rendered]
    assert harness_rendered[1]["content"].startswith(
        "Context (from the harness):\n"
    )
    assert "\n\nUser request:\n" in harness_rendered[1]["content"]


def test_failure_injection_obeys_first_and_membership_stages() -> None:
    """Select injected failures only from public metadata and the contract."""

    questions, _ = _smoke_panel()
    contract = load_contract()
    direct = next(
        row for row in questions if row["metadata"]["source_trace"] == "S1.1.8"
    )
    membership = next(
        row for row in questions if row["metadata"]["source_trace"] == "S1.3.10"
    )

    assert failure_injection_required(
        direct,
        tool_name="lookup_human_gene_symbol",
        call_index=0,
        contract=contract,
    )
    assert not failure_injection_required(
        direct,
        tool_name="check_multiplex_membership",
        call_index=1,
        contract=contract,
    )
    assert not failure_injection_required(
        membership,
        tool_name="lookup_human_gene_symbol",
        call_index=0,
        contract=contract,
    )
    assert failure_injection_required(
        membership,
        tool_name="check_multiplex_membership",
        call_index=1,
        contract=contract,
    )
    visible = json.dumps(model_messages(membership), sort_keys=True)
    assert "tool_error" not in visible
    assert "blocked_reason_code" not in visible


def test_full_test_config_pins_all_s1_transcript_classes() -> None:
    """Validate the full panel without requiring an unbuilt 120B checkpoint."""

    config = read_json(TEST_CONFIG_PATH)
    training = read_json(TRAIN_CONFIG_PATH)
    assert sha256_file(TRAIN_CONFIG_PATH) == config["training"]["config_sha256"]
    assert training["run_scope"] == "production"
    assert training["promotion_eligible"] is True
    assert training["production_matrix_cell"] is True
    # This gate prohibits test access during training. The eval validator later
    # requires a completed exposure receipt before it opens the test panel.
    assert training["validation_gate"]["test_panel_access"] is False
    assert _validate_promotion_gate(config["gate"]) == {
        name: 0.99 for name in PROMOTION_RATE_GATE_NAMES
    } | PROMOTION_MAXIMUM_RATE_LIMITS
    corpus, _, corpus_sha256, _, _ = _validate_corpus(
        REPO_ROOT,
        config["corpus"],
    )
    assert corpus["row_counts"]["train"] == 55276
    panel = _validate_panel(
        repo_root=REPO_ROOT,
        panel_config=config["test_panel"],
        corpus_manifest_sha256=corpus_sha256,
        contract=load_contract(),
    )
    assert panel[-1] == 6875

    contract = load_contract()
    questions = [
        validate_question(row, contract)
        for row in read_jsonl(FULL_ROOT / "test_questions.jsonl")
    ]
    by_id = {row["record_id"]: row for row in questions}
    answers = [
        validate_answer_row(row, by_id[row["record_id"]], contract)
        for row in read_jsonl(FULL_ROOT / "test_answer_key.jsonl")
    ]
    shapes = Counter(tuple(message_shape(row["expected_messages"])) for row in answers)
    assert set(shapes) == EXPECTED_TRANSCRIPT_SHAPES
    assert shapes[("assistant",)] == 3500
    assert shapes[("assistant_tool_call", "tool", "assistant")] == 2750
    assert shapes[
        (
            "assistant_tool_call",
            "tool",
            "assistant_tool_call",
            "tool",
            "assistant",
        )
    ] == 625


@pytest.mark.parametrize("gate_name", PROMOTION_RATE_GATE_NAMES)
@pytest.mark.parametrize("below_floor", [0.989, 0.0])
def test_promotion_gate_rejects_every_floor_below_point_99(
    gate_name: str,
    below_floor: float,
) -> None:
    """Every numeric promotion floor is immutable at 0.99 or stricter."""

    gate = {
        name: 0.99 for name in PROMOTION_RATE_GATE_NAMES
    } | {name: True for name in REQUIRED_BOOLEAN_GATE_NAMES}
    gate[gate_name] = below_floor

    with pytest.raises(
        S1EvaluationError,
        match=rf"^{gate_name} must be at least 0\.99$",
    ):
        _validate_promotion_gate(gate)


@pytest.mark.parametrize("above_ceiling", [0.011, 1.0])
def test_promotion_gate_rejects_unexpected_direct_answer_ceiling_drift(
    above_ceiling: float,
) -> None:
    """Unexpected direct answers may never exceed the stage-spec ceiling."""

    gate = {
        name: 0.99 for name in PROMOTION_RATE_GATE_NAMES
    } | {
        "maximum_unexpected_direct_answer_rate": above_ceiling,
    } | {name: True for name in REQUIRED_BOOLEAN_GATE_NAMES}

    with pytest.raises(
        S1EvaluationError,
        match=(
            r"^maximum_unexpected_direct_answer_rate "
            r"must be at most 0\.01$"
        ),
    ):
        _validate_promotion_gate(gate)


@pytest.mark.parametrize("gate_name", REQUIRED_BOOLEAN_GATE_NAMES)
def test_promotion_gate_requires_every_hard_consistency_check(
    gate_name: str,
) -> None:
    """No config may disable an unconditional S1 consistency gate."""

    gate = {
        name: 0.99 for name in PROMOTION_RATE_GATE_NAMES
    } | PROMOTION_MAXIMUM_RATE_LIMITS | {
        name: True for name in REQUIRED_BOOLEAN_GATE_NAMES
    }
    gate[gate_name] = False

    with pytest.raises(
        S1EvaluationError,
        match=rf"^the S1 gate {gate_name} must be true$",
    ):
        _validate_promotion_gate(gate)


def test_held_out_test_link_rejects_qualification_training() -> None:
    """Never open the held-out test panel for a qualification checkpoint."""

    qualification_path = REPO_ROOT / "config/world_model_s1_20b_qualification_v1.json"
    qualification = read_json(qualification_path)
    method = qualification["methods"][0]
    link = {
        "config": qualification_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256_file(qualification_path),
        "training_id": qualification.get(
            "training_id", qualification["qualification_id"]
        ),
        "checkpoint_root": method["output_root"],
    }

    with pytest.raises(
        S1EvaluationError,
        match="promotion-eligible production run",
    ):
        _validate_training_link(
            repo_root=REPO_ROOT,
            training=link,
            config={
                "model": qualification["model"],
                "corpus": qualification["corpus"],
            },
            method=method,
            method_id=method["method_id"],
        )


def test_tokenizer_contract_detects_chat_template_tamper(tmp_path: Path) -> None:
    """Bind chat template and config files, not only tokenizer.json."""

    source_root = (
        REPO_ROOT
        / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/"
        "plain_base_tokenizer"
    )
    arm_root = tmp_path / "repo/arm"
    shutil.copytree(source_root, arm_root)
    tokenizer_root = arm_root / "tokenizer"
    method = {
        "tokenizer_path": "arm/tokenizer",
        "tokenizer_manifest": "arm/tokenizer_manifest.json",
        "tokenizer_manifest_sha256": (
            "245163b5189a64afb8e5b061585f90d4c15bcf94599b68307e9ff008880fe3c0"
        ),
        "tokenizer_artifacts_sha256": (
            "be21ce55523f893dc49e5ec7433f627372eebc1b9ac39dcdca5af8c7561e8d2f"
        ),
    }
    training_method = {
        "arm_root": "arm",
        "arm_manifest_sha256": sha256_file(arm_root / "manifest.json"),
    }
    _, _, _, artifact_identity, _ = _validate_tokenizer(
        repo_root=tmp_path / "repo",
        method=method,
        training_method=training_method,
    )
    assert artifact_identity == method["tokenizer_artifacts_sha256"]

    with (tokenizer_root / "chat_template.jinja").open(
        "a", encoding="utf-8"
    ) as destination:
        destination.write("\n{# tamper #}\n")
    with pytest.raises(S1EvaluationError, match="tokenizer contract changed"):
        _validate_tokenizer(
            repo_root=tmp_path / "repo",
            method=method,
            training_method=training_method,
        )


def test_validator_accepts_complete_checkpoint_and_rejects_identity_drift(
    tmp_path: Path,
) -> None:
    """Exercise the complete post-training test-access contract."""

    repo = tmp_path / "repo"
    contract_path = repo / "runtime/world_model/contract.yaml"
    contract_path.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "runtime/world_model/contract.yaml", contract_path)
    for relative, source in (
        (
            "data/world_model_v3/corpora/s1_smoke",
            REPO_ROOT / "data/world_model_v3/corpora/s1_smoke",
        ),
        (
            "data/world_model_v3/eval/s1_smoke",
            REPO_ROOT / "data/world_model_v3/eval/s1_smoke",
        ),
    ):
        shutil.copytree(source, repo / relative)
    source_index = repo / SOURCE_INDEX.relative_to(REPO_ROOT)
    source_index.parent.mkdir(parents=True)
    shutil.copy2(SOURCE_INDEX, source_index)

    tokenizer_arm = (
        REPO_ROOT
        / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/"
        "plain_base_tokenizer"
    )
    shadow_arm = (
        repo
        / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/"
        "plain_base_tokenizer"
    )
    shutil.copytree(tokenizer_arm, shadow_arm)
    tokenizer_path = shadow_arm / "tokenizer"
    tokenizer_manifest_path = shadow_arm / "tokenizer_manifest.json"
    tokenizer_manifest_sha256 = read_json(tokenizer_manifest_path)[
        "manifest_sha256"
    ]
    tokenizer_artifacts_sha256 = (
        "be21ce55523f893dc49e5ec7433f627372eebc1b9ac39dcdca5af8c7561e8d2f"
    )

    model_path = tmp_path / "model"
    model_path.mkdir()
    for name, payload in (
        ("config.json", {"model_type": "test"}),
        ("model.safetensors.index.json", {"weight_map": {}}),
        ("generation_config.json", {"do_sample": False}),
    ):
        (model_path / name).write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    model = {
        "model_id": "mock-gpt-oss-120b",
        "path": "../model",
        "model_identity_sha256": _model_metadata_identity(model_path),
        "base_model_artifact_schema": "mentor-rl-base-model-artifact-v2",
        "base_model_artifact_sha256": "a" * 64,
        "weight_shard_count": 1,
    }

    corpus_root = repo / "data/world_model_v3/corpora/s1_smoke"
    corpus_manifest = read_json(corpus_root / "manifest.json")
    corpus = {
        "root": corpus_root.relative_to(repo).as_posix(),
        "manifest_sha256": sha256_file(corpus_root / "manifest.json"),
        "train_sha256": corpus_manifest["files"]["train.jsonl"]["sha256"],
        "train_rows": corpus_manifest["row_counts"]["train"],
    }
    method_id = "mock-oss120b-plain-base-tokenizer-lora-r32"
    tokenizer_relative = tokenizer_path.relative_to(repo).as_posix()
    tokenizer_manifest_relative = tokenizer_manifest_path.relative_to(repo).as_posix()
    checkpoint_root = repo / "checkpoints" / method_id
    checkpoint = checkpoint_root / "slurm-test"
    checkpoint.mkdir(parents=True)
    training_method = {
        "method_id": method_id,
        "arm_root": shadow_arm.relative_to(repo).as_posix(),
        "arm_manifest_sha256": sha256_file(shadow_arm / "manifest.json"),
        "tokenizer_path": tokenizer_relative,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "output_root": checkpoint_root.relative_to(repo).as_posix(),
    }
    training_config = {
        "schema_version": "test-only-s1-training",
        "training_id": "mock-s1-production",
        "run_scope": "production",
        "promotion_eligible": True,
        "production_matrix_cell": True,
        "model": model,
        "corpus": corpus,
        "methods": [training_method],
        "validation_gate": {
            "checkpoint_set": "final",
            "test_panel_access": False,
        },
    }
    training_path = repo / "config/training.json"
    training_path.parent.mkdir(parents=True)
    training_path.write_text(
        json.dumps(training_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_config_sha256 = sha256_file(training_path)

    evaluator_root = repo / "data/world_model_v3/eval/s1_smoke"
    evaluator = read_json(evaluator_root / "manifest.json")
    test_panel = evaluator["test"]
    eval_method = {
        **training_method,
        "tokenizer_method": "plain_base_tokenizer",
        "tokenizer_manifest": tokenizer_manifest_relative,
        "tokenizer_artifacts_sha256": tokenizer_artifacts_sha256,
        "checkpoint_kind": "lora",
        "fine_tune_configuration": "lora_r32",
        "output_root": "eval-output",
    }
    test_config = {
        "schema_version": "mentor-rl-world-model-s1-test-v1",
        "evaluation_id": "mock-s1-test",
        "evaluation_contract": "held_out_s1_multiplex_sft_v3",
        "training": {
            "config": training_path.relative_to(repo).as_posix(),
            "config_sha256": training_config_sha256,
            "training_id": training_config["training_id"],
            "checkpoint_root": checkpoint_root.relative_to(repo).as_posix(),
        },
        "model": model,
        "corpus": corpus,
        "test_panel": {
            "evaluator_root": evaluator_root.relative_to(repo).as_posix(),
            "evaluator_manifest": "manifest.json",
            "evaluator_manifest_sha256": sha256_file(
                evaluator_root / "manifest.json"
            ),
            "questions_file": "test_questions.jsonl",
            "questions_sha256": test_panel["questions_sha256"],
            "answer_key_file": "test_answer_key.jsonl",
            "answer_key_sha256": test_panel["answer_key_sha256"],
            "test_panel_id": test_panel["test_panel_id"],
            "row_count": test_panel["row_count"],
            "required_family_count": 8,
        },
        "source_index": {
            "path": source_index.relative_to(repo).as_posix(),
            "sha256": (
                "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
            ),
        },
        "generation": {
            "num_nodes": 1,
            "tasks_per_node": 1,
            "gpus_per_node": 8,
            "batch_size": 1,
            "max_tool_call_tokens": 1024,
            "max_final_prompt_tokens": 6144,
            "max_new_tokens": 4096,
            "max_total_tokens": 8192,
            "maximum_tool_calls": 2,
            "do_sample": False,
            "enable_thinking": True,
            "reasoning_effort": "low",
            "seed": 900913,
            "local_files_only": True,
        },
        "gate": {
            "minimum_valid_trajectory": 0.99,
            "minimum_reasoning_present": 0.99,
            "minimum_tool_followup_thinking_present": 0.99,
            "minimum_exact_tool_selection": 0.99,
            "minimum_exact_tool_arguments": 0.99,
            "minimum_tool_payload_consistency": 0.99,
            "minimum_final_answer_accuracy": 0.99,
            "minimum_source_trace_envelope_accuracy": 0.99,
            "minimum_family_accuracy": 0.99,
            "maximum_unexpected_direct_answer_rate": 0.01,
            "require_closed_book_zero_calls": True,
            "require_failure_harness_consistency": True,
            "require_network_used_false": True,
            "require_ambiguous_deferral_consistency": True,
            "require_blocked_reason_consistency": True,
            "require_evidence_consistency": True,
            "require_summary_consistency": True,
            "require_structured_final_json": True,
            "require_family_contract_consistency": True,
        },
        "tracking": {
            "entity": "jail-ai",
            "project": "mentor-sft",
            "resume_training_run": True,
        },
        "methods": [eval_method],
    }
    test_config_path = repo / "config/test.json"
    test_config_path.write_text(
        json.dumps(test_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_id = f"{method_id}-slurm-test"
    adapter_identity = {
        "run_id": run_id,
        "method_id": method_id,
        "model_identity_sha256": model["model_identity_sha256"],
        "run_config_sha256": training_config_sha256,
        "corpus_manifest_sha256": corpus["manifest_sha256"],
        "train_sha256": corpus["train_sha256"],
        "tokenizer_arm_manifest_sha256": training_method[
            "arm_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_artifacts_sha256": tokenizer_artifacts_sha256,
        "validation_sha256": corpus_manifest["files"]["validation.jsonl"][
            "sha256"
        ],
        "validation_answer_key_sha256": evaluator["validation"][
            "answer_key_sha256"
        ],
    }
    adapter = {
        "format": "mentor-rl-s0-tp-lora-v1",
        "identity": adapter_identity,
        "objective": {"loss_contract": "multiplex_sft_v3"},
    }
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"mock")
    (checkpoint / "tp_adapter_manifest.json").write_text(
        json.dumps(adapter, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    exposure = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": "complete",
        "method_id": method_id,
        "run_id": run_id,
        "identity": {
            "run_config_sha256": training_config_sha256,
            "corpus_manifest_sha256": corpus["manifest_sha256"],
            "train_sha256": corpus["train_sha256"],
            "tokenizer_arm_manifest_sha256": training_method[
                "arm_manifest_sha256"
            ],
        },
        "corpus": {"dataset_id": "world_model_v3_s1_coordinate_system"},
        "exposure_contract": {"satisfied": True},
    }
    exposure["manifest_sha256"] = stable_sha256(exposure)
    exposure_path = checkpoint / "run_contract/training_exposure.json"
    exposure_path.parent.mkdir()
    exposure_path.write_text(
        json.dumps(exposure, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    values = validate_and_export(
        repo_root=repo,
        config_path=test_config_path,
        method_id=method_id,
        checkpoint_path=checkpoint,
    )
    assert values["S1_TEST_ROWS"] == "111"
    assert values["S1_TRAIN_RUN_ID"] == run_id
    assert len(values["S1_CHECKPOINT_ARTIFACT_SHA256"]) == 64
    assert (
        values["S1_TOKENIZER_ARTIFACTS_SHA256"]
        == tokenizer_artifacts_sha256
    )
    assert values["S1_MINIMUM_SOURCE_TRACE_ENVELOPE_ACCURACY"] == "0.99"
    assert values["S1_MINIMUM_TOOL_FOLLOWUP_THINKING_PRESENT"] == "0.99"
    assert values["S1_MAXIMUM_UNEXPECTED_DIRECT_ANSWER_RATE"] == "0.01"
    assert values["S1_HISTORY_POLICY"] == "retain_current_turn"

    # Both modes must reach the launcher explicitly; reject an unknown policy
    # before any GPU allocation. Historical configs use the new default.
    for policy in ("strip", "retain_current_turn", "retain_everything"):
        test_config["generation"]["history_policy"] = policy
        test_config_path.write_text(json.dumps(test_config) + "\n", encoding="utf-8")
        if policy == "retain_everything":
            with pytest.raises(S1EvaluationError, match="history policy is invalid"):
                validate_and_export(
                    repo_root=repo, config_path=test_config_path,
                    method_id=method_id, checkpoint_path=checkpoint,
                )
        else:
            explicit = validate_and_export(
                repo_root=repo, config_path=test_config_path,
                method_id=method_id, checkpoint_path=checkpoint,
            )
            assert explicit["S1_HISTORY_POLICY"] == policy
    test_config["generation"]["history_policy"] = "retain_current_turn"
    test_config_path.write_text(json.dumps(test_config) + "\n", encoding="utf-8")

    adapter["identity"]["tokenizer_artifacts_sha256"] = "0" * 64
    (checkpoint / "tp_adapter_manifest.json").write_text(
        json.dumps(adapter, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(S1EvaluationError, match="checkpoint contract changed"):
        validate_and_export(
            repo_root=repo,
            config_path=test_config_path,
            method_id=method_id,
            checkpoint_path=checkpoint,
        )


def test_v8_promotion_config_exports_the_frozen_panel(monkeypatch) -> None:
    """Validate real v8 inputs while explicitly stubbing only its future checkpoint."""
    from scripts import validate_world_model_s1_eval_contract as validator

    config_path = REPO_ROOT / "config/world_model_s1_120b_32n_test_v8_retained_512n.json"
    config = json.loads(config_path.read_text())
    training = json.loads((REPO_ROOT / config["training"]["config"]).read_text())
    checked = []

    def future_checkpoint(**kwargs):
        checked.append(kwargs)
        return "explicit-future-v8-checkpoint-stub", "1" * 64

    resolve = validator.resolve_repo_path

    def future_path(repo_root, value, label, **kwargs):
        if label in {"checkpoint root", "training output root"}:
            kwargs["must_exist"] = False
        return resolve(repo_root, value, label, **kwargs)

    monkeypatch.setattr(validator, "_validate_checkpoint", future_checkpoint)
    monkeypatch.setattr(validator, "resolve_repo_path", future_path)
    values = validator.validate_and_export(
        repo_root=REPO_ROOT, config_path=config_path,
        method_id="oss120b-plain-base-tokenizer-lora-r32",
        checkpoint_path=REPO_ROOT / config["training"]["checkpoint_root"] / "explicit-future-v8-checkpoint-stub",
    )
    assert values["S1_DIAGNOSTIC_ONLY"] == "0"
    assert values["S1_HISTORY_POLICY"] == "retain_current_turn"
    assert values["S1_GATE_PROFILE"] == "tiered_v3"
    assert values["S1_TEST_ROWS"] == "6875"
    assert values["S1_EVAL_NUM_NODES"] == "512"
    assert values["S1_TEST_PANEL_ID"] == "eff174050b8d82ee2702cad2e4efebb8fa0938785e2ec6266640c23363ee7904"
    assert len(checked) == 1
    assert checked[0]["corpus_manifest_sha256"] == training["corpus"]["manifest_sha256"]
    assert checked[0]["train_sha256"] == training["corpus"]["train_sha256"]
    assert checked[0]["training_config_sha256"] == config["training"]["config_sha256"]


def test_launcher_contract_only_stops_before_slurm_work(tmp_path: Path) -> None:
    """Exercise the dedicated eval launcher's no-allocation contract path."""

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        f"S1_EVAL_CONFIG={TEST_CONFIG_PATH}\n"
        "S1_EVAL_CONFIG_SHA256=placeholder\n"
        "S1_EVALUATION_ID=s1-test\n"
        "S1_EVALUATION_CONTRACT=held_out_s1_multiplex_sft_v3\n"
        "S1_DIAGNOSTIC_ONLY=0\n"
        f"S1_METHOD_ID={METHOD_ID}\n"
        "CHECKPOINT_PATH=/tmp/completed-s1-checkpoint\n"
        "S1_TEST_ROWS=6875\n"
        "S1_HISTORY_POLICY=retain_current_turn\n"
        f"S1_EVAL_OUTPUT_ROOT={tmp_path / 'eval-output'}\n"
        "S1_EVAL_NUM_NODES=8\n"
        "S1_EVAL_TASKS_PER_NODE=1\n"
        "S1_EVAL_WORLD_SIZE=8\n"
        "S1_EVAL_GPUS_PER_NODE=8\n"
        "EOF\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_")
    }
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CONTRACT_ONLY": "1",
            "RUN_CONFIG": str(TEST_CONFIG_PATH),
            "S1_METHOD_ID": METHOD_ID,
            "CHECKPOINT_PATH": "/tmp/completed-s1-checkpoint",
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "world_model_s1_eval.slurm")],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "nodes=8 tasks=8" in result.stdout
    assert "Reasoning history: retain_current_turn" in result.stdout
    assert "config, panel, and checkpoint are valid" in result.stdout
    assert not (tmp_path / "eval-output").exists()


def test_launcher_reports_promotion_gate_separately_from_artifact_success() -> None:
    """A valid failed evaluation remains complete but cannot look promoted."""

    launcher = (REPO_ROOT / "world_model_s1_eval.slurm").read_text(
        encoding="utf-8"
    )
    assert 'print("PASS" if passed else "FAIL")' in launcher
    assert 'echo "S1 promotion gate: ${promotion_gate}"' in launcher
    assert "evaluation artifacts are complete" in launcher


def test_launcher_passes_every_publication_identity() -> None:
    """Bind published metrics to the validated launch and generated bundle."""

    launcher = (REPO_ROOT / "world_model_s1_eval.slurm").read_text(
        encoding="utf-8"
    )
    assert '--base-model-artifact-sha256 "${BASE_MODEL_ARTIFACT_SHA256}"' in launcher
    assert '--history-policy "${S1_HISTORY_POLICY}"' in launcher
    assert '"history_policy=${S1_HISTORY_POLICY}"' in launcher
    for flag in (
        "--expected-train-run-id",
        "--expected-method-id",
        "--expected-checkpoint-identity-sha256",
        "--expected-base-model-identity-sha256",
        "--expected-base-model-artifact-sha256",
        "--expected-tokenizer-manifest-sha256",
        "--expected-tokenizer-artifacts-sha256",
        "--expected-source-index-sha256",
        "--expected-generation-bundle-sha256",
    ):
        assert flag in launcher
    assert "--minimum-source-trace-envelope-accuracy" in launcher
    assert "--minimum-tool-followup-thinking-present" in launcher
    assert "--maximum-unexpected-direct-answer-rate" in launcher
    assert "exact_s1_report.json" in launcher


V2_TEST_CONFIGS = [
    (
        REPO_ROOT / "config/world_model_s1_120b_test_v2_512n.json",
        REPO_ROOT
        / "data/world_model_v4/checkpoints/s1_120b"
        / "oss120b-plain-base-tokenizer-lora-r32/slurm-5638673",
    ),
    (
        REPO_ROOT / "config/world_model_s1_120b_184n_test_v2_512n.json",
        REPO_ROOT
        / "data/world_model_v4/checkpoints/s1_120b_184n"
        / "oss120b-plain-base-tokenizer-lora-r32/slurm-5640408",
    ),
]


def test_tiered_gate_returns_the_strict_and_target_floors() -> None:
    config = read_json(V2_TEST_CONFIGS[0][0])

    values = _validate_tiered_gate(config["gate"])

    assert values["gate_profile"] == "tiered_v2"
    assert values["minimum_strict_rate"] == 0.995
    assert values["minimum_target_rate"] == 0.9
    assert values["minimum_final_answer_accuracy"] == 0.0


def test_tiered_gate_rejects_a_changed_key_set() -> None:
    gate = dict(read_json(V2_TEST_CONFIGS[0][0])["gate"])
    gate["require_summary_consistency"] = True

    with pytest.raises(S1EvaluationError, match="tiered S1 gate keys"):
        _validate_tiered_gate(gate)


def test_tiered_v3_gate_names_only_its_profile() -> None:
    gate = {"profile": "tiered_v3", "require_network_used_false": True}

    values = _validate_tiered_v3_gate(gate)

    assert values["gate_profile"] == "tiered_v3"
    assert values["minimum_final_answer_accuracy"] == 0.0
    with pytest.raises(S1EvaluationError, match="tiered_v3 S1 gate keys"):
        _validate_tiered_v3_gate({**gate, "minimum_strict_rate": 0.9})


V3_32_NODE_CHECKPOINT = (
    REPO_ROOT
    / "data/world_model_v4/checkpoints/s1_v3_120b_32n"
    / "oss120b-plain-base-tokenizer-lora-r32/slurm-5652138"
)


@pytest.mark.skipif(
    not V3_32_NODE_CHECKPOINT.is_dir(), reason="the S1 v3 32-node checkpoint is absent"
)
def test_tiered_v3_test_config_exports_its_profile() -> None:
    values = validate_and_export(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / "config/world_model_s1_120b_32n_test_v3_512n.json",
        method_id="oss120b-plain-base-tokenizer-lora-r32",
        checkpoint_path=V3_32_NODE_CHECKPOINT,
    )

    assert values["S1_GATE_PROFILE"] == "tiered_v3"
    assert values["S1_MINIMUM_STRICT_RATE"] == "0.995"


V4_32_NODE_CHECKPOINT = (
    REPO_ROOT
    / "data/world_model_v4/checkpoints/s1_v4_120b_32n"
    / "oss120b-plain-base-tokenizer-lora-r32/slurm-5656641"
)


@pytest.mark.skipif(
    not (V4_32_NODE_CHECKPOINT / "training_state.json").is_file(),
    reason="the S1 v4 32-node checkpoint is absent",
)
def test_v4_test_config_keeps_the_s1_test_panel() -> None:
    values = validate_and_export(
        repo_root=REPO_ROOT,
        config_path=REPO_ROOT / "config/world_model_s1_120b_32n_test_v4_512n.json",
        method_id="oss120b-plain-base-tokenizer-lora-r32",
        checkpoint_path=V4_32_NODE_CHECKPOINT,
    )

    assert values["S1_GATE_PROFILE"] == "tiered_v3"
    assert values["S1_TEST_PANEL_ID"].startswith("eff17405")


@pytest.mark.parametrize(("config", "checkpoint"), V2_TEST_CONFIGS)
def test_v2_test_configs_validate_their_s1_checkpoints(
    config: Path, checkpoint: Path
) -> None:
    if not checkpoint.is_dir():
        pytest.skip("the S1 v2 checkpoint is absent")

    values = validate_and_export(
        repo_root=REPO_ROOT,
        config_path=config,
        method_id="oss120b-plain-base-tokenizer-lora-r32",
        checkpoint_path=checkpoint,
    )

    assert values["S1_GATE_PROFILE"] == "tiered_v2"
    assert values["S1_EVAL_NUM_NODES"] == "512"
    assert values["S1_TEST_ROWS"] == "6875"
