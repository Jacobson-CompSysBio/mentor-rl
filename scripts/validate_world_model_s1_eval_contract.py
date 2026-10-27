#!/usr/bin/env python3
"""Validate and export one exact native S1 evaluation contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import (  # noqa: E402
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from runtime.world_model.s1_overlay import (  # noqa: E402
    S1_OVERLAY_V3_SCHEMA_VERSION,
    S1_OVERLAY_V4_SCHEMA_VERSION,
    S1_OVERLAY_V5_SCHEMA_VERSION,
    S1_OVERLAY_V7_SCHEMA_VERSION,
    S1_OVERLAY_V8_SCHEMA_VERSION,
    S1_OVERLAY_V9_SCHEMA_VERSION,
)
from runtime.world_model.source_index import (  # noqa: E402
    check_source_index,
    content_sha256,
)
from runtime.world_model_training import (  # noqa: E402
    tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.world_model_s1_eval_common import (  # noqa: E402
    DATASET_ID,
    EVALUATION_CONTRACT,
    EVALUATOR_SCHEMA_VERSION,
    LOSS_CONTRACT,
    S1EvaluationError,
    S1_FAMILIES,
    STRUCTURED_95_GATE_PROFILE,
    STRUCTURED_95_MINIMUM,
    TIERED_V3_GATE_PROFILE,
    TIERED_V3_MAXIMUM_DIRECT_ANSWER_RATE,
    TIERED_V3_TARGETS,
    message_shape,
    read_json,
    read_jsonl,
    require_sha256,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    validate_answer_row,
    validate_question,
)


TEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-test-v1"
# The v2 test reads the S1 v2 overlay corpus and gates with two tiers.
TIERED_TEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-test-v2"
RETRY_TEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-retry-test-v1"
ROBUSTNESS_TEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-robustness-test-v1"
TIERED_GATE_PROFILE = "tiered_v2"
S1_V2_OVERLAY_PATH = "runtime/world_model/s1_v2.yaml"
S1_OVERLAY_SCHEMA_VERSION = "mentor-rl-world-model-s1-overlay-v1"
# The canonical S1 overlays an evaluator may name, with their schemas.
S1_OVERLAY_SCHEMAS = {
    S1_V2_OVERLAY_PATH: S1_OVERLAY_SCHEMA_VERSION,
    "runtime/world_model/s1_v3.yaml": "mentor-rl-world-model-s1-overlay-v2",
    "runtime/world_model/s1_v4.yaml": S1_OVERLAY_V3_SCHEMA_VERSION,
    "runtime/world_model/s1_v5.yaml": S1_OVERLAY_V4_SCHEMA_VERSION,
    "runtime/world_model/s1_v6.yaml": S1_OVERLAY_V5_SCHEMA_VERSION,
    "runtime/world_model/s1_v8.yaml": S1_OVERLAY_V7_SCHEMA_VERSION,
    "runtime/world_model/s1_v9.yaml": S1_OVERLAY_V8_SCHEMA_VERSION,
    "runtime/world_model/s1_v10.yaml": S1_OVERLAY_V9_SCHEMA_VERSION,
}
TIERED_GATE_KEYS = (
    "profile",
    "minimum_strict_rate",
    "minimum_target_rate",
    "minimum_reasoning_present",
    "minimum_tool_followup_thinking_present",
    "maximum_unexpected_direct_answer_rate",
    "require_network_used_false",
)
# The tiered_v3 targets are fixed in world_model_s1_eval_common, so a test
# config names only the profile.
TIERED_V3_GATE_KEYS = ("profile", "require_network_used_false")
CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-v1"
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s0-training-exposure-v2"
EXPECTED_TRANSCRIPT_SHAPES = {
    ("assistant",),
    ("assistant_tool_call", "tool", "assistant"),
    (
        "assistant_tool_call",
        "tool",
        "assistant_tool_call",
        "tool",
        "assistant",
    ),
}
MINIMUM_PROMOTION_RATE = 0.99
PROMOTION_RATE_GATE_NAMES = (
    "minimum_valid_trajectory",
    "minimum_reasoning_present",
    "minimum_tool_followup_thinking_present",
    "minimum_exact_tool_selection",
    "minimum_exact_tool_arguments",
    "minimum_tool_payload_consistency",
    "minimum_final_answer_accuracy",
    "minimum_source_trace_envelope_accuracy",
    "minimum_family_accuracy",
)
PROMOTION_MAXIMUM_RATE_LIMITS = {
    "maximum_unexpected_direct_answer_rate": 0.01,
}
REQUIRED_BOOLEAN_GATE_NAMES = (
    "require_closed_book_zero_calls",
    "require_failure_harness_consistency",
    "require_network_used_false",
    "require_ambiguous_deferral_consistency",
    "require_blocked_reason_consistency",
    "require_evidence_consistency",
    "require_summary_consistency",
    "require_structured_final_json",
    "require_family_contract_consistency",
)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise S1EvaluationError(f"{label} must be nonempty text")
    return value


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise S1EvaluationError(f"{label} must be a positive integer")
    return value


def _require_rate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise S1EvaluationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise S1EvaluationError(f"{label} must be between zero and one")
    return result


def _validate_promotion_gate(gate: Mapping[str, Any]) -> dict[str, float]:
    """Validate immutable S1 promotion floors and hard boolean gates."""

    gate_values = {
        name: _require_rate(gate.get(name), name)
        for name in PROMOTION_RATE_GATE_NAMES
    }
    for name, value in gate_values.items():
        if value < MINIMUM_PROMOTION_RATE:
            raise S1EvaluationError(
                f"{name} must be at least {MINIMUM_PROMOTION_RATE}"
            )
    for name, maximum in PROMOTION_MAXIMUM_RATE_LIMITS.items():
        value = _require_rate(gate.get(name), name)
        if value > maximum:
            raise S1EvaluationError(f"{name} must be at most {maximum}")
        gate_values[name] = value
    for name in REQUIRED_BOOLEAN_GATE_NAMES:
        if gate.get(name) is not True:
            raise S1EvaluationError(f"the S1 gate {name} must be true")
    return gate_values


def _validate_tiered_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the two-tier S1 v2 gate and return evaluator arguments."""

    if set(gate) != set(TIERED_GATE_KEYS):
        raise S1EvaluationError("the tiered S1 gate keys changed")
    if gate.get("profile") != TIERED_GATE_PROFILE:
        raise S1EvaluationError("the tiered S1 gate profile changed")
    if gate.get("require_network_used_false") is not True:
        raise S1EvaluationError("the S1 test must disable network access")
    strict = _require_rate(gate.get("minimum_strict_rate"), "minimum_strict_rate")
    target = _require_rate(gate.get("minimum_target_rate"), "minimum_target_rate")
    reasoning = _require_rate(
        gate.get("minimum_reasoning_present"), "minimum_reasoning_present"
    )
    followup = _require_rate(
        gate.get("minimum_tool_followup_thinking_present"),
        "minimum_tool_followup_thinking_present",
    )
    direct = _require_rate(
        gate.get("maximum_unexpected_direct_answer_rate"),
        "maximum_unexpected_direct_answer_rate",
    )
    # The evaluator still takes the promotion floors. The tiered profile
    # records them but gates with the strict and target rates instead.
    return {
        "gate_profile": TIERED_GATE_PROFILE,
        "minimum_strict_rate": strict,
        "minimum_target_rate": target,
        "minimum_valid_trajectory": strict,
        "minimum_reasoning_present": reasoning,
        "minimum_tool_followup_thinking_present": followup,
        "minimum_exact_tool_selection": strict,
        "minimum_exact_tool_arguments": strict,
        "minimum_tool_payload_consistency": strict,
        "minimum_final_answer_accuracy": 0.0,
        "minimum_source_trace_envelope_accuracy": 0.0,
        "minimum_family_accuracy": 0.0,
        "maximum_unexpected_direct_answer_rate": direct,
    }


def _validate_tiered_v3_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the S1 v3 gate and return evaluator arguments.

    The evaluator still takes the promotion floors and records them; the
    tiered_v3 profile gates with its own fixed targets.
    """

    if set(gate) != set(TIERED_V3_GATE_KEYS):
        raise S1EvaluationError("the tiered_v3 S1 gate keys changed")
    if gate.get("require_network_used_false") is not True:
        raise S1EvaluationError("the S1 test must disable network access")
    protocol = TIERED_V3_TARGETS["protocol"]
    floor = TIERED_V3_TARGETS["format"]
    return {
        "gate_profile": TIERED_V3_GATE_PROFILE,
        "minimum_strict_rate": protocol,
        "minimum_target_rate": TIERED_V3_TARGETS["generalization"],
        "minimum_valid_trajectory": protocol,
        "minimum_reasoning_present": floor,
        "minimum_tool_followup_thinking_present": floor,
        "minimum_exact_tool_selection": TIERED_V3_TARGETS["decision"],
        "minimum_exact_tool_arguments": protocol,
        "minimum_tool_payload_consistency": protocol,
        "minimum_final_answer_accuracy": 0.0,
        "minimum_source_trace_envelope_accuracy": 0.0,
        "minimum_family_accuracy": 0.0,
        "maximum_unexpected_direct_answer_rate": (
            TIERED_V3_MAXIMUM_DIRECT_ANSWER_RATE
        ),
    }


def _validate_structured_95_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Select the fixed overall 95% gate without changing historical profiles."""
    if set(gate) != {"profile", "require_network_used_false"}:
        raise S1EvaluationError("the structured_95_v1 S1 gate keys changed")
    if gate.get("profile") != STRUCTURED_95_GATE_PROFILE:
        raise S1EvaluationError("the structured_95_v1 S1 gate profile changed")
    if gate.get("require_network_used_false") is not True:
        raise S1EvaluationError("the S1 test must disable network access")
    # The CLI accepts legacy floors; only the selected profile's checks apply.
    return {
        **{name: 0.0 for name in PROMOTION_RATE_GATE_NAMES},
        "maximum_unexpected_direct_answer_rate": 1.0,
        "gate_profile": STRUCTURED_95_GATE_PROFILE,
        "minimum_strict_rate": 1.0,
        "minimum_target_rate": STRUCTURED_95_MINIMUM,
    }


def _model_path(repo_root: Path, value: Any) -> Path:
    declared = Path(_require_string(value, "model path"))
    path = (declared if declared.is_absolute() else repo_root / declared).resolve()
    try:
        path.relative_to(repo_root.parent)
    except ValueError as error:
        raise S1EvaluationError(
            "the model path escapes the shared LLM project root"
        ) from error
    if not path.is_dir():
        raise S1EvaluationError(f"the model path is absent: {path}")
    return path


def _model_metadata_identity(model_path: Path) -> str:
    names = ["config.json", "model.safetensors.index.json"]
    if (model_path / "generation_config.json").is_file():
        names.append("generation_config.json")
    if any(not (model_path / name).is_file() for name in names):
        raise S1EvaluationError("the base model metadata is incomplete")
    return hashlib.sha256(
        "".join(
            f"{sha256_file(model_path / name)}  {name}\n" for name in names
        ).encode("utf-8")
    ).hexdigest()


def _internal_manifest_identity(payload: Mapping[str, Any], label: str) -> str:
    claimed = require_sha256(payload.get("manifest_sha256"), label)
    identity = {
        str(key): value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if stable_sha256(identity) != claimed:
        raise S1EvaluationError(f"{label} failed its internal identity")
    return claimed


def _selected_method(config: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise S1EvaluationError("the S1 test config has no methods")
    selected = [
        method
        for method in methods
        if isinstance(method, dict) and method.get("method_id") == method_id
    ]
    if len(selected) != 1:
        raise S1EvaluationError("METHOD_ID must select one S1 test method")
    return selected[0]


def _validate_training_link(
    *,
    repo_root: Path,
    training: Mapping[str, Any],
    config: Mapping[str, Any],
    method: Mapping[str, Any],
    method_id: str,
) -> tuple[dict[str, Any], Path, str]:
    training_config_path = resolve_repo_path(
        repo_root, training.get("config"), "S1 training config"
    )
    training_config_sha256 = require_sha256(
        training.get("config_sha256"), "S1 training config identity"
    )
    if sha256_file(training_config_path) != training_config_sha256:
        raise S1EvaluationError("the linked S1 training config changed")
    training_config = read_json(training_config_path)
    training_id = training_config.get(
        "training_id", training_config.get("qualification_id")
    )
    if training_id != training.get("training_id"):
        raise S1EvaluationError("the linked S1 training ID changed")
    if (
        training_config.get("run_scope") != "production"
        or training_config.get("promotion_eligible") is not True
        or training_config.get("production_matrix_cell") is not True
    ):
        raise S1EvaluationError(
            "held-out S1 test access requires a promotion-eligible production run"
        )
    validation_gate = training_config.get("validation_gate")
    if (
        not isinstance(validation_gate, Mapping)
        or validation_gate.get("checkpoint_set") != "final"
        or validation_gate.get("test_panel_access") is not False
    ):
        raise S1EvaluationError(
            "held-out S1 evaluation requires post-training final-checkpoint access"
        )
    matching = [
        item
        for item in training_config.get("methods", [])
        if isinstance(item, Mapping) and item.get("method_id") == method_id
    ]
    if len(matching) != 1:
        raise S1EvaluationError("the linked training config lacks this method")
    train_method = matching[0]
    if (
        training_config.get("model") != config.get("model")
        or training_config.get("corpus", {}).get("manifest_sha256")
        != config.get("corpus", {}).get("manifest_sha256")
        or training_config.get("corpus", {}).get("train_sha256")
        != config.get("corpus", {}).get("train_sha256")
        or train_method.get("tokenizer_manifest_sha256")
        != method.get("tokenizer_manifest_sha256")
        or train_method.get("tokenizer_path") != method.get("tokenizer_path")
    ):
        raise S1EvaluationError("the test and training artifact identities disagree")
    checkpoint_root = resolve_repo_path(
        repo_root, training.get("checkpoint_root"), "checkpoint root"
    )
    expected_root = resolve_repo_path(
        repo_root, train_method.get("output_root"), "training output root"
    )
    if checkpoint_root != expected_root:
        raise S1EvaluationError("the linked checkpoint root changed")
    return training_config, checkpoint_root, training_config_sha256


def _validate_corpus(
    repo_root: Path,
    corpus: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str, str, int]:
    corpus_root = resolve_repo_path(repo_root, corpus.get("root"), "S1 corpus root")
    manifest_path = corpus_root / "manifest.json"
    manifest_sha256 = require_sha256(
        corpus.get("manifest_sha256"), "S1 corpus manifest"
    )
    if sha256_file(manifest_path) != manifest_sha256:
        raise S1EvaluationError("the S1 corpus manifest identity changed")
    manifest = read_json(manifest_path)
    train_sha256 = require_sha256(corpus.get("train_sha256"), "S1 train file")
    train_rows = _require_int(corpus.get("train_rows"), "S1 train rows")
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or manifest.get("loss_contract") != LOSS_CONTRACT
        or manifest.get("row_counts", {}).get("train") != train_rows
        or manifest.get("files", {}).get("train.jsonl", {}).get("sha256")
        != train_sha256
        or manifest.get("files", {}).get("train.jsonl", {}).get("row_count")
        != train_rows
    ):
        raise S1EvaluationError("the S1 corpus contract changed")
    if sha256_file(corpus_root / "train.jsonl") != train_sha256:
        raise S1EvaluationError("the S1 train file identity changed")
    return manifest, manifest_path, manifest_sha256, train_sha256, train_rows


def _validate_tokenizer(
    *,
    repo_root: Path,
    method: Mapping[str, Any],
    training_method: Mapping[str, Any],
) -> tuple[Path, Path, str, str, str]:
    """Bind every load-bearing tokenizer artifact to the eval contract."""

    tokenizer_path = resolve_repo_path(
        repo_root, method.get("tokenizer_path"), "tokenizer path"
    )
    tokenizer_manifest_path = resolve_repo_path(
        repo_root, method.get("tokenizer_manifest"), "tokenizer manifest"
    )
    tokenizer_manifest_sha256 = require_sha256(
        method.get("tokenizer_manifest_sha256"), "tokenizer manifest identity"
    )
    tokenizer_artifacts_sha256 = require_sha256(
        method.get("tokenizer_artifacts_sha256"),
        "tokenizer artifact identity",
    )
    arm_root = resolve_repo_path(
        repo_root, training_method.get("arm_root"), "tokenizer arm root"
    )
    arm_manifest_path = arm_root / "manifest.json"
    arm_manifest_sha256 = require_sha256(
        training_method.get("arm_manifest_sha256"),
        "tokenizer arm manifest identity",
    )
    if not arm_manifest_path.is_file():
        raise S1EvaluationError("the linked tokenizer arm manifest is absent")
    if (
        tokenizer_path != arm_root / "tokenizer"
        or tokenizer_manifest_path != arm_root / "tokenizer_manifest.json"
        or sha256_file(arm_manifest_path) != arm_manifest_sha256
    ):
        raise S1EvaluationError("the linked tokenizer arm identity changed")
    try:
        arm_manifest = read_json(arm_manifest_path)
        tokenizer_manifest = validated_tokenizer_manifest(tokenizer_manifest_path)
        tokenizer_files = tokenizer_artifact_hashes(tokenizer_path)
    except (OSError, TypeError, ValueError) as error:
        raise S1EvaluationError(f"the S1 tokenizer is invalid: {error}") from error
    declared_artifacts = arm_manifest.get("tokenizer_artifact_hashes")
    file_hashes = arm_manifest.get("file_hashes")
    expected_artifacts = (
        {
            Path(str(name)).name: require_sha256(
                value, f"tokenizer arm artifact {name}"
            )
            for name, value in declared_artifacts.items()
        }
        if isinstance(declared_artifacts, Mapping)
        else None
    )
    if (
        arm_manifest.get("schema_version")
        != "mentor-rl-world-model-s0-tokenizer-arm-v4"
        or arm_manifest.get("method") != "plain_base_tokenizer"
        or arm_manifest.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or not isinstance(file_hashes, Mapping)
        or not isinstance(declared_artifacts, Mapping)
        or any(
            file_hashes.get(name) != value
            for name, value in declared_artifacts.items()
        )
        or file_hashes.get("tokenizer_manifest.json")
        != sha256_file(tokenizer_manifest_path)
        or tokenizer_manifest.get("manifest_sha256")
        != tokenizer_manifest_sha256
        or tokenizer_manifest.get("method") != "plain_base_tokenizer"
        or tokenizer_files != expected_artifacts
        or stable_sha256(tokenizer_files) != tokenizer_artifacts_sha256
    ):
        raise S1EvaluationError("the S1 tokenizer contract changed")
    return (
        tokenizer_path,
        tokenizer_manifest_path,
        tokenizer_manifest_sha256,
        tokenizer_artifacts_sha256,
        arm_manifest_sha256,
    )


def _validate_panel(
    *,
    repo_root: Path,
    panel_config: Mapping[str, Any],
    corpus_manifest_sha256: str,
    contract: Mapping[str, Any],
    allow_s1_overlay: bool = False,
) -> tuple[dict[str, Any], Path, str, Path, str, Path, str, int]:
    evaluator_root = resolve_repo_path(
        repo_root, panel_config.get("evaluator_root"), "S1 evaluator root"
    )
    evaluator_path = evaluator_root / _require_string(
        panel_config.get("evaluator_manifest"), "evaluator manifest name"
    )
    evaluator_sha256 = require_sha256(
        panel_config.get("evaluator_manifest_sha256"), "evaluator manifest"
    )
    if sha256_file(evaluator_path) != evaluator_sha256:
        raise S1EvaluationError("the S1 evaluator manifest identity changed")
    evaluator = read_json(evaluator_path)
    test = evaluator.get("test")
    validation = evaluator.get("validation")
    expected_contract_pin = {
        "path": "runtime/world_model/contract.yaml",
        "sha256": sha256_file(repo_root / "runtime/world_model/contract.yaml"),
        "row_schema_version": contract["runtime"]["row_schema_version"],
        "system_prompt_sha256": system_prompt_sha256(contract),
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
        "tool_definitions_sha256": stable_sha256(tool_definitions(contract)),
    }
    if allow_s1_overlay:
        # Pin the overlay the evaluator names when it is a canonical one; any
        # other path keeps the v2 pin, so the comparison below fails.
        declared = evaluator.get("contract")
        declared = declared.get("s1_overlay") if isinstance(declared, Mapping) else None
        overlay_path = (
            declared.get("path")
            if isinstance(declared, Mapping)
            and declared.get("path") in S1_OVERLAY_SCHEMAS
            else S1_V2_OVERLAY_PATH
        )
        expected_contract_pin["s1_overlay"] = {
            "path": overlay_path,
            "sha256": sha256_file(repo_root / overlay_path),
            "schema_version": S1_OVERLAY_SCHEMAS[overlay_path],
        }
    if (
        evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != DATASET_ID
        or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
        or evaluator.get("loss_contract") != LOSS_CONTRACT
        or evaluator.get("corpus_manifest_sha256") != corpus_manifest_sha256
        or evaluator.get("contract") != expected_contract_pin
        or not isinstance(test, Mapping)
        or not isinstance(validation, Mapping)
    ):
        raise S1EvaluationError("the S1 evaluator contract changed")

    questions_path = evaluator_root / _require_string(
        panel_config.get("questions_file"), "test question file"
    )
    answers_path = evaluator_root / _require_string(
        panel_config.get("answer_key_file"), "test answer-key file"
    )
    questions_sha256 = require_sha256(
        panel_config.get("questions_sha256"), "test questions"
    )
    answers_sha256 = require_sha256(
        panel_config.get("answer_key_sha256"), "test answer key"
    )
    if sha256_file(questions_path) != questions_sha256:
        raise S1EvaluationError("the S1 test question identity changed")
    if sha256_file(answers_path) != answers_sha256:
        raise S1EvaluationError("the S1 answer-key identity changed")
    row_count = _require_int(panel_config.get("row_count"), "S1 test rows")
    if (
        test.get("questions_path")
        != questions_path.relative_to(repo_root).as_posix()
        or test.get("answer_key_path") != answers_path.relative_to(repo_root).as_posix()
        or test.get("questions_sha256") != questions_sha256
        or test.get("answer_key_sha256") != answers_sha256
        or test.get("test_panel_id") != panel_config.get("test_panel_id")
        or test.get("test_panel_id") != questions_sha256
        or test.get("row_count") != row_count
    ):
        raise S1EvaluationError("the fixed S1 test panel changed")

    questions = [validate_question(row, contract) for row in read_jsonl(questions_path)]
    raw_answers = read_jsonl(answers_path)
    if len(questions) != row_count or len(raw_answers) != row_count:
        raise S1EvaluationError("the S1 panel row count changed")
    by_id = {str(row["record_id"]): row for row in questions}
    if len(by_id) != row_count:
        raise S1EvaluationError("the S1 question record IDs are not unique")
    answers: list[dict[str, Any]] = []
    for row in raw_answers:
        record_id = row.get("record_id")
        question = by_id.get(str(record_id))
        if question is None:
            raise S1EvaluationError("the S1 answer record set changed")
        answers.append(validate_answer_row(row, question, contract))
    if {str(row["record_id"]) for row in answers} != set(by_id):
        raise S1EvaluationError("the S1 answer record set changed")

    family_counts = Counter(
        str(row["metadata"]["question_family"]) for row in questions
    )
    required_family_count = _require_int(
        panel_config.get("required_family_count"), "required S1 family count"
    )
    if (
        set(family_counts) != set(S1_FAMILIES)
        or len(family_counts) != required_family_count
        or dict(sorted(family_counts.items()))
        != dict(sorted(test.get("family_counts", {}).items()))
    ):
        raise S1EvaluationError("the S1 test family coverage changed")
    shapes = {tuple(message_shape(row["expected_messages"])) for row in answers}
    if shapes != EXPECTED_TRANSCRIPT_SHAPES:
        raise S1EvaluationError("the S1 0/1/2-call transcript coverage changed")
    if {row["metadata"]["book_mode"] for row in questions} != {
        "closed_book",
        "tool_call",
    }:
        raise S1EvaluationError("the S1 book-mode coverage changed")
    if not any(len(row["messages"]) > 2 for row in questions):
        raise S1EvaluationError("the S1 multi-turn coverage is absent")
    if not any(
        row["metadata"].get("blocked_reason_code") == "tool_error"
        for row in questions
    ):
        raise S1EvaluationError("the S1 transport-failure coverage is absent")
    return (
        evaluator,
        evaluator_path,
        evaluator_sha256,
        questions_path,
        questions_sha256,
        answers_path,
        answers_sha256,
        row_count,
    )


def _validate_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_root: Path,
    method_id: str,
    model_identity_sha256: str,
    training_config_sha256: str,
    corpus_manifest_sha256: str,
    train_sha256: str,
    tokenizer_arm_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    tokenizer_artifacts_sha256: str,
    validation_sha256: str,
    validation_answer_key_sha256: str,
) -> tuple[str, str]:
    if not checkpoint_path.is_dir():
        raise S1EvaluationError("the completed checkpoint directory is absent")
    try:
        checkpoint_path.relative_to(checkpoint_root)
    except ValueError as error:
        raise S1EvaluationError(
            "the checkpoint is outside the linked S1 training output"
        ) from error
    required_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
        "run_contract/training_exposure.json",
    )
    for name in required_files:
        if not (checkpoint_path / name).is_file():
            raise S1EvaluationError(f"the checkpoint lacks {name}")
    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if (
        adapter.get("format") != CHECKPOINT_FORMAT
        or adapter.get("objective", {}).get("loss_contract") != LOSS_CONTRACT
        or not isinstance(identity, Mapping)
        or identity.get("method_id") != method_id
        or identity.get("model_identity_sha256") != model_identity_sha256
        or identity.get("run_config_sha256") != training_config_sha256
        or identity.get("corpus_manifest_sha256") != corpus_manifest_sha256
        or identity.get("train_sha256") != train_sha256
        or identity.get("tokenizer_arm_manifest_sha256")
        != tokenizer_arm_manifest_sha256
        or identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or identity.get("tokenizer_artifacts_sha256")
        != tokenizer_artifacts_sha256
        or identity.get("validation_sha256") != validation_sha256
        or identity.get("validation_answer_key_sha256")
        != validation_answer_key_sha256
    ):
        raise S1EvaluationError("the selected S1 checkpoint contract changed")
    exposure = read_json(
        checkpoint_path / "run_contract/training_exposure.json"
    )
    _internal_manifest_identity(exposure, "training exposure receipt")
    if (
        exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION
        or exposure.get("status") != "complete"
        or exposure.get("method_id") != method_id
        or exposure.get("run_id") != identity.get("run_id")
        or exposure.get("identity", {}).get("run_config_sha256")
        != training_config_sha256
        or exposure.get("identity", {}).get("corpus_manifest_sha256")
        != corpus_manifest_sha256
        or exposure.get("identity", {}).get("train_sha256") != train_sha256
        or exposure.get("identity", {}).get("tokenizer_arm_manifest_sha256")
        != tokenizer_arm_manifest_sha256
        or exposure.get("corpus", {}).get("dataset_id") != DATASET_ID
        or exposure.get("exposure_contract", {}).get("satisfied") is not True
    ):
        raise S1EvaluationError("the test requires one complete S1 checkpoint")
    checkpoint_artifact = {
        "files": {
            name: {
                "size_bytes": (checkpoint_path / name).stat().st_size,
                "sha256": sha256_file(checkpoint_path / name),
            }
            for name in required_files
        }
    }
    checkpoint_artifact_sha256 = stable_sha256(checkpoint_artifact)
    return (
        _require_string(identity.get("run_id"), "S1 train run ID"),
        checkpoint_artifact_sha256,
    )


def validate_and_export(
    *,
    repo_root: Path,
    config_path: Path,
    method_id: str,
    checkpoint_path: Path,
) -> dict[str, str]:
    """Validate all S1 evaluation inputs and return shell export values."""

    repo_root = repo_root.resolve()
    config_path = (
        config_path if config_path.is_absolute() else repo_root / config_path
    ).resolve()
    try:
        config_path.relative_to(repo_root)
    except ValueError as error:
        raise S1EvaluationError("RUN_CONFIG escapes the repository") from error
    config = read_json(config_path)
    if (
        config.get("schema_version")
        not in {TEST_SCHEMA_VERSION, TIERED_TEST_SCHEMA_VERSION, RETRY_TEST_SCHEMA_VERSION,
                ROBUSTNESS_TEST_SCHEMA_VERSION}
        or config.get("evaluation_contract") != EVALUATION_CONTRACT
    ):
        raise S1EvaluationError("the S1 test config contract changed")
    sections = [
        config.get(name)
        for name in (
            "training",
            "model",
            "corpus",
            "test_panel",
            "source_index",
            "generation",
            "gate",
            "tracking",
        )
    ]
    if not all(isinstance(section, Mapping) for section in sections):
        raise S1EvaluationError("the S1 test config has an invalid section")
    (
        training,
        model,
        corpus,
        panel_config,
        source_index_config,
        generation,
        gate,
        tracking,
    ) = sections
    diagnostic = config.get("schema_version") in {
        RETRY_TEST_SCHEMA_VERSION, ROBUSTNESS_TEST_SCHEMA_VERSION,
    }
    method = _selected_method(config, method_id)
    if (
        method.get("tokenizer_method") != "plain_base_tokenizer"
        or method.get("checkpoint_kind") != "lora"
        or method.get("fine_tune_configuration") != "lora_r32"
    ):
        raise S1EvaluationError(
            "the S1 test requires the base-tokenizer LoRA-r32 method"
        )
    (
        training_config,
        checkpoint_root,
        training_config_sha256,
    ) = _validate_training_link(
        repo_root=repo_root,
        training=training,
        config=config,
        method=method,
        method_id=method_id,
    )
    training_method = next(
        item
        for item in training_config["methods"]
        if isinstance(item, Mapping) and item.get("method_id") == method_id
    )

    model_path = _model_path(repo_root, model.get("path"))
    model_identity_sha256 = require_sha256(
        model.get("model_identity_sha256"), "model identity"
    )
    if (
        _model_metadata_identity(model_path) != model_identity_sha256
        or model.get("base_model_artifact_schema")
        != "mentor-rl-base-model-artifact-v2"
    ):
        raise S1EvaluationError("the S1 base-model identity changed")
    base_model_artifact_sha256 = require_sha256(
        model.get("base_model_artifact_sha256"), "base-model artifact"
    )
    weight_shard_count = _require_int(
        model.get("weight_shard_count"), "model weight shard count"
    )
    (
        corpus_manifest,
        _,
        corpus_manifest_sha256,
        train_sha256,
        _,
    ) = _validate_corpus(repo_root, corpus)
    contract = load_contract(repo_root / "runtime/world_model/contract.yaml")
    (
        evaluator,
        evaluator_path,
        evaluator_sha256,
        questions_path,
        questions_sha256,
        answers_path,
        answers_sha256,
        row_count,
    ) = _validate_panel(
        repo_root=repo_root,
        panel_config=panel_config,
        corpus_manifest_sha256=corpus_manifest_sha256,
        contract=contract,
        allow_s1_overlay=config.get("schema_version") in {
            TIERED_TEST_SCHEMA_VERSION, RETRY_TEST_SCHEMA_VERSION, ROBUSTNESS_TEST_SCHEMA_VERSION,
        },
    )

    source_index_path = resolve_repo_path(
        repo_root, source_index_config.get("path"), "S1 source index"
    )
    source_index_sha256 = require_sha256(
        source_index_config.get("sha256"), "S1 source-index identity"
    )
    if content_sha256(check_source_index(source_index_path, contract)) != (
        source_index_sha256
    ):
        raise S1EvaluationError("the S1 source-index identity changed")

    (
        tokenizer_path,
        tokenizer_manifest_path,
        tokenizer_manifest_sha256,
        tokenizer_artifacts_sha256,
        tokenizer_arm_manifest_sha256,
    ) = _validate_tokenizer(
        repo_root=repo_root,
        method=method,
        training_method=training_method,
    )

    validation_sha256 = require_sha256(
        corpus_manifest.get("files", {}).get("validation.jsonl", {}).get("sha256"),
        "S1 validation file identity",
    )
    validation_answer_key_sha256 = require_sha256(
        evaluator.get("validation", {}).get("answer_key_sha256"),
        "S1 validation answer-key identity",
    )
    checkpoint_path = checkpoint_path.resolve()
    train_run_id, checkpoint_artifact_sha256 = _validate_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_root=checkpoint_root,
        method_id=method_id,
        model_identity_sha256=model_identity_sha256,
        training_config_sha256=training_config_sha256,
        corpus_manifest_sha256=corpus_manifest_sha256,
        train_sha256=train_sha256,
        tokenizer_arm_manifest_sha256=tokenizer_arm_manifest_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        tokenizer_artifacts_sha256=tokenizer_artifacts_sha256,
        validation_sha256=validation_sha256,
        validation_answer_key_sha256=validation_answer_key_sha256,
    )

    num_nodes = _require_int(generation.get("num_nodes"), "evaluation node count")
    tasks_per_node = _require_int(
        generation.get("tasks_per_node"), "tasks per node"
    )
    gpus_per_node = _require_int(generation.get("gpus_per_node"), "GPUs per node")
    batch_size = _require_int(generation.get("batch_size"), "generation batch size")
    max_tool_call_tokens = _require_int(
        generation.get("max_tool_call_tokens"), "maximum tool-call tokens"
    )
    max_final_prompt_tokens = _require_int(
        generation.get("max_final_prompt_tokens"), "maximum final prompt tokens"
    )
    max_new_tokens = _require_int(
        generation.get("max_new_tokens"), "maximum new tokens"
    )
    max_total_tokens = _require_int(
        generation.get("max_total_tokens"), "maximum total tokens"
    )
    # Keep current-turn analysis available after tools, matching SFT context.
    # Explicit strip remains available for historical comparisons.
    history_policy = generation.get("history_policy", "retain_current_turn")
    if history_policy not in ("strip", "retain_current_turn"):
        raise S1EvaluationError("the S1 reasoning history policy is invalid")
    if (
        tasks_per_node != 1
        or gpus_per_node != 8
        or generation.get("maximum_tool_calls") != (
            3 if diagnostic and isinstance(config.get("diagnostic_panel"), Mapping)
            and config["diagnostic_panel"].get("kind") == "membership_retries" else 2
        )
        or generation.get("do_sample") is not False
        or generation.get("enable_thinking") is not True
        or generation.get("reasoning_effort") != "low"
        or generation.get("local_files_only") is not True
        or max_total_tokens <= max_new_tokens
        or max_tool_call_tokens > max_new_tokens
        or max_final_prompt_tokens >= max_total_tokens
    ):
        raise S1EvaluationError(
            "the deterministic native S1 generation contract changed"
        )

    if diagnostic:
        if config.get("schema_version") == ROBUSTNESS_TEST_SCHEMA_VERSION:
            from scripts.evaluate_world_model_s1_robustness import validate_robustness_config

            validate_diagnostic_config = validate_robustness_config
        else:
            from scripts.world_model_s1_retry_eval import validate_retry_config

            validate_diagnostic_config = validate_retry_config

        panel_config, evaluator_path, evaluator_sha256 = validate_diagnostic_config(
            config, repo_root=repo_root, contract=contract,
        )
        questions_path = repo_root / panel_config["questions_path"]
        answers_path = repo_root / panel_config["answer_key_path"]
        questions_sha256 = panel_config["questions_sha256"]
        answers_sha256 = panel_config["answer_key_sha256"]
        row_count = panel_config["row_count"]
        gate_values = {}
    elif config.get("schema_version") != TIERED_TEST_SCHEMA_VERSION:
        gate_values = _validate_promotion_gate(gate)
    elif gate.get("profile") == STRUCTURED_95_GATE_PROFILE:
        gate_values = _validate_structured_95_gate(gate)
    elif gate.get("profile") == TIERED_V3_GATE_PROFILE:
        gate_values = _validate_tiered_v3_gate(gate)
    else:
        gate_values = _validate_tiered_gate(gate)
    if (
        tracking.get("entity") != "jail-ai"
        or tracking.get("project") != "mentor-sft"
        or tracking.get("resume_training_run") is not (not diagnostic)
    ):
        raise S1EvaluationError("the S1 W&B evaluation contract changed")

    values = {
        "S1_DIAGNOSTIC_ONLY": "1" if diagnostic else "0",
        "S1_DIAGNOSTIC_KIND": config["diagnostic_panel"]["kind"] if diagnostic else "",
        "S1_EVAL_CONFIG": str(config_path),
        "S1_EVAL_CONFIG_SHA256": sha256_file(config_path),
        "S1_EVALUATION_ID": _require_string(
            config.get("evaluation_id"), "evaluation ID"
        ),
        "S1_EVALUATION_CONTRACT": EVALUATION_CONTRACT,
        "S1_METHOD_ID": method_id,
        "S1_TOKENIZER_METHOD": "plain_base_tokenizer",
        "TOKENIZER_PATH": str(tokenizer_path),
        "TOKENIZER_MANIFEST": str(tokenizer_manifest_path),
        "S1_TOKENIZER_MANIFEST_SHA256": tokenizer_manifest_sha256,
        "S1_TOKENIZER_ARTIFACTS_SHA256": tokenizer_artifacts_sha256,
        "S1_TOKENIZER_ARM_MANIFEST_SHA256": tokenizer_arm_manifest_sha256,
        "MODEL_ID": _require_string(model.get("model_id"), "model ID"),
        "MODEL_PATH": str(model_path),
        "MODEL_IDENTITY_SHA256": model_identity_sha256,
        "BASE_MODEL_ARTIFACT_SHA256": base_model_artifact_sha256,
        "BASE_MODEL_WEIGHT_SHARD_COUNT": str(weight_shard_count),
        "S1_CORPUS_MANIFEST_SHA256": corpus_manifest_sha256,
        "S1_EVALUATOR_MANIFEST": str(evaluator_path),
        "S1_EVALUATOR_MANIFEST_SHA256": evaluator_sha256,
        "S1_TEST_QUESTIONS": str(questions_path),
        "S1_TEST_QUESTIONS_SHA256": questions_sha256,
        "S1_TEST_ANSWER_KEY": str(answers_path),
        "S1_TEST_ANSWER_KEY_SHA256": answers_sha256,
        "S1_TEST_PANEL_ID": require_sha256(
            panel_config.get("test_panel_id"), "S1 test panel ID"
        ),
        "S1_TEST_ROWS": str(row_count),
        "CHECKPOINT_PATH": str(checkpoint_path),
        "S1_CHECKPOINT_ARTIFACT_SHA256": checkpoint_artifact_sha256,
        "S1_TRAIN_RUN_ID": train_run_id,
        "S1_EVAL_OUTPUT_ROOT": str(
            resolve_repo_path(
                repo_root,
                method.get("output_root"),
                "S1 evaluation output root",
                must_exist=False,
            )
        ),
        "S1_EVAL_NUM_NODES": str(num_nodes),
        "S1_EVAL_TASKS_PER_NODE": str(tasks_per_node),
        "S1_EVAL_WORLD_SIZE": str(num_nodes * tasks_per_node),
        "S1_EVAL_GPUS_PER_NODE": str(gpus_per_node),
        "S1_MAX_NEW_TOKENS": str(max_new_tokens),
        "S1_MAX_TOTAL_TOKENS": str(max_total_tokens),
        "S1_MAX_TOOL_CALL_TOKENS": str(max_tool_call_tokens),
        "S1_MAX_FINAL_PROMPT_TOKENS": str(max_final_prompt_tokens),
        "S1_REASONING_EFFORT": "low",
        "S1_HISTORY_POLICY": history_policy,
        "S1_ENABLE_THINKING": "1",
        "S1_EVAL_SEED": str(_require_int(generation.get("seed"), "seed")),
        "S1_EVAL_BATCH_SIZE": str(batch_size),
        "S1_SOURCE_INDEX": str(source_index_path),
        "S1_SOURCE_INDEX_SHA256": source_index_sha256,
        "S1_WANDB_ENTITY": "jail-ai",
        "S1_WANDB_PROJECT": "mentor-sft",
    }
    for name, value in gate_values.items():
        values["S1_" + name.upper()] = str(value)
    return values


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: validate_world_model_s1_eval_contract.py "
            "REPO_ROOT RUN_CONFIG METHOD_ID CHECKPOINT_PATH"
        )
    try:
        values = validate_and_export(
            repo_root=Path(sys.argv[1]),
            config_path=Path(sys.argv[2]),
            method_id=sys.argv[3],
            checkpoint_path=Path(sys.argv[4]),
        )
    except S1EvaluationError as error:
        raise SystemExit(str(error)) from error
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
