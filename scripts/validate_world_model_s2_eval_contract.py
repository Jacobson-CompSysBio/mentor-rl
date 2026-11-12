#!/usr/bin/env python3
"""Validate and export the completed-checkpoint S2 evaluation contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.s2_production_adapter import tools_sha256  # noqa: E402
from runtime.world_model_training import (  # noqa: E402
    tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.build_world_model_s2_eval_bundle import (  # noqa: E402
    CORPUS_SCHEMA_VERSION,
    EVALUATION_CONTRACT,
    EVALUATOR_SCHEMA_VERSION,
    EXPECTED_FAMILIES,
    PRODUCTION_ID,
    PUBLIC_METADATA_KEYS,
    SCORED_TARGET_KEYS,
    TARGET_KEYS,
    sha256_file,
    stable_sha256,
)


TEST_SCHEMA_VERSION = "mentor-rl-world-model-s2-test-v1"
TRAINING_SCHEMA_VERSION = "mentor-rl-world-model-s2-120b-training-v1"
CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-v1"
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s2-training-exposure-v1"
LOSS_CONTRACT = "multiplex_sft_v3"
REPORT_ONLY_GATE_PROFILE = "report_only_v1"
RUNTIME_CONTRACT_SHA256 = (
    "f629c28de87a0440b40130d78614d4c5f65d55cd32e68b7bd144d09df81d9f0f"
)
TRAINING_WIRE_FILES = (
    "runtime/world_model/contract.py",
    "runtime/world_model/contract.yaml",
    "runtime/world_model/sources.py",
    "runtime/world_model/s2_3_tokens.py",
    "runtime/world_model/s2_production_adapter.py",
    "runtime/world_model_training.py",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class S2EvaluationContractError(RuntimeError):
    """Report one invalid S2 evaluation input or identity."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one required JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2EvaluationContractError(
            f"cannot read {label} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise S2EvaluationContractError(f"{label} must be one JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    """Return one required nonempty string."""

    if not isinstance(value, str) or not value.strip():
        raise S2EvaluationContractError(f"{label} must be nonempty text")
    return value


def _require_sha256(value: Any, label: str) -> str:
    """Return one required lowercase SHA-256 value."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S2EvaluationContractError(
            f"{label} must be a lowercase SHA-256 value"
        )
    return value


def _require_int(value: Any, label: str) -> int:
    """Return one required positive integer."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise S2EvaluationContractError(f"{label} must be a positive integer")
    return value


def _resolve_repo_path(
    repo_root: Path,
    value: Any,
    label: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one repository-owned path without allowing escape."""

    declared = Path(_require_string(value, label))
    path = (declared if declared.is_absolute() else repo_root / declared).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as error:
        raise S2EvaluationContractError(f"{label} escapes the repository") from error
    if must_exist and not path.exists():
        raise S2EvaluationContractError(f"{label} is absent: {path}")
    return path


def _model_path(repo_root: Path, value: Any) -> Path:
    """Resolve the shared base-model path beside the repository."""

    declared = Path(_require_string(value, "model path"))
    path = (declared if declared.is_absolute() else repo_root / declared).resolve()
    try:
        path.relative_to(repo_root.parent)
    except ValueError as error:
        raise S2EvaluationContractError(
            "the model path escapes the shared LLM project root"
        ) from error
    if not path.is_dir():
        raise S2EvaluationContractError(f"the model path is absent: {path}")
    return path


def _model_metadata_identity(model_path: Path) -> str:
    """Hash the immutable base-model metadata used by generation."""

    names = ["config.json", "model.safetensors.index.json"]
    if (model_path / "generation_config.json").is_file():
        names.append("generation_config.json")
    if any(not (model_path / name).is_file() for name in names):
        raise S2EvaluationContractError("the base-model metadata is incomplete")
    return hashlib.sha256(
        "".join(
            f"{sha256_file(model_path / name)}  {name}\n" for name in names
        ).encode("utf-8")
    ).hexdigest()


def _internal_manifest_identity(payload: Mapping[str, Any], label: str) -> str:
    """Validate a manifest whose identity excludes its identity field."""

    claimed = _require_sha256(payload.get("manifest_sha256"), label)
    identity = {str(key): value for key, value in payload.items()
                if key != "manifest_sha256"}
    if stable_sha256(identity) != claimed:
        raise S2EvaluationContractError(f"{label} failed its internal identity")
    return claimed


def _selected_method(config: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    """Select exactly one evaluation method from the run config."""

    methods = config.get("methods")
    selected = [
        method for method in methods if isinstance(method, dict)
        and method.get("method_id") == method_id
    ] if isinstance(methods, list) else []
    if len(selected) != 1:
        raise S2EvaluationContractError(
            "METHOD_ID must select one S2 evaluation method"
        )
    return selected[0]


def _validate_training_link(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    method: Mapping[str, Any],
    method_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, str, str]:
    """Bind evaluation to the exact production training config and checkpoint."""

    path = _resolve_repo_path(repo_root, training.get("config"), "training config")
    config_sha256 = _require_sha256(
        training.get("config_sha256"), "training config identity"
    )
    if sha256_file(path) != config_sha256:
        raise S2EvaluationContractError("the linked S2 training config changed")
    train_config = _read_json(path, "training config")
    train_method = train_config.get("method")
    if (
        train_config.get("schema_version") != TRAINING_SCHEMA_VERSION
        or train_config.get("training_id") != training.get("training_id")
        or train_config.get("run_scope") != "production"
        or train_config.get("submission_ready") is not True
        or not isinstance(train_method, Mapping)
        or train_method.get("method_id") != method_id
    ):
        raise S2EvaluationContractError("the linked S2 training contract changed")
    implementation_files = train_config.get("implementation_files")
    wire_pins: dict[str, str] = {}
    if isinstance(implementation_files, list):
        for relative_path in TRAINING_WIRE_FILES:
            matches = [
                item for item in implementation_files
                if isinstance(item, Mapping) and item.get("path") == relative_path
            ]
            if len(matches) != 1:
                raise S2EvaluationContractError(
                    f"the linked S2 training config lacks one {relative_path} pin"
                )
            wire_pins[relative_path] = _require_sha256(
                matches[0].get("sha256"), f"training wire {relative_path}"
            )
    if set(wire_pins) != set(TRAINING_WIRE_FILES):
        raise S2EvaluationContractError(
            "the linked S2 training config lacks its inference wire pins"
        )
    runtime_contract_sha256 = wire_pins["runtime/world_model/contract.yaml"]
    if runtime_contract_sha256 != RUNTIME_CONTRACT_SHA256:
        raise S2EvaluationContractError(
            "the runtime world-model contract pin changed since S2 training"
        )
    for relative_path, identity in wire_pins.items():
        if sha256_file(repo_root / relative_path) != identity:
            raise S2EvaluationContractError(
                f"the S2 training wire changed: {relative_path}"
            )
    corpus = config.get("corpus")
    train_corpus = train_config.get("corpus")
    if (
        train_config.get("model") != config.get("model")
        or not isinstance(corpus, Mapping)
        or not isinstance(train_corpus, Mapping)
        or any(
            train_corpus.get(name) != corpus.get(name)
            for name in (
                "root",
                "manifest_sha256",
                "train_sha256",
                "train_rows",
                "validation_sha256",
                "validation_rows",
                "test_panel_id",
            )
        )
        or any(
            train_method.get(name) != method.get(name)
            for name in (
                "tokenizer_method",
                "tokenizer_path",
                "tokenizer_manifest",
                "tokenizer_manifest_sha256",
                "tokenizer_artifacts_sha256",
            )
        )
    ):
        raise S2EvaluationContractError(
            "the S2 evaluation and training artifact identities disagree"
        )
    checkpoint_root = _resolve_repo_path(
        repo_root, training.get("checkpoint_root"), "checkpoint root"
    )
    expected_root = _resolve_repo_path(
        repo_root, train_method.get("output_root"), "training output root"
    )
    checkpoint_path = _resolve_repo_path(
        repo_root, training.get("checkpoint_path"), "final checkpoint"
    )
    if checkpoint_root != expected_root or checkpoint_path.parent != checkpoint_root:
        raise S2EvaluationContractError("the linked S2 checkpoint path changed")
    return (
        train_config,
        dict(train_method),
        checkpoint_root,
        checkpoint_path,
        config_sha256,
        runtime_contract_sha256,
    )


def _validate_corpus(
    repo_root: Path,
    corpus: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str, str]:
    """Validate the production corpus identities needed by evaluation."""

    root = _resolve_repo_path(repo_root, corpus.get("root"), "S2 corpus root")
    manifest_path = root / "manifest.json"
    manifest_sha256 = _require_sha256(
        corpus.get("manifest_sha256"), "S2 corpus manifest"
    )
    if sha256_file(manifest_path) != manifest_sha256:
        raise S2EvaluationContractError("the S2 corpus manifest identity changed")
    manifest = _read_json(manifest_path, "S2 corpus manifest")
    declared_files = {
        "train.jsonl": (
            _require_sha256(corpus.get("train_sha256"), "S2 train file"),
            _require_int(corpus.get("train_rows"), "S2 train rows"),
        ),
        "validation.jsonl": (
            _require_sha256(corpus.get("validation_sha256"), "S2 validation file"),
            _require_int(corpus.get("validation_rows"), "S2 validation rows"),
        ),
        "test.jsonl": (
            _require_sha256(corpus.get("test_sha256"), "S2 test file"),
            _require_int(corpus.get("test_rows"), "S2 test rows"),
        ),
    }
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != CORPUS_SCHEMA_VERSION
        or manifest.get("production_id") != PRODUCTION_ID
        or manifest.get("panel_id") != corpus.get("test_panel_id")
        or manifest.get("max_length") != 8192
        or manifest.get("truncation_applied") is not False
        or not isinstance(files, Mapping)
        or manifest.get("retained_cases") != 394
        or len(manifest.get("case_entries", [])) != 394
        or len(manifest.get("contract_routes", {})) != 13
        or any(
            not isinstance(files.get(name), Mapping)
            or files[name].get("sha256") != identity
            or files[name].get("row_count") != rows
            for name, (identity, rows) in declared_files.items()
        )
    ):
        raise S2EvaluationContractError("the S2 production corpus contract changed")
    return (
        manifest,
        manifest_sha256,
        declared_files["train.jsonl"][0],
        declared_files["validation.jsonl"][0],
    )


def _validate_tokenizer(
    *,
    repo_root: Path,
    method: Mapping[str, Any],
    training_method: Mapping[str, Any],
) -> tuple[Path, Path, str, str, str]:
    """Bind every load-bearing base-tokenizer artifact."""

    tokenizer_path = _resolve_repo_path(
        repo_root, method.get("tokenizer_path"), "tokenizer path"
    )
    tokenizer_manifest_path = _resolve_repo_path(
        repo_root, method.get("tokenizer_manifest"), "tokenizer manifest"
    )
    tokenizer_manifest_sha256 = _require_sha256(
        method.get("tokenizer_manifest_sha256"), "tokenizer manifest identity"
    )
    tokenizer_artifacts_sha256 = _require_sha256(
        method.get("tokenizer_artifacts_sha256"), "tokenizer artifact identity"
    )
    arm_root = _resolve_repo_path(
        repo_root, training_method.get("arm_root"), "tokenizer arm root"
    )
    arm_manifest_path = arm_root / "manifest.json"
    arm_manifest_sha256 = _require_sha256(
        training_method.get("arm_manifest_sha256"),
        "tokenizer arm manifest identity",
    )
    if (
        tokenizer_path != arm_root / "tokenizer"
        or tokenizer_manifest_path != arm_root / "tokenizer_manifest.json"
        or sha256_file(arm_manifest_path) != arm_manifest_sha256
    ):
        raise S2EvaluationContractError("the tokenizer arm identity changed")
    try:
        arm = _read_json(arm_manifest_path, "tokenizer arm manifest")
        tokenizer_manifest = validated_tokenizer_manifest(tokenizer_manifest_path)
        actual_files = tokenizer_artifact_hashes(tokenizer_path)
    except (OSError, TypeError, ValueError) as error:
        raise S2EvaluationContractError(f"the S2 tokenizer is invalid: {error}") from error
    declared = arm.get("tokenizer_artifact_hashes")
    expected_files = {
        Path(str(name)).name: _require_sha256(value, f"tokenizer artifact {name}")
        for name, value in declared.items()
    } if isinstance(declared, Mapping) else None
    file_hashes = arm.get("file_hashes")
    if (
        arm.get("schema_version") != "mentor-rl-world-model-s0-tokenizer-arm-v4"
        or arm.get("method") != "plain_base_tokenizer"
        or arm.get("tokenizer_manifest_sha256") != tokenizer_manifest_sha256
        or not isinstance(file_hashes, Mapping)
        or not isinstance(declared, Mapping)
        or any(file_hashes.get(name) != value for name, value in declared.items())
        or file_hashes.get("tokenizer_manifest.json")
        != sha256_file(tokenizer_manifest_path)
        or tokenizer_manifest.get("manifest_sha256") != tokenizer_manifest_sha256
        or tokenizer_manifest.get("method") != "plain_base_tokenizer"
        or actual_files != expected_files
        or stable_sha256(actual_files) != tokenizer_artifacts_sha256
    ):
        raise S2EvaluationContractError("the S2 tokenizer contract changed")
    return (
        tokenizer_path,
        tokenizer_manifest_path,
        tokenizer_manifest_sha256,
        tokenizer_artifacts_sha256,
        arm_manifest_sha256,
    )


def _child_file(root: Path, value: Any, label: str) -> Path:
    """Resolve an evaluator-owned file without allowing directory escape."""

    declared = Path(_require_string(value, label))
    path = (declared if declared.is_absolute() else root / declared).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise S2EvaluationContractError(f"{label} escapes the evaluator root") from error
    if not path.is_file():
        raise S2EvaluationContractError(f"{label} is absent: {path}")
    return path


def _validate_panel_rows(
    *,
    questions_path: Path,
    answers_path: Path,
    routes: Mapping[str, Any],
    expected: Mapping[str, Any],
    required_case_count: int,
    maximum_tool_calls: int,
) -> None:
    """Stream-check public wire order, answer separation, and panel coverage."""

    row_ids: list[str] = []
    seen: set[str] = set()
    family_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    observed_maximum_calls = 0
    try:
        questions_source = questions_path.open(encoding="utf-8")
        answers_source = answers_path.open(encoding="utf-8")
        with questions_source, answers_source:
            for line_number, pair in enumerate(
                zip_longest(questions_source, answers_source), start=1
            ):
                question_line, answer_line = pair
                if question_line is None or answer_line is None:
                    raise S2EvaluationContractError(
                        "the S2 question and answer row counts differ"
                    )
                question = json.loads(question_line)
                answer = json.loads(answer_line)
                if not isinstance(question, Mapping) or set(question) != {
                    "row_id", "messages", "tools", "metadata"
                }:
                    raise S2EvaluationContractError(
                        f"S2 question {line_number} has an unsafe shape"
                    )
                row_id = question.get("row_id")
                messages = question.get("messages")
                tools = question.get("tools")
                metadata = question.get("metadata")
                family = metadata.get("question_family") \
                    if isinstance(metadata, Mapping) else None
                route = routes.get(family) if isinstance(family, str) else None
                if (
                    not isinstance(row_id, str)
                    or not row_id
                    or row_id in seen
                    or not isinstance(messages, list)
                    or len(messages) != 3
                    or any(not isinstance(message, Mapping) for message in messages)
                    or [message.get("role") for message in messages]
                    != ["system", "harness", "user"]
                    or not isinstance(tools, list)
                    or not isinstance(metadata, Mapping)
                    or set(metadata) != set(PUBLIC_METADATA_KEYS)
                    or metadata.get("split") != "test"
                    or not isinstance(route, Mapping)
                    or metadata.get("tool_contract_version")
                    != route.get("contract_version")
                    or hashlib.sha256(
                        str(messages[0].get("content", "")).encode("utf-8")
                    ).hexdigest() != route.get("system_prompt_sha256")
                    or tools_sha256(tools) != route.get("tools_sha256")
                ):
                    raise S2EvaluationContractError(
                        f"S2 question {line_number} changed its route or wire contract"
                    )
                if not isinstance(answer, Mapping) or set(answer) != {
                    "row_id", "scoring_metadata", "target"
                } or answer.get("row_id") != row_id:
                    raise S2EvaluationContractError(
                        f"S2 answer {line_number} does not match its question"
                    )
                scoring = answer.get("scoring_metadata")
                target = answer.get("target")
                case_id = scoring.get("catalog_case_id") \
                    if isinstance(scoring, Mapping) else None
                calls = target.get("expected_tool_calls") \
                    if isinstance(target, Mapping) else None
                responses = target.get("expected_tool_responses") \
                    if isinstance(target, Mapping) else None
                if (
                    not isinstance(scoring, Mapping)
                    or set(scoring) != {"question_family", "catalog_case_id"}
                    or scoring.get("question_family") != family
                    or not isinstance(case_id, str)
                    or not isinstance(target, Mapping)
                    or set(target) != set(TARGET_KEYS)
                    or not isinstance(calls, list)
                    or not isinstance(responses, list)
                    or len(calls) != len(responses)
                ):
                    raise S2EvaluationContractError(
                        f"S2 answer {line_number} changed its structured target"
                    )
                for call, response in zip(calls, responses):
                    if (
                        not isinstance(call, Mapping)
                        or set(call) != {"name", "arguments"}
                        or not isinstance(call.get("name"), str)
                        or not isinstance(call.get("arguments"), Mapping)
                        or not isinstance(response, Mapping)
                        or set(response) != {"name", "content"}
                        or response.get("name") != call.get("name")
                        or not isinstance(response.get("content"), Mapping)
                    ):
                        raise S2EvaluationContractError(
                            f"S2 answer {line_number} has an invalid tool fixture"
                        )
                seen.add(row_id)
                row_ids.append(row_id)
                family_counts[str(family)] += 1
                case_counts[case_id] += 1
                observed_maximum_calls = max(observed_maximum_calls, len(calls))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2EvaluationContractError(f"cannot stream the S2 panel: {error}") from error
    if (
        len(row_ids) != expected.get("row_count")
        or stable_sha256(row_ids) != expected.get("record_ids_sha256")
        or dict(sorted(family_counts.items())) != expected.get("family_counts")
        or dict(sorted(case_counts.items())) != expected.get("case_counts")
        or len(case_counts) != required_case_count
        or set(case_counts.values()) != {64}
        or observed_maximum_calls != maximum_tool_calls
    ):
        raise S2EvaluationContractError("the frozen S2 panel coverage changed")


def _validate_panel(
    *,
    repo_root: Path,
    panel_config: Mapping[str, Any],
    corpus_manifest_sha256: str,
    corpus_test_sha256: str,
    maximum_tool_calls: int,
) -> tuple[Path, str, Path, str, Path, str, int]:
    """Validate the frozen answer-separated S2 evaluator."""

    root = _resolve_repo_path(
        repo_root, panel_config.get("evaluator_root"), "S2 evaluator root"
    )
    manifest_path = _child_file(
        root, panel_config.get("evaluator_manifest"), "evaluator manifest"
    )
    manifest_sha256 = _require_sha256(
        panel_config.get("evaluator_manifest_sha256"), "evaluator manifest"
    )
    if sha256_file(manifest_path) != manifest_sha256:
        raise S2EvaluationContractError("the S2 evaluator manifest identity changed")
    manifest = _read_json(manifest_path, "S2 evaluator manifest")
    test = manifest.get("test")
    routes = manifest.get("contract_routes")
    source = manifest.get("source_corpus")
    separation = manifest.get("answer_separation")
    scoring = manifest.get("scoring")
    implementation = manifest.get("implementation_sha256")
    expected_source = {
        "manifest_path": "data/world_model_v4/corpora/s2_production_v1_8k/manifest.json",
        "manifest_sha256": corpus_manifest_sha256,
        "panel_id": panel_config.get("test_panel_id"),
        "production_id": PRODUCTION_ID,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "test_path": "data/world_model_v4/corpora/s2_production_v1_8k/test.jsonl",
        "test_sha256": corpus_test_sha256,
    }
    if (
        manifest.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or manifest.get("dataset_id") != PRODUCTION_ID
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or source != expected_source
        or not isinstance(test, Mapping)
        or not isinstance(routes, Mapping)
        or set(routes) != set(EXPECTED_FAMILIES)
        or not isinstance(implementation, Mapping)
        or separation != {
            "generation_reads_private_answer_keys": False,
            "private_targets_contain_expected_tool_responses": True,
            "public_metadata_fields": list(PUBLIC_METADATA_KEYS),
            "public_questions_end_at_current_user": True,
        }
        or not isinstance(scoring, Mapping)
        or scoring.get("headline") != "structured_accuracy"
        or scoring.get("target_fields") != list(SCORED_TARGET_KEYS)
    ):
        raise S2EvaluationContractError("the S2 evaluator contract changed")
    for relative_path, identity in implementation.items():
        path = _resolve_repo_path(repo_root, relative_path, "evaluator implementation")
        if sha256_file(path) != _require_sha256(identity, relative_path):
            raise S2EvaluationContractError(
                f"the evaluator implementation changed: {relative_path}"
            )
    questions_path = _child_file(
        root, panel_config.get("questions_file"), "test questions"
    )
    answers_path = _child_file(
        root, panel_config.get("answer_key_file"), "test answer key"
    )
    questions_sha256 = _require_sha256(
        panel_config.get("questions_sha256"), "test questions"
    )
    answers_sha256 = _require_sha256(
        panel_config.get("answer_key_sha256"), "test answer key"
    )
    row_count = _require_int(panel_config.get("row_count"), "S2 test rows")
    required_family_count = _require_int(
        panel_config.get("required_family_count"), "required S2 family count"
    )
    required_case_count = _require_int(
        panel_config.get("required_case_count"), "required S2 case count"
    )
    if (
        sha256_file(questions_path) != questions_sha256
        or sha256_file(answers_path) != answers_sha256
        or answers_path.stat().st_mode & 0o077
        or test.get("questions_path") != questions_path.name
        or test.get("questions_sha256") != questions_sha256
        or test.get("answer_key_path") != answers_path.name
        or test.get("answer_key_sha256") != answers_sha256
        or test.get("test_panel_id") != panel_config.get("test_panel_id")
        or test.get("row_count") != row_count
        or len(test.get("family_counts", {})) != required_family_count
        or len(test.get("case_counts", {})) != required_case_count
    ):
        raise S2EvaluationContractError("the frozen S2 test panel changed")
    _validate_panel_rows(
        questions_path=questions_path,
        answers_path=answers_path,
        routes=routes,
        expected=test,
        required_case_count=required_case_count,
        maximum_tool_calls=maximum_tool_calls,
    )
    return (
        manifest_path,
        manifest_sha256,
        questions_path,
        questions_sha256,
        answers_path,
        answers_sha256,
        row_count,
    )


def _validate_checkpoint(
    *,
    checkpoint_path: Path,
    configured_checkpoint_path: Path,
    checkpoint_root: Path,
    training: Mapping[str, Any],
    training_config: Mapping[str, Any],
    method_id: str,
    model_identity_sha256: str,
    training_config_sha256: str,
    corpus_manifest_sha256: str,
    train_sha256: str,
    validation_sha256: str,
    tokenizer_arm_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    tokenizer_artifacts_sha256: str,
) -> tuple[str, str]:
    """Validate the exact complete final S2 checkpoint and exposure receipt."""

    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path != configured_checkpoint_path or not checkpoint_path.is_dir():
        raise S2EvaluationContractError("CHECKPOINT_PATH is not the pinned final checkpoint")
    try:
        checkpoint_path.relative_to(checkpoint_root)
    except ValueError as error:
        raise S2EvaluationContractError(
            "the checkpoint is outside the linked S2 training output"
        ) from error
    identity_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
        "run_contract/training_exposure.json",
    )
    required_files = identity_files + ("training_state.json",)
    for name in required_files:
        if not (checkpoint_path / name).is_file():
            raise S2EvaluationContractError(f"the checkpoint lacks {name}")
    adapter = _read_json(checkpoint_path / "tp_adapter_manifest.json", "adapter manifest")
    identity = adapter.get("identity")
    train_run_id = _require_string(training.get("train_run_id"), "S2 train run ID")
    run_settings = training_config.get("run_settings")
    topology = adapter.get("topology")
    if (
        adapter.get("format") != CHECKPOINT_FORMAT
        or adapter.get("consolidated_peft_adapter") != "adapter_model.safetensors"
        or adapter.get("objective", {}).get("loss_contract") != LOSS_CONTRACT
        or not isinstance(identity, Mapping)
        or identity.get("run_id") != train_run_id
        or identity.get("method_id") != method_id
        or identity.get("model_identity_sha256") != model_identity_sha256
        or identity.get("run_config_sha256") != training_config_sha256
        or identity.get("corpus_manifest_sha256") != corpus_manifest_sha256
        or identity.get("train_sha256") != train_sha256
        or identity.get("validation_sha256") != validation_sha256
        or identity.get("validation_answer_key_sha256") != validation_sha256
        or identity.get("tokenizer_arm_manifest_sha256")
        != tokenizer_arm_manifest_sha256
        or identity.get("tokenizer_manifest_sha256") != tokenizer_manifest_sha256
        or identity.get("tokenizer_artifacts_sha256") != tokenizer_artifacts_sha256
        or not isinstance(run_settings, Mapping)
        or not isinstance(topology, Mapping)
        or topology.get("node_count") != run_settings.get("num_nodes")
        or topology.get("world_size")
        != run_settings.get("num_nodes") * run_settings.get("gpus_per_node")
        or topology.get("tp_size") != run_settings.get("tp_size")
        or topology.get("ep_size") != run_settings.get("ep_size")
    ):
        raise S2EvaluationContractError("the selected S2 checkpoint contract changed")
    exposure = _read_json(
        checkpoint_path / "run_contract/training_exposure.json",
        "training exposure receipt",
    )
    _internal_manifest_identity(exposure, "training exposure receipt")
    train_rows = training_config.get("corpus", {}).get("train_rows")
    total_steps = training_config.get("derived_schedule", {}).get("optimizer_updates")
    if (
        exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION
        or exposure.get("status") != "complete"
        or exposure.get("method_id") != method_id
        or exposure.get("run_id") != train_run_id
        or exposure.get("identity", {}).get("run_config_sha256")
        != training_config_sha256
        or exposure.get("identity", {}).get("corpus_manifest_sha256")
        != corpus_manifest_sha256
        or exposure.get("identity", {}).get("train_sha256") != train_sha256
        or exposure.get("identity", {}).get("tokenizer_arm_manifest_sha256")
        != tokenizer_arm_manifest_sha256
        or exposure.get("identity", {}).get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or exposure.get("corpus", {}).get("dataset_id") != PRODUCTION_ID
        or exposure.get("corpus", {}).get("eligible_train_rows") != train_rows
        or exposure.get("exposure_contract", {}).get("satisfied") is not True
        or exposure.get("logical_exposure", {}).get(
            "all_eligible_train_rows_exposed"
        ) is not True
        or exposure.get("logical_exposure", {}).get("record_occurrences")
        != train_rows
        or exposure.get("schedule", {}).get("completed_global_step") != total_steps
        or exposure.get("schedule", {}).get("total_steps") != total_steps
    ):
        raise S2EvaluationContractError(
            "the test requires one completely exposed S2 checkpoint"
        )
    state = _read_json(checkpoint_path / "training_state.json", "training state")
    if (
        state.get("global_step") != total_steps
        or state.get("batches_consumed_in_epoch") != total_steps
        or state.get("epoch") != training_config.get("run_settings", {}).get(
            "num_train_epochs"
        )
    ):
        raise S2EvaluationContractError("the final S2 training state is incomplete")
    artifact = {
        "files": {
            name: {
                "size_bytes": (checkpoint_path / name).stat().st_size,
                "sha256": sha256_file(checkpoint_path / name),
            }
            for name in identity_files
        }
    }
    return train_run_id, stable_sha256(artifact)


def validate_and_export(
    *,
    repo_root: Path,
    config_path: Path,
    method_id: str,
    checkpoint_path: Path,
) -> dict[str, str]:
    """Validate all S2 evaluation inputs and return shell export values."""

    repo_root = repo_root.resolve()
    config_path = (
        config_path if config_path.is_absolute() else repo_root / config_path
    ).resolve()
    try:
        config_path.relative_to(repo_root)
    except ValueError as error:
        raise S2EvaluationContractError("RUN_CONFIG escapes the repository") from error
    config = _read_json(config_path, "S2 evaluation config")
    if (
        config.get("schema_version") != TEST_SCHEMA_VERSION
        or config.get("evaluation_contract") != EVALUATION_CONTRACT
    ):
        raise S2EvaluationContractError("the S2 evaluation config contract changed")
    names = ("training", "model", "corpus", "test_panel", "generation", "gate", "tracking")
    sections = [config.get(name) for name in names]
    if not all(isinstance(section, Mapping) for section in sections):
        raise S2EvaluationContractError("the S2 evaluation config has an invalid section")
    training, model, corpus, panel, generation, gate, tracking = sections
    method = _selected_method(config, method_id)
    if (
        method.get("tokenizer_method") != "plain_base_tokenizer"
        or method.get("checkpoint_kind") != "lora"
        or method.get("fine_tune_configuration") != "lora_r32"
    ):
        raise S2EvaluationContractError(
            "the S2 evaluation requires the base-tokenizer LoRA-r32 method"
        )
    (
        training_config,
        training_method,
        checkpoint_root,
        configured_checkpoint_path,
        training_config_sha256,
        runtime_contract_sha256,
    ) = _validate_training_link(
        repo_root=repo_root,
        config=config,
        training=training,
        method=method,
        method_id=method_id,
    )
    model_path = _model_path(repo_root, model.get("path"))
    model_identity_sha256 = _require_sha256(
        model.get("model_identity_sha256"), "model identity"
    )
    if (
        _model_metadata_identity(model_path) != model_identity_sha256
        or model.get("base_model_artifact_schema")
        != "mentor-rl-base-model-artifact-v2"
    ):
        raise S2EvaluationContractError("the S2 base-model identity changed")
    base_model_artifact_sha256 = _require_sha256(
        model.get("base_model_artifact_sha256"), "base-model artifact"
    )
    weight_shard_count = _require_int(
        model.get("weight_shard_count"), "model weight shard count"
    )
    corpus_manifest, corpus_manifest_sha256, train_sha256, validation_sha256 = (
        _validate_corpus(repo_root, corpus)
    )
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

    num_nodes = _require_int(generation.get("num_nodes"), "evaluation node count")
    tasks_per_node = _require_int(generation.get("tasks_per_node"), "tasks per node")
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
    maximum_tool_calls = _require_int(
        generation.get("maximum_tool_calls"), "maximum tool calls"
    )
    if (
        num_nodes != 512
        or tasks_per_node != 1
        or gpus_per_node != 8
        or maximum_tool_calls != 4
        or generation.get("do_sample") is not False
        or generation.get("enable_thinking") is not True
        or generation.get("reasoning_effort") != "low"
        or generation.get("history_policy") != "retain_current_turn"
        or generation.get("local_files_only") is not True
        or max_total_tokens <= max_new_tokens
        or max_tool_call_tokens > max_new_tokens
        or max_final_prompt_tokens >= max_total_tokens
    ):
        raise S2EvaluationContractError(
            "the deterministic native S2 generation contract changed"
        )
    corpus_test_sha256 = _require_sha256(corpus.get("test_sha256"), "S2 test file")
    (
        evaluator_path,
        evaluator_sha256,
        questions_path,
        questions_sha256,
        answers_path,
        answers_sha256,
        row_count,
    ) = _validate_panel(
        repo_root=repo_root,
        panel_config=panel,
        corpus_manifest_sha256=corpus_manifest_sha256,
        corpus_test_sha256=corpus_test_sha256,
        maximum_tool_calls=maximum_tool_calls,
    )
    if row_count != corpus_manifest.get("files", {}).get("test.jsonl", {}).get("row_count"):
        raise S2EvaluationContractError("the evaluator and corpus test row counts disagree")
    train_run_id, checkpoint_artifact_sha256 = _validate_checkpoint(
        checkpoint_path=checkpoint_path,
        configured_checkpoint_path=configured_checkpoint_path,
        checkpoint_root=checkpoint_root,
        training=training,
        training_config=training_config,
        method_id=method_id,
        model_identity_sha256=model_identity_sha256,
        training_config_sha256=training_config_sha256,
        corpus_manifest_sha256=corpus_manifest_sha256,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
        tokenizer_arm_manifest_sha256=tokenizer_arm_manifest_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        tokenizer_artifacts_sha256=tokenizer_artifacts_sha256,
    )
    if gate != {
        "profile": REPORT_ONLY_GATE_PROFILE,
        "minimum_structured_accuracy": None,
        "require_complete_coverage": True,
        "require_network_used_false": True,
    }:
        raise S2EvaluationContractError("the report-only S2 scoring contract changed")
    if tracking != {
        "entity": "jail-ai",
        "project": "mentor-sft",
        "resume_training_run": True,
        "resume_policy": "must",
        "publish_after_complete_scoring": True,
    }:
        raise S2EvaluationContractError("the sequential S2 W&B contract changed")

    return {
        "S2_EVAL_CONFIG": str(config_path),
        "S2_EVAL_CONFIG_SHA256": sha256_file(config_path),
        "S2_EVALUATION_ID": _require_string(config.get("evaluation_id"), "evaluation ID"),
        "S2_EVALUATION_CONTRACT": EVALUATION_CONTRACT,
        "S2_METHOD_ID": method_id,
        "S2_TOKENIZER_METHOD": "plain_base_tokenizer",
        "TOKENIZER_PATH": str(tokenizer_path),
        "TOKENIZER_MANIFEST": str(tokenizer_manifest_path),
        "S2_TOKENIZER_MANIFEST_SHA256": tokenizer_manifest_sha256,
        "S2_TOKENIZER_ARTIFACTS_SHA256": tokenizer_artifacts_sha256,
        "S2_TOKENIZER_ARM_MANIFEST_SHA256": tokenizer_arm_manifest_sha256,
        "MODEL_ID": _require_string(model.get("model_id"), "model ID"),
        "MODEL_PATH": str(model_path),
        "MODEL_IDENTITY_SHA256": model_identity_sha256,
        "BASE_MODEL_ARTIFACT_SHA256": base_model_artifact_sha256,
        "BASE_MODEL_WEIGHT_SHARD_COUNT": str(weight_shard_count),
        "S2_CORPUS_MANIFEST_SHA256": corpus_manifest_sha256,
        "S2_RUNTIME_CONTRACT_SHA256": runtime_contract_sha256,
        "S2_EVALUATOR_MANIFEST": str(evaluator_path),
        "S2_EVALUATOR_MANIFEST_SHA256": evaluator_sha256,
        "S2_TEST_QUESTIONS": str(questions_path),
        "S2_TEST_QUESTIONS_SHA256": questions_sha256,
        "S2_TEST_ANSWER_KEY": str(answers_path),
        "S2_TEST_ANSWER_KEY_SHA256": answers_sha256,
        "S2_TEST_PANEL_ID": _require_sha256(panel.get("test_panel_id"), "S2 test panel ID"),
        "S2_TEST_ROWS": str(row_count),
        "CHECKPOINT_PATH": str(checkpoint_path.resolve()),
        "S2_CHECKPOINT_ARTIFACT_SHA256": checkpoint_artifact_sha256,
        "S2_TRAIN_RUN_ID": train_run_id,
        "S2_EVAL_OUTPUT_ROOT": str(_resolve_repo_path(
            repo_root, method.get("output_root"), "S2 evaluation output root", must_exist=False
        )),
        "S2_EVAL_NUM_NODES": str(num_nodes),
        "S2_EVAL_TASKS_PER_NODE": str(tasks_per_node),
        "S2_EVAL_WORLD_SIZE": str(num_nodes * tasks_per_node),
        "S2_EVAL_GPUS_PER_NODE": str(gpus_per_node),
        "S2_MAX_NEW_TOKENS": str(max_new_tokens),
        "S2_MAX_TOTAL_TOKENS": str(max_total_tokens),
        "S2_MAX_TOOL_CALL_TOKENS": str(max_tool_call_tokens),
        "S2_MAX_FINAL_PROMPT_TOKENS": str(max_final_prompt_tokens),
        "S2_MAXIMUM_TOOL_CALLS": str(maximum_tool_calls),
        "S2_REASONING_EFFORT": "low",
        "S2_HISTORY_POLICY": "retain_current_turn",
        "S2_ENABLE_THINKING": "1",
        "S2_EVAL_SEED": str(_require_int(generation.get("seed"), "seed")),
        "S2_EVAL_BATCH_SIZE": str(batch_size),
        "S2_GATE_PROFILE": REPORT_ONLY_GATE_PROFILE,
        "S2_WANDB_ENTITY": "jail-ai",
        "S2_WANDB_PROJECT": "mentor-sft",
    }


def main() -> int:
    """Validate command-line inputs and emit safely quoted shell assignments."""

    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: validate_world_model_s2_eval_contract.py "
            "REPO_ROOT RUN_CONFIG METHOD_ID CHECKPOINT_PATH"
        )
    try:
        values = validate_and_export(
            repo_root=Path(sys.argv[1]),
            config_path=Path(sys.argv[2]),
            method_id=sys.argv[3],
            checkpoint_path=Path(sys.argv[4]),
        )
    except S2EvaluationContractError as error:
        raise SystemExit(str(error)) from error
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
