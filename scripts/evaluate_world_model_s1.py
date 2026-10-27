#!/usr/bin/env python3
"""Score native S1 direct and sequential-tool trajectories exactly."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model_training import validated_tokenizer_manifest  # noqa: E402
from scripts.run_world_model_s1_exact_generation import (  # noqa: E402
    EVIDENCE_ID_POLICY,
    MAX_TOOL_CALLS,
    maximum_tool_calls,
    NETWORK_POLICY,
    deterministic_evidence_id,
)
from scripts.world_model_s1_eval_common import (  # noqa: E402
    DATASET_ID,
    EVALUATION_CONTRACT,
    EVALUATOR_SCHEMA_VERSION,
    LOSS_CONTRACT,
    PREDICTION_SCHEMA_VERSION,
    S1EvaluationError,
    S1_FAMILIES,
    STRUCTURED_95_GATE_PROFILE,
    STRUCTURED_95_MINIMUM,
    final_message,
    message_shape,
    normalize_evidence_ids,
    read_json,
    read_jsonl,
    require_sha256,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    tool_calls,
    tool_payloads,
    validate_answer_payload,
    validate_answer_row,
    validate_question,
    write_json,
)


REPORT_SCHEMA_VERSION = "mentor-rl-world-model-s1-exact-report-v1"
METRICS_SCHEMA_VERSION = "mentor-rl-world-model-s1-exact-metrics-v1"
PREDICTION_KEYS = frozenset(
    {
        "record_id",
        "generated_messages",
        "generations",
        "stop_reason",
        "trajectory_error",
        "tool_call_count",
        "executed_tool_call_count",
        "injected_failure_count",
        "network_used_false",
    }
)
BACKEND_FAILURE_PAYLOADS = (
    {
        "status": "error",
        "provenance": None,
        "error_code": "BACKEND_UNAVAILABLE",
        "message": "registry unreachable",
    },
    {
        "status": "error",
        "provenance": None,
        "error_code": "BACKEND_UNAVAILABLE",
        "message": "graph service unreachable",
    },
)
# The complete frozen panel uses the fixed structured-answer accuracy gate.
GATE_PROFILES = (STRUCTURED_95_GATE_PROFILE,)
SUMMARY_BOOLEAN_FIELDS = (
    "network_used_false",
    "closed_book_zero_calls",
    "valid_trajectory",
    "reasoning_present",
    "reasoning_exact",
    "exact_message_shape",
    "exact_tool_selection",
    "exact_tool_arguments",
    "tool_payload_consistency",
    "evidence_consistency",
    "structured_final_json",
    "final_answer_accuracy",
    "structured_answer_accuracy",
    "status_accuracy",
    "result_accuracy",
    "continuation_decision_accuracy",
    "envelope_accuracy",
    "ambiguous_deferral_consistency",
    "blocked_reason_accuracy",
    "blocked_reason_presence",
    "summary_accuracy",
    "false_defer",
    "unexpected_direct_answer",
    "tool_followup_thinking_present",
    "absolute_invalid_no_call_consistency",
    "ambiguous_absolute_contract",
    "same_gene_null_contract",
    "no_inferred_null_consistency",
    "claim_sets_exact",
    "no_unsupported_claim_leakage",
    "direction_contract_accuracy",
    "direction_truth_accuracy",
    "heldout_metric_mapping_accuracy",
    "direction_defer_null_consistency",
    "module_result_contract",
    "identifier_no_invention_consistency",
    "membership_call_contract",
    "identifier_lookup_argument_consistency",
    "identifier_payload_binding_consistency",
    "null_symbol_consistency",
    "ambiguous_choice_consistency",
    "shared_alias_mapping_accuracy",
    "confident_wrong_direction",
    "family_contract_consistency",
    "failure_harness_accuracy",
    "exact_behavior",
    "exact_trace",
)
LAYER_FAMILY_LABELS = frozenset(
    {"ppi", "coexpression", "tf_target", "bulk_pen", "scpen", "other"}
)
PUBLICATION_IDENTITY_KEYS = (
    "train_run_id",
    "method_id",
    "checkpoint_identity_sha256",
    "base_model_identity_sha256",
    "base_model_artifact_sha256",
    "tokenizer_manifest_sha256",
    "tokenizer_artifacts_sha256",
    "source_index_sha256",
    "generation_bundle_sha256",
)


def _unique_by_id(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise S1EvaluationError(f"a {label} row has no record_id")
        if record_id in result:
            raise S1EvaluationError(f"duplicate {label} record {record_id}")
        result[record_id] = dict(row)
    return result


def load_evaluator_panel(
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Load the public test questions and separately protected answer suffixes."""

    evaluator_manifest_path = evaluator_manifest_path.resolve()
    repo_root = repo_root.resolve()
    if sha256_file(evaluator_manifest_path) != evaluator_manifest_sha256:
        raise S1EvaluationError("the S1 evaluator manifest identity changed")
    evaluator = read_json(evaluator_manifest_path)
    if (
        evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != DATASET_ID
        or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
        or evaluator.get("loss_contract") != LOSS_CONTRACT
    ):
        raise S1EvaluationError("the S1 evaluator contract changed")
    contract_pin = evaluator.get("contract")
    if not isinstance(contract_pin, Mapping):
        raise S1EvaluationError("the S1 evaluator lacks its contract pin")
    contract_path = resolve_repo_path(
        repo_root,
        contract_pin.get("path"),
        "S1 runtime contract",
    )
    if sha256_file(contract_path) != contract_pin.get("sha256"):
        raise S1EvaluationError("the evaluator's S1 runtime contract changed")
    contract = load_contract(contract_path)

    panel = evaluator.get("test")
    if not isinstance(panel, Mapping):
        raise S1EvaluationError("the S1 evaluator has no test panel")
    questions_path = resolve_repo_path(
        repo_root,
        panel.get("questions_path"),
        "S1 test questions",
    )
    answers_path = resolve_repo_path(
        repo_root,
        panel.get("answer_key_path"),
        "S1 test answer key",
    )
    if (
        sha256_file(questions_path) != panel.get("questions_sha256")
        or sha256_file(answers_path) != panel.get("answer_key_sha256")
        or panel.get("test_panel_id") != panel.get("questions_sha256")
    ):
        raise S1EvaluationError("the S1 test panel identity changed")
    question_rows = [
        validate_question(row, contract) for row in read_jsonl(questions_path)
    ]
    questions = _unique_by_id(question_rows, label="question")
    raw_answers = _unique_by_id(read_jsonl(answers_path), label="answer")
    if set(questions) != set(raw_answers) or len(questions) != panel.get(
        "row_count"
    ):
        raise S1EvaluationError("the S1 question and answer sets disagree")
    answers = {
        record_id: validate_answer_row(
            raw_answers[record_id], questions[record_id], contract
        )
        for record_id in questions
    }
    family_counts = Counter(
        str(row["metadata"]["question_family"])
        for row in questions.values()
    )
    if (
        set(family_counts) != set(S1_FAMILIES)
        or dict(sorted(family_counts.items()))
        != dict(sorted(panel.get("family_counts", {}).items()))
    ):
        raise S1EvaluationError("the S1 evaluator family counts changed")
    return evaluator, questions, answers, contract


def load_predictions(
    predictions_path: Path,
    generation_manifest_path: Path,
    *,
    expected_record_ids: set[str],
    expected_test_panel_id: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load one complete, identity-bound S1 prediction artifact."""

    predictions_path = predictions_path.resolve()
    generation_manifest_path = generation_manifest_path.resolve()
    manifest = read_json(generation_manifest_path)
    claimed = manifest.get("manifest_sha256")
    identity = {
        str(key): value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    generation_config = manifest.get("generation_config")
    elapsed = manifest.get("elapsed_seconds")
    if (
        manifest.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or manifest.get("test_panel_id") != expected_test_panel_id
        or manifest.get("reads_private_answer_keys") is not False
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or sha256_file(predictions_path) != manifest.get("predictions_sha256")
        or not isinstance(generation_config, Mapping)
        or stable_sha256(generation_config)
        != manifest.get("generation_config_sha256")
        or generation_config.get("maximum_tool_calls") != max((maximum_tool_calls(r) for r in expected_record_ids), default=MAX_TOOL_CALLS)
        or generation_config.get("evidence_id_policy") != EVIDENCE_ID_POLICY
        or generation_config.get("network_policy") != NETWORK_POLICY
        or generation_config.get("enable_thinking") is not True
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise S1EvaluationError("the S1 generation manifest is invalid")
    for key in (
        "base_model_identity_sha256",
        "base_model_artifact_sha256",
        "checkpoint_identity_sha256",
        "tokenizer_manifest_sha256",
        "tokenizer_artifacts_sha256",
        "generation_bundle_sha256",
        "source_index_sha256",
    ):
        require_sha256(manifest.get(key), f"S1 generation {key}")
    if (
        manifest.get("source_index_sha256")
        != load_contract()["source_index"]["content_sha256"]
        or not isinstance(manifest.get("method_id"), str)
        or not manifest["method_id"]
        or not isinstance(manifest.get("train_run_id"), str)
        or not manifest["train_run_id"]
    ):
        raise S1EvaluationError("the S1 generation input identity changed")
    rows = read_jsonl(predictions_path)
    predictions = _unique_by_id(rows, label="prediction")
    if (
        set(predictions) != expected_record_ids
        or len(predictions) != manifest.get("record_count")
    ):
        raise S1EvaluationError("the S1 prediction record set changed")
    for record_id, prediction in predictions.items():
        if set(prediction) != PREDICTION_KEYS:
            raise S1EvaluationError(
                f"S1 prediction {record_id} has an invalid field set"
            )
        messages = prediction.get("generated_messages")
        generations = prediction.get("generations")
        if not isinstance(messages, list) or not isinstance(generations, list):
            raise S1EvaluationError(
                f"S1 prediction {record_id} lacks generated messages"
            )
        if prediction.get("network_used_false") is not True:
            raise S1EvaluationError(
                f"S1 prediction {record_id} did not prove offline execution"
            )
    return manifest, predictions


def _strict_generated_suffix(
    messages: Any,
    contract: Mapping[str, Any],
    *, call_limit: int = MAX_TOOL_CALLS,
) -> tuple[bool, str | None, bool]:
    """Validate the native current-turn assistant/tool message grammar."""

    if not isinstance(messages, list) or not messages:
        return False, "generated_suffix_is_empty", False
    index = 0
    call_count = 0
    while index < len(messages) - 1:
        call_message = messages[index]
        if not isinstance(call_message, Mapping):
            return False, "generated_message_is_not_object", False
        if call_message.get("role") != "assistant" or "tool_call" not in call_message:
            return False, "expected_assistant_tool_call", False
        call = call_message.get("tool_call")
        if (
            set(call_message) != {"role", "thinking", "tool_call"}
            or not isinstance(call_message.get("thinking"), str)
            or not isinstance(call, Mapping)
            or set(call) != {"name", "arguments"}
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), Mapping)
        ):
            return False, "assistant_tool_call_shape", False
        call_count += 1
        if call_count > call_limit:
            return False, "more_than_two_tool_calls" if call_limit == 2 else "more_than_three_tool_calls", False
        index += 1
        if index >= len(messages):
            return False, "tool_result_is_missing", False
        tool_message = messages[index]
        if (
            not isinstance(tool_message, Mapping)
            or set(tool_message) != {"role", "name", "content"}
            or tool_message.get("role") != "tool"
            or tool_message.get("name") != call.get("name")
            or not isinstance(tool_message.get("content"), Mapping)
        ):
            return False, "tool_result_shape_or_name", False
        index += 1

    if index != len(messages) - 1:
        return False, "generated_suffix_does_not_end_in_final", False
    final = messages[index]
    if (
        not isinstance(final, Mapping)
        or set(final) != {"role", "thinking", "content"}
        or final.get("role") != "assistant"
        or not isinstance(final.get("thinking"), str)
    ):
        return False, "final_message_shape", False
    try:
        validate_answer_payload(final.get("content"), contract)
    except S1EvaluationError as error:
        return False, f"final_answer:{error}", False
    return True, None, True


def _normalized_equal(left: Any, right: Any) -> bool:
    return normalize_evidence_ids(left) == normalize_evidence_ids(right)


def _validate_provenance(
    value: Any,
    contract: Mapping[str, Any],
) -> tuple[bool, str | None, dict[str, dict[str, Any]]]:
    """Validate composite provenance and return its populated channel blocks."""

    if value is None:
        return True, None, {}
    channels = [str(channel) for channel in contract["provenance"]["channels"]]
    if not isinstance(value, Mapping) or set(value) != set(channels):
        return False, "provenance_channels", {}
    populated: dict[str, dict[str, Any]] = {}
    for channel in channels:
        block = value.get(channel)
        if block is None:
            continue
        spec = contract["provenance"][f"{channel}_block"]
        if not isinstance(block, Mapping) or set(block) != set(spec["keys"]):
            return False, f"{channel}_provenance_keys", {}
        evidence_id = block.get("evidence_id")
        if (
            not isinstance(evidence_id, str)
            or re.fullmatch(str(spec["evidence_id_pattern"]), evidence_id)
            is None
        ):
            return False, f"{channel}_evidence_id", {}
        if channel == "registry":
            if (
                block.get("registry") != spec["registry"]
                or block.get("release") != spec["release"]
            ):
                return False, "registry_provenance_constants", {}
        else:
            if (
                not isinstance(block.get("multiplex_id"), str)
                or re.fullmatch(
                    str(spec["multiplex_id_pattern"]),
                    str(block.get("multiplex_id")),
                )
                is None
                or not isinstance(block.get("graph_version"), str)
                or re.fullmatch(
                    str(spec["graph_version_pattern"]),
                    str(block.get("graph_version")),
                )
                is None
                or block.get("layer_scope") != spec["layer_scope"]
            ):
                return False, "graph_provenance_constants", {}
        populated[channel] = dict(block)
    if not populated:
        return False, "empty_composite_provenance", {}
    evidence_ids = [block["evidence_id"] for block in populated.values()]
    if len(set(evidence_ids)) != len(evidence_ids):
        return False, "duplicate_evidence_id", {}
    return True, None, populated


def _all_evidence_ids(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            if key == "evidence_id" and isinstance(item, str):
                result.append(item)
            else:
                result.extend(_all_evidence_ids(item))
        return result
    if isinstance(value, list):
        return [
            evidence_id
            for item in value
            for evidence_id in _all_evidence_ids(item)
        ]
    if isinstance(value, str):
        return re.findall(r"ev_(?:reg|mem)_[a-z0-9]{6}", value)
    return []


def _echoes_malformed_argument(
    call: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> bool:
    """Return whether bad graph provenance repeats a malformed call argument.

    The membership tool echoes the multiplex ID it was given. When the model
    invents a malformed ID, the payload's provenance is invalid because of the
    model, not the harness.
    """

    provenance = payload.get("provenance")
    graph = provenance.get("graph") if isinstance(provenance, Mapping) else None
    arguments = call.get("arguments") if isinstance(call, Mapping) else None
    argument = (
        arguments.get("multiplex_id") if isinstance(arguments, Mapping) else None
    )
    pattern = str(contract["provenance"]["graph_block"]["multiplex_id_pattern"])
    return (
        isinstance(graph, Mapping)
        and isinstance(argument, str)
        and graph.get("multiplex_id") == argument
        and re.fullmatch(pattern, argument) is None
    )


def validate_generated_evidence_policy(
    questions: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    """Bind each successful tool nonce to its panel record/call coordinate."""

    if set(questions) != set(predictions):
        raise S1EvaluationError("the S1 evidence-policy record set changed")
    visible_ids = {
        evidence_id
        for question in questions.values()
        for evidence_id in _all_evidence_ids(question.get("messages"))
    }
    generated_ids: set[str] = set()
    for record_index, (record_id, _) in enumerate(questions.items()):
        messages = predictions[record_id].get("generated_messages")
        if not isinstance(messages, list):
            raise S1EvaluationError(
                f"S1 prediction {record_id} lacks generated messages"
            )
        call_count = 0
        pending_call_index: int | None = None
        pending_call: Mapping[str, Any] | None = None
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "assistant" and isinstance(
                message.get("tool_call"), Mapping
            ):
                pending_call_index = call_count
                pending_call = message["tool_call"]
                call_count += 1
                continue
            if message.get("role") != "tool":
                continue
            if pending_call_index is None:
                raise S1EvaluationError(
                    f"S1 prediction {record_id} has an unpaired tool payload"
                )
            content = message.get("content")
            if not isinstance(content, Mapping):
                raise S1EvaluationError(
                    f"S1 prediction {record_id} has a non-object tool payload"
                )
            provenance = content.get("provenance")
            if content.get("status") == "error":
                if provenance is not None:
                    raise S1EvaluationError(
                        f"S1 prediction {record_id} gives failure evidence"
                    )
                pending_call_index = None
                continue
            valid, error, blocks = _validate_provenance(provenance, contract)
            if not valid and _echoes_malformed_argument(
                pending_call, content, contract
            ):
                # A model error: the row fails evidence consistency when it
                # is scored, and the run continues.
                pending_call_index = None
                continue
            if not valid or len(blocks) != 1:
                raise S1EvaluationError(
                    f"S1 prediction {record_id} violates evidence policy: {error}"
                )
            channel, block = next(iter(blocks.items()))
            try:
                expected_id = deterministic_evidence_id(
                    channel=channel,
                    record_index=record_index,
                    call_index=pending_call_index, record_id=record_id,
                )
            except S1EvaluationError as error_value:
                raise S1EvaluationError(
                    f"S1 prediction {record_id} exceeds the evidence policy"
                ) from error_value
            evidence_id = str(block["evidence_id"])
            if evidence_id != expected_id:
                raise S1EvaluationError(
                    f"S1 prediction {record_id} has a non-deterministic evidence ID"
                )
            if evidence_id in visible_ids or evidence_id in generated_ids:
                raise S1EvaluationError(
                    f"S1 prediction {record_id} reuses an evidence ID"
                )
            generated_ids.add(evidence_id)
            pending_call_index = None


def _prefix_evidence_blocks(
    question: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Collect valid evidence visible in harness and previous-turn messages."""

    visible: dict[str, list[dict[str, Any]]] = {
        str(channel): [] for channel in contract["provenance"]["channels"]
    }
    for message in question["messages"]:
        provenance: Any = None
        if message.get("role") in {"assistant", "tool"} and isinstance(
            message.get("content"), Mapping
        ):
            provenance = message["content"].get("provenance")
        elif message.get("role") == "harness" and isinstance(
            message.get("content"), str
        ):
            start = message["content"].find("{")
            if start >= 0:
                try:
                    harness, _ = json.JSONDecoder().raw_decode(
                        message["content"][start:]
                    )
                except (json.JSONDecodeError, TypeError):
                    harness = None
                if isinstance(harness, Mapping):
                    provenance = harness.get("provenance")
        valid, _, blocks = _validate_provenance(provenance, contract)
        if valid:
            for channel, block in blocks.items():
                visible[channel].append(block)
    return visible


def _tool_evidence_blocks(
    generated_calls: Sequence[Mapping[str, Any]],
    generated_payloads: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[bool, str | None, dict[str, list[dict[str, Any]]]]:
    """Validate tool producer channels and return current-turn evidence."""

    visible: dict[str, list[dict[str, Any]]] = {
        str(channel): [] for channel in contract["provenance"]["channels"]
    }
    if len(generated_calls) != len(generated_payloads):
        return False, "tool_call_payload_count", visible
    seen_ids: set[str] = set()
    for call, payload in zip(generated_calls, generated_payloads, strict=True):
        content = payload.get("content")
        if not isinstance(content, Mapping):
            return False, "tool_payload_is_not_object", visible
        provenance = content.get("provenance")
        if content.get("status") == "error":
            if provenance is not None:
                return False, "error_payload_has_evidence", visible
            continue
        valid, error, blocks = _validate_provenance(provenance, contract)
        if not valid:
            return False, f"tool_payload_{error}", visible
        name = str(call.get("name"))
        expected_channel = contract["provenance"]["producer_channel"].get(name)
        if expected_channel not in blocks or len(blocks) != 1:
            return False, "tool_payload_producer_channel", visible
        block = blocks[str(expected_channel)]
        evidence_id = str(block["evidence_id"])
        if evidence_id in seen_ids:
            return False, "tool_payload_reuses_evidence_id", visible
        seen_ids.add(evidence_id)
        visible[str(expected_channel)].append(block)
    return True, None, visible


def _payload_provenance_blocks(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    content = payload.get("content")
    if not isinstance(content, Mapping):
        return {}
    valid, _, blocks = _validate_provenance(content.get("provenance"), contract)
    return blocks if valid else {}


def _source_bound_final_evidence(
    question: Mapping[str, Any],
    *,
    expected_final_content: Any,
    generated_final_content: Any,
    expected_payloads: Sequence[Mapping[str, Any]],
    generated_payloads: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Require each final evidence block to cite its gold-selected source."""

    if not isinstance(expected_final_content, Mapping) or not isinstance(
        generated_final_content, Mapping
    ):
        return False, "final_content_is_not_object"
    expected_valid, expected_error, expected_blocks = _validate_provenance(
        expected_final_content.get("provenance"), contract
    )
    generated_valid, generated_error, generated_blocks = _validate_provenance(
        generated_final_content.get("provenance"), contract
    )
    if not expected_valid:
        return False, f"expected_final_{expected_error}"
    if not generated_valid:
        return False, f"final_{generated_error}"
    if set(generated_blocks) != set(expected_blocks):
        return False, "final_evidence_channels_changed"
    prefix_blocks = _prefix_evidence_blocks(question, contract)
    for channel, expected_block in expected_blocks.items():
        source_index: int | None = None
        for index, payload in enumerate(expected_payloads):
            expected_source = _payload_provenance_blocks(payload, contract).get(
                channel
            )
            if expected_source == expected_block:
                source_index = index
                break
        generated_block = generated_blocks[channel]
        if source_index is not None:
            if source_index >= len(generated_payloads):
                return False, f"final_{channel}_tool_source_missing"
            generated_source = _payload_provenance_blocks(
                generated_payloads[source_index], contract
            ).get(channel)
            if generated_source is None or generated_block != generated_source:
                return False, f"final_{channel}_tool_source_mismatch"
            continue
        if not any(
            expected_block == candidate for candidate in prefix_blocks[channel]
        ):
            return False, f"expected_final_{channel}_source_unknown"
        if generated_block != expected_block:
            return False, f"final_{channel}_prefix_source_mismatch"
    return True, None


def _thinking(messages: Sequence[Mapping[str, Any]]) -> list[Any]:
    return [
        message.get("thinking") if isinstance(message, Mapping) else None
        for message in messages
        if not isinstance(message, Mapping)
        or message.get("role") == "assistant"
    ]


def _tool_followups_have_thinking(messages: Sequence[Any]) -> bool:
    tool_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "tool"
    ]
    return bool(tool_indexes) and all(
        index + 1 < len(messages)
        and isinstance(messages[index + 1], Mapping)
        and messages[index + 1].get("role") == "assistant"
        and isinstance(messages[index + 1].get("thinking"), str)
        and bool(messages[index + 1]["thinking"].strip())
        for index in tool_indexes
    )


def _safe_tool_calls(messages: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        dict(message["tool_call"])
        for message in messages
        if isinstance(message, Mapping)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_call"), Mapping)
    ]


def _safe_tool_payloads(messages: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {"name": message.get("name"), "content": message.get("content")}
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "tool"
    ]


def _safe_final_message(messages: Sequence[Any]) -> Mapping[str, Any] | None:
    if not messages or not isinstance(messages[-1], Mapping):
        return None
    message = messages[-1]
    if message.get("role") != "assistant" or "tool_call" in message:
        return None
    return message


def _safe_message_shape(messages: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            result.append("invalid")
        elif message.get("role") == "assistant" and "tool_call" in message:
            result.append("assistant_tool_call")
        else:
            result.append(str(message.get("role")))
    return result


def _is_backend_failure(value: Any) -> bool:
    return isinstance(value, Mapping) and dict(value) in BACKEND_FAILURE_PAYLOADS


def _failure_signature(
    calls: Sequence[Mapping[str, Any]],
    payloads: Sequence[Mapping[str, Any]],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Return ordered call-position/name/payload failure coordinates."""

    return [
        (index, str(call.get("name")), dict(payload["content"]))
        for index, (call, payload) in enumerate(zip(calls, payloads))
        if _is_backend_failure(payload.get("content"))
    ]


def _null_fields_preserved(expected: Any, generated: Any) -> bool:
    """Require every explicit gold null to remain explicit and null."""

    if isinstance(expected, Mapping):
        if not isinstance(generated, Mapping):
            return False
        return all(
            (
                key in generated and generated[key] is None
                if value is None
                else _null_fields_preserved(value, generated.get(key))
                if isinstance(value, (Mapping, list))
                else True
            )
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(generated, list) and len(generated) >= len(
            expected
        ) and all(
            _null_fields_preserved(value, generated[index])
            for index, value in enumerate(expected)
            if isinstance(value, (Mapping, list))
        )
    return True


def _same_container_shape(expected: Any, generated: Any) -> bool:
    """Require identical mapping keys and list lengths recursively."""

    if isinstance(expected, Mapping):
        return isinstance(generated, Mapping) and set(generated) == set(
            expected
        ) and all(
            _same_container_shape(value, generated[key])
            for key, value in expected.items()
            if isinstance(value, (Mapping, list))
        )
    if isinstance(expected, list):
        return isinstance(generated, list) and len(generated) == len(
            expected
        ) and all(
            _same_container_shape(value, generated[index])
            for index, value in enumerate(expected)
            if isinstance(value, (Mapping, list))
        )
    return True


def _scalar_leaf_multiset(value: Any) -> Counter[tuple[str, str]]:
    """Return typed, JSON-stable non-null leaves for no-invention checks."""

    leaves: Counter[tuple[str, str]] = Counter()
    if isinstance(value, Mapping):
        for child in value.values():
            leaves.update(_scalar_leaf_multiset(child))
    elif isinstance(value, list):
        for child in value:
            leaves.update(_scalar_leaf_multiset(child))
    elif value is not None:
        leaves[(type(value).__name__, json.dumps(value, sort_keys=True))] += 1
    return leaves


def _layer_families(result: Any) -> list[str]:
    def label(value: Any, *, missing: bool = False) -> str:
        if missing:
            return "<missing>"
        return value if isinstance(value, str) and value in LAYER_FAMILY_LABELS else "<invalid>"

    if not isinstance(result, Mapping):
        return []
    if "layer_family" in result:
        family = result.get("layer_family")
        return [label(family)]
    layers = result.get("layers")
    if not isinstance(layers, list):
        return []
    return [
        (
            label(
                layer.get("layer_family"),
                missing="layer_family" not in layer,
            )
            if isinstance(layer, Mapping)
            else "<invalid>"
        )
        for layer in layers
    ]


def _direction_name_kind(
    expected_result: Mapping[str, Any],
    *,
    trace: str,
    contract: Mapping[str, Any],
) -> str:
    """Classify a direction input from the pinned S1.8 test vocabulary."""

    if trace == "S1.8.5":
        return "ambiguous"
    if trace == "S1.8.6":
        return "unknown"
    input_name = expected_result.get("input")
    metric = expected_result.get("metric")
    if not isinstance(input_name, str):
        return "<unclassified>"
    curriculum = contract.get("curriculum")
    families = (
        curriculum.get("families")
        if isinstance(curriculum, Mapping)
        else None
    )
    s1_8 = families.get("S1.8") if isinstance(families, Mapping) else None
    if not isinstance(s1_8, Mapping):
        return "<unclassified>"
    variants = s1_8.get("name_variants")
    test_variants = variants.get("test") if isinstance(variants, Mapping) else None
    if isinstance(test_variants, Mapping):
        for canonical_metric, names in test_variants.items():
            if (
                isinstance(names, list)
                and input_name in names
                and metric == canonical_metric
            ):
                return "metric_variant"
    aliases = s1_8.get("shared_aliases")
    test_aliases = aliases.get("test") if isinstance(aliases, Mapping) else None
    if isinstance(test_aliases, Mapping) and input_name in test_aliases:
        return "shared_alias"
    conventions = s1_8.get("metric_conventions")
    if (
        isinstance(conventions, Mapping)
        and input_name == metric
        and metric in conventions
    ):
        return "canonical"
    return "<unclassified>"


def _direction_polarity(question: Mapping[str, Any]) -> str | None:
    user_text = "\n".join(
        str(message.get("content"))
        for message in question.get("messages", [])
        if isinstance(message, Mapping) and message.get("role") == "user"
    )
    match = re.search(
        r"\ba (lower|higher) value\b.*?\bmeans\b "
        r"(stronger|weaker|greater|less)",
        user_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    strength = (
        "stronger"
        if match.group(2).lower() in {"stronger", "greater"}
        else "weaker"
    )
    return f"{match.group(1).lower()}_{strength}"


def _family_contract_details(
    question: Mapping[str, Any],
    *,
    expected_calls: Sequence[Mapping[str, Any]],
    generated_calls: Sequence[Mapping[str, Any]],
    expected_payloads: Sequence[Mapping[str, Any]],
    generated_payloads: Sequence[Mapping[str, Any]],
    expected_final_content: Mapping[str, Any],
    generated_final_content: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return absolute family invariants and labels for public breakdowns."""

    trace = str(question["metadata"]["source_trace"])
    family = str(question["metadata"]["question_family"])
    expected_result = expected_final_content.get("result")
    generated_result = generated_final_content.get("result")
    expected_result = expected_result if isinstance(expected_result, Mapping) else {}
    generated_result = (
        generated_result if isinstance(generated_result, Mapping) else {}
    )
    checks: dict[str, bool | None] = {
        "absolute_invalid_no_call_consistency": None,
        "ambiguous_absolute_contract": None,
        "same_gene_null_contract": None,
        "no_inferred_null_consistency": None,
        "claim_sets_exact": None,
        "no_unsupported_claim_leakage": None,
        "direction_contract_accuracy": None,
        "direction_truth_accuracy": None,
        "heldout_metric_mapping_accuracy": None,
        "direction_defer_null_consistency": None,
        "module_result_contract": None,
        "identifier_no_invention_consistency": None,
        "membership_call_contract": None,
        "identifier_lookup_argument_consistency": None,
        "identifier_payload_binding_consistency": None,
        "null_symbol_consistency": None,
        "ambiguous_choice_consistency": None,
        "shared_alias_mapping_accuracy": None,
    }
    if trace in {"S1.1.7", "S1.2.10", "S1.2.12"}:
        checks["absolute_invalid_no_call_consistency"] = (
            not generated_calls
            and generated_final_content.get("status") == "invalid_input"
            and generated_final_content.get("continuation_decision") == "defer"
        )
    if trace == "S1.1.3":
        checks["identifier_no_invention_consistency"] = (
            generated_final_content.get("status") == "not_found"
            and generated_result == expected_result
        )
    if trace in {"S1.1.9", "S1.1.10", "S1.1.11"}:
        expected_membership = [
            (call.get("name"), call.get("arguments"))
            for call in expected_calls
            if call.get("name") == "check_multiplex_membership"
        ]
        generated_membership = [
            (call.get("name"), call.get("arguments"))
            for call in generated_calls
            if call.get("name") == "check_multiplex_membership"
        ]
        checks["membership_call_contract"] = (
            bool(expected_membership)
            and generated_membership == expected_membership
        )
    if trace.startswith("S1.2."):
        expected_id_lookups = [
            (call.get("name"), call.get("arguments"))
            for call in expected_calls
            if call.get("name") == "lookup_human_ensembl_gene_id"
        ]
        if expected_id_lookups:
            generated_id_lookups = [
                (call.get("name"), call.get("arguments"))
                for call in generated_calls
                if call.get("name") == "lookup_human_ensembl_gene_id"
            ]
            checks["identifier_lookup_argument_consistency"] = (
                generated_id_lookups == expected_id_lookups
            )
        successful_id_payload_indexes = [
            index
            for index, (call, payload) in enumerate(
                zip(expected_calls, expected_payloads, strict=False)
            )
            if call.get("name") == "lookup_human_ensembl_gene_id"
            and isinstance(payload.get("content"), Mapping)
            and payload["content"].get("status") == "resolved"
        ]
        if successful_id_payload_indexes:
            bindings = []
            for index in successful_id_payload_indexes:
                generated_payload = (
                    generated_payloads[index]
                    if index < len(generated_payloads)
                    else None
                )
                content = (
                    generated_payload.get("content")
                    if isinstance(generated_payload, Mapping)
                    else None
                )
                bindings.append(
                    isinstance(content, Mapping)
                    and content.get("status") == "resolved"
                    and generated_result.get("gene_symbol")
                    == content.get("gene_symbol")
                )
            checks["identifier_payload_binding_consistency"] = all(bindings)
    if trace == "S1.2.4":
        checks["null_symbol_consistency"] = (
            "gene_symbol" in generated_result
            and generated_result.get("gene_symbol") is None
        )
    if trace in {"S1.2.5", "S1.2.6"}:
        checks["identifier_no_invention_consistency"] = (
            generated_final_content.get("status")
            == expected_final_content.get("status")
            and generated_result == expected_result
        )
    if trace in {"S1.3.1", "S1.3.2", "S1.3.3", "S1.3.4"}:
        checks["ambiguous_absolute_contract"] = (
            generated_final_content.get("continuation_decision") == "defer"
            and "gene_id" not in generated_result
            and "present_gene_ids" not in generated_result
            and "multiplex_id" not in generated_result
            and all(
                call.get("name") != "check_multiplex_membership"
                for call in generated_calls
            )
            and generated_result.get("candidate_gene_ids")
            == expected_result.get("candidate_gene_ids")
            and generated_result.get("candidate_details")
            == expected_result.get("candidate_details")
        )
    if trace == "S1.3.5":
        expected_present = expected_result.get("present_gene_ids")
        checks["ambiguous_choice_consistency"] = (
            isinstance(expected_present, list)
            and len(expected_present) == 1
            and generated_result.get("present_gene_ids") == expected_present
            and generated_result.get("gene_id") == expected_present[0]
        )
    if trace == "S1.3.8":
        expected_candidates = expected_result.get("candidate_gene_ids")
        checks["ambiguous_choice_consistency"] = (
            isinstance(expected_candidates, list)
            and generated_result.get("candidate_gene_ids")
            == expected_candidates
            and generated_result.get("gene_id") in expected_candidates
        )
    if trace == "S1.3.9":
        checks["ambiguous_choice_consistency"] = (
            generated_final_content.get("continuation_decision") == "defer"
            and "gene_id" not in generated_result
            and all(
                call.get("name") != "check_multiplex_membership"
                for call in generated_calls
            )
        )
    if family == "cross_context_alignment" and expected_result.get(
        "same_gene", object()
    ) is None:
        checks["same_gene_null_contract"] = (
            "same_gene" in generated_result
            and generated_result.get("same_gene") is None
            and generated_final_content.get("continuation_decision") == "defer"
        )
    if family == "layer_tag_parsing":
        expected_leaves = _scalar_leaf_multiset(expected_result)
        generated_leaves = _scalar_leaf_multiset(generated_result)
        checks["no_inferred_null_consistency"] = (
            _same_container_shape(expected_result, generated_result)
            and _null_fields_preserved(expected_result, generated_result)
            and all(
                count <= expected_leaves[leaf]
                for leaf, count in generated_leaves.items()
            )
        )
    if family == "layer_family_claims" and {
        "supported_claims",
        "unsupported_claims",
    }.issubset(expected_result):
        expected_supported = expected_result.get("supported_claims")
        expected_unsupported = expected_result.get("unsupported_claims")
        generated_supported = generated_result.get("supported_claims")
        generated_unsupported = generated_result.get("unsupported_claims")
        checks["claim_sets_exact"] = (
            isinstance(generated_supported, list)
            and isinstance(generated_unsupported, list)
            and generated_supported == expected_supported
            and generated_unsupported == expected_unsupported
        )
        checks["no_unsupported_claim_leakage"] = (
            isinstance(expected_supported, list)
            and isinstance(expected_unsupported, list)
            and isinstance(generated_supported, list)
            and all(isinstance(claim, str) for claim in generated_supported)
            and set(generated_supported).issubset(expected_supported)
            and not set(expected_unsupported) & set(generated_supported)
            and not (
                expected_result.get("claim_supported") is False
                and generated_result.get("claim_supported") is True
            )
        )
    expected_metric = expected_result.get("metric")
    expected_statement = expected_result.get("statement_true")
    direction_name_kind = (
        _direction_name_kind(expected_result, trace=trace, contract=contract)
        if family == "direction_conventions"
        else None
    )
    candidate_metrics = expected_result.get("candidate_metrics")
    expected_direction_metrics = (
        [str(expected_metric)]
        if isinstance(expected_metric, str)
        else [str(value) for value in candidate_metrics]
        if direction_name_kind == "shared_alias"
        and isinstance(candidate_metrics, list)
        and all(isinstance(value, str) for value in candidate_metrics)
        else []
    )
    if family == "direction_conventions" and trace in {
        "S1.8.1",
        "S1.8.2",
        "S1.8.3",
        "S1.8.4",
    }:
        direction_fields_exact = all(
            generated_result.get(key) == expected_result.get(key)
            for key in ("lower_is_stronger", "meaning", "rule")
        )
        if direction_name_kind == "metric_variant":
            checks["direction_contract_accuracy"] = direction_fields_exact
        elif direction_name_kind == "shared_alias":
            checks["direction_contract_accuracy"] = (
                direction_fields_exact
                and generated_result.get("metric") is None
                and generated_result.get("candidate_metrics")
                == candidate_metrics
            )
            checks["shared_alias_mapping_accuracy"] = (
                generated_result.get("metric") is None
                and generated_result.get("candidate_metrics")
                == candidate_metrics
            )
        elif direction_name_kind == "canonical":
            checks["direction_contract_accuracy"] = (
                direction_fields_exact
                and generated_result.get("metric") == expected_metric
            )
        else:
            checks["direction_contract_accuracy"] = False
    if family == "direction_conventions" and isinstance(
        expected_statement, bool
    ):
        checks["direction_truth_accuracy"] = (
            generated_result.get("statement_true") is expected_statement
        )
    heldout_metric_name = direction_name_kind == "metric_variant"
    if family == "direction_conventions" and heldout_metric_name:
        checks["heldout_metric_mapping_accuracy"] = (
            generated_result.get("metric") == expected_metric
        )
    if trace in {"S1.8.5", "S1.8.6"}:
        checks["direction_defer_null_consistency"] = (
            generated_final_content.get("continuation_decision") == "defer"
            and all(
                key in generated_result and generated_result.get(key) is None
                for key in (
                    "lower_is_stronger",
                    "meaning",
                    "metric",
                    "rule",
                    "statement_true",
                )
            )
            and generated_result.get("candidate_metrics")
            == expected_result.get("candidate_metrics")
        )
    if family == "module_id_parsing":
        # JSON key order carries no meaning, and the answer key stores keys
        # sorted, so compare keys and values only.
        checks["module_result_contract"] = (
            generated_result == expected_result
            and generated_result.get("existence_asserted") is False
        )
    non_absolute_checks = {
        "direction_truth_accuracy",
        "heldout_metric_mapping_accuracy",
    }
    applicable = [
        value
        for key, value in checks.items()
        if key not in non_absolute_checks and isinstance(value, bool)
    ]
    return {
        **checks,
        "family_contract_consistency": all(applicable),
        "expected_layer_families": _layer_families(expected_result),
        "generated_layer_families": _layer_families(generated_result),
        "expected_direction_metric": (
            str(expected_metric) if expected_metric is not None else None
        ),
        "expected_direction_metrics": expected_direction_metrics,
        "direction_name_kind": direction_name_kind,
        "expected_direction_polarity": (
            _direction_polarity(question)
            if isinstance(expected_statement, bool)
            else None
        ),
        "expected_direction_truth": (
            expected_statement if isinstance(expected_statement, bool) else None
        ),
        "heldout_metric_name": heldout_metric_name,
    }


def score_s1_record(
    question: Mapping[str, Any],
    answer: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Score exact S1 action selection, payloads, and six-field final JSON."""

    expected = answer["expected_messages"]
    generated = prediction.get("generated_messages")
    generated_messages = generated if isinstance(generated, list) else []
    structural_valid, structural_error, final_json_valid = (
        _strict_generated_suffix(generated_messages, contract, call_limit=maximum_tool_calls(str(question["record_id"])))
    )
    expected_calls = tool_calls(expected)
    generated_calls = _safe_tool_calls(generated_messages)
    expected_payloads = tool_payloads(expected)
    generated_payloads = _safe_tool_payloads(generated_messages)
    expected_final = final_message(expected)
    generated_final = _safe_final_message(generated_messages)
    generated_final_content = (
        generated_final.get("content")
        if isinstance(generated_final, Mapping)
        and isinstance(generated_final.get("content"), Mapping)
        else {}
    )
    expected_final_content = (
        expected_final.get("content")
        if isinstance(expected_final, Mapping)
        and isinstance(expected_final.get("content"), Mapping)
        else {}
    )
    family_contract = _family_contract_details(
        question,
        expected_calls=expected_calls,
        generated_calls=generated_calls,
        expected_payloads=expected_payloads,
        generated_payloads=generated_payloads,
        expected_final_content=expected_final_content,
        generated_final_content=generated_final_content,
        contract=contract,
    )

    tool_evidence_valid, tool_evidence_error, _ = (
        _tool_evidence_blocks(generated_calls, generated_payloads, contract)
    )
    final_evidence_valid, final_evidence_error = _source_bound_final_evidence(
        question,
        expected_final_content=expected_final_content,
        generated_final_content=generated_final_content,
        expected_payloads=expected_payloads,
        generated_payloads=generated_payloads,
        contract=contract,
    )
    evidence_consistency = tool_evidence_valid and final_evidence_valid

    exact_tool_selection = [call.get("name") for call in generated_calls] == [
        call.get("name") for call in expected_calls
    ]
    exact_tool_arguments = (
        len(generated_calls) == len(expected_calls)
        and all(
            generated_call.get("arguments") == expected_call.get("arguments")
            for generated_call, expected_call in zip(
                generated_calls, expected_calls, strict=True
            )
        )
    )
    payload_consistency = (
        tool_evidence_valid
        and len(generated_payloads) == len(expected_payloads)
        and all(
            generated_payload.get("name") == expected_payload.get("name")
            and _normalized_equal(
                generated_payload.get("content"),
                expected_payload.get("content"),
            )
            for generated_payload, expected_payload in zip(
                generated_payloads, expected_payloads, strict=True
            )
        )
    )
    final_exact = (
        evidence_consistency
        and generated_final is not None
        and expected_final is not None
        and _normalized_equal(
            generated_final.get("content"), expected_final.get("content")
        )
    )
    # A held-out name answered with the opposite direction is the costly
    # error; a deferral that states no direction is safe.
    confident_wrong_direction = None
    if family_contract.get("direction_name_kind") in {
        "metric_variant",
        "shared_alias",
    }:
        generated_answer = generated_final_content.get("result")
        expected_answer = expected_final_content.get("result")
        generated_direction = (
            generated_answer.get("lower_is_stronger")
            if isinstance(generated_answer, Mapping)
            else None
        )
        expected_direction = (
            expected_answer.get("lower_is_stronger")
            if isinstance(expected_answer, Mapping)
            else None
        )
        confident_wrong_direction = (
            isinstance(generated_direction, bool)
            and generated_direction != expected_direction
        )
    status_exact = (
        bool(generated_final_content)
        and generated_final_content.get("status")
        == expected_final_content.get("status")
    )
    result_exact = (
        generated_final is not None
        and expected_final is not None
        and generated_final_content.get("result")
        == expected_final_content.get("result")
    )
    decision_exact = (
        bool(generated_final_content)
        and generated_final_content.get("continuation_decision")
        == expected_final_content.get("continuation_decision")
    )
    envelope_exact = status_exact and result_exact and decision_exact
    ambiguous_deferral_consistency = (
        status_exact and decision_exact
        if expected_final_content.get("status") == "ambiguous"
        else None
    )
    blocked_reason_exact = (
        generated_final is not None
        and expected_final is not None
        and generated_final_content.get("blocked_reason")
        == expected_final_content.get("blocked_reason")
    )
    # The blocked-reason sentence is free text, like the summary. Score only
    # that it appears exactly when the expected answer blocks (AGENTS.md).
    generated_reason = generated_final_content.get("blocked_reason")
    blocked_reason_presence = (
        generated_final is not None
        and expected_final is not None
        and (isinstance(generated_reason, str) and bool(generated_reason.strip()))
        == (expected_final_content.get("blocked_reason") is not None)
    )
    # Both prose fields are free text. Retain the blocked-reason presence
    # and evidence checks while comparing every other structured field.
    structured_exact = (
        evidence_consistency
        and blocked_reason_presence
        and _normalized_equal(
            {**generated_final_content, "summary": None, "blocked_reason": None},
            {**expected_final_content, "summary": None, "blocked_reason": None},
        )
    )
    summary_exact = (
        generated_final is not None
        and expected_final is not None
        and generated_final_content.get("summary")
        == expected_final_content.get("summary")
    )
    answerable = expected_final_content.get("continuation_decision") != "defer"
    false_defer = (
        generated_final_content.get("continuation_decision") == "defer"
        if answerable
        else None
    )
    unexpected_direct_answer = (
        len(generated_calls) == 0 and generated_final is not None
        if expected_calls
        else None
    )
    tool_followup_thinking_present = (
        len(generated_payloads) == len(expected_payloads)
        and _tool_followups_have_thinking(generated_messages)
        if expected_payloads
        else None
    )
    reasoning_present = bool(generated_messages) and all(
        isinstance(value, str) and bool(value.strip())
        for value in _thinking(generated_messages)
    )
    reasoning_exact = _thinking(generated_messages) == _thinking(expected)
    expected_failure = (
        question["metadata"].get("blocked_reason_code") == "tool_error"
    )
    expected_failure_signature = _failure_signature(
        expected_calls, expected_payloads
    )
    generated_failure_signature = _failure_signature(
        generated_calls, generated_payloads
    )
    expected_failure_count = len(expected_failure_signature)
    generated_failure_count = len(generated_failure_signature)
    declared_failure_count = prediction.get("injected_failure_count")
    failure_harness_accuracy = (
        generated_failure_signature == expected_failure_signature
        and declared_failure_count == generated_failure_count
        and (generated_failure_count > 0) == expected_failure
    )
    exact_shape = _safe_message_shape(generated_messages) == message_shape(expected)
    valid_trajectory = (
        structural_valid
        and prediction.get("stop_reason") == "final"
        and prediction.get("trajectory_error") is None
        and prediction.get("tool_call_count") == len(generated_calls)
        and prediction.get("executed_tool_call_count") == len(generated_calls)
        and reasoning_present
    )
    exact_behavior = (
        valid_trajectory
        and exact_shape
        and exact_tool_selection
        and exact_tool_arguments
        and payload_consistency
        and final_exact
        and failure_harness_accuracy
    )
    return {
        "record_id": question["record_id"],
        "question_family": question["metadata"]["question_family"],
        "source_trace": question["metadata"]["source_trace"],
        "form": question["metadata"]["form"],
        "book_mode": question["metadata"]["book_mode"],
        "expected_failure": expected_failure,
        "expected_tool_call_count": len(expected_calls),
        "generated_tool_call_count": len(generated_calls),
        "stop_reason": prediction.get("stop_reason"),
        "trajectory_error": prediction.get("trajectory_error"),
        "structural_error": structural_error,
        "evidence_error": tool_evidence_error or final_evidence_error,
        "network_used_false": prediction.get("network_used_false") is True,
        "closed_book_zero_calls": (
            len(generated_calls) == 0
            if question["metadata"]["book_mode"] == "closed_book"
            else None
        ),
        "valid_trajectory": valid_trajectory,
        "reasoning_present": reasoning_present,
        "reasoning_exact": reasoning_exact,
        "exact_message_shape": exact_shape,
        "exact_tool_selection": exact_tool_selection,
        "exact_tool_arguments": exact_tool_arguments,
        "tool_payload_consistency": payload_consistency,
        "evidence_consistency": evidence_consistency,
        "structured_final_json": final_json_valid and final_evidence_valid,
        "final_answer_accuracy": final_exact,
        "structured_answer_accuracy": structured_exact,
        "status_accuracy": status_exact,
        "result_accuracy": result_exact,
        "continuation_decision_accuracy": decision_exact,
        "envelope_accuracy": envelope_exact,
        "ambiguous_deferral_consistency": ambiguous_deferral_consistency,
        "blocked_reason_accuracy": blocked_reason_exact,
        "blocked_reason_presence": blocked_reason_presence,
        "summary_accuracy": summary_exact,
        "false_defer": false_defer,
        "unexpected_direct_answer": unexpected_direct_answer,
        "tool_followup_thinking_present": tool_followup_thinking_present,
        "failure_harness_accuracy": failure_harness_accuracy,
        "confident_wrong_direction": confident_wrong_direction,
        "exact_behavior": exact_behavior,
        "exact_trace": exact_behavior and reasoning_exact,
        **family_contract,
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    eligible = [row for row in rows if isinstance(row.get(field), bool)]
    successes = sum(bool(row[field]) for row in eligible)
    total = len(eligible)
    return {
        f"{field}_count": successes,
        f"{field}_total": total,
        f"{field}_rate": successes / total if total else 0.0,
    }


def summarize_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return useful exact rates and trajectory diagnostics for one slice."""

    summary: dict[str, Any] = {
        "row_count": len(rows),
        "expected_tool_call_count": sum(
            int(row["expected_tool_call_count"]) for row in rows
        ),
        "generated_tool_call_count": sum(
            int(row["generated_tool_call_count"]) for row in rows
        ),
        "stop_reason_counts": dict(
            sorted(Counter(str(row.get("stop_reason")) for row in rows).items())
        ),
        "structural_error_counts": dict(
            sorted(
                Counter(
                    str(row["structural_error"])
                    for row in rows
                    if row.get("structural_error") is not None
                ).items()
            )
        ),
        "evidence_error_counts": dict(
            sorted(
                Counter(
                    str(row["evidence_error"])
                    for row in rows
                    if row.get("evidence_error") is not None
                ).items()
            )
        ),
    }
    for field in SUMMARY_BOOLEAN_FIELDS:
        summary.update(_rate(rows, field))
    return summary


def _group_summaries(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: summarize_scores(groups[key]) for key in sorted(groups)
    }


def _layer_family_confusion(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    correct = 0
    total = 0
    for row in rows:
        expected = list(row.get("expected_layer_families") or [])
        generated = list(row.get("generated_layer_families") or [])
        for index in range(max(len(expected), len(generated))):
            expected_label = (
                str(expected[index]) if index < len(expected) else "<extra>"
            )
            generated_label = (
                str(generated[index]) if index < len(generated) else "<missing>"
            )
            matrix[expected_label][generated_label] += 1
            total += 1
            correct += expected_label == generated_label
    return {
        "count": total,
        "correct_count": correct,
        "accuracy_rate": correct / total if total else 0.0,
        "matrix": {
            expected: dict(sorted(generated.items()))
            for expected, generated in sorted(matrix.items())
        },
    }


def _boolean_breakdown(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    return {"row_count": len(rows), **_rate(rows, field)}


def _family_contract_summaries(
    traces: Mapping[str, Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for trace, summary in traces.items():
        checks = {
            "tool_selection": summary["exact_tool_selection_rate"]
            >= thresholds["minimum_exact_tool_selection"],
            "tool_arguments": summary["exact_tool_arguments_rate"]
            >= thresholds["minimum_exact_tool_arguments"],
            "tool_payload": summary["tool_payload_consistency_rate"]
            >= thresholds["minimum_tool_payload_consistency"],
            "final_answer": summary["final_answer_accuracy_rate"]
            >= thresholds["minimum_final_answer_accuracy"],
            "envelope": summary["envelope_accuracy_rate"]
            >= thresholds["minimum_source_trace_envelope_accuracy"],
            "absolute_contract": summary[
                "family_contract_consistency_rate"
            ]
            == 1.0,
        }
        result[trace] = {
            "row_count": summary["row_count"],
            "checks": checks,
            "passed": all(checks.values()),
        }
    return result


def _threshold(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise S1EvaluationError(f"{label} must be in [0, 1]")
    return value


def build_s1_report(
    scores: Sequence[Mapping[str, Any]],
    *,
    generation: Mapping[str, Any],
    evaluator_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    minimum_valid_trajectory: float,
    minimum_reasoning_present: float,
    minimum_exact_tool_selection: float,
    minimum_exact_tool_arguments: float,
    minimum_tool_payload_consistency: float,
    minimum_final_answer_accuracy: float,
    minimum_family_accuracy: float,
    minimum_source_trace_envelope_accuracy: float,
    maximum_unexpected_direct_answer_rate: float,
    minimum_tool_followup_thinking_present: float,
    gate_profile: str = STRUCTURED_95_GATE_PROFILE,
    minimum_strict_rate: float = 1.0,
    minimum_target_rate: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a private record report and public aggregate gate metrics."""

    if gate_profile not in GATE_PROFILES:
        raise S1EvaluationError(f"unsupported S1 gate profile {gate_profile!r}")

    thresholds = {
        "minimum_valid_trajectory": _threshold(
            minimum_valid_trajectory, "minimum_valid_trajectory"
        ),
        "minimum_reasoning_present": _threshold(
            minimum_reasoning_present, "minimum_reasoning_present"
        ),
        "minimum_exact_tool_selection": _threshold(
            minimum_exact_tool_selection, "minimum_exact_tool_selection"
        ),
        "minimum_exact_tool_arguments": _threshold(
            minimum_exact_tool_arguments, "minimum_exact_tool_arguments"
        ),
        "minimum_tool_payload_consistency": _threshold(
            minimum_tool_payload_consistency,
            "minimum_tool_payload_consistency",
        ),
        "minimum_final_answer_accuracy": _threshold(
            minimum_final_answer_accuracy, "minimum_final_answer_accuracy"
        ),
        "minimum_family_accuracy": _threshold(
            minimum_family_accuracy, "minimum_family_accuracy"
        ),
        "minimum_source_trace_envelope_accuracy": _threshold(
            minimum_source_trace_envelope_accuracy,
            "minimum_source_trace_envelope_accuracy",
        ),
        "maximum_unexpected_direct_answer_rate": _threshold(
            maximum_unexpected_direct_answer_rate,
            "maximum_unexpected_direct_answer_rate",
        ),
        "minimum_tool_followup_thinking_present": _threshold(
            minimum_tool_followup_thinking_present,
            "minimum_tool_followup_thinking_present",
        ),
    }
    score_rows = [dict(row) for row in scores]
    overall = summarize_scores(score_rows)
    families = _group_summaries(score_rows, "question_family")
    traces = _group_summaries(score_rows, "source_trace")
    book_modes = _group_summaries(score_rows, "book_mode")
    tool_rows = [row for row in score_rows if row["expected_tool_call_count"] > 0]
    direct_rows = [row for row in score_rows if row["expected_tool_call_count"] == 0]
    failure_rows = [row for row in score_rows if row["expected_failure"]]
    closed_book_rows = [
        row for row in score_rows if row["book_mode"] == "closed_book"
    ]
    tool_summary = summarize_scores(tool_rows)
    failure_summary = summarize_scores(failure_rows)
    closed_book_summary = summarize_scores(closed_book_rows)
    family_minimum = min(
        (summary["exact_behavior_rate"] for summary in families.values()),
        default=0.0,
    )
    source_trace_envelope_minimum = min(
        (summary["envelope_accuracy_rate"] for summary in traces.values()),
        default=0.0,
    )
    family_contracts = _family_contract_summaries(
        traces,
        thresholds=thresholds,
    )
    layer_tag_confusion = _layer_family_confusion(
        [
            row
            for row in score_rows
            if row["question_family"] == "layer_tag_parsing"
        ]
    )
    layer_claim_confusion = _layer_family_confusion(
        [
            row
            for row in score_rows
            if row["question_family"] == "layer_family_claims"
        ]
    )
    direction_rows = [
        row
        for row in score_rows
        if row["question_family"] == "direction_conventions"
    ]
    direction_metric_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_truth_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_truth_polarity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direction_truth_intersection: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    direction_name_kind_groups: dict[
        str, list[dict[str, Any]]
    ] = defaultdict(list)
    for row in direction_rows:
        name_kind = str(row.get("direction_name_kind") or "<unclassified>")
        direction_name_kind_groups[name_kind].append(row)
        expected_metrics = row.get("expected_direction_metrics")
        metric_labels = (
            list(dict.fromkeys(str(value) for value in expected_metrics))
            if isinstance(expected_metrics, list) and expected_metrics
            else []
        )
        for metric_label in metric_labels:
            if isinstance(row.get("direction_contract_accuracy"), bool):
                direction_metric_groups[metric_label].append(row)
        truth = row.get("expected_direction_truth")
        polarity = row.get("expected_direction_polarity")
        if isinstance(truth, bool) and isinstance(polarity, str):
            direction_truth_polarity[polarity].append(row)
            for metric_label in metric_labels or ["<unclassified>"]:
                direction_truth_metric[metric_label].append(row)
                direction_truth_intersection[
                    f"{metric_label}|{polarity}|truth={str(truth).lower()}"
                ].append(row)
    direction_metrics = {
        key: _boolean_breakdown(rows, "direction_contract_accuracy")
        for key, rows in sorted(direction_metric_groups.items())
    }
    direction_truth_breakdown = {
        "by_metric": {
            key: _boolean_breakdown(rows, "direction_truth_accuracy")
            for key, rows in sorted(direction_truth_metric.items())
        },
        "by_polarity": {
            key: _boolean_breakdown(rows, "direction_truth_accuracy")
            for key, rows in sorted(direction_truth_polarity.items())
        },
        "by_metric_polarity_truth": {
            key: _boolean_breakdown(rows, "direction_truth_accuracy")
            for key, rows in sorted(direction_truth_intersection.items())
        },
    }
    direction_contract_minimum = min(
        (
            summary["direction_contract_accuracy_rate"]
            for summary in direction_metrics.values()
        ),
        default=0.0,
    )
    direction_truth_minimum = min(
        (
            summary["direction_truth_accuracy_rate"]
            for breakdown in direction_truth_breakdown.values()
            for summary in breakdown.values()
        ),
        default=0.0,
    )
    direction_name_kinds = {
        key: {
            "row_count": len(rows),
            **_rate(rows, "direction_contract_accuracy"),
            **_rate(rows, "direction_truth_accuracy"),
            **_rate(rows, "heldout_metric_mapping_accuracy"),
            **_rate(rows, "shared_alias_mapping_accuracy"),
        }
        for key, rows in sorted(direction_name_kind_groups.items())
    }
    heldout_metric_rows = [
        row
        for row in direction_rows
        if row.get("direction_name_kind") == "metric_variant"
    ]
    heldout_metric_mapping = _boolean_breakdown(
        heldout_metric_rows,
        "heldout_metric_mapping_accuracy",
    )
    shared_alias_rows = [
        row
        for row in direction_rows
        if row.get("direction_name_kind") == "shared_alias"
    ]
    shared_alias_mapping = _boolean_breakdown(
        shared_alias_rows,
        "shared_alias_mapping_accuracy",
    )
    thresholds = {"minimum_structured_answer_accuracy": STRUCTURED_95_MINIMUM}
    checks = {
        "structured_answer_accuracy": bool(score_rows)
        and overall["structured_answer_accuracy_rate"] >= STRUCTURED_95_MINIMUM,
        "network_used_false": bool(score_rows)
        and overall["network_used_false_rate"] == 1.0,
    }
    gate_details = {
        "structured_answer_accuracy": {
            "correct": overall["structured_answer_accuracy_count"],
            "total": overall["row_count"],
            "rate": overall["structured_answer_accuracy_rate"],
            "target": STRUCTURED_95_MINIMUM,
            "comparison": "point_estimate",
            "scope": "all_panel_rows",
            "passed": checks["structured_answer_accuracy"],
        },
    }
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": generation["test_panel_id"],
        "method_id": generation["method_id"],
        "train_run_id": generation["train_run_id"],
        "base_model_identity_sha256": generation[
            "base_model_identity_sha256"
        ],
        "base_model_artifact_sha256": generation[
            "base_model_artifact_sha256"
        ],
        "checkpoint_identity_sha256": generation[
            "checkpoint_identity_sha256"
        ],
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_artifacts_sha256": generation[
            "tokenizer_artifacts_sha256"
        ],
        "generation_manifest_sha256": generation["manifest_sha256"],
        "generation_bundle_sha256": generation["generation_bundle_sha256"],
        "source_index_sha256": generation["source_index_sha256"],
        "overall": overall,
        "families": families,
        "source_traces": traces,
        "family_contracts": family_contracts,
        "layer_family_confusion": {
            "layer_tag_parsing": layer_tag_confusion,
            "layer_family_claims": layer_claim_confusion,
        },
        "direction_metrics": direction_metrics,
        "direction_truth": direction_truth_breakdown,
        "direction_name_kinds": direction_name_kinds,
        "heldout_metric_variant_mapping": heldout_metric_mapping,
        "heldout_metric_name_mapping": heldout_metric_mapping,
        "shared_alias_mapping": shared_alias_mapping,
        "book_modes": book_modes,
        "tool_rows": tool_summary,
        "direct_rows": summarize_scores(direct_rows),
        "closed_book_rows": closed_book_summary,
        "failure_rows": failure_summary,
        "gate": {
            "profile": gate_profile,
            "minimum_strict_rate": minimum_strict_rate,
            "minimum_target_rate": minimum_target_rate,
            "thresholds": thresholds,
            "observed_minimum_family_accuracy": family_minimum,
            "observed_minimum_source_trace_envelope_accuracy": (
                source_trace_envelope_minimum
            ),
            "checks": checks,
            **({"details": gate_details} if gate_details else {}),
            "passed": all(checks.values()),
        },
    }
    metrics["metrics_sha256"] = stable_sha256(metrics)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": generation["test_panel_id"],
        "generation_manifest_sha256": generation["manifest_sha256"],
        "base_model_artifact_sha256": generation[
            "base_model_artifact_sha256"
        ],
        "checkpoint_identity_sha256": generation[
            "checkpoint_identity_sha256"
        ],
        "generation_bundle_sha256": generation["generation_bundle_sha256"],
        "source_index_sha256": generation["source_index_sha256"],
        "tokenizer_artifacts_sha256": generation[
            "tokenizer_artifacts_sha256"
        ],
        "metrics_sha256": metrics["metrics_sha256"],
        "records": score_rows,
    }
    report["report_sha256"] = stable_sha256(report)
    return report, metrics


def validate_publication_identity(
    generation: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    """Fail closed before resuming an externally visible training run."""

    if expected is None or set(expected) != set(PUBLICATION_IDENTITY_KEYS):
        raise S1EvaluationError(
            "S1 W&B publication lacks its expected generation identity"
        )
    for key in PUBLICATION_IDENTITY_KEYS:
        value = expected.get(key)
        if key.endswith("_sha256"):
            require_sha256(value, f"expected publication {key}")
        elif not isinstance(value, str) or not value:
            raise S1EvaluationError(
                f"expected publication {key} must be nonempty text"
            )
        if generation.get(key) != value:
            raise S1EvaluationError(
                f"S1 W&B publication {key} does not match the launch contract"
            )


def publish_wandb(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Publish S1 structured accuracy and its gate; retain details in reports."""

    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if mode != "online" or not entity or not project:
        raise S1EvaluationError("the S1 W&B publication contract is invalid")
    try:
        import wandb
    except ImportError as error:
        raise S1EvaluationError("the wandb package is absent") from error
    run = wandb.init(
        entity=entity,
        project=project,
        id=str(metrics["train_run_id"]),
        resume="must",
        mode=mode,
        dir=os.environ.get("WANDB_DIR"),
    )
    if run is None or getattr(run, "disabled", False):
        raise S1EvaluationError("W&B did not resume the S1 train run")
    overall = metrics["overall"]
    payload = {
        "test/s1/structured_answer_accuracy": overall["structured_answer_accuracy_rate"],
        "test/s1/gate_passed": int(metrics["gate"]["passed"]),
    }
    run.log(payload)
    receipt = {
        "id": run.id,
        "name": run.name,
        "entity": run.entity,
        "project": run.project,
        "url": run.url,
    }
    run.finish()
    return receipt


def evaluate_s1_test(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    predictions_path: Path,
    generation_manifest_path: Path,
    tokenizer_manifest_path: Path,
    output_dir: Path,
    minimum_valid_trajectory: float,
    minimum_reasoning_present: float,
    minimum_exact_tool_selection: float,
    minimum_exact_tool_arguments: float,
    minimum_tool_payload_consistency: float,
    minimum_final_answer_accuracy: float,
    minimum_family_accuracy: float,
    minimum_source_trace_envelope_accuracy: float,
    maximum_unexpected_direct_answer_rate: float,
    minimum_tool_followup_thinking_present: float,
    publish: bool,
    expected_publication_identity: Mapping[str, Any] | None = None,
    gate_profile: str = STRUCTURED_95_GATE_PROFILE,
    minimum_strict_rate: float = 1.0,
    minimum_target_rate: float = 1.0,
) -> dict[str, Any]:
    """Load, score, gate, and optionally publish one exact S1 test."""

    evaluator, questions, answers, contract = load_evaluator_panel(
        evaluator_manifest_path,
        evaluator_manifest_sha256,
    )
    panel = evaluator["test"]
    generation, predictions = load_predictions(
        predictions_path,
        generation_manifest_path,
        expected_record_ids=set(questions),
        expected_test_panel_id=str(panel["test_panel_id"]),
    )
    if publish:
        validate_publication_identity(
            generation,
            expected_publication_identity,
        )
    validate_generated_evidence_policy(questions, predictions, contract)
    tokenizer_manifest = validated_tokenizer_manifest(
        tokenizer_manifest_path.resolve()
    )
    if tokenizer_manifest.get("manifest_sha256") != generation.get(
        "tokenizer_manifest_sha256"
    ):
        raise S1EvaluationError("the S1 generation tokenizer identity changed")
    scores = [
        score_s1_record(
            questions[record_id],
            answers[record_id],
            predictions[record_id],
            contract=contract,
        )
        for record_id in sorted(questions)
    ]
    report, metrics = build_s1_report(
        scores,
        generation=generation,
        evaluator_manifest_sha256=evaluator_manifest_sha256,
        tokenizer_manifest_sha256=str(tokenizer_manifest["manifest_sha256"]),
        minimum_valid_trajectory=minimum_valid_trajectory,
        minimum_reasoning_present=minimum_reasoning_present,
        minimum_exact_tool_selection=minimum_exact_tool_selection,
        minimum_exact_tool_arguments=minimum_exact_tool_arguments,
        minimum_tool_payload_consistency=minimum_tool_payload_consistency,
        minimum_final_answer_accuracy=minimum_final_answer_accuracy,
        minimum_family_accuracy=minimum_family_accuracy,
        minimum_source_trace_envelope_accuracy=(
            minimum_source_trace_envelope_accuracy
        ),
        maximum_unexpected_direct_answer_rate=(
            maximum_unexpected_direct_answer_rate
        ),
        minimum_tool_followup_thinking_present=(
            minimum_tool_followup_thinking_present
        ),
        gate_profile=gate_profile,
        minimum_strict_rate=minimum_strict_rate,
        minimum_target_rate=minimum_target_rate,
    )
    output_dir = output_dir.resolve()
    write_json(output_dir / "exact_s1_report.json", report, private=True)
    write_json(output_dir / "test_metrics.json", metrics, private=False)
    if publish:
        receipt = publish_wandb(metrics)
        write_json(output_dir / "wandb_test_run.json", receipt, private=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid-trajectory", type=float, required=True)
    parser.add_argument("--minimum-reasoning-present", type=float, required=True)
    parser.add_argument(
        "--minimum-exact-tool-selection", type=float, required=True
    )
    parser.add_argument(
        "--minimum-exact-tool-arguments", type=float, required=True
    )
    parser.add_argument(
        "--minimum-tool-payload-consistency", type=float, required=True
    )
    parser.add_argument(
        "--minimum-final-answer-accuracy", type=float, required=True
    )
    parser.add_argument("--minimum-family-accuracy", type=float, required=True)
    parser.add_argument(
        "--minimum-source-trace-envelope-accuracy",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--maximum-unexpected-direct-answer-rate",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--minimum-tool-followup-thinking-present",
        type=float,
        required=True,
    )
    for key in PUBLICATION_IDENTITY_KEYS:
        parser.add_argument("--expected-" + key.replace("_", "-"))
    parser.add_argument("--publish-wandb", action="store_true")
    parser.add_argument(
        "--gate-profile", choices=GATE_PROFILES, default=STRUCTURED_95_GATE_PROFILE
    )
    parser.add_argument("--minimum-strict-rate", type=float, default=1.0)
    parser.add_argument("--minimum-target-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = evaluate_s1_test(
        evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256,
        predictions_path=args.predictions,
        generation_manifest_path=args.generation_manifest,
        tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir,
        minimum_valid_trajectory=args.minimum_valid_trajectory,
        minimum_reasoning_present=args.minimum_reasoning_present,
        minimum_exact_tool_selection=args.minimum_exact_tool_selection,
        minimum_exact_tool_arguments=args.minimum_exact_tool_arguments,
        minimum_tool_payload_consistency=(
            args.minimum_tool_payload_consistency
        ),
        minimum_final_answer_accuracy=args.minimum_final_answer_accuracy,
        minimum_family_accuracy=args.minimum_family_accuracy,
        minimum_source_trace_envelope_accuracy=(
            args.minimum_source_trace_envelope_accuracy
        ),
        maximum_unexpected_direct_answer_rate=(
            args.maximum_unexpected_direct_answer_rate
        ),
        minimum_tool_followup_thinking_present=(
            args.minimum_tool_followup_thinking_present
        ),
        publish=args.publish_wandb,
        expected_publication_identity={
            key: getattr(args, "expected_" + key)
            for key in PUBLICATION_IDENTITY_KEYS
        },
        gate_profile=args.gate_profile,
        minimum_strict_rate=args.minimum_strict_rate,
        minimum_target_rate=args.minimum_target_rate,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "passed": metrics["gate"]["passed"],
                "metrics_sha256": metrics["metrics_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
