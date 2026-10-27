#!/usr/bin/env python3
"""Score the S0 full-registry test.

The scorer replays each generated tool call against the pinned source-index
runtime, parses the final JSON answer against the shared world-model contract,
and compares both with the private answer key. It writes a private per-record
report and public metrics with the gate result, and can append the metrics to
the W&B run of the training job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model.tools import (  # noqa: E402
    ToolCallError,
    WorldModelToolRuntime,
)
from runtime.world_model_training import (  # noqa: E402
    S0_FAMILIES,
    validated_tokenizer_manifest,
)


FULL_REGISTRY_V3_EVALUATOR_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-full-registry-evaluator-v3"
)
FULL_REGISTRY_V3_EVALUATION_CONTRACT = "full_registry_multiplex_sft_v3"
FULL_REGISTRY_V3_DATASET_ID = (
    "world_model_v3_s0_human_identifiers_full_registry"
)
V3_PREDICTION_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-predictions-v3"
)
# These schema names predate v4. They stay unchanged so that new metrics and
# reports remain byte-comparable with the completed v4 test runs.
TEST_METRICS_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-shared-trace-test-metrics-v7"
)
TEST_REPORT_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-shared-trace-test-report-v7"
)
AMBIGUOUS_SYMBOL_FAMILY = "human_ambiguous_symbol"
WILSON_Z_95 = 1.959963984540054
_V3_CONTRACT = load_contract()
V3_ANSWER_KEYS = tuple(_V3_CONTRACT["answer"]["keys"])
V3_ANSWER_STATUSES = frozenset(_V3_CONTRACT["vocabularies"]["status"])
V3_CONTINUATION_DECISIONS = frozenset(
    _V3_CONTRACT["vocabularies"]["continuation_decision"]
)
TOOL_TRAJECTORY_CALL_RE = re.compile(
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"to=functions\.([A-Za-z_][A-Za-z0-9_]*)"
    r"<\|channel\|>commentary json<\|message\|>"
    r"(\{.*?\})<\|call\|>\s*",
    re.DOTALL,
)
SHARED_FINAL_ACTION_RE = re.compile(
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"<\|channel\|>final<\|message\|>(.*?)\s*",
    re.DOTALL,
)


class S0EvaluationError(RuntimeError):
    """Report one invalid S0 test input."""


class WorldModelContractError(ValueError):
    """Report why one final answer breaks the answer contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class V3Answer:
    """The fields needed to score one canonical v3 final action."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        self.status = payload["status"]
        self.result = payload["result"]
        self.continuation_decision = payload["continuation_decision"]


def _v3_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting duplicate keys."""

    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise WorldModelContractError(
                "answer_duplicate_key",
                f"The v3 answer repeats the key {key!r}",
            )
        payload[key] = value
    return payload


def _reject_v3_json_constant(value: str) -> None:
    """Reject NaN and infinities, which canonical JSON does not permit."""

    raise WorldModelContractError(
        "answer_json_constant",
        f"The v3 answer contains the invalid JSON constant {value}",
    )


def parse_v3_answer(text: Any) -> V3Answer:
    """Parse one strict six-field canonical v3 final action."""

    if not isinstance(text, str):
        raise WorldModelContractError("answer_not_text", "The answer must be text")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_v3_json_object,
            parse_constant=_reject_v3_json_constant,
        )
    except WorldModelContractError:
        raise
    except json.JSONDecodeError as error:
        raise WorldModelContractError("answer_json_invalid", str(error)) from error
    if not isinstance(payload, dict) or set(payload) != set(V3_ANSWER_KEYS):
        raise WorldModelContractError(
            "answer_fields_invalid", "The v3 answer fields changed"
        )
    if tuple(payload) != V3_ANSWER_KEYS:
        raise WorldModelContractError(
            "answer_field_order_invalid", "The v3 answer field order changed"
        )
    if (
        not isinstance(payload["status"], str)
        or payload["status"] not in V3_ANSWER_STATUSES
    ):
        raise WorldModelContractError(
            "answer_status_invalid", "The v3 answer status is invalid"
        )
    if payload["provenance"] is not None and not isinstance(
        payload["provenance"], dict
    ):
        raise WorldModelContractError(
            "answer_provenance_invalid",
            "The v3 provenance must be an object or null",
        )
    if not isinstance(payload["result"], dict):
        raise WorldModelContractError(
            "answer_result_invalid", "The v3 result must be an object"
        )
    if (
        not isinstance(payload["continuation_decision"], str)
        or payload["continuation_decision"]
        not in V3_CONTINUATION_DECISIONS
    ):
        raise WorldModelContractError(
            "answer_continuation_decision_invalid",
            "The v3 continuation decision is invalid",
        )
    blocked_reason = payload["blocked_reason"]
    if blocked_reason is not None and (
        not isinstance(blocked_reason, str) or not blocked_reason.strip()
    ):
        raise WorldModelContractError(
            "answer_blocked_reason_invalid",
            "The v3 blocked reason must be nonempty text or null",
        )
    if payload["continuation_decision"] == "defer":
        if blocked_reason is None:
            raise WorldModelContractError(
                "answer_blocked_reason_invalid",
                "A deferred v3 answer requires a blocked reason",
            )
    elif blocked_reason is not None:
        raise WorldModelContractError(
            "answer_blocked_reason_invalid",
            "A non-deferred v3 answer cannot have a blocked reason",
        )
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise WorldModelContractError(
            "answer_summary_invalid", "The v3 summary must be nonempty text"
        )
    return V3Answer(payload)


def _normalize_evidence_ids(value: Any) -> Any:
    """Remove per-call nonce values before semantic equality checks."""

    if isinstance(value, Mapping):
        return {
            key: (
                "<evidence_id>"
                if key == "evidence_id"
                else _normalize_evidence_ids(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_evidence_ids(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return stable compact JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S0EvaluationError(
            f"Could not read one JSON object from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise S0EvaluationError(f"Expected one JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise S0EvaluationError(
                        f"Expected one JSON object at {path}:{line_number}"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise S0EvaluationError(
            f"Could not read JSONL from {path}: {error}"
        ) from error
    return rows


def write_json(path: Path, payload: Mapping[str, Any], *, private: bool) -> None:
    """Write one stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600 if private else 0o640)


def _unique_rows(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    """Index rows by their unique record IDs."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise S0EvaluationError(f"{label} has an invalid record ID")
        if record_id in result:
            raise S0EvaluationError(f"{label} has a duplicate record ID")
        result[record_id] = row
    return result


def strict_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one complete JSON object without extra text."""

    if not isinstance(text, str):
        return None, "prediction_is_not_text"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"json_decode:{error.msg}"
    if not isinstance(value, dict):
        return None, "prediction_is_not_object"
    return value, None


def _remove_terminal_tokens(text: str) -> str:
    """Remove terminal GPT-OSS tokens from one generated turn."""

    result = text.strip()
    while result.endswith("<|return|>") or result.endswith("<|end|>"):
        token = "<|return|>" if result.endswith("<|return|>") else "<|end|>"
        result = result[: -len(token)].rstrip()
    return result


def parse_tool_trajectory_call(
    text: str,
) -> tuple[dict[str, Any] | None, str | None, str | None, bool]:
    """Parse one reason and one GPT-OSS function call."""

    if not isinstance(text, str):
        return None, None, "prediction_is_not_text", False
    direct_answer = "<|channel|>final<|message|>" in text
    direct_answer = direct_answer or strict_json_object(text.strip())[0] is not None
    normalized = _remove_terminal_tokens(text)
    if normalized.count("<|channel|>analysis<|message|>") != 1:
        return None, None, "analysis_count", direct_answer
    if normalized.count("to=functions.") != 1:
        return None, None, "tool_call_count", direct_answer
    match = TOOL_TRAJECTORY_CALL_RE.fullmatch(normalized)
    if match is None:
        return None, None, "trajectory_format", direct_answer
    thinking = match.group(1).strip()
    if not thinking:
        return None, None, "analysis_is_empty", direct_answer
    try:
        arguments = json.loads(match.group(3))
    except json.JSONDecodeError as error:
        return None, thinking, f"argument_json:{error.msg}", direct_answer
    if not isinstance(arguments, dict):
        return None, thinking, "arguments_are_not_an_object", direct_answer
    return {
        "name": match.group(2),
        "arguments": arguments,
    }, thinking, None, direct_answer


def parse_shared_final_action(
    text: str,
) -> tuple[str | None, str | None, str | None]:
    """Parse one thinking turn and one final JSON answer."""

    if not isinstance(text, str):
        return None, None, "prediction_is_not_text"
    normalized = _remove_terminal_tokens(text)
    if normalized.count("<|channel|>analysis<|message|>") != 1:
        return None, None, "final_analysis_count"
    if normalized.count("<|channel|>final<|message|>") != 1:
        return None, None, "final_answer_count"
    match = SHARED_FINAL_ACTION_RE.fullmatch(normalized)
    if match is None:
        return None, None, "final_action_format"
    explanation = match.group(1).strip()
    answer = match.group(2).strip()
    if not explanation:
        return None, None, "final_analysis_is_empty"
    if not answer:
        return explanation, None, "final_answer_is_empty"
    return explanation, answer, None


def score_tool_trajectory_record(
    question: Mapping[str, Any],
    answer_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
    *,
    runtime: WorldModelToolRuntime,
) -> dict[str, Any]:
    """Parse, execute, and score one generated tool trajectory."""

    encoded = prediction_row.get("encoded_prediction")
    if not isinstance(encoded, str):
        raise S0EvaluationError("A prediction has no generated tool turn")
    metadata = question.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != "multiplex_sft_v3"
    ):
        raise S0EvaluationError("A test question does not use the v3 rows")
    parsed, thinking, parse_error, direct_answer = (
        parse_tool_trajectory_call(encoded)
    )
    expected_call = answer_row.get("expected_tool_call")
    expected_payload = answer_row.get("expected_tool_payload")
    expected_thinking = answer_row.get("expected_assistant_thinking")
    expected_final = answer_row.get("expected_final_answer")
    expected_status = answer_row.get("expected_status")
    expected_continuation = answer_row.get(
        "expected_continuation_decision"
    )
    expected_final_explanation = answer_row.get(
        "expected_final_explanation"
    )
    if not isinstance(expected_call, Mapping):
        raise S0EvaluationError("A trajectory answer row has no expected call")
    if not isinstance(expected_payload, Mapping):
        raise S0EvaluationError(
            "A trajectory answer row has no expected payload"
        )
    if not isinstance(expected_thinking, str) or not expected_thinking:
        raise S0EvaluationError(
            "A trajectory answer row has no expected reason"
        )
    if not isinstance(expected_final, str) or not expected_final:
        raise S0EvaluationError(
            "A trajectory answer row has no expected final answer"
        )
    if not isinstance(expected_status, str):
        raise S0EvaluationError(
            "A trajectory answer row has no expected status"
        )

    disallowed_tokens = prediction_row.get("disallowed_special_tokens", [])
    final_disallowed_tokens = prediction_row.get(
        "final_disallowed_special_tokens", []
    )
    if not isinstance(disallowed_tokens, list) or not isinstance(
        final_disallowed_tokens, list
    ):
        raise S0EvaluationError(
            "A trajectory prediction has invalid special-token metadata"
        )

    generated_thinking = prediction_row.get("assistant_thinking")
    thinking_matches_turn = (
        isinstance(generated_thinking, str)
        and generated_thinking == thinking
    )
    reasoning_present = (
        isinstance(thinking, str)
        and bool(thinking.strip())
        and thinking_matches_turn
    )
    reasoning_exact = reasoning_present and thinking == expected_thinking
    exact_call = parsed == expected_call

    # Replay the parsed call. The source-index runtime never uses the network.
    runtime_payload = None
    runtime_tool_error = None
    if parsed is not None:
        try:
            result = runtime.execute(parsed["name"], parsed["arguments"])
        except ToolCallError as error:
            runtime_tool_error = str(error)
        else:
            runtime_payload = dict(result.output)

    generated_payload = prediction_row.get("tool_payload")
    generated_tool_error = prediction_row.get("tool_error")
    generated_network_used_false = prediction_row.get("network_used_false")
    public_runtime_exact = (
        isinstance(generated_payload, Mapping)
        and _normalize_evidence_ids(dict(generated_payload))
        == _normalize_evidence_ids(runtime_payload)
        and generated_tool_error is None
    )
    network_used_false = generated_network_used_false is True
    valid_single_call = (
        parsed is not None
        and runtime_tool_error is None
        and not direct_answer
        and not disallowed_tokens
        and reasoning_present
        and public_runtime_exact
    )
    payload_exact = valid_single_call and (
        _normalize_evidence_ids(runtime_payload)
        == _normalize_evidence_ids(dict(expected_payload))
    )

    encoded_final = prediction_row.get("encoded_final_answer")
    if isinstance(encoded_final, str):
        final_explanation, final_answer, final_parse_error = (
            parse_shared_final_action(encoded_final)
        )
    else:
        final_explanation = None
        final_answer, final_parse_error = None, "final_answer_is_absent"
    generated_final = prediction_row.get("assistant_final")
    final_matches_turn = (
        isinstance(generated_final, str)
        and generated_final == final_answer
    )
    generated_final_explanation = prediction_row.get(
        "assistant_final_explanation"
    )
    final_explanation_matches_turn = (
        isinstance(generated_final_explanation, str)
        and generated_final_explanation == final_explanation
    )
    final_explanation_present = (
        isinstance(final_explanation, str)
        and bool(final_explanation.strip())
        and final_explanation_matches_turn
    )
    final_explanation_exact = (
        final_explanation_present
        and final_explanation == expected_final_explanation
    )
    parsed_structured_answer = None
    structured_answer_error = None
    if final_answer is not None:
        try:
            parsed_structured_answer = parse_v3_answer(final_answer)
        except WorldModelContractError as error:
            structured_answer_error = error.code
    structured_json_valid = parsed_structured_answer is not None
    valid_final_answer = (
        final_answer is not None
        and final_matches_turn
        and not final_disallowed_tokens
        and structured_json_valid
        and final_explanation_present
    )
    expected_answer = parse_v3_answer(expected_final)
    # Evidence IDs are per-call nonces, so compare the answers without them.
    final_answer_exact = valid_final_answer and (
        _normalize_evidence_ids(parsed_structured_answer.payload)
        == _normalize_evidence_ids(expected_answer.payload)
    )
    valid_tool_trajectory = valid_single_call and valid_final_answer
    family = answer_row.get("family")
    status_exact = (
        structured_json_valid
        and parsed_structured_answer.status == expected_status
    )
    continuation_exact = (
        structured_json_valid
        and parsed_structured_answer.continuation_decision
        == expected_continuation
    )
    ambiguous_defer = (
        family != AMBIGUOUS_SYMBOL_FAMILY
        or (
            payload_exact
            and status_exact
            and parsed_structured_answer.continuation_decision == "defer"
        )
    )
    not_found_defer = (
        expected_status != "not_found"
        or (
            status_exact
            and parsed_structured_answer.continuation_decision == "defer"
        )
    )
    expected_result = expected_answer.result
    # The result must restate the tool output, including a null symbol.
    result_exact = (
        structured_json_valid
        and _normalize_evidence_ids(parsed_structured_answer.result)
        == _normalize_evidence_ids(expected_result)
    )
    unnamed_id_resolved = (
        expected_result.get("gene_symbol") is not None
        or expected_status != "resolved"
        or (
            status_exact
            and continuation_exact
            and parsed_structured_answer.result.get("gene_symbol") is None
        )
    )
    return {
        "record_id": question["record_id"],
        "fact_id": answer_row.get("fact_id"),
        "family": family,
        "encoded_prediction": encoded,
        "assistant_thinking": thinking,
        "expected_assistant_thinking": expected_thinking,
        "parsed_tool_call": parsed,
        "expected_tool_call": dict(expected_call),
        "tool_payload": runtime_payload,
        "generated_tool_payload": (
            dict(generated_payload)
            if isinstance(generated_payload, Mapping)
            else generated_payload
        ),
        "expected_tool_payload": dict(expected_payload),
        "encoded_final_answer": encoded_final,
        "assistant_final_explanation": final_explanation,
        "expected_final_explanation": expected_final_explanation,
        "assistant_final": final_answer,
        "expected_final_answer": expected_final,
        "parse_error": parse_error,
        "final_parse_error": final_parse_error,
        "structured_answer_error": structured_answer_error,
        "tool_error": runtime_tool_error,
        "generated_tool_error": generated_tool_error,
        "reasoning_present": reasoning_present,
        "reasoning_exact": reasoning_exact,
        "valid_single_tool_call": valid_single_call,
        "exact_tool_argument": exact_call,
        "payload_exact": payload_exact,
        "valid_final_answer": valid_final_answer,
        "final_explanation_present": final_explanation_present,
        "final_explanation_exact": final_explanation_exact,
        "structured_json_valid": structured_json_valid,
        "status_exact": status_exact,
        "result_exact": result_exact,
        "continuation_decision_exact": continuation_exact,
        "final_answer_exact": final_answer_exact,
        "valid_tool_trajectory": valid_tool_trajectory,
        "ambiguous_defer_correct": ambiguous_defer,
        "not_found_defer_correct": not_found_defer,
        "unnamed_id_resolved_correct": unnamed_id_resolved,
        "direct_answer": direct_answer,
        "network_used_false": network_used_false,
        "disallowed_special_tokens": list(disallowed_tokens),
        "final_disallowed_special_tokens": list(final_disallowed_tokens),
    }


def wilson_interval(successes: int, total: int) -> list[float]:
    """Return one 95-percent Wilson interval."""

    if total < 1:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + WILSON_Z_95**2 / total
    center = (rate + WILSON_Z_95**2 / (2.0 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(
            rate * (1.0 - rate) / total
            + WILSON_Z_95**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _load_tool_panel(
    evaluator_manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    WorldModelToolRuntime,
]:
    """Load the private full-registry panel and its pinned runtime."""

    if sha256_file(evaluator_manifest_path) != expected_manifest_sha256:
        raise S0EvaluationError("The evaluator manifest identity changed")
    manifest = read_json(evaluator_manifest_path)
    if (
        manifest.get("evaluation_contract")
        != FULL_REGISTRY_V3_EVALUATION_CONTRACT
        or manifest.get("schema_version")
        != FULL_REGISTRY_V3_EVALUATOR_SCHEMA_VERSION
        or manifest.get("dataset_id") != FULL_REGISTRY_V3_DATASET_ID
    ):
        raise S0EvaluationError("The trajectory evaluator contract changed")

    test = manifest.get("test")
    registry = manifest.get("identifier_registry")
    if not isinstance(test, Mapping) or not isinstance(registry, Mapping):
        raise S0EvaluationError(
            "The trajectory evaluator lacks a required section"
        )
    questions_path = (
        REPO_ROOT / str(test.get("questions_path", ""))
    ).resolve()
    answer_key_path = (
        REPO_ROOT / str(test.get("answer_key_path", ""))
    ).resolve()
    if sha256_file(questions_path) != test.get("questions_sha256"):
        raise S0EvaluationError("The trajectory question identity changed")
    if sha256_file(answer_key_path) != test.get("answer_key_sha256"):
        raise S0EvaluationError("The trajectory answer identity changed")
    questions = _unique_rows(
        read_jsonl(questions_path), "The trajectory panel"
    )
    answers = _unique_rows(
        read_jsonl(answer_key_path), "The trajectory answer key"
    )
    if set(questions) != set(answers) or len(questions) != test.get(
        "row_count"
    ):
        raise S0EvaluationError("The trajectory panel record set changed")

    registry_path = (
        REPO_ROOT / str(registry.get("path", ""))
    ).resolve()
    if not isinstance(registry.get("sha256"), str):
        raise S0EvaluationError("The evaluator lacks the registry identity")
    runtime = WorldModelToolRuntime.from_source_index(
        registry_path,
        contract=load_contract(),
    )
    return manifest, questions, answers, runtime


def _load_tool_predictions(
    predictions_path: Path,
    generation_manifest_path: Path,
    expected_record_ids: set[str],
    *,
    evaluation_contract: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load one complete trajectory result."""

    generation = read_json(generation_manifest_path)
    claimed = generation.get("manifest_sha256")
    identity = {
        str(key): value
        for key, value in generation.items()
        if key != "manifest_sha256"
    }
    if (
        generation.get("schema_version") != V3_PREDICTION_SCHEMA_VERSION
        or generation.get("evaluation_contract") != evaluation_contract
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or generation.get("predictions_sha256")
        != sha256_file(predictions_path)
    ):
        raise S0EvaluationError("The trajectory generation identity changed")
    predictions = _unique_rows(
        read_jsonl(predictions_path), "The trajectory predictions"
    )
    if set(predictions) != expected_record_ids:
        raise S0EvaluationError(
            "The trajectory prediction record set changed"
        )
    return generation, predictions


def _tool_trajectory_summary(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize tool-trajectory scores for one nonempty set."""

    if not rows:
        raise S0EvaluationError("Cannot summarize an empty trajectory family")
    count = len(rows)

    def rate(field: str) -> tuple[int, float]:
        successes = sum(row.get(field) is True for row in rows)
        return successes, successes / count

    trajectory_count, trajectory_rate = rate("valid_tool_trajectory")
    reason_count, reason_rate = rate("reasoning_present")
    exact_reason_count, exact_reason_rate = rate("reasoning_exact")
    valid_count, valid_rate = rate("valid_single_tool_call")
    exact_count, exact_rate = rate("exact_tool_argument")
    payload_count, payload_rate = rate("payload_exact")
    final_count, final_rate = rate("final_answer_exact")
    structured_count, structured_rate = rate("structured_json_valid")
    status_count, status_rate = rate("status_exact")
    result_count, result_rate = rate("result_exact")
    continuation_count, continuation_rate = rate(
        "continuation_decision_exact"
    )
    final_explanation_count, final_explanation_rate = rate(
        "final_explanation_present"
    )
    exact_final_explanation_count, exact_final_explanation_rate = rate(
        "final_explanation_exact"
    )
    defer_count, defer_rate = rate("ambiguous_defer_correct")
    not_found_count, not_found_rate = rate("not_found_defer_correct")
    unnamed_count, unnamed_rate = rate("unnamed_id_resolved_correct")
    direct_count, direct_rate = rate("direct_answer")
    network_count, network_rate = rate("network_used_false")
    return {
        "record_count": count,
        "valid_tool_trajectory_count": trajectory_count,
        "valid_tool_trajectory_rate": trajectory_rate,
        "reasoning_present_count": reason_count,
        "reasoning_present_rate": reason_rate,
        "reasoning_exact_count": exact_reason_count,
        "reasoning_exact_rate": exact_reason_rate,
        "valid_single_tool_call_count": valid_count,
        "valid_single_tool_call_rate": valid_rate,
        "exact_tool_argument_count": exact_count,
        "exact_tool_argument_rate": exact_rate,
        "payload_exact_count": payload_count,
        "payload_accuracy": payload_rate,
        "payload_wilson_95": wilson_interval(payload_count, count),
        "final_answer_exact_count": final_count,
        "final_answer_accuracy": final_rate,
        "final_answer_wilson_95": wilson_interval(final_count, count),
        "structured_json_valid_count": structured_count,
        "structured_json_valid_rate": structured_rate,
        "status_exact_count": status_count,
        "status_exact_rate": status_rate,
        "result_exact_count": result_count,
        "result_exact_rate": result_rate,
        "continuation_decision_exact_count": continuation_count,
        "continuation_decision_exact_rate": continuation_rate,
        "final_explanation_present_count": final_explanation_count,
        "final_explanation_present_rate": final_explanation_rate,
        "final_explanation_exact_count": exact_final_explanation_count,
        "final_explanation_exact_rate": exact_final_explanation_rate,
        "ambiguous_defer_correct_count": defer_count,
        "ambiguous_defer_accuracy": defer_rate,
        "not_found_defer_correct_count": not_found_count,
        "not_found_defer_accuracy": not_found_rate,
        "unnamed_id_resolved_correct_count": unnamed_count,
        "unnamed_id_resolved_accuracy": unnamed_rate,
        "direct_answer_count": direct_count,
        "direct_answer_rate": direct_rate,
        "network_used_false_count": network_count,
        "network_used_false_rate": network_rate,
    }




def build_tool_trajectory_report(
    rows: list[dict[str, Any]],
    *,
    evaluation_contract: str,
    generation: Mapping[str, Any],
    evaluator_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    registry_sha256: str,
    minimum_valid_tool_trajectory: float,
    minimum_reasoning_present: float,
    minimum_exact_reasoning: float,
    minimum_valid_single_tool_call: float,
    minimum_exact_tool_argument: float,
    minimum_family_payload_accuracy: float,
    minimum_family_final_answer_accuracy: float,
    minimum_ambiguous_defer_accuracy: float,
    maximum_direct_answer_rate: float,
    minimum_structured_json_valid: float,
    minimum_status_exact: float,
    minimum_result_exact: float,
    minimum_continuation_decision_exact: float,
    minimum_not_found_defer_accuracy: float,
    minimum_unnamed_id_resolved_accuracy: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build private and public tool-trajectory reports."""

    by_family = {
        family: _tool_trajectory_summary(
            [row for row in rows if row["family"] == family]
        )
        for family in S0_FAMILIES
    }
    overall = _tool_trajectory_summary(rows)
    ambiguous = by_family[AMBIGUOUS_SYMBOL_FAMILY]
    gate_checks = {
        "valid_tool_trajectory": (
            overall["valid_tool_trajectory_rate"]
            >= minimum_valid_tool_trajectory
        ),
        "reasoning_present": (
            overall["reasoning_present_rate"]
            >= minimum_reasoning_present
        ),
        "exact_reasoning": (
            overall["reasoning_exact_rate"] >= minimum_exact_reasoning
        ),
        "valid_single_tool_call": (
            overall["valid_single_tool_call_rate"]
            >= minimum_valid_single_tool_call
        ),
        "exact_tool_argument": (
            overall["exact_tool_argument_rate"]
            >= minimum_exact_tool_argument
        ),
        "family_payload_accuracy": all(
            family["payload_accuracy"] >= minimum_family_payload_accuracy
            for family in by_family.values()
        ),
        "family_final_answer_accuracy": all(
            family["final_answer_accuracy"]
            >= minimum_family_final_answer_accuracy
            for family in by_family.values()
        ),
        "ambiguous_defer": (
            ambiguous["ambiguous_defer_accuracy"]
            >= minimum_ambiguous_defer_accuracy
        ),
        "direct_answer": (
            overall["direct_answer_rate"] <= maximum_direct_answer_rate
        ),
        "network_disabled": overall["network_used_false_rate"] == 1.0,
        "structured_json_valid": (
            overall["structured_json_valid_rate"]
            >= minimum_structured_json_valid
        ),
        "status_exact": (
            overall["status_exact_rate"] >= minimum_status_exact
        ),
        "result_exact": (
            overall["result_exact_rate"] >= minimum_result_exact
        ),
        "continuation_decision_exact": (
            overall["continuation_decision_exact_rate"]
            >= minimum_continuation_decision_exact
        ),
        # The final turn must also think first, so it shares the floor.
        "final_explanation_present": (
            overall["final_explanation_present_rate"]
            >= minimum_reasoning_present
        ),
        "not_found_defer": (
            overall["not_found_defer_accuracy"]
            >= minimum_not_found_defer_accuracy
        ),
        "unnamed_id_resolved": (
            overall["unnamed_id_resolved_accuracy"]
            >= minimum_unnamed_id_resolved_accuracy
        ),
    }
    metrics = {
        "schema_version": TEST_METRICS_SCHEMA_VERSION,
        "evaluation_contract": evaluation_contract,
        "test_panel_id": generation["test_panel_id"],
        "train_run_id": generation["train_run_id"],
        "method_id": generation["method_id"],
        "checkpoint_identity_sha256": generation[
            "checkpoint_identity_sha256"
        ],
        "generation_manifest_sha256": generation["manifest_sha256"],
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "registry_sha256": registry_sha256,
        "overall": overall,
        "families": by_family,
        "gate": {
            "minimum_valid_tool_trajectory": (
                minimum_valid_tool_trajectory
            ),
            "minimum_reasoning_present": minimum_reasoning_present,
            "minimum_exact_reasoning": minimum_exact_reasoning,
            "minimum_valid_single_tool_call": (
                minimum_valid_single_tool_call
            ),
            "minimum_exact_tool_argument": minimum_exact_tool_argument,
            "minimum_family_payload_accuracy": (
                minimum_family_payload_accuracy
            ),
            "minimum_family_final_answer_accuracy": (
                minimum_family_final_answer_accuracy
            ),
            "minimum_ambiguous_defer_accuracy": (
                minimum_ambiguous_defer_accuracy
            ),
            "maximum_direct_answer_rate": maximum_direct_answer_rate,
            "minimum_structured_json_valid": minimum_structured_json_valid,
            "minimum_status_exact": minimum_status_exact,
            "minimum_result_exact": minimum_result_exact,
            "minimum_continuation_decision_exact": (
                minimum_continuation_decision_exact
            ),
            "minimum_not_found_defer_accuracy": (
                minimum_not_found_defer_accuracy
            ),
            "minimum_unnamed_id_resolved_accuracy": (
                minimum_unnamed_id_resolved_accuracy
            ),
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
    }
    metrics["metrics_sha256"] = stable_sha256(metrics)
    report = {
        "schema_version": TEST_REPORT_SCHEMA_VERSION,
        "private_report": True,
        "metrics": metrics,
        "parse_error_counts": dict(
            sorted(
                Counter(
                    str(row["parse_error"])
                    for row in rows
                    if row["parse_error"] is not None
                ).items()
            )
        ),
        "final_parse_error_counts": dict(
            sorted(
                Counter(
                    str(row["final_parse_error"])
                    for row in rows
                    if row["final_parse_error"] is not None
                ).items()
            )
        ),
        "structured_answer_error_counts": dict(
            sorted(
                Counter(
                    str(row.get("structured_answer_error"))
                    for row in rows
                    if row.get("structured_answer_error") is not None
                ).items()
            )
        ),
        "tool_error_counts": dict(
            sorted(
                Counter(
                    str(row["tool_error"])
                    for row in rows
                    if row["tool_error"] is not None
                ).items()
            )
        ),
        "records": rows,
    }
    report["report_sha256"] = stable_sha256(report)
    return report, metrics


def publish_wandb(
    metrics: Mapping[str, Any], metric_prefix: str = "test"
) -> dict[str, Any]:
    """Publish S0 result accuracy and its gate; retain details in reports."""

    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if mode != "online" or entity != "jail-ai" or not project:
        raise S0EvaluationError("The W&B test contract is invalid")
    try:
        import wandb
    except ImportError as error:
        raise S0EvaluationError("The wandb package is absent") from error

    run = wandb.init(
        entity=entity,
        project=project,
        id=str(metrics["train_run_id"]),
        resume="must",
        mode=mode,
        dir=os.environ.get("WANDB_DIR"),
    )
    if run is None or getattr(run, "disabled", False):
        raise S0EvaluationError("W&B did not resume the train run")
    payload = {
        "test/result_exact_rate": metrics["overall"]["result_exact_rate"],
        "test/gate_passed": int(metrics["gate"]["passed"]),
    }
    if metric_prefix != "test":
        payload = {
            f"{metric_prefix}/{key.split('/', 1)[1]}": value
            for key, value in payload.items()
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


def evaluate_tool_trajectory_test(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    predictions_path: Path,
    generation_manifest_path: Path,
    tokenizer_manifest_path: Path,
    output_dir: Path,
    minimum_valid_tool_trajectory: float,
    minimum_reasoning_present: float,
    minimum_exact_reasoning: float,
    minimum_valid_single_tool_call: float,
    minimum_exact_tool_argument: float,
    minimum_family_payload_accuracy: float,
    minimum_family_final_answer_accuracy: float,
    minimum_ambiguous_defer_accuracy: float,
    maximum_direct_answer_rate: float,
    minimum_structured_json_valid: float,
    minimum_status_exact: float,
    minimum_result_exact: float,
    minimum_continuation_decision_exact: float,
    minimum_not_found_defer_accuracy: float,
    minimum_unnamed_id_resolved_accuracy: float,
    publish: bool,
    wandb_metric_prefix: str = "test",
) -> dict[str, Any]:
    """Score one complete S0 full-registry test result."""

    evaluator, questions, answers, runtime = _load_tool_panel(
        evaluator_manifest_path.resolve(),
        evaluator_manifest_sha256,
    )
    evaluation_contract = str(evaluator["evaluation_contract"])
    generation, predictions = _load_tool_predictions(
        predictions_path.resolve(),
        generation_manifest_path.resolve(),
        set(questions),
        evaluation_contract=evaluation_contract,
    )
    tokenizer_manifest = validated_tokenizer_manifest(
        tokenizer_manifest_path.resolve()
    )
    if (
        tokenizer_manifest.get("method") != "plain_base_tokenizer"
        or tokenizer_manifest.get("manifest_sha256")
        != generation.get("tokenizer_manifest_sha256")
        or generation.get("test_panel_id")
        != evaluator["test"]["test_panel_id"]
    ):
        raise S0EvaluationError("The trajectory generation identity changed")
    scored = [
        score_tool_trajectory_record(
            questions[record_id],
            answers[record_id],
            predictions[record_id],
            runtime=runtime,
        )
        for record_id in sorted(questions)
    ]
    report, metrics = build_tool_trajectory_report(
        scored,
        evaluation_contract=evaluation_contract,
        generation=generation,
        evaluator_manifest_sha256=evaluator_manifest_sha256,
        tokenizer_manifest_sha256=str(
            tokenizer_manifest["manifest_sha256"]
        ),
        registry_sha256=runtime.source_index_sha256,
        minimum_valid_tool_trajectory=minimum_valid_tool_trajectory,
        minimum_reasoning_present=minimum_reasoning_present,
        minimum_exact_reasoning=minimum_exact_reasoning,
        minimum_valid_single_tool_call=minimum_valid_single_tool_call,
        minimum_exact_tool_argument=minimum_exact_tool_argument,
        minimum_family_payload_accuracy=minimum_family_payload_accuracy,
        minimum_family_final_answer_accuracy=(
            minimum_family_final_answer_accuracy
        ),
        minimum_ambiguous_defer_accuracy=(
            minimum_ambiguous_defer_accuracy
        ),
        maximum_direct_answer_rate=maximum_direct_answer_rate,
        minimum_structured_json_valid=minimum_structured_json_valid,
        minimum_status_exact=minimum_status_exact,
        minimum_result_exact=minimum_result_exact,
        minimum_continuation_decision_exact=(
            minimum_continuation_decision_exact
        ),
        minimum_not_found_defer_accuracy=minimum_not_found_defer_accuracy,
        minimum_unnamed_id_resolved_accuracy=(
            minimum_unnamed_id_resolved_accuracy
        ),
    )
    output_dir = output_dir.resolve()
    write_json(output_dir / "exact_tool_report.json", report, private=True)
    write_json(output_dir / "test_metrics.json", metrics, private=False)
    if publish:
        receipt = publish_wandb(metrics, wandb_metric_prefix)
        write_json(output_dir / "wandb_test_run.json", receipt, private=False)
    return metrics


GATE_ARGUMENTS = (
    "minimum_valid_tool_trajectory",
    "minimum_reasoning_present",
    "minimum_exact_reasoning",
    "minimum_valid_single_tool_call",
    "minimum_exact_tool_argument",
    "minimum_family_payload_accuracy",
    "minimum_family_final_answer_accuracy",
    "minimum_ambiguous_defer_accuracy",
    "maximum_direct_answer_rate",
    "minimum_structured_json_valid",
    "minimum_status_exact",
    "minimum_result_exact",
    "minimum_continuation_decision_exact",
    "minimum_not_found_defer_accuracy",
    "minimum_unnamed_id_resolved_accuracy",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # The test config owns each gate floor; see the "Gate" section of the
    # S0 curriculum document.
    for name in GATE_ARGUMENTS:
        parser.add_argument(
            "--" + name.replace("_", "-"), type=float, required=True
        )
    parser.add_argument("--publish-wandb", action="store_true")
    parser.add_argument("--wandb-metric-prefix", default="test")
    return parser.parse_args()


def main() -> int:
    """Score one S0 full-registry test."""

    args = parse_args()
    metrics = evaluate_tool_trajectory_test(
        evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256,
        predictions_path=args.predictions,
        generation_manifest_path=args.generation_manifest,
        tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir,
        publish=args.publish_wandb,
        wandb_metric_prefix=args.wandb_metric_prefix,
        **{name: getattr(args, name) for name in GATE_ARGUMENTS},
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
