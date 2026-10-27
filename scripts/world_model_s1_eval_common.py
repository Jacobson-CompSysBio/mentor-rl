"""Shared, answer-key-free helpers for exact S1 generation and scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from runtime.world_model.s1_lookup_retry import diagnostic_retry_case, diagnostic_retry_tool
from runtime.world_model.s1_membership_retry import diagnostic_membership_retry_case


REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_ID = "world_model_v3_s1_coordinate_system"
EVALUATION_CONTRACT = "held_out_s1_multiplex_sft_v3"
LOSS_CONTRACT = "multiplex_sft_v3"
EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s1-evaluator-v1"
GENERATION_BUNDLE_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-generation-bundle-v1"
)
PREDICTION_SCHEMA_VERSION = "mentor-rl-world-model-s1-predictions-v1"
PREDICTION_SHARD_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-prediction-shard-v1"
)
GENERATION_PLAN_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-generation-plan-v1"
)
PARALLEL_GENERATION_PLAN_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-parallel-generation-plan-v1"
)
SHARD_STRATEGY = "strided_record_order_v1"

# The S1 v3 gate ("tiered_v3"). Protocol checks are mechanical; decision checks
# follow from the evidence in the transcript; generalization checks cover
# wording that training never showed. A check fails only when its 95% interval
# lies wholly on the wrong side of its target. The summary and blocked-reason
# sentences are never scored word for word, and only metric names that the
# runtime emits are scored.
TIERED_V3_GATE_PROFILE = "tiered_v3"
STRUCTURED_95_GATE_PROFILE = "structured_95_v1"
STRUCTURED_95_MINIMUM = 0.95
TIERED_V3_TARGETS = {
    "protocol": 0.995,
    "format": 0.99,
    "decision": 0.98,
    "generalization": 0.90,
}
TIERED_V3_MAXIMUM_DIRECT_ANSWER_RATE = 0.005

QUESTION_KEYS = (
    "record_id",
    "split",
    "messages",
    "tools",
    "metadata",
)
ANSWER_KEYS = ("record_id", "split", "expected_messages")
S1_FAMILIES = frozenset(
    {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
        "cross_context_alignment",
        "layer_tag_parsing",
        "layer_family_claims",
        "direction_conventions",
        "module_id_parsing",
    }
)
FAILURE_BLOCKED_REASON = "tool_error"
MEMBERSHIP_TOOL = "check_multiplex_membership"
V3_HARNESS_HEADER = "Context (from the harness):"
V3_USER_HEADER = "User request:"
V3_ANSWER_KEYS = (
    "status",
    "provenance",
    "result",
    "continuation_decision",
    "blocked_reason",
    "summary",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_PATTERN = re.compile(r"ev_(?:reg|mem)_[a-z0-9]{6}")
_TOOL_CALL_RE = re.compile(
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"to=functions\.([A-Za-z_][A-Za-z0-9_]*)"
    r"<\|channel\|>commentary json<\|message\|>"
    r"(\{.*?\})<\|call\|>\s*",
    re.DOTALL,
)
_FINAL_RE = re.compile(
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>"
    r"\s*(?:<\|start\|>assistant)?\s*"
    r"<\|channel\|>final<\|message\|>(.*?)\s*",
    re.DOTALL,
)


class S1EvaluationError(RuntimeError):
    """Report one invalid S1 evaluation input or generated transcript."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for an identity or JSONL record."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def ordered_json(value: Any) -> str:
    """Return compact JSON while preserving contract-significant key order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    """Return one required lowercase SHA-256 value."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise S1EvaluationError(f"{label} must be a lowercase SHA-256 value")
    return value


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S1EvaluationError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise S1EvaluationError(f"expected one JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per nonempty line."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise S1EvaluationError(
                        f"expected an object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise S1EvaluationError(f"cannot read JSONL {path}: {error}") from error
    return rows


def write_json(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    """Atomically write one stable, human-readable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600 if private else 0o640)


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    private: bool = False,
) -> None:
    """Atomically write stable JSONL without buffering the encoded corpus."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o755)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")
    temporary.replace(path)
    path.chmod(0o600 if private else 0o640)


def resolve_repo_path(
    repo_root: Path,
    value: Any,
    label: str,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve one declared path without allowing repository escape."""

    if not isinstance(value, str) or not value:
        raise S1EvaluationError(f"{label} must be a nonempty path")
    declared = Path(value)
    path = (declared if declared.is_absolute() else repo_root / declared).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise S1EvaluationError(f"{label} escapes the repository") from error
    if must_exist and not path.exists():
        raise S1EvaluationError(f"{label} is absent: {path}")
    return path


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def validate_answer_payload(
    value: Any,
    contract: Mapping[str, Any],
    *,
    require_order: bool = True,
) -> dict[str, Any]:
    """Validate and return one ordered six-field v3 final answer."""

    keys = list(contract["answer"]["keys"])
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise S1EvaluationError("the final answer fields changed")
    if require_order and list(value) != keys:
        raise S1EvaluationError("the final answer fields or field order changed")
    # The aggregate evaluator was originally serialized with sort_keys=True,
    # including nested earlier/final answers. Reconstruct the contract wire at
    # the answer-free boundary; generated text remains strictly order-checked.
    payload = {key: value[key] for key in keys}
    if payload.get("status") not in contract["vocabularies"]["status"]:
        raise S1EvaluationError("the final answer status is invalid")
    if payload.get("provenance") is not None and not isinstance(
        payload.get("provenance"), Mapping
    ):
        raise S1EvaluationError("final provenance must be an object or null")
    if not isinstance(payload.get("result"), Mapping):
        raise S1EvaluationError("the final result must be an object")
    decision = payload.get("continuation_decision")
    if decision not in contract["vocabularies"]["continuation_decision"]:
        raise S1EvaluationError("the continuation decision is invalid")
    blocked_reason = payload.get("blocked_reason")
    if blocked_reason is not None and (
        not isinstance(blocked_reason, str) or not blocked_reason.strip()
    ):
        raise S1EvaluationError("blocked_reason must be nonempty text or null")
    if (decision == "defer") != (blocked_reason is not None):
        raise S1EvaluationError(
            "defer requires blocked_reason and other decisions forbid it"
        )
    if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
        raise S1EvaluationError("the final summary must be nonempty text")
    return payload


def parse_answer_text(text: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Parse strict JSON and preserve the v3 answer's key order."""

    if not isinstance(text, str):
        raise S1EvaluationError("the final answer is not text")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise S1EvaluationError(
            f"the final answer is not strict JSON: {error}"
        ) from error
    return validate_answer_payload(value, contract)


def remove_terminal_tokens(text: str) -> str:
    """Remove only terminal GPT-OSS turn tokens."""

    result = text.strip()
    while result.endswith("<|return|>") or result.endswith("<|end|>"):
        token = "<|return|>" if result.endswith("<|return|>") else "<|end|>"
        result = result[: -len(token)].rstrip()
    return result


def parse_assistant_turn(
    text: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one GPT-OSS assistant turn as a tool call or final action."""

    if not isinstance(text, str):
        return None, "prediction_is_not_text"
    normalized = remove_terminal_tokens(text)
    call_match = _TOOL_CALL_RE.fullmatch(normalized)
    final_match = _FINAL_RE.fullmatch(normalized)
    if call_match is not None and final_match is not None:
        return None, "assistant_turn_is_ambiguous"
    action_match = call_match or final_match
    if action_match is not None and "<|channel|>" in action_match.group(1):
        # Retained analysis is rendered again after tools. Nested channel tags
        # are a malformed model action, not a reason to abort the whole shard.
        return None, "analysis_contains_channel_tag"
    if call_match is not None:
        thinking = call_match.group(1).strip()
        if not thinking:
            return None, "analysis_is_empty"
        try:
            arguments = json.loads(
                call_match.group(3),
                object_pairs_hook=_json_object_without_duplicates,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            return None, f"argument_json:{error}"
        if not isinstance(arguments, dict):
            return None, "arguments_are_not_an_object"
        return {
            "kind": "tool_call",
            "thinking": thinking,
            "name": call_match.group(2),
            "arguments": arguments,
        }, None
    if final_match is not None:
        thinking = final_match.group(1).strip()
        answer_text = final_match.group(2).strip()
        if not thinking:
            return None, "analysis_is_empty"
        if not answer_text:
            return None, "final_answer_is_empty"
        try:
            payload = parse_answer_text(answer_text, contract)
        except S1EvaluationError as error:
            return None, f"final_answer:{error}"
        return {
            "kind": "final",
            "thinking": thinking,
            "content": payload,
        }, None
    return None, "assistant_turn_format"


def enabled_s1_families(
    contract: Mapping[str, Any],
) -> dict[str, tuple[str, str]]:
    """Return question-family to (section, book-mode) for enabled S1 rows."""

    result: dict[str, tuple[str, str]] = {}
    families = contract["curriculum"]["families"]
    for section, settings in families.items():
        if not str(section).startswith("S1.") or not isinstance(settings, Mapping):
            continue
        if settings.get("enabled") is False:
            continue
        family = settings.get("question_family")
        book_mode = settings.get("book_mode")
        if isinstance(family, str) and isinstance(book_mode, str):
            result[family] = (str(section), book_mode)
    if set(result) != set(S1_FAMILIES):
        raise S1EvaluationError("the enabled S1 family contract changed")
    return result


def _sentinel_answer(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shape-valid answer used only to validate an open prefix."""

    values = {
        "status": str(contract["vocabularies"]["status"][0]),
        "provenance": None,
        "result": {},
        "continuation_decision": "stop",
        "blocked_reason": None,
        "summary": "prefix validation sentinel",
    }
    return {key: values[key] for key in contract["answer"]["keys"]}


def validate_question(
    row: Any,
    contract: Mapping[str, Any],
    *,
    expected_split: str = "test",
) -> dict[str, Any]:
    """Validate one native S1 answer-free question row."""

    if expected_split not in {"test", "validation"}:
        raise S1EvaluationError("S1 generation requires a test or validation split")
    if not isinstance(row, Mapping) or set(row) != set(QUESTION_KEYS):
        raise S1EvaluationError("an S1 question has an unsafe field set")
    record_id = row.get("record_id")
    messages = row.get("messages")
    tools = row.get("tools")
    metadata = row.get("metadata")
    if not isinstance(record_id, str) or not record_id:
        raise S1EvaluationError("an S1 question has an invalid record_id")
    if row.get("split") != expected_split:
        raise S1EvaluationError(f"{record_id} is not a {expected_split} question")
    if not isinstance(messages, list) or not messages:
        raise S1EvaluationError(f"{record_id} has no message prefix")
    if not isinstance(tools, list) or not isinstance(metadata, Mapping):
        raise S1EvaluationError(f"{record_id} lacks tools or metadata")

    from runtime.world_model.contract import system_prompt_sha256, tool_definitions
    from runtime.world_model_training import _validate_v3_message_sequence

    expected_prompt_hash = system_prompt_sha256(contract)
    if (
        metadata.get("step") != "S1"
        or metadata.get("schema_version")
        != contract["runtime"]["row_schema_version"]
        or metadata.get("tool_contract_version")
        != contract["runtime"]["tool_contract_version"]
        or metadata.get("system_prompt_sha256") != expected_prompt_hash
    ):
        raise S1EvaluationError(f"{record_id} metadata is outside the S1 contract")
    family = metadata.get("question_family")
    family_contracts = enabled_s1_families(contract)
    if family not in family_contracts:
        raise S1EvaluationError(f"{record_id} has an unsupported S1 family")
    section, book_mode = family_contracts[str(family)]
    source_trace = metadata.get("source_trace")
    if (
        metadata.get("book_mode") != book_mode
        or not isinstance(source_trace, str)
        or not source_trace.startswith(section + ".")
    ):
        raise S1EvaluationError(f"{record_id} family metadata changed")
    if messages[0] != {
        "role": "system",
        "content": contract["runtime"]["system_prompt"],
    }:
        raise S1EvaluationError(f"{record_id} has a changed system prompt")
    if tools != tool_definitions(contract):
        raise S1EvaluationError(f"{record_id} has changed tool definitions")
    if messages[-1].get("role") != "user":
        raise S1EvaluationError(f"{record_id} does not end at the current user turn")
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        if normalized.get("role") == "assistant" and "content" in normalized:
            normalized["content"] = validate_answer_payload(
                normalized["content"], contract, require_order=False
            )
        normalized_messages.append(normalized)
    complete = [dict(message) for message in normalized_messages]
    complete.append(
        {
            "role": "assistant",
            "thinking": "",
            "content": _sentinel_answer(contract),
        }
    )
    try:
        _validate_v3_message_sequence(complete)
    except (TypeError, ValueError) as error:
        raise S1EvaluationError(
            f"{record_id} has an invalid native S1 prefix: {error}"
        ) from error
    result = {str(key): value for key, value in row.items()}
    result["messages"] = normalized_messages
    return result


def validate_answer_row(
    row: Any,
    question: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_split: str = "test",
) -> dict[str, Any]:
    """Validate one private S1 answer suffix against its question prefix."""

    if expected_split not in {"test", "validation"}:
        raise S1EvaluationError("S1 scoring requires a test or validation split")
    if not isinstance(row, Mapping) or set(row) != set(ANSWER_KEYS):
        raise S1EvaluationError("an S1 answer row has an invalid field set")
    if (
        row.get("record_id") != question.get("record_id")
        or row.get("split") != expected_split
        or question.get("split") != expected_split
    ):
        raise S1EvaluationError("an S1 answer row does not match its question")
    expected = row.get("expected_messages")
    if not isinstance(expected, list) or not expected:
        raise S1EvaluationError("an S1 answer row has no expected suffix")
    from runtime.world_model_training import _validate_v3_message_sequence

    normalized_expected = [dict(message) for message in expected]
    normalized_expected[-1]["content"] = validate_answer_payload(
        normalized_expected[-1].get("content"),
        contract,
        require_order=False,
    )
    complete = [dict(message) for message in question["messages"]]
    complete.extend(normalized_expected)
    try:
        _validate_v3_message_sequence(complete)
    except (TypeError, ValueError) as error:
        raise S1EvaluationError(
            f"the S1 answer suffix is invalid for {row['record_id']}: {error}"
        ) from error
    result = {str(key): value for key, value in row.items()}
    result["expected_messages"] = normalized_expected
    return result


def model_messages(question: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert the native S1 prefix to the tokenizer's GPT-OSS message wire."""

    result: list[dict[str, Any]] = []
    harness_content: str | None = None
    for source in question["messages"]:
        role = source.get("role")
        if role == "harness":
            harness_content = str(source["content"])
        elif role == "system":
            result.append({"role": "system", "content": source["content"]})
        elif role == "user":
            content = str(source["content"])
            if harness_content is not None:
                content = (
                    f"{V3_HARNESS_HEADER}\n{harness_content}\n\n"
                    f"{V3_USER_HEADER}\n{content}"
                )
                harness_content = None
            result.append({"role": "user", "content": content})
        elif role == "assistant" and "content" in source:
            # Earlier-turn thinking is masked during S1 training and must not
            # become a hidden test-time hint.
            result.append(
                {
                    "role": "assistant",
                    "content": ordered_json(
                        {
                            key: source["content"][key]
                            for key in V3_ANSWER_KEYS
                        }
                    ),
                }
            )
        elif role == "assistant" and "tool_call" in source:
            call = source["tool_call"]
            result.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                    ],
                }
            )
        elif role == "tool":
            result.append(
                {
                    "role": "tool",
                    "name": source["name"],
                    "content": source["content"],
                }
            )
        else:
            raise S1EvaluationError("the S1 question has an unsupported message")
    if harness_content is not None:
        raise S1EvaluationError("a harness block has no following user request")
    return result


def failure_injection_required(
    question: Mapping[str, Any],
    *,
    tool_name: str,
    call_index: int,
    contract: Mapping[str, Any],
) -> bool:
    """Apply the answer-free, contract-declared failure harness for one call."""

    membership_case = diagnostic_membership_retry_case(str(question.get("record_id", "")))
    if membership_case is not None:
        return tool_name == "check_multiplex_membership" and (
            call_index == 1 or (membership_case == "exhausted" and call_index == 2)
        )
    retry_case = diagnostic_retry_case(str(question.get("record_id", "")))
    if retry_case is not None:
        return tool_name == diagnostic_retry_tool(str(question["record_id"])) and (
            call_index == 0 or (retry_case == "exhausted" and call_index == 1)
        )
    metadata = question.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get(
        "blocked_reason_code"
    ) != FAILURE_BLOCKED_REASON:
        return False
    source_trace = metadata.get("source_trace")
    if not isinstance(source_trace, str):
        raise S1EvaluationError("a failure row has no source_trace")
    parts = source_trace.split(".")
    section = ".".join(parts[:2])
    settings = contract["curriculum"]["families"].get(section)
    policy = (
        settings.get("failure_injection")
        if isinstance(settings, Mapping)
        else None
    )
    if not isinstance(policy, Mapping):
        raise S1EvaluationError(f"{source_trace} has no failure-injection policy")
    call_stage = policy.get("call_stage")
    if call_stage == "first":
        return call_index == 0
    if call_stage == "membership":
        return tool_name == MEMBERSHIP_TOOL
    raise S1EvaluationError(f"unsupported failure call stage {call_stage!r}")


def normalize_evidence_ids(value: Any) -> Any:
    """Replace only shape-valid per-call evidence nonces for exact scoring."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                "<evidence_id>"
                if key == "evidence_id"
                and isinstance(item, str)
                and _EVIDENCE_ID_PATTERN.fullmatch(item) is not None
                else normalize_evidence_ids(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_evidence_ids(item) for item in value]
    return value


def message_shape(messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the role/action grammar of one current-turn suffix."""

    return [
        (
            "assistant_tool_call"
            if message.get("role") == "assistant" and "tool_call" in message
            else str(message.get("role"))
        )
        for message in messages
    ]


def tool_calls(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return canonical tool calls from one native S1 suffix."""

    return [
        dict(message["tool_call"])
        for message in messages
        if message.get("role") == "assistant" and "tool_call" in message
    ]


def tool_payloads(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return canonical named tool results from one native S1 suffix."""

    return [
        {"name": message.get("name"), "content": message.get("content")}
        for message in messages
        if message.get("role") == "tool"
    ]


def final_message(messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the terminal assistant answer when present."""

    if not messages:
        return None
    message = messages[-1]
    if message.get("role") != "assistant" or "tool_call" in message:
        return None
    return message


def reasoning_present(messages: Sequence[Mapping[str, Any]]) -> bool:
    """Require nonempty thinking on every generated assistant action."""

    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    return bool(assistant_messages) and all(
        isinstance(message.get("thinking"), str)
        and bool(str(message["thinking"]).strip())
        for message in assistant_messages
    )
