#!/usr/bin/env python3
"""Generate native, answer-key-free S1 trajectories for exact evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model.source_index import (  # noqa: E402
    SourceIndexError,
    check_source_index,
    content_sha256,
)
from runtime.world_model.tools import (  # noqa: E402
    MEMBERSHIP_TOOL,
    ToolCallError,
    ToolCallResult,
    WorldModelToolRuntime,
)
from runtime.world_model_training import (  # noqa: E402
    renderable_v3_tools,
    tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.run_world_model_v2_s0_exact_generation import (  # noqa: E402
    generate_prompt_batch_with_oom_backoff,
    load_backend,
    validate_checkpoint,
)
from scripts.world_model_s1_eval_common import (  # noqa: E402
    DATASET_ID,
    EVALUATION_CONTRACT,
    GENERATION_BUNDLE_SCHEMA_VERSION,
    GENERATION_PLAN_SCHEMA_VERSION,
    LOSS_CONTRACT,
    PARALLEL_GENERATION_PLAN_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    PREDICTION_SHARD_SCHEMA_VERSION,
    SHARD_STRATEGY,
    S1EvaluationError,
    S1_FAMILIES,
    canonical_json,
    failure_injection_required,
    model_messages,
    parse_assistant_turn,
    read_json,
    read_jsonl,
    require_sha256,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    validate_question,
    write_json,
)


MAX_TOOL_CALLS = 2
HISTORY_POLICIES = frozenset({"strip", "retain_current_turn"})
DEFAULT_HISTORY_POLICY = "retain_current_turn"
EVIDENCE_ID_POLICY = "full_panel_reverse_index_call_ordinal_base36_v2"
NETWORK_POLICY = "python_audit_deny_outbound_socket_v1"
EVIDENCE_NAMESPACE_SIZE = 36**6
PARTIAL_PREDICTION_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-partial-prediction-shard-v1"
)
GPT_OSS_ASSISTANT_BOUNDARY = "<|start|>assistant"
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
ALLOWED_GENERATION_SPECIAL_TOKENS = frozenset(
    {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|call|>",
        "<|end|>",
        "<|return|>",
    }
)
DENIED_NETWORK_AUDIT_EVENTS = frozenset(
    {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.sendto",
    }
)
_NETWORK_GUARD_INSTALLED = False


class S1GenerationError(S1EvaluationError):
    """Report one invalid S1 generation input or runtime result."""


def deny_outbound_network_audit_event(
    event: str,
    arguments: tuple[Any, ...],
) -> None:
    """Abort when audited Python code attempts outbound network access."""

    del arguments
    if event in DENIED_NETWORK_AUDIT_EVENTS:
        raise S1GenerationError(
            f"S1 offline generation blocked network audit event {event!r}"
        )


def install_network_guard() -> str:
    """Install the irreversible process-wide S1 outbound-network guard."""

    global _NETWORK_GUARD_INSTALLED
    if not _NETWORK_GUARD_INSTALLED:
        sys.addaudithook(deny_outbound_network_audit_event)
        _NETWORK_GUARD_INSTALLED = True
    return NETWORK_POLICY


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value < 0:
        raise S1GenerationError("an evidence ordinal cannot be negative")
    digits = "0" if value == 0 else ""
    while value:
        value, remainder = divmod(value, len(alphabet))
        digits = alphabet[remainder] + digits
    return digits


def maximum_tool_calls(record_id: str) -> int:
    """Return the frozen panel's tool-call limit."""
    return MAX_TOOL_CALLS


def deterministic_evidence_id(
    *,
    channel: str,
    record_index: int,
    call_index: int,
    record_id: str = "",
) -> str:
    """Return a collision-free ID independent of shard and execution order."""

    if channel not in {"registry", "graph"}:
        raise S1GenerationError(f"unsupported evidence channel {channel!r}")
    if (
        not isinstance(record_index, int)
        or isinstance(record_index, bool)
        or record_index < 0
        or not isinstance(call_index, int)
        or isinstance(call_index, bool)
        or call_index < 0
        or call_index >= maximum_tool_calls(record_id)
    ):
        raise S1GenerationError("the deterministic evidence coordinate is invalid")
    coordinate = record_index * MAX_TOOL_CALLS + call_index
    if coordinate >= EVIDENCE_NAMESPACE_SIZE:
        raise S1GenerationError("the S1 panel exceeds the evidence ID namespace")
    # Builders allocate from the bottom; evaluation allocates from the top.
    ordinal = EVIDENCE_NAMESPACE_SIZE - 1 - coordinate
    suffix = _base36(ordinal)
    prefix = "ev_reg_" if channel == "registry" else "ev_mem_"
    return prefix + suffix.rjust(6, "0")


def _evidence_ids(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            if key == "evidence_id" and isinstance(item, str):
                result.append(item)
            else:
                result.extend(_evidence_ids(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_evidence_ids(item))
        return result
    if isinstance(value, str):
        return re.findall(r"ev_(?:reg|mem)_[a-z0-9]{6}", value)
    return []


def validate_evidence_namespace(rows: Sequence[Mapping[str, Any]]) -> None:
    """Prove generated IDs are unique and disjoint from visible evidence."""

    occupied = [
        evidence_id
        for row in rows
        for evidence_id in _evidence_ids(row.get("messages"))
    ]
    generated = [
        deterministic_evidence_id(
            channel=channel,
            record_index=record_index,
            call_index=call_index, record_id=str(row["record_id"]),
        )
        for record_index, row in enumerate(rows)
        for call_index in range(maximum_tool_calls(str(row["record_id"])))
        for channel in ("registry", "graph")
    ]
    if len(set(generated)) != len(generated):
        raise S1GenerationError("generated S1 evidence IDs collide")
    overlap = set(occupied) & set(generated)
    if overlap:
        raise S1GenerationError(
            "generated S1 evidence IDs overlap visible context: "
            + ", ".join(sorted(overlap)[:5])
        )


class DeterministicEvidenceRuntime:
    """Attach panel-coordinate evidence nonces to one shared local runtime."""

    def __init__(
        self,
        runtime: WorldModelToolRuntime,
        record_indexes: Mapping[str, int],
    ) -> None:
        self._runtime = runtime
        self._record_indexes = dict(record_indexes)
        self._pending: tuple[str, str] | None = None

    @classmethod
    def from_source_index(
        cls,
        path: Path,
        *,
        contract: Mapping[str, Any],
        record_indexes: Mapping[str, int],
    ) -> "DeterministicEvidenceRuntime":
        holder: list[DeterministicEvidenceRuntime] = []

        def evidence_id_factory(channel: str) -> str:
            if not holder:
                raise ToolCallError("the deterministic evidence runtime is unbound")
            return holder[0]._take_evidence_id(channel)

        runtime = WorldModelToolRuntime.from_source_index(
            path,
            contract=contract,
            evidence_id_factory=evidence_id_factory,
        )
        result = cls(runtime, record_indexes)
        holder.append(result)
        return result

    def _take_evidence_id(self, channel: str) -> str:
        pending = self._pending
        if pending is None or pending[0] != channel:
            raise ToolCallError("the deterministic evidence context changed")
        self._pending = None
        return pending[1]

    def execute_s1(
        self,
        *,
        record_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        call_index: int,
        inject_backend_failure: bool,
    ) -> Any:
        try:
            record_index = self._record_indexes[record_id]
        except KeyError as error:
            raise ToolCallError(f"unknown S1 generation record {record_id}") from error
        channel = "graph" if tool_name == MEMBERSHIP_TOOL else "registry"
        evidence_id = deterministic_evidence_id(
            channel=channel,
            record_index=record_index,
            call_index=call_index, record_id=record_id,
        )
        if self._pending is not None:
            raise ToolCallError("an evidence call is already active")
        self._pending = (channel, evidence_id)
        try:
            result = self._runtime.execute(
                tool_name,
                arguments,
                inject_backend_failure=inject_backend_failure,
            )
            return result
        finally:
            # Failure injection and invalid arguments intentionally allocate
            # no evidence block, so the callback may not have consumed it.
            self._pending = None


def write_prediction_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write predictions without sorting contract-significant final fields."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(path)
    path.chmod(0o600)


@dataclass(slots=True)
class TrajectoryState:
    """Mutable answer-free state for one current S1 user turn."""

    question: dict[str, Any]
    prompt_messages: list[dict[str, Any]]
    generated_messages: list[dict[str, Any]] = field(default_factory=list)
    generations: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    trajectory_error: str | None = None
    executed_tool_calls: int = 0
    injected_failure_count: int = 0
    history_policy: str = DEFAULT_HISTORY_POLICY

    def __post_init__(self) -> None:
        """Reject unknown history policies before generating any action."""

        if self.history_policy not in HISTORY_POLICIES:
            raise S1GenerationError(f"unknown S1 history policy: {self.history_policy!r}")

    @property
    def active(self) -> bool:
        return self.stop_reason is None


TurnGenerator = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], int],
    str | Mapping[str, Any],
]


def _bundle_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in manifest.items()
        if key != "bundle_sha256"
    }


def load_generation_bundle(
    bundle_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    """Load and fully verify one answer-free native S1 generation bundle."""

    bundle_root = bundle_root.resolve()
    repo_root = repo_root.resolve()
    manifest = read_json(bundle_root / "manifest.json")
    claimed = manifest.get("bundle_sha256")
    if (
        manifest.get("schema_version") != GENERATION_BUNDLE_SCHEMA_VERSION
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or manifest.get("loss_contract") != LOSS_CONTRACT
        or manifest.get("native_transcript") is not True
        or manifest.get("reads_private_answer_keys") is not False
        or not isinstance(claimed, str)
        or stable_sha256(_bundle_identity(manifest)) != claimed
    ):
        raise S1GenerationError("the S1 generation bundle identity changed")
    failure_harness = manifest.get("failure_harness")
    if (
        not isinstance(failure_harness, Mapping)
        or failure_harness.get("metadata_visible_to_model") is not False
    ):
        raise S1GenerationError("the S1 failure harness is unsafe")

    contract_path = resolve_repo_path(
        repo_root,
        manifest.get("contract", {}).get("path")
        if isinstance(manifest.get("contract"), Mapping)
        else None,
        "S1 runtime contract",
    )
    contract_pin = manifest.get("contract")
    if (
        not isinstance(contract_pin, Mapping)
        or sha256_file(contract_path) != contract_pin.get("sha256")
    ):
        raise S1GenerationError("the bundled S1 runtime contract changed")
    contract = load_contract(contract_path)

    questions_path = bundle_root / "questions.jsonl"
    if sha256_file(questions_path) != manifest.get("questions_sha256"):
        raise S1GenerationError("the bundled S1 questions changed")
    rows = [validate_question(row, contract) for row in read_jsonl(questions_path)]
    record_ids = [str(row["record_id"]) for row in rows]
    family_counts = Counter(
        str(row["metadata"]["question_family"]) for row in rows
    )
    book_mode_counts = Counter(str(row["metadata"]["book_mode"]) for row in rows)
    failure_count = sum(
        row["metadata"].get("blocked_reason_code") == "tool_error"
        for row in rows
    )
    if "diagnostic_only" in manifest or "diagnostic_kind" in manifest:
        raise S1GenerationError("diagnostic panels are not supported")
    expected_families = set(S1_FAMILIES)
    if (
        not rows
        or len(rows) != manifest.get("record_count")
        or len(set(record_ids)) != len(record_ids)
        or stable_sha256(record_ids) != manifest.get("record_ids_sha256")
        or dict(sorted(family_counts.items())) != manifest.get("family_counts")
        or dict(sorted(book_mode_counts.items()))
        != manifest.get("book_mode_counts")
        or failure_count != manifest.get("failure_row_count")
        or set(family_counts) != expected_families
    ):
        raise S1GenerationError("the bundled S1 record set changed")
    validate_evidence_namespace(rows)

    source_index = manifest.get("source_index")
    if not isinstance(source_index, Mapping):
        raise S1GenerationError("the S1 bundle has no source-index identity")
    source_index_sha256 = require_sha256(
        source_index.get("sha256"), "S1 source-index identity"
    )
    if source_index.get("id") != f"sha256:{source_index_sha256}":
        raise S1GenerationError("the S1 source-index ID changed")
    source_index_path = resolve_repo_path(
        repo_root,
        source_index.get("path"),
        "S1 source index",
    )
    try:
        payload = check_source_index(source_index_path, contract)
    except SourceIndexError as error:
        raise S1GenerationError(f"the S1 source index changed: {error}") from error
    if content_sha256(payload) != source_index_sha256:
        raise S1GenerationError("the S1 source-index content identity changed")
    return manifest, rows, source_index_path


def clean_decoded_generation(
    raw: str,
    *,
    eos_token: str | None = None,
    pad_token: str | None = None,
) -> tuple[str, list[str]]:
    """Strip terminal tokens and report any foreign special token."""

    terminal = {"<|return|>", "<|end|>"}
    if eos_token:
        terminal.add(str(eos_token))
    if pad_token:
        terminal.add(str(pad_token))
    allowed = set(ALLOWED_GENERATION_SPECIAL_TOKENS) | terminal
    disallowed = sorted(
        {
            token
            for token in SPECIAL_TOKEN_RE.findall(raw)
            if token not in allowed
        }
    )
    result = raw.strip()
    ordered_terminal = sorted(terminal, key=len, reverse=True)
    while True:
        for token in ordered_terminal:
            if token and result.endswith(token):
                result = result[: -len(token)].rstrip()
                break
        else:
            break
    return result, disallowed


def _new_state(
    question: Mapping[str, Any], *, history_policy: str = DEFAULT_HISTORY_POLICY,
) -> TrajectoryState:
    """Start with public context; earlier user turns never supply reasoning."""

    return TrajectoryState(
        question={str(key): value for key, value in question.items()},
        prompt_messages=model_messages(question),
        history_policy=history_policy,
    )


def _generation_details(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "raw_generation": value,
            "prompt_tokens": None,
            "generated_tokens": None,
            "action_tokens": None,
        }
    raw = value.get("raw_generation")
    if not isinstance(raw, str):
        raise S1GenerationError("a generation result lacks raw text")
    return {
        "raw_generation": raw,
        "prompt_tokens": value.get("prompt_tokens"),
        "generated_tokens": value.get("generated_tokens"),
        "action_tokens": value.get("action_tokens"),
    }


def _append_tool_prompt_messages(
    state: TrajectoryState,
    *,
    name: str,
    arguments: Mapping[str, Any],
    payload: Mapping[str, Any],
    thinking: str,
) -> None:
    """Append executed calls, optionally retaining their generated reasoning."""

    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": name, "arguments": dict(arguments)},
            }
        ],
    }
    if state.history_policy == "retain_current_turn":
        assistant["thinking"] = thinking
    state.prompt_messages.extend(
        [
            assistant,
            {"role": "tool", "name": name, "content": dict(payload)},
        ]
    )


def advance_trajectory(
    state: TrajectoryState,
    generation: str | Mapping[str, Any],
    *,
    runtime: Any,
    contract: Mapping[str, Any],
    eos_token: str | None = None,
    pad_token: str | None = None,
    max_tool_call_tokens: int | None = None,
) -> None:
    """Parse and apply one generated assistant action to an S1 state."""

    if not state.active:
        raise S1GenerationError("cannot advance a completed S1 trajectory")
    details = _generation_details(generation)
    encoded, disallowed = clean_decoded_generation(
        details["raw_generation"],
        eos_token=eos_token,
        pad_token=pad_token,
    )
    parsed, parse_error = parse_assistant_turn(encoded, contract)
    diagnostic = {
        "turn_index": len(state.generations),
        "raw_generation": details["raw_generation"],
        "encoded_generation": encoded,
        "prompt_tokens": details["prompt_tokens"],
        "generated_tokens": details["generated_tokens"],
        "action_tokens": details["action_tokens"],
        "disallowed_special_tokens": disallowed,
        "parse_error": parse_error,
    }
    state.generations.append(diagnostic)
    if disallowed:
        state.stop_reason = "parse_error"
        state.trajectory_error = "disallowed_special_tokens"
        return
    if parsed is None or parse_error is not None:
        state.stop_reason = "parse_error"
        state.trajectory_error = parse_error or "assistant_turn_parse_error"
        return

    thinking = str(parsed["thinking"])
    if parsed["kind"] == "final":
        state.generated_messages.append(
            {
                "role": "assistant",
                "thinking": thinking,
                "content": dict(parsed["content"]),
            }
        )
        state.stop_reason = "final"
        return

    name = str(parsed["name"])
    arguments = dict(parsed["arguments"])
    generated_tokens = details.get("action_tokens")
    if (
        max_tool_call_tokens is not None
        and isinstance(generated_tokens, int)
        and not isinstance(generated_tokens, bool)
        and generated_tokens > max_tool_call_tokens
    ):
        state.stop_reason = "tool_call_token_limit"
        state.trajectory_error = "max_tool_call_tokens_exceeded"
        return
    state.generated_messages.append(
        {
            "role": "assistant",
            "thinking": thinking,
            "tool_call": {"name": name, "arguments": arguments},
        }
    )
    attempted_call_count = sum(
        message.get("role") == "assistant" and "tool_call" in message
        for message in state.generated_messages
    )
    limit = maximum_tool_calls(str(state.question["record_id"]))
    if attempted_call_count > limit:
        state.stop_reason = "tool_call_limit"
        state.trajectory_error = "more_than_two_tool_calls" if limit == 2 else "more_than_three_tool_calls"
        return

    inject_failure = failure_injection_required(
        state.question,
        tool_name=name,
        call_index=state.executed_tool_calls,
        contract=contract,
    )
    try:
        execute_s1 = getattr(runtime, "execute_s1", None)
        if callable(execute_s1):
            result = execute_s1(
                record_id=str(state.question["record_id"]),
                tool_name=name,
                arguments=arguments,
                call_index=state.executed_tool_calls,
                inject_backend_failure=inject_failure,
            )
        else:
            result = runtime.execute(
                name,
                arguments,
                inject_backend_failure=inject_failure,
            )
    except ToolCallError as error:
        state.stop_reason = "tool_execution_error"
        state.trajectory_error = str(error)
        return
    payload = dict(result.output)
    state.generated_messages.append(
        {"role": "tool", "name": name, "content": payload}
    )
    _append_tool_prompt_messages(
        state,
        name=name,
        arguments=arguments,
        payload=payload,
        thinking=thinking,
    )
    state.executed_tool_calls += 1
    if inject_failure:
        state.injected_failure_count += 1


def prediction_from_state(
    state: TrajectoryState,
    *,
    network_guarded: bool = False,
) -> dict[str, Any]:
    """Serialize one completed or diagnostically stopped trajectory."""

    if network_guarded and not _NETWORK_GUARD_INSTALLED:
        raise S1GenerationError(
            "cannot claim offline execution without the S1 network guard"
        )
    attempted_calls = sum(
        message.get("role") == "assistant" and "tool_call" in message
        for message in state.generated_messages
    )
    return {
        "record_id": state.question["record_id"],
        "generated_messages": state.generated_messages,
        "generations": state.generations,
        "stop_reason": state.stop_reason or "generation_limit",
        "trajectory_error": state.trajectory_error,
        "tool_call_count": attempted_calls,
        "executed_tool_call_count": state.executed_tool_calls,
        "injected_failure_count": state.injected_failure_count,
        "network_used_false": network_guarded,
    }


def orchestrate_s1_trajectory(
    question: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    runtime: Any,
    generate_turn: TurnGenerator,
    history_policy: str = DEFAULT_HISTORY_POLICY,
    expected_split: str = "test",
) -> dict[str, Any]:
    """Run a native S1 trajectory with the panel-specific tool-call limit."""

    validated = validate_question(question, contract, expected_split=expected_split)
    state = _new_state(validated, history_policy=history_policy)
    for turn_index in range(maximum_tool_calls(str(validated["record_id"])) + 1):
        generation = generate_turn(
            tuple(dict(message) for message in state.prompt_messages),
            tuple(dict(tool) for tool in validated["tools"]),
            turn_index,
        )
        advance_trajectory(
            state,
            generation,
            runtime=runtime,
            contract=contract,
        )
        if not state.active:
            break
    if state.active:
        state.stop_reason = "generation_limit"
        state.trajectory_error = "terminal_answer_not_generated"
    return prediction_from_state(state)


def _render_prompt(
    state: TrajectoryState,
    *,
    tokenizer: Any,
    reasoning_effort: str,
) -> str:
    prompt = tokenizer.apply_chat_template(
        state.prompt_messages,
        tools=renderable_v3_tools(state.question["tools"]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )
    if not isinstance(prompt, str) or not prompt.endswith(
        GPT_OSS_ASSISTANT_BOUNDARY
    ):
        raise S1GenerationError(
            "the S1 prompt lacks its GPT-OSS assistant boundary"
        )
    return prompt


def _prompt_token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False)
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        if len(values) != 1:
            raise S1GenerationError("prompt tokenization returned a batch")
        values = values[0]
    if not isinstance(values, list):
        raise S1GenerationError("prompt tokenization returned no input IDs")
    return len(values)


def generate_trajectory_batch(
    questions: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    torch_module: Any,
    runtime: Any,
    contract: Mapping[str, Any],
    max_new_tokens: int,
    max_total_tokens: int,
    max_tool_call_tokens: int,
    max_final_prompt_tokens: int,
    reasoning_effort: str,
    shard_index: int,
    network_guarded: bool = False,
    history_policy: str = DEFAULT_HISTORY_POLICY,
    expected_split: str = "test",
) -> list[dict[str, Any]]:
    """Generate a batch using each record namespace's local tool-call limit."""

    if network_guarded and not _NETWORK_GUARD_INSTALLED:
        raise S1GenerationError("the S1 network guard was not installed")
    states = [
        _new_state(
            validate_question(row, contract, expected_split=expected_split),
            history_policy=history_policy,
        )
        for row in questions
    ]
    for turn_index in range(max((maximum_tool_calls(str(s.question["record_id"])) for s in states), default=MAX_TOOL_CALLS) + 1):
        active: list[tuple[TrajectoryState, str]] = []
        for state in states:
            if not state.active:
                continue
            prompt = _render_prompt(
                state,
                tokenizer=tokenizer,
                reasoning_effort=reasoning_effort,
            )
            if (
                state.executed_tool_calls > 0
                and _prompt_token_count(tokenizer, prompt)
                > max_final_prompt_tokens
            ):
                state.stop_reason = "prompt_limit"
                state.trajectory_error = "max_final_prompt_tokens_exceeded"
                continue
            active.append((state, prompt))
        if not active:
            break
        stage = f"assistant_turn_{turn_index + 1}"
        generations = generate_prompt_batch_with_oom_backoff(
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_module=torch_module,
            prompts=[prompt for _, prompt in active],
            record_ids=[str(state.question["record_id"]) for state, _ in active],
            # Any S1 pass may legally be a direct/final action. Generate to
            # the final-answer budget, then enforce the tighter call budget
            # only when the parsed action is in fact a tool call.
            max_new_tokens=max_new_tokens,
            max_total_tokens=max_total_tokens,
            stage=stage,
            shard_index=shard_index,
        )
        for (state, _), generation in zip(active, generations, strict=True):
            generation = dict(generation)
            encoded, _ = clean_decoded_generation(
                str(generation["raw_generation"]),
                eos_token=getattr(tokenizer, "eos_token", None),
                pad_token=getattr(tokenizer, "pad_token", None),
            )
            generation["action_tokens"] = _prompt_token_count(
                tokenizer, encoded
            )
            advance_trajectory(
                state,
                generation,
                runtime=runtime,
                contract=contract,
                eos_token=getattr(tokenizer, "eos_token", None),
                pad_token=getattr(tokenizer, "pad_token", None),
                max_tool_call_tokens=max_tool_call_tokens,
            )
    for state in states:
        if state.active:
            state.stop_reason = "generation_limit"
            state.trajectory_error = "terminal_answer_not_generated"
    return [
        prediction_from_state(state, network_guarded=network_guarded)
        for state in states
    ]


def select_shard_rows(
    rows: Sequence[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise S1GenerationError("the shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise S1GenerationError("the shard index is outside the shard count")
    return list(rows[shard_index::shard_count])


def validate_tokenizer_artifact_binding(
    checkpoint_path: Path,
    tokenizer_files: Mapping[str, str],
) -> str:
    """Bind inference prompt files to the tokenizer used by the checkpoint."""

    artifact_sha256 = stable_sha256(tokenizer_files)
    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("tokenizer_artifacts_sha256") != artifact_sha256
    ):
        raise S1GenerationError(
            "the checkpoint and inference tokenizer artifacts differ"
        )
    return artifact_sha256


def validate_base_model_artifact_binding(
    base_model_path: Path,
    expected_sha256: str,
) -> str:
    """Verify the staged model receipt and marker bind the full artifact."""

    expected_sha256 = require_sha256(
        expected_sha256, "S1 base-model artifact"
    )
    marker_path = base_model_path / ".mentor_model_identity_sha256"
    receipt_path = base_model_path / ".mentor_base_model_artifact.json"
    try:
        marker = marker_path.read_text(encoding="utf-8").strip()
        receipt = read_json(receipt_path)
    except (OSError, S1EvaluationError) as error:
        raise S1GenerationError(
            "the staged base-model identity receipt is absent"
        ) from error
    claimed = receipt.get("base_model_artifact_sha256")
    receipt_identity = {
        key: value
        for key, value in receipt.items()
        if key != "base_model_artifact_sha256"
    }
    if (
        marker != expected_sha256
        or claimed != expected_sha256
        or stable_sha256(receipt_identity) != expected_sha256
    ):
        raise S1GenerationError("the staged base-model artifact changed")
    return expected_sha256


def _plan_manifest_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "evaluation_contract",
        "test_panel_id",
        "method_id",
        "train_run_id",
        "base_model_identity_sha256",
        "base_model_artifact_sha256",
        "checkpoint_identity_sha256",
        "tokenizer_manifest_sha256",
        "tokenizer_artifacts_sha256",
        "generation_bundle_sha256",
        "source_index_sha256",
        "generation_config",
        "generation_config_sha256",
        "reads_private_answer_keys",
    )
    return {key: plan[key] for key in keys}


def _validated_shard_identity(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one complete shard identity and reject cross-shard drift."""

    try:
        identity = _plan_manifest_fields(manifest)
        for key in (
            "base_model_identity_sha256",
            "base_model_artifact_sha256",
            "checkpoint_identity_sha256",
            "tokenizer_manifest_sha256",
            "tokenizer_artifacts_sha256",
            "generation_bundle_sha256",
            "source_index_sha256",
            "generation_config_sha256",
        ):
            require_sha256(identity[key], f"S1 shard {key}")
    except (KeyError, S1EvaluationError) as error:
        raise S1GenerationError("an S1 shard identity is incomplete") from error
    if stable_sha256(identity["generation_config"]) != identity[
        "generation_config_sha256"
    ]:
        raise S1GenerationError("an S1 shard generation config changed")
    if identity["generation_config"].get("history_policy", "strip") not in HISTORY_POLICIES:
        raise S1GenerationError("an S1 shard uses an unknown history policy")
    if expected is not None and identity != dict(expected):
        raise S1GenerationError("the S1 shards use different generation inputs")
    return identity


def _write_partial_checkpoint(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    predictions_path = output_dir / "partial_predictions.jsonl"
    write_prediction_jsonl(predictions_path, predictions)
    ids = [str(row["record_id"]) for row in predictions]
    manifest = {
        "schema_version": PARTIAL_PREDICTION_SCHEMA_VERSION,
        "generation_plan_sha256": stable_sha256(plan),
        "history_policy": plan["generation_config"]["history_policy"],
        "shard_index": plan["shard_index"],
        "shard_count": plan["shard_count"],
        "shard_strategy": SHARD_STRATEGY,
        "full_record_count": plan["full_record_count"],
        "shard_record_count": len(rows),
        "record_count": len(predictions),
        "record_ids_sha256": stable_sha256(ids),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": elapsed_seconds,
        "reads_private_answer_keys": False,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(
        output_dir / "partial_generation_manifest.json",
        manifest,
        private=True,
    )
    return manifest


def _valid_elapsed(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _load_resume(
    resume_dir: Path,
    *,
    plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
    """Load a compatible complete shard or batch-aligned partial prefix."""

    if not resume_dir.is_dir():
        raise S1GenerationError(f"the resume directory is absent: {resume_dir}")
    artifacts = any(
        (resume_dir / name).exists()
        for name in (
            "generation_manifest.json",
            "predictions.jsonl",
            "partial_generation_manifest.json",
            "partial_predictions.jsonl",
        )
    )
    if not artifacts:
        return [], 0.0, None
    if read_json(resume_dir / "generation_plan.json") != dict(plan):
        raise S1GenerationError("the resume generation plan changed")

    complete_manifest_path = resume_dir / "generation_manifest.json"
    complete_predictions_path = resume_dir / "predictions.jsonl"
    if complete_manifest_path.is_file() and complete_predictions_path.is_file():
        manifest = read_json(complete_manifest_path)
        claimed = manifest.get("manifest_sha256")
        identity = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        predictions = read_jsonl(complete_predictions_path)
        if (
            manifest.get("schema_version")
            not in {PREDICTION_SCHEMA_VERSION, PREDICTION_SHARD_SCHEMA_VERSION}
            or not isinstance(claimed, str)
            or stable_sha256(identity) != claimed
            or sha256_file(complete_predictions_path)
            != manifest.get("predictions_sha256")
            or not _valid_elapsed(manifest.get("elapsed_seconds"))
            or any(
                manifest.get(key) != value
                for key, value in _plan_manifest_fields(plan).items()
            )
        ):
            raise S1GenerationError("the complete resume shard is invalid")
        expected_ids = [str(row["record_id"]) for row in rows]
        if [row.get("record_id") for row in predictions] != expected_ids:
            raise S1GenerationError("the complete resume record order changed")
        return predictions, float(manifest["elapsed_seconds"]), manifest

    partial_manifest_path = resume_dir / "partial_generation_manifest.json"
    partial_predictions_path = resume_dir / "partial_predictions.jsonl"
    if not partial_manifest_path.is_file() or not partial_predictions_path.is_file():
        return [], 0.0, None
    manifest = read_json(partial_manifest_path)
    claimed = manifest.get("manifest_sha256")
    identity = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    predictions = read_jsonl(partial_predictions_path)
    ids = [str(row.get("record_id")) for row in predictions]
    expected_ids = [str(row["record_id"]) for row in rows[: len(predictions)]]
    if (
        manifest.get("schema_version") != PARTIAL_PREDICTION_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or manifest.get("generation_plan_sha256") != stable_sha256(plan)
        or manifest.get("history_policy", "strip")
        != plan["generation_config"].get("history_policy", "strip")
        or sha256_file(partial_predictions_path)
        != manifest.get("predictions_sha256")
        or not _valid_elapsed(manifest.get("elapsed_seconds"))
        or ids != expected_ids
        or not predictions
        or len(predictions) > len(rows)
        or (len(predictions) != len(rows) and len(predictions) % batch_size)
    ):
        raise S1GenerationError("the partial resume shard is invalid")
    return predictions, float(manifest["elapsed_seconds"]), None


def _validated_shard(
    shard_dir: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = shard_dir / "generation_manifest.json"
    predictions_path = shard_dir / "predictions.jsonl"
    manifest = read_json(manifest_path)
    claimed = manifest.get("manifest_sha256")
    identity = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    if (
        manifest.get("schema_version") != PREDICTION_SHARD_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or manifest.get("shard_index") != shard_index
        or manifest.get("shard_count") != shard_count
        or manifest.get("shard_strategy") != SHARD_STRATEGY
        or manifest.get("reads_private_answer_keys") is not False
        or not _valid_elapsed(manifest.get("elapsed_seconds"))
        or sha256_file(predictions_path) != manifest.get("predictions_sha256")
    ):
        raise S1GenerationError(f"prediction shard {shard_index} is invalid")
    predictions = read_jsonl(predictions_path)
    if len(predictions) != manifest.get("record_count"):
        raise S1GenerationError(f"prediction shard {shard_index} changed size")
    return manifest, predictions


def merge_prediction_shards(
    *,
    bundle_root: Path,
    shards_root: Path,
    output_dir: Path,
    expected_shard_count: int,
) -> dict[str, Any]:
    """Verify and merge S1 prediction shards in original panel order."""

    if expected_shard_count < 1:
        raise S1GenerationError("the expected shard count must be positive")
    bundle, rows, _ = load_generation_bundle(bundle_root)
    shards_root = shards_root.resolve()
    expected_directories = {
        f"shard-{index:05d}" for index in range(expected_shard_count)
    }
    actual_directories = {
        path.name for path in shards_root.iterdir() if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise S1GenerationError("the S1 prediction shard set changed")

    expected_common: dict[str, Any] | None = None
    first_manifest: dict[str, Any] | None = None
    merged: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for shard_index in range(expected_shard_count):
        manifest, predictions = _validated_shard(
            shards_root / f"shard-{shard_index:05d}",
            shard_index=shard_index,
            shard_count=expected_shard_count,
        )
        if (
            manifest.get("evaluation_contract") != EVALUATION_CONTRACT
            or manifest.get("test_panel_id") != bundle["test_panel_id"]
            or manifest.get("generation_bundle_sha256")
            != bundle["bundle_sha256"]
            or manifest.get("full_record_count") != len(rows)
        ):
            raise S1GenerationError(f"prediction shard {shard_index} changed inputs")
        common = _validated_shard_identity(
            manifest,
            expected=expected_common,
        )
        if expected_common is None:
            expected_common = common
            first_manifest = manifest
        expected_ids = [
            str(row["record_id"])
            for row in rows[shard_index::expected_shard_count]
        ]
        if [row.get("record_id") for row in predictions] != expected_ids:
            raise S1GenerationError(f"prediction shard {shard_index} has wrong rows")
        for prediction in predictions:
            record_id = str(prediction["record_id"])
            if record_id in merged:
                raise S1GenerationError(f"duplicate S1 prediction {record_id}")
            merged[record_id] = prediction
        receipts.append(
            {
                "shard_index": shard_index,
                "record_count": len(predictions),
                "predictions_sha256": manifest["predictions_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )
    if first_manifest is None or expected_common is None:
        raise S1GenerationError("no S1 prediction shards were found")
    ordered_ids = [str(row["record_id"]) for row in rows]
    if set(merged) != set(ordered_ids):
        raise S1GenerationError("the merged S1 record set changed")
    predictions = [merged[record_id] for record_id in ordered_ids]
    output_dir = output_dir.resolve()
    predictions_path = output_dir / "predictions.jsonl"
    write_prediction_jsonl(predictions_path, predictions)
    elapsed = [float(receipt["elapsed_seconds"]) for receipt in receipts]
    manifest = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        **expected_common,
        "record_count": len(predictions),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": max(elapsed),
        "parallelism": {
            "shard_strategy": SHARD_STRATEGY,
            "shard_count": expected_shard_count,
            "completed_shards": len(receipts),
            "maximum_shard_seconds": max(elapsed),
            "total_shard_seconds": sum(elapsed),
            "shards": receipts,
        },
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest, private=True)
    write_json(
        output_dir / "generation_plan.json",
        {
            "schema_version": PARALLEL_GENERATION_PLAN_SCHEMA_VERSION,
            **expected_common,
            "record_count": len(rows),
            "parallelism": {
                "shard_strategy": SHARD_STRATEGY,
                "shard_count": expected_shard_count,
            },
        },
        private=True,
    )
    return {"status": "complete", "manifest": manifest}


def generate_predictions(
    *,
    bundle_root: Path,
    base_model_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    tokenizer_manifest_path: Path,
    output_dir: Path,
    method_id: str,
    corpus_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    model_identity_sha256: str,
    base_model_artifact_sha256: str,
    max_new_tokens: int,
    max_total_tokens: int,
    max_tool_call_tokens: int,
    max_final_prompt_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str,
    seed: int,
    batch_size: int,
    local_files_only: bool,
    history_policy: str = DEFAULT_HISTORY_POLICY,
    shard_index: int = 0,
    shard_count: int = 1,
    shard_output: bool = False,
    resume_shard_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate one complete native S1 prediction shard."""

    if not enable_thinking:
        raise S1GenerationError("native S1 evaluation requires thinking")
    if history_policy not in HISTORY_POLICIES:
        raise S1GenerationError(f"unknown S1 history policy: {history_policy!r}")
    if max_new_tokens < 1 or max_total_tokens <= max_new_tokens:
        raise S1GenerationError("the S1 generation token limits are invalid")
    if max_tool_call_tokens < 1 or max_final_prompt_tokens < 1:
        raise S1GenerationError("the S1 generation safety limits are invalid")
    if batch_size < 1:
        raise S1GenerationError("the S1 generation batch size is invalid")

    bundle, all_rows, source_index_path = load_generation_bundle(bundle_root)
    base_model_artifact_sha256 = validate_base_model_artifact_binding(
        base_model_path.resolve(),
        base_model_artifact_sha256,
    )
    rows = select_shard_rows(
        all_rows,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    tokenizer_manifest = validated_tokenizer_manifest(
        tokenizer_manifest_path.resolve()
    )
    if tokenizer_manifest.get("manifest_sha256") != tokenizer_manifest_sha256:
        raise S1GenerationError("the tokenizer manifest identity changed")
    tokenizer_files = tokenizer_artifact_hashes(tokenizer_path.resolve())
    tokenizer_artifacts_sha256 = validate_tokenizer_artifact_binding(
        checkpoint_path.resolve(),
        tokenizer_files,
    )
    try:
        checkpoint, train_run_id = validate_checkpoint(
            checkpoint_path.resolve(),
            method_id=method_id,
            corpus_manifest_sha256=corpus_manifest_sha256,
            tokenizer_manifest_sha256=tokenizer_manifest_sha256,
            model_identity_sha256=model_identity_sha256,
        )
    except Exception as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise S1GenerationError(f"the S1 checkpoint is invalid: {error}") from error

    generation_config = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "max_total_tokens": max_total_tokens,
        "max_tool_call_tokens": max_tool_call_tokens,
        "max_final_prompt_tokens": max_final_prompt_tokens,
        "maximum_tool_calls": max((maximum_tool_calls(str(row["record_id"])) for row in all_rows), default=MAX_TOOL_CALLS),
        "evidence_id_policy": EVIDENCE_ID_POLICY,
        "network_policy": NETWORK_POLICY,
        "enable_thinking": True,
        "reasoning_effort": reasoning_effort,
        "history_policy": history_policy,
        "seed": seed,
        "batch_size": batch_size,
        "device_map": "balanced_low_0",
    }
    plan = {
        "schema_version": GENERATION_PLAN_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": bundle["test_panel_id"],
        "record_count": len(rows),
        "full_record_count": len(all_rows),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_strategy": SHARD_STRATEGY,
        "method_id": method_id,
        "train_run_id": train_run_id,
        "base_model_identity_sha256": model_identity_sha256,
        "base_model_artifact_sha256": base_model_artifact_sha256,
        "checkpoint_identity_sha256": checkpoint["checkpoint_identity_sha256"],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_artifacts_sha256": tokenizer_artifacts_sha256,
        "tokenizer_artifact_hashes": tokenizer_files,
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "source_index_sha256": bundle["source_index"]["sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False,
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if dry_run:
        write_json(output_dir / "generation_plan.json", plan, private=True)
        return {"status": "dry_run", "plan": plan}

    predictions: list[dict[str, Any]] = []
    resumed_seconds = 0.0
    completed_manifest: dict[str, Any] | None = None
    if resume_shard_dir is not None:
        predictions, resumed_seconds, completed_manifest = _load_resume(
            resume_shard_dir.resolve(),
            plan=plan,
            rows=rows,
            batch_size=batch_size,
        )
    write_json(output_dir / "generation_plan.json", plan, private=True)
    if completed_manifest is not None:
        predictions_path = output_dir / "predictions.jsonl"
        write_prediction_jsonl(predictions_path, predictions)
        if sha256_file(predictions_path) != completed_manifest["predictions_sha256"]:
            raise S1GenerationError("the copied complete S1 shard changed")
        write_json(
            output_dir / "generation_manifest.json",
            completed_manifest,
            private=True,
        )
        return {
            "status": "complete",
            "resume_status": "reused_complete",
            "manifest": completed_manifest,
        }

    installed_network_policy = install_network_guard()
    if installed_network_policy != NETWORK_POLICY:
        raise S1GenerationError("the S1 network guard policy changed")

    import torch

    torch.manual_seed(seed)
    model, tokenizer = load_backend(
        base_model_path=base_model_path.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        tokenizer_path=tokenizer_path.resolve(),
        local_files_only=local_files_only,
    )
    if batch_size > 1:
        tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    contract = load_contract()
    record_indexes = {
        str(row["record_id"]): index for index, row in enumerate(all_rows)
    }
    runtime = DeterministicEvidenceRuntime.from_source_index(
        source_index_path,
        contract=contract,
        record_indexes=record_indexes,
    )
    initial_count = len(predictions)
    started = time.monotonic()
    for batch_start in range(initial_count, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_predictions = generate_trajectory_batch(
            batch_rows,
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_module=torch,
            runtime=runtime,
            contract=contract,
            max_new_tokens=max_new_tokens,
            max_total_tokens=max_total_tokens,
            max_tool_call_tokens=max_tool_call_tokens,
            max_final_prompt_tokens=max_final_prompt_tokens,
            reasoning_effort=reasoning_effort,
            shard_index=shard_index,
            network_guarded=True,
            history_policy=history_policy,
        )
        predictions.extend(batch_predictions)
        elapsed = resumed_seconds + time.monotonic() - started
        partial = _write_partial_checkpoint(
            output_dir,
            plan=plan,
            rows=rows,
            predictions=predictions,
            elapsed_seconds=elapsed,
        )
        print(
            json.dumps(
                {
                    "event": "s1_generation_batch_checkpointed",
                    "shard_index": shard_index,
                    "completed": len(predictions),
                    "total": len(rows),
                    "partial_manifest_sha256": partial["manifest_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    predictions_path = output_dir / "predictions.jsonl"
    write_prediction_jsonl(predictions_path, predictions)
    manifest = {
        "schema_version": (
            PREDICTION_SHARD_SCHEMA_VERSION
            if shard_count > 1 or shard_output
            else PREDICTION_SCHEMA_VERSION
        ),
        **_plan_manifest_fields(plan),
        "record_count": len(predictions),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": resumed_seconds + time.monotonic() - started,
    }
    if shard_count > 1 or shard_output:
        manifest.update(
            {
                "full_record_count": len(all_rows),
                "shard_index": shard_index,
                "shard_count": shard_count,
                "shard_strategy": SHARD_STRATEGY,
            }
        )
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest, private=True)
    return {"status": "complete", "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-bundle", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--model-identity-sha256", required=True)
    parser.add_argument("--base-model-artifact-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--max-tool-call-tokens", type=int, default=1024)
    parser.add_argument("--max-final-prompt-tokens", type=int, default=4096)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument(
        "--history-policy", choices=sorted(HISTORY_POLICIES),
        default=DEFAULT_HISTORY_POLICY,
        help="Retain reasoning from this user turn's generated calls; strip reproduces historical evaluation.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-output", action="store_true")
    parser.add_argument("--resume-shard-dir", type=Path)
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = generate_predictions(
        bundle_root=args.generation_bundle,
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir,
        method_id=args.method_id,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
        model_identity_sha256=args.model_identity_sha256,
        base_model_artifact_sha256=args.base_model_artifact_sha256,
        max_new_tokens=args.max_new_tokens,
        max_total_tokens=args.max_total_tokens,
        max_tool_call_tokens=args.max_tool_call_tokens,
        max_final_prompt_tokens=args.max_final_prompt_tokens,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        history_policy=args.history_policy,
        seed=args.seed,
        batch_size=args.batch_size,
        local_files_only=not args.allow_remote_files,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        shard_output=args.shard_output,
        resume_shard_dir=args.resume_shard_dir,
        dry_run=args.dry_run,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
