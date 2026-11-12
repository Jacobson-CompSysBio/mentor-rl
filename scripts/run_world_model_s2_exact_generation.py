#!/usr/bin/env python3
"""Generate deterministic offline S2 trajectories from answer-free prompts."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model.s2_3_tokens import verification_clock  # noqa: E402
from runtime.world_model_training import (  # noqa: E402
    V3_HARNESS_HEADER, V3_USER_HEADER, tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.build_world_model_s2_eval_bundle import EVALUATION_CONTRACT  # noqa: E402
from scripts.build_world_model_s2_generation_bundle import (  # noqa: E402
    GENERATION_BUNDLE_SCHEMA_VERSION, TOOL_FIXTURE_SCHEMA_VERSION,
    _validate_question,
)
from scripts.run_world_model_s1_exact_generation import (  # noqa: E402
    clean_decoded_generation, install_network_guard,
)
from scripts.run_world_model_v2_s0_exact_generation import (  # noqa: E402
    CHECKPOINT_FORMAT, checkpoint_artifact_identity,
    generate_prompt_batch_with_oom_backoff, load_backend,
)
from scripts.world_model_s1_eval_common import (  # noqa: E402
    canonical_json, parse_assistant_turn, read_json, read_jsonl, require_sha256,
    sha256_file, stable_sha256, write_json,
)

PREDICTION_SCHEMA_VERSION = "mentor-rl-world-model-s2-predictions-v1"
PREDICTION_SHARD_SCHEMA_VERSION = "mentor-rl-world-model-s2-prediction-shard-v1"
PARTIAL_PREDICTION_SCHEMA_VERSION = "mentor-rl-world-model-s2-partial-prediction-shard-v1"
GENERATION_PLAN_SCHEMA_VERSION = "mentor-rl-world-model-s2-generation-plan-v1"
PARALLEL_GENERATION_PLAN_SCHEMA_VERSION = "mentor-rl-world-model-s2-parallel-generation-plan-v1"
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s2-training-exposure-v1"
SHARD_STRATEGY = "strided_row_order_v1"
NETWORK_POLICY = "python_audit_deny_outbound_socket_v1"
GPT_OSS_ASSISTANT_BOUNDARY = "<|start|>assistant"
MAXIMUM_TOOL_CALLS = 4
# A lone prompt longer than this is prefilled in chunks, so eager attention
# never holds a full prompt-by-prompt score matrix (S2 prompts reach 6,350).
SINGLETON_PREFILL_CHUNK_TOKENS = 2048
PREDICTION_KEYS = {
    "row_id", "generated_messages", "generations", "stop_reason",
    "trajectory_error", "tool_call_count", "executed_tool_call_count",
    "network_used_false",
}


class S2GenerationError(RuntimeError):
    """Report one invalid S2 generation input or harness fault."""


def write_prediction_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write predictions without sorting contract-significant final fields."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    path.chmod(0o600)


def _bundle_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in manifest.items() if key != "bundle_sha256"}


def _validate_fixture(row: Any) -> dict[str, Any]:
    """Return one answer-free ordered tool fixture."""
    if not isinstance(row, Mapping) or set(row) != {"schema_version", "row_id", "steps"}:
        raise S2GenerationError("an S2 tool fixture has an invalid shape")
    if row.get("schema_version") != TOOL_FIXTURE_SCHEMA_VERSION or not isinstance(row.get("row_id"), str):
        raise S2GenerationError("an S2 tool fixture identity changed")
    steps = row.get("steps")
    if not isinstance(steps, list):
        raise S2GenerationError("an S2 tool fixture has no steps")
    for step in steps:
        call = step.get("tool_call") if isinstance(step, Mapping) else None
        response = step.get("tool_response") if isinstance(step, Mapping) else None
        if (not isinstance(step, Mapping) or set(step) != {"tool_call", "tool_response"}
                or not isinstance(call, Mapping)
                or set(call) != {"name", "arguments"} or not isinstance(call.get("arguments"), Mapping)
                or not isinstance(response, Mapping) or set(response) != {"name", "content"}
                or response.get("name") != call.get("name") or not isinstance(response.get("content"), Mapping)):
            raise S2GenerationError("an S2 tool fixture step changed")
    return {str(key): value for key, value in row.items()}


def load_generation_bundle(bundle_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and fully verify public questions and isolated tool fixtures."""
    bundle_root = bundle_root.resolve()
    manifest = read_json(bundle_root / "manifest.json")
    claimed = manifest.get("bundle_sha256")
    if (manifest.get("schema_version") != GENERATION_BUNDLE_SCHEMA_VERSION
            or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
            or manifest.get("native_transcript") is not True
            or manifest.get("reads_private_answer_keys") is not False
            or manifest.get("tool_fixtures_contain_final_targets") is not False
            or not isinstance(claimed, str) or stable_sha256(_bundle_identity(manifest)) != claimed):
        raise S2GenerationError("the S2 generation bundle identity changed")
    questions_path, fixtures_path = bundle_root / "questions.jsonl", bundle_root / "tool_fixtures.jsonl"
    if (sha256_file(questions_path) != manifest.get("questions_sha256")
            or sha256_file(fixtures_path) != manifest.get("tool_fixtures_sha256")):
        raise S2GenerationError("an S2 generation bundle file changed")
    routes = manifest.get("contract_routes")
    if not isinstance(routes, Mapping):
        raise S2GenerationError("the S2 generation bundle has no routes")
    questions = [_validate_question(row, routes) for row in read_jsonl(questions_path)]
    fixtures_list = [_validate_fixture(row) for row in read_jsonl(fixtures_path)]
    question_ids = [str(row["row_id"]) for row in questions]
    fixture_ids = [str(row["row_id"]) for row in fixtures_list]
    maximum_tool_calls = manifest.get("maximum_tool_calls")
    observed_maximum = max((len(row["steps"]) for row in fixtures_list), default=0)
    if (question_ids != fixture_ids or len(question_ids) != manifest.get("record_count")
            or len(set(question_ids)) != len(question_ids)
            or stable_sha256(question_ids) != manifest.get("record_ids_sha256")
            or type(maximum_tool_calls) is not int
            or not 0 <= maximum_tool_calls <= MAXIMUM_TOOL_CALLS
            or observed_maximum != maximum_tool_calls):
        raise S2GenerationError("the S2 generation row set changed")
    return manifest, questions, {str(row["row_id"]): row for row in fixtures_list}


def _validate_prediction(row: Any, *, require_network_guard: bool = True) -> dict[str, Any]:
    """Return one complete row-local prediction without interpreting its answer."""
    error = row.get("trajectory_error") if isinstance(row, Mapping) else None
    if (not isinstance(row, Mapping) or set(row) != PREDICTION_KEYS
            or not isinstance(row.get("row_id"), str) or not row["row_id"]
            or not isinstance(row.get("generated_messages"), list)
            or not isinstance(row.get("generations"), list)
            or not isinstance(row.get("stop_reason"), str) or not row["stop_reason"]
            or error is not None and not isinstance(error, str)
            or type(row.get("tool_call_count")) is not int or row["tool_call_count"] < 0
            or type(row.get("executed_tool_call_count")) is not int
            or row["executed_tool_call_count"] < 0
            or require_network_guard and row.get("network_used_false") is not True
            or not require_network_guard and type(row.get("network_used_false")) is not bool):
        raise S2GenerationError("an S2 prediction has an invalid harness shape")
    return {str(key): value for key, value in row.items()}


def _model_messages(question: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a public system/harness/user prefix to the training chat wire."""
    system, harness, user = question["messages"]
    return [{"role": "system", "content": system["content"]}, {"role": "user",
        "content": f"{V3_HARNESS_HEADER}\n{harness['content']}\n\n{V3_USER_HEADER}\n{user['content']}"}]


@dataclass(slots=True)
class TrajectoryState:
    """Mutable state for one answer-free S2 current turn."""
    question: dict[str, Any]
    fixture: dict[str, Any]
    prompt_messages: list[dict[str, Any]]
    generated_messages: list[dict[str, Any]] = field(default_factory=list)
    generations: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    trajectory_error: str | None = None
    executed_tool_calls: int = 0

    @property
    def active(self) -> bool:
        return self.stop_reason is None


TurnGenerator = Callable[[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], int], str | Mapping[str, Any]]


def _new_state(question: Mapping[str, Any], fixture: Mapping[str, Any]) -> TrajectoryState:
    return TrajectoryState(dict(question), dict(fixture), _model_messages(question))


def _generation_details(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return {"raw_generation": value, "prompt_tokens": None,
                "generated_tokens": None, "action_tokens": None}
    raw = value.get("raw_generation")
    if not isinstance(raw, str):
        raise S2GenerationError("an S2 generation result has no raw text")
    return {"raw_generation": raw, "prompt_tokens": value.get("prompt_tokens"),
            "generated_tokens": value.get("generated_tokens"), "action_tokens": value.get("action_tokens")}


def _append_tool_prompt(state: TrajectoryState, call: Mapping[str, Any],
                        response: Mapping[str, Any], thinking: str) -> None:
    state.prompt_messages.extend([{"role": "assistant", "thinking": thinking,
        "tool_calls": [{"type": "function", "function": {
            "name": call["name"], "arguments": dict(call["arguments"])}}]},
        {"role": "tool", "name": response["name"], "content": dict(response["content"])}])


def advance_trajectory(state: TrajectoryState, generation: str | Mapping[str, Any],
                       *, contract: Mapping[str, Any], eos_token: str | None = None,
                       pad_token: str | None = None, max_tool_call_tokens: int | None = None) -> None:
    """Parse one action, serving a fixture only after an exact expected call."""
    if not state.active:
        raise S2GenerationError("cannot advance a completed S2 trajectory")
    details = _generation_details(generation)
    encoded, disallowed = clean_decoded_generation(details["raw_generation"],
        eos_token=eos_token, pad_token=pad_token)
    parsed, parse_error = parse_assistant_turn(encoded, contract)
    state.generations.append({"turn_index": len(state.generations),
        "raw_generation": details["raw_generation"], "encoded_generation": encoded,
        "prompt_tokens": details["prompt_tokens"], "generated_tokens": details["generated_tokens"],
        "action_tokens": details["action_tokens"], "disallowed_special_tokens": disallowed,
        "parse_error": parse_error})
    if disallowed or parsed is None or parse_error is not None:
        state.stop_reason = "parse_error"
        state.trajectory_error = "disallowed_special_tokens" if disallowed else (parse_error or "assistant_turn_parse_error")
        return
    thinking = str(parsed["thinking"])
    if parsed["kind"] == "final":
        state.generated_messages.append({"role": "assistant", "thinking": thinking,
                                         "content": dict(parsed["content"])})
        state.stop_reason = "final"
        return
    call = {"name": str(parsed["name"]), "arguments": dict(parsed["arguments"])}
    state.generated_messages.append({"role": "assistant", "thinking": thinking,
                                     "tool_call": call})
    tokens = details.get("action_tokens")
    if (max_tool_call_tokens is not None and isinstance(tokens, int)
            and not isinstance(tokens, bool) and tokens > max_tool_call_tokens):
        state.stop_reason, state.trajectory_error = "tool_call_token_limit", "max_tool_call_tokens_exceeded"
        return
    steps = state.fixture["steps"]
    if state.executed_tool_calls >= len(steps) or call != steps[state.executed_tool_calls]["tool_call"]:
        state.stop_reason, state.trajectory_error = "tool_call_mismatch", "unexpected_tool_call_or_arguments"
        return
    response = steps[state.executed_tool_calls]["tool_response"]
    tool_message = {"role": "tool", "name": response["name"], "content": dict(response["content"])}
    state.generated_messages.append(tool_message)
    _append_tool_prompt(state, call, response, thinking)
    state.executed_tool_calls += 1


def prediction_from_state(state: TrajectoryState, *, network_guarded: bool = False) -> dict[str, Any]:
    """Serialize a final or diagnostically stopped S2 row."""
    return {"row_id": state.question["row_id"], "generated_messages": state.generated_messages,
        "generations": state.generations, "stop_reason": state.stop_reason or "generation_limit",
        "trajectory_error": state.trajectory_error,
        "tool_call_count": sum(message.get("role") == "assistant" and "tool_call" in message
                               for message in state.generated_messages),
        "executed_tool_call_count": state.executed_tool_calls,
        "network_used_false": network_guarded}


def orchestrate_s2_trajectory(question: Mapping[str, Any], fixture: Mapping[str, Any],
                              *, generate_turn: TurnGenerator) -> dict[str, Any]:
    """Run one deterministic S2 trajectory for unit and harness tests."""
    state, contract = _new_state(question, fixture), load_contract()
    for turn_index in range(len(fixture["steps"]) + 1):
        generation = generate_turn(tuple(state.prompt_messages), tuple(question["tools"]), turn_index)
        advance_trajectory(state, generation, contract=contract)
        if not state.active:
            break
    if state.active:
        state.stop_reason, state.trajectory_error = "generation_limit", "terminal_answer_not_generated"
    return prediction_from_state(state)


def _render_prompt(state: TrajectoryState, *, tokenizer: Any, reasoning_effort: str) -> str:
    """Render the exact S2 tool-enabled GPT-OSS prompt."""
    prompt = tokenizer.apply_chat_template(state.prompt_messages, tools=state.question["tools"],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
        reasoning_effort=reasoning_effort, strftime_now=verification_clock)
    if not isinstance(prompt, str) or not prompt.endswith(GPT_OSS_ASSISTANT_BOUNDARY):
        raise S2GenerationError("the S2 prompt lacks its GPT-OSS assistant boundary")
    return prompt


def _prompt_token_count(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False)
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0] if len(values) == 1 else None
    if not isinstance(values, list):
        raise S2GenerationError("S2 prompt tokenization returned no input IDs")
    return len(values)


def _generation_prompt_token_count(tokenizer: Any, prompt: str) -> int:
    """Count a rendered prompt with the same special-token policy as generate."""
    encoded = tokenizer(prompt, add_special_tokens=True)
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0] if len(values) == 1 else None
    if not isinstance(values, list):
        raise S2GenerationError("S2 generation prompt tokenization returned no input IDs")
    return len(values)


def generate_trajectory_batch(questions: Sequence[Mapping[str, Any]], *, fixtures: Mapping[str, Mapping[str, Any]],
        model: Any, tokenizer: Any, device: Any, torch_module: Any, max_new_tokens: int,
        max_total_tokens: int, max_tool_call_tokens: int, max_final_prompt_tokens: int,
        reasoning_effort: str, shard_index: int, network_guarded: bool = False) -> list[dict[str, Any]]:
    """Generate one batched shard, failing model errors row-by-row."""
    contract = load_contract()
    states = [_new_state(row, fixtures[str(row["row_id"])]) for row in questions]
    max_turns = max((len(state.fixture["steps"]) + 1 for state in states), default=1)
    for turn_index in range(max_turns):
        active: list[tuple[TrajectoryState, str, int]] = []
        for state in states:
            if not state.active:
                continue
            prompt = _render_prompt(state, tokenizer=tokenizer, reasoning_effort=reasoning_effort)
            prompt_tokens = _generation_prompt_token_count(tokenizer, prompt)
            if state.executed_tool_calls and (
                    prompt_tokens > max_final_prompt_tokens or prompt_tokens >= max_total_tokens):
                state.stop_reason, state.trajectory_error = (
                    "prompt_limit", "max_final_prompt_tokens_exceeded"
                )
                continue
            if prompt_tokens >= max_total_tokens:
                raise S2GenerationError(
                    f"initial S2 prompt for {state.question['row_id']} leaves no generation budget"
                )
            active.append((state, prompt, prompt_tokens))
        if not active:
            break
        # The inherited helper caps a whole batch at its smallest remaining
        # context budget. Isolate near-limit prompts so they cannot truncate
        # otherwise full-budget rows.
        full_budget = [index for index, (_, _, tokens) in enumerate(active)
                       if tokens <= max_total_tokens - max_new_tokens]
        near_limit = [index for index, (_, _, tokens) in enumerate(active)
                      if tokens > max_total_tokens - max_new_tokens]
        groups = ([full_budget] if full_budget else []) + [[index] for index in near_limit]
        generated_by_index: dict[int, dict[str, Any]] = {}
        for group in groups:
            generations = generate_prompt_batch_with_oom_backoff(model=model, tokenizer=tokenizer,
                device=device, torch_module=torch_module,
                prompts=[active[index][1] for index in group],
                record_ids=[str(active[index][0].question["row_id"]) for index in group],
                max_new_tokens=max_new_tokens, max_total_tokens=max_total_tokens,
                stage=f"s2_assistant_turn_{turn_index + 1}", shard_index=shard_index,
                prefill_chunk_size=SINGLETON_PREFILL_CHUNK_TOKENS)
            for index, generation in zip(group, generations, strict=True):
                generated_by_index[index] = dict(generation)
        if set(generated_by_index) != set(range(len(active))):
            raise S2GenerationError("the S2 batch generator returned incomplete results")
        for index, (state, _, _) in enumerate(active):
            generation = generated_by_index[index]
            encoded, _ = clean_decoded_generation(str(generation["raw_generation"]),
                eos_token=getattr(tokenizer, "eos_token", None), pad_token=getattr(tokenizer, "pad_token", None))
            generation["action_tokens"] = _prompt_token_count(tokenizer, encoded)
            advance_trajectory(state, generation, contract=contract,
                eos_token=getattr(tokenizer, "eos_token", None), pad_token=getattr(tokenizer, "pad_token", None),
                max_tool_call_tokens=max_tool_call_tokens)
    for state in states:
        if state.active:
            state.stop_reason, state.trajectory_error = "generation_limit", "terminal_answer_not_generated"
    return [prediction_from_state(state, network_guarded=network_guarded) for state in states]


def use_routed_experts(model: Any) -> int:
    """Compute only each token's routed experts, as training did.

    In eval mode on GPU, transformers 4.57 GptOssExperts copies every token
    to all 128 experts. Prefill memory then grows with 128 x prompt tokens,
    and job 5892323 ran out of memory on every prompt above 3,812 tokens.
    The module's training branch loops over the routed experts only. The
    experts hold no dropout or adapter, so the flag changes nothing else.
    """
    experts = [module for module in model.modules() if type(module).__name__ == "GptOssExperts"]
    if len(experts) != getattr(model.config, "num_hidden_layers", None):
        raise S2GenerationError("the S2 model lacks one GPT-OSS expert module per layer")
    for module in experts:
        module.training = True
    return len(experts)


def validate_base_model_artifact_binding(base_model_path: Path, expected_sha256: str) -> str:
    """Verify the staged base-model marker and complete artifact receipt."""
    expected_sha256 = require_sha256(expected_sha256, "S2 base-model artifact")
    try:
        marker = (base_model_path / ".mentor_model_identity_sha256").read_text().strip()
        receipt = read_json(base_model_path / ".mentor_base_model_artifact.json")
    except Exception as error:
        raise S2GenerationError("the staged S2 base-model receipt is absent") from error
    claimed = receipt.get("base_model_artifact_sha256")
    identity = {key: value for key, value in receipt.items() if key != "base_model_artifact_sha256"}
    if marker != expected_sha256 or claimed != expected_sha256 or stable_sha256(identity) != expected_sha256:
        raise S2GenerationError("the staged S2 base-model artifact changed")
    return expected_sha256


def validate_tokenizer_artifact_binding(checkpoint_path: Path, tokenizer_files: Mapping[str, str]) -> str:
    """Bind inference prompt artifacts to the tokenizer used by training."""
    artifact_sha256 = stable_sha256(tokenizer_files)
    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if not isinstance(identity, Mapping) or identity.get("tokenizer_artifacts_sha256") != artifact_sha256:
        raise S2GenerationError("the S2 checkpoint and tokenizer artifacts differ")
    return artifact_sha256


def validate_checkpoint(checkpoint_path: Path, *, method_id: str, corpus_manifest_sha256: str,
                        tokenizer_manifest_sha256: str, model_identity_sha256: str) -> tuple[dict[str, Any], str]:
    """Verify one completed S2 adapter and its exposure receipt."""
    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if (adapter.get("format") != CHECKPOINT_FORMAT or not isinstance(identity, Mapping)
            or identity.get("method_id") != method_id
            or identity.get("corpus_manifest_sha256") != corpus_manifest_sha256
            or identity.get("tokenizer_manifest_sha256") != tokenizer_manifest_sha256
            or identity.get("model_identity_sha256") != model_identity_sha256):
        raise S2GenerationError("the S2 checkpoint identity changed")
    exposure = read_json(checkpoint_path / "run_contract/training_exposure.json")
    claimed = exposure.get("manifest_sha256")
    exposure_identity = {str(key): value for key, value in exposure.items() if key != "manifest_sha256"}
    logical, contract = exposure.get("logical_exposure"), exposure.get("exposure_contract")
    if (exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION or exposure.get("status") != "complete"
            or exposure.get("method_id") != method_id or exposure.get("run_id") != identity.get("run_id")
            or not isinstance(claimed, str) or stable_sha256(exposure_identity) != claimed
            or not isinstance(logical, Mapping) or logical.get("all_eligible_train_rows_exposed") is not True
            or not isinstance(contract, Mapping) or contract.get("satisfied") is not True):
        raise S2GenerationError("S2 evaluation requires a complete exposure receipt")
    return checkpoint_artifact_identity(checkpoint_path), str(identity["run_id"])


def select_shard_rows(rows: Sequence[dict[str, Any]], *, shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    """Select one deterministic strided S2 row shard."""
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise S2GenerationError("the S2 shard coordinate is invalid")
    return list(rows[shard_index::shard_count])


def _plan_manifest_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("evaluation_contract", "test_panel_id", "method_id", "train_run_id",
        "base_model_identity_sha256", "base_model_artifact_sha256", "checkpoint_identity_sha256",
        "tokenizer_manifest_sha256", "tokenizer_artifacts_sha256", "evaluator_manifest_sha256",
        "tool_fixtures_sha256", "generation_bundle_sha256", "generation_config",
        "generation_config_sha256", "reads_private_answer_keys")
    return {key: plan[key] for key in keys}


def _write_partial(output_dir: Path, *, plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                   predictions: Sequence[Mapping[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    path = output_dir / "partial_predictions.jsonl"
    write_prediction_jsonl(path, predictions)
    manifest = {"schema_version": PARTIAL_PREDICTION_SCHEMA_VERSION,
        "generation_plan_sha256": stable_sha256(plan), "shard_index": plan["shard_index"],
        "shard_count": plan["shard_count"], "shard_strategy": SHARD_STRATEGY,
        "full_record_count": plan["full_record_count"], "shard_record_count": len(rows),
        "record_count": len(predictions),
        "row_ids_sha256": stable_sha256([row["row_id"] for row in predictions]),
        "predictions_sha256": sha256_file(path), "elapsed_seconds": elapsed_seconds,
        "reads_private_answer_keys": False}
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "partial_generation_manifest.json", manifest, private=True)
    return manifest


def _load_partial(resume_dir: Path, *, plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  batch_size: int) -> tuple[list[dict[str, Any]], float]:
    if not resume_dir.is_dir() or not (resume_dir / "partial_generation_manifest.json").is_file():
        return [], 0.0
    if read_json(resume_dir / "generation_plan.json") != dict(plan):
        raise S2GenerationError("the S2 resume plan changed")
    manifest = read_json(resume_dir / "partial_generation_manifest.json")
    path = resume_dir / "partial_predictions.jsonl"
    predictions = [_validate_prediction(row) for row in read_jsonl(path)]
    claimed = manifest.get("manifest_sha256")
    identity = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    expected_ids = [row["row_id"] for row in rows[:len(predictions)]]
    elapsed = manifest.get("elapsed_seconds")
    if (not isinstance(claimed, str) or stable_sha256(identity) != claimed
            or sha256_file(path) != manifest.get("predictions_sha256")
            or [row.get("row_id") for row in predictions] != expected_ids
            or stable_sha256(expected_ids) != manifest.get("row_ids_sha256")
            or len(predictions) > len(rows) or (len(predictions) < len(rows) and len(predictions) % batch_size)
            or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed)) or elapsed < 0):
        raise S2GenerationError("the S2 partial resume shard is invalid")
    return predictions, float(elapsed)


def generate_predictions(*, bundle_root: Path, base_model_path: Path, checkpoint_path: Path,
        tokenizer_path: Path, tokenizer_manifest_path: Path, output_dir: Path, method_id: str,
        corpus_manifest_sha256: str, tokenizer_manifest_sha256: str, model_identity_sha256: str,
        base_model_artifact_sha256: str, max_new_tokens: int, max_total_tokens: int,
        max_tool_call_tokens: int, max_final_prompt_tokens: int, enable_thinking: bool,
        reasoning_effort: str, seed: int, batch_size: int, local_files_only: bool,
        history_policy: str = "retain_current_turn",
        shard_index: int = 0, shard_count: int = 1, shard_output: bool = False,
        resume_shard_dir: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Generate one complete, identity-bound S2 prediction shard."""
    if (not enable_thinking or history_policy != "retain_current_turn"
            or max_new_tokens < 1 or max_total_tokens <= max_new_tokens):
        raise S2GenerationError("the S2 generation settings are invalid")
    if min(max_tool_call_tokens, max_final_prompt_tokens, batch_size) < 1:
        raise S2GenerationError("the S2 generation safety limits are invalid")
    bundle, all_rows, fixtures = load_generation_bundle(bundle_root)
    rows = select_shard_rows(all_rows, shard_index=shard_index, shard_count=shard_count)
    base_model_artifact_sha256 = validate_base_model_artifact_binding(
        base_model_path.resolve(), base_model_artifact_sha256)
    tokenizer_manifest = validated_tokenizer_manifest(tokenizer_manifest_path.resolve())
    if tokenizer_manifest.get("manifest_sha256") != tokenizer_manifest_sha256:
        raise S2GenerationError("the S2 tokenizer manifest identity changed")
    tokenizer_files = tokenizer_artifact_hashes(tokenizer_path.resolve())
    tokenizer_artifacts_sha256 = validate_tokenizer_artifact_binding(checkpoint_path.resolve(), tokenizer_files)
    checkpoint, train_run_id = validate_checkpoint(checkpoint_path.resolve(), method_id=method_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        model_identity_sha256=model_identity_sha256)
    generation_config = {"do_sample": False, "max_new_tokens": max_new_tokens,
        "max_total_tokens": max_total_tokens, "max_tool_call_tokens": max_tool_call_tokens,
        "max_final_prompt_tokens": max_final_prompt_tokens,
        "maximum_tool_calls": bundle["maximum_tool_calls"], "enable_thinking": True,
        "reasoning_effort": reasoning_effort, "seed": seed, "batch_size": batch_size,
        "history_policy": history_policy,
        "device_map": "balanced_low_0", "network_policy": NETWORK_POLICY,
        "tool_policy": "exact_ordered_fixture_v1", "expert_computation": "routed_top_k_v1",
        "singleton_prefill_chunk_tokens": SINGLETON_PREFILL_CHUNK_TOKENS}
    plan = {"schema_version": GENERATION_PLAN_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT, "test_panel_id": bundle["test_panel_id"],
        "record_count": len(rows), "full_record_count": len(all_rows),
        "shard_index": shard_index, "shard_count": shard_count, "shard_strategy": SHARD_STRATEGY,
        "method_id": method_id, "train_run_id": train_run_id,
        "base_model_identity_sha256": model_identity_sha256,
        "base_model_artifact_sha256": base_model_artifact_sha256,
        "checkpoint_identity_sha256": checkpoint["checkpoint_identity_sha256"],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_artifacts_sha256": tokenizer_artifacts_sha256,
        "tokenizer_artifact_hashes": tokenizer_files,
        "evaluator_manifest_sha256": bundle["source_evaluator_manifest_sha256"],
        "tool_fixtures_sha256": bundle["tool_fixtures_sha256"],
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output_dir / "generation_plan.json", plan, private=True)
    if dry_run:
        return {"status": "dry_run", "plan": plan}
    predictions, resumed_seconds = ([], 0.0) if resume_shard_dir is None else _load_partial(
        resume_shard_dir.resolve(), plan=plan, rows=rows, batch_size=batch_size)
    if install_network_guard() != NETWORK_POLICY:
        raise S2GenerationError("the S2 offline network policy changed")
    import torch
    torch.manual_seed(seed)
    model, tokenizer = load_backend(base_model_path=base_model_path.resolve(),
        checkpoint_path=checkpoint_path.resolve(), tokenizer_path=tokenizer_path.resolve(),
        local_files_only=local_files_only)
    use_routed_experts(model)
    if batch_size > 1:
        tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    started = time.monotonic()
    for batch_start in range(len(predictions), len(rows), batch_size):
        batch_rows = rows[batch_start:batch_start + batch_size]
        predictions.extend(generate_trajectory_batch(batch_rows, fixtures=fixtures, model=model,
            tokenizer=tokenizer, device=device, torch_module=torch, max_new_tokens=max_new_tokens,
            max_total_tokens=max_total_tokens, max_tool_call_tokens=max_tool_call_tokens,
            max_final_prompt_tokens=max_final_prompt_tokens, reasoning_effort=reasoning_effort,
            shard_index=shard_index, network_guarded=True))
        partial = _write_partial(output_dir, plan=plan, rows=rows, predictions=predictions,
                                 elapsed_seconds=resumed_seconds + time.monotonic() - started)
        print(json.dumps({"event": "s2_generation_batch_checkpointed", "shard_index": shard_index,
            "completed": len(predictions), "total": len(rows),
            "partial_manifest_sha256": partial["manifest_sha256"]}, sort_keys=True), flush=True)
    path = output_dir / "predictions.jsonl"
    write_prediction_jsonl(path, predictions)
    manifest = {"schema_version": PREDICTION_SHARD_SCHEMA_VERSION if shard_count > 1 or shard_output
                else PREDICTION_SCHEMA_VERSION, **_plan_manifest_fields(plan),
        "record_count": len(predictions), "predictions_sha256": sha256_file(path),
        "elapsed_seconds": resumed_seconds + time.monotonic() - started}
    if shard_count > 1 or shard_output:
        manifest.update(full_record_count=len(all_rows), shard_index=shard_index,
                        shard_count=shard_count, shard_strategy=SHARD_STRATEGY)
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest, private=True)
    return {"status": "complete", "manifest": manifest}


def _valid_elapsed(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)) and value >= 0)


def _validated_shard(path: Path, *, shard_index: int, shard_count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest, predictions_path = read_json(path / "generation_manifest.json"), path / "predictions.jsonl"
    claimed = manifest.get("manifest_sha256")
    identity = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    generation_config = manifest.get("generation_config")
    if (manifest.get("schema_version") != PREDICTION_SHARD_SCHEMA_VERSION
            or not isinstance(claimed, str) or stable_sha256(identity) != claimed
            or manifest.get("shard_index") != shard_index or manifest.get("shard_count") != shard_count
            or manifest.get("shard_strategy") != SHARD_STRATEGY
            or manifest.get("reads_private_answer_keys") is not False
            or not isinstance(generation_config, Mapping)
            or stable_sha256(generation_config) != manifest.get("generation_config_sha256")
            or generation_config.get("history_policy") != "retain_current_turn"
            or not _valid_elapsed(manifest.get("elapsed_seconds"))
            or sha256_file(predictions_path) != manifest.get("predictions_sha256")):
        raise S2GenerationError(f"S2 prediction shard {shard_index} is invalid")
    predictions = [_validate_prediction(row) for row in read_jsonl(predictions_path)]
    if len(predictions) != manifest.get("record_count"):
        raise S2GenerationError(f"S2 prediction shard {shard_index} changed size")
    return manifest, predictions


def merge_prediction_shards(*, bundle_root: Path, shards_root: Path, output_dir: Path,
                            expected_shard_count: int) -> dict[str, Any]:
    """Verify and merge complete S2 shards in frozen panel order."""
    if expected_shard_count < 1:
        raise S2GenerationError("the expected S2 shard count must be positive")
    bundle, rows, _ = load_generation_bundle(bundle_root)
    shards_root = shards_root.resolve()
    expected_dirs = {f"shard-{index:05d}" for index in range(expected_shard_count)}
    actual_dirs = {path.name for path in shards_root.iterdir() if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise S2GenerationError("the S2 prediction shard set changed")
    expected_common, merged, receipts = None, {}, []
    for index in range(expected_shard_count):
        manifest, predictions = _validated_shard(shards_root / f"shard-{index:05d}",
            shard_index=index, shard_count=expected_shard_count)
        common = _plan_manifest_fields(manifest)
        if expected_common is None:
            expected_common = common
        elif common != expected_common:
            raise S2GenerationError("the S2 shards use different generation inputs")
        if (manifest.get("evaluation_contract") != EVALUATION_CONTRACT
                or manifest.get("test_panel_id") != bundle["test_panel_id"]
                or manifest.get("generation_bundle_sha256") != bundle["bundle_sha256"]
                or manifest.get("full_record_count") != len(rows)):
            raise S2GenerationError(f"S2 prediction shard {index} changed inputs")
        expected_ids = [str(row["row_id"]) for row in rows[index::expected_shard_count]]
        if [row.get("row_id") for row in predictions] != expected_ids:
            raise S2GenerationError(f"S2 prediction shard {index} has wrong rows")
        for prediction in predictions:
            row_id = str(prediction["row_id"])
            if row_id in merged:
                raise S2GenerationError(f"duplicate S2 prediction {row_id}")
            merged[row_id] = prediction
        receipts.append({"shard_index": index, "record_count": len(predictions),
            "predictions_sha256": manifest["predictions_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "elapsed_seconds": manifest["elapsed_seconds"]})
    if expected_common is None:
        raise S2GenerationError("no S2 prediction shards were found")
    ordered_ids = [str(row["row_id"]) for row in rows]
    if set(merged) != set(ordered_ids):
        raise S2GenerationError("the merged S2 record set changed")
    predictions = [merged[row_id] for row_id in ordered_ids]
    output_dir = output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = output_dir / "predictions.jsonl"; write_prediction_jsonl(path, predictions)
    elapsed = [float(receipt["elapsed_seconds"]) for receipt in receipts]
    manifest = {"schema_version": PREDICTION_SCHEMA_VERSION, **expected_common,
        "record_count": len(predictions), "predictions_sha256": sha256_file(path),
        "elapsed_seconds": max(elapsed), "parallelism": {"shard_strategy": SHARD_STRATEGY,
            "shard_count": expected_shard_count, "completed_shards": len(receipts),
            "maximum_shard_seconds": max(elapsed), "total_shard_seconds": sum(elapsed),
            "shards": receipts}}
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest, private=True)
    write_json(output_dir / "generation_plan.json", {"schema_version": PARALLEL_GENERATION_PLAN_SCHEMA_VERSION,
        **expected_common, "record_count": len(rows), "parallelism": {
            "shard_strategy": SHARD_STRATEGY, "shard_count": expected_shard_count}}, private=True)
    return {"status": "complete", "manifest": manifest}


def parse_args() -> argparse.Namespace:
    """Parse one deterministic S2 shard-generation request."""
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
    parser.add_argument("--max-final-prompt-tokens", type=int, default=8191)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--history-policy", choices=["retain_current_turn"],
                        default="retain_current_turn")
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
    """Generate one shard and print its completion status."""
    args = parse_args()
    result = generate_predictions(bundle_root=args.generation_bundle,
        base_model_path=args.base_model_path, checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path, tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir, method_id=args.method_id,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
        model_identity_sha256=args.model_identity_sha256,
        base_model_artifact_sha256=args.base_model_artifact_sha256,
        max_new_tokens=args.max_new_tokens, max_total_tokens=args.max_total_tokens,
        max_tool_call_tokens=args.max_tool_call_tokens,
        max_final_prompt_tokens=args.max_final_prompt_tokens,
        enable_thinking=args.enable_thinking, reasoning_effort=args.reasoning_effort,
        history_policy=args.history_policy, seed=args.seed, batch_size=args.batch_size,
        local_files_only=not args.allow_remote_files, shard_index=args.shard_index,
        shard_count=args.shard_count, shard_output=args.shard_output,
        resume_shard_dir=args.resume_shard_dir, dry_run=args.dry_run)
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
