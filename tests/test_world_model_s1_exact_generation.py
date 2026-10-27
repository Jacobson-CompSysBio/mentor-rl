from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.run_world_model_s1_exact_generation as runner
from runtime.world_model.contract import load_contract
from runtime.world_model_training import tokenizer_artifact_hashes
from scripts.run_world_model_s1_exact_generation import (
    EVIDENCE_ID_POLICY,
    NETWORK_POLICY,
    S1GenerationError,
    TrajectoryState,
    _new_state,
    _validated_shard_identity,
    advance_trajectory,
    deterministic_evidence_id,
    orchestrate_s1_trajectory,
    validate_evidence_namespace,
    validate_base_model_artifact_binding,
    validate_tokenizer_artifact_binding,
    write_prediction_jsonl,
)
from scripts.world_model_s1_eval_common import (
    S1EvaluationError,
    V3_HARNESS_HEADER,
    model_messages,
    ordered_json,
    read_jsonl,
    validate_answer_row,
    validate_question,
    stable_sha256,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_ROOT = REPO_ROOT / "data/world_model_v4/eval/s1_v10"


@lru_cache(maxsize=1)
def _panel() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    contract = load_contract()
    questions = [
        validate_question(row, contract)
        for row in read_jsonl(PANEL_ROOT / "test_questions.jsonl")
    ]
    by_id = {row["record_id"]: row for row in questions}
    answers = {
        row["record_id"]: validate_answer_row(
            row, by_id[row["record_id"]], contract
        )
        for row in read_jsonl(PANEL_ROOT / "test_answer_key.jsonl")
    }
    return questions, answers


def _sample(*, trace: str) -> tuple[dict[str, Any], dict[str, Any]]:
    questions, answers = _panel()
    question = next(
        row for row in questions if row["metadata"]["source_trace"] == trace
    )
    return question, answers[question["record_id"]]


def _call_text(message: dict[str, Any]) -> str:
    call = message["tool_call"]
    return (
        "<|channel|>analysis<|message|>"
        + message["thinking"]
        + "<|end|><|start|>assistant to=functions."
        + call["name"]
        + "<|channel|>commentary json<|message|>"
        + json.dumps(call["arguments"], separators=(",", ":"))
        + "<|call|>"
    )


def _final_text(message: dict[str, Any]) -> str:
    return (
        "<|channel|>analysis<|message|>"
        + message["thinking"]
        + "<|end|><|start|>assistant"
        + "<|channel|>final<|message|>"
        + ordered_json(message["content"])
        + "<|return|>"
    )


@pytest.mark.parametrize("history_policy", ["strip", "retain_current_turn"])
@pytest.mark.parametrize("kind", ["tool_call", "final"])
def test_nested_channel_in_model_analysis_fails_only_the_row(history_policy, kind):
    """Malformed retained reasoning must never reach the chat template or tools."""
    question, answer = _sample(trace="S1.1.1")
    message = deepcopy(answer["expected_messages"][0 if kind == "tool_call" else -1])
    message["thinking"] += "<|channel|>analysis<|message|>nested analysis"
    raw = _call_text(message) if kind == "tool_call" else _final_text(message)
    state = _new_state(question, history_policy=history_policy)
    advance_trajectory(state, raw, runtime=None, contract=load_contract())
    assert not state.active
    assert state.stop_reason == "parse_error"
    assert state.trajectory_error == "analysis_contains_channel_tag"
    assert state.generated_messages == []
    assert state.generations[0]["raw_generation"] == raw
    assert state.generations[0]["parse_error"] == "analysis_contains_channel_tag"


class _Runtime:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = [deepcopy(output) for output in outputs]
        self.calls: list[tuple[str, dict[str, Any], bool]] = []

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        inject_backend_failure: bool,
    ) -> SimpleNamespace:
        self.calls.append((name, dict(arguments), inject_backend_failure))
        return SimpleNamespace(output=self.outputs.pop(0))


def test_direct_closed_book_final_uses_native_question_prefix() -> None:
    question, answer = _sample(trace="S1.6.1")
    expected = answer["expected_messages"]
    runtime = _Runtime([])
    seen: list[list[dict[str, Any]]] = []

    def generate(messages, tools, turn_index):
        seen.append([dict(message) for message in messages])
        assert turn_index == 0
        assert len(tools) == 3
        return _final_text(expected[-1])

    prediction = orchestrate_s1_trajectory(
        question,
        contract=load_contract(),
        runtime=runtime,
        generate_turn=generate,
    )

    assert prediction["stop_reason"] == "final"
    assert prediction["generated_messages"] == expected
    assert prediction["tool_call_count"] == 0
    assert runtime.calls == []
    assert seen[0] == model_messages(question)


def test_two_sequential_calls_retain_current_thinking_by_default() -> None:
    question, answer = _sample(trace="S1.1.9")
    expected = answer["expected_messages"]
    assert [message["role"] for message in expected] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    runtime = _Runtime([expected[1]["content"], expected[3]["content"]])
    outputs = iter(
        [_call_text(expected[0]), _call_text(expected[2]), _final_text(expected[4])]
    )
    prompts: list[list[dict[str, Any]]] = []

    def generate(messages, tools, turn_index):
        prompts.append(deepcopy(list(messages)))
        return next(outputs)

    prediction = orchestrate_s1_trajectory(
        question,
        contract=load_contract(),
        runtime=runtime,
        generate_turn=generate,
    )

    assert prediction["generated_messages"] == expected
    assert prediction["tool_call_count"] == 2
    assert prediction["executed_tool_call_count"] == 2
    assert len(prompts) == 3
    first_generated_call = prompts[1][-2]
    second_generated_call = prompts[2][-2]
    assert first_generated_call["thinking"] == expected[0]["thinking"]
    assert second_generated_call["thinking"] == expected[2]["thinking"]
    assert [message["thinking"] for message in prompts[2] if "thinking" in message] == [
        expected[0]["thinking"], expected[2]["thinking"],
    ]
    assert "tool_calls" in first_generated_call
    assert [call[0] for call in runtime.calls] == [
        "lookup_human_gene_symbol",
        "check_multiplex_membership",
    ]


@pytest.mark.parametrize("history_policy", ["strip", "retain_current_turn"])
def test_history_diagnostic_uses_only_current_generated_reasoning(history_policy):
    """Later prompts may retain model output, never the reference thoughts."""

    question, answer = _sample(trace="S1.1.9")
    original = deepcopy(question)
    expected = deepcopy(answer["expected_messages"])
    original_thoughts = [expected[index]["thinking"] for index in (0, 2)]
    expected[0]["thinking"] = "Runtime first-call reasoning sentinel."
    expected[2]["thinking"] = "Runtime second-call reasoning sentinel."
    runtime = _Runtime([expected[1]["content"], expected[3]["content"]])
    outputs = iter(
        [_call_text(expected[0]), _call_text(expected[2]), _final_text(expected[4])]
    )
    prompts = []

    def generate(messages, tools, turn_index):
        prompts.append(deepcopy(list(messages)))
        return next(outputs)

    prediction = orchestrate_s1_trajectory(
        question,
        contract=load_contract(),
        runtime=runtime,
        generate_turn=generate,
        history_policy=history_policy,
    )

    assert question == original
    assert prompts[0] == model_messages(question)
    assert prediction["generated_messages"] == expected
    for turn_index, prompt in enumerate(prompts):
        thoughts = [message["thinking"] for message in prompt if "thinking" in message]
        assert thoughts == (
            [expected[index]["thinking"] for index in (0, 2)[:turn_index]]
            if history_policy == "retain_current_turn" else []
        )
        assert all(thought not in json.dumps(prompt) for thought in original_thoughts)


@pytest.mark.parametrize("history_policy", [None, "strip", "retain_current_turn"])
def test_history_diagnostic_never_restores_previous_user_turn_reasoning(history_policy):
    """A previous answer's private reasoning is absent under both policies."""

    questions, _ = _panel()
    question = deepcopy(next(
        row for row in questions
        if sum(message["role"] == "user" for message in row["messages"]) == 2
    ))
    previous_answer = next(
        message for message in question["messages"] if message["role"] == "assistant"
    )
    previous_answer["thinking"] = "Previous-user private reasoning sentinel."
    options = {} if history_policy is None else {"history_policy": history_policy}
    state = _new_state(question, **options)
    assert "Previous-user private reasoning sentinel." not in json.dumps(
        state.prompt_messages
    )
    assert all("thinking" not in message for message in state.prompt_messages)


@pytest.fixture
def generation_inputs(tmp_path, monkeypatch):
    """Exercise production generation plumbing with artifact checks and model mocked."""
    question, _ = _sample(trace="S1.6.1")
    bundle = {"test_panel_id": "a" * 64, "bundle_sha256": "b" * 64,
              "source_index": {"sha256": "c" * 64}}
    monkeypatch.setattr(runner, "load_generation_bundle", lambda path: (bundle, [question], tmp_path / "source"))
    monkeypatch.setattr(runner, "validate_base_model_artifact_binding", lambda path, sha: sha)
    monkeypatch.setattr(runner, "validated_tokenizer_manifest", lambda path: {"manifest_sha256": "d" * 64})
    monkeypatch.setattr(runner, "tokenizer_artifact_hashes", lambda path: {"tokenizer.json": "e" * 64})
    monkeypatch.setattr(runner, "validate_tokenizer_artifact_binding", lambda path, files: "f" * 64)
    monkeypatch.setattr(runner, "validate_checkpoint", lambda *a, **kw: ({"checkpoint_identity_sha256": "1" * 64}, "run"))
    monkeypatch.setattr(runner, "install_network_guard", lambda: NETWORK_POLICY)
    monkeypatch.setattr(runner.DeterministicEvidenceRuntime, "from_source_index", lambda *a, **kw: object())
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    calls = []

    def load(**kwargs):
        calls.append("load")
        return SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")])), SimpleNamespace()

    def generate(rows, **kwargs):
        calls.append(kwargs["history_policy"])
        return [{"record_id": row["record_id"]} for row in rows]

    monkeypatch.setattr(runner, "load_backend", load)
    monkeypatch.setattr(runner, "generate_trajectory_batch", generate)
    return {
        "bundle_root": tmp_path / "bundle", "base_model_path": tmp_path / "base",
        "checkpoint_path": tmp_path / "checkpoint", "tokenizer_path": tmp_path / "tokenizer",
        "tokenizer_manifest_path": tmp_path / "tokenizer-manifest.json",
        "output_dir": tmp_path / "output", "method_id": "method",
        "corpus_manifest_sha256": "2" * 64, "tokenizer_manifest_sha256": "d" * 64,
        "model_identity_sha256": "3" * 64, "base_model_artifact_sha256": "4" * 64,
        "max_new_tokens": 4096, "max_total_tokens": 8192,
        "max_tool_call_tokens": 1024, "max_final_prompt_tokens": 6144,
        "enable_thinking": True, "reasoning_effort": "low", "seed": 900913,
        "batch_size": 1, "local_files_only": True,
    }, calls


@pytest.mark.parametrize("history_policy", [None, "strip"])
def test_production_policy_reaches_batch_and_all_generation_artifacts(generation_inputs, history_policy):
    """Default retention and explicit legacy reproduction are recorded and executed."""
    kwargs, calls = generation_inputs
    if history_policy is not None:
        kwargs["history_policy"] = history_policy
    expected = history_policy or "retain_current_turn"
    result = runner.generate_predictions(**kwargs)
    manifest = result["manifest"]
    plan = json.loads((kwargs["output_dir"] / "generation_plan.json").read_text())
    partial = json.loads((kwargs["output_dir"] / "partial_generation_manifest.json").read_text())
    assert calls == ["load", expected]
    assert manifest["generation_config"]["history_policy"] == expected
    assert plan["generation_config"] == manifest["generation_config"]
    assert partial["history_policy"] == expected
    assert manifest["generation_config_sha256"] == stable_sha256(manifest["generation_config"])


def test_history_policy_change_cannot_resume_or_mix_generation_shards(generation_inputs):
    """A policy switch creates a different generation identity even at one checkpoint."""
    kwargs, calls = generation_inputs
    retained = runner.generate_predictions(**kwargs)["manifest"]
    original_output = kwargs["output_dir"]
    changed = {**kwargs, "history_policy": "strip", "output_dir": original_output.parent / "legacy"}
    with pytest.raises(S1GenerationError, match="resume generation plan changed"):
        runner.generate_predictions(**changed, resume_shard_dir=original_output)
    assert calls == ["load", "retain_current_turn"]
    stripped = runner.generate_predictions(**changed)["manifest"]
    with pytest.raises(S1GenerationError, match="different generation inputs"):
        _validated_shard_identity(stripped, expected=_validated_shard_identity(retained))
    unknown = deepcopy(retained)
    unknown["generation_config"]["history_policy"] = "retain_all_turns"
    unknown["generation_config_sha256"] = stable_sha256(unknown["generation_config"])
    with pytest.raises(S1GenerationError, match="unknown history policy"):
        _validated_shard_identity(unknown)


@pytest.mark.parametrize("history_policy", [None, "strip"])
def test_cli_defaults_to_retention_and_passes_explicit_legacy_policy(monkeypatch, history_policy):
    """The CLI forwards its selected policy into the production entry point."""
    args = ["runner", "--dry-run", "--enable-thinking"]
    for name, value in {
        "generation-bundle": "bundle", "base-model-path": "model",
        "checkpoint-path": "checkpoint", "tokenizer-path": "tokenizer",
        "tokenizer-manifest": "manifest", "output-dir": "output", "method-id": "method",
        "corpus-manifest-sha256": "1" * 64, "tokenizer-manifest-sha256": "2" * 64,
        "model-identity-sha256": "3" * 64, "base-model-artifact-sha256": "4" * 64,
        "max-new-tokens": "4096", "max-total-tokens": "8192",
        "reasoning-effort": "low", "seed": "900913",
    }.items():
        args.extend([f"--{name}", value])
    if history_policy is not None:
        args.extend(["--history-policy", history_policy])
    monkeypatch.setattr(sys, "argv", args)
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(runner, "generate_predictions", generate)
    assert runner.main() == 0
    assert captured["history_policy"] == (history_policy or "retain_current_turn")


def test_validation_questions_require_explicit_validation_mode():
    """The diagnostic cannot silently change which split the gate accepts."""

    question, answer = _sample(trace="S1.1.9")
    validation = {**question, "split": "validation"}
    validation_answer = {**answer, "split": "validation"}
    contract = load_contract()
    with pytest.raises(S1EvaluationError, match="not a test question"):
        validate_question(validation, contract)
    assert validate_question(validation, contract, expected_split="validation") == validation
    with pytest.raises(S1EvaluationError, match="does not match its question"):
        validate_answer_row(validation_answer, validation, contract)
    assert validate_answer_row(
        validation_answer, validation, contract, expected_split="validation"
    ) == validation_answer
    with pytest.raises(S1EvaluationError, match="does not match its question"):
        validate_answer_row(answer, validation, contract, expected_split="validation")
    with pytest.raises(S1EvaluationError, match="not a validation question"):
        validate_question(question, contract, expected_split="validation")
    with pytest.raises(S1EvaluationError, match="test or validation split"):
        validate_question(validation, contract, expected_split="train")
    with pytest.raises(S1GenerationError, match="unknown S1 history policy"):
        _new_state(question, history_policy="retain_everything")


def test_failure_injection_is_metadata_driven_and_not_prompt_visible() -> None:
    question, answer = _sample(trace="S1.1.8")
    expected = answer["expected_messages"]
    runtime = _Runtime([expected[1]["content"]])
    outputs = iter([_call_text(expected[0]), _final_text(expected[2])])
    visible_prompts: list[str] = []

    def generate(messages, tools, turn_index):
        visible_prompts.append(json.dumps(messages, sort_keys=True))
        return next(outputs)

    prediction = orchestrate_s1_trajectory(
        question,
        contract=load_contract(),
        runtime=runtime,
        generate_turn=generate,
    )

    assert runtime.calls[0][2] is True
    assert prediction["injected_failure_count"] == 1
    assert prediction["generated_messages"][1]["content"] == {
        "status": "error",
        "provenance": None,
        "error_code": "BACKEND_UNAVAILABLE",
        "message": "registry unreachable",
    }
    assert all("tool_error" not in prompt for prompt in visible_prompts)
    assert all("expected_messages" not in prompt for prompt in visible_prompts)


def test_model_messages_preserve_multiturn_prefix_and_fold_harness() -> None:
    questions, _ = _panel()
    multiturn = next(
        row
        for row in questions
        if sum(message["role"] == "user" for message in row["messages"]) == 2
    )
    rendered = model_messages(multiturn)
    assert [message["role"] for message in rendered] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert isinstance(rendered[2]["content"], str)
    assert "thinking" not in rendered[2]
    assert rendered[-1]["content"] == multiturn["messages"][-1]["content"]

    harness = next(
        row
        for row in questions
        if any(message["role"] == "harness" for message in row["messages"])
    )
    harness_rendered = model_messages(harness)
    assert "harness" not in [message["role"] for message in harness_rendered]
    assert V3_HARNESS_HEADER in harness_rendered[1]["content"]


def test_prediction_serialization_preserves_six_field_wire_order() -> None:
    _, answer = _sample(trace="S1.6.1")
    expected_keys = [
        "status",
        "provenance",
        "result",
        "continuation_decision",
        "blocked_reason",
        "summary",
    ]
    row = {
        "record_id": answer["record_id"],
        "generated_messages": deepcopy(answer["expected_messages"]),
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.jsonl"
        write_prediction_jsonl(path, [row])
        loaded = json.loads(path.read_text(encoding="utf-8"))
    assert list(loaded["generated_messages"][-1]["content"]) == expected_keys


def test_evidence_ids_are_repeatable_and_shard_order_independent() -> None:
    questions, _ = _panel()
    full_indexes = {
        question["record_id"]: index for index, question in enumerate(questions)
    }

    def ids(order: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
        return {
            question["record_id"]: (
                deterministic_evidence_id(
                    channel="registry",
                    record_index=full_indexes[question["record_id"]],
                    call_index=0,
                ),
                deterministic_evidence_id(
                    channel="graph",
                    record_index=full_indexes[question["record_id"]],
                    call_index=1,
                ),
            )
            for question in order
        }

    forward = ids(questions)
    reverse = ids(list(reversed(questions)))
    strided = ids(questions[1::3] + questions[0::3] + questions[2::3])
    assert forward == reverse == strided
    assert len({value for pair in forward.values() for value in pair}) == 2 * len(
        questions
    )
    validate_evidence_namespace(questions)
    visible = set(
        re.findall(
            r"ev_(?:reg|mem)_[a-z0-9]{6}",
            json.dumps([question["messages"] for question in questions]),
        )
    )
    generated = {value for pair in forward.values() for value in pair}
    assert generated.isdisjoint(visible)


def test_padded_batch_width_does_not_reject_a_short_tool_call() -> None:
    question, answer = _sample(trace="S1.1.2")
    expected = answer["expected_messages"]
    runtime = _Runtime([expected[1]["content"]])
    state = TrajectoryState(
        question=question,
        prompt_messages=model_messages(question),
    )
    advance_trajectory(
        state,
        {
            "raw_generation": _call_text(expected[0]),
            # The inherited batch helper reports padded tensor width here.
            "generated_tokens": 4096,
            "action_tokens": 32,
            "prompt_tokens": 100,
        },
        runtime=runtime,
        contract=load_contract(),
        max_tool_call_tokens=1024,
    )
    assert state.stop_reason is None
    assert state.executed_tool_calls == 1


def test_checkpoint_binds_all_prompt_affecting_tokenizer_files() -> None:
    tokenizer_path = (
        REPO_ROOT
        / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/"
        "plain_base_tokenizer/tokenizer"
    )
    files = tokenizer_artifact_hashes(tokenizer_path)
    expected = "be21ce55523f893dc49e5ec7433f627372eebc1b9ac39dcdca5af8c7561e8d2f"
    assert stable_sha256(files) == expected
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        (checkpoint / "tp_adapter_manifest.json").write_text(
            json.dumps(
                {"identity": {"tokenizer_artifacts_sha256": expected}}
            ),
            encoding="utf-8",
        )
        assert validate_tokenizer_artifact_binding(checkpoint, files) == expected
        tampered = dict(files)
        tampered["chat_template.jinja"] = "0" * 64
        try:
            validate_tokenizer_artifact_binding(checkpoint, tampered)
        except S1GenerationError:
            pass
        else:
            raise AssertionError("tampered tokenizer artifacts were accepted")


def test_base_model_receipt_and_cross_shard_identity_are_bound() -> None:
    with TemporaryDirectory() as directory:
        base_model = Path(directory)
        receipt_identity = {
            "schema_version": "mentor-rl-base-model-artifact-v2",
            "weight_shard_count": 0,
            "files": {},
        }
        expected = stable_sha256(receipt_identity)
        receipt = {
            **receipt_identity,
            "base_model_artifact_sha256": expected,
        }
        (base_model / ".mentor_model_identity_sha256").write_text(
            expected + "\n", encoding="utf-8"
        )
        (base_model / ".mentor_base_model_artifact.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        assert validate_base_model_artifact_binding(
            base_model, expected
        ) == expected

    generation_config = {
        "evidence_id_policy": EVIDENCE_ID_POLICY,
        "network_policy": NETWORK_POLICY,
    }
    identity = {
        "evaluation_contract": "contract",
        "test_panel_id": "panel",
        "method_id": "method",
        "train_run_id": "run",
        "base_model_identity_sha256": "1" * 64,
        "base_model_artifact_sha256": "2" * 64,
        "checkpoint_identity_sha256": "3" * 64,
        "tokenizer_manifest_sha256": "4" * 64,
        "tokenizer_artifacts_sha256": "5" * 64,
        "generation_bundle_sha256": "6" * 64,
        "source_index_sha256": "7" * 64,
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False,
    }
    common = _validated_shard_identity(identity)
    tampered = dict(identity)
    tampered["tokenizer_artifacts_sha256"] = "8" * 64
    try:
        _validated_shard_identity(tampered, expected=common)
    except S1GenerationError:
        pass
    else:
        raise AssertionError("cross-shard tokenizer drift was accepted")


def test_network_guard_denies_dns_and_connect_in_isolated_process() -> None:
    code = r'''
import socket
from scripts.run_world_model_s1_exact_generation import (
    S1GenerationError,
    TrajectoryState,
    install_network_guard,
    prediction_from_state,
)

state = TrajectoryState(question={"record_id": "row"}, prompt_messages=[])
assert prediction_from_state(state)["network_used_false"] is False
try:
    prediction_from_state(state, network_guarded=True)
except S1GenerationError:
    pass
else:
    raise AssertionError("unguarded prediction claimed offline execution")
install_network_guard()
assert prediction_from_state(
    state, network_guarded=True
)["network_used_false"] is True
for operation in (
    lambda: socket.getaddrinfo("example.com", 443),
    lambda: socket.socket().connect(("127.0.0.1", 9)),
):
    try:
        operation()
    except S1GenerationError:
        pass
    else:
        raise AssertionError("network operation escaped the audit guard")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
