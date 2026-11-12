"""Check answer-free, exact-call S2 generation and ordered shard merging."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.world_model.s2_production_adapter import tools_sha256
from scripts import build_world_model_s2_generation_bundle as bundle_builder
from scripts import run_world_model_s2_exact_generation as runner
from scripts import run_world_model_v2_s0_exact_generation as s0_runner
from scripts.build_world_model_s2_eval_bundle import (
    EVALUATION_CONTRACT,
    EVALUATOR_SCHEMA_VERSION,
    EXPECTED_FAMILIES,
    ordered_json,
    sha256_file,
    stable_sha256,
)


def _question(row_id: str = "row-1", family: str = "S2.1") -> dict:
    system = "Use the visible fixture and return the ordered structured result."
    tools = [{
        "type": "function",
        "function": {
            "name": "inspect_ordered_fixture",
            "description": "Return one deterministic offline fixture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "z_scope": {"type": "string"},
                    "a_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["z_scope", "a_ids"],
                "additionalProperties": False,
            },
        },
    }]
    return {
        "row_id": row_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "harness", "content": '{"observations":[]}'},
            {"role": "user", "content": "Inspect the requested fixture."},
        ],
        "tools": tools,
        "metadata": {
            "schema_version": "mentor-rl-world-model-s2-corpus-v1",
            "tool_contract_version": f"synthetic-{family}",
            "question_family": family,
            "split": "test",
        },
    }


def _fixture(row_id: str = "row-1") -> dict:
    return {
        "schema_version": bundle_builder.TOOL_FIXTURE_SCHEMA_VERSION,
        "row_id": row_id,
        "steps": [{
            "tool_call": {
                "name": "inspect_ordered_fixture",
                "arguments": {"z_scope": "brain", "a_ids": ["B", "A"]},
            },
            "tool_response": {
                "name": "inspect_ordered_fixture",
                "content": {
                    "status": "resolved",
                    "provenance": {"registry": [], "graph": []},
                    "result": {"z_scope": "brain", "a_ids": ["B", "A"]},
                },
            },
        }],
    }


def _answer(question: dict) -> dict:
    fixture = _fixture(question["row_id"])
    return {
        "row_id": question["row_id"],
        "scoring_metadata": {"question_family": question["metadata"]["question_family"]},
        "target": {
            "status": "resolved",
            "result": {"private_final_sentinel": question["row_id"]},
            "continuation_decision": "stop",
            "provenance": {"registry": [], "graph": []},
            "blocked_reason_required": False,
            "expected_tool_calls": [fixture["steps"][0]["tool_call"]],
            "expected_tool_responses": [fixture["steps"][0]["tool_response"]],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(ordered_json(row) + "\n" for row in rows), encoding="utf-8")


def _synthetic_evaluator(root: Path) -> tuple[Path, str, list[dict]]:
    root.mkdir()
    questions, answers, routes = [], [], {}
    for index, family in enumerate(EXPECTED_FAMILIES, start=1):
        question = _question(f"row-{index}", family)
        questions.append(question)
        answers.append(_answer(question))
        routes[family] = {
            "contract_version": question["metadata"]["tool_contract_version"],
            "system_prompt_sha256": hashlib.sha256(
                question["messages"][0]["content"].encode()
            ).hexdigest(),
            "tools_sha256": tools_sha256(question["tools"]),
        }
    questions_path = root / "test_questions.jsonl"
    answers_path = root / "test_answer_key.jsonl"
    _write_jsonl(questions_path, questions)
    _write_jsonl(answers_path, answers)
    answers_path.chmod(0o600)
    ids = [row["row_id"] for row in questions]
    manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "dataset_id": "synthetic-s2",
        "evaluation_contract": EVALUATION_CONTRACT,
        "contract_routes": routes,
        "test": {
            "test_panel_id": "a" * 64,
            "questions_path": questions_path.name,
            "questions_sha256": sha256_file(questions_path),
            "answer_key_path": answers_path.name,
            "answer_key_sha256": sha256_file(answers_path),
            "row_count": len(questions),
            "record_ids_sha256": stable_sha256(ids),
            "family_counts": {family: 1 for family in EXPECTED_FAMILIES},
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, sha256_file(manifest_path), questions


def _built_bundle(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    evaluator, identity, questions = _synthetic_evaluator(tmp_path / "eval")
    output = tmp_path / "generation-bundle"
    manifest = bundle_builder.build_generation_bundle(
        evaluator_manifest_path=evaluator,
        evaluator_manifest_sha256=identity,
        output_dir=output,
    )
    return output, manifest, questions


def _call_text(call: dict, thinking: str = "Use the exact requested fixture.") -> str:
    return (
        "<|channel|>analysis<|message|>" + thinking
        + "<|end|><|start|>assistant to=functions." + call["name"]
        + "<|channel|>commentary json<|message|>"
        + ordered_json(call["arguments"]) + "<|call|>"
    )


def _final_text(*, summary: str = "The exact fixture was used.") -> str:
    content = {
        "status": "resolved",
        "provenance": {"registry": [], "graph": []},
        "result": {"fixture_used": True},
        "continuation_decision": "stop",
        "blocked_reason": None,
        "summary": summary,
    }
    return (
        "<|channel|>analysis<|message|>Return the structured result."
        "<|end|><|start|>assistant<|channel|>final<|message|>"
        + ordered_json(content) + "<|return|>"
    )


def test_generation_bundle_preserves_order_and_excludes_final_targets(tmp_path):
    bundle, manifest, source_questions = _built_bundle(tmp_path)
    emitted = [json.loads(line) for line in (bundle / "questions.jsonl").read_text().splitlines()]
    fixtures = [json.loads(line) for line in (bundle / "tool_fixtures.jsonl").read_text().splitlines()]

    assert manifest["questions_sha256"] == manifest["source_questions_sha256"]
    assert manifest["record_count"] == len(EXPECTED_FAMILIES)
    assert manifest["maximum_tool_calls"] == 1
    assert all(tools_sha256(row["tools"]) == manifest["contract_routes"][
        row["metadata"]["question_family"]]["tools_sha256"] for row in emitted)
    assert list(emitted[0]["tools"][0]["function"]["parameters"]["properties"]) == [
        "z_scope", "a_ids",
    ]
    assert list(fixtures[0]["steps"][0]["tool_call"]["arguments"]) == ["z_scope", "a_ids"]
    assert "private_final_sentinel" not in (bundle / "tool_fixtures.jsonl").read_text()
    assert emitted == source_questions
    assert manifest["reads_private_answer_keys"] is False
    assert manifest["tool_fixtures_contain_final_targets"] is False
    loaded, loaded_questions, loaded_fixtures = runner.load_generation_bundle(bundle)
    assert loaded == manifest
    assert loaded_questions == emitted
    assert list(loaded_fixtures) == [row["row_id"] for row in emitted]


def test_exact_fixture_call_and_final_form_native_generated_suffix():
    question, fixture = _question(), _fixture()
    outputs = iter([_call_text(fixture["steps"][0]["tool_call"]), _final_text()])
    prompts = []

    def generate(messages, tools, turn_index):
        prompts.append(deepcopy(list(messages)))
        assert list(tools) == question["tools"]
        return next(outputs)

    prediction = runner.orchestrate_s2_trajectory(
        question, fixture, generate_turn=generate
    )
    assert prediction["stop_reason"] == "final"
    assert prediction["trajectory_error"] is None
    assert prediction["tool_call_count"] == prediction["executed_tool_call_count"] == 1
    assert [message["role"] for message in prediction["generated_messages"]] == [
        "assistant", "tool", "assistant",
    ]
    assert prediction["generated_messages"][0]["tool_call"] == fixture["steps"][0]["tool_call"]
    assert prediction["generated_messages"][1] == {
        "role": "tool", **fixture["steps"][0]["tool_response"],
    }
    assert prompts[1][-2]["thinking"] == "Use the exact requested fixture."


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (_call_text({"name": "inspect_ordered_fixture", "arguments": {"wrong": True}}),
         "tool_call_mismatch"),
        ("not a GPT-OSS assistant action", "parse_error"),
    ],
)
def test_invalid_model_actions_fail_only_the_row(raw, reason):
    prediction = runner.orchestrate_s2_trajectory(
        _question(), _fixture(), generate_turn=lambda messages, tools, turn: raw
    )
    assert prediction["stop_reason"] == reason
    assert isinstance(prediction["trajectory_error"], str)
    assert prediction["network_used_false"] is False


def test_infrastructure_generation_errors_still_abort():
    def fail(messages, tools, turn):
        raise RuntimeError("accelerator failed")

    with pytest.raises(RuntimeError, match="accelerator failed"):
        runner.orchestrate_s2_trajectory(_question(), _fixture(), generate_turn=fail)


def test_model_expanded_final_prompt_budget_fails_only_that_row(monkeypatch):
    """Retained generated reasoning may stop its row, not the whole shard."""
    call = _call_text(_fixture()["steps"][0]["tool_call"])
    monkeypatch.setattr(
        runner, "_render_prompt",
        lambda state, **kwargs: "post-tool-prompt" if state.executed_tool_calls else "initial-prompt",
    )
    monkeypatch.setattr(
        runner,
        "_generation_prompt_token_count",
        lambda tokenizer, value: 9000 if value == "post-tool-prompt" else 10,
    )
    monkeypatch.setattr(runner, "_prompt_token_count", lambda *args, **kwargs: 10)
    monkeypatch.setattr(
        runner,
        "generate_prompt_batch_with_oom_backoff",
        lambda **kwargs: [{
            "raw_generation": call, "prompt_tokens": 100,
            "generated_tokens": 10,
        }],
    )

    class Tokenizer:
        eos_token = None
        pad_token = None

    predictions = runner.generate_trajectory_batch(
        [_question()], fixtures={"row-1": _fixture()}, model=object(),
        tokenizer=Tokenizer(), device="cpu", torch_module=object(),
        max_new_tokens=4096, max_total_tokens=8192,
        max_tool_call_tokens=1024, max_final_prompt_tokens=8191,
        reasoning_effort="low", shard_index=0,
    )
    assert predictions[0]["stop_reason"] == "prompt_limit"
    assert predictions[0]["trajectory_error"] == "max_final_prompt_tokens_exceeded"


def test_near_limit_prompt_cannot_reduce_another_rows_generation_budget(monkeypatch):
    """Batch-wide minimum remaining context must never truncate a short row."""
    short, long = _question("short"), _question("long")
    fixtures = {
        row_id: {
            "schema_version": bundle_builder.TOOL_FIXTURE_SCHEMA_VERSION,
            "row_id": row_id,
            "steps": [],
        }
        for row_id in ("short", "long")
    }
    monkeypatch.setattr(
        runner, "_render_prompt",
        lambda state, **kwargs: str(state.question["row_id"]),
    )
    token_counts = {"short": 1000, "long": 8150}
    monkeypatch.setattr(
        runner, "_generation_prompt_token_count",
        lambda tokenizer, value: token_counts.get(value, 10),
    )
    monkeypatch.setattr(runner, "_prompt_token_count", lambda *args, **kwargs: 10)
    calls = []

    def generate(**kwargs):
        cap = min(
            kwargs["max_new_tokens"],
            min(kwargs["max_total_tokens"] - token_counts[prompt]
                for prompt in kwargs["prompts"]),
        )
        calls.append((list(kwargs["prompts"]), cap))
        return [{
            "raw_generation": _final_text(summary=f"{prompt} complete."),
            "prompt_tokens": token_counts[prompt],
            "generated_tokens": cap,
        } for prompt in kwargs["prompts"]]

    monkeypatch.setattr(runner, "generate_prompt_batch_with_oom_backoff", generate)

    class Tokenizer:
        eos_token = None
        pad_token = None

    predictions = runner.generate_trajectory_batch(
        [short, long], fixtures=fixtures, model=object(), tokenizer=Tokenizer(),
        device="cpu", torch_module=object(), max_new_tokens=4096,
        max_total_tokens=8192, max_tool_call_tokens=1024,
        max_final_prompt_tokens=8191, reasoning_effort="low", shard_index=0,
    )
    assert calls == [(["short"], 4096), (["long"], 42)]
    assert [row["stop_reason"] for row in predictions] == ["final", "final"]


def test_static_initial_prompt_overflow_aborts_the_harness(monkeypatch):
    monkeypatch.setattr(runner, "_render_prompt", lambda *args, **kwargs: "too-long")
    monkeypatch.setattr(
        runner, "_generation_prompt_token_count", lambda *args, **kwargs: 8192
    )

    class Tokenizer:
        eos_token = None
        pad_token = None

    with pytest.raises(runner.S2GenerationError, match="initial S2 prompt"):
        runner.generate_trajectory_batch(
            [_question()], fixtures={"row-1": _fixture()}, model=object(),
            tokenizer=Tokenizer(), device="cpu", torch_module=object(),
            max_new_tokens=4096, max_total_tokens=8192,
            max_tool_call_tokens=1024, max_final_prompt_tokens=8191,
            reasoning_effort="low", shard_index=0,
        )


def _prediction(row_id: str) -> dict:
    return {
        "row_id": row_id,
        "generated_messages": [{
            "role": "assistant",
            "thinking": "Return the result.",
            "content": {
                "status": "resolved", "provenance": {}, "result": {},
                "continuation_decision": "stop", "blocked_reason": None,
                "summary": "Done.",
            },
        }],
        "generations": [],
        "stop_reason": "final",
        "trajectory_error": None,
        "tool_call_count": 0,
        "executed_tool_call_count": 0,
        "network_used_false": True,
    }


def _shard_manifest(bundle: dict, index: int, count: int, predictions_path: Path) -> dict:
    config = {"history_policy": "retain_current_turn", "do_sample": False}
    common = {
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": bundle["test_panel_id"],
        "method_id": "method", "train_run_id": "run",
        "base_model_identity_sha256": "1" * 64,
        "base_model_artifact_sha256": "2" * 64,
        "checkpoint_identity_sha256": "3" * 64,
        "tokenizer_manifest_sha256": "4" * 64,
        "tokenizer_artifacts_sha256": "5" * 64,
        "evaluator_manifest_sha256": bundle["source_evaluator_manifest_sha256"],
        "tool_fixtures_sha256": bundle["tool_fixtures_sha256"],
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": config,
        "generation_config_sha256": stable_sha256(config),
        "reads_private_answer_keys": False,
    }
    manifest = {
        "schema_version": runner.PREDICTION_SHARD_SCHEMA_VERSION,
        **common,
        "record_count": len(predictions_path.read_text().splitlines()),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": float(index + 1),
        "full_record_count": len(EXPECTED_FAMILIES),
        "shard_index": index,
        "shard_count": count,
        "shard_strategy": runner.SHARD_STRATEGY,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    return manifest


def test_merger_restores_panel_order_and_rejects_config_identity_drift(tmp_path):
    bundle_root, bundle, questions = _built_bundle(tmp_path)
    shards_root = tmp_path / "shards"
    for index in range(2):
        shard = shards_root / f"shard-{index:05d}"
        shard.mkdir(parents=True)
        predictions = [_prediction(row["row_id"]) for row in questions[index::2]]
        path = shard / "predictions.jsonl"
        runner.write_prediction_jsonl(path, predictions)
        manifest = _shard_manifest(bundle, index, 2, path)
        (shard / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    output = tmp_path / "merged"
    result = runner.merge_prediction_shards(
        bundle_root=bundle_root,
        shards_root=shards_root,
        output_dir=output,
        expected_shard_count=2,
    )
    merged = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert [row["row_id"] for row in merged] == [row["row_id"] for row in questions]
    assert result["manifest"]["record_count"] == len(questions)
    assert result["manifest"]["parallelism"]["completed_shards"] == 2

    changed_path = shards_root / "shard-00001/generation_manifest.json"
    changed = json.loads(changed_path.read_text())
    changed["generation_config_sha256"] = "f" * 64
    changed["manifest_sha256"] = stable_sha256({
        key: value for key, value in changed.items() if key != "manifest_sha256"
    })
    changed_path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
    with pytest.raises(runner.S2GenerationError, match="shard 1 is invalid"):
        runner.merge_prediction_shards(
            bundle_root=bundle_root,
            shards_root=shards_root,
            output_dir=tmp_path / "rejected",
            expected_shard_count=2,
        )


def test_s2_checkpoint_exposure_and_standard_artifact_identity(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "run_contract").mkdir(parents=True)
    identity = {
        "method_id": "method", "run_id": "run",
        "corpus_manifest_sha256": "1" * 64,
        "tokenizer_manifest_sha256": "2" * 64,
        "model_identity_sha256": "3" * 64,
    }
    (checkpoint / "tp_adapter_manifest.json").write_text(json.dumps({
        "format": runner.CHECKPOINT_FORMAT, "identity": identity,
    }))
    (checkpoint / "adapter_config.json").write_text("{}\n")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    exposure = {
        "schema_version": runner.EXPOSURE_SCHEMA_VERSION,
        "status": "complete", "method_id": "method", "run_id": "run",
        "logical_exposure": {"all_eligible_train_rows_exposed": True},
        "exposure_contract": {"satisfied": True},
    }
    exposure["manifest_sha256"] = stable_sha256(exposure)
    exposure_path = checkpoint / "run_contract/training_exposure.json"
    exposure_path.write_text(json.dumps(exposure))
    artifact, run_id = runner.validate_checkpoint(
        checkpoint, method_id="method", corpus_manifest_sha256="1" * 64,
        tokenizer_manifest_sha256="2" * 64, model_identity_sha256="3" * 64,
    )
    assert run_id == "run"
    assert artifact["checkpoint_identity_sha256"] == stable_sha256({
        "files": artifact["files"],
    })
    assert set(artifact["files"]) == {
        "adapter_config.json", "adapter_model.safetensors", "tp_adapter_manifest.json",
        "run_contract/training_exposure.json",
    }

    exposure["schema_version"] = "mentor-rl-s1-training-exposure-v1"
    exposure["manifest_sha256"] = stable_sha256({
        key: value for key, value in exposure.items() if key != "manifest_sha256"
    })
    exposure_path.write_text(json.dumps(exposure))
    with pytest.raises(runner.S2GenerationError, match="complete exposure"):
        runner.validate_checkpoint(
            checkpoint, method_id="method", corpus_manifest_sha256="1" * 64,
            tokenizer_manifest_sha256="2" * 64, model_identity_sha256="3" * 64,
        )


class GptOssExperts:
    """Stand in for the transformers expert module by class name."""

    training = False


class _Attention:
    training = False


class _Model:
    def __init__(self, layers: int, experts: int) -> None:
        self.config = type("Config", (), {"num_hidden_layers": layers})()
        self.parts = [GptOssExperts() for _ in range(experts)] + [_Attention()]

    def modules(self):
        return iter(self.parts)


def test_routed_experts_switch_only_expert_modules():
    """Inference uses the routed expert loop that training used."""

    model = _Model(layers=2, experts=2)
    assert runner.use_routed_experts(model) == 2
    assert [part.training for part in model.parts] == [True, True, False]
    with pytest.raises(runner.S2GenerationError, match="one GPT-OSS expert module per layer"):
        runner.use_routed_experts(_Model(layers=2, experts=1))


class _Seq(list):
    """A token row with the tensor methods the generation helper uses."""

    shape = property(lambda self: (len(self),))

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return _Seq(value) if isinstance(key, slice) else value


class _Batch:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def to(self, device):
        return self

    def sum(self, dim):
        return type("Sums", (list,), {"tolist": lambda sums: list(sums)})(sum(row) for row in self.rows)


class _Tokenizer:
    eos_token_id = pad_token_id = 0

    def __call__(self, prompts, return_tensors, padding):
        prompts = [prompts] if isinstance(prompts, str) else prompts
        width = max(len(prompt.split()) for prompt in prompts)
        masks = [[0] * (width - len(p.split())) + [1] * len(p.split()) for p in prompts]
        return {"input_ids": _Batch(masks), "attention_mask": _Batch(masks)}

    def decode(self, tokens, skip_special_tokens):
        return "x" * len(tokens)


class _GenerationModel:
    generation_config = type("Config", (), {"eos_token_id": 0})()

    def __init__(self, oom_batch_sizes):
        self.calls, self.oom_batch_sizes = [], oom_batch_sizes

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.shape, kwargs.get("prefill_chunk_size")))
        if input_ids.shape[0] in self.oom_batch_sizes:
            raise RuntimeError("HIP out of memory")
        return [_Seq(row + [1]) for row in input_ids.rows]


_TORCH = type("Torch", (), {"OutOfMemoryError": MemoryError, "cuda": None,
                            "no_grad": staticmethod(lambda: __import__("contextlib").nullcontext())})


@pytest.mark.parametrize(
    ("prompts", "oom", "expected"),
    [
        (["a b c d e", "a b c"], set(), [((2, 5), None)]),
        (["a b c"], set(), [((1, 3), None)]),
        (["a b c d e", "a b c"], {2}, [((2, 5), None), ((1, 5), 4), ((1, 3), None)]),
    ],
)
def test_only_a_lone_long_prompt_uses_chunked_prefill(prompts, oom, expected):
    """Chunked prefill ignores left padding, so batches must stay unchunked."""

    model = _GenerationModel(oom)
    results = s0_runner.generate_prompt_batch_with_oom_backoff(
        model=model, tokenizer=_Tokenizer(), device=None, torch_module=_TORCH,
        prompts=prompts, record_ids=[f"row-{i}" for i in range(len(prompts))],
        max_new_tokens=4, max_total_tokens=64, stage="s2_assistant_turn_1",
        shard_index=0, prefill_chunk_size=4)
    assert model.calls == expected
    assert len(results) == len(prompts)
