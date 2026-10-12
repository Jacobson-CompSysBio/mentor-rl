"""Test S0 trajectory parsing and the two-pass generation loop."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import scripts.run_world_model_v2_s0_exact_generation as generation_module
from runtime.world_model.contract import load_contract, tool_definitions
from scripts.build_world_model_v2_s0_generation_bundle import (
    FULL_REGISTRY_BUNDLE_V3_SCHEMA_VERSION,
    FULL_REGISTRY_V3_EVALUATION_CONTRACT,
)
from scripts.evaluate_world_model_v2_s0 import (
    V3_PREDICTION_SCHEMA_VERSION,
    parse_shared_final_action,
    parse_tool_trajectory_call,
)
from scripts.run_world_model_v2_s0_exact_generation import (
    SHARD_PREDICTION_SCHEMA_VERSION,
    SHARD_STRATEGY,
    generate_predictions,
    merge_prediction_shards,
    sha256_file,
    stable_sha256,
    write_json,
    write_jsonl,
)


REASON = "TP53 is the supplied symbol, so I will look it up."
FINAL_THINKING = "The registry gives one ID for TP53, so I should stop."
FINAL_ANSWER = json.dumps(
    {
        "status": "resolved",
        "provenance": None,
        "result": {"gene_id": "ENSG00000141510", "gene_symbol": "TP53"},
        "continuation_decision": "stop",
        "blocked_reason": None,
        "summary": "TP53 is <GENE:ENSG00000141510|TP53>.",
    },
    separators=(",", ":"),
)
TOOL_CALL = {
    "name": "lookup_human_gene_symbol",
    "arguments": {"gene_symbol": "TP53"},
}
TOOL_PAYLOAD = {
    "status": "resolved",
    "gene_symbol": "TP53",
    "gene_id": "ENSG00000141510",
}
ENCODED_CALL = (
    "<|start|>assistant<|channel|>analysis<|message|>"
    f"{REASON}<|end|><|start|>assistant "
    "to=functions.lookup_human_gene_symbol"
    "<|channel|>commentary json<|message|>"
    '{"gene_symbol":"TP53"}<|call|>'
)
ENCODED_FINAL = (
    "<|start|>assistant<|channel|>analysis<|message|>"
    f"{FINAL_THINKING}<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
    f"{FINAL_ANSWER}<|return|>"
)


def test_parse_trajectory_requires_analysis_and_one_call() -> None:
    """Parse one analysis turn followed by one tool call."""

    call, thinking, error, direct_answer = parse_tool_trajectory_call(
        ENCODED_CALL
    )

    assert call == TOOL_CALL
    assert thinking == REASON
    assert error is None
    assert direct_answer is False


def test_parse_trajectory_rejects_a_call_without_analysis() -> None:
    """Reject a tool call that has no analysis turn."""

    encoded = (
        "<|start|>assistant "
        "to=functions.lookup_human_gene_symbol"
        "<|channel|>commentary json<|message|>"
        '{"gene_symbol":"TP53"}<|call|>'
    )

    call, thinking, error, direct_answer = parse_tool_trajectory_call(encoded)

    assert call is None
    assert thinking is None
    assert error == "analysis_count"
    assert direct_answer is False


def test_parse_final_action_requires_thinking_and_one_answer() -> None:
    """Parse the final thinking turn and the final JSON answer."""

    thinking, answer, error = parse_shared_final_action(ENCODED_FINAL)

    assert thinking == FINAL_THINKING
    assert answer == FINAL_ANSWER
    assert error is None


def test_parse_final_action_rejects_an_answer_without_thinking() -> None:
    """The final turn must think before it answers."""

    encoded = (
        "<|start|>assistant<|channel|>final<|message|>"
        f"{FINAL_ANSWER}<|return|>"
    )

    assert parse_shared_final_action(encoded) == (
        None,
        None,
        "final_analysis_count",
    )


def _bundle_rows() -> list[dict[str, object]]:
    """Return one public question that uses the contract tools."""

    return [
        {
            "record_id": "record-01",
            "system": "Use the tool, then answer.",
            "question": "Resolve TP53.",
            "input": {"gene_symbol": "TP53"},
            "metadata": {"question_family": "human_symbol_to_ensembl"},
            "provenance": {"fact_role": "registry_population"},
            "split": "test",
            "tools": tool_definitions(load_contract()),
        }
    ]


def _write_bundle(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Write one small public bundle and return its manifest and rows."""

    rows = _bundle_rows()
    questions_path = path / "questions.jsonl"
    write_jsonl(questions_path, rows)
    manifest = {
        "schema_version": FULL_REGISTRY_BUNDLE_V3_SCHEMA_VERSION,
        "evaluation_contract": FULL_REGISTRY_V3_EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "identifier_registry": {
            "path": "data/source_index.json.gz",
            "id": "sha256:" + "f" * 64,
            "sha256": "f" * 64,
        },
        "record_count": len(rows),
        "record_ids_sha256": stable_sha256(
            [row["record_id"] for row in rows]
        ),
        "questions_sha256": sha256_file(questions_path),
        "reads_private_answer_keys": False,
    }
    manifest["bundle_sha256"] = stable_sha256(manifest)
    write_json(path / "manifest.json", manifest)
    return manifest, rows


def _prediction() -> dict[str, object]:
    """Return one complete generated trajectory."""

    return {
        "record_id": "record-01",
        "encoded_prediction": ENCODED_CALL,
        "assistant_thinking": REASON,
        "tool_payload": dict(TOOL_PAYLOAD),
        "tool_error": None,
        "network_used_false": True,
        "disallowed_special_tokens": [],
        "encoded_final_answer": ENCODED_FINAL,
        "assistant_final": FINAL_ANSWER,
        "assistant_final_explanation": FINAL_THINKING,
        "final_disallowed_special_tokens": [],
    }


def test_merge_writes_the_v3_prediction_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A merged result uses the prediction schema that the scorer reads."""

    bundle_dir = tmp_path / "bundle"
    bundle, rows = _write_bundle(bundle_dir)
    monkeypatch.setattr(
        generation_module, "load_bundle", lambda root: (bundle, rows)
    )
    shard_dir = tmp_path / "shards/shard-00000"
    predictions_path = shard_dir / "predictions.jsonl"
    write_jsonl(predictions_path, [_prediction()])
    generation_config = {"do_sample": False, "batch_size": 1}
    manifest = {
        "schema_version": SHARD_PREDICTION_SCHEMA_VERSION,
        "evaluation_contract": FULL_REGISTRY_V3_EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "method_id": "test-method",
        "train_run_id": "test-run",
        "base_model_identity_sha256": "b" * 64,
        "checkpoint_identity_sha256": "c" * 64,
        "tokenizer_manifest_sha256": "d" * 64,
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False,
        "record_count": 1,
        "full_record_count": 1,
        "shard_index": 0,
        "shard_count": 1,
        "shard_strategy": SHARD_STRATEGY,
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": 1.0,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(shard_dir / "generation_manifest.json", manifest)

    result = merge_prediction_shards(
        bundle_root=bundle_dir,
        shards_root=tmp_path / "shards",
        output_dir=tmp_path / "merged",
        expected_shard_count=1,
    )

    assert result["manifest"]["schema_version"] == V3_PREDICTION_SCHEMA_VERSION
    merged = [
        json.loads(line)
        for line in (tmp_path / "merged/predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert merged == [_prediction()]


class _ToolResult:
    """Store one fake source-index tool result."""

    def __init__(self) -> None:
        self.output = dict(TOOL_PAYLOAD)


class _ToolRuntime:
    """Return the fixed result for the expected call."""

    def execute(self, name: str, arguments: object) -> _ToolResult:
        assert name == TOOL_CALL["name"]
        assert arguments == TOOL_CALL["arguments"]
        return _ToolResult()


class _FakeTensor:
    """Provide the tensor operations that the generation loop uses."""

    def __init__(self, data: object) -> None:
        self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the stored list."""

        if not isinstance(self.data, list):
            return ()
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def __getitem__(self, key: object) -> object:
        """Return one scalar or one wrapped list."""

        assert isinstance(self.data, list)
        value = self.data[key]
        return _FakeTensor(value) if isinstance(value, list) else value

    def to(self, device: object) -> "_FakeTensor":
        """Keep fake tensors on the CPU."""

        assert device == "cpu"
        return self

    def sum(self, dim: int) -> "_FakeTensor":
        """Sum each row for an attention mask."""

        assert dim == 1
        assert isinstance(self.data, list)
        return _FakeTensor([sum(row) for row in self.data])

    def tolist(self) -> object:
        """Return the stored list."""

        return self.data


class _FakeTokenizer:
    """Record prompt options and decode two fixed model turns."""

    eos_token = "<|return|>"
    pad_token = "<|end|>"
    pad_token_id = 0

    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []
        self.padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        **options: object,
    ) -> str:
        """Return the assistant boundary that both passes end with."""

        self.template_calls.append({"messages": messages, **options})
        return "prefix<|start|>assistant"

    def __call__(
        self,
        prompts: object,
        *,
        return_tensors: str,
        padding: bool = False,
    ) -> dict[str, _FakeTensor]:
        """Return one small token batch."""

        assert return_tensors == "pt"
        del padding
        prompt_rows = prompts if isinstance(prompts, list) else [prompts]
        rows = [[1, 2, 3] for _ in prompt_rows]
        return {
            "input_ids": _FakeTensor(rows),
            "attention_mask": _FakeTensor([[1, 1, 1] for _ in rows]),
        }

    def decode(
        self,
        generated: _FakeTensor,
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode the fixed tool or final turn."""

        assert skip_special_tokens is False
        assert generated.data in ([101], [202])
        return ENCODED_CALL if generated.data == [101] else ENCODED_FINAL


class _FakeModel:
    """Return one fixed token in each generation pass."""

    def __init__(self) -> None:
        self.generate_calls = 0
        self.generation_config = types.SimpleNamespace(eos_token_id=999)

    def parameters(self):
        """Return one fake parameter on the CPU."""

        return iter([types.SimpleNamespace(device="cpu")])

    def generate(self, **inputs: object) -> _FakeTensor:
        """Append the token for the selected generation pass."""

        self.generate_calls += 1
        input_ids = inputs["input_ids"]
        assert isinstance(input_ids, _FakeTensor)
        assert isinstance(input_ids.data, list)
        token = 101 if self.generate_calls == 1 else 202
        return _FakeTensor([list(row) + [token] for row in input_ids.data])


class _NoGrad:
    """Provide the no-gradient context protocol."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        del exc
        return False


def test_generate_predictions_runs_both_trajectory_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Replay the tool call, then generate the final thinking and answer."""

    bundle_dir = tmp_path / "bundle"
    bundle, rows = _write_bundle(bundle_dir)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    tokenizer_hash = "d" * 64
    fake_torch = types.SimpleNamespace(
        manual_seed=lambda seed: None,
        no_grad=lambda: _NoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        generation_module, "load_bundle", lambda root: (bundle, rows)
    )
    monkeypatch.setattr(
        generation_module.WorldModelToolRuntime,
        "from_source_index",
        classmethod(lambda cls, *args, **kwargs: _ToolRuntime()),
    )
    monkeypatch.setattr(
        generation_module,
        "validated_tokenizer_manifest",
        lambda path: {"manifest_sha256": tokenizer_hash},
    )
    monkeypatch.setattr(
        generation_module,
        "tokenizer_artifact_hashes",
        lambda path: {"tokenizer.json": "e" * 64},
    )
    monkeypatch.setattr(
        generation_module,
        "validate_checkpoint",
        lambda *args, **kwargs: (
            {"checkpoint_identity_sha256": "c" * 64},
            "test-train-run",
        ),
    )
    monkeypatch.setattr(
        generation_module,
        "load_backend",
        lambda **kwargs: (model, tokenizer),
    )

    result = generate_predictions(
        bundle_root=bundle_dir,
        base_model_path=tmp_path / "model",
        checkpoint_path=tmp_path / "checkpoint",
        tokenizer_path=tmp_path / "tokenizer",
        tokenizer_manifest_path=tmp_path / "tokenizer-manifest.json",
        output_dir=tmp_path / "output",
        method_id="test-method",
        corpus_manifest_sha256="a" * 64,
        tokenizer_manifest_sha256=tokenizer_hash,
        model_identity_sha256="b" * 64,
        max_new_tokens=384,
        max_total_tokens=1024,
        enable_thinking=True,
        reasoning_effort="low",
        seed=900913,
        batch_size=1,
        local_files_only=True,
    )

    assert result["status"] == "complete"
    assert result["manifest"]["schema_version"] == V3_PREDICTION_SCHEMA_VERSION
    assert model.generate_calls == 2
    # Both passes think first, as in training.
    assert [
        call["enable_thinking"] for call in tokenizer.template_calls
    ] == [True, True]
    for call in tokenizer.template_calls:
        assert all(
            tool["function"].get("description") for tool in call["tools"]
        )
    final_messages = tokenizer.template_calls[1]["messages"]
    assert isinstance(final_messages, list)
    assert final_messages[-1] == {
        "role": "tool",
        "name": TOOL_CALL["name"],
        "content": TOOL_PAYLOAD,
    }
    prediction = json.loads(
        (tmp_path / "output/predictions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert prediction["assistant_thinking"] == REASON
    assert prediction["tool_payload"] == TOOL_PAYLOAD
    assert prediction["network_used_false"] is True
    assert prediction["assistant_final_explanation"] == FINAL_THINKING
    assert prediction["assistant_final"] == FINAL_ANSWER
    assert prediction["final_disallowed_special_tokens"] == []
