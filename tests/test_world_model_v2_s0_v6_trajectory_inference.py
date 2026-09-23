"""Test v6 tool-trajectory inference and score results."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import scripts.run_world_model_v2_s0_exact_generation as generation_module
from scripts.build_world_model_v2_s0_generation_bundle import (
    BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
    FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
)
from scripts.evaluate_world_model_v2_s0 import (
    TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION,
    parse_final_answer,
    parse_tool_trajectory_call,
    score_tool_trajectory_record,
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


REASON = "I need to resolve TP53 against the pinned registry."
FINAL = "The Ensembl gene ID for TP53 is ENSG00000141510."
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
    "<|start|>assistant<|channel|>final<|message|>"
    f"{FINAL}<|return|>"
)


class _ToolResult:
    """Store one fake local tool result."""

    def __init__(self) -> None:
        self.payload = dict(TOOL_PAYLOAD)
        self.provenance = {"network_used": False}


class _ToolRuntime:
    """Return the fixed result for the expected call."""

    def execute(self, name: str, arguments: object) -> _ToolResult:
        assert name == TOOL_CALL["name"]
        assert arguments == TOOL_CALL["arguments"]
        return _ToolResult()


def _question() -> dict[str, object]:
    """Return one public trajectory question."""

    return {"record_id": "record-01"}


def _answer() -> dict[str, object]:
    """Return one private trajectory answer."""

    return {
        "record_id": "record-01",
        "fact_id": "fact-01",
        "family": "human_symbol_to_ensembl",
        "expected_tool_call": dict(TOOL_CALL),
        "expected_tool_payload": dict(TOOL_PAYLOAD),
        "expected_assistant_thinking": REASON,
        "expected_final_answer": FINAL,
    }


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
        "assistant_final": FINAL,
        "final_disallowed_special_tokens": [],
    }


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


def test_parse_final_answer_accepts_one_final_turn() -> None:
    """Parse one complete final answer turn."""

    answer, error = parse_final_answer(ENCODED_FINAL)

    assert answer == FINAL
    assert error is None


def test_score_accepts_one_complete_trajectory() -> None:
    """Score one complete and exact trajectory."""

    scored = score_tool_trajectory_record(
        _question(),
        _answer(),
        _prediction(),
        runtime=_ToolRuntime(),
    )

    assert scored["reasoning_present"] is True
    assert scored["reasoning_exact"] is True
    assert scored["valid_single_tool_call"] is True
    assert scored["exact_tool_argument"] is True
    assert scored["payload_exact"] is True
    assert scored["valid_final_answer"] is True
    assert scored["final_answer_exact"] is True
    assert scored["valid_tool_trajectory"] is True
    assert scored["network_used_false"] is True


def test_score_reports_one_wrong_final_answer() -> None:
    """Keep a valid trajectory separate from final answer accuracy."""

    prediction = _prediction()
    wrong = "TP53 maps to the wrong identifier."
    prediction["encoded_final_answer"] = (
        "<|start|>assistant<|channel|>final<|message|>"
        f"{wrong}<|return|>"
    )
    prediction["assistant_final"] = wrong

    scored = score_tool_trajectory_record(
        _question(),
        _answer(),
        prediction,
        runtime=_ToolRuntime(),
    )

    assert scored["valid_tool_trajectory"] is True
    assert scored["valid_final_answer"] is True
    assert scored["final_answer_exact"] is False


def test_score_rejects_one_missing_final_answer() -> None:
    """If a trajectory stops after the tool result, reject it."""

    prediction = _prediction()
    prediction.pop("encoded_final_answer")
    prediction.pop("assistant_final")

    scored = score_tool_trajectory_record(
        _question(),
        _answer(),
        prediction,
        runtime=_ToolRuntime(),
    )

    assert scored["final_parse_error"] == "final_answer_is_absent"
    assert scored["valid_final_answer"] is False
    assert scored["valid_tool_trajectory"] is False


def _write_v6_bundle(path: Path) -> list[dict[str, object]]:
    """Write one small public v6 bundle."""

    rows = [
        {
            "record_id": "record-01",
            "system": "Use the tool, then answer.",
            "question": "Resolve TP53.",
            "input": {"gene_symbol": "TP53"},
            "metadata": {"question_family": "human_symbol_to_ensembl"},
            "provenance": {"fact_role": "registry_population"},
            "split": "test",
            "tools": [],
        }
    ]
    questions_path = path / "questions.jsonl"
    write_jsonl(questions_path, rows)
    manifest = {
        "schema_version": BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
        "evaluation_contract": FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "identifier_registry": {
            "path": (
                "data/world_model_v2/sources/"
                "ensembl_116_human_gene_ids_and_symbols.json"
            ),
            "id": (
                "sha256:"
                "e67a4a1d8839c7f8ea568c3040c18941"
                "a189f65afeed66dfeedb03c6e305c4ab"
            ),
            "sha256": (
                "e67a4a1d8839c7f8ea568c3040c18941"
                "a189f65afeed66dfeedb03c6e305c4ab"
            ),
        },
        "record_count": 1,
        "record_ids_sha256": stable_sha256(["record-01"]),
        "questions_sha256": sha256_file(questions_path),
        "reads_private_answer_keys": False,
    }
    manifest["bundle_sha256"] = stable_sha256(manifest)
    write_json(path / "manifest.json", manifest)
    return rows


def _write_v6_shard(path: Path, bundle_sha256: str) -> None:
    """Write one complete v6 prediction shard."""

    predictions_path = path / "predictions.jsonl"
    write_jsonl(predictions_path, [_prediction()])
    generation_config = {"do_sample": False, "batch_size": 1}
    manifest = {
        "schema_version": SHARD_PREDICTION_SCHEMA_VERSION,
        "evaluation_contract": FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "method_id": "test-method",
        "train_run_id": "test-run",
        "base_model_identity_sha256": "b" * 64,
        "checkpoint_identity_sha256": "c" * 64,
        "tokenizer_manifest_sha256": "d" * 64,
        "generation_bundle_sha256": bundle_sha256,
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
    write_json(path / "generation_manifest.json", manifest)


def test_merge_selects_the_v6_trajectory_schema(tmp_path: Path) -> None:
    """Select the v6 output schema for trajectory shards."""

    bundle_dir = tmp_path / "bundle"
    _write_v6_bundle(bundle_dir)
    bundle = json.loads(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    shards_dir = tmp_path / "shards"
    _write_v6_shard(
        shards_dir / "shard-00000",
        str(bundle["bundle_sha256"]),
    )

    result = merge_prediction_shards(
        bundle_root=bundle_dir,
        shards_root=shards_dir,
        output_dir=tmp_path / "merged",
        expected_shard_count=1,
    )

    assert result["manifest"]["schema_version"] == (
        TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION
    )
    predictions = [
        json.loads(line)
        for line in (tmp_path / "merged/predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert predictions == [_prediction()]


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
        """Return the required boundary for each generation pass."""

        self.template_calls.append(
            {
                "messages": messages,
                **options,
            }
        )
        if options["enable_thinking"] is False:
            return "final-prefix<|start|>assistant<|channel|>final<|message|>"
        return "tool-prefix<|start|>assistant"

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
    """Execute the public tool result before the final model pass."""

    bundle_dir = tmp_path / "bundle"
    _write_v6_bundle(bundle_dir)
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    tokenizer_hash = "d" * 64
    checkpoint_hash = "c" * 64
    fake_torch = types.SimpleNamespace(
        manual_seed=lambda seed: None,
        no_grad=lambda: _NoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
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
            {"checkpoint_identity_sha256": checkpoint_hash},
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
    assert model.generate_calls == 2
    assert [
        call["enable_thinking"] for call in tokenizer.template_calls
    ] == [True, False]
    final_messages = tokenizer.template_calls[1]["messages"]
    assert isinstance(final_messages, list)
    assert final_messages[-1] == {
        "role": "tool",
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
    assert prediction["assistant_final"] == FINAL
    assert prediction["raw_final_generation"] == ENCODED_FINAL
