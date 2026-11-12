"""Test S2 structured scoring, coverage, identities, and report-only mode."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import evaluate_world_model_s2 as scorer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(scorer.canonical_json(row) + "\n" for row in rows))


def _panel(root: Path) -> tuple[Path, str, list[dict], list[dict]]:
    """Create two answer-separated S2 rows with and without a tool call."""

    root.mkdir()
    panel_id = "a" * 64
    call = {"name": "get_edge", "arguments": {"gene_ids": ["A", "B"]}}
    response = {
        "name": "get_edge",
        "content": {"status": "resolved", "result": {"edge_exists": True}},
    }
    questions = [
        {
            "row_id": "row-tool",
            "messages": [
                {"role": "system", "content": "Use exact graph evidence."},
                {"role": "harness", "content": "{}"},
                {"role": "user", "content": "Check the edge."},
            ],
            "tools": [],
            "metadata": {
                "schema_version": "mentor-rl-world-model-s2-corpus-v1",
                "tool_contract_version": "s2-test",
                "question_family": "S2.1",
                "split": "test",
            },
        },
        {
            "row_id": "row-defer",
            "messages": [
                {"role": "system", "content": "Use exact graph evidence."},
                {"role": "harness", "content": "{}"},
                {"role": "user", "content": "Handle the missing input."},
            ],
            "tools": [],
            "metadata": {
                "schema_version": "mentor-rl-world-model-s2-corpus-v1",
                "tool_contract_version": "s2-test",
                "question_family": "S2.2",
                "split": "test",
            },
        },
    ]
    answers = [
        {
            "row_id": "row-tool",
            "scoring_metadata": {
                "question_family": "S2.1",
                "catalog_case_id": "S2.1.1",
            },
            "target": {
                "status": "resolved",
                "result": {"edge_exists": True},
                "continuation_decision": "stop",
                "provenance": {"registry": [], "graph": [{"evidence_id": "ev-1"}]},
                "blocked_reason_required": False,
                "expected_tool_calls": [call],
                "expected_tool_responses": [response],
            },
        },
        {
            "row_id": "row-defer",
            "scoring_metadata": {
                "question_family": "S2.2",
                "catalog_case_id": "S2.2.1",
            },
            "target": {
                "status": "invalid_input",
                "result": {"issue_code": "MISSING_SCOPE"},
                "continuation_decision": "defer",
                "provenance": {"registry": [], "graph": []},
                "blocked_reason_required": True,
                "expected_tool_calls": [],
                "expected_tool_responses": [],
            },
        },
    ]
    questions_path = root / "test_questions.jsonl"
    answers_path = root / "test_answer_key.jsonl"
    _write_jsonl(questions_path, questions)
    _write_jsonl(answers_path, answers)
    answers_path.chmod(0o600)
    manifest = {
        "schema_version": scorer.EVALUATOR_SCHEMA_VERSION,
        "dataset_id": "s2_production_v1_8k",
        "evaluation_contract": scorer.EVALUATION_CONTRACT,
        "source_corpus": {"panel_id": panel_id},
        "contract_routes": {"S2.1": {}, "S2.2": {}},
        "test": {
            "questions_path": questions_path.name,
            "questions_sha256": scorer.sha256_file(questions_path),
            "answer_key_path": answers_path.name,
            "answer_key_sha256": scorer.sha256_file(answers_path),
            "test_panel_id": panel_id,
            "row_count": 2,
            "family_counts": {"S2.1": 1, "S2.2": 1},
            "case_counts": {"S2.1.1": 1, "S2.2.1": 1},
        },
        "scoring": {"target_fields": list(scorer.SCORED_TARGET_KEYS)},
        "answer_separation": {
            "public_questions_end_at_current_user": True,
            "generation_reads_private_answer_keys": False,
        },
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, scorer.sha256_file(manifest_path), questions, answers


def _perfect_predictions(answers: list[dict]) -> list[dict]:
    call = deepcopy(answers[0]["target"]["expected_tool_calls"][0])
    tool_final = answers[0]["target"]
    defer_final = answers[1]["target"]
    return [
        {
            "row_id": "row-tool",
            "generated_messages": [
                {"role": "assistant", "thinking": "Call once.", "tool_call": call},
                {
                    "role": "tool",
                    "name": "get_edge",
                    "content": deepcopy(
                        answers[0]["target"]["expected_tool_responses"][0]["content"]
                    ),
                },
                {
                    "role": "assistant",
                    "thinking": "Use the result.",
                    "content": {
                        "status": tool_final["status"],
                        "provenance": deepcopy(tool_final["provenance"]),
                        "result": deepcopy(tool_final["result"]),
                        "continuation_decision": tool_final["continuation_decision"],
                        "blocked_reason": None,
                        "summary": "Free wording is deliberately different.",
                    },
                },
            ],
            "generations": [],
            "stop_reason": "final",
            "trajectory_error": None,
            "tool_call_count": 1,
            "executed_tool_call_count": 1,
            "network_used_false": True,
        },
        {
            "row_id": "row-defer",
            "generated_messages": [
                {
                    "role": "assistant",
                    "thinking": "The scope is missing.",
                    "content": {
                        "status": defer_final["status"],
                        "provenance": deepcopy(defer_final["provenance"]),
                        "result": deepcopy(defer_final["result"]),
                        "continuation_decision": defer_final["continuation_decision"],
                        "blocked_reason": "Any nonempty corrective wording is valid.",
                        "summary": "Different free summary.",
                    },
                }
            ],
            "generations": [],
            "stop_reason": "final",
            "trajectory_error": None,
            "tool_call_count": 0,
            "executed_tool_call_count": 0,
            "network_used_false": True,
        },
    ]


def _generation(
    root: Path,
    predictions: list[dict],
    *,
    evaluator_sha256: str,
) -> tuple[Path, Path, dict]:
    predictions_path = root / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    config = {"do_sample": False, "seed": 900913}
    manifest = {
        "schema_version": scorer.PREDICTION_SCHEMA_VERSION,
        "evaluation_contract": scorer.EVALUATION_CONTRACT,
        "test_panel_id": "a" * 64,
        "method_id": "oss120b-plain-base-tokenizer-lora-r32",
        "train_run_id": "s2-train-run",
        "base_model_identity_sha256": "1" * 64,
        "base_model_artifact_sha256": "2" * 64,
        "checkpoint_identity_sha256": "3" * 64,
        "tokenizer_manifest_sha256": "4" * 64,
        "tokenizer_artifacts_sha256": "5" * 64,
        "generation_bundle_sha256": "6" * 64,
        "evaluator_manifest_sha256": evaluator_sha256,
        "tool_fixtures_sha256": "7" * 64,
        "generation_config": config,
        "generation_config_sha256": scorer.stable_sha256(config),
        "reads_private_answer_keys": False,
        "record_count": len(predictions),
        "predictions_sha256": scorer.sha256_file(predictions_path),
        "elapsed_seconds": 1.0,
    }
    manifest["manifest_sha256"] = scorer.stable_sha256(manifest)
    manifest_path = root / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return predictions_path, manifest_path, manifest


def test_exact_structured_score_ignores_free_prose_and_does_not_score_tool_responses(tmp_path):
    manifest_path, manifest_sha256, _, answers = _panel(tmp_path / "eval")
    predictions = _perfect_predictions(answers)
    predictions_path, generation_path, _ = _generation(
        tmp_path,
        predictions,
        evaluator_sha256=manifest_sha256,
    )
    metrics = scorer.evaluate_s2_test(
        evaluator_manifest_path=manifest_path,
        evaluator_manifest_sha256=manifest_sha256,
        predictions_path=predictions_path,
        generation_manifest_path=generation_path,
        output_dir=tmp_path / "report",
        gate_profile=scorer.REPORT_ONLY_GATE_PROFILE,
    )
    assert metrics["complete_coverage"] is True
    assert metrics["overall"]["structured_accuracy_count"] == 2
    assert metrics["overall"]["structured_accuracy_rate"] == 1.0
    assert metrics["families"]["S2.1"]["row_count"] == 1
    assert metrics["cases"]["S2.2.1"]["row_count"] == 1
    assert metrics["predictions_sha256"] == scorer.sha256_file(predictions_path)
    assert metrics["tool_fixtures_sha256"] == "7" * 64
    assert metrics["gate"]["passed"] is None
    assert metrics["gate"]["minimum_structured_accuracy"] is None

    changed = deepcopy(predictions[0])
    changed["generated_messages"][1]["content"] = {"changed": True}
    with pytest.raises(scorer.S2EvaluationError, match="fixtures changed"):
        scorer.score_s2_record(
            {"question_family": "S2.1"},
            answers[0],
            changed,
        )


def test_wrong_call_and_model_error_are_failed_rows_not_harness_aborts(tmp_path):
    manifest_path, manifest_sha256, questions, answers = _panel(tmp_path / "eval")
    predictions = _perfect_predictions(answers)
    predictions[0]["generated_messages"][0]["tool_call"]["arguments"] = {"wrong": True}
    predictions[1]["trajectory_error"] = "model_final_parse_error"
    generation = {
        **{key: "x" for key in scorer.PUBLICATION_IDENTITY_KEYS[:2]},
        **{key: str(index) * 64 for index, key in enumerate(scorer.PUBLICATION_IDENTITY_KEYS[2:], 1)},
        "test_panel_id": "a" * 64,
        "manifest_sha256": "f" * 64,
        "generation_config_sha256": "d" * 64,
        "predictions_sha256": "e" * 64,
        "tool_fixtures_sha256": "7" * 64,
    }
    by_id = {row["row_id"]: row for row in predictions}
    answer_by_id = {row["row_id"]: row for row in answers}
    question_by_id = {
        row["row_id"]: {"question_family": row["metadata"]["question_family"]}
        for row in questions
    }
    report, metrics = scorer.build_s2_report(
        evaluator_manifest_sha256=manifest_sha256,
        generation=generation,
        questions=question_by_id,
        answers=answer_by_id,
        predictions=by_id,
        gate_profile=scorer.REPORT_ONLY_GATE_PROFILE,
        minimum_structured_accuracy=None,
    )
    rows = {row["row_id"]: row for row in report["records"]}
    assert rows["row-tool"]["exact_tool_calls"] is False
    assert rows["row-tool"]["status_accuracy"] is True
    assert rows["row-tool"]["structured_accuracy"] is False
    assert rows["row-defer"]["model_error"] is True
    assert rows["row-defer"]["structured_accuracy"] is False
    assert metrics["overall"]["structured_accuracy_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("stop_reason", "generation_limit"), ("executed_tool_call_count", 0)],
)
def test_harness_counters_must_match_a_complete_final_trajectory(
    tmp_path,
    field,
    value,
):
    _, _, _, answers = _panel(tmp_path / "eval")
    prediction = _perfect_predictions(answers)[0]
    prediction[field] = value
    score = scorer.score_s2_record(
        {"question_family": "S2.1"},
        answers[0],
        prediction,
    )
    assert score["valid_trajectory"] is False
    assert score["structured_accuracy"] is False


def test_incomplete_predictions_and_changed_private_key_fail_closed(tmp_path):
    manifest_path, manifest_sha256, _, answers = _panel(tmp_path / "eval")
    predictions = _perfect_predictions(answers)[:1]
    predictions_path, generation_path, _ = _generation(
        tmp_path,
        predictions,
        evaluator_sha256=manifest_sha256,
    )
    with pytest.raises(scorer.S2EvaluationError, match="incomplete"):
        scorer.evaluate_s2_test(
            evaluator_manifest_path=manifest_path,
            evaluator_manifest_sha256=manifest_sha256,
            predictions_path=predictions_path,
            generation_manifest_path=generation_path,
            output_dir=tmp_path / "report",
            gate_profile=scorer.REPORT_ONLY_GATE_PROFILE,
        )
    with (manifest_path.parent / "test_answer_key.jsonl").open("a") as destination:
        destination.write("{}\n")
    with pytest.raises(scorer.S2EvaluationError, match="panel"):
        scorer.load_evaluator_panel(manifest_path, manifest_sha256)


def test_gate_thresholds_are_explicit_and_report_only_never_claims_pass(tmp_path):
    manifest_path, manifest_sha256, _, answers = _panel(tmp_path / "eval")
    predictions_path, generation_path, _ = _generation(
        tmp_path,
        _perfect_predictions(answers),
        evaluator_sha256=manifest_sha256,
    )
    with pytest.raises(scorer.S2EvaluationError, match="cannot claim"):
        scorer.evaluate_s2_test(
            evaluator_manifest_path=manifest_path,
            evaluator_manifest_sha256=manifest_sha256,
            predictions_path=predictions_path,
            generation_manifest_path=generation_path,
            output_dir=tmp_path / "report-only",
            gate_profile=scorer.REPORT_ONLY_GATE_PROFILE,
            minimum_structured_accuracy=0.95,
        )
    metrics = scorer.evaluate_s2_test(
        evaluator_manifest_path=manifest_path,
        evaluator_manifest_sha256=manifest_sha256,
        predictions_path=predictions_path,
        generation_manifest_path=generation_path,
        output_dir=tmp_path / "gated",
        gate_profile="structured_95_test",
        minimum_structured_accuracy=0.95,
    )
    assert metrics["gate"]["mode"] == "gate"
    assert metrics["gate"]["passed"] is True


def test_wandb_publication_identity_requires_every_exact_pin():
    generation = {
        **{key: "identity" for key in scorer.PUBLICATION_IDENTITY_KEYS[:2]},
        **{key: "a" * 64 for key in scorer.PUBLICATION_IDENTITY_KEYS[2:]},
    }
    scorer.validate_publication_identity(generation, dict(generation))
    changed = dict(generation)
    changed["checkpoint_identity_sha256"] = "b" * 64
    with pytest.raises(scorer.S2EvaluationError, match="checkpoint"):
        scorer.validate_publication_identity(generation, changed)


def test_wandb_publication_resumes_the_training_run_with_report_only_scalars(
    monkeypatch,
):
    calls = {}

    class Run:
        disabled = False
        id = "s2-train-run"
        name = "training"
        entity = "entity"
        project = "project"
        url = "https://example.invalid/run"

        def log(self, payload):
            calls["payload"] = payload

        def finish(self):
            calls["finished"] = True

    def init(**kwargs):
        calls["init"] = kwargs
        return Run()

    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_PROJECT", "project")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    receipt = scorer.publish_wandb(
        {
            "train_run_id": "s2-train-run",
            "overall": {"structured_accuracy_rate": 0.5},
            "complete_coverage": True,
            "gate": {"passed": None},
        }
    )
    assert calls["init"]["id"] == "s2-train-run"
    assert calls["init"]["resume"] == "must"
    assert calls["payload"] == {
        "test/s2/structured_accuracy": 0.5,
        "test/s2/complete_coverage": 1,
    }
    assert calls["finished"] is True
    assert receipt["id"] == "s2-train-run"
