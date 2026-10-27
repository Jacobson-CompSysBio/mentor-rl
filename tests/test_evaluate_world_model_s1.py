from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from runtime.world_model.contract import load_contract
from scripts.evaluate_world_model_s1 import (
    _wilson_interval,
    build_s1_report,
    score_s1_record,
    validate_generated_evidence_policy,
    validate_publication_identity,
)
from scripts.run_world_model_s1_exact_generation import deterministic_evidence_id
from scripts.world_model_s1_eval_common import (
    S1EvaluationError,
    S1_FAMILIES,
    read_jsonl,
    validate_answer_row,
    validate_question,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_ROOT = REPO_ROOT / "data/world_model_v3/eval/s1_smoke"


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


@lru_cache(maxsize=1)
def _direction_panel() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    contract = load_contract()
    raw_questions = [
        row
        for row in read_jsonl(
            REPO_ROOT / "data/world_model_v3/eval/s1/test_questions.jsonl"
        )
        if row["metadata"]["question_family"] == "direction_conventions"
    ]
    questions = [validate_question(row, contract) for row in raw_questions]
    by_id = {row["record_id"]: row for row in questions}
    answers = {
        row["record_id"]: validate_answer_row(
            row, by_id[row["record_id"]], contract
        )
        for row in read_jsonl(
            REPO_ROOT / "data/world_model_v3/eval/s1/test_answer_key.jsonl"
        )
        if row["record_id"] in by_id
    }
    return questions, answers


def _perfect_prediction(answer: dict[str, Any]) -> dict[str, Any]:
    messages = deepcopy(answer["expected_messages"])
    call_count = sum("tool_call" in message for message in messages)
    failure_count = sum(
        message.get("role") == "tool"
        and message["content"].get("error_code") == "BACKEND_UNAVAILABLE"
        for message in messages
    )
    return {
        "record_id": answer["record_id"],
        "generated_messages": messages,
        "generations": [],
        "stop_reason": "final",
        "trajectory_error": None,
        "tool_call_count": call_count,
        "executed_tool_call_count": call_count,
        "injected_failure_count": failure_count,
        "network_used_false": True,
    }


def _generation_identity() -> dict[str, Any]:
    return {
        "test_panel_id": "panel",
        "method_id": "method",
        "train_run_id": "run",
        "manifest_sha256": "a" * 64,
        "base_model_identity_sha256": "b" * 64,
        "base_model_artifact_sha256": "c" * 64,
        "checkpoint_identity_sha256": "d" * 64,
        "tokenizer_manifest_sha256": "e" * 64,
        "tokenizer_artifacts_sha256": "f" * 64,
        "source_index_sha256": "1" * 64,
        "generation_bundle_sha256": "2" * 64,
    }


def _build_metrics(
    scores: list[dict[str, Any]],
    *,
    source_trace_floor: float = 0.0,
    family_floor: float = 0.0,
    final_floor: float = 0.0,
    maximum_direct_rate: float = 1.0,
    tool_followup_floor: float = 0.0,
) -> dict[str, Any]:
    _, metrics = build_s1_report(
        scores,
        generation=_generation_identity(),
        evaluator_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="c" * 64,
        minimum_valid_trajectory=0.0,
        minimum_reasoning_present=0.0,
        minimum_exact_tool_selection=0.0,
        minimum_exact_tool_arguments=0.0,
        minimum_tool_payload_consistency=0.0,
        minimum_final_answer_accuracy=final_floor,
        minimum_family_accuracy=family_floor,
        minimum_source_trace_envelope_accuracy=source_trace_floor,
        maximum_unexpected_direct_answer_rate=maximum_direct_rate,
        minimum_tool_followup_thinking_present=tool_followup_floor,
    )
    return metrics


def _sample(*, trace: str) -> tuple[dict[str, Any], dict[str, Any]]:
    questions, answers = _panel()
    question = next(
        row for row in questions if row["metadata"]["source_trace"] == trace
    )
    return question, answers[question["record_id"]]


def test_exact_two_call_trace_scores_every_behavior_dimension() -> None:
    question, answer = _sample(trace="S1.3.5")
    score = score_s1_record(
        question,
        answer,
        _perfect_prediction(answer),
        contract=load_contract(),
    )

    assert score["expected_tool_call_count"] == 2
    assert score["valid_trajectory"] is True
    assert score["exact_tool_selection"] is True
    assert score["exact_tool_arguments"] is True
    assert score["tool_payload_consistency"] is True
    assert score["structured_final_json"] is True
    assert score["final_answer_accuracy"] is True
    assert score["exact_behavior"] is True
    assert score["exact_trace"] is True


def test_tool_argument_payload_and_final_json_fail_independently() -> None:
    question, answer = _sample(trace="S1.1.2")
    prediction = _perfect_prediction(answer)
    prediction["generated_messages"][0]["tool_call"]["arguments"] = {
        "gene_symbol": "WRONG"
    }
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["exact_tool_selection"] is True
    assert score["exact_tool_arguments"] is False
    assert score["tool_payload_consistency"] is True
    assert score["final_answer_accuracy"] is True
    assert score["exact_behavior"] is False

    prediction = _perfect_prediction(answer)
    prediction["generated_messages"][1]["content"]["status"] = "not_found"
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["exact_tool_arguments"] is True
    assert score["tool_payload_consistency"] is False
    assert score["final_answer_accuracy"] is True

    prediction = _perfect_prediction(answer)
    content = prediction["generated_messages"][-1]["content"]
    prediction["generated_messages"][-1]["content"] = dict(
        sorted(content.items())
    )
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["structured_final_json"] is False
    assert score["valid_trajectory"] is False


def test_nondeterministic_evidence_ids_are_normalized_for_exact_scoring() -> None:
    question, answer = _sample(trace="S1.1.2")
    prediction = _perfect_prediction(answer)

    def replace(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "evidence_id":
                    value[key] = "ev_reg_zzzzzz"
                else:
                    replace(item)
        elif isinstance(value, list):
            for item in value:
                replace(item)

    replace(prediction["generated_messages"])
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["tool_payload_consistency"] is True
    assert score["final_answer_accuracy"] is True


def test_invalid_or_unmatched_final_evidence_never_normalizes_to_exact() -> None:
    question, answer = _sample(trace="S1.1.2")
    prediction = _perfect_prediction(answer)
    prediction["generated_messages"][-1]["content"]["provenance"][
        "registry"
    ]["evidence_id"] = None
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["evidence_consistency"] is False
    assert score["structured_final_json"] is False
    assert score["final_answer_accuracy"] is False
    assert score["exact_behavior"] is False

    prediction = _perfect_prediction(answer)
    prediction["generated_messages"][-1]["content"]["provenance"][
        "registry"
    ]["evidence_id"] = "ev_reg_zzzzzz"
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["evidence_error"] == "final_registry_tool_source_mismatch"
    assert score["final_answer_accuracy"] is False


def test_final_can_reuse_only_exact_visible_harness_evidence() -> None:
    questions, answers = _panel()
    question = next(
        row
        for row in questions
        if any(message["role"] == "harness" for message in row["messages"])
        and not any(
            "tool_call" in message
            for message in answers[row["record_id"]]["expected_messages"]
        )
        and answers[row["record_id"]]["expected_messages"][-1]["content"][
            "provenance"
        ]
        is not None
    )
    answer = answers[question["record_id"]]
    prediction = _perfect_prediction(answer)
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["evidence_consistency"] is True

    prediction["generated_messages"][-1]["content"]["provenance"][
        "registry"
    ]["evidence_id"] = "ev_reg_zzzzzz"
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["evidence_consistency"] is False


def test_final_must_cite_gold_selected_source_not_any_visible_block() -> None:
    question, answer = _sample(trace="S1.1.5")
    prediction = _perfect_prediction(answer)
    harness = next(
        message["content"]
        for message in question["messages"]
        if message["role"] == "harness"
    )
    start = harness.index("{")
    harness_object, _ = json.JSONDecoder().raw_decode(harness[start:])
    prediction["generated_messages"][-1]["content"]["provenance"][
        "registry"
    ] = harness_object["provenance"]["registry"]
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["evidence_error"] == "final_registry_tool_source_mismatch"
    assert score["final_answer_accuracy"] is False


def test_two_call_and_partial_failure_final_bind_to_positional_payloads() -> None:
    for trace in ("S1.1.9", "S1.3.10"):
        question, answer = _sample(trace=trace)
        prediction = _perfect_prediction(answer)
        score = score_s1_record(
            question, answer, prediction, contract=load_contract()
        )
        assert score["evidence_consistency"] is True
        assert score["final_answer_accuracy"] is True

        provenance = prediction["generated_messages"][-1]["content"][
            "provenance"
        ]
        channel = "graph" if provenance.get("graph") is not None else "registry"
        prefix = "ev_mem_" if channel == "graph" else "ev_reg_"
        provenance[channel]["evidence_id"] = prefix + "zzzzzz"
        score = score_s1_record(
            question, answer, prediction, contract=load_contract()
        )
        assert score["evidence_consistency"] is False
        assert score["final_answer_accuracy"] is False


def test_failure_trace_requires_the_deterministic_backend_payload() -> None:
    question, answer = _sample(trace="S1.1.8")
    prediction = _perfect_prediction(answer)
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["expected_failure"] is True
    assert score["failure_harness_accuracy"] is True

    prediction["generated_messages"][1]["content"]["message"] = "different"
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["failure_harness_accuracy"] is False
    assert score["tool_payload_consistency"] is False

    prediction = _perfect_prediction(answer)
    prediction["generated_messages"][-1]["content"]["summary"] = (
        "A wrong failure summary."
    )
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["failure_harness_accuracy"] is True
    assert score["final_answer_accuracy"] is False


def test_failure_trace_binds_backend_error_to_call_position_and_name() -> None:
    question, answer = _sample(trace="S1.3.10")
    prediction = _perfect_prediction(answer)
    messages = prediction["generated_messages"]
    first_payload = deepcopy(messages[1]["content"])
    messages[1]["content"] = deepcopy(messages[3]["content"])
    messages[3]["content"] = first_payload
    score = score_s1_record(
        question, answer, prediction, contract=load_contract()
    )
    assert score["failure_harness_accuracy"] is False
    assert score["exact_behavior"] is False


def test_generated_evidence_policy_rejects_cross_record_nonce_reuse() -> None:
    questions, answers = _panel()
    selected = []
    for question in questions:
        payloads = [
            message
            for message in answers[question["record_id"]]["expected_messages"]
            if message.get("role") == "tool"
            and message.get("content", {}).get("status") != "error"
        ]
        if payloads:
            selected.append(question)
        if len(selected) == 2:
            break
    question_map = {row["record_id"]: row for row in selected}
    predictions = {
        row["record_id"]: _perfect_prediction(answers[row["record_id"]])
        for row in selected
    }
    successful_ids = []
    for record_index, row in enumerate(selected):
        call_index = 0
        for message in predictions[row["record_id"]]["generated_messages"]:
            if message.get("role") != "tool":
                continue
            content = message["content"]
            if content.get("status") == "error":
                call_index += 1
                continue
            provenance = content["provenance"]
            channel = next(key for key, value in provenance.items() if value)
            evidence_id = deterministic_evidence_id(
                channel=channel,
                record_index=record_index,
                call_index=call_index,
            )
            provenance[channel]["evidence_id"] = evidence_id
            successful_ids.append((message, channel, evidence_id))
            call_index += 1
    validate_generated_evidence_policy(
        question_map, predictions, load_contract()
    )
    second_message, second_channel, _ = successful_ids[1]
    second_message["content"]["provenance"][second_channel][
        "evidence_id"
    ] = successful_ids[0][2]
    try:
        validate_generated_evidence_policy(
            question_map, predictions, load_contract()
        )
    except S1EvaluationError:
        pass
    else:
        raise AssertionError("cross-record evidence reuse was accepted")


def test_publication_identity_must_match_every_launcher_pin() -> None:
    generation = _generation_identity()
    expected = {
        key: generation[key]
        for key in (
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
    }
    validate_publication_identity(generation, expected)
    mismatched = dict(expected)
    mismatched["train_run_id"] = "different-run"
    try:
        validate_publication_identity(generation, mismatched)
    except S1EvaluationError:
        pass
    else:
        raise AssertionError("mismatched W&B run identity was accepted")


def test_report_contains_all_families_and_source_trace_breakdowns() -> None:
    questions, answers = _panel()
    selected = {}
    for question in questions:
        selected.setdefault(question["metadata"]["question_family"], question)
    assert set(selected) == set(S1_FAMILIES)
    failure_question, _ = _sample(trace="S1.1.8")
    selected["failure_fixture"] = failure_question
    direction_questions, direction_answers = _direction_panel()
    contract = load_contract()
    direction_by_kind = {}
    for question in direction_questions:
        answer = direction_answers[question["record_id"]]
        score = score_s1_record(
            question,
            answer,
            _perfect_prediction(answer),
            contract=contract,
        )
        direction_by_kind.setdefault(score["direction_name_kind"], question)
    selected["direction_metric_variant"] = direction_by_kind[
        "metric_variant"
    ]
    selected["direction_shared_alias"] = direction_by_kind["shared_alias"]
    scores = [
        score_s1_record(
            question,
            (answers | direction_answers)[question["record_id"]],
            _perfect_prediction(
                (answers | direction_answers)[question["record_id"]]
            ),
            contract=contract,
        )
        for question in selected.values()
    ]
    generation = _generation_identity()
    report, metrics = build_s1_report(
        scores,
        generation=generation,
        evaluator_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="c" * 64,
        minimum_valid_trajectory=1.0,
        minimum_reasoning_present=1.0,
        minimum_exact_tool_selection=1.0,
        minimum_exact_tool_arguments=1.0,
        minimum_tool_payload_consistency=1.0,
        minimum_final_answer_accuracy=1.0,
        minimum_family_accuracy=1.0,
        minimum_source_trace_envelope_accuracy=1.0,
        maximum_unexpected_direct_answer_rate=0.0,
        minimum_tool_followup_thinking_present=1.0,
    )

    assert set(metrics["families"]) == set(S1_FAMILIES)
    assert metrics["failure_rows"]["row_count"] >= 1
    assert metrics["closed_book_rows"]["row_count"] >= 1
    assert metrics["gate"]["passed"] is True
    assert metrics["gate"]["checks"]["closed_book_zero_calls"] is True
    assert metrics["gate"]["checks"]["failure_harness_consistency"] is True
    assert metrics["gate"]["checks"]["network_used_false"] is True
    assert metrics["overall"]["exact_behavior_rate"] == 1.0
    assert len(report["records"]) == len(scores)


def test_required_harness_gates_are_not_absorbed_into_average_accuracy() -> None:
    questions, answers = _panel()
    direct = next(
        row for row in questions if row["metadata"]["book_mode"] == "closed_book"
    )
    failure = next(
        row
        for row in questions
        if row["metadata"].get("blocked_reason_code") == "tool_error"
    )
    scores = [
        score_s1_record(
            row,
            answers[row["record_id"]],
            _perfect_prediction(answers[row["record_id"]]),
            contract=load_contract(),
        )
        for row in (direct, failure)
    ]
    scores[0]["closed_book_zero_calls"] = False
    scores[1]["failure_harness_accuracy"] = False
    _, metrics = build_s1_report(
        scores,
        generation=_generation_identity(),
        evaluator_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="c" * 64,
        minimum_valid_trajectory=0.0,
        minimum_reasoning_present=0.0,
        minimum_exact_tool_selection=0.0,
        minimum_exact_tool_arguments=0.0,
        minimum_tool_payload_consistency=0.0,
        minimum_final_answer_accuracy=0.0,
        minimum_family_accuracy=0.0,
        minimum_source_trace_envelope_accuracy=0.0,
        maximum_unexpected_direct_answer_rate=1.0,
        minimum_tool_followup_thinking_present=0.0,
    )
    assert metrics["gate"]["checks"]["closed_book_zero_calls"] is False
    assert metrics["gate"]["checks"]["failure_harness_consistency"] is False
    assert metrics["gate"]["passed"] is False


def test_one_wrong_small_form_envelope_fails_the_source_trace_gate() -> None:
    question, answer = _sample(trace="S1.1.2")
    template = score_s1_record(
        question,
        answer,
        _perfect_prediction(answer),
        contract=load_contract(),
    )
    full_questions = read_jsonl(
        REPO_ROOT / "data/world_model_v3/eval/s1/test_questions.jsonl"
    )
    scores = []
    target = None
    for row in full_questions:
        score = deepcopy(template)
        metadata = row["metadata"]
        score.update(
            {
                "record_id": row["record_id"],
                "question_family": metadata["question_family"],
                "source_trace": metadata["source_trace"],
                "book_mode": metadata["book_mode"],
                "expected_failure": (
                    metadata.get("blocked_reason_code") == "tool_error"
                ),
                "closed_book_zero_calls": (
                    True if metadata["book_mode"] == "closed_book" else None
                ),
                "ambiguous_deferral_consistency": True,
                "false_defer": False,
                "unexpected_direct_answer": False,
                "tool_followup_thinking_present": True,
            }
        )
        if metadata["question_family"] in {
            "layer_tag_parsing",
            "layer_family_claims",
        }:
            score["expected_layer_families"] = ["bulk_pen"]
            score["generated_layer_families"] = ["bulk_pen"]
        if metadata["question_family"] == "direction_conventions":
            score.update(
                {
                    "direction_contract_accuracy": True,
                    "direction_truth_accuracy": True,
                    "expected_direction_metric": "synthetic_metric",
                    "expected_direction_polarity": "lower_stronger",
                    "expected_direction_truth": True,
                    "heldout_metric_name": str(
                        metadata.get("split_unit", "")
                    ).startswith("heldout_metric_name_"),
                    "heldout_metric_mapping_accuracy": True,
                }
            )
        scores.append(score)
        if metadata["source_trace"] == "S1.4.9" and target is None:
            target = score
    assert target is not None
    target["result_accuracy"] = False
    target["envelope_accuracy"] = False
    target["final_answer_accuracy"] = False
    target["exact_behavior"] = False
    target["exact_trace"] = False

    metrics = _build_metrics(
        scores,
        source_trace_floor=0.99,
        family_floor=0.99,
        final_floor=0.99,
    )
    assert metrics["source_traces"]["S1.4.9"][
        "envelope_accuracy_total"
    ] == 31
    assert metrics["overall"]["envelope_accuracy_rate"] >= 0.99
    assert metrics["gate"]["checks"]["final_answer_accuracy"] is True
    assert metrics["gate"]["checks"]["all_family_accuracy"] is True
    assert metrics["gate"]["checks"][
        "all_source_trace_envelope_accuracy"
    ] is False


def test_written_stage_hard_gates_are_explicit_and_false_defer_is_reported() -> None:
    ambiguous_question, ambiguous_answer = _sample(trace="S1.3.1")
    prediction = _perfect_prediction(ambiguous_answer)
    prediction["generated_messages"][-1]["content"][
        "blocked_reason"
    ] = "A different but nonempty reason."
    score = score_s1_record(
        ambiguous_question,
        ambiguous_answer,
        prediction,
        contract=load_contract(),
    )
    assert score["blocked_reason_accuracy"] is False

    prediction = _perfect_prediction(ambiguous_answer)
    prediction["generated_messages"][-1]["content"]["summary"] = (
        "A different nonempty summary."
    )
    score = score_s1_record(
        ambiguous_question,
        ambiguous_answer,
        prediction,
        contract=load_contract(),
    )
    assert score["summary_accuracy"] is False

    prediction = _perfect_prediction(ambiguous_answer)
    final_content = prediction["generated_messages"][-1]["content"]
    final_content["continuation_decision"] = "stop"
    final_content["blocked_reason"] = None
    score = score_s1_record(
        ambiguous_question,
        ambiguous_answer,
        prediction,
        contract=load_contract(),
    )
    assert score["ambiguous_deferral_consistency"] is False

    answerable_question, answerable_answer = _sample(trace="S1.1.2")
    prediction = _perfect_prediction(answerable_answer)
    final_content = prediction["generated_messages"][-1]["content"]
    final_content["continuation_decision"] = "defer"
    final_content["blocked_reason"] = "Unnecessarily deferred."
    score = score_s1_record(
        answerable_question,
        answerable_answer,
        prediction,
        contract=load_contract(),
    )
    assert score["false_defer"] is True

    questions, answers = _panel()
    perfect_scores = [
        score_s1_record(
            question,
            answers[question["record_id"]],
            _perfect_prediction(answers[question["record_id"]]),
            contract=load_contract(),
        )
        for question in questions
    ]
    assert _build_metrics(perfect_scores)["gate"]["passed"] is True

    adversarial = (
        ("ambiguous_deferral_consistency", "ambiguous_deferral_consistency"),
        ("blocked_reason_accuracy", "blocked_reason_consistency"),
        ("evidence_consistency", "evidence_consistency"),
        ("summary_accuracy", "summary_consistency"),
        ("structured_final_json", "structured_final_json"),
    )
    for field, check in adversarial:
        scores = deepcopy(perfect_scores)
        row = next(item for item in scores if item.get(field) is True)
        row[field] = False
        metrics = _build_metrics(scores)
        assert metrics["gate"]["checks"][check] is False

    scores = deepcopy(perfect_scores)
    failure = next(item for item in scores if item["expected_failure"])
    failure["final_answer_accuracy"] = False
    metrics = _build_metrics(scores)
    assert metrics["gate"]["checks"][
        "failure_final_answer_accuracy"
    ] is False

    scores = deepcopy(perfect_scores)
    answerable = next(item for item in scores if item["false_defer"] is False)
    answerable["false_defer"] = True
    metrics = _build_metrics(scores)
    assert metrics["overall"]["false_defer_count"] == 1
    assert metrics["overall"]["false_defer_rate"] > 0.0
    assert "false_defer" not in metrics["gate"]["checks"]


def test_direct_answer_and_tool_followup_thinking_have_explicit_gates() -> None:
    question, answer = _sample(trace="S1.1.2")
    direct = _perfect_prediction(answer)
    direct["generated_messages"] = [deepcopy(answer["expected_messages"][-1])]
    direct["tool_call_count"] = 0
    direct["executed_tool_call_count"] = 0
    direct_score = score_s1_record(
        question, answer, direct, contract=load_contract()
    )
    assert direct_score["unexpected_direct_answer"] is True

    missing_followup = _perfect_prediction(answer)
    missing_followup["generated_messages"][-1]["thinking"] = ""
    thinking_score = score_s1_record(
        question, answer, missing_followup, contract=load_contract()
    )
    assert thinking_score["tool_followup_thinking_present"] is False

    questions, answers = _panel()
    scores = [
        score_s1_record(
            row,
            answers[row["record_id"]],
            _perfect_prediction(answers[row["record_id"]]),
            contract=load_contract(),
        )
        for row in questions
    ]
    direct_row = next(
        item for item in scores if item["unexpected_direct_answer"] is False
    )
    direct_row["unexpected_direct_answer"] = True
    metrics = _build_metrics(scores, maximum_direct_rate=0.01)
    assert metrics["gate"]["checks"]["unexpected_direct_answer_rate"] is False

    scores = deepcopy(scores)
    for item in scores:
        if item["unexpected_direct_answer"] is True:
            item["unexpected_direct_answer"] = False
            break
    thinking_row = next(
        item
        for item in scores
        if item["tool_followup_thinking_present"] is True
    )
    thinking_row["tool_followup_thinking_present"] = False
    metrics = _build_metrics(scores, tool_followup_floor=0.99)
    assert metrics["gate"]["checks"][
        "tool_followup_thinking_present"
    ] is False


def test_family_absolute_contracts_reject_single_row_bypasses() -> None:
    contract = load_contract()

    def scored(trace: str, mutate: Any) -> dict[str, Any]:
        question, answer = _sample(trace=trace)
        prediction = _perfect_prediction(answer)
        mutate(prediction["generated_messages"])
        return score_s1_record(
            question, answer, prediction, contract=contract
        )

    def final_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
        return messages[-1]["content"]["result"]

    cases = [
        (
            "S1.1.3",
            lambda messages: final_result(messages).__setitem__(
                "gene_symbol", "TP53"
            ),
            "identifier_no_invention_consistency",
        ),
        (
            "S1.1.9",
            lambda messages: next(
                message["tool_call"]
                for message in messages
                if message.get("tool_call", {}).get("name")
                == "check_multiplex_membership"
            )["arguments"].__setitem__("gene_ids", ["ENSG00000000000"]),
            "membership_call_contract",
        ),
        (
            "S1.2.3",
            lambda messages: messages[0]["tool_call"]["arguments"].__setitem__(
                "gene_id", "ENSG00000229391"
            ),
            "identifier_lookup_argument_consistency",
        ),
        (
            "S1.2.1",
            lambda messages: final_result(messages).__setitem__(
                "gene_symbol", "WRONG"
            ),
            "identifier_payload_binding_consistency",
        ),
        (
            "S1.2.4",
            lambda messages: final_result(messages).__setitem__(
                "gene_symbol", "INVENTED"
            ),
            "null_symbol_consistency",
        ),
        (
            "S1.3.5",
            lambda messages: (
                final_result(messages).__setitem__(
                    "present_gene_ids", ["ENSG00000293487"]
                ),
                final_result(messages).__setitem__(
                    "gene_id", "ENSG00000293487"
                ),
            ),
            "ambiguous_choice_consistency",
        ),
        (
            "S1.3.8",
            lambda messages: (
                final_result(messages).__setitem__(
                    "candidate_gene_ids", ["ENSG00000000000"]
                ),
                final_result(messages).__setitem__(
                    "gene_id", "ENSG00000000000"
                ),
            ),
            "ambiguous_choice_consistency",
        ),
        (
            "S1.4.5",
            lambda messages: final_result(messages).__setitem__(
                "same_gene", True
            ),
            "same_gene_null_contract",
        ),
        (
            "S1.6.1",
            lambda messages: final_result(messages).__setitem__(
                "organ", "invented_organ"
            ),
            "no_inferred_null_consistency",
        ),
        (
            "S1.7.4",
            lambda messages: final_result(messages)[
                "supported_claims"
            ].append("holds_in_other_contexts"),
            "no_unsupported_claim_leakage",
        ),
        (
            "S1.9.2",
            lambda messages: final_result(messages).__setitem__(
                "existence_asserted", True
            ),
            "module_result_contract",
        ),
    ]
    for trace, mutate, field in cases:
        score = scored(trace, mutate)
        assert score[field] is False, trace
        assert score["family_contract_consistency"] is False, trace

    for trace in ("S1.1.7", "S1.2.10", "S1.2.12"):
        def add_call(messages: list[dict[str, Any]]) -> None:
            messages.insert(
                -1,
                {
                    "role": "assistant",
                    "thinking": "I will call a forbidden tool.",
                    "tool_call": {
                        "name": "lookup_human_gene_symbol",
                        "arguments": {"gene_symbol": "WRONG"},
                    },
                },
            )

        score = scored(trace, add_call)
        assert score["absolute_invalid_no_call_consistency"] is False
        assert score["family_contract_consistency"] is False


def test_direction_name_kinds_and_variant_floor_are_not_diluted() -> None:
    questions, answers = _direction_panel()
    contract = load_contract()
    scores = [
        score_s1_record(
            question,
            answers[question["record_id"]],
            _perfect_prediction(answers[question["record_id"]]),
            contract=contract,
        )
        for question in questions
    ]
    assert Counter(row["direction_name_kind"] for row in scores) == {
        "canonical": 280,
        "metric_variant": 80,
        "shared_alias": 40,
        "ambiguous": 50,
        "unknown": 50,
    }

    variant_index = next(
        index
        for index, row in enumerate(scores)
        if row["direction_name_kind"] == "metric_variant"
        and row["source_trace"] == "S1.8.1"
    )
    question = questions[variant_index]
    answer = answers[question["record_id"]]
    prediction = _perfect_prediction(answer)
    prediction["generated_messages"][-1]["content"]["result"][
        "metric"
    ] = "wrong_metric"
    scores[variant_index] = score_s1_record(
        question, answer, prediction, contract=contract
    )
    assert scores[variant_index]["direction_contract_accuracy"] is True
    assert scores[variant_index]["heldout_metric_mapping_accuracy"] is False
    assert scores[variant_index]["family_contract_consistency"] is True

    metrics = _build_metrics(
        scores,
        source_trace_floor=0.99,
        family_floor=0.99,
        final_floor=0.99,
    )
    mapping = metrics["heldout_metric_variant_mapping"]
    assert mapping["heldout_metric_mapping_accuracy_total"] == 80
    assert mapping["heldout_metric_mapping_accuracy_count"] == 79
    assert metrics["gate"]["checks"]["final_answer_accuracy"] is True
    assert metrics["gate"]["checks"]["all_family_accuracy"] is True
    assert metrics["gate"]["checks"][
        "all_source_trace_envelope_accuracy"
    ] is True
    assert metrics["gate"]["checks"][
        "heldout_metric_variant_mapping"
    ] is False
    assert "<unrecognized>" not in metrics["direction_truth"]["by_metric"]
    assert metrics["shared_alias_mapping"]["row_count"] == 40


def test_shared_alias_tuple_and_confusion_sentinels_are_hard() -> None:
    direction_questions, direction_answers = _direction_panel()
    contract = load_contract()
    alias_question = next(
        question
        for question in direction_questions
        if question["metadata"]["source_trace"] in {
            "S1.8.1",
            "S1.8.2",
            "S1.8.3",
            "S1.8.4",
        }
        and direction_answers[question["record_id"]]["expected_messages"][-1][
            "content"
        ]["result"].get("candidate_metrics")
    )
    alias_answer = direction_answers[alias_question["record_id"]]
    alias_prediction = _perfect_prediction(alias_answer)
    alias_result = alias_prediction["generated_messages"][-1]["content"][
        "result"
    ]
    alias_result["candidate_metrics"] = ["rank", "score"]
    alias_result["lower_is_stronger"] = None
    alias_score = score_s1_record(
        alias_question, alias_answer, alias_prediction, contract=contract
    )
    assert alias_score["shared_alias_mapping_accuracy"] is False
    assert alias_score["direction_contract_accuracy"] is False
    assert alias_score["family_contract_consistency"] is False

    layer_question, layer_answer = _sample(trace="S1.6.4")
    layer_prediction = _perfect_prediction(layer_answer)
    layer_prediction["generated_messages"][-1]["content"]["result"][
        "layers"
    ].append({"layer_family": "<extra>"})
    layer_score = score_s1_record(
        layer_question, layer_answer, layer_prediction, contract=contract
    )
    assert layer_score["generated_layer_families"][-1] == "<invalid>"
    metrics = _build_metrics([layer_score])
    confusion = metrics["layer_family_confusion"]["layer_tag_parsing"]
    assert confusion["count"] == max(
        len(layer_score["expected_layer_families"]),
        len(layer_score["generated_layer_families"]),
    )
    assert confusion["matrix"]["<extra>"]["<invalid>"] == 1
    assert confusion["accuracy_rate"] < 1.0


def _gate_covering_scores() -> list[dict[str, Any]]:
    """Return exact scores that cover every family and gated row class."""

    questions, answers = _panel()
    selected = {}
    for question in questions:
        selected.setdefault(question["metadata"]["question_family"], question)
    selected["failure_fixture"], _ = _sample(trace="S1.1.8")
    direction_questions, direction_answers = _direction_panel()
    contract = load_contract()
    by_kind = {}
    for question in direction_questions:
        answer = direction_answers[question["record_id"]]
        score = score_s1_record(
            question, answer, _perfect_prediction(answer), contract=contract
        )
        by_kind.setdefault(score["direction_name_kind"], question)
    selected["direction_metric_variant"] = by_kind["metric_variant"]
    selected["direction_shared_alias"] = by_kind["shared_alias"]
    selected["direction_canonical"] = by_kind["canonical"]
    all_answers = answers | direction_answers
    return [
        score_s1_record(
            question,
            all_answers[question["record_id"]],
            _perfect_prediction(all_answers[question["record_id"]]),
            contract=contract,
        )
        for question in selected.values()
    ]


def _gate(scores: list[dict[str, Any]], **profile: Any) -> dict[str, Any]:
    _, metrics = build_s1_report(
        scores,
        generation=_generation_identity(),
        evaluator_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="c" * 64,
        minimum_valid_trajectory=1.0,
        minimum_reasoning_present=1.0,
        minimum_exact_tool_selection=1.0,
        minimum_exact_tool_arguments=1.0,
        minimum_tool_payload_consistency=1.0,
        minimum_final_answer_accuracy=1.0,
        minimum_family_accuracy=1.0,
        minimum_source_trace_envelope_accuracy=1.0,
        maximum_unexpected_direct_answer_rate=0.0,
        minimum_tool_followup_thinking_present=1.0,
        **profile,
    )
    return metrics["gate"]


TIERED = {
    "gate_profile": "tiered_v2",
    "minimum_strict_rate": 0.995,
    "minimum_target_rate": 0.9,
}


def test_tiered_gate_does_not_evaluate_the_summary_text() -> None:
    scores = _gate_covering_scores()
    assert _gate(scores, **TIERED)["passed"] is True
    scores[0].update(
        summary_accuracy=False,
        final_answer_accuracy=False,
        exact_behavior=False,
        exact_trace=False,
    )

    promotion = _gate(scores)
    tiered = _gate(scores, **TIERED)

    assert promotion["checks"]["summary_consistency"] is False
    assert promotion["passed"] is False
    assert "summary_consistency" not in tiered["checks"]
    assert tiered["profile"] == "tiered_v2"
    assert tiered["passed"] is True


def test_tiered_gate_allows_a_target_miss_but_not_a_strict_miss() -> None:
    scores = _gate_covering_scores()
    assert len(scores) >= 10
    scores[0].update(continuation_decision_accuracy=False, envelope_accuracy=False)

    assert _gate(scores, **TIERED)["checks"]["continuation_decision_accuracy"]
    assert _gate(scores, **TIERED)["passed"] is True

    scores[1].update(result_accuracy=False)
    tiered = _gate(scores, **TIERED)

    assert tiered["checks"]["result_accuracy"] is False
    assert tiered["passed"] is False


TIERED_V3 = {"gate_profile": "tiered_v3"}


def test_structured_95_gate_uses_overall_point_estimate_and_retains_network_check():
    """Exactly 95% passes; subgroup floors and prose do not change this gate."""
    question, answer = _sample(trace="S1.1.1")
    score = score_s1_record(question, answer, _perfect_prediction(answer), contract=load_contract())
    scores = [{**deepcopy(score), "record_id": f"fixture-{i}"} for i in range(20)]
    for row in scores:
        row.update(summary_accuracy=False, final_answer_accuracy=False)
    scores[0]["structured_answer_accuracy"] = False
    profile = {"gate_profile": "structured_95_v1"}
    gate = _gate(scores, **profile)
    assert gate["passed"] is True
    assert gate["thresholds"] == {"minimum_structured_answer_accuracy": 0.95}
    assert gate["checks"] == {"structured_answer_accuracy": True, "network_used_false": True}
    assert gate["details"]["structured_answer_accuracy"] == {
        "correct": 19, "total": 20, "rate": 0.95, "target": 0.95,
        "comparison": "point_estimate", "scope": "all_panel_rows", "passed": True,
    }
    scores[1]["structured_answer_accuracy"] = False
    assert _gate(scores, **profile)["passed"] is False
    scores[1]["structured_answer_accuracy"] = True
    scores[0]["network_used_false"] = False
    assert _gate(scores, **profile)["passed"] is False
    assert _gate([], **profile)["passed"] is False


def test_structured_answer_accuracy_ignores_summary_wording() -> None:
    question, answer = _sample(trace="S1.1.1")
    prediction = _perfect_prediction(answer)
    final = prediction["generated_messages"][-1]["content"]
    final["summary"] = "A different but correct sentence."

    score = score_s1_record(question, answer, prediction, contract=load_contract())
    assert score["final_answer_accuracy"] is False
    assert score["structured_answer_accuracy"] is True

    final["status"] = "not_found"
    score = score_s1_record(question, answer, prediction, contract=load_contract())
    assert score["structured_answer_accuracy"] is False


def _membership_prediction(multiplex_id: str, echoed_id: str) -> tuple[dict, dict]:
    """Return one membership record whose call names multiplex_id."""

    questions, answers = _panel()
    question = next(
        row
        for row in questions
        if any(
            message.get("tool_call", {}).get("name") == "check_multiplex_membership"
            for message in answers[row["record_id"]]["expected_messages"]
        )
    )
    prediction = _perfect_prediction(answers[question["record_id"]])
    call_index = 0
    for message in prediction["generated_messages"]:
        if message.get("tool_call", {}).get("name") == "check_multiplex_membership":
            message["tool_call"]["arguments"]["multiplex_id"] = multiplex_id
        if message.get("role") != "tool":
            continue
        provenance = message["content"]["provenance"]
        channel = next(key for key, value in provenance.items() if value)
        provenance[channel]["evidence_id"] = deterministic_evidence_id(
            channel=channel, record_index=0, call_index=call_index
        )
        if channel == "graph":
            provenance["graph"]["multiplex_id"] = echoed_id
        call_index += 1
    return {question["record_id"]: question}, {question["record_id"]: prediction}


def test_evidence_policy_scores_an_invented_multiplex_as_a_row_error() -> None:
    questions, predictions = _membership_prediction("MPX:example", "MPX:example")
    validate_generated_evidence_policy(questions, predictions, load_contract())

    # The same bad provenance without a matching model argument is a harness
    # fault and still stops the run.
    questions, predictions = _membership_prediction(
        "full_brain_multiplex_v1", "MPX:example"
    )
    try:
        validate_generated_evidence_policy(questions, predictions, load_contract())
    except S1EvaluationError:
        pass
    else:
        raise AssertionError("a harness provenance fault was accepted")


def test_wilson_interval_matches_reference_values() -> None:
    low, high = _wilson_interval(39, 40)
    assert round(low, 4) == 0.8712 and round(high, 4) == 0.9956
    assert _wilson_interval(0, 0) == (0.0, 1.0)
    assert _wilson_interval(6755, 6755)[1] == 1.0
    assert _wilson_interval(0, 40)[0] == 0.0


def test_tiered_v3_gate_ignores_the_summary_and_uses_intervals() -> None:
    scores = _gate_covering_scores()
    assert len(scores) >= 10
    gate = _gate(scores, **TIERED_V3)
    assert gate["profile"] == "tiered_v3"
    assert gate["passed"] is True
    assert gate["details"]["result_accuracy"]["tier"] == "decision"

    scores[0].update(summary_accuracy=False, final_answer_accuracy=False)
    assert _gate(scores, **TIERED_V3)["passed"] is True

    # One miss in about ten rows cannot rule out 98%; three misses can.
    scores[1].update(result_accuracy=False)
    assert _gate(scores, **TIERED_V3)["checks"]["result_accuracy"] is True
    for score in scores[2:4]:
        score.update(result_accuracy=False)
    gate = _gate(scores, **TIERED_V3)
    assert gate["checks"]["result_accuracy"] is False
    assert gate["passed"] is False


def test_tiered_v3_gate_scores_only_runtime_metric_names() -> None:
    scores = _gate_covering_scores()
    held_out = [
        score
        for score in scores
        if score["direction_name_kind"] in {"metric_variant", "shared_alias"}
    ]
    assert held_out
    for score in held_out:
        score.update(status_accuracy=False, result_accuracy=False)

    gate = _gate(scores, **TIERED_V3)
    assert gate["passed"] is True
    assert gate["details"]["status_accuracy"]["total"] == len(scores) - len(held_out)
    assert "heldout_metric_variant_mapping" not in gate["checks"]


def test_blocked_reason_presence_ignores_the_sentence_wording() -> None:
    question, answer = _sample(trace="S1.1.3")
    prediction = _perfect_prediction(answer)
    final = prediction["generated_messages"][-1]["content"]
    assert final["blocked_reason"]
    final["blocked_reason"] = "A reworded reason for the same block."

    score = score_s1_record(question, answer, prediction, contract=load_contract())
    assert score["blocked_reason_accuracy"] is False
    assert score["blocked_reason_presence"] is True
    assert score["structured_answer_accuracy"] is True

    final["blocked_reason"] = None
    score = score_s1_record(question, answer, prediction, contract=load_contract())
    assert score["blocked_reason_presence"] is False
    assert score["structured_answer_accuracy"] is False


def test_claim_leakage_rejects_invented_supported_claims() -> None:
    """Unknown claims must fail even when absent from the unsupported list."""

    question, answer = _sample(trace="S1.7.8")
    for claim in ("evolutionary_context", "conserved_sequence"):
        prediction = _perfect_prediction(answer)
        prediction["generated_messages"][-1]["content"]["result"][
            "supported_claims"
        ].append(claim)
        score = score_s1_record(
            question, answer, prediction, contract=load_contract()
        )
        assert score["no_unsupported_claim_leakage"] is False
        assert score["structured_answer_accuracy"] is False


def test_module_result_contract_ignores_json_key_order() -> None:
    question, answer = _sample(trace="S1.9.1")
    prediction = _perfect_prediction(answer)
    final = prediction["generated_messages"][-1]["content"]
    final["result"] = dict(reversed(list(final["result"].items())))

    score = score_s1_record(question, answer, prediction, contract=load_contract())
    assert score["module_result_contract"] is True
    assert score["family_contract_consistency"] is True
