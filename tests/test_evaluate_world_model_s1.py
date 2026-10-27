from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from runtime.world_model.contract import load_contract
from scripts.evaluate_world_model_s1 import (
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


@lru_cache(maxsize=1)
def _direction_panel() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]]
]:
    contract = load_contract()
    raw_questions = [
        row
        for row in read_jsonl(
            REPO_ROOT / "data/world_model_v4/eval/s1_v10/test_questions.jsonl"
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
            REPO_ROOT / "data/world_model_v4/eval/s1_v10/test_answer_key.jsonl"
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
    assert metrics["closed_book_rows"]["closed_book_zero_calls_rate"] == 1.0
    assert metrics["failure_rows"]["failure_harness_accuracy_rate"] == 1.0
    assert metrics["gate"]["checks"]["network_used_false"] is True
    assert metrics["overall"]["exact_behavior_rate"] == 1.0
    assert len(report["records"]) == len(scores)


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
