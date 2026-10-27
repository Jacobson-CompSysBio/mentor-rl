"""Test the S0 full-registry generation bundle and scorer report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_world_model_v2_s0_generation_bundle import (
    build_generation_bundle,
)
from scripts.evaluate_world_model_v2_s0 import (
    FULL_REGISTRY_V3_EVALUATION_CONTRACT,
    build_tool_trajectory_report,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_MANIFEST = (
    REPO_ROOT / "data/world_model_v3/eval/s0_full_registry/manifest.json"
)
EVALUATOR_MANIFEST_SHA256 = (
    "dbca23382f0881aba83a6100ea54004031337bef53415236e8f1042970c53beb"
)
FAMILIES = (
    "human_symbol_to_ensembl",
    "human_ensembl_to_symbol",
    "human_ambiguous_symbol",
)
LENIENT_FLOORS = {
    "minimum_valid_tool_trajectory": 0.0,
    "minimum_reasoning_present": 0.0,
    "minimum_exact_reasoning": 0.0,
    "minimum_valid_single_tool_call": 0.0,
    "minimum_exact_tool_argument": 0.0,
    "minimum_family_payload_accuracy": 0.0,
    "minimum_family_final_answer_accuracy": 0.0,
    "minimum_ambiguous_defer_accuracy": 0.0,
    "maximum_direct_answer_rate": 1.0,
    "minimum_structured_json_valid": 0.0,
    "minimum_status_exact": 0.0,
    "minimum_result_exact": 0.0,
    "minimum_continuation_decision_exact": 0.0,
    "minimum_not_found_defer_accuracy": 0.0,
    "minimum_unnamed_id_resolved_accuracy": 0.0,
}
# The tiers of the v4 test config; see the S0 curriculum document.
V4_FLOORS = {
    **LENIENT_FLOORS,
    "minimum_valid_tool_trajectory": 0.995,
    "minimum_reasoning_present": 0.99,
    "minimum_valid_single_tool_call": 0.995,
    "minimum_exact_tool_argument": 0.995,
    "minimum_family_payload_accuracy": 0.995,
    "minimum_ambiguous_defer_accuracy": 0.995,
    "maximum_direct_answer_rate": 0.005,
    "minimum_structured_json_valid": 0.995,
    "minimum_status_exact": 0.995,
    "minimum_result_exact": 0.995,
    "minimum_continuation_decision_exact": 0.9,
    "minimum_not_found_defer_accuracy": 0.995,
}


def test_generation_bundle_has_no_private_targets(
    tmp_path: Path,
) -> None:
    """Keep each supervised target out of the public bundle."""

    if not EVALUATOR_MANIFEST.is_file():
        pytest.skip(f"The frozen panel is not present: {EVALUATOR_MANIFEST}")
    bundle = build_generation_bundle(
        evaluator_manifest_path=EVALUATOR_MANIFEST,
        evaluator_manifest_sha256=EVALUATOR_MANIFEST_SHA256,
        output_dir=tmp_path,
    )

    assert bundle["record_count"] == 120766
    assert bundle["evaluation_contract"] == FULL_REGISTRY_V3_EVALUATION_CONTRACT
    assert bundle["reads_private_answer_keys"] is False
    assert "identifier_registry" in bundle
    with (tmp_path / "questions.jsonl").open(encoding="utf-8") as questions:
        for line in questions:
            row = json.loads(line)
            for field in (
                "answer",
                "expected_final_answer",
                "expected_tool_call",
                "expected_tool_payload",
                "expected_status",
            ):
                assert field not in row


def _score_row(index: int) -> dict[str, object]:
    """Return one exact score row for one of the three families."""

    family = FAMILIES[index % len(FAMILIES)]
    return {
        "record_id": f"record-{index}",
        "fact_id": f"fact-{index}",
        "family": family,
        "parse_error": None,
        "final_parse_error": None,
        "structured_answer_error": None,
        "tool_error": None,
        "reasoning_present": True,
        "reasoning_exact": True,
        "valid_single_tool_call": True,
        "exact_tool_argument": True,
        "payload_exact": True,
        "valid_final_answer": True,
        "final_explanation_present": True,
        "final_explanation_exact": True,
        "structured_json_valid": True,
        "status_exact": True,
        "result_exact": True,
        "continuation_decision_exact": True,
        "final_answer_exact": True,
        "valid_tool_trajectory": True,
        "ambiguous_defer_correct": True,
        "not_found_defer_correct": True,
        "unnamed_id_resolved_correct": True,
        "direct_answer": False,
        "network_used_false": True,
    }


def _report(rows: list[dict[str, object]], **floors: float):
    """Build the report with the given gate floors."""

    return build_tool_trajectory_report(
        rows,
        evaluation_contract=FULL_REGISTRY_V3_EVALUATION_CONTRACT,
        generation={
            "test_panel_id": "a" * 64,
            "train_run_id": "train-run",
            "method_id": "test-method",
            "checkpoint_identity_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        evaluator_manifest_sha256="d" * 64,
        tokenizer_manifest_sha256="e" * 64,
        registry_sha256="f" * 64,
        **floors,
    )


def test_exact_scores_pass_the_v4_gate() -> None:
    """Exact rows from all three families pass every check."""

    report, metrics = _report([_score_row(index) for index in range(3)], **V4_FLOORS)

    assert metrics["gate"]["passed"] is True
    assert all(metrics["gate"]["checks"].values())
    assert metrics["overall"]["valid_tool_trajectory_rate"] == 1.0
    assert set(metrics["families"]) == set(FAMILIES)
    assert report["metrics"]["metrics_sha256"] == metrics["metrics_sha256"]


def test_wrong_final_answer_fails_the_family_gate() -> None:
    """Fail the final-answer gate for one incorrect family row."""

    rows = [_score_row(index) for index in range(3)]
    rows[0]["final_answer_exact"] = False

    _, metrics = _report(
        rows,
        **{**LENIENT_FLOORS, "minimum_family_final_answer_accuracy": 1.0},
    )

    assert metrics["gate"]["passed"] is False
    assert metrics["gate"]["checks"]["family_final_answer_accuracy"] is False


def test_v4_gate_tiers() -> None:
    """Allow a target-tier miss but fail a strict-tier miss."""

    rows = [_score_row(index) for index in range(20)]
    rows[0]["continuation_decision_exact"] = False

    _, metrics = _report(rows, **V4_FLOORS)
    assert metrics["gate"]["passed"] is True

    rows[1]["result_exact"] = False
    _, metrics = _report(rows, **V4_FLOORS)

    assert metrics["gate"]["checks"]["result_exact"] is False
    assert metrics["gate"]["checks"]["continuation_decision_exact"] is True
    assert metrics["gate"]["passed"] is False


def test_report_only_metrics_do_not_gate() -> None:
    """Exact reasoning and the whole-answer match have a floor of zero."""

    rows = [_score_row(index) for index in range(3)]
    for row in rows:
        row["reasoning_exact"] = False
        row["final_answer_exact"] = False
        row["unnamed_id_resolved_correct"] = False

    _, metrics = _report(rows, **V4_FLOORS)

    assert metrics["gate"]["passed"] is True
