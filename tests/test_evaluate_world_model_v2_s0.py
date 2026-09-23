"""Test the canonical S0 trajectory evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_world_model_v2_s0_generation_bundle import (
    TOOL_TRAJECTORY_EVALUATION_CONTRACT,
    build_generation_bundle,
)
from scripts.evaluate_world_model_v2_s0 import (
    build_tool_trajectory_report,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_MANIFEST = (
    REPO_ROOT
    / "data/world_model_v2/eval/"
    "s0_human_identifier_trajectories_v6/manifest.json"
)
EVALUATOR_MANIFEST_SHA256 = (
    "9d8e24dbdf6b7bc8564255a6b8c5c52"
    "c15a27464b4ca8110f348c6edbf56ce9d"
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read one JSON Lines file."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_generation_bundle_has_no_private_targets(
    tmp_path: Path,
) -> None:
    """Keep each supervised target out of the public bundle."""

    bundle = build_generation_bundle(
        evaluator_manifest_path=EVALUATOR_MANIFEST,
        evaluator_manifest_sha256=EVALUATOR_MANIFEST_SHA256,
        output_dir=tmp_path,
    )
    questions = read_jsonl(tmp_path / "questions.jsonl")

    assert bundle["record_count"] == 659
    assert bundle["evaluation_contract"] == (
        TOOL_TRAJECTORY_EVALUATION_CONTRACT
    )
    assert bundle["reads_private_answer_keys"] is False
    assert "identifier_registry" in bundle
    for row in questions:
        for field in (
            "assistant_tool_call",
            "assistant_thinking",
            "tool_result",
            "assistant_final",
            "expected_tool_payload",
        ):
            assert field not in row


def _score_row(family: str) -> dict[str, object]:
    """Return one exact trajectory score row."""

    return {
        "record_id": f"record-{family}",
        "fact_id": f"fact-{family}",
        "family": family,
        "encoded_prediction": "encoded",
        "assistant_thinking": "reason",
        "expected_assistant_thinking": "reason",
        "parsed_tool_call": {"name": "tool", "arguments": {}},
        "expected_tool_call": {"name": "tool", "arguments": {}},
        "tool_payload": {"status": "resolved"},
        "generated_tool_payload": {"status": "resolved"},
        "expected_tool_payload": {"status": "resolved"},
        "encoded_final_answer": "answer",
        "assistant_final": "answer",
        "expected_final_answer": "answer",
        "parse_error": None,
        "final_parse_error": None,
        "tool_error": None,
        "generated_tool_error": None,
        "reasoning_present": True,
        "reasoning_exact": True,
        "valid_single_tool_call": True,
        "exact_tool_argument": True,
        "payload_exact": True,
        "valid_final_answer": True,
        "final_answer_exact": True,
        "valid_tool_trajectory": True,
        "ambiguous_defer_correct": True,
        "direct_answer": False,
        "network_used_false": True,
        "disallowed_special_tokens": [],
        "final_disallowed_special_tokens": [],
    }


def test_exact_trajectory_scores_pass_all_gates() -> None:
    """Pass each gate for exact rows from all families."""

    rows = [
        _score_row("human_symbol_to_ensembl"),
        _score_row("human_ensembl_to_symbol"),
        _score_row("human_ambiguous_symbol"),
    ]
    generation = {
        "test_panel_id": "a" * 64,
        "train_run_id": "train-run",
        "method_id": "test-method",
        "checkpoint_identity_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    report, metrics = build_tool_trajectory_report(
        rows,
        evaluation_contract=TOOL_TRAJECTORY_EVALUATION_CONTRACT,
        generation=generation,
        evaluator_manifest_sha256="d" * 64,
        tokenizer_manifest_sha256="e" * 64,
        registry_sha256="f" * 64,
        minimum_valid_tool_trajectory=0.99,
        minimum_reasoning_present=0.99,
        minimum_exact_reasoning=0.99,
        minimum_valid_single_tool_call=0.99,
        minimum_exact_tool_argument=0.99,
        minimum_family_payload_accuracy=0.99,
        minimum_family_final_answer_accuracy=0.99,
        minimum_ambiguous_defer_accuracy=1.0,
        maximum_direct_answer_rate=0.01,
    )

    assert metrics["gate"]["passed"] is True
    assert metrics["overall"]["valid_tool_trajectory_rate"] == 1.0
    assert metrics["overall"]["final_answer_accuracy"] == 1.0
    assert set(metrics["families"]) == {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
    }
    assert report["metrics"]["metrics_sha256"] == (
        metrics["metrics_sha256"]
    )


def test_wrong_final_answer_fails_the_family_gate() -> None:
    """Fail the final-answer gate for one incorrect family row."""

    rows = [
        _score_row("human_symbol_to_ensembl"),
        _score_row("human_ensembl_to_symbol"),
        _score_row("human_ambiguous_symbol"),
    ]
    rows[0]["final_answer_exact"] = False
    generation = {
        "test_panel_id": "a" * 64,
        "train_run_id": "train-run",
        "method_id": "test-method",
        "checkpoint_identity_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    _, metrics = build_tool_trajectory_report(
        rows,
        evaluation_contract=TOOL_TRAJECTORY_EVALUATION_CONTRACT,
        generation=generation,
        evaluator_manifest_sha256="d" * 64,
        tokenizer_manifest_sha256="e" * 64,
        registry_sha256="f" * 64,
        minimum_valid_tool_trajectory=0.0,
        minimum_reasoning_present=0.0,
        minimum_exact_reasoning=0.0,
        minimum_valid_single_tool_call=0.0,
        minimum_exact_tool_argument=0.0,
        minimum_family_payload_accuracy=0.0,
        minimum_family_final_answer_accuracy=1.0,
        minimum_ambiguous_defer_accuracy=0.0,
        maximum_direct_answer_rate=1.0,
    )

    assert metrics["gate"]["passed"] is False
    assert (
        metrics["gate"]["checks"]["family_final_answer_accuracy"]
        is False
    )
