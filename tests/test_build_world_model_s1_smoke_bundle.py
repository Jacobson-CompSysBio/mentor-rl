"""Tests for the behavior-stratified S1 smoke bundle."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.world_model_training import S1_V3_TRACE_FAMILIES
from scripts.build_world_model_s1_smoke_bundle import select_split


REPO_ROOT = Path(__file__).resolve().parent.parent


def _row(family: str, family_index: int, row_index: int) -> dict[str, object]:
    padding = "x" * (row_index * 20)
    return {
        "row_id": f"row_{family_index}_{row_index}",
        "messages": [
            {
                "role": "assistant",
                "content": {
                    "status": "resolved",
                    "result": {"value": padding},
                    "continuation_decision": "stop",
                },
            }
        ],
        "tools": [],
        "metadata": {
            "book_mode": "closed_book",
            "step": "S1",
            "question_family": family,
            "source_trace": f"S1.{family_index}.1",
            "blocked_reason_code": None,
        },
    }


def test_selection_covers_every_family_and_keeps_the_longest(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "train.jsonl"
    rows = [
        _row(family, family_index, row_index)
        for family_index, family in enumerate(
            sorted(S1_V3_TRACE_FAMILIES), start=1
        )
        for row_index in range(3)
    ]
    source_path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )

    selection = select_split(
        source_path,
        split="train",
        rows_per_signature=2,
        seed=17,
    )

    assert selection.source_signature_count == 8
    assert len(selection.rows) == 16
    assert set(selection.family_counts) == set(S1_V3_TRACE_FAMILIES)
    selected_ids = {row.row_id for row in selection.rows}
    assert {
        f"row_{family_index}_2"
        for family_index in range(1, 9)
    } <= selected_ids


def test_published_validation_questions_and_answers_match_corpus() -> None:
    corpus_path = (
        REPO_ROOT / "data/world_model_v3/corpora/s1_smoke/validation.jsonl"
    )
    questions_path = (
        REPO_ROOT / "data/world_model_v3/eval/s1_smoke/validation_questions.jsonl"
    )
    answers_path = (
        REPO_ROOT / "data/world_model_v3/eval/s1_smoke/validation_answer_key.jsonl"
    )

    corpus_ids = [
        json.loads(line)["row_id"]
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
    ]
    question_ids = [
        json.loads(line)["record_id"]
        for line in questions_path.read_text(encoding="utf-8").splitlines()
    ]
    answer_ids = [
        json.loads(line)["record_id"]
        for line in answers_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(corpus_ids) == 111
    assert corpus_ids == question_ids == answer_ids
