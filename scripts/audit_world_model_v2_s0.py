#!/usr/bin/env python3
"""Audit the S0 human gene identifier corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_world_model_v2_s0 import (
    FAMILIES,
    load_config,
    read_json,
    resolve_repo_path,
    sha256_file,
)


ANSWER_FIELDS = {
    FAMILIES[0]: {"gene_id", "gene_symbol", "status"},
    FAMILIES[1]: {"gene_id", "gene_symbols", "status"},
    FAMILIES[2]: {
        "action",
        "candidate_gene_ids",
        "gene_symbol",
        "status",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON Lines file."""
    with path.open(encoding="utf-8") as source_file:
        return [json.loads(line) for line in source_file if line.strip()]


def require(condition: bool, message: str) -> None:
    """Fail the audit when one condition is false."""
    if not condition:
        raise ValueError(message)


def check_record(record: dict[str, Any], prompt_form_id: str) -> None:
    """Check the common S0 record fields."""
    metadata = record["metadata"]
    provenance = record["provenance"]

    require(record["context"] is None, "context must be null")
    require(metadata["book_mode"] == "closed_book", "book_mode changed")
    require(metadata["question_family"] in FAMILIES, "family changed")
    require(provenance["fact_role"] == "seen", "fact_role changed")
    require(
        provenance["prompt_form_id"] == prompt_form_id,
        "prompt form changed",
    )


def check_answer(family: str, answer: dict[str, Any]) -> None:
    """Check the answer fields for one family."""
    require(set(answer) == ANSWER_FIELDS[family], "answer fields changed")


def check_panel(
    name: str,
    questions: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    train_by_fact: dict[str, dict[str, Any]],
) -> set[str]:
    """Check one public question panel and its private answer key."""
    question_by_id = {row["record_id"]: row for row in questions}
    answer_by_id = {row["record_id"]: row for row in answers}

    require(
        len(question_by_id) == len(questions),
        f"duplicate {name} record ID",
    )
    require(
        len(answer_by_id) == len(answers),
        f"duplicate {name} answer ID",
    )
    require(
        set(question_by_id) == set(answer_by_id),
        f"{name} answer join failed",
    )

    group_ids = set()
    for record_id, question in question_by_id.items():
        require("answer" not in question, f"public {name} answer found")
        check_record(question, name)

        provenance = question["provenance"]
        fact_id = provenance["fact_id"]
        require(fact_id in train_by_fact, f"{name} fact has no train row")

        train = train_by_fact[fact_id]
        require(question["input"] == train["input"], f"{name} input changed")
        require(
            question["metadata"]["question_family"]
            == train["metadata"]["question_family"],
            f"{name} family changed",
        )
        require(
            question["question"] != train["question"],
            f"{name} prompt reused",
        )

        answer = answer_by_id[record_id]
        family = question["metadata"]["question_family"]
        require(answer["fact_id"] == fact_id, f"{name} fact join failed")
        require(answer["family"] == family, f"{name} family join failed")
        require(
            answer["fact_group_id"] == provenance["fact_group_id"],
            f"{name} group join failed",
        )
        check_answer(family, answer["answer"])
        group_ids.add(provenance["fact_group_id"])

    return group_ids


def audit(config_path: Path) -> dict[str, int]:
    """Audit the generated corpus and evaluator files."""
    config = load_config(config_path)
    corpus_dir = resolve_repo_path(config["outputs"]["corpus_dir"])
    evaluator_dir = resolve_repo_path(config["outputs"]["evaluator_dir"])

    manifest = read_json(corpus_dir / "manifest.json")
    split_manifest = read_json(corpus_dir / "split_manifest.json")
    evaluator_manifest = read_json(evaluator_dir / "manifest.json")

    train_path = corpus_dir / "train.jsonl"
    validation_path = corpus_dir / "val.jsonl"
    test_path = evaluator_dir / "test_questions.jsonl"
    validation_answer_path = evaluator_dir / "validation_answer_key.jsonl"
    test_answer_path = evaluator_dir / "test_answer_key.jsonl"

    expected_hashes = {
        train_path: manifest["file_hashes"]["train.jsonl"],
        validation_path: manifest["file_hashes"]["val.jsonl"],
        corpus_dir / "split_manifest.json": manifest["file_hashes"][
            "split_manifest.json"
        ],
        evaluator_dir / "manifest.json": manifest["file_hashes"][
            "evaluator_manifest.json"
        ],
        validation_answer_path: evaluator_manifest["validation"][
            "answer_key_sha256"
        ],
        test_path: evaluator_manifest["test"]["questions_sha256"],
        test_answer_path: evaluator_manifest["test"]["answer_key_sha256"],
    }
    for path, expected_hash in expected_hashes.items():
        require(sha256_file(path) == expected_hash, f"hash changed: {path}")

    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    test = read_jsonl(test_path)
    validation_answers = read_jsonl(validation_answer_path)
    test_answers = read_jsonl(test_answer_path)

    train_by_fact = {row["provenance"]["fact_id"]: row for row in train}
    require(len(train_by_fact) == len(train), "duplicate train fact ID")

    for row in train:
        check_record(row, "train")
        check_answer(row["metadata"]["question_family"], row["answer"])

    validation_groups = check_panel(
        "validation", validation, validation_answers, train_by_fact
    )
    test_groups = check_panel("test", test, test_answers, train_by_fact)
    require(
        validation_groups.isdisjoint(test_groups),
        "evaluation groups overlap",
    )

    counts = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    require(counts == manifest["row_counts"], "manifest row counts changed")
    require(counts == split_manifest["row_counts"], "split row counts changed")
    return counts


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "config"
            / "world_model_v2_s0_closed_book_recall_v4.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the audit."""
    counts = audit(parse_args().config.expanduser().resolve())
    print(json.dumps({"audit": "passed", "row_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
