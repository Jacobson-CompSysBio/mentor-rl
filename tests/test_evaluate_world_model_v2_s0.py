from __future__ import annotations

import json
from pathlib import Path

from scripts.build_world_model_v2_s0_generation_bundle import (
    build_generation_bundle,
)
from scripts.evaluate_world_model_v2_s0 import (
    PREDICTION_SCHEMA_VERSION,
    evaluate_test,
    score_record,
    sha256_file,
    stable_sha256,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR_MANIFEST = (
    REPO_ROOT / "data/world_model_v2/eval/s0_human_identifiers_v4/manifest.json"
)
EVALUATOR_MANIFEST_SHA256 = (
    "af7e2cfe87dabd1ebc9b7059c892f94946bd73c10521a08b5ff8d6b533889590"
)
PLAIN_TOKENIZER_MANIFEST = (
    REPO_ROOT
    / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4"
    / "plain_base_tokenizer/tokenizer_manifest.json"
)


def read_jsonl(path: Path) -> list[dict]:
    """Read one test JSONL file."""

    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write one stable test JSONL file."""

    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

## ensure that the generation bundle does not include any private answer keys (avoids accidental leakage)
def test_generation_bundle_has_no_private_answers(tmp_path: Path) -> None:
    bundle = build_generation_bundle(
        evaluator_manifest_path=EVALUATOR_MANIFEST,
        evaluator_manifest_sha256=EVALUATOR_MANIFEST_SHA256,
        output_dir=tmp_path,
    )
    assert bundle["record_count"] == 659
    assert bundle["reads_private_answer_keys"] is False
    assert "answer_key_path" not in bundle
    assert "answer_key_sha256" not in bundle
    questions = read_jsonl(tmp_path / "questions.jsonl")
    assert len(questions) == 659
    assert all("answer" not in row for row in questions)

# ensure that mapping accuracy is computed separately from whole record match, so that a model can get mapping accuracy even if it produces extra fields in the record
def test_mapping_accuracy_is_separate_from_whole_record() -> None:
    evaluator = json.loads(EVALUATOR_MANIFEST.read_text(encoding="utf-8"))
    questions = read_jsonl(REPO_ROOT / evaluator["test"]["questions_path"])
    answers = read_jsonl(REPO_ROOT / evaluator["test"]["answer_key_path"])
    question = questions[0]
    answer = answers[0]
    predicted = dict(answer["answer"])
    predicted["unexpected"] = "field"
    scored = score_record(
        question,
        answer,
        {
            "record_id": question["record_id"],
            "encoded_prediction": json.dumps(
                predicted, separators=(",", ":"), sort_keys=True
            ),
        },
        codec=None,
    )
    assert scored["mapping_exact"] is True
    assert scored["exact_record_match"] is False
    assert scored["schema_valid"] is False

# test model predictions across all rows and families
def test_exact_gold_predictions_pass_all_family_gates(tmp_path: Path) -> None:
    evaluator = json.loads(EVALUATOR_MANIFEST.read_text(encoding="utf-8"))
    questions = read_jsonl(REPO_ROOT / evaluator["test"]["questions_path"])
    answers = {
        row["record_id"]: row
        for row in read_jsonl(REPO_ROOT / evaluator["test"]["answer_key_path"])
    }
    predictions = [
        {
            "record_id": row["record_id"],
            "encoded_prediction": json.dumps(
                answers[row["record_id"]]["answer"],
                separators=(",", ":"),
                sort_keys=True,
            ),
            "raw_generation": "",
            "prompt_tokens": 1,
            "generated_tokens": 1,
            "disallowed_special_tokens": [],
        }
        for row in questions
    ]
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    tokenizer = json.loads(PLAIN_TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
    generation = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "test_panel_id": evaluator["test"]["test_panel_id"],
        "record_count": len(predictions),
        "method_id": "oss20b-plain-base-tokenizer-lora-r32",
        "train_run_id": "test-train-run",
        "base_model_identity_sha256": "a" * 64,
        "checkpoint_identity_sha256": "b" * 64,
        "tokenizer_manifest_sha256": tokenizer["manifest_sha256"],
        "generation_bundle_sha256": "c" * 64,
        "generation_config": {"do_sample": False},
        "generation_config_sha256": "d" * 64,
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": 1.0,
        "reads_private_answer_keys": False,
    }
    generation["manifest_sha256"] = stable_sha256(generation)
    generation_path = tmp_path / "generation_manifest.json"
    generation_path.write_text(
        json.dumps(generation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = evaluate_test(
        evaluator_manifest_path=EVALUATOR_MANIFEST,
        evaluator_manifest_sha256=EVALUATOR_MANIFEST_SHA256,
        predictions_path=predictions_path,
        generation_manifest_path=generation_path,
        tokenizer_manifest_path=PLAIN_TOKENIZER_MANIFEST,
        output_dir=tmp_path / "report",
        minimum_family_accuracy=0.9,
        target_family_accuracy=0.95,
        publish=False,
    )
    assert metrics["overall"]["mapping_accuracy"] == 1.0
    assert metrics["macro_family_mapping_accuracy"] == 1.0
    assert metrics["minimum_family_mapping_accuracy"] == 1.0
    assert metrics["gate"]["passed"] is True
    assert metrics["gate"]["target_reached"] is True
    assert set(metrics["families"]) == {
        "human_symbol_to_ensembl",
        "human_ensembl_to_symbol",
        "human_ambiguous_symbol",
    }
