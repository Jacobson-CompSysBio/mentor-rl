#!/usr/bin/env python3
"""Score exact S0 test mappings and publish summary metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_s0 import (  # noqa: E402
    AMBIGUOUS_SYMBOL_FAMILY,
    ENSEMBL_TO_SYMBOL_FAMILY,
    S0_FAMILIES,
    SYMBOL_TO_ENSEMBL_FAMILY,
    validate_s0_contract,
)
from runtime.world_model_schemas import WorldModelRecord  # noqa: E402
from runtime.world_model_training import (  # noqa: E402
    load_model_text_codec_for_token_manifest,
    validated_tokenizer_manifest,
)


REPORT_SCHEMA_VERSION = "mentor-rl-world-model-s0-test-report-v4"
METRICS_SCHEMA_VERSION = "mentor-rl-world-model-s0-test-metrics-v4"
PREDICTION_SCHEMA_VERSION = "mentor-rl-world-model-s0-predictions-v4"
EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-manifest-v4"
EVALUATION_CONTRACT = "seen_fact_closed_book_recall_v1"
DATASET_ID = "world_model_v2_s0_human_identifiers_v4"
WILSON_Z_95 = 1.959963984540054
MAPPING_FIELDS = {
    SYMBOL_TO_ENSEMBL_FAMILY: "gene_id",
    ENSEMBL_TO_SYMBOL_FAMILY: "gene_symbols",
    AMBIGUOUS_SYMBOL_FAMILY: "candidate_gene_ids",
}


class S0EvaluationError(RuntimeError):
    """Report one invalid S0 test input."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S0EvaluationError(
            f"Could not read one JSON object from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise S0EvaluationError(f"Expected one JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise S0EvaluationError(
                        f"Expected one JSON object at {path}:{line_number}"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise S0EvaluationError(
            f"Could not read JSONL from {path}: {error}"
        ) from error
    return rows


def write_json(path: Path, payload: Mapping[str, Any], *, private: bool) -> None:
    """Write one stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600 if private else 0o640)


def _unique_rows(
    rows: list[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    """Index rows by their unique record IDs."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise S0EvaluationError(f"{label} has an invalid record ID")
        if record_id in result:
            raise S0EvaluationError(f"{label} has a duplicate record ID")
        result[record_id] = row
    return result


def strict_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one complete JSON object without extra text."""

    if not isinstance(text, str):
        return None, "prediction_is_not_text"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return None, f"json_decode:{error.msg}"
    if not isinstance(value, dict):
        return None, "prediction_is_not_object"
    return value, None


def wilson_interval(successes: int, total: int) -> list[float]:
    """Return one 95-percent Wilson interval."""

    if total < 1:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + WILSON_Z_95**2 / total
    center = (rate + WILSON_Z_95**2 / (2.0 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(
            rate * (1.0 - rate) / total
            + WILSON_Z_95**2 / (4.0 * total**2)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _load_panel(
    evaluator_manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """Load and verify the complete private test panel."""

    if sha256_file(evaluator_manifest_path) != expected_manifest_sha256:
        raise S0EvaluationError("The evaluator manifest identity changed")
    manifest = read_json(evaluator_manifest_path)
    if (
        manifest.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
    ):
        raise S0EvaluationError("The evaluator contract changed")
    test = manifest.get("test")
    if not isinstance(test, Mapping):
        raise S0EvaluationError("The evaluator has no test panel")
    questions_path = (REPO_ROOT / str(test.get("questions_path", ""))).resolve()
    answer_key_path = (REPO_ROOT / str(test.get("answer_key_path", ""))).resolve()
    if sha256_file(questions_path) != test.get("questions_sha256"):
        raise S0EvaluationError("The test question identity changed")
    if sha256_file(answer_key_path) != test.get("answer_key_sha256"):
        raise S0EvaluationError("The test answer-key identity changed")
    questions = _unique_rows(read_jsonl(questions_path), "The test panel")
    answers = _unique_rows(read_jsonl(answer_key_path), "The answer key")
    if (
        set(questions) != set(answers)
        or len(questions) != test.get("row_count")
    ):
        raise S0EvaluationError("The test panel record set changed")
    return manifest, questions, answers


def _load_predictions(
    predictions_path: Path,
    generation_manifest_path: Path,
    expected_record_ids: set[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and verify one complete generation result."""

    generation = read_json(generation_manifest_path)
    claimed = generation.get("manifest_sha256")
    identity = {
        str(key): value
        for key, value in generation.items()
        if key != "manifest_sha256"
    }
    if (
        generation.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or generation.get("predictions_sha256")
        != sha256_file(predictions_path)
    ):
        raise S0EvaluationError("The generation manifest identity changed")
    predictions = _unique_rows(
        read_jsonl(predictions_path), "The predictions"
    )
    if set(predictions) != expected_record_ids:
        raise S0EvaluationError("The prediction record set changed")
    return generation, predictions


def score_record(
    question: Mapping[str, Any],
    answer_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
    *,
    codec: Any | None,
) -> dict[str, Any]:
    """Score one generated S0 answer."""

    encoded = prediction_row.get("encoded_prediction")
    if not isinstance(encoded, str):
        raise S0EvaluationError("A prediction has no generated text")
    decoded = encoded
    codec_report = None
    if codec is not None:
        decoded, codec_report = codec.decode_generated_answer(encoded)
    parsed, parse_error = strict_json_object(decoded)
    schema_errors: list[str] = []
    if parsed is not None:
        predicted_payload = dict(question)
        predicted_payload["answer"] = parsed
        try:
            record = WorldModelRecord.from_dict(predicted_payload)
        except (KeyError, TypeError, ValueError) as error:
            schema_errors.append(
                f"record_construction:{type(error).__name__}:{error}"
            )
        else:
            schema_errors.extend(validate_s0_contract(record))
    codec_valid = codec_report is None or codec_report.get("valid") is True
    schema_valid = parsed is not None and not schema_errors and codec_valid
    family = answer_row.get("family")
    if family not in MAPPING_FIELDS:
        raise S0EvaluationError("An answer row has an invalid family")
    expected = answer_row.get("answer")
    if not isinstance(expected, Mapping):
        raise S0EvaluationError("An answer row has no answer object")
    mapping_field = MAPPING_FIELDS[str(family)]
    mapping_exact = (
        parsed is not None
        and codec_valid
        and type(parsed.get(mapping_field)) is type(expected.get(mapping_field))
        and parsed.get(mapping_field) == expected.get(mapping_field)
    )
    exact_record = schema_valid and parsed == expected
    defer_correct = (
        family != AMBIGUOUS_SYMBOL_FAMILY
        or (
            parsed is not None
            and parsed.get("action") == "defer"
            and parsed.get("status") == "ambiguous"
        )
    )
    return {
        "record_id": question["record_id"],
        "fact_id": answer_row.get("fact_id"),
        "family": family,
        "mapping_field": mapping_field,
        "encoded_prediction": encoded,
        "decoded_prediction": decoded,
        "predicted_answer": parsed,
        "expected_answer": dict(expected),
        "parse_error": parse_error,
        "schema_errors": sorted(set(schema_errors)),
        "codec_report": codec_report,
        "valid_json": parsed is not None,
        "schema_valid": schema_valid,
        "mapping_exact": mapping_exact,
        "exact_record_match": exact_record,
        "defer_correct": defer_correct,
    }


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one nonempty set of scored rows."""

    if not rows:
        raise S0EvaluationError("Cannot summarize an empty S0 family")
    count = len(rows)
    mapping_count = sum(row.get("mapping_exact") is True for row in rows)
    record_count = sum(
        row.get("exact_record_match") is True for row in rows
    )
    valid_json_count = sum(row.get("valid_json") is True for row in rows)
    schema_count = sum(row.get("schema_valid") is True for row in rows)
    defer_count = sum(row.get("defer_correct") is True for row in rows)
    return {
        "record_count": count,
        "mapping_exact_count": mapping_count,
        "mapping_accuracy": mapping_count / count,
        "mapping_wilson_95": wilson_interval(mapping_count, count),
        "exact_record_count": record_count,
        "exact_record_accuracy": record_count / count,
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_count / count,
        "schema_valid_count": schema_count,
        "schema_valid_rate": schema_count / count,
        "defer_correct_count": defer_count,
        "defer_correct_rate": defer_count / count,
    }


def build_report(
    rows: list[dict[str, Any]],
    *,
    generation: Mapping[str, Any],
    evaluator_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    minimum_family_accuracy: float,
    target_family_accuracy: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build private and public S0 test reports."""

    by_family = {
        family: _summary([row for row in rows if row["family"] == family])
        for family in S0_FAMILIES
    }
    overall = _summary(rows)
    family_accuracies = [
        float(by_family[family]["mapping_accuracy"])
        for family in S0_FAMILIES
    ]
    gate_passed = all(
        value >= minimum_family_accuracy for value in family_accuracies
    )
    target_passed = all(
        value >= target_family_accuracy for value in family_accuracies
    )
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": generation["test_panel_id"],
        "train_run_id": generation["train_run_id"],
        "method_id": generation["method_id"],
        "checkpoint_identity_sha256": generation[
            "checkpoint_identity_sha256"
        ],
        "generation_manifest_sha256": generation["manifest_sha256"],
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "overall": overall,
        "families": by_family,
        "macro_family_mapping_accuracy": sum(family_accuracies)
        / len(family_accuracies),
        "minimum_family_mapping_accuracy": min(family_accuracies),
        "gate": {
            "minimum_family_mapping_accuracy": minimum_family_accuracy,
            "target_family_mapping_accuracy": target_family_accuracy,
            "passed": gate_passed,
            "target_reached": target_passed,
        },
    }
    metrics["metrics_sha256"] = stable_sha256(metrics)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "private_report": True,
        "metrics": metrics,
        "error_counts": dict(
            sorted(
                Counter(
                    error
                    for row in rows
                    for error in row["schema_errors"]
                ).items()
            )
        ),
        "records": rows,
    }
    report["report_sha256"] = stable_sha256(report)
    return report, metrics


def publish_wandb(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Append exact test metrics to the train W&B run."""

    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if mode != "online" or entity != "jail-ai" or not project:
        raise S0EvaluationError("The W&B test contract is invalid")
    try:
        import wandb
    except ImportError as error:
        raise S0EvaluationError("The wandb package is absent") from error
    run_id = str(metrics["train_run_id"])
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        mode=mode,
        dir=os.environ.get("WANDB_DIR"),
    )
    if run is None or getattr(run, "disabled", False):
        raise S0EvaluationError("W&B did not resume the train run")
    payload = {
        "test/overall_mapping_accuracy": metrics["overall"][
            "mapping_accuracy"
        ],
        "test/overall_exact_record_accuracy": metrics["overall"][
            "exact_record_accuracy"
        ],
        "test/valid_json_rate": metrics["overall"]["valid_json_rate"],
        "test/schema_valid_rate": metrics["overall"]["schema_valid_rate"],
        "test/macro_family_mapping_accuracy": metrics[
            "macro_family_mapping_accuracy"
        ],
        "test/minimum_family_mapping_accuracy": metrics[
            "minimum_family_mapping_accuracy"
        ],
    }
    for family, family_metrics in metrics["families"].items():
        payload[f"test/{family}/mapping_accuracy"] = family_metrics[
            "mapping_accuracy"
        ]
        payload[
            f"test/{family}/exact_record_accuracy"
        ] = family_metrics["exact_record_accuracy"]
    run.log(payload)
    receipt = {
        "id": run.id,
        "name": run.name,
        "entity": run.entity,
        "project": run.project,
        "url": run.url,
    }
    run.finish()
    return receipt


def evaluate_test(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    predictions_path: Path,
    generation_manifest_path: Path,
    tokenizer_manifest_path: Path,
    output_dir: Path,
    minimum_family_accuracy: float,
    target_family_accuracy: float,
    publish: bool,
) -> dict[str, Any]:
    """Score one complete S0 test result."""

    evaluator, questions, answers = _load_panel(
        evaluator_manifest_path.resolve(), evaluator_manifest_sha256
    )
    generation, predictions = _load_predictions(
        predictions_path.resolve(),
        generation_manifest_path.resolve(),
        set(questions),
    )
    tokenizer_manifest = validated_tokenizer_manifest(
        tokenizer_manifest_path.resolve()
    )
    if (
        tokenizer_manifest.get("manifest_sha256")
        != generation.get("tokenizer_manifest_sha256")
        or generation.get("test_panel_id")
        != evaluator["test"]["test_panel_id"]
    ):
        raise S0EvaluationError("The generation identity changed")
    codec = load_model_text_codec_for_token_manifest(
        tokenizer_manifest_path.resolve()
    )
    scored = [
        score_record(
            questions[record_id],
            answers[record_id],
            predictions[record_id],
            codec=codec,
        )
        for record_id in sorted(questions)
    ]
    report, metrics = build_report(
        scored,
        generation=generation,
        evaluator_manifest_sha256=evaluator_manifest_sha256,
        tokenizer_manifest_sha256=str(
            tokenizer_manifest["manifest_sha256"]
        ),
        minimum_family_accuracy=minimum_family_accuracy,
        target_family_accuracy=target_family_accuracy,
    )
    output_dir = output_dir.resolve()
    write_json(output_dir / "exact_mapping_report.json", report, private=True)
    write_json(output_dir / "test_metrics.json", metrics, private=False)
    if publish:
        receipt = publish_wandb(metrics)
        write_json(output_dir / "wandb_test_run.json", receipt, private=False)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-family-accuracy", type=float, required=True)
    parser.add_argument("--target-family-accuracy", type=float, required=True)
    parser.add_argument("--publish-wandb", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Score one S0 test panel."""

    args = parse_args()
    metrics = evaluate_test(
        evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256,
        predictions_path=args.predictions,
        generation_manifest_path=args.generation_manifest,
        tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir,
        minimum_family_accuracy=args.minimum_family_accuracy,
        target_family_accuracy=args.target_family_accuracy,
        publish=args.publish_wandb,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "passed": metrics["gate"]["passed"],
                "metrics_sha256": metrics["metrics_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
