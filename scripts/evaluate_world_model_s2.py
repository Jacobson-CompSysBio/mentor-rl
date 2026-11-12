#!/usr/bin/env python3
"""Score frozen S2 structured rollouts without comparing free prose."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s2-evaluator-v1"
EVALUATION_CONTRACT = "held_out_s2_structured_rollout_v1"
PREDICTION_SCHEMA_VERSION = "mentor-rl-world-model-s2-predictions-v1"
REPORT_SCHEMA_VERSION = "mentor-rl-world-model-s2-exact-report-v1"
METRICS_SCHEMA_VERSION = "mentor-rl-world-model-s2-exact-metrics-v1"
PUBLIC_METADATA_KEYS = {
    "schema_version",
    "tool_contract_version",
    "question_family",
    "split",
}
TARGET_KEYS = {
    "status",
    "result",
    "continuation_decision",
    "provenance",
    "blocked_reason_required",
    "expected_tool_calls",
    "expected_tool_responses",
}
SCORED_TARGET_KEYS = (
    "status",
    "result",
    "continuation_decision",
    "provenance",
    "blocked_reason_required",
    "expected_tool_calls",
)
PREDICTION_KEYS = {
    "row_id",
    "generated_messages",
    "generations",
    "stop_reason",
    "trajectory_error",
    "tool_call_count",
    "executed_tool_call_count",
    "network_used_false",
}
PUBLICATION_IDENTITY_KEYS = (
    "train_run_id",
    "method_id",
    "checkpoint_identity_sha256",
    "base_model_identity_sha256",
    "base_model_artifact_sha256",
    "tokenizer_manifest_sha256",
    "tokenizer_artifacts_sha256",
    "generation_bundle_sha256",
)
SCORE_FIELDS = (
    "valid_trajectory",
    "status_accuracy",
    "result_accuracy",
    "continuation_decision_accuracy",
    "provenance_accuracy",
    "blocked_reason_presence_accuracy",
    "exact_tool_calls",
    "structured_accuracy",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FAMILY = re.compile(r"S2\.(?:[1-9]|1[0-3])\Z")
REPORT_ONLY_GATE_PROFILE = "report_only_v1"


class S2EvaluationError(RuntimeError):
    """Report one harness, identity, or publication fault."""


def canonical_json(value: Any) -> str:
    """Serialize one identity deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without materializing it."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    """Return one required lowercase SHA-256 value."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S2EvaluationError(f"{label} must be a lowercase SHA-256 value")
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one required JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2EvaluationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise S2EvaluationError(f"{label} must be one JSON object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    """Read one JSON object per nonempty line."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise S2EvaluationError(
                        f"{label} has a blank record at line {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise S2EvaluationError(
                        f"{label} record {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2EvaluationError(f"cannot read {label} {path}: {error}") from error
    return rows


def write_json(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    """Atomically write one deterministic report object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o600 if private else 0o640)


def _child_path(root: Path, value: Any, label: str) -> Path:
    """Resolve one evaluator-owned file without allowing directory escape."""

    if not isinstance(value, str) or not value:
        raise S2EvaluationError(f"{label} path is absent")
    declared = Path(value)
    path = (declared if declared.is_absolute() else root / declared).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise S2EvaluationError(f"{label} escapes the evaluator root") from error
    if not path.is_file():
        raise S2EvaluationError(f"{label} is absent: {path}")
    return path


def _unique(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    """Index rows by their nonempty unique S2 row ID."""

    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            raise S2EvaluationError(f"a {label} row has no row_id")
        if row_id in result:
            raise S2EvaluationError(f"duplicate {label} row {row_id}")
        result[row_id] = dict(row)
    return result


def load_evaluator_panel(
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
]:
    """Load the public S2 identities and separately protected structured keys."""

    evaluator_manifest_path = evaluator_manifest_path.resolve()
    evaluator_manifest_sha256 = require_sha256(
        evaluator_manifest_sha256,
        "evaluator manifest identity",
    )
    if sha256_file(evaluator_manifest_path) != evaluator_manifest_sha256:
        raise S2EvaluationError("the S2 evaluator manifest identity changed")
    evaluator = read_json(evaluator_manifest_path, "evaluator manifest")
    scoring = evaluator.get("scoring")
    separation = evaluator.get("answer_separation")
    source_corpus = evaluator.get("source_corpus")
    if (
        evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
        or not isinstance(evaluator.get("contract_routes"), Mapping)
        or not isinstance(source_corpus, Mapping)
        or not isinstance(scoring, Mapping)
        or scoring.get("target_fields") != list(SCORED_TARGET_KEYS)
        or not isinstance(separation, Mapping)
        or separation.get("public_questions_end_at_current_user") is not True
        or separation.get("generation_reads_private_answer_keys") is not False
    ):
        raise S2EvaluationError("the S2 evaluator contract changed")
    panel = evaluator.get("test")
    if not isinstance(panel, Mapping):
        raise S2EvaluationError("the S2 evaluator has no test panel")
    root = evaluator_manifest_path.parent
    questions_path = _child_path(root, panel.get("questions_path"), "test questions")
    answers_path = _child_path(root, panel.get("answer_key_path"), "test answer key")
    if (
        sha256_file(questions_path) != panel.get("questions_sha256")
        or sha256_file(answers_path) != panel.get("answer_key_sha256")
        or (answers_path.stat().st_mode & 0o077) != 0
    ):
        raise S2EvaluationError("the S2 test panel or private-key mode changed")
    test_panel_id = require_sha256(panel.get("test_panel_id"), "test panel ID")

    question_rows = read_jsonl(questions_path, "test questions")
    raw_questions = _unique(question_rows, "question")
    questions: dict[str, dict[str, str]] = {}
    family_counts: dict[str, int] = defaultdict(int)
    for row_id, row in raw_questions.items():
        metadata = row.get("metadata")
        messages = row.get("messages")
        family = metadata.get("question_family") if isinstance(metadata, Mapping) else None
        if (
            set(row) != {"row_id", "messages", "tools", "metadata"}
            or not isinstance(metadata, Mapping)
            or set(metadata) != PUBLIC_METADATA_KEYS
            or metadata.get("split") != "test"
            or not isinstance(family, str)
            or _FAMILY.fullmatch(family) is None
            or not isinstance(messages, list)
            or len(messages) != 3
            or any(not isinstance(message, Mapping) for message in messages)
            or [message.get("role") for message in messages]
            != ["system", "harness", "user"]
            or not isinstance(row.get("tools"), list)
        ):
            raise S2EvaluationError(f"public S2 question {row_id} is invalid")
        questions[row_id] = {"question_family": family}
        family_counts[family] += 1

    raw_answers = _unique(read_jsonl(answers_path, "test answer key"), "answer")
    if set(questions) != set(raw_answers) or len(questions) != panel.get("row_count"):
        raise S2EvaluationError("the S2 question and answer record sets disagree")
    answers: dict[str, dict[str, Any]] = {}
    case_counts: dict[str, int] = defaultdict(int)
    for row_id, answer in raw_answers.items():
        metadata = answer.get("scoring_metadata")
        target = answer.get("target")
        family = questions[row_id]["question_family"]
        if (
            set(answer) != {"row_id", "scoring_metadata", "target"}
            or not isinstance(metadata, Mapping)
            or set(metadata) != {"question_family", "catalog_case_id"}
            or metadata.get("question_family") != family
            or not isinstance(metadata.get("catalog_case_id"), str)
            or not isinstance(target, Mapping)
            or set(target) != TARGET_KEYS
            or not isinstance(target.get("status"), str)
            or not isinstance(target.get("result"), Mapping)
            or target.get("continuation_decision") not in {"stop", "continue", "defer"}
            or type(target.get("blocked_reason_required")) is not bool
            or not isinstance(target.get("expected_tool_calls"), list)
            or not isinstance(target.get("expected_tool_responses"), list)
        ):
            raise S2EvaluationError(f"private S2 answer {row_id} is invalid")
        answers[row_id] = dict(answer)
        case_counts[str(metadata["catalog_case_id"])] += 1
    if (
        dict(sorted(family_counts.items())) != panel.get("family_counts")
        or dict(sorted(case_counts.items())) != panel.get("case_counts")
        or source_corpus.get("panel_id") != test_panel_id
    ):
        raise S2EvaluationError("the S2 evaluator counts or panel identity changed")
    return evaluator, questions, answers


def load_predictions(
    predictions_path: Path,
    generation_manifest_path: Path,
    *,
    expected_record_ids: set[str],
    expected_test_panel_id: str,
    expected_evaluator_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load one complete, offline, identity-bound S2 generation result."""

    predictions_path = predictions_path.resolve()
    generation_manifest_path = generation_manifest_path.resolve()
    manifest = read_json(generation_manifest_path, "generation manifest")
    claimed = manifest.get("manifest_sha256")
    identity = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    generation_config = manifest.get("generation_config")
    elapsed = manifest.get("elapsed_seconds")
    if (
        manifest.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or manifest.get("test_panel_id") != expected_test_panel_id
        or manifest.get("reads_private_answer_keys") is not False
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or sha256_file(predictions_path) != manifest.get("predictions_sha256")
        or not isinstance(generation_config, Mapping)
        or stable_sha256(generation_config) != manifest.get("generation_config_sha256")
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise S2EvaluationError("the S2 generation manifest is invalid")
    declared_evaluator = manifest.get("evaluator_manifest_sha256")
    if declared_evaluator != expected_evaluator_manifest_sha256:
        raise S2EvaluationError("generation used a different S2 evaluator")
    require_sha256(
        manifest.get("tool_fixtures_sha256"),
        "generation tool fixtures identity",
    )
    for key in PUBLICATION_IDENTITY_KEYS[2:]:
        require_sha256(manifest.get(key), f"generation {key}")
    for key in PUBLICATION_IDENTITY_KEYS[:2]:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise S2EvaluationError(f"generation {key} must be nonempty text")

    predictions = _unique(read_jsonl(predictions_path, "predictions"), "prediction")
    if set(predictions) != expected_record_ids or len(predictions) != manifest.get("record_count"):
        raise S2EvaluationError("the S2 prediction record set is incomplete")
    for row_id, prediction in predictions.items():
        error = prediction.get("trajectory_error")
        if (
            set(prediction) != PREDICTION_KEYS
            or not isinstance(prediction.get("generated_messages"), list)
            or not isinstance(prediction.get("generations"), list)
            or not isinstance(prediction.get("stop_reason"), str)
            or error is not None and not isinstance(error, str)
            or type(prediction.get("tool_call_count")) is not int
            or prediction["tool_call_count"] < 0
            or type(prediction.get("executed_tool_call_count")) is not int
            or prediction["executed_tool_call_count"] < 0
            or prediction.get("network_used_false") is not True
        ):
            raise S2EvaluationError(f"S2 prediction {row_id} has an invalid harness shape")
    return manifest, predictions


def _trajectory(
    messages: Any,
) -> tuple[
    bool,
    list[dict[str, Any]],
    list[dict[str, Any]],
    Mapping[str, Any] | None,
    str | None,
]:
    """Parse the native generated suffix while ignoring reasoning and tool payloads."""

    if not isinstance(messages, list) or not messages:
        return False, [], [], None, "generated_suffix_is_empty"
    calls: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    index = 0
    while index < len(messages) - 1:
        call_message = messages[index]
        if not isinstance(call_message, Mapping):
            return False, calls, responses, None, "generated_message_is_not_object"
        call = call_message.get("tool_call")
        if (
            call_message.get("role") != "assistant"
            or not isinstance(call, Mapping)
            or set(call) != {"name", "arguments"}
            or not isinstance(call.get("name"), str)
            or not isinstance(call.get("arguments"), Mapping)
        ):
            return False, calls, responses, None, "expected_assistant_tool_call"
        calls.append(dict(call))
        index += 1
        if index >= len(messages) - 1:
            return False, calls, responses, None, "tool_call_has_no_response_or_final"
        response = messages[index]
        if (
            not isinstance(response, Mapping)
            or response.get("role") != "tool"
            or response.get("name") != call["name"]
            or not isinstance(response.get("content"), Mapping)
        ):
            return False, calls, responses, None, "tool_response_shape_changed"
        responses.append({"name": response["name"], "content": response["content"]})
        index += 1
    final_message = messages[index]
    if (
        not isinstance(final_message, Mapping)
        or final_message.get("role") != "assistant"
        or "tool_call" in final_message
        or not isinstance(final_message.get("content"), Mapping)
    ):
        return False, calls, responses, None, "final_assistant_object_is_absent"
    return True, calls, responses, final_message["content"], None


def score_s2_record(
    question: Mapping[str, str],
    answer: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Score only S2 structured fields, blocker presence, and ordered calls."""

    target = answer["target"]
    valid, calls, responses, final, parse_error = _trajectory(
        prediction["generated_messages"]
    )
    trajectory_error = prediction.get("trajectory_error")
    model_error = trajectory_error is not None
    valid = (
        valid
        and not model_error
        and prediction["stop_reason"] == "final"
        and prediction["tool_call_count"] == len(calls)
        and prediction["executed_tool_call_count"] == len(calls)
    )
    if (
        valid
        and calls == target["expected_tool_calls"]
        and responses != target["expected_tool_responses"]
    ):
        raise S2EvaluationError(
            f"the injected tool fixtures changed for {prediction['row_id']}"
        )
    if final is None:
        final = {}
    required_blocker = target["blocked_reason_required"]
    blocker = final.get("blocked_reason")
    blocker_exact = (
        isinstance(blocker, str) and bool(blocker.strip())
        if required_blocker
        else blocker is None
    )
    result = {
        "row_id": prediction["row_id"],
        "question_family": question["question_family"],
        "catalog_case_id": answer["scoring_metadata"]["catalog_case_id"],
        "model_error": model_error,
        "trajectory_error": trajectory_error,
        "parse_error": parse_error,
        "valid_trajectory": valid,
        "status_accuracy": valid and final.get("status") == target["status"],
        "result_accuracy": valid and final.get("result") == target["result"],
        "continuation_decision_accuracy": valid
        and final.get("continuation_decision") == target["continuation_decision"],
        "provenance_accuracy": valid
        and final.get("provenance") == target["provenance"],
        "blocked_reason_presence_accuracy": valid and blocker_exact,
        "exact_tool_calls": valid and calls == target["expected_tool_calls"],
        "expected_tool_call_count": len(target["expected_tool_calls"]),
        "generated_tool_call_count": len(calls),
        "executed_tool_call_count": prediction["executed_tool_call_count"],
        "stop_reason": prediction["stop_reason"],
    }
    result["structured_accuracy"] = all(
        result[field]
        for field in (
            "status_accuracy",
            "result_accuracy",
            "continuation_decision_accuracy",
            "provenance_accuracy",
            "blocked_reason_presence_accuracy",
            "exact_tool_calls",
        )
    )
    return result


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return exact counts and rates for one scored row group."""

    total = len(rows)
    result: dict[str, Any] = {"row_count": total}
    for field in SCORE_FIELDS:
        correct = sum(row.get(field) is True for row in rows)
        result[f"{field}_count"] = correct
        result[f"{field}_rate"] = correct / total if total else 0.0
    errors = sum(row.get("model_error") is True for row in rows)
    result["model_error_count"] = errors
    result["model_error_rate"] = errors / total if total else 0.0
    return result


def build_s2_report(
    *,
    evaluator_manifest_sha256: str,
    generation: Mapping[str, Any],
    questions: Mapping[str, Mapping[str, str]],
    answers: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    gate_profile: str,
    minimum_structured_accuracy: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete S2 score rows, diagnostics, and an explicit gate state."""

    score_rows = [
        score_s2_record(questions[row_id], answers[row_id], predictions[row_id])
        for row_id in questions
    ]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        families[row["question_family"]].append(row)
        cases[row["catalog_case_id"]].append(row)
    overall = _summary(score_rows)
    if minimum_structured_accuracy is None:
        gate = {
            "profile": gate_profile,
            "mode": "diagnostic",
            "minimum_structured_accuracy": None,
            "passed": None,
        }
    else:
        gate = {
            "profile": gate_profile,
            "mode": "gate",
            "minimum_structured_accuracy": minimum_structured_accuracy,
            "passed": overall["structured_accuracy_rate"]
            >= minimum_structured_accuracy,
        }
    identity = {
        key: generation[key]
        for key in PUBLICATION_IDENTITY_KEYS
    }
    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": generation["test_panel_id"],
        **identity,
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "generation_manifest_sha256": generation["manifest_sha256"],
        "generation_bundle_sha256": generation["generation_bundle_sha256"],
        "generation_config_sha256": generation["generation_config_sha256"],
        "predictions_sha256": generation["predictions_sha256"],
        "tool_fixtures_sha256": generation["tool_fixtures_sha256"],
        "complete_coverage": True,
        "overall": overall,
        "families": {
            key: _summary(rows) for key, rows in sorted(families.items())
        },
        "cases": {key: _summary(rows) for key, rows in sorted(cases.items())},
        "gate": gate,
    }
    metrics["metrics_sha256"] = stable_sha256(metrics)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": generation["test_panel_id"],
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "generation_manifest_sha256": generation["manifest_sha256"],
        "metrics_sha256": metrics["metrics_sha256"],
        "records": score_rows,
    }
    report["report_sha256"] = stable_sha256(report)
    return report, metrics


def validate_publication_identity(
    generation: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    """Fail closed before resuming the externally visible training run."""

    if expected is None or set(expected) != set(PUBLICATION_IDENTITY_KEYS):
        raise S2EvaluationError("S2 W&B publication lacks its expected identity")
    for key in PUBLICATION_IDENTITY_KEYS:
        value = expected[key]
        if key.endswith("_sha256"):
            require_sha256(value, f"expected publication {key}")
        elif not isinstance(value, str) or not value:
            raise S2EvaluationError(f"expected publication {key} is invalid")
        if generation.get(key) != value:
            raise S2EvaluationError(f"S2 W&B publication {key} changed")


def publish_wandb(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Resume the training run and publish only headline S2 scalars."""

    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "").strip()
    if mode != "online" or not entity or not project:
        raise S2EvaluationError("the S2 W&B publication contract is invalid")
    try:
        import wandb
    except ImportError as error:
        raise S2EvaluationError("the wandb package is absent") from error
    run = wandb.init(
        entity=entity,
        project=project,
        id=str(metrics["train_run_id"]),
        resume="must",
        mode=mode,
        dir=os.environ.get("WANDB_DIR"),
    )
    if run is None or getattr(run, "disabled", False):
        raise S2EvaluationError("W&B did not resume the S2 training run")
    payload = {
        "test/s2/structured_accuracy": metrics["overall"]["structured_accuracy_rate"],
        "test/s2/complete_coverage": int(metrics["complete_coverage"]),
    }
    if metrics["gate"]["passed"] is not None:
        payload["test/s2/gate_passed"] = int(metrics["gate"]["passed"])
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


def evaluate_s2_test(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    predictions_path: Path,
    generation_manifest_path: Path,
    output_dir: Path,
    gate_profile: str,
    minimum_structured_accuracy: float | None = None,
    publish: bool = False,
    expected_publication_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one complete S2 panel and optionally publish its headline."""

    if not isinstance(gate_profile, str) or not gate_profile:
        raise S2EvaluationError("the S2 gate profile must be nonempty")
    if gate_profile == REPORT_ONLY_GATE_PROFILE and minimum_structured_accuracy is not None:
        raise S2EvaluationError("report_only_v1 cannot claim a gate threshold")
    if gate_profile != REPORT_ONLY_GATE_PROFILE and minimum_structured_accuracy is None:
        raise S2EvaluationError("a non-diagnostic S2 profile requires a threshold")
    if minimum_structured_accuracy is not None and not 0.0 <= minimum_structured_accuracy <= 1.0:
        raise S2EvaluationError("the S2 structured-accuracy threshold is invalid")
    evaluator, questions, answers = load_evaluator_panel(
        evaluator_manifest_path,
        evaluator_manifest_sha256,
    )
    generation, predictions = load_predictions(
        predictions_path,
        generation_manifest_path,
        expected_record_ids=set(questions),
        expected_test_panel_id=evaluator["test"]["test_panel_id"],
        expected_evaluator_manifest_sha256=evaluator_manifest_sha256,
    )
    report, metrics = build_s2_report(
        evaluator_manifest_sha256=evaluator_manifest_sha256,
        generation=generation,
        questions=questions,
        answers=answers,
        predictions=predictions,
        gate_profile=gate_profile,
        minimum_structured_accuracy=minimum_structured_accuracy,
    )
    output_dir = output_dir.resolve()
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "metrics.json", metrics)
    if publish:
        validate_publication_identity(generation, expected_publication_identity)
        receipt = publish_wandb(metrics)
        write_json(output_dir / "wandb_receipt.json", receipt, private=True)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse exact S2 scorer inputs and explicit publication identities."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-profile", required=True)
    parser.add_argument("--minimum-structured-accuracy", type=float)
    parser.add_argument("--publish-wandb", action="store_true")
    for key in PUBLICATION_IDENTITY_KEYS:
        parser.add_argument("--expected-" + key.replace("_", "-"))
    return parser.parse_args()


def main() -> int:
    """Score S2 predictions and print the compact headline metrics."""

    args = parse_args()
    expected = {
        key: getattr(args, "expected_" + key)
        for key in PUBLICATION_IDENTITY_KEYS
    }
    if not args.publish_wandb:
        expected = None
    metrics = evaluate_s2_test(
        evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256,
        predictions_path=args.predictions,
        generation_manifest_path=args.generation_manifest,
        output_dir=args.output_dir,
        gate_profile=args.gate_profile,
        minimum_structured_accuracy=args.minimum_structured_accuracy,
        publish=args.publish_wandb,
        expected_publication_identity=expected,
    )
    print(
        json.dumps(
            {
                "row_count": metrics["overall"]["row_count"],
                "structured_accuracy": metrics["overall"]["structured_accuracy_rate"],
                "gate": metrics["gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
