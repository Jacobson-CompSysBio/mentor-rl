"""Validate and score the three frozen retry diagnostics separately from promotion."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_lookup_retry import (
    DIAGNOSTIC_PREFIX, ID_DIAGNOSTIC_PREFIX, audit_retry_row, diagnostic_retry_case,
)
from runtime.world_model.s1_membership_retry import (
    DIAGNOSTIC_PREFIX as MEMBERSHIP_PREFIX, audit_membership_retry_row,
    diagnostic_membership_retry_case,
)
from scripts.world_model_s1_eval_common import (
    S1EvaluationError, read_json, read_jsonl, require_sha256, resolve_repo_path,
    sha256_file, stable_sha256, validate_answer_row, validate_question, write_json,
)

PANELS = {
    "lookup_retries": (150, DIAGNOSTIC_PREFIX, "S1.1.8", "s1_1", "s1_v6_lookup_retries.json"),
    "id_lookup_retries": (150, ID_DIAGNOSTIC_PREFIX, "S1.2.11", "s1_2", "s1_v6_id_lookup_retries.json"),
    "membership_retries": (111, MEMBERSHIP_PREFIX, "S1.3.10", "s1_3", "s1_v6_membership_retries.json"),
}


def retry_case(record_id: str) -> str | None:
    """Read the diagnostic fault schedule from its separate record namespace."""

    return diagnostic_membership_retry_case(record_id) or diagnostic_retry_case(record_id)


def load_retry_panel(manifest_path, manifest_sha256, *, repo_root=REPO_ROOT,
                     contract=None, read_answers=True):
    """Check a frozen diagnostic's provenance and records; public loading avoids keys."""

    repo_root, manifest_path = Path(repo_root).resolve(), Path(manifest_path).resolve()
    kind = manifest_path.parent.name
    if kind not in PANELS or manifest_path != repo_root / "data/world_model_v4/eval/s1_v6" / kind / "manifest.json":
        raise S1EvaluationError("unknown retry diagnostic location")
    if sha256_file(manifest_path) != require_sha256(manifest_sha256, "retry manifest"):
        raise S1EvaluationError("retry diagnostic manifest identity changed")
    manifest = read_json(manifest_path)
    contract = contract or load_contract(repo_root / "runtime/world_model/contract.yaml")
    count, prefix, trace, component, policy_file = PANELS[kind]
    schema = "mentor-rl-s1-membership-retry-diagnostic-v1" if kind == "membership_retries" else "mentor-rl-s1-lookup-retry-diagnostic-v1"
    overlay_path = "runtime/world_model/s1_v6.yaml"
    policy_path = "runtime/world_model/" + policy_file
    base_path = "data/world_model_v4/corpora/s1_v5/" + component
    if (
        manifest.get("schema_version") != schema
        or manifest.get("row_count") != count
        or manifest.get("max_retries") != 1
        or manifest.get("case_counts") != {case: count // 3 for case in ("no_retry", "recovered", "exhausted")}
        or manifest.get("s1_overlay") != {
            "path": overlay_path, "sha256": sha256_file(repo_root / overlay_path),
            "schema_version": "mentor-rl-world-model-s1-overlay-v5",
        }
        or manifest.get("policy") != {"path": policy_path, "sha256": sha256_file(repo_root / policy_path)}
        or manifest.get("source_index_sha256") != contract["source_index"]["content_sha256"]
        or manifest.get("base_test_rows") != {
            "path": base_path + "/test.jsonl", "sha256": sha256_file(repo_root / base_path / "test.jsonl"),
        }
        or manifest.get("base_manifest_sha256") != sha256_file(repo_root / base_path / "manifest.json")
    ):
        raise S1EvaluationError("retry diagnostic contract or source pin changed")
    implementation = manifest.get("implementation_sha256")
    if not isinstance(implementation, Mapping) or not implementation:
        raise S1EvaluationError("retry diagnostic lacks implementation pins")
    for path, digest in implementation.items():
        if sha256_file(resolve_repo_path(repo_root, path, "retry implementation")) != digest:
            raise S1EvaluationError("retry diagnostic implementation pin changed")
    files = manifest.get("files", {})
    questions_path = manifest_path.parent / "test_questions.jsonl"
    answers_path = manifest_path.parent / "test_answer_key.jsonl"
    for name in ("test_questions.jsonl", "test_answer_key.jsonl"):
        spec = files.get(name, {})
        require_sha256(spec.get("sha256"), "retry " + name)
        if spec.get("row_count") != count:
            raise S1EvaluationError("retry diagnostic file count changed")
        if (read_answers or name == "test_questions.jsonl") and sha256_file(manifest_path.parent / name) != spec["sha256"]:
            raise S1EvaluationError("retry diagnostic file identity changed")
    if manifest.get("test_panel_id") != files["test_questions.jsonl"]["sha256"]:
        raise S1EvaluationError("retry diagnostic panel identity changed")
    questions = [validate_question(row, contract) for row in read_jsonl(questions_path)]
    by_id = {q["record_id"]: q for q in questions}
    if len(questions) != count or len(by_id) != count or any(
        not rid.startswith(prefix) or q["metadata"]["source_trace"] != trace
        for rid, q in by_id.items()
    ):
        raise S1EvaluationError("retry diagnostic record namespace or count changed")
    try:
        cases = Counter(retry_case(rid) for rid in by_id)
    except ValueError as error:
        raise S1EvaluationError("retry diagnostic case changed") from error
    if cases != manifest["case_counts"]:
        raise S1EvaluationError("retry diagnostic case coverage changed")
    answers = {}
    if read_answers:
        raw_answers = read_jsonl(answers_path)
        if len(raw_answers) != count or {row["record_id"] for row in raw_answers} != set(by_id):
            raise S1EvaluationError("retry diagnostic answer record set changed")
        for answer in raw_answers:
            rid = answer["record_id"]
            answers[rid] = validate_answer_row(answer, by_id[rid], contract)
            row = {"row_id": rid, "messages": by_id[rid]["messages"] + answer["expected_messages"],
                   "metadata": by_id[rid]["metadata"]}
            try:
                audit = audit_membership_retry_row if kind == "membership_retries" else audit_retry_row
                audit(row, retry_case(rid))
            except (ValueError, KeyError, TypeError) as error:
                raise S1EvaluationError("retry diagnostic answer violates its fault schedule") from error
    panel = {
        "kind": kind, "diagnostic_only": True, "row_count": count,
        "test_panel_id": manifest["test_panel_id"],
        "questions_path": questions_path.relative_to(repo_root).as_posix(),
        "questions_sha256": files["test_questions.jsonl"]["sha256"],
        "answer_key_path": answers_path.relative_to(repo_root).as_posix(),
        "answer_key_sha256": files["test_answer_key.jsonl"]["sha256"],
        "family_counts": dict(Counter(q["metadata"]["question_family"] for q in questions)),
    }
    return panel, questions, answers


def validate_retry_config(config, *, repo_root, contract):
    """Require an isolated diagnostic config and its exact frozen panel."""

    spec = config.get("diagnostic_panel")
    if (not isinstance(spec, Mapping) or set(spec) != {"kind", "manifest", "manifest_sha256"}
            or spec["kind"] not in PANELS
            or config.get("gate") != {"profile": "diagnostic_only", "require_network_used_false": True}
            or config.get("tracking", {}).get("resume_training_run") is not False):
        raise S1EvaluationError("retry diagnostics require separate reporting and a frozen panel")
    path = resolve_repo_path(repo_root, spec["manifest"], "retry manifest")
    if path.parent.name != spec["kind"]:
        raise S1EvaluationError("retry diagnostic kind disagrees with its manifest")
    expected_calls = 3 if spec["kind"] == "membership_retries" else 2
    if config.get("generation", {}).get("maximum_tool_calls") != expected_calls:
        raise S1EvaluationError("retry diagnostic call limit changed")
    panel, _, _ = load_retry_panel(path, spec["manifest_sha256"], repo_root=repo_root, contract=contract)
    return panel, path, spec["manifest_sha256"]


def score_retry_diagnostic(*, manifest_path, manifest_sha256, predictions_path,
                           generation_manifest_path, tokenizer_manifest_path,
                           output_dir, expected_identity, repo_root=REPO_ROOT):
    """Score complete retry trajectories and report rates without a promotion gate."""

    from scripts.evaluate_world_model_s1 import (
        load_predictions, score_s1_record, summarize_scores,
        validate_generated_evidence_policy, validate_publication_identity,
    )
    from runtime.world_model_training import validated_tokenizer_manifest

    contract = load_contract(Path(repo_root) / "runtime/world_model/contract.yaml")
    panel, rows, answers = load_retry_panel(manifest_path, manifest_sha256, repo_root=repo_root, contract=contract)
    questions = {q["record_id"]: q for q in rows}
    generation, predictions = load_predictions(
        predictions_path, generation_manifest_path, expected_record_ids=set(questions),
        expected_test_panel_id=panel["test_panel_id"],
    )
    validate_publication_identity(generation, expected_identity)
    tokenizer = validated_tokenizer_manifest(Path(tokenizer_manifest_path).resolve())
    if tokenizer["manifest_sha256"] != generation["tokenizer_manifest_sha256"]:
        raise S1EvaluationError("retry diagnostic tokenizer identity changed")
    validate_generated_evidence_policy(questions, predictions, contract)
    scores = [score_s1_record(questions[rid], answers[rid], predictions[rid], contract=contract) for rid in sorted(questions)]
    metrics = {
        "schema_version": "mentor-rl-s1-retry-diagnostic-metrics-v1", "diagnostic_only": True,
        "kind": panel["kind"], "test_panel_id": panel["test_panel_id"],
        "diagnostic_manifest_sha256": manifest_sha256, "generation_manifest_sha256": generation["manifest_sha256"],
        "row_count": len(scores), "overall": summarize_scores(scores),
        "by_case": {case: summarize_scores([s for s in scores if retry_case(s["record_id"]) == case])
                    for case in ("no_retry", "recovered", "exhausted")},
        "identity": dict(expected_identity),
    }
    metrics["metrics_sha256"] = stable_sha256(metrics)
    output_dir = Path(output_dir)
    write_json(output_dir / "retry_metrics.json", metrics)
    write_json(output_dir / "retry_report.json", {"diagnostic_only": True, "metrics_sha256": metrics["metrics_sha256"], "records": scores}, private=True)
    return metrics


def main():
    """Run the dedicated diagnostic scorer using the launcher's verified identities."""

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "predictions", "generation-manifest", "tokenizer-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    names = {
        "train_run_id": "S1_TRAIN_RUN_ID", "method_id": "S1_METHOD_ID",
        "checkpoint_identity_sha256": "S1_CHECKPOINT_ARTIFACT_SHA256",
        "base_model_identity_sha256": "MODEL_IDENTITY_SHA256", "base_model_artifact_sha256": "BASE_MODEL_ARTIFACT_SHA256",
        "tokenizer_manifest_sha256": "S1_TOKENIZER_MANIFEST_SHA256", "tokenizer_artifacts_sha256": "S1_TOKENIZER_ARTIFACTS_SHA256",
        "source_index_sha256": "S1_SOURCE_INDEX_SHA256", "generation_bundle_sha256": "S1_GENERATION_BUNDLE_SHA256",
    }
    metrics = score_retry_diagnostic(
        manifest_path=args.manifest, manifest_sha256=args.manifest_sha256,
        predictions_path=args.predictions, generation_manifest_path=args.generation_manifest,
        tokenizer_manifest_path=args.tokenizer_manifest, output_dir=args.output_dir,
        expected_identity={key: os.environ.get(env) for key, env in names.items()},
    )
    print(f"Separate {metrics['kind']} diagnostic complete: {metrics['row_count']} rows; no promotion gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
