#!/usr/bin/env python3
"""Score the separate, training-owned S1 robustness panel without promotion gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract
from scripts.world_model_s1_eval_common import (
    S1EvaluationError, final_message, normalize_evidence_ids, read_json,
    read_jsonl, require_sha256, resolve_repo_path, sha256_file, stable_sha256,
    tool_calls, validate_answer_row, validate_question, write_json,
)

PANEL_SCHEMA = "mentor-rl-s1-robustness-diagnostic-v1"
PAIR_SCHEMA = "mentor-rl-s1-robustness-pairs-v1"
CONFIG_SCHEMA = "mentor-rl-world-model-s1-robustness-test-v1"
KIND = "robustness_v1"
RELATION_BY_AXIS = {
    "wording": {"invariant"},
    "task_scope": {"scope_flip"},
    "evidence_order": {"invariant", "mention_swap"},
    "id_notation_case": {"normalized_identity_invariant"},
    "survivor_position": {"source_backed_strata"},
}
CORRECTNESS_CHECKS = (
    "valid_trajectory", "exact_tool_selection", "exact_tool_arguments",
    "tool_payload_consistency", "evidence_consistency",
    "structured_answer_accuracy", "failure_harness_accuracy",
)


def _validate_groups(groups, questions):
    """Require complete, disjoint pairs and explicitly distinct-fact coverage groups."""

    if not isinstance(groups, list) or not groups:
        raise S1EvaluationError("robustness panel has no groups")
    seen_groups, seen_records = set(), set()
    for group in groups:
        if not isinstance(group, Mapping):
            raise S1EvaluationError("robustness group is not an object")
        group_id, axis = group.get("group_id"), group.get("axis")
        if not isinstance(group_id, str) or not group_id or group_id in seen_groups:
            raise S1EvaluationError("robustness group identity is duplicated or missing")
        if axis not in RELATION_BY_AXIS or group.get("expected_relationship") not in RELATION_BY_AXIS[axis]:
            raise S1EvaluationError("robustness group relationship changed")
        expected_kind = "coverage_group" if axis == "survivor_position" else "pair"
        members = group.get("members")
        if group.get("kind") != expected_kind or not isinstance(members, list) or len(members) != 2:
            raise S1EvaluationError("robustness group must contain exactly two declared members")
        roles = set()
        for member in members:
            if not isinstance(member, Mapping):
                raise S1EvaluationError("robustness member is not an object")
            rid, role = member.get("record_id"), member.get("role")
            if rid not in questions or rid in seen_records:
                raise S1EvaluationError("robustness group record set changed")
            if not isinstance(role, str) or not role or role in roles:
                raise S1EvaluationError("robustness member roles are missing or repeated")
            if not all(isinstance(member.get(key), str) and member[key]
                       for key in ("source_row_id", "source_component")):
                raise S1EvaluationError("robustness member lacks training provenance")
            roles.add(role)
            seen_records.add(rid)
        if len({questions[m["record_id"]]["metadata"]["source_trace"] for m in members}) != 1:
            raise S1EvaluationError("robustness group crosses question types")
        seen_groups.add(group_id)
    if seen_records != set(questions):
        raise S1EvaluationError("robustness groups do not partition all questions")


def _structured_view(messages):
    """Retain calls and structured final fields while removing prose and nonce spelling."""

    final = final_message(messages)
    if final is None or not isinstance(final.get("content"), Mapping):
        raise S1EvaluationError("robustness comparison lacks a final object")
    content = deepcopy(dict(final["content"]))
    content.pop("summary", None)
    reason = content.pop("blocked_reason", None)
    content["blocked_reason_present"] = isinstance(reason, str) and bool(reason.strip())
    return normalize_evidence_ids({"tool_calls": tool_calls(messages), "final": content})


def _comparison_view(messages, expected_messages, relationship, member_index=0):
    """Apply only the declared transformation, checking transformed literals separately."""

    view = _structured_view(messages)
    expected = _structured_view(expected_messages)
    if relationship == "scope_flip":
        if view["final"].get("continuation_decision") != expected["final"]["continuation_decision"]:
            return None
        view["final"].pop("continuation_decision")
    elif relationship == "normalized_identity_invariant":
        result, gold = view["final"].get("result"), expected["final"].get("result")
        if not isinstance(result, Mapping) or not isinstance(gold, Mapping):
            return None
        for side in ("first", "second"):
            if not isinstance(result.get(side), Mapping) or not isinstance(gold.get(side), Mapping):
                return None
            if result[side].get("input") != gold[side].get("input") or "input" not in result[side]:
                return None
            result[side]["input"] = None
    elif relationship == "mention_swap":
        result = view["final"].get("result")
        if not isinstance(result, Mapping) or not all(side in result for side in ("first", "second")):
            return None
        if member_index == 1:
            result["first"], result["second"] = result["second"], result["first"]
    return view


def _validate_gold_relationships(groups, answers):
    """Reject a panel whose declared equivalences disagree with its structured gold."""

    for group in groups:
        if group["kind"] == "coverage_group":
            continue
        suffixes = [answers[m["record_id"]]["expected_messages"] for m in group["members"]]
        relation = group["expected_relationship"]
        if relation == "scope_flip" and {
            final_message(suffix)["content"]["continuation_decision"] for suffix in suffixes
        } != {"stop", "continue"}:
            raise S1EvaluationError("robustness scope pair does not flip stop/continue")
        views = [_comparison_view(suffix, suffix, relation, index) for index, suffix in enumerate(suffixes)]
        if any(view is None for view in views) or views[0] != views[1]:
            raise S1EvaluationError("robustness gold violates its declared relationship")


def load_robustness_panel(manifest_path, manifest_sha256, *, repo_root=REPO_ROOT,
                          contract=None, read_answers=True):
    """Load a pinned training-owned diagnostic; public mode never opens private rows/keys."""

    root = Path(repo_root).resolve()
    path = resolve_repo_path(root, str(manifest_path), "robustness manifest")
    if sha256_file(path) != require_sha256(manifest_sha256, "robustness manifest"):
        raise S1EvaluationError("robustness manifest identity changed")
    manifest = read_json(path)
    contract = contract or load_contract(root / "runtime/world_model/contract.yaml")
    count = manifest.get("row_count")
    if (manifest.get("schema_version") != PANEL_SCHEMA or manifest.get("kind") != KIND
            or manifest.get("diagnostic_only") is not True or manifest.get("source_split") != "train"
            or manifest.get("source_index_sha256") != contract["source_index"]["content_sha256"]
            or manifest.get("max_tool_calls") != 2 or type(count) is not int or count <= 0):
        raise S1EvaluationError("robustness panel identity or training ownership changed")
    for label in ("inputs", "implementation_sha256"):
        pins = manifest.get(label)
        if not isinstance(pins, Mapping) or not pins:
            raise S1EvaluationError("robustness panel lacks source/implementation pins")
        for name, digest in pins.items():
            source = resolve_repo_path(root, name, "robustness " + label)
            if sha256_file(source) != require_sha256(digest, "robustness " + label):
                raise S1EvaluationError("robustness source/implementation identity changed")
    files = manifest.get("files", {})
    names = ("test_questions.jsonl", "test_answer_key.jsonl", "rows.jsonl", "pairs.json")
    if not isinstance(files, Mapping) or set(files) != set(names):
        raise S1EvaluationError("robustness panel file set changed")
    for name in names:
        spec = files[name]
        if not isinstance(spec, Mapping):
            raise S1EvaluationError("robustness file pin is invalid")
        digest = require_sha256(spec.get("sha256"), "robustness " + name)
        if name != "pairs.json" and spec.get("row_count") != count:
            raise S1EvaluationError("robustness file row count changed")
        if (read_answers or name in {"test_questions.jsonl", "pairs.json"}) and sha256_file(path.parent / name) != digest:
            raise S1EvaluationError("robustness file identity changed")
    if manifest.get("test_panel_id") != files["test_questions.jsonl"]["sha256"]:
        raise S1EvaluationError("robustness question identity changed")
    rows = [validate_question(row, contract) for row in read_jsonl(path.parent / "test_questions.jsonl")]
    questions = {row["record_id"]: row for row in rows}
    if len(rows) != count or len(questions) != count or any(
        not rid.startswith("wm_s1_robust_v1_") for rid in questions
    ):
        raise S1EvaluationError("robustness question count or namespace changed")
    pairs = read_json(path.parent / "pairs.json")
    if pairs.get("schema_version") != PAIR_SCHEMA:
        raise S1EvaluationError("robustness pair schema changed")
    groups = pairs.get("groups")
    _validate_groups(groups, questions)
    axes = Counter(group["axis"] for group in groups)
    if (set(axes) != set(RELATION_BY_AXIS) or manifest.get("group_counts") != dict(axes)
            or manifest.get("axis_counts") != {axis: amount * 2 for axis, amount in axes.items()}
            or manifest.get("group_count") != len(groups)
            or files["pairs.json"].get("group_count") != len(groups)):
        raise S1EvaluationError("robustness axis/group coverage changed")
    families = dict(Counter(row["metadata"]["question_family"] for row in rows))
    if families != manifest.get("family_counts"):
        raise S1EvaluationError("robustness family coverage changed")
    answers = {}
    if read_answers:
        raw = read_jsonl(path.parent / "test_answer_key.jsonl")
        if len(raw) != count or {row.get("record_id") for row in raw} != set(questions):
            raise S1EvaluationError("robustness answer record set changed")
        answers = {row["record_id"]: validate_answer_row(row, questions[row["record_id"]], contract) for row in raw}
        complete = read_jsonl(path.parent / "rows.jsonl")
        if len(complete) != count or {row.get("row_id") for row in complete} != set(questions):
            raise S1EvaluationError("robustness complete-row identity changed")
        for row in complete:
            question, answer = questions[row["row_id"]], answers[row["row_id"]]
            if (row.get("messages") != question["messages"] + answer["expected_messages"]
                    or row.get("metadata") != question["metadata"] or row.get("tools") != question["tools"]):
                raise S1EvaluationError("robustness complete trace disagrees with question/key")
        _validate_gold_relationships(groups, answers)
    panel = {
        "kind": KIND, "diagnostic_only": True, "source_split": "train", "row_count": count,
        "test_panel_id": manifest["test_panel_id"], "family_counts": families,
        "questions_path": (path.parent / "test_questions.jsonl").relative_to(root).as_posix(),
        "questions_sha256": files["test_questions.jsonl"]["sha256"],
        "answer_key_path": (path.parent / "test_answer_key.jsonl").relative_to(root).as_posix(),
        "answer_key_sha256": files["test_answer_key.jsonl"]["sha256"], "groups": groups,
    }
    return panel, rows, answers


def validate_robustness_config(config, *, repo_root, contract):
    """Require an isolated diagnostic launch with retained current-turn reasoning."""

    spec = config.get("diagnostic_panel")
    if (config.get("schema_version") != CONFIG_SCHEMA or not isinstance(spec, Mapping)
            or set(spec) != {"kind", "manifest", "manifest_sha256"} or spec.get("kind") != KIND
            or config.get("gate") != {"profile": "diagnostic_only", "require_network_used_false": True}
            or config.get("tracking", {}).get("resume_training_run") is not False
            or config.get("generation", {}).get("maximum_tool_calls") != 2
            or config.get("generation", {}).get("history_policy") != "retain_current_turn"):
        raise S1EvaluationError("robustness launch must retain reasoning and remain diagnostic-only")
    path = resolve_repo_path(Path(repo_root), spec["manifest"], "robustness manifest")
    panel, _, _ = load_robustness_panel(path, spec["manifest_sha256"], repo_root=repo_root, contract=contract)
    return panel, path, spec["manifest_sha256"]


def _summary(rows, groups):
    """Report correctness separately from consistency, including stable wrong pairs."""

    pairs = [group for group in groups if group["kind"] == "pair"]
    correct = sum(row["structured_correct"] for row in rows)
    return {
        "row_count": len(rows), "structured_accuracy_count": correct,
        "structured_accuracy_rate": correct / len(rows) if rows else None,
        "pair_count": len(pairs), "both_correct_count": sum(p["all_correct"] for p in pairs),
        "both_correct_rate": sum(p["all_correct"] for p in pairs) / len(pairs) if pairs else None,
        "consistent_pair_count": sum(p["consistent"] for p in pairs),
        "pair_consistency_rate": sum(p["consistent"] for p in pairs) / len(pairs) if pairs else None,
        "stable_wrong_pair_count": sum(p["stable_wrong"] for p in pairs),
        "coverage_group_count": sum(g["kind"] == "coverage_group" for g in groups),
        "coverage_group_all_correct_count": sum(g["kind"] == "coverage_group" and g["all_correct"] for g in groups),
        "failure_checks": dict(Counter(check for row in rows for check in row["failed_checks"])),
    }


def score_robustness_records(questions, answers, predictions, groups, *, contract):
    """Score a complete panel with native S1 checks and explicit pair transformations."""

    from scripts.evaluate_world_model_s1 import score_s1_record, validate_generated_evidence_policy

    if set(questions) != set(answers) or set(questions) != set(predictions):
        raise S1EvaluationError("robustness scoring requires the complete matched record set")
    if any(prediction.get("record_id") != rid or prediction.get("network_used_false") is not True
           for rid, prediction in predictions.items()):
        raise S1EvaluationError("robustness prediction identity or network isolation changed")
    _validate_groups(groups, questions)
    _validate_gold_relationships(groups, answers)
    validate_generated_evidence_policy(questions, predictions, contract)
    membership = {member["record_id"]: (group, member) for group in groups for member in group["members"]}
    scores = {}
    for rid in questions:
        score = score_s1_record(questions[rid], answers[rid], predictions[rid], contract=contract)
        group, member = membership[rid]
        score.update(axis=group["axis"], group_id=group["group_id"], role=member["role"],
                     expected_relationship=group["expected_relationship"])
        score["failed_checks"] = [check for check in CORRECTNESS_CHECKS if score[check] is not True]
        score["structured_correct"] = not score["failed_checks"]
        scores[rid] = score
    group_scores = []
    for group in groups:
        ids = [member["record_id"] for member in group["members"]]
        correct = [scores[rid]["structured_correct"] for rid in ids]
        consistent = None
        if group["kind"] == "pair":
            valid = all(scores[rid]["valid_trajectory"] and scores[rid]["structured_final_json"]
                        and scores[rid]["evidence_consistency"] for rid in ids)
            views = [_comparison_view(predictions[rid]["generated_messages"], answers[rid]["expected_messages"],
                                      group["expected_relationship"], index)
                     for index, rid in enumerate(ids)] if valid else []
            consistent = valid and all(view is not None for view in views) and views[0] == views[1]
        group_scores.append({
            "group_id": group["group_id"], "axis": group["axis"], "kind": group["kind"],
            "expected_relationship": group["expected_relationship"],
            "members": deepcopy(group["members"]), "coverage": deepcopy(group.get("coverage", {})),
            "record_ids": ids, "all_correct": all(correct), "consistent": consistent,
            "stable_wrong": consistent is True and not any(correct),
            "failed_record_ids": [rid for rid in ids if not scores[rid]["structured_correct"]],
        })
    rows = list(scores.values())
    strata = defaultdict(list)
    for row in rows:
        strata[(row["axis"], row["expected_relationship"], row["source_trace"], row["role"])].append(row)
    stratum_reports = [{
        "axis": axis, "expected_relationship": relation, "source_trace": trace,
        "role": role, **_summary(items, []),
    } for (axis, relation, trace, role), items in sorted(strata.items())]
    minimum = min(item["structured_accuracy_rate"] for item in stratum_reports)
    return {
        "overall": _summary(rows, group_scores),
        "by_axis": {axis: _summary([row for row in rows if row["axis"] == axis],
                                    [group for group in group_scores if group["axis"] == axis])
                    for axis in sorted({group["axis"] for group in groups})},
        "by_stratum": stratum_reports,
        "worst_strata": [item for item in stratum_reports if item["structured_accuracy_rate"] == minimum],
        "records": rows, "groups": group_scores,
    }


def score_robustness_diagnostic(*, manifest_path, manifest_sha256, predictions_path,
                              generation_manifest_path, tokenizer_manifest_path,
                              output_dir, expected_identity, repo_root=REPO_ROOT):
    """Validate complete retained-context generation and write isolated diagnostic reports."""

    from runtime.world_model_training import validated_tokenizer_manifest
    from scripts.evaluate_world_model_s1 import load_predictions, validate_publication_identity

    contract = load_contract(Path(repo_root) / "runtime/world_model/contract.yaml")
    panel, rows, answers = load_robustness_panel(manifest_path, manifest_sha256, repo_root=repo_root, contract=contract)
    questions = {row["record_id"]: row for row in rows}
    generation, predictions = load_predictions(
        Path(predictions_path), Path(generation_manifest_path), expected_record_ids=set(questions),
        expected_test_panel_id=panel["test_panel_id"],
    )
    validate_publication_identity(generation, expected_identity)
    if generation["generation_config"].get("history_policy") != "retain_current_turn":
        raise S1EvaluationError("robustness generation did not retain current-turn reasoning")
    tokenizer = validated_tokenizer_manifest(Path(tokenizer_manifest_path).resolve())
    if tokenizer["manifest_sha256"] != generation["tokenizer_manifest_sha256"]:
        raise S1EvaluationError("robustness tokenizer identity changed")
    result = score_robustness_records(questions, answers, predictions, panel["groups"], contract=contract)
    metrics = {
        "schema_version": "mentor-rl-s1-robustness-metrics-v1", "diagnostic_only": True,
        "kind": KIND, "source_split": "train", "test_panel_id": panel["test_panel_id"],
        "diagnostic_manifest_sha256": manifest_sha256,
        "generation_manifest_sha256": generation["manifest_sha256"], "identity": dict(expected_identity),
        "correctness_definition": list(CORRECTNESS_CHECKS),
        **{key: value for key, value in result.items() if key not in {"records", "groups"}},
    }
    metrics["metrics_sha256"] = stable_sha256(metrics)
    report = {
        "schema_version": "mentor-rl-s1-robustness-report-v1", "diagnostic_only": True,
        "metrics_sha256": metrics["metrics_sha256"], "records": result["records"], "groups": result["groups"],
    }
    output = Path(output_dir)
    write_json(output / "robustness_metrics.json", metrics)
    write_json(output / "robustness_report.json", report, private=True)
    return metrics


def main():
    """Use the launcher's pinned identities without a training W&B writer."""

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
    metrics = score_robustness_diagnostic(
        manifest_path=args.manifest, manifest_sha256=args.manifest_sha256, predictions_path=args.predictions,
        generation_manifest_path=args.generation_manifest, tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir, expected_identity={key: os.environ.get(env) for key, env in names.items()},
    )
    print(f"Separate robustness diagnostic complete: {metrics['overall']['row_count']} rows; no promotion gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
