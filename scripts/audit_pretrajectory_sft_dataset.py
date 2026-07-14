#!/usr/bin/env python3
"""Audit a pre-trajectory MENTOR-RL SFT dataset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_pretrajectory_sft_curriculum_plan import (  # noqa: E402
    curriculum_plan_hash,
    validate_curriculum_plan,
)


SPLITS = ("train", "val", "test")
MIXTURE_UNDERFILL_POLICIES = ("ignore", "warning", "fatal")
CURRICULUM_ARTIFACT_SCHEMA_VERSION = "pretrajectory-sft-curriculum-artifacts-v1"
CURRICULUM_DATASET_SCHEMA_VERSION = "pretrajectory-sft-v3"
CURRICULUM_AUDIT_SCHEMA_VERSION = "pretrajectory-sft-audit-v3"
CURRICULUM_REQUIRED_REPORTS = ("audit_report.json", "leakage_report.json", "coverage_report.json")
CURRICULUM_MODEL_TEXT_FIELDS = ("system", "question", "context", "answer")
CURRICULUM_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?"
)
CURRICULUM_RELATIVE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:data|config|scripts|runtime|checkpoints|logs)/[A-Za-z0-9_./-]+"
)
CURRICULUM_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\]+\\)+[^\s\\]+")


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str | None = None
    line: int | None = None
    record_id: str | None = None
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.line is not None:
            payload["line"] = self.line
        if self.record_id is not None:
            payload["record_id"] = self.record_id
        if self.context:
            payload["context"] = self.context
        return payload


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json_file(path: Path, add_issue: Callable[..., None]) -> dict[str, Any]:
    if not path.exists():
        add_issue("fatal", "missing_file", f"Missing required JSON file: {path}", path=path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        add_issue(
            "fatal",
            "invalid_json",
            f"Invalid JSON in {path}: {exc}",
            path=path,
            line=exc.lineno,
        )
        return {}
    if not isinstance(payload, dict):
        add_issue("fatal", "json_not_object", f"Expected JSON object in {path}", path=path)
        return {}
    return payload


def read_jsonl_file(path: Path, add_issue: Callable[..., None]) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    if not path.exists():
        add_issue("fatal", "missing_file", f"Missing required JSONL file: {path}", path=path)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                add_issue(
                    "fatal",
                    "invalid_jsonl",
                    f"Invalid JSONL row in {path}: {exc}",
                    path=path,
                    line=line_number,
                )
                continue
            if not isinstance(payload, dict):
                add_issue(
                    "fatal",
                    "jsonl_row_not_object",
                    f"Expected JSON object row in {path}",
                    path=path,
                    line=line_number,
                )
                continue
            rows.append((line_number, payload))
    return rows


def build_mixture_contract_report(
    counts_by_bucket: dict[str, int] | Counter[str],
    *,
    total_records: int,
    target_weights: dict[str, float],
    minimum_records: int,
    absolute_underfill_tolerance: float,
    relative_underfill_tolerance: float,
    underfill_policy: str,
) -> dict[str, Any]:
    """Build the mixture-underfill report required by the v3 curriculum audit."""

    if underfill_policy not in MIXTURE_UNDERFILL_POLICIES:
        raise ValueError(f"Unsupported mixture underfill policy: {underfill_policy}")
    if minimum_records < 0:
        raise ValueError("Mixture contract minimum_records must be nonnegative.")
    if not 0.0 <= absolute_underfill_tolerance <= 1.0:
        raise ValueError("Mixture absolute underfill tolerance must be between 0 and 1.")
    if not 0.0 <= relative_underfill_tolerance <= 1.0:
        raise ValueError("Mixture relative underfill tolerance must be between 0 and 1.")

    eligible = total_records >= minimum_records
    buckets: dict[str, dict[str, Any]] = {}
    material_underfilled_buckets: list[str] = []
    for bucket, target_share in target_weights.items():
        count = int(counts_by_bucket.get(bucket, 0))
        actual_share = count / total_records if total_records else 0.0
        delta_share = actual_share - target_share
        absolute_underfill = max(0.0, target_share - actual_share)
        relative_underfill = (
            max(0.0, (target_share - actual_share) / target_share)
            if target_share
            else 0.0
        )
        materially_underfilled = bool(
            eligible
            and absolute_underfill > absolute_underfill_tolerance
            and relative_underfill > relative_underfill_tolerance
        )
        if materially_underfilled:
            material_underfilled_buckets.append(bucket)
        buckets[bucket] = {
            "target_share": target_share,
            "target_record_count": total_records * target_share,
            "actual_share": actual_share,
            "actual_record_count": count,
            "delta_share": delta_share,
            "absolute_underfill": absolute_underfill,
            "relative_underfill": relative_underfill,
            "materially_underfilled": materially_underfilled,
        }

    return {
        "contract": {
            "target_weights": target_weights,
            "minimum_records": minimum_records,
            "absolute_underfill_tolerance": absolute_underfill_tolerance,
            "relative_underfill_tolerance": relative_underfill_tolerance,
            "underfill_policy": underfill_policy,
            "material_underfill_rule": (
                "record_count >= minimum_records AND "
                "absolute_underfill > absolute_underfill_tolerance AND "
                "relative_underfill > relative_underfill_tolerance"
            ),
        },
        "eligible": eligible,
        "total_records": total_records,
        "material_underfilled_bucket_count": len(material_underfilled_buckets),
        "material_underfilled_buckets": sorted(material_underfilled_buckets),
        "buckets": buckets,
    }


def has_negation_near(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32) : start]
    return bool(re.search(r"\b(no|not|never|without|avoid|unsupported)\b", prefix))


def unsupported_answer_claim(answer: str) -> str | None:
    text = answer.lower()
    if "definitely causally" in text:
        return "answer contains 'definitely causally'"
    if re.search(r"\bproves? causality\b", text):
        return "answer says the evidence proves causality"
    for match in re.finditer(r"\bconfirmed causal (gene|relationship|candidate)s?\b", text):
        if not has_negation_near(text, match.start()):
            return "answer makes an uncaveated confirmed-causal claim"
    phrase = "there is no biological relationship"
    pos = text.find(phrase)
    if pos >= 0 and not has_negation_near(text, pos) and "does not prove" not in text:
        return "answer treats graph absence as biological absence"
    return None


def _curriculum_estimate_tokens(text: str) -> int:
    """Match the deterministic tokenizer-free estimator used by the v3 compiler."""

    encoded_length = len(text.encode("utf-8"))
    byte_chunks = math.ceil(encoded_length / 4) if encoded_length else 0
    lexical = 0
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        token = match.group(0)
        if re.fullmatch(r"\w+", token, flags=re.UNICODE):
            lexical += max(1, math.ceil(len(token.encode("utf-8")) / 4))
        else:
            lexical += 1
    return max(byte_chunks, lexical)


def _curriculum_budget_measurement(record: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    prompt_text = "\n".join(str(record.get(field, "")) for field in ("system", "question", "context"))
    answer = str(record.get("answer", ""))
    prompt_tokens = _curriculum_estimate_tokens(prompt_text)
    answer_tokens = _curriculum_estimate_tokens(answer)
    total_tokens = prompt_tokens + answer_tokens
    answer_characters = len(answer)
    violations: list[str] = []
    if prompt_tokens > int(profile["max_prompt_tokens"]):
        violations.append("max_prompt_tokens")
    if answer_tokens > int(profile["max_answer_tokens"]):
        violations.append("max_answer_tokens")
    if total_tokens > int(profile["max_total_tokens"]):
        violations.append("max_total_tokens")
    if answer_characters > int(profile["max_answer_characters"]):
        violations.append("max_answer_characters")
    return {
        "prompt_token_estimate": prompt_tokens,
        "answer_token_estimate": answer_tokens,
        "total_token_estimate": total_tokens,
        "answer_character_count": answer_characters,
        "violations": violations,
        "passed": not violations,
    }


def _curriculum_generator_budget_measurement(record: dict[str, Any]) -> dict[str, Any]:
    answer = str(record.get("answer", ""))
    prompt_tokens = (
        _curriculum_estimate_tokens(str(record.get("system", "")))
        + _curriculum_estimate_tokens(str(record.get("question", "")))
        + 16
    )
    answer_tokens = _curriculum_estimate_tokens(answer)
    return {
        "prompt_token_estimate": prompt_tokens,
        "answer_token_estimate": answer_tokens,
        "total_token_estimate": prompt_tokens + answer_tokens,
        "answer_character_count": len(answer),
    }


def _curriculum_stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audit_curriculum_v3_dataset(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    issues: list[AuditIssue],
    add_issue: Callable[..., None],
    output_path: Path | None,
    mixture_tolerance: float | None,
    coverage_min_records: int,
    strict_coverage: bool,
    fail_on_warnings: bool,
    min_records: int,
    max_issues: int,
    training_max_sequence_tokens: int | None,
    eval_max_answer_tokens: int | None,
    max_answer_characters: int | None,
    mixture_contract_min_records: int | None,
    mixture_absolute_underfill_tolerance: float | None,
    mixture_relative_underfill_tolerance: float | None,
    mixture_underfill_policy: str | None,
) -> dict[str, Any]:
    """Audit the plan-driven v3 artifact and its native curriculum reports."""

    plan_path = dataset_dir / "curriculum_plan.json"
    canonical_path = dataset_dir / "canonical_objects.jsonl"
    plan = read_json_file(plan_path, add_issue)
    native_reports = {
        name: read_json_file(dataset_dir / name, add_issue) for name in CURRICULUM_REQUIRED_REPORTS
    }
    canonical_rows = read_jsonl_file(canonical_path, add_issue)

    if manifest.get("schema_version") != CURRICULUM_ARTIFACT_SCHEMA_VERSION:
        add_issue(
            "fatal",
            "artifact_schema_version_mismatch",
            "The v3 dataset manifest has an unsupported artifact-wrapper schema.",
            path=manifest_path,
            context={
                "expected": CURRICULUM_ARTIFACT_SCHEMA_VERSION,
                "actual": manifest.get("schema_version"),
            },
        )
    if plan.get("dataset_schema_version") != CURRICULUM_DATASET_SCHEMA_VERSION:
        add_issue(
            "fatal",
            "plan_dataset_schema_version_mismatch",
            "The embedded curriculum plan does not declare pretrajectory-sft-v3.",
            path=plan_path,
            context={"actual": plan.get("dataset_schema_version")},
        )
    plan_errors = validate_curriculum_plan(plan)
    for error in plan_errors:
        add_issue(
            "fatal",
            "invalid_curriculum_plan",
            error,
            path=plan_path,
        )
    observed_plan_hash = curriculum_plan_hash(plan) if not plan_errors else None
    declared_plan_hash = manifest.get("plan_hash")
    if observed_plan_hash is not None and declared_plan_hash != observed_plan_hash:
        add_issue(
            "fatal",
            "curriculum_plan_hash_mismatch",
            "Manifest plan_hash does not match curriculum_plan.json.",
            path=manifest_path,
            context={"manifest": declared_plan_hash, "actual": observed_plan_hash},
        )
    if plan.get("plan_id") != manifest.get("plan_id"):
        add_issue(
            "fatal",
            "curriculum_plan_id_mismatch",
            "Manifest plan_id does not match curriculum_plan.json.",
            path=manifest_path,
            context={"manifest": manifest.get("plan_id"), "plan": plan.get("plan_id")},
        )

    for report_name, native_report in native_reports.items():
        report_path = dataset_dir / report_name
        if native_report.get("schema_version") != CURRICULUM_ARTIFACT_SCHEMA_VERSION:
            add_issue(
                "fatal",
                "native_report_schema_mismatch",
                f"{report_name} has an unsupported artifact schema.",
                path=report_path,
                context={"actual": native_report.get("schema_version")},
            )
        if native_report.get("plan_id") != manifest.get("plan_id"):
            add_issue(
                "fatal",
                "native_report_plan_id_mismatch",
                f"{report_name} plan_id does not match the manifest.",
                path=report_path,
            )
        if native_report.get("plan_hash") != declared_plan_hash:
            add_issue(
                "fatal",
                "native_report_plan_hash_mismatch",
                f"{report_name} plan_hash does not match the manifest.",
                path=report_path,
            )
        if native_report.get("passed") is not True:
            add_issue(
                "fatal",
                "native_report_failed",
                f"{report_name} is not marked passed.",
                path=report_path,
            )

    native_audit = native_reports["audit_report.json"]
    for field in (
        "fatal_error_count",
        "budget_violation_count_in_selected",
        "raw_path_violation_count_in_selected",
        "metadata_violation_count_in_selected",
        "tool_schema_violation_count_in_selected",
    ):
        if native_audit.get(field) != 0:
            add_issue(
                "fatal",
                "native_selected_row_audit_failed",
                f"Native audit field {field} must be zero.",
                path=dataset_dir / "audit_report.json",
                context={"field": field, "actual": native_audit.get(field)},
            )
    if native_audit.get("leakage_passed") is not True:
        add_issue(
            "fatal",
            "native_audit_leakage_failed",
            "Native audit does not attest that leakage checks passed.",
            path=dataset_dir / "audit_report.json",
        )

    native_leakage = native_reports["leakage_report.json"]
    for field in (
        "oracle_fact_cross_split_count",
        "optional_group_cross_split_count",
        "exact_duplicate_cross_split_count",
        "near_duplicate_cross_split_count",
    ):
        if native_leakage.get(field) != 0:
            add_issue(
                "fatal",
                "native_leakage_count_nonzero",
                f"Native leakage field {field} must be zero.",
                path=dataset_dir / "leakage_report.json",
                context={"field": field, "actual": native_leakage.get(field)},
            )
    native_coverage = native_reports["coverage_report.json"]
    if int(native_coverage.get("underfilled_material_cross_cell_count", 0) or 0) != 0:
        add_issue(
            "fatal",
            "native_material_cross_cell_underfill",
            "Native coverage report contains underfilled material cross-cells.",
            path=dataset_dir / "coverage_report.json",
            context={
                "count": native_coverage.get("underfilled_material_cross_cell_count"),
            },
        )
    build_profiles = plan.get("build_profiles") if isinstance(plan.get("build_profiles"), dict) else {}
    build_profile = build_profiles.get(manifest.get("build_profile"))
    if isinstance(build_profile, dict):
        material_minimum = int(build_profile["minimum_selected_per_material_cross_cell"])
        cross_cells = native_coverage.get("cross_cells")
        if not isinstance(cross_cells, list):
            add_issue(
                "fatal",
                "native_coverage_missing_cross_cells",
                "Native coverage report does not expose required orthogonal cross-cells.",
                path=dataset_dir / "coverage_report.json",
            )
        else:
            independently_underfilled: list[dict[str, Any]] = []
            for cell in cross_cells:
                if not isinstance(cell, dict):
                    continue
                valid = cell.get("valid")
                if not isinstance(valid, int):
                    compacted = cell.get("compacted")
                    filtered = cell.get("filtered")
                    valid = (
                        compacted - filtered
                        if isinstance(compacted, int) and isinstance(filtered, int)
                        else 0
                    )
                selected = cell.get("selected")
                if valid >= material_minimum and (
                    not isinstance(selected, int) or selected < material_minimum
                ):
                    independently_underfilled.append(cell)
            if independently_underfilled:
                add_issue(
                    "fatal",
                    "material_cross_cell_underfill",
                    "Selected rows underfill material orthogonal cross-cells.",
                    path=dataset_dir / "coverage_report.json",
                    context={
                        "minimum_selected": material_minimum,
                        "underfilled_count": len(independently_underfilled),
                        "sample": independently_underfilled[:10],
                    },
                )

    records: list[dict[str, Any]] = []
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.jsonl"
        for line, record in read_jsonl_file(split_path, add_issue):
            records.append({"_file_split": split, "_path": split_path, "_line": line, "record": record})

    canonical_by_id: dict[str, dict[str, Any]] = {}
    for line, obj in canonical_rows:
        object_id = obj.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            add_issue(
                "fatal",
                "missing_canonical_object_id",
                "Canonical object is missing object_id.",
                path=canonical_path,
                line=line,
            )
            continue
        if object_id in canonical_by_id:
            add_issue(
                "fatal",
                "duplicate_canonical_object_id",
                "Duplicate canonical object id.",
                path=canonical_path,
                line=line,
                context={"object_id": object_id},
            )
        canonical_by_id[object_id] = obj

    if min_records and len(records) < min_records:
        add_issue(
            "fatal",
            "record_count_below_minimum",
            "Dataset contains fewer records than requested.",
            context={"actual": len(records), "minimum": min_records},
        )
    if manifest.get("selected_record_count") != len(records):
        add_issue(
            "fatal",
            "manifest_selected_record_count_mismatch",
            "Manifest selected_record_count does not match split JSONL rows.",
            path=manifest_path,
            context={"manifest": manifest.get("selected_record_count"), "actual": len(records)},
        )
    if manifest.get("canonical_object_count") != len(canonical_by_id):
        add_issue(
            "fatal",
            "manifest_canonical_object_count_mismatch",
            "Manifest canonical_object_count does not match canonical_objects.jsonl.",
            path=manifest_path,
            context={"manifest": manifest.get("canonical_object_count"), "actual": len(canonical_by_id)},
        )

    families = {
        str(item.get("name")): item
        for item in plan.get("question_families", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    stages = {
        str(item.get("name")): item
        for item in plan.get("stages", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    profiles = plan.get("context_budget_profiles") if isinstance(plan.get("context_budget_profiles"), dict) else {}
    required_metadata = set(
        plan.get("record_contract", {}).get("required_metadata_fields", [])
        if isinstance(plan.get("record_contract"), dict)
        else []
    )
    source_identities = manifest.get("source_identities") if isinstance(manifest.get("source_identities"), dict) else {}
    expected_multiplex_id = plan.get("graph_contract", {}).get("multiplex_id") if isinstance(plan.get("graph_contract"), dict) else None
    expected_store_id = source_identities.get("store_id")
    expected_flist_id = source_identities.get("flist_id")

    record_ids: set[str] = set()
    object_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    near_text_splits: dict[str, set[str]] = defaultdict(set)
    counts_by_split: Counter[str] = Counter()
    counts_by_family: Counter[str] = Counter()
    counts_by_bucket: Counter[str] = Counter()
    counts_by_stage: Counter[str] = Counter()
    counts_by_book_mode: Counter[str] = Counter()
    counts_by_source: Counter[str] = Counter()
    counts_by_budget_profile: Counter[str] = Counter()
    counts_by_answer_budget_action: Counter[str] = Counter()
    over_budget_record_count = 0
    missing_budget_count = 0
    max_answer_token_estimate = 0
    max_total_token_estimate = 0
    max_answer_character_count = 0
    optional_group_keys = list(plan.get("split_contract", {}).get("co_grouping_keys_when_present", [])) if isinstance(plan.get("split_contract"), dict) else []

    for item in records:
        record = item["record"]
        path = item["_path"]
        line = item["_line"]
        file_split = item["_file_split"]
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            add_issue("fatal", "missing_metadata", "Record is missing metadata.", path=path, line=line)
            continue
        record_id = metadata.get("record_id") if isinstance(metadata.get("record_id"), str) else None
        if not record_id:
            add_issue("fatal", "missing_record_id", "Record metadata is missing record_id.", path=path, line=line)
        elif record_id in record_ids:
            add_issue("fatal", "duplicate_record_id", "Duplicate record_id.", path=path, line=line, record_id=record_id)
        else:
            record_ids.add(record_id)
        for key in ("system", "question", "answer"):
            if not isinstance(record.get(key), str) or not str(record.get(key)).strip():
                add_issue("fatal", f"missing_{key}", f"Record is missing nonempty `{key}`.", path=path, line=line, record_id=record_id)
        missing_fields = sorted(field for field in required_metadata if field not in metadata)
        if missing_fields:
            add_issue(
                "fatal",
                "missing_required_v3_metadata",
                "Record is missing plan-required v3 metadata.",
                path=path,
                line=line,
                record_id=record_id,
                context={"missing_fields": missing_fields},
            )
        if metadata.get("schema_version") != CURRICULUM_DATASET_SCHEMA_VERSION:
            add_issue("fatal", "record_schema_version_mismatch", "Record metadata schema_version is not pretrajectory-sft-v3.", path=path, line=line, record_id=record_id)
        split = metadata.get("split")
        if split != file_split:
            add_issue(
                "fatal",
                "metadata_split_file_mismatch",
                "Record metadata split does not match its split file.",
                path=path,
                line=line,
                record_id=record_id,
                context={"metadata_split": split, "file_split": file_split},
            )
        family_name = metadata.get("question_family")
        family = families.get(str(family_name))
        stage_name = metadata.get("curriculum_stage")
        stage = stages.get(str(stage_name))
        bucket = metadata.get("mixture_bucket")
        book_mode = metadata.get("book_mode")
        profile_name = metadata.get("context_budget_profile")
        if family is None:
            add_issue("fatal", "unknown_question_family", "Record question_family is not in the curriculum plan.", path=path, line=line, record_id=record_id, context={"question_family": family_name})
        else:
            if family.get("primary_stage") != stage_name:
                add_issue("fatal", "family_stage_mismatch", "Record stage does not match its question-family contract.", path=path, line=line, record_id=record_id)
            if family.get("mixture_bucket") != bucket:
                add_issue("fatal", "family_bucket_mismatch", "Record mixture bucket does not match its question-family contract.", path=path, line=line, record_id=record_id)
            if book_mode not in family.get("allowed_book_modes", []):
                add_issue("fatal", "family_book_mode_mismatch", "Record book mode is not allowed for its question family.", path=path, line=line, record_id=record_id)
            if metadata.get("difficulty_source") != family.get("difficulty_source"):
                add_issue("fatal", "family_difficulty_source_mismatch", "Record difficulty source does not match its question-family contract.", path=path, line=line, record_id=record_id)
        if stage is None or stage_name == "stage6_blend":
            add_issue("fatal", "invalid_primary_curriculum_stage", "Record does not name a primary stage from 1 through 5.", path=path, line=line, record_id=record_id, context={"stage": stage_name})
        else:
            if book_mode not in stage.get("allowed_book_modes", []):
                add_issue("fatal", "stage_book_mode_mismatch", "Record book mode is not allowed in its primary stage.", path=path, line=line, record_id=record_id)
            if profile_name not in stage.get("allowed_budget_profiles", []):
                add_issue("fatal", "stage_budget_profile_mismatch", "Record budget profile is not allowed in its primary stage.", path=path, line=line, record_id=record_id)

        profile = profiles.get(profile_name) if isinstance(profile_name, str) else None
        if not isinstance(profile, dict):
            missing_budget_count += 1
            add_issue("fatal", "unknown_budget_profile", "Record context_budget_profile is not defined by the plan.", path=path, line=line, record_id=record_id, context={"profile": profile_name})
        else:
            measurement = _curriculum_budget_measurement(record, profile)
            max_answer_token_estimate = max(max_answer_token_estimate, int(measurement["answer_token_estimate"]))
            max_total_token_estimate = max(max_total_token_estimate, int(measurement["total_token_estimate"]))
            max_answer_character_count = max(max_answer_character_count, int(measurement["answer_character_count"]))
            stored_measurement = metadata.get("budget_measurement")
            if stored_measurement != measurement:
                add_issue(
                    "fatal",
                    "budget_measurement_mismatch",
                    "Stored profile budget_measurement does not match model-facing text.",
                    path=path,
                    line=line,
                    record_id=record_id,
                    context={"stored": stored_measurement, "actual": measurement},
                )
            generator_measurement = _curriculum_generator_budget_measurement(record)
            answer_budget = metadata.get("answer_budget")
            if not isinstance(answer_budget, dict):
                missing_budget_count += 1
                counts_by_answer_budget_action["missing"] += 1
                add_issue("fatal", "missing_record_answer_budget_metadata", "Record is missing generator answer_budget metadata.", path=path, line=line, record_id=record_id)
            else:
                counts_by_answer_budget_action["profile_valid"] += 1
                for field, expected in generator_measurement.items():
                    if answer_budget.get(field) != expected:
                        add_issue(
                            "fatal",
                            "answer_budget_measurement_mismatch",
                            "Stored generator answer_budget does not match model-facing text.",
                            path=path,
                            line=line,
                            record_id=record_id,
                            context={"field": field, "stored": answer_budget.get(field), "actual": expected},
                        )
                if answer_budget.get("profile") != profile_name or answer_budget.get("violations") != []:
                    add_issue("fatal", "invalid_answer_budget_metadata", "Generator answer_budget does not attest the selected profile without violations.", path=path, line=line, record_id=record_id)
            external_violations = list(measurement["violations"])
            if training_max_sequence_tokens is not None and measurement["total_token_estimate"] > training_max_sequence_tokens:
                external_violations.append("training_max_sequence_tokens")
            if eval_max_answer_tokens is not None and measurement["answer_token_estimate"] > eval_max_answer_tokens:
                external_violations.append("eval_max_answer_tokens")
            if max_answer_characters is not None and measurement["answer_character_count"] > max_answer_characters:
                external_violations.append("max_answer_characters")
            if external_violations:
                over_budget_record_count += 1
                add_issue(
                    "fatal",
                    "answer_budget_exceeded",
                    "Record exceeds its profile or supplied training/evaluation ceiling.",
                    path=path,
                    line=line,
                    record_id=record_id,
                    context={"profile": profile_name, "violations": sorted(set(external_violations))},
                )

        if metadata.get("answer_format") == "json":
            try:
                json.loads(str(record.get("answer", "")))
            except json.JSONDecodeError:
                add_issue("fatal", "invalid_json_answer", "answer_format=json requires a valid JSON answer.", path=path, line=line, record_id=record_id)
        model_text = "\n".join(str(record.get(field, "")) for field in CURRICULUM_MODEL_TEXT_FIELDS)
        if (
            "file://" in model_text
            or CURRICULUM_ABSOLUTE_PATH_RE.search(model_text)
            or CURRICULUM_RELATIVE_PATH_RE.search(model_text)
            or CURRICULUM_WINDOWS_PATH_RE.search(model_text)
        ):
            add_issue("fatal", "raw_path_in_model_text", "Model-facing text contains a raw filesystem path.", path=path, line=line, record_id=record_id)
        claim_issue = unsupported_answer_claim(str(record.get("answer", "")))
        if claim_issue:
            add_issue("fatal", "unsupported_causal_language_in_answer", claim_issue, path=path, line=line, record_id=record_id)

        object_id = metadata.get("canonical_object_id")
        oracle_fact_id = metadata.get("oracle_fact_id")
        canonical_object = canonical_by_id.get(object_id) if isinstance(object_id, str) else None
        if canonical_object is None:
            add_issue("fatal", "missing_canonical_object", "Record points to no object in canonical_objects.jsonl.", path=path, line=line, record_id=record_id, context={"canonical_object_id": object_id})
        else:
            if oracle_fact_id != object_id:
                add_issue("fatal", "oracle_fact_canonical_object_mismatch", "oracle_fact_id and canonical_object_id must identify the same pre-render fact.", path=path, line=line, record_id=record_id)
            if canonical_object.get("object_type") != family_name:
                add_issue("fatal", "canonical_object_type_mismatch", "Canonical object type does not match record question_family.", path=path, line=line, record_id=record_id)
            for field, expected in (
                ("multiplex_id", expected_multiplex_id),
                ("store_id", expected_store_id),
                ("flist_id", expected_flist_id),
            ):
                if expected is not None and canonical_object.get(field) != expected:
                    add_issue("fatal", f"canonical_{field}_mismatch", f"Canonical object {field} does not match the declared source identity.", path=path, line=line, record_id=record_id)
        for field, expected in (
            ("multiplex_id", expected_multiplex_id),
            ("store_id", expected_store_id),
            ("flist_id", expected_flist_id),
        ):
            if expected is not None and metadata.get(field) != expected:
                add_issue("fatal", f"record_{field}_mismatch", f"Record {field} does not match the declared source identity.", path=path, line=line, record_id=record_id)
        if (book_mode == "tool_call" or stage_name == "stage5_structured_tools") and metadata.get("tool_schema_validated") is not True:
            add_issue("fatal", "tool_schema_not_validated", "Tool-curriculum record lacks live-schema validation attestation.", path=path, line=line, record_id=record_id)

        split_string = str(split)
        if isinstance(object_id, str):
            object_splits[object_id].add(split_string)
        fact_group = metadata.get("oracle_fact_group_id")
        if isinstance(fact_group, str) and fact_group:
            group_splits[("oracle_fact_group_id", fact_group)].add(split_string)
        else:
            add_issue("fatal", "missing_oracle_fact_group_id", "Record is missing oracle_fact_group_id used for split isolation.", path=path, line=line, record_id=record_id)
        for group_key in optional_group_keys:
            group_value = metadata.get(group_key)
            if isinstance(group_value, str) and group_value:
                group_splits[(str(group_key), group_value)].add(split_string)
        text_fingerprint = _curriculum_stable_hash([str(record.get(field, "")) for field in CURRICULUM_MODEL_TEXT_FIELDS])
        normalized = re.sub(r"[^a-z0-9]+", " ", model_text.lower()).strip()
        text_splits[text_fingerprint].add(split_string)
        near_text_splits[_curriculum_stable_hash(normalized)].add(split_string)

        counts_by_split[split_string] += 1
        if isinstance(family_name, str):
            counts_by_family[family_name] += 1
        if isinstance(bucket, str):
            counts_by_bucket[bucket] += 1
        if isinstance(stage_name, str):
            counts_by_stage[stage_name] += 1
        if isinstance(book_mode, str):
            counts_by_book_mode[book_mode] += 1
        if isinstance(profile_name, str):
            counts_by_budget_profile[profile_name] += 1
        provenance = metadata.get("provenance")
        if isinstance(provenance, dict) and isinstance(provenance.get("source"), str):
            counts_by_source[provenance["source"]] += 1

    for object_id, splits in object_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "canonical_object_split_leakage", "Canonical oracle fact appears in multiple splits.", context={"canonical_object_id": object_id, "splits": sorted(splits)})
    for (group_key, group_value), splits in group_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "oracle_group_split_leakage", "Oracle grouping key appears in multiple splits.", context={"group_key": group_key, "group_value": group_value, "splits": sorted(splits)})
    for fingerprint, splits in text_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "exact_duplicate_cross_split", "Exact model-facing text appears in multiple splits.", context={"fingerprint": fingerprint, "splits": sorted(splits)})
    for fingerprint, splits in near_text_splits.items():
        if len(splits) > 1:
            add_issue("fatal", "near_duplicate_cross_split", "Normalized model-facing text appears in multiple splits.", context={"fingerprint": fingerprint, "splits": sorted(splits)})

    actual_counts = {
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
        "record_count_by_question_family": dict(sorted(counts_by_family.items())),
    }
    if isinstance(build_profile, dict):
        expected_split_counts = build_profile.get("split_counts")
        if expected_split_counts != actual_counts["record_count_by_split"]:
            add_issue(
                "fatal",
                "build_profile_split_count_mismatch",
                "Selected split counts do not match the manifest build profile.",
                path=manifest_path,
                context={"expected": expected_split_counts, "actual": actual_counts["record_count_by_split"]},
            )
        family_minimum = int(build_profile["minimum_selected_per_required_family"])
        underfilled_families = {
            name: int(counts_by_family.get(name, 0))
            for name in families
            if int(counts_by_family.get(name, 0)) < family_minimum
        }
        if underfilled_families:
            add_issue(
                "fatal",
                "required_question_family_underfill",
                "One or more plan-required question families are below the build-profile minimum.",
                context={"minimum": family_minimum, "families": underfilled_families},
            )
    for field, actual in actual_counts.items():
        if manifest.get(field) != actual:
            add_issue("fatal", f"manifest_{field}_mismatch", f"Manifest {field} does not match selected rows.", path=manifest_path, context={"manifest": manifest.get(field), "actual": actual})
    for report_name in ("audit_report.json",):
        native_report = native_reports[report_name]
        for field in ("selected_record_count", "record_count_by_split", "record_count_by_mixture_bucket", "question_family_counts"):
            expected = len(records) if field == "selected_record_count" else actual_counts.get("record_count_by_question_family" if field == "question_family_counts" else field)
            if native_report.get(field) != expected:
                add_issue("fatal", "native_audit_count_mismatch", f"Native audit {field} does not match selected rows.", path=dataset_dir / report_name, context={"field": field, "report": native_report.get(field), "actual": expected})
    if native_leakage.get("selected_record_count") != len(records):
        add_issue("fatal", "native_leakage_record_count_mismatch", "Native leakage report selected_record_count does not match selected rows.", path=dataset_dir / "leakage_report.json")

    sorted_records = [
        item["record"]
        for item in sorted(
            records,
            key=lambda entry: str(entry["record"].get("metadata", {}).get("record_id", "")),
        )
    ]
    observed_content_hash = _curriculum_stable_hash(sorted_records)
    if manifest.get("content_hash") != observed_content_hash:
        add_issue(
            "fatal",
            "selected_content_hash_mismatch",
            "Selected split records no longer match the manifest content_hash.",
            path=manifest_path,
            context={"manifest": manifest.get("content_hash"), "actual": observed_content_hash},
        )

    total = len(records)
    target_weights = {
        str(key): float(value)
        for key, value in (
            plan.get("mixture", {}).get("content_buckets", {})
            if isinstance(plan.get("mixture"), dict)
            else {}
        ).items()
    }
    effective_policy = mixture_underfill_policy or "fatal"
    effective_min_records = mixture_contract_min_records if mixture_contract_min_records is not None else 0
    effective_absolute_tolerance = mixture_absolute_underfill_tolerance if mixture_absolute_underfill_tolerance is not None else 0.0
    effective_relative_tolerance = mixture_relative_underfill_tolerance if mixture_relative_underfill_tolerance is not None else 0.0
    mixture_report = build_mixture_contract_report(
        counts_by_bucket,
        total_records=total,
        target_weights=target_weights,
        minimum_records=effective_min_records,
        absolute_underfill_tolerance=effective_absolute_tolerance,
        relative_underfill_tolerance=effective_relative_tolerance,
        underfill_policy=effective_policy,
    )
    mixture_report["manifest_contract_present"] = True
    for bucket in mixture_report["material_underfilled_buckets"]:
        if effective_policy == "ignore":
            continue
        add_issue(
            "fatal" if effective_policy == "fatal" else "warning",
            "mixture_bucket_materially_underfilled",
            "V3 mixture bucket is materially below its plan target.",
            context={"bucket": bucket, **mixture_report["buckets"][bucket]},
        )
    if mixture_tolerance is not None and total >= coverage_min_records:
        for bucket, bucket_report in mixture_report["buckets"].items():
            if abs(float(bucket_report["delta_share"])) > mixture_tolerance:
                add_issue("warning", "mixture_bucket_outside_tolerance", "V3 mixture bucket share is outside the requested symmetric tolerance.", context={"bucket": bucket, "delta_share": bucket_report["delta_share"], "tolerance": mixture_tolerance})

    missing_families = sorted(set(families) - set(counts_by_family))
    missing_buckets = sorted(set(target_weights) - set(counts_by_bucket))
    if total >= coverage_min_records and missing_families:
        add_issue("fatal" if strict_coverage else "warning", "missing_recommended_view_types", "Dataset is missing plan-required question families.", context={"missing_question_families": missing_families})
    if total >= coverage_min_records and missing_buckets:
        add_issue("fatal" if strict_coverage else "warning", "missing_mixture_buckets", "Dataset is missing plan-defined mixture buckets.", context={"missing_buckets": missing_buckets})

    issue_dicts = [issue.to_dict() for issue in issues]
    fatal_count = sum(issue.severity == "fatal" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    report = {
        "schema_version": CURRICULUM_AUDIT_SCHEMA_VERSION,
        "dataset_schema_version": CURRICULUM_DATASET_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "dataset_dir": str(dataset_dir),
        "passed": fatal_count == 0 and (warning_count == 0 or not fail_on_warnings),
        "fatal_error_count": fatal_count,
        "warning_count": warning_count,
        "record_count": total,
        "canonical_object_count": len(canonical_by_id),
        "plan_id": manifest.get("plan_id"),
        "plan_hash": declared_plan_hash,
        "content_hash": observed_content_hash,
        "native_reports": {
            name: {
                "passed": payload.get("passed"),
                "plan_hash": payload.get("plan_hash"),
            }
            for name, payload in native_reports.items()
        },
        "record_count_by_split": dict(sorted(counts_by_split.items())),
        "record_count_by_view_type": dict(sorted(counts_by_family.items())),
        "record_count_by_question_family": dict(sorted(counts_by_family.items())),
        "record_count_by_mixture_bucket": dict(sorted(counts_by_bucket.items())),
        "record_count_by_curriculum_stage": dict(sorted(counts_by_stage.items())),
        "record_count_by_context_mode": dict(sorted(counts_by_book_mode.items())),
        "record_count_by_book_mode": dict(sorted(counts_by_book_mode.items())),
        "record_count_by_budget_profile": dict(sorted(counts_by_budget_profile.items())),
        "record_count_by_answer_budget_action": dict(sorted(counts_by_answer_budget_action.items())),
        "record_count_by_source": dict(sorted(counts_by_source.items())),
        "answer_budget_contract": {
            "context_budget_profiles": profiles,
            "training_max_sequence_tokens": training_max_sequence_tokens,
            "eval_max_answer_tokens": eval_max_answer_tokens,
            "max_answer_characters": max_answer_characters,
        },
        "answer_budget_report": {
            "manifest_contract_present": True,
            "record_count_checked": total,
            "over_budget_record_count": over_budget_record_count,
            "missing_record_budget_metadata_count": missing_budget_count,
            "max_answer_token_estimate": max_answer_token_estimate,
            "max_training_sequence_token_estimate": max_total_token_estimate,
            "max_answer_character_count": max_answer_character_count,
            "record_count_by_action": dict(sorted(counts_by_answer_budget_action.items())),
        },
        "mixture_contract": mixture_report["contract"],
        "mixture_report": mixture_report,
        "missing_recommended_view_types": missing_families,
        "issues": issue_dicts[:max_issues],
        "truncated_issue_count": max(0, len(issue_dicts) - max_issues),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def audit_pretrajectory_sft_dataset(
    dataset_dir: Path,
    *,
    output_path: Path | None = None,
    mixture_tolerance: float | None = None,
    coverage_min_records: int = 1000,
    strict_coverage: bool = False,
    fail_on_warnings: bool = False,
    min_records: int = 0,
    max_issues: int = 200,
    training_max_sequence_tokens: int | None = None,
    eval_max_answer_tokens: int | None = None,
    max_answer_characters: int | None = None,
    mixture_contract_min_records: int | None = None,
    mixture_absolute_underfill_tolerance: float | None = None,
    mixture_relative_underfill_tolerance: float | None = None,
    mixture_underfill_policy: str | None = None,
) -> dict[str, Any]:
    """Audit a plan-driven v3 dataset produced by the active v5 curriculum."""

    issues: list[AuditIssue] = []

    def add_issue(
        severity: str,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        line: int | None = None,
        record_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        issues.append(
            AuditIssue(
                severity=severity,
                code=code,
                message=message,
                path=str(path) if path is not None else None,
                line=line,
                record_id=record_id,
                context=context,
            )
        )

    manifest_path = dataset_dir / "manifest.json"
    manifest = read_json_file(manifest_path, add_issue)
    if manifest.get("dataset_schema_version") != CURRICULUM_DATASET_SCHEMA_VERSION:
        add_issue(
            "fatal",
            "unsupported_dataset_schema_version",
            "Dataset must use the active plan-driven curriculum schema.",
            path=manifest_path,
            context={
                "expected": CURRICULUM_DATASET_SCHEMA_VERSION,
                "actual": manifest.get("dataset_schema_version"),
            },
        )

    return _audit_curriculum_v3_dataset(
        dataset_dir,
        manifest=manifest,
        manifest_path=manifest_path,
        issues=issues,
        add_issue=add_issue,
        output_path=output_path,
        mixture_tolerance=mixture_tolerance,
        coverage_min_records=coverage_min_records,
        strict_coverage=strict_coverage,
        fail_on_warnings=fail_on_warnings,
        min_records=min_records,
        max_issues=max_issues,
        training_max_sequence_tokens=training_max_sequence_tokens,
        eval_max_answer_tokens=eval_max_answer_tokens,
        max_answer_characters=max_answer_characters,
        mixture_contract_min_records=mixture_contract_min_records,
        mixture_absolute_underfill_tolerance=mixture_absolute_underfill_tolerance,
        mixture_relative_underfill_tolerance=mixture_relative_underfill_tolerance,
        mixture_underfill_policy=mixture_underfill_policy,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a pre-trajectory MENTOR-RL SFT dataset directory.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Audit report path. Defaults to DATASET_DIR/audit_report_contract_v3.json.",
    )
    parser.add_argument(
        "--mixture-tolerance",
        type=float,
        default=None,
        help="Optional symmetric share tolerance; the underfill contract is enforced separately.",
    )
    parser.add_argument("--mixture-contract-min-records", type=int, default=None)
    parser.add_argument("--mixture-absolute-underfill-tolerance", type=float, default=None)
    parser.add_argument("--mixture-relative-underfill-tolerance", type=float, default=None)
    parser.add_argument("--mixture-underfill-policy", choices=MIXTURE_UNDERFILL_POLICIES, default=None)
    parser.add_argument("--training-max-sequence-tokens", type=int, default=None)
    parser.add_argument("--eval-max-answer-tokens", type=int, default=None)
    parser.add_argument("--max-answer-characters", type=int, default=None)
    parser.add_argument("--coverage-min-records", type=int, default=1000)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument("--min-records", type=int, default=0)
    parser.add_argument("--max-issues", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="Print the full report instead of a compact summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out or args.dataset_dir / "audit_report_contract_v3.json"
    report = audit_pretrajectory_sft_dataset(
        args.dataset_dir,
        output_path=output_path,
        mixture_tolerance=args.mixture_tolerance,
        coverage_min_records=args.coverage_min_records,
        strict_coverage=args.strict_coverage,
        fail_on_warnings=args.fail_on_warnings,
        min_records=args.min_records,
        max_issues=args.max_issues,
        training_max_sequence_tokens=args.training_max_sequence_tokens,
        eval_max_answer_tokens=args.eval_max_answer_tokens,
        max_answer_characters=args.max_answer_characters,
        mixture_contract_min_records=args.mixture_contract_min_records,
        mixture_absolute_underfill_tolerance=args.mixture_absolute_underfill_tolerance,
        mixture_relative_underfill_tolerance=args.mixture_relative_underfill_tolerance,
        mixture_underfill_policy=args.mixture_underfill_policy,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = {
            "audit_report": str(output_path),
            "passed": report["passed"],
            "fatal_error_count": report["fatal_error_count"],
            "warning_count": report["warning_count"],
            "record_count": report["record_count"],
            "record_count_by_split": report["record_count_by_split"],
            "record_count_by_mixture_bucket": report["record_count_by_mixture_bucket"],
            "record_count_by_curriculum_stage": report["record_count_by_curriculum_stage"],
            "record_count_by_context_mode": report["record_count_by_context_mode"],
            "record_count_by_answer_budget_action": report["record_count_by_answer_budget_action"],
            "over_budget_record_count": report["answer_budget_report"]["over_budget_record_count"],
            "material_underfilled_mixture_buckets": report["mixture_report"]["material_underfilled_buckets"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    if report["fatal_error_count"] or (args.fail_on_warnings and report["warning_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
