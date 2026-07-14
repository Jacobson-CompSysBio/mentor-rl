#!/usr/bin/env python3
"""Run contract-checked exact evaluation for a base model or PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_pretrajectory_sft_predictions import (  # noqa: E402
    evaluate_gold_self_contract,
    evaluate_prediction_rows,
    evaluate_row,
    load_canonical_objects,
    render_html_report,
)
from utils.utils import build_formatting_func  # noqa: E402


PIPELINE_CONTRACT_VERSION = "pretrajectory-exact-eval-v2"
CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION = "pretrajectory-sft-audit-v3"
CURRICULUM_DATASET_SCHEMA_VERSION = "pretrajectory-sft-v3"
REQUIRED_DATASET_AUDIT_SCHEMA_VERSION = CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION
REQUIRED_EVALUATOR_CONTRACT_VERSION = "pretrajectory-sft-exact-v3"
OFFICIAL_READINESS_REGIME = "official_holdout"
SEEN_ROW_RECALL_REGIME = "seen_row_recall"
SEEN_FACT_RETENTION_REGIME = "seen_fact_heldout_rendering"
EVALUATION_REGIMES = (
    OFFICIAL_READINESS_REGIME,
    "unseen_fact_generalization",
    SEEN_ROW_RECALL_REGIME,
    SEEN_FACT_RETENTION_REGIME,
)
DIAGNOSTIC_REGIMES = frozenset(EVALUATION_REGIMES) - {OFFICIAL_READINESS_REGIME}
RETENTION_SUITE_SCHEMA_VERSION = "pretrajectory-sft-retention-suite-v1"
RETENTION_REPORT_SCHEMA_VERSION = "pretrajectory-sft-retention-report-v1"
DEFAULT_RETENTION_FAMILIES = (
    "entity_symbol_to_ensembl",
    "entity_ensembl_to_symbol",
    "ambiguous_alias_resolution",
)
CURRICULUM_NATIVE_REPORT_NAMES = (
    "audit_report.json",
    "leakage_report.json",
    "coverage_report.json",
)


class ExactEvalContractError(RuntimeError):
    """Raised after invalid pipeline artifacts and a run summary are written."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object row in {path}.")
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    metadata: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if resolved.is_file():
        metadata.update(
            {
                "size_bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        )
    return metadata


def _known_directory_files(path: Path | None, names: tuple[str, ...]) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "files": {
            name: _artifact_metadata(resolved / name)
            for name in names
            if (resolved / name).is_file()
        },
    }


def _find_dataset_artifact(dataset_path: Path, filename: str) -> Path | None:
    for directory in (dataset_path.resolve().parent, *dataset_path.resolve().parents):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _find_dataset_audit(dataset_path: Path) -> Path | None:
    """Locate the required v3 bridge audit without falling back to legacy audits."""

    return _find_dataset_artifact(dataset_path, "audit_report_contract_v3.json")


def _selected_rows_sha256(selected: list[tuple[int, dict[str, Any]]]) -> str:
    selection = [
        {
            "source_index": source_index,
            "record_id": row.get("metadata", {}).get("record_id"),
        }
        for source_index, row in selected
    ]
    encoded = json.dumps(selection, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _curriculum_dataset_identity(dataset_path: Path, audit: dict[str, Any]) -> dict[str, Any]:
    """Recompute the v3 selected-row identity used by the bridge audit."""

    failures: list[str] = []
    manifest_path = _find_dataset_artifact(dataset_path, "manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path is None:
        failures.append("missing_dataset_manifest")
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"unreadable_dataset_manifest:{type(error).__name__}")
        else:
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                failures.append("dataset_manifest_not_object")

    if manifest and manifest.get("dataset_schema_version") != CURRICULUM_DATASET_SCHEMA_VERSION:
        failures.append("dataset_manifest_schema_mismatch")

    dataset_root = manifest_path.parent if manifest_path is not None else dataset_path.resolve().parent
    plan_path = dataset_root / "curriculum_plan.json"
    observed_plan_hash: str | None = None
    if not plan_path.is_file():
        failures.append("missing_curriculum_plan_for_staleness")
    else:
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"unreadable_curriculum_plan_for_staleness:{type(error).__name__}")
        else:
            if isinstance(plan, dict):
                observed_plan_hash = _stable_json_hash(plan)
            else:
                failures.append("curriculum_plan_not_object")

    audit_plan_hash = audit.get("plan_hash")
    audit_content_hash = audit.get("content_hash")
    if not isinstance(audit_plan_hash, str) or not audit_plan_hash:
        failures.append("missing_audit_plan_hash")
    if not isinstance(audit_content_hash, str) or not audit_content_hash:
        failures.append("missing_audit_content_hash")
    if observed_plan_hash is not None and observed_plan_hash != audit_plan_hash:
        failures.append("curriculum_plan_hash_mismatch")
    if manifest:
        if manifest.get("plan_hash") != audit_plan_hash:
            failures.append("dataset_manifest_plan_hash_mismatch")
        if manifest.get("content_hash") != audit_content_hash:
            failures.append("dataset_manifest_content_hash_mismatch")

    observed_content_hash: str | None = None
    records: list[dict[str, Any]] = []
    split_files_present = True
    for split in ("train", "val", "test"):
        split_path = dataset_root / f"{split}.jsonl"
        if not split_path.is_file():
            failures.append(f"missing_dataset_split_for_staleness:{split}")
            split_files_present = False
            continue
        try:
            records.extend(read_jsonl(split_path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"unreadable_dataset_split_for_staleness:{split}:{type(error).__name__}")
            split_files_present = False
    if split_files_present:
        records.sort(
            key=lambda row: str(
                row.get("metadata", {}).get("record_id", "")
                if isinstance(row.get("metadata"), dict)
                else ""
            )
        )
        observed_content_hash = _stable_json_hash(records)
        if observed_content_hash != audit_content_hash:
            failures.append("dataset_content_hash_mismatch")

    return {
        "valid": not failures,
        "dataset_schema_version": manifest.get("dataset_schema_version"),
        "plan_hash": observed_plan_hash,
        "manifest_content_hash": manifest.get("content_hash"),
        "observed_content_hash": observed_content_hash,
        "audit_content_hash": audit_content_hash,
        "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else None,
        "curriculum_plan_path": str(plan_path.resolve()),
        "failures": failures,
    }


def _prediction_row(
    *,
    source_index: int,
    sample_index: int,
    example: dict[str, Any],
    prediction: str,
) -> dict[str, Any]:
    return {
        "idx": int(source_index),
        "sample_idx": int(sample_index),
        "system": example.get("system", ""),
        "question": example["question"],
        "answer": example["answer"],
        "prediction": prediction,
        "metadata": example.get("metadata", {}),
    }


def _model_report_contract(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    summary = report.get("summary")
    failures: list[str] = []
    if report.get("sample_count") != expected_count:
        failures.append("sample_count_mismatch")
    evaluator_contract = report.get("evaluator_contract")
    if not isinstance(evaluator_contract, dict):
        failures.append("missing_evaluator_contract")
    elif evaluator_contract.get("version") != REQUIRED_EVALUATOR_CONTRACT_VERSION:
        failures.append("unsupported_evaluator_contract_version")
    gold_self_evaluation = report.get("gold_self_evaluation")
    if not isinstance(gold_self_evaluation, dict) or gold_self_evaluation.get("passed") is not True:
        failures.append("gold_self_evaluation_not_passed")
    if not isinstance(summary, dict):
        failures.append("missing_summary")
    else:
        for metric in ("exact_graph_fact_pass_rate", "unsupported_language_rate"):
            value = summary.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                failures.append(f"missing_metric:{metric}")
        exact_only = summary.get("exact_only")
        if not isinstance(exact_only, dict):
            failures.append("missing_exact_only_summary")
        else:
            for metric in ("mean_id_recall", "mean_layer_recall", "mean_number_recall"):
                value = exact_only.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    failures.append(f"missing_exact_only_metric:{metric}")
    return {
        "valid": not failures,
        "failures": failures,
        "message": (
            "Model report has a valid evaluator contract and the fields required by the readiness gate."
            if not failures
            else "Model report is incomplete or its embedded gold-self contract is invalid."
        ),
    }


def _dataset_audit_contract(
    path: Path,
    *,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    payload: dict[str, Any] = {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"unreadable_dataset_audit:{type(error).__name__}")
    else:
        if isinstance(loaded, dict):
            payload = loaded
        else:
            failures.append("dataset_audit_not_object")

    schema_version = payload.get("schema_version")
    dataset_schema_version = payload.get("dataset_schema_version")
    if schema_version != REQUIRED_DATASET_AUDIT_SCHEMA_VERSION:
        failures.append("unsupported_dataset_audit_schema_version")
    if dataset_schema_version != CURRICULUM_DATASET_SCHEMA_VERSION:
        failures.append("unsupported_dataset_schema_version")

    fatal_error_count = payload.get("fatal_error_count")
    if isinstance(fatal_error_count, bool) or not isinstance(fatal_error_count, int):
        failures.append("missing_or_invalid_fatal_error_count")
        fatal_error_count = None
    elif fatal_error_count < 0:
        failures.append("negative_fatal_error_count")
    elif fatal_error_count != 0:
        failures.append("nonzero_fatal_error_count")

    warning_count = payload.get("warning_count")
    if isinstance(warning_count, bool) or not isinstance(warning_count, int):
        warning_count = None

    audit_passed = payload.get("passed")
    if audit_passed is not True:
        failures.append("dataset_audit_not_passed")

    answer_budget_report = payload.get("answer_budget_report")
    manifest_contract_present: bool | None = None
    over_budget_record_count: int | None = None
    missing_record_budget_metadata_count: int | None = None
    if not isinstance(answer_budget_report, dict):
        failures.append("missing_answer_budget_report")
    else:
        manifest_contract_present = answer_budget_report.get("manifest_contract_present")
        if manifest_contract_present is not True:
            failures.append("answer_budget_manifest_contract_not_present")
        over_budget_record_count = answer_budget_report.get("over_budget_record_count")
        if isinstance(over_budget_record_count, bool) or not isinstance(over_budget_record_count, int):
            failures.append("missing_or_invalid_over_budget_record_count")
            over_budget_record_count = None
        elif over_budget_record_count < 0:
            failures.append("negative_over_budget_record_count")
        elif over_budget_record_count != 0:
            failures.append("nonzero_over_budget_record_count")

        missing_record_budget_metadata_count = answer_budget_report.get(
            "missing_record_budget_metadata_count"
        )
        if (
            isinstance(missing_record_budget_metadata_count, bool)
            or not isinstance(missing_record_budget_metadata_count, int)
        ):
            failures.append("missing_or_invalid_record_budget_metadata_count")
            missing_record_budget_metadata_count = None
        elif missing_record_budget_metadata_count < 0:
            failures.append("negative_record_budget_metadata_count")
        elif missing_record_budget_metadata_count != 0:
            failures.append("nonzero_record_budget_metadata_count")

    native_reports = payload.get("native_reports")
    native_report_failures: list[str] = []
    plan_hash = payload.get("plan_hash")
    if not isinstance(native_reports, dict):
        native_report_failures.append("missing_native_reports")
    else:
        for report_name in CURRICULUM_NATIVE_REPORT_NAMES:
            report = native_reports.get(report_name)
            if not isinstance(report, dict):
                native_report_failures.append(f"missing_native_report:{report_name}")
                continue
            if report.get("passed") is not True:
                native_report_failures.append(f"native_report_not_passed:{report_name}")
            if report.get("plan_hash") != plan_hash:
                native_report_failures.append(f"native_report_plan_hash_mismatch:{report_name}")
    failures.extend(native_report_failures)
    native_reports_valid = not native_report_failures

    identity_dataset_path = dataset_path or path.parent / "val.jsonl"
    dataset_identity = _curriculum_dataset_identity(identity_dataset_path, payload)
    failures.extend(dataset_identity["failures"])

    return {
        "valid": not failures,
        "schema_version": schema_version,
        "dataset_schema_version": dataset_schema_version,
        "required_schema_version": REQUIRED_DATASET_AUDIT_SCHEMA_VERSION,
        "supported_schema_versions": [REQUIRED_DATASET_AUDIT_SCHEMA_VERSION],
        "fatal_error_count": fatal_error_count,
        "warning_count": warning_count,
        "audit_passed": audit_passed,
        "answer_budget_manifest_contract_present": manifest_contract_present,
        "over_budget_record_count": over_budget_record_count,
        "missing_record_budget_metadata_count": missing_record_budget_metadata_count,
        "native_reports_valid": native_reports_valid,
        "plan_hash": payload.get("plan_hash"),
        "content_hash": payload.get("content_hash"),
        "dataset_identity": dataset_identity,
        "failures": failures,
        "message": (
            "Dataset audit satisfies the schema, fatal-error, budget, leakage, and freshness contracts."
            if not failures
            else "Dataset audit does not satisfy the current dataset contract."
        ),
    }


def _invalid_readiness(
    *,
    decision: str,
    reason: str,
    gold_report_path: Path,
    gate_name: str = "evaluation_contract_valid",
    dataset_audit_path: Path | None = None,
) -> dict[str, Any]:
    payload = {
        "valid": False,
        "passed": False,
        "decision": decision,
        "required_failure_count": 1,
        "advisory_failure_count": 0,
        "failed_required_gates": [
            {
                "name": gate_name,
                "observed": False,
                "threshold": True,
                "op": "==",
                "required": True,
                "passed": False,
                "reason": reason,
            }
        ],
        "failed_advisory_gates": [],
        "gates": [],
        "gold_reference_report": str(gold_report_path),
    }
    if dataset_audit_path is not None:
        payload["dataset_audit"] = str(dataset_audit_path)
    return payload


def _run_readiness_checker(*, dataset_audit: Path, exact_report: Path, output_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_pretrajectory_sft_readiness.py"),
        "--dataset-audit",
        str(dataset_audit),
        "--exact-report",
        str(exact_report),
        "--out",
        str(output_path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    # Exit 1 covers both valid-but-not-ready and invalid-contract reports.  The
    # caller must preserve and inspect the report's `valid` field.
    if result.returncode not in {0, 1} or not output_path.is_file():
        raise ExactEvalContractError(
            "Readiness checker failed to produce an artifact: "
            f"returncode={result.returncode}, stderr={result.stderr.strip()}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExactEvalContractError("Readiness checker output is not a JSON object.")
    return payload


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _row_question_family(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    family = metadata.get("question_family") if isinstance(metadata, dict) else None
    return family if isinstance(family, str) and family else None


def _row_record_id(row: dict[str, Any], *, fallback: int) -> str:
    metadata = row.get("metadata")
    record_id = metadata.get("record_id") if isinstance(metadata, dict) else None
    return record_id if isinstance(record_id, str) and record_id else f"row-{fallback}"


def _parse_question_families(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else value
    families = tuple(str(item).strip() for item in items if str(item).strip())
    if len(set(families)) != len(families):
        raise ExactEvalContractError("question_families contains duplicates.")
    return families


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int | None,
    seed: int,
    question_families: tuple[str, ...] = (),
    samples_per_family: int = 0,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if samples_per_family < 0:
        raise ExactEvalContractError("samples_per_family cannot be negative.")
    if samples_per_family > 0:
        if sample_size is not None and sample_size > 0:
            raise ExactEvalContractError(
                "Use either sample_size or samples_per_family, not both."
            )
        if not question_families:
            raise ExactEvalContractError(
                "question_families is required when samples_per_family is positive."
            )
        by_family: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for source_index, row in indexed:
            family = _row_question_family(row)
            if family in question_families:
                by_family[str(family)].append((source_index, row))
        underfilled = {
            family: len(by_family.get(family, []))
            for family in question_families
            if len(by_family.get(family, [])) < samples_per_family
        }
        if underfilled:
            raise ExactEvalContractError(
                f"Stratified selection is underfilled for samples_per_family="
                f"{samples_per_family}: {underfilled}."
            )
        selected: list[tuple[int, dict[str, Any]]] = []
        for family in question_families:
            ranked = sorted(
                by_family[family],
                key=lambda item: hashlib.sha256(
                    f"{seed}:{family}:{_row_record_id(item[1], fallback=item[0])}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
            )
            selected.extend(ranked[:samples_per_family])
        return sorted(
            selected,
            key=lambda item: hashlib.sha256(
                f"{seed}:generation-order:{_row_record_id(item[1], fallback=item[0])}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )

    if question_families:
        indexed = [item for item in indexed if _row_question_family(item[1]) in question_families]
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(indexed), generator=generator).tolist()
    if sample_size is not None and sample_size > 0:
        order = order[: min(sample_size, len(order))]
    return [indexed[index] for index in order]


def _retention_suite_contract(
    *,
    manifest_path: Path | None,
    dataset_path: Path,
    rows: list[dict[str, Any]],
    source_dataset_root: Path | None,
) -> dict[str, Any]:
    failures: list[str] = []
    manifest: dict[str, Any] = {}
    if manifest_path is None or not manifest_path.is_file():
        failures.append("missing_retention_suite_manifest")
    else:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"unreadable_retention_suite_manifest:{type(error).__name__}")
        else:
            if isinstance(payload, dict):
                manifest = payload
            else:
                failures.append("retention_suite_manifest_not_object")

    if manifest:
        if manifest.get("schema_version") != RETENTION_SUITE_SCHEMA_VERSION:
            failures.append("retention_suite_schema_mismatch")
        if manifest.get("evaluation_regime") != SEEN_FACT_RETENTION_REGIME:
            failures.append("retention_suite_regime_mismatch")
        if manifest.get("passed") is not True:
            failures.append("retention_suite_audit_not_passed")
        if manifest.get("official_readiness_eligible") is not False:
            failures.append("retention_suite_must_be_diagnostic_only")

        suite = manifest.get("suite")
        if not isinstance(suite, dict):
            failures.append("missing_retention_suite_identity")
        else:
            if suite.get("sha256") != _sha256_file(dataset_path):
                failures.append("retention_suite_sha256_mismatch")
            if suite.get("row_count") != len(rows):
                failures.append("retention_suite_row_count_mismatch")
            manifest_counts = suite.get("row_count_by_family")
            observed_counts = Counter(
                family for row in rows if (family := _row_question_family(row)) is not None
            )
            if manifest_counts != dict(sorted(observed_counts.items())):
                failures.append("retention_suite_family_counts_mismatch")

        audit = manifest.get("audit")
        if not isinstance(audit, dict):
            failures.append("missing_retention_suite_audit")
        else:
            for field in (
                "duplicate_record_id_count",
                "duplicate_normalized_question_count",
                "source_prompt_overlap_count",
                "hidden_target_leak_count",
                "answer_changed_count",
                "oracle_fact_changed_count",
            ):
                if audit.get(field) != 0:
                    failures.append(f"retention_suite_nonzero_audit_count:{field}")

        if source_dataset_root is not None:
            source_manifest_path = source_dataset_root / "manifest.json"
            try:
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                failures.append(f"unreadable_source_dataset_manifest:{type(error).__name__}")
            else:
                source_identity = manifest.get("source_dataset")
                if not isinstance(source_identity, dict):
                    failures.append("missing_retention_source_dataset_identity")
                elif not isinstance(source_manifest, dict):
                    failures.append("source_dataset_manifest_not_object")
                else:
                    for key in ("dataset_schema_version", "plan_hash", "content_hash"):
                        if source_identity.get(key) != source_manifest.get(key):
                            failures.append(f"retention_source_dataset_{key}_mismatch")
                    source_stage_path = (
                        source_dataset_root
                        / "curriculum"
                        / "stage1_entity_schema"
                        / "train.jsonl"
                    )
                    if (
                        not source_stage_path.is_file()
                        or source_identity.get("sha256") != _sha256_file(source_stage_path)
                    ):
                        failures.append("retention_source_stage_sha256_mismatch")

    for row_index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            failures.append(f"retention_row_missing_metadata:{row_index}")
            continue
        if metadata.get("evaluation_regime") != SEEN_FACT_RETENTION_REGIME:
            failures.append(f"retention_row_regime_mismatch:{row_index}")
        if metadata.get("source_split") != "train":
            failures.append(f"retention_row_source_split_mismatch:{row_index}")
        if not isinstance(metadata.get("source_record_id"), str):
            failures.append(f"retention_row_missing_source_record_id:{row_index}")
        if not isinstance(metadata.get("retention_template_id"), str):
            failures.append(f"retention_row_missing_template_id:{row_index}")
        if metadata.get("official_readiness_eligible") is not False:
            failures.append(f"retention_row_readiness_eligible:{row_index}")

    return {
        "valid": not failures,
        "schema_version": manifest.get("schema_version"),
        "evaluation_regime": manifest.get("evaluation_regime"),
        "official_readiness_eligible": manifest.get("official_readiness_eligible"),
        "manifest": str(manifest_path.resolve()) if manifest_path is not None else None,
        "dataset_sha256": _sha256_file(dataset_path),
        "row_count": len(rows),
        "selection": manifest.get("selection"),
        "source_dataset": manifest.get("source_dataset"),
        "suite": manifest.get("suite"),
        "audit": manifest.get("audit"),
        "failures": failures,
    }


def _evaluation_regime_contract(
    *,
    regime: str,
    selected: list[tuple[int, dict[str, Any]]],
    question_families: tuple[str, ...],
    samples_per_family: int = 0,
) -> dict[str, Any]:
    failures: list[str] = []
    counts = Counter(_row_question_family(row) or "missing" for _, row in selected)
    if regime == SEEN_ROW_RECALL_REGIME:
        for source_index, row in selected:
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("split") != "train":
                failures.append(f"seen_row_not_from_train:{source_index}")
    if regime in {SEEN_ROW_RECALL_REGIME, SEEN_FACT_RETENTION_REGIME}:
        if samples_per_family <= 0:
            failures.append("diagnostic_samples_per_family_must_be_positive")
        if not question_families:
            failures.append("diagnostic_question_families_missing")
        unexpected = sorted(set(counts) - set(question_families))
        missing = sorted(set(question_families) - set(counts))
        if unexpected:
            failures.append(f"unexpected_question_families:{unexpected}")
        if missing:
            failures.append(f"missing_question_families:{missing}")
    return {
        "valid": not failures,
        "evaluation_regime": regime,
        "question_family_counts": dict(sorted(counts.items())),
        "failures": failures,
    }


def _wilson_interval(passes: int, count: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if count <= 0:
        return None
    rate = passes / count
    denominator = 1.0 + z * z / count
    centre = (rate + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [max(0.0, centre - half_width), min(1.0, centre + half_width)]


def _summarize_diagnostic_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    exact_pass_count = sum(item.get("exact_graph_fact_pass") is True for item in items)
    return {
        "count": count,
        "exact_pass_count": exact_pass_count,
        "exact_graph_fact_pass_rate": exact_pass_count / count if count else None,
        "exact_rate_wilson_95": _wilson_interval(exact_pass_count, count),
        "mean_id_recall": (
            sum(float(item["id_recall"]) for item in items if item.get("id_recall") is not None)
            / sum(item.get("id_recall") is not None for item in items)
            if any(item.get("id_recall") is not None for item in items)
            else None
        ),
    }


def _build_retention_report(
    *,
    prediction_rows: list[dict[str, Any]],
    canonical_objects: dict[str, dict[str, Any]],
    regime: str,
    question_families: tuple[str, ...],
    samples_per_family: int,
    minimum_family_exact: float,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_fact_ids: set[str] = set()
    for row in prediction_rows:
        item = evaluate_row(row, canonical_objects)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        family = str(metadata.get("question_family") or item.get("question_family") or "missing")
        template_id = str(metadata.get("retention_template_id") or "training_prompt")
        source_fact_id = metadata.get("source_oracle_fact_id") or metadata.get("oracle_fact_id")
        if isinstance(source_fact_id, str):
            source_fact_ids.add(source_fact_id)
        evaluated.append(item)
        by_family[family].append(item)
        by_template[template_id].append(item)

    requested_families = question_families or tuple(sorted(by_family))
    gates: list[dict[str, Any]] = []
    if regime in {SEEN_ROW_RECALL_REGIME, SEEN_FACT_RETENTION_REGIME}:
        for family in requested_families:
            family_summary = _summarize_diagnostic_items(by_family.get(family, []))
            gates.extend(
                [
                    {
                        "name": f"{family}.sample_count",
                        "observed": family_summary["count"],
                        "threshold": samples_per_family,
                        "op": ">=",
                        "passed": family_summary["count"] >= samples_per_family,
                    },
                    {
                        "name": f"{family}.exact_graph_fact_pass_rate",
                        "observed": family_summary["exact_graph_fact_pass_rate"],
                        "threshold": minimum_family_exact,
                        "op": ">=",
                        "passed": (
                            family_summary["exact_graph_fact_pass_rate"] is not None
                            and family_summary["exact_graph_fact_pass_rate"] >= minimum_family_exact
                        ),
                    },
                ]
            )
        passed: bool | None = all(gate["passed"] for gate in gates)
        decision = (
            "strong_memorization_signal"
            if passed
            else "insufficient_memorization_signal"
        )
    else:
        passed = None
        decision = "advisory_unseen_fact_generalization_only"

    return {
        "schema_version": RETENTION_REPORT_SCHEMA_VERSION,
        "valid": True,
        "passed": passed,
        "decision": decision,
        "purpose": "intermediate_memorization_diagnostic",
        "official_readiness_eligible": False,
        "evaluation_regime": regime,
        "thresholds": {
            "minimum_examples_per_family": samples_per_family,
            "minimum_exact_rate_per_family": minimum_family_exact,
        },
        "sample_count": len(evaluated),
        "unique_source_fact_count": len(source_fact_ids),
        "summary": _summarize_diagnostic_items(evaluated),
        "by_question_family": {
            family: _summarize_diagnostic_items(items)
            for family, items in sorted(by_family.items())
        },
        "by_prompt_template_id": {
            template_id: _summarize_diagnostic_items(items)
            for template_id, items in sorted(by_template.items())
        },
        "gates": gates,
        "failed_gates": [gate for gate in gates if not gate["passed"]],
    }


def _diagnostic_readiness(*, regime: str, retention_report_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "pretrajectory-sft-readiness-v2",
        "valid": True,
        "applicable": False,
        "passed": None,
        "official_readiness_eligible": False,
        "decision": "diagnostic_only_no_readiness_decision",
        "evaluation_regime": regime,
        "required_failure_count": 0,
        "advisory_failure_count": 0,
        "failed_required_gates": [],
        "failed_advisory_gates": [],
        "gates": [],
        "diagnostic_report": str(retention_report_path.resolve()),
    }


def _load_model(
    *,
    model_path: Path,
    adapter_path: Path | None,
    local_files_only: bool,
    trust_remote_code: bool,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            local_files_only=local_files_only,
            is_trainable=False,
        )
    model.eval()
    model.config.use_cache = True
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    return model


def _tokenizer(model_path: Path, *, local_files_only: bool, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id >= tokenizer.vocab_size:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def run_exact_eval(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.out_dir / "holdout_predictions.jsonl"
    exact_report_path = args.out_dir / "holdout_exact_report.json"
    html_report_path = args.out_dir / "holdout_review.html"
    gold_report_path = args.out_dir / "gold_reference_report.json"
    readiness_report_path = args.out_dir / "readiness_report.json"
    retention_report_path = args.out_dir / "retention_report.json"
    run_summary_path = args.out_dir / "run_summary.json"

    evaluation_regime = str(
        getattr(args, "evaluation_regime", OFFICIAL_READINESS_REGIME)
    )
    if evaluation_regime not in EVALUATION_REGIMES:
        raise ExactEvalContractError(f"Unsupported evaluation_regime: {evaluation_regime!r}.")
    question_families = _parse_question_families(
        getattr(args, "question_families", ())
    )
    samples_per_family = int(getattr(args, "samples_per_family", 0))
    retention_suite_manifest = getattr(args, "retention_suite_manifest", None)
    source_dataset_root = getattr(args, "source_dataset_root", None)
    minimum_diagnostic_family_exact = float(
        getattr(args, "minimum_diagnostic_family_exact", 0.80)
    )
    if not 0.0 <= minimum_diagnostic_family_exact <= 1.0:
        raise ExactEvalContractError("minimum_diagnostic_family_exact must be in [0, 1].")

    dataset_rows = read_jsonl(args.dataset_path)
    selected = _select_rows(
        dataset_rows,
        sample_size=args.sample_size,
        seed=args.seed,
        question_families=question_families,
        samples_per_family=samples_per_family,
    )
    regime_contract = _evaluation_regime_contract(
        regime=evaluation_regime,
        selected=selected,
        question_families=question_families,
        samples_per_family=samples_per_family,
    )
    retention_suite_contract = (
        _retention_suite_contract(
            manifest_path=retention_suite_manifest,
            dataset_path=args.dataset_path,
            rows=dataset_rows,
            source_dataset_root=source_dataset_root,
        )
        if evaluation_regime == SEEN_FACT_RETENTION_REGIME
        else None
    )

    canonical_path = args.canonical_objects or _find_dataset_artifact(args.dataset_path, "canonical_objects.jsonl")
    dataset_audit_path = args.dataset_audit or (
        source_dataset_root / "audit_report_contract_v3.json"
        if source_dataset_root is not None
        else _find_dataset_audit(args.dataset_path)
    )
    if canonical_path is None or not canonical_path.is_file():
        raise ExactEvalContractError("Could not locate canonical_objects.jsonl; pass --canonical-objects.")
    if dataset_audit_path is None or not dataset_audit_path.is_file():
        raise ExactEvalContractError(
            "Could not locate audit_report_contract_v3.json; pass the v3 bridge with "
            "--dataset-audit."
        )
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        raise ExactEvalContractError(f"Adapter path is not a directory: {args.adapter_path}")

    model_source_path = args.model_source_path or args.model_path
    summary: dict[str, Any] = {
        "contract_version": PIPELINE_CONTRACT_VERSION,
        "status": "initializing",
        "valid": None,
        "evaluation_regime": evaluation_regime,
        "purpose": (
            "official_readiness"
            if evaluation_regime == OFFICIAL_READINESS_REGIME
            else "intermediate_diagnostic"
        ),
        "official_readiness_eligible": evaluation_regime == OFFICIAL_READINESS_REGIME,
        "evaluation_target": "adapter" if args.adapter_path is not None else "base_model",
        "sample_count": len(selected),
        "model_path": str(args.model_path.resolve()),
        "model_source_path": str(model_source_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "dataset_path": str(args.dataset_path.resolve()),
        "canonical_objects": str(canonical_path.resolve()),
        "dataset_audit": str(dataset_audit_path.resolve()),
        "predictions": str(predictions_path.resolve()),
        "exact_report": str(exact_report_path.resolve()),
        "html_report": str(html_report_path.resolve()),
        "gold_reference_report": str(gold_report_path.resolve()),
        "readiness_report": str(readiness_report_path.resolve()),
        "retention_report": (
            str(retention_report_path.resolve())
            if evaluation_regime in DIAGNOSTIC_REGIMES
            else None
        ),
        "evaluation_regime_contract": regime_contract,
        "retention_suite_contract": retention_suite_contract,
        "generation": {
            "sample_size_requested": args.sample_size,
            "samples_per_family_requested": samples_per_family,
            "question_families": list(question_families),
            "sample_count": len(selected),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "max_total_tokens": args.max_total_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "enable_thinking": args.enable_thinking,
            "reasoning_effort": args.reasoning_effort,
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "max_report_examples": args.max_report_examples,
            "selected_rows_sha256": _selected_rows_sha256(selected),
        },
        "identity": {
            "model": _known_directory_files(
                args.model_path,
                ("config.json", "generation_config.json", "tokenizer_config.json"),
            ),
            "model_source": _known_directory_files(
                model_source_path,
                ("config.json", "generation_config.json", "tokenizer_config.json"),
            ),
            "adapter": _known_directory_files(
                args.adapter_path,
                ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"),
            ),
            "inputs": {
                "dataset": _artifact_metadata(args.dataset_path),
                "canonical_objects": _artifact_metadata(canonical_path),
                "dataset_audit": _artifact_metadata(dataset_audit_path),
                "retention_suite_manifest": _artifact_metadata(retention_suite_manifest),
                "source_dataset_root": (
                    {"path": str(source_dataset_root.resolve())}
                    if source_dataset_root is not None
                    else None
                ),
            },
            "implementation": {
                "runner": _artifact_metadata(Path(__file__)),
                "evaluator": _artifact_metadata(REPO_ROOT / "scripts" / "evaluate_pretrajectory_sft_predictions.py"),
                "readiness_checker": _artifact_metadata(REPO_ROOT / "scripts" / "check_pretrajectory_sft_readiness.py"),
            },
        },
        "runtime": {
            "argv": sys.argv,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_node_list": os.environ.get("SLURM_NODELIST"),
        },
    }

    dataset_audit_contract = _dataset_audit_contract(
        dataset_audit_path,
        dataset_path=(
            source_dataset_root / "train.jsonl"
            if source_dataset_root is not None
            else args.dataset_path
        ),
    )
    summary["dataset_audit_contract"] = dataset_audit_contract

    canonical_objects = load_canonical_objects(canonical_path)
    gold_contract_rows = [
        {**example, "idx": int(source_index)}
        for source_index, example in selected
    ]
    gold_contract = evaluate_gold_self_contract(
        gold_contract_rows,
        canonical_objects_by_id=canonical_objects,
    )
    write_json(gold_report_path, gold_contract)
    summary["gold_reference_contract"] = gold_contract
    summary["evaluator_contract_version"] = gold_contract.get("contract_version")

    if not dataset_audit_contract["valid"]:
        reason = (
            f"{dataset_audit_contract['message']} "
            f"fatal_error_count={dataset_audit_contract['fatal_error_count']}, "
            f"failures={dataset_audit_contract['failures']}."
        )
        invalid_readiness = _invalid_readiness(
            decision="invalid_dataset_contract",
            reason=reason,
            gold_report_path=gold_report_path,
            gate_name="dataset_audit_contract_valid",
            dataset_audit_path=dataset_audit_path,
        )
        write_json(readiness_report_path, invalid_readiness)
        summary.update(
            {
                "status": "invalid_dataset_contract",
                "valid": False,
                "readiness": {
                    "valid": False,
                    "passed": False,
                    "decision": invalid_readiness["decision"],
                },
                "artifacts": {
                    "dataset_audit": _artifact_metadata(dataset_audit_path),
                    "gold_reference_report": _artifact_metadata(gold_report_path),
                    "readiness_report": _artifact_metadata(readiness_report_path),
                },
            }
        )
        write_json(run_summary_path, summary)
        raise ExactEvalContractError(reason)

    evaluation_input_failures = list(regime_contract["failures"])
    if retention_suite_contract is not None:
        evaluation_input_failures.extend(retention_suite_contract["failures"])
    if evaluation_input_failures:
        reason = (
            "Evaluation regime inputs do not satisfy their contract: "
            f"failures={evaluation_input_failures}."
        )
        invalid_readiness = _invalid_readiness(
            decision="invalid_evaluation_regime_contract",
            reason=reason,
            gold_report_path=gold_report_path,
            gate_name="evaluation_regime_contract_valid",
            dataset_audit_path=dataset_audit_path,
        )
        write_json(readiness_report_path, invalid_readiness)
        summary.update(
            {
                "status": "invalid_evaluation_regime_contract",
                "valid": False,
                "readiness": {
                    "valid": False,
                    "passed": False,
                    "decision": invalid_readiness["decision"],
                },
                "artifacts": {
                    "dataset_audit": _artifact_metadata(dataset_audit_path),
                    "retention_suite_manifest": _artifact_metadata(retention_suite_manifest),
                    "gold_reference_report": _artifact_metadata(gold_report_path),
                    "readiness_report": _artifact_metadata(readiness_report_path),
                },
            }
        )
        write_json(run_summary_path, summary)
        raise ExactEvalContractError(reason)

    if gold_contract.get("passed") is not True:
        reason = (
            "Gold references do not satisfy the evaluator contract: "
            f"status={gold_contract.get('status')}, failure_count={gold_contract.get('failure_count')}."
        )
        invalid_readiness = _invalid_readiness(
            decision="invalid_gold_reference_contract",
            reason=reason,
            gold_report_path=gold_report_path,
        )
        write_json(readiness_report_path, invalid_readiness)
        summary.update(
            {
                "status": "invalid_gold_reference_contract",
                "valid": False,
                "readiness": {
                    "valid": False,
                    "passed": False,
                    "decision": invalid_readiness["decision"],
                },
                "artifacts": {
                    "gold_reference_report": _artifact_metadata(gold_report_path),
                    "readiness_report": _artifact_metadata(readiness_report_path),
                },
            }
        )
        write_json(run_summary_path, summary)
        raise ExactEvalContractError(reason)

    tokenizer = _tokenizer(
        args.model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = _load_model(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    format_fn = build_formatting_func(
        tokenizer,
        train=False,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort or None,
    )
    device = next(model.parameters()).device
    context_limit = args.max_total_tokens if args.max_total_tokens and args.max_total_tokens > 0 else None
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for local_index, (source_index, example) in enumerate(selected):
            prompt = format_fn(example)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_len = int(inputs["input_ids"].shape[1])
            max_new_tokens = args.max_new_tokens
            if context_limit is not None:
                room = context_limit - input_len
                if room <= 0:
                    keep_tokens = max(1, context_limit - 1)
                    inputs = {key: value[:, -keep_tokens:] for key, value in inputs.items()}
                    input_len = int(inputs["input_ids"].shape[1])
                    room = max(1, context_limit - input_len)
                max_new_tokens = max(1, min(max_new_tokens, room))
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": args.do_sample,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.do_sample:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p
            output = model.generate(**generate_kwargs)
            prediction = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            prediction_rows.append(
                _prediction_row(
                    source_index=source_index,
                    sample_index=local_index,
                    example=example,
                    prediction=prediction,
                )
            )
            if args.progress_every and (local_index + 1) % args.progress_every == 0:
                print(f"Generated {local_index + 1}/{len(selected)} examples.", flush=True)

    write_jsonl(predictions_path, prediction_rows)
    exact_report = evaluate_prediction_rows(
        prediction_rows,
        canonical_objects_by_id=canonical_objects,
        max_examples=args.max_report_examples,
    )
    dataset_identity = dataset_audit_contract.get("dataset_identity") or {}
    exact_report["dataset_identity"] = {
        "dataset_schema_version": dataset_audit_contract["dataset_schema_version"],
        "plan_hash": dataset_audit_contract.get("plan_hash"),
        "content_hash": dataset_identity.get("observed_content_hash"),
    }
    exact_report["evaluation_regime"] = evaluation_regime
    exact_report["official_readiness_eligible"] = (
        evaluation_regime == OFFICIAL_READINESS_REGIME
    )
    exact_report["selection"] = {
        "question_families": list(question_families),
        "samples_per_family": samples_per_family,
        "sample_size": len(prediction_rows),
        "selected_rows_sha256": _selected_rows_sha256(selected),
    }
    if retention_suite_contract is not None:
        exact_report["retention_suite_contract"] = retention_suite_contract
    write_json(exact_report_path, exact_report)
    html_report_path.write_text(
        render_html_report(exact_report, {row["idx"]: row for row in prediction_rows}),
        encoding="utf-8",
    )
    model_contract = _model_report_contract(exact_report, expected_count=len(prediction_rows))
    summary["model_report_contract"] = model_contract
    summary["evaluator_contract"] = exact_report.get("evaluator_contract")
    if not model_contract["valid"]:
        reason = model_contract["message"]
        invalid_readiness = _invalid_readiness(
            decision="invalid_model_report_contract",
            reason=reason,
            gold_report_path=gold_report_path,
        )
        write_json(readiness_report_path, invalid_readiness)
        summary.update(
            {
                "status": "invalid_model_report_contract",
                "valid": False,
                "summary": exact_report.get("summary"),
                "readiness": {
                    "valid": False,
                    "passed": False,
                    "decision": invalid_readiness["decision"],
                },
            }
        )
        summary["artifacts"] = {
            "predictions": _artifact_metadata(predictions_path),
            "exact_report": _artifact_metadata(exact_report_path),
            "html_report": _artifact_metadata(html_report_path),
            "gold_reference_report": _artifact_metadata(gold_report_path),
            "readiness_report": _artifact_metadata(readiness_report_path),
        }
        write_json(run_summary_path, summary)
        raise ExactEvalContractError(reason)

    if evaluation_regime in DIAGNOSTIC_REGIMES:
        retention_report = _build_retention_report(
            prediction_rows=prediction_rows,
            canonical_objects=canonical_objects,
            regime=evaluation_regime,
            question_families=question_families,
            samples_per_family=samples_per_family,
            minimum_family_exact=minimum_diagnostic_family_exact,
        )
        write_json(retention_report_path, retention_report)
        readiness_report = _diagnostic_readiness(
            regime=evaluation_regime,
            retention_report_path=retention_report_path,
        )
        write_json(readiness_report_path, readiness_report)
        exact_report["diagnostic"] = {
            "report": str(retention_report_path.resolve()),
            "passed": retention_report.get("passed"),
            "decision": retention_report.get("decision"),
        }
        write_json(exact_report_path, exact_report)
        summary.update(
            {
                "status": "completed",
                "valid": True,
                "summary": exact_report["summary"],
                "diagnostic": {
                    "valid": retention_report["valid"],
                    "passed": retention_report.get("passed"),
                    "decision": retention_report.get("decision"),
                },
                "readiness": {
                    "valid": True,
                    "applicable": False,
                    "passed": None,
                    "decision": readiness_report["decision"],
                    "required_failure_count": 0,
                    "advisory_failure_count": 0,
                },
                "artifacts": {
                    "predictions": _artifact_metadata(predictions_path),
                    "exact_report": _artifact_metadata(exact_report_path),
                    "html_report": _artifact_metadata(html_report_path),
                    "gold_reference_report": _artifact_metadata(gold_report_path),
                    "retention_report": _artifact_metadata(retention_report_path),
                    "retention_suite_manifest": _artifact_metadata(retention_suite_manifest),
                    "readiness_report": _artifact_metadata(readiness_report_path),
                },
            }
        )
        write_json(run_summary_path, summary)
        return summary

    try:
        readiness_report = _run_readiness_checker(
            dataset_audit=dataset_audit_path,
            exact_report=exact_report_path,
            output_path=readiness_report_path,
        )
    except ExactEvalContractError as error:
        invalid_readiness = _invalid_readiness(
            decision="invalid_readiness_checker",
            reason=str(error),
            gold_report_path=gold_report_path,
        )
        write_json(readiness_report_path, invalid_readiness)
        summary.update(
            {
                "status": "invalid_readiness_checker",
                "valid": False,
                "summary": exact_report.get("summary"),
                "readiness": {
                    "valid": False,
                    "passed": False,
                    "decision": invalid_readiness["decision"],
                },
                "artifacts": {
                    "predictions": _artifact_metadata(predictions_path),
                    "exact_report": _artifact_metadata(exact_report_path),
                    "html_report": _artifact_metadata(html_report_path),
                    "gold_reference_report": _artifact_metadata(gold_report_path),
                    "readiness_report": _artifact_metadata(readiness_report_path),
                },
            }
        )
        write_json(run_summary_path, summary)
        raise

    if readiness_report.get("valid") is not True:
        failed_contract_names = [
            gate.get("name")
            for gate in readiness_report.get("failed_contract_gates", [])
            if isinstance(gate, dict)
        ]
        reason = (
            "Readiness checker classified the evaluation inputs as contract-invalid: "
            f"decision={readiness_report.get('decision')}, "
            f"failed_contract_gates={failed_contract_names}."
        )
        summary.update(
            {
                "status": "invalid_readiness_contract",
                "valid": False,
                "summary": exact_report.get("summary"),
                "readiness": {
                    "valid": readiness_report.get("valid"),
                    "passed": readiness_report.get("passed"),
                    "decision": readiness_report.get("decision"),
                    "required_failure_count": readiness_report.get("required_failure_count"),
                    "advisory_failure_count": readiness_report.get("advisory_failure_count"),
                },
                "artifacts": {
                    "predictions": _artifact_metadata(predictions_path),
                    "exact_report": _artifact_metadata(exact_report_path),
                    "html_report": _artifact_metadata(html_report_path),
                    "gold_reference_report": _artifact_metadata(gold_report_path),
                    "readiness_report": _artifact_metadata(readiness_report_path),
                },
            }
        )
        write_json(run_summary_path, summary)
        raise ExactEvalContractError(reason)

    readiness_report["dataset_audit_contract_valid"] = True
    readiness_report["gold_reference_contract_valid"] = True
    write_json(readiness_report_path, readiness_report)

    summary.update(
        {
            "status": "completed",
            "valid": True,
            "summary": exact_report["summary"],
            "readiness": {
                "valid": readiness_report["valid"],
                "passed": bool(readiness_report.get("passed")),
                "decision": readiness_report.get("decision"),
                "required_failure_count": readiness_report.get("required_failure_count"),
                "advisory_failure_count": readiness_report.get("advisory_failure_count"),
            },
            "artifacts": {
                "predictions": _artifact_metadata(predictions_path),
                "exact_report": _artifact_metadata(exact_report_path),
                "html_report": _artifact_metadata(html_report_path),
                "gold_reference_report": _artifact_metadata(gold_report_path),
                "readiness_report": _artifact_metadata(readiness_report_path),
            },
        }
    )
    write_json(run_summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-path", type=Path, default=None)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--canonical-objects", type=Path, default=None)
    parser.add_argument("--dataset-audit", type=Path, default=None)
    parser.add_argument(
        "--source-dataset-root",
        type=Path,
        default=None,
        help="Frozen curriculum root used for audit identity when evaluating a derived suite.",
    )
    parser.add_argument(
        "--evaluation-regime",
        choices=EVALUATION_REGIMES,
        default=OFFICIAL_READINESS_REGIME,
    )
    parser.add_argument(
        "--question-families",
        default="",
        help="Comma-separated family strata. Required with --samples-per-family.",
    )
    parser.add_argument("--samples-per-family", type=int, default=0)
    parser.add_argument("--retention-suite-manifest", type=Path, default=None)
    parser.add_argument("--minimum-diagnostic-family-exact", type=float, default=0.80)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-total-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", type=_bool_arg, default=False)
    parser.add_argument("--enable-thinking", type=_bool_arg, default=False)
    parser.add_argument("--reasoning-effort", type=str, default="")
    parser.add_argument("--local-files-only", type=_bool_arg, default=True)
    parser.add_argument("--trust-remote-code", type=_bool_arg, default=True)
    parser.add_argument("--max-report-examples", type=int, default=120)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    try:
        summary = run_exact_eval(parse_args())
    except ExactEvalContractError as error:
        print(f"[INVALID EXACT-EVAL PIPELINE] {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
