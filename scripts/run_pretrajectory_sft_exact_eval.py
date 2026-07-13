#!/usr/bin/env python3
"""Run contract-checked exact evaluation for a base model or PEFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
    load_canonical_objects,
    render_html_report,
)
from utils.utils import build_formatting_func  # noqa: E402


PIPELINE_CONTRACT_VERSION = "pretrajectory-exact-eval-v2"
LEGACY_DATASET_AUDIT_SCHEMA_VERSION = "pretrajectory-sft-audit-v2"
CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION = "pretrajectory-sft-audit-v3"
CURRICULUM_DATASET_SCHEMA_VERSION = "pretrajectory-sft-v3"
REQUIRED_DATASET_AUDIT_SCHEMA_VERSION = LEGACY_DATASET_AUDIT_SCHEMA_VERSION
REQUIRED_EVALUATOR_CONTRACT_VERSION = "pretrajectory-sft-exact-v3"
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
    """Select the bridge audit for v3 and the legacy audit for v2."""

    manifest_path = _find_dataset_artifact(dataset_path, "manifest.json")
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if (
            isinstance(manifest, dict)
            and manifest.get("dataset_schema_version") == CURRICULUM_DATASET_SCHEMA_VERSION
        ):
            return _find_dataset_artifact(dataset_path, "audit_report_contract_v3.json")
    return _find_dataset_artifact(dataset_path, "audit_report.json")


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
    is_curriculum_v3 = dataset_schema_version == CURRICULUM_DATASET_SCHEMA_VERSION
    required_schema_version = (
        CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION
        if is_curriculum_v3
        else LEGACY_DATASET_AUDIT_SCHEMA_VERSION
    )
    if schema_version != required_schema_version:
        failures.append("unsupported_dataset_audit_schema_version")

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
    if is_curriculum_v3 and audit_passed is not True:
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

        if is_curriculum_v3:
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

    native_reports_valid: bool | None = None
    dataset_identity: dict[str, Any] | None = None
    if is_curriculum_v3:
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
        "required_schema_version": required_schema_version,
        "supported_schema_versions": [
            LEGACY_DATASET_AUDIT_SCHEMA_VERSION,
            CURRICULUM_DATASET_AUDIT_SCHEMA_VERSION,
        ],
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


def _select_rows(rows: list[dict[str, Any]], *, sample_size: int | None, seed: int) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(indexed), generator=generator).tolist()
    if sample_size is not None and sample_size > 0:
        order = order[: min(sample_size, len(order))]
    return [indexed[index] for index in order]


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
    run_summary_path = args.out_dir / "run_summary.json"

    dataset_rows = read_jsonl(args.dataset_path)
    selected = _select_rows(dataset_rows, sample_size=args.sample_size, seed=args.seed)

    canonical_path = args.canonical_objects or _find_dataset_artifact(args.dataset_path, "canonical_objects.jsonl")
    dataset_audit_path = args.dataset_audit or _find_dataset_audit(args.dataset_path)
    if canonical_path is None or not canonical_path.is_file():
        raise ExactEvalContractError("Could not locate canonical_objects.jsonl; pass --canonical-objects.")
    if dataset_audit_path is None or not dataset_audit_path.is_file():
        raise ExactEvalContractError(
            "Could not locate the schema-appropriate dataset audit; pass --dataset-audit."
        )
    if args.adapter_path is not None and not args.adapter_path.is_dir():
        raise ExactEvalContractError(f"Adapter path is not a directory: {args.adapter_path}")

    model_source_path = args.model_source_path or args.model_path
    summary: dict[str, Any] = {
        "contract_version": PIPELINE_CONTRACT_VERSION,
        "status": "initializing",
        "valid": None,
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
        "generation": {
            "sample_size_requested": args.sample_size,
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
        dataset_path=args.dataset_path,
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
    if dataset_audit_contract["dataset_schema_version"] == CURRICULUM_DATASET_SCHEMA_VERSION:
        dataset_identity = dataset_audit_contract.get("dataset_identity") or {}
        exact_report["dataset_identity"] = {
            "dataset_schema_version": dataset_audit_contract["dataset_schema_version"],
            "plan_hash": dataset_audit_contract.get("plan_hash"),
            "content_hash": dataset_identity.get("observed_content_hash"),
        }
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
