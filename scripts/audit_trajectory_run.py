#!/usr/bin/env python3
"""Audit generated trajectory artifacts before using them for DPO.

The audit is intentionally read-only. It checks that a finished trajectory run
is structurally valid, free of known hidden-supervision leaks, and healthy
enough to be considered for large-scale generation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import CandidateBranch, PreferencePair, TrajectoryTurn
from runtime.validators import validate_candidate_branch, validate_preference_pair


ARTIFACT_JSONL_FILES = (
    "branch_pools.jsonl",
    "trajectory_turns.jsonl",
    "finding_records.jsonl",
    "preference_pairs_raw.jsonl",
    "preference_pairs.jsonl",
    "final_summaries.jsonl",
)
REQUIRED_JSON_FILES = ("progress.json", "manifest.json")
BLOCKED_KEYS = {
    "hidden_target",
    "target_gene_ids",
    "target_gene_symbols",
    "mechanism_labels",
    "terminal_score_metadata",
    "raw_actor_response",
    "raw_verifier_response",
    "token_ids",
    "prompt_token_ids",
}
EXPECTED_TASK_TYPES = ("explanation", "recovery", "refinement", "none")
EXPECTED_EVIDENCE_MODES = ("contextual", "full", "graph", "minimal")
EXPECTED_DIFFICULTY_BINS = ("easy", "medium", "hard")
CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is (?P<limit>\d+) tokens.*request has (?P<input>\d+) input tokens",
    re.IGNORECASE,
)


@dataclass
class AuditFinding:
    severity: str
    code: str
    message: str
    artifact: str | None = None
    row_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditConfig:
    preference_pair_margin: float = 0.10
    max_all_tie_rate: float = 0.20
    max_top_tie_rate: float = 1.00
    max_generator_error_rate: float = 0.0
    min_balanced_pair_bins: int = 6
    require_completed: bool = True
    require_pairs: bool = True
    required_task_types: tuple[str, ...] = EXPECTED_TASK_TYPES
    required_evidence_modes: tuple[str, ...] = EXPECTED_EVIDENCE_MODES


@dataclass
class AuditReport:
    run_dir: str
    ok: bool = True
    findings: list[AuditFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    freeze: dict[str, Any] = field(default_factory=dict)

    def add(self, severity: str, code: str, message: str, *, artifact: str | None = None, row_index: int | None = None) -> None:
        if severity == "error":
            self.ok = False
        self.findings.append(
            AuditFinding(
                severity=severity,
                code=code,
                message=message,
                artifact=artifact,
                row_index=row_index,
            )
        )

    def error(self, code: str, message: str, *, artifact: str | None = None, row_index: int | None = None) -> None:
        self.add("error", code, message, artifact=artifact, row_index=row_index)

    def warning(self, code: str, message: str, *, artifact: str | None = None, row_index: int | None = None) -> None:
        self.add("warning", code, message, artifact=artifact, row_index=row_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "metrics": self.metrics,
            "freeze": self.freeze,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _read_json(path: Path, report: AuditReport) -> dict[str, Any] | None:
    if not path.exists():
        report.error("missing_json", f"Missing required JSON file: {path.name}.", artifact=path.name)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.error("invalid_json", f"Could not parse {path.name}: {exc}", artifact=path.name)
        return None
    if not isinstance(payload, dict):
        report.error("invalid_json_shape", f"{path.name} must contain a JSON object.", artifact=path.name)
        return None
    return payload


def _read_jsonl(path: Path, report: AuditReport) -> list[dict[str, Any]]:
    if not path.exists():
        report.error("missing_jsonl", f"Missing required JSONL file: {path.name}.", artifact=path.name)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                report.error("invalid_jsonl", f"Could not parse line {row_index}: {exc}", artifact=path.name, row_index=row_index)
                continue
            if not isinstance(payload, dict):
                report.error("invalid_jsonl_shape", "JSONL rows must be objects.", artifact=path.name, row_index=row_index)
                continue
            rows.append(payload)
    return rows


def _walk_blocked_keys(value: Any, *, path: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in BLOCKED_KEYS:
                yield child_path
            yield from _walk_blocked_keys(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_blocked_keys(item, path=f"{path}[{index}]")


def _counter_to_sorted_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _build_freeze(run_dir: Path, manifest: dict[str, Any] | None, progress: dict[str, Any] | None) -> dict[str, Any]:
    served_models = None
    served_models_path = run_dir / "served_models.json"
    if served_models_path.exists():
        try:
            served_models = json.loads(served_models_path.read_text(encoding="utf-8"))
        except Exception:
            served_models = None

    run_freeze = None
    run_freeze_path = run_dir / "run_freeze.json"
    if run_freeze_path.exists():
        try:
            run_freeze = json.loads(run_freeze_path.read_text(encoding="utf-8"))
        except Exception:
            run_freeze = None

    served_model_id = None
    if isinstance(served_models, dict):
        data = served_models.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            served_model_id = data[0].get("id")

    return {
        "current_git_sha": _run_git(["rev-parse", "HEAD"]),
        "current_git_status_short": _run_git(["status", "--short"]),
        "manifest_config": (manifest or {}).get("config", {}),
        "manifest_generator": (manifest or {}).get("generator", {}),
        "manifest_task_selection": (manifest or {}).get("task_selection", {}),
        "manifest_runtime": (manifest or {}).get("runtime", {}),
        "progress_status": (progress or {}).get("status"),
        "served_model_id": served_model_id,
        "run_freeze": run_freeze,
    }


def _audit_logs(run_dir: Path, report: AuditReport) -> None:
    log_path = run_dir / "vllm_server.log"
    metrics = {
        "vllm_400_count": 0,
        "vllm_error_count": 0,
        "vllm_blocking_error_count": 0,
        "vllm_startup_safetensors_error_count": 0,
        "observed_context_limit": None,
        "max_rejected_input_tokens": None,
    }
    if not log_path.exists():
        report.warning("missing_vllm_log", "No vllm_server.log was found; backend health cannot be audited.", artifact="vllm_server.log")
        report.metrics.update(metrics)
        return

    text = log_path.read_text(encoding="utf-8", errors="replace")
    error_lines = [line for line in text.splitlines() if "ERROR " in line]
    startup_safetensors_errors = [
        line
        for line in error_lines
        if "Error retrieving safetensors: Repo id must be in the form" in line
    ]
    metrics["vllm_400_count"] = text.count("400 Bad Request")
    metrics["vllm_error_count"] = len(error_lines)
    metrics["vllm_startup_safetensors_error_count"] = len(startup_safetensors_errors)
    metrics["vllm_blocking_error_count"] = len(error_lines) - len(startup_safetensors_errors)

    limits: list[int] = []
    rejected_inputs: list[int] = []
    for match in CONTEXT_LIMIT_RE.finditer(text):
        limits.append(int(match.group("limit")))
        rejected_inputs.append(int(match.group("input")))
    if limits:
        metrics["observed_context_limit"] = min(limits)
    if rejected_inputs:
        metrics["max_rejected_input_tokens"] = max(rejected_inputs)

    if metrics["vllm_400_count"]:
        report.error("backend_400", f"vLLM log contains {metrics['vllm_400_count']} HTTP 400 responses.", artifact="vllm_server.log")
    if metrics["vllm_blocking_error_count"]:
        report.error("backend_error_log", f"vLLM log contains {metrics['vllm_blocking_error_count']} non-startup ERROR lines.", artifact="vllm_server.log")
    if metrics["vllm_startup_safetensors_error_count"]:
        report.warning(
            "startup_safetensors_probe_error",
            f"vLLM log contains {metrics['vllm_startup_safetensors_error_count']} safetensors probe ERROR lines during startup.",
            artifact="vllm_server.log",
        )
    report.metrics.update(metrics)


def _audit_blocked_keys(artifact_name: str, rows: list[dict[str, Any]], report: AuditReport) -> None:
    for index, row in enumerate(rows, start=1):
        blocked_paths = list(_walk_blocked_keys(row))
        if blocked_paths:
            preview = ", ".join(blocked_paths[:5])
            report.error(
                "blocked_artifact_key",
                f"Artifact contains blocked hidden/raw payload keys: {preview}",
                artifact=artifact_name,
                row_index=index,
            )


def _is_close(left: float, right: float, *, abs_tol: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=abs_tol)


def _audit_branch_pools(rows: list[dict[str, Any]], report: AuditReport, config: AuditConfig) -> dict[tuple[str, int], dict[str, Any]]:
    task_type_counts: Counter[str] = Counter()
    evidence_mode_counts: Counter[str] = Counter()
    selected_tool_counts: Counter[str] = Counter()
    relationship_counts: Counter[str] = Counter()
    generator_error_count = 0
    fallback_branch_count = 0
    total_branches = 0
    all_tie_count = 0
    top_tie_count = 0
    no_pair_step_keys: set[tuple[str, int]] = set()
    branch_pool_lookup: dict[tuple[str, int], dict[str, Any]] = {}

    for row_index, row in enumerate(rows, start=1):
        task_type = row.get("task_type")
        evidence_mode = row.get("evidence_mode")
        if isinstance(task_type, str):
            task_type_counts[task_type] += 1
        if isinstance(evidence_mode, str):
            evidence_mode_counts[evidence_mode] += 1

        trajectory_id = row.get("trajectory_id")
        step_index = row.get("step_index")
        if isinstance(trajectory_id, str) and isinstance(step_index, int):
            branch_pool_lookup[(trajectory_id, step_index)] = row
            no_pair_step_keys.add((trajectory_id, step_index))

        branches = row.get("branches")
        if not isinstance(branches, list) or not branches:
            report.error("empty_branch_pool", "Branch pool must contain at least one branch.", artifact="branch_pools.jsonl", row_index=row_index)
            continue
        total_branches += len(branches)

        branch_ids = [branch.get("branch_id") for branch in branches if isinstance(branch, dict)]
        if len(branch_ids) != len(set(branch_ids)):
            report.error("duplicate_branch_id", "Branch pool contains duplicate branch ids.", artifact="branch_pools.jsonl", row_index=row_index)

        selected_branch_id = row.get("selected_branch_id")
        if selected_branch_id not in branch_ids:
            report.error("selected_branch_missing", "selected_branch_id is not present in this branch pool.", artifact="branch_pools.jsonl", row_index=row_index)

        scores: list[float] = []
        normalized_scores: list[float] = []
        selected_branch: dict[str, Any] | None = None
        for branch_payload in branches:
            if not isinstance(branch_payload, dict):
                report.error("invalid_branch_shape", "Branch entries must be JSON objects.", artifact="branch_pools.jsonl", row_index=row_index)
                continue
            try:
                branch = CandidateBranch.from_dict(branch_payload)
            except Exception as exc:
                report.error("branch_schema_parse", f"CandidateBranch parse failed: {exc}", artifact="branch_pools.jsonl", row_index=row_index)
                continue
            validation = validate_candidate_branch(branch)
            if not validation.valid:
                report.error("branch_validation", "; ".join(validation.errors), artifact="branch_pools.jsonl", row_index=row_index)

            score_metadata = branch.local_score.score_metadata
            if score_metadata.get("schema_valid") is False:
                report.error("branch_local_schema_invalid", "Local scorer marked branch schema_valid=false.", artifact="branch_pools.jsonl", row_index=row_index)

            scores.append(float(branch.local_score.total_score))
            if branch.local_score.normalized_score is None:
                report.error("missing_normalized_score", "Branch local_score.normalized_score is missing.", artifact="branch_pools.jsonl", row_index=row_index)
            else:
                normalized_scores.append(float(branch.local_score.normalized_score))

            errors = branch.metadata.get("generator_errors", [])
            if isinstance(errors, list):
                generator_error_count += len(errors)
            if branch.metadata.get("generator_backend") == "heuristic_fallback":
                fallback_branch_count += 1
            if branch.branch_id == selected_branch_id:
                selected_branch = branch_payload

        if not scores or len(normalized_scores) != len(scores):
            continue

        min_score = min(scores)
        max_score = max(scores)
        if _is_close(min_score, max_score):
            all_tie_count += 1
            expected = 1.0 if len(scores) == 1 else 0.5
            if any(not _is_close(score, expected) for score in normalized_scores):
                report.error("bad_tie_normalization", f"All tied branches should have normalized_score={expected}.", artifact="branch_pools.jsonl", row_index=row_index)
        else:
            expected_min = min(normalized_scores)
            expected_max = max(normalized_scores)
            if not _is_close(expected_min, 0.0) or not _is_close(expected_max, 1.0):
                report.error("bad_score_normalization_bounds", "Non-tied branch pools must normalize to min 0.0 and max 1.0.", artifact="branch_pools.jsonl", row_index=row_index)
            for score, normalized in zip(scores, normalized_scores, strict=True):
                expected = (score - min_score) / (max_score - min_score)
                if not _is_close(normalized, expected):
                    report.error("bad_score_normalization", "normalized_score does not match total_score min/max normalization.", artifact="branch_pools.jsonl", row_index=row_index)
                    break

        max_normalized = max(normalized_scores)
        top_count = sum(1 for score in normalized_scores if _is_close(score, max_normalized))
        if top_count > 1:
            top_tie_count += 1

        if selected_branch is not None:
            selected_score = selected_branch.get("local_score", {}).get("normalized_score")
            if not isinstance(selected_score, (int, float)) or not _is_close(float(selected_score), max_normalized):
                report.error("selected_not_top_scored", "Selected branch is not tied for the top normalized score.", artifact="branch_pools.jsonl", row_index=row_index)

            action = selected_branch.get("actor_step", {}).get("tool_action")
            selected_tool_counts[(action or {}).get("tool_name", "no_tool") if isinstance(action, dict) else "no_tool"] += 1
            state = selected_branch.get("verifier_step", {}).get("updated_state", {})
            if isinstance(state, dict):
                relationship_counts[str(state.get("relationship_status"))] += 1

    branch_pool_count = len(rows)
    all_tie_rate = all_tie_count / branch_pool_count if branch_pool_count else 0.0
    top_tie_rate = top_tie_count / branch_pool_count if branch_pool_count else 0.0
    generator_error_rate = generator_error_count / total_branches if total_branches else 0.0

    if branch_pool_count == 0:
        report.error("no_branch_pools", "No branch pools were produced.", artifact="branch_pools.jsonl")
    if all_tie_rate > config.max_all_tie_rate:
        report.error("all_tie_rate_high", f"All-tie branch-pool rate {all_tie_rate:.3f} exceeds {config.max_all_tie_rate:.3f}.")
    if top_tie_rate > config.max_top_tie_rate:
        report.error("top_tie_rate_high", f"Top-score tie rate {top_tie_rate:.3f} exceeds {config.max_top_tie_rate:.3f}.")
    if generator_error_rate > config.max_generator_error_rate:
        report.error("generator_error_rate_high", f"Generator error rate {generator_error_rate:.3f} exceeds {config.max_generator_error_rate:.3f}.")
    if fallback_branch_count:
        report.error("heuristic_fallback_used", f"{fallback_branch_count} branches used heuristic_fallback; DPO generation should use model-backed branches only.")

    missing_task_types = sorted(set(config.required_task_types) - set(task_type_counts))
    if missing_task_types:
        report.error("missing_task_type_coverage", "Missing required task types: " + ", ".join(missing_task_types) + ".")
    missing_evidence_modes = sorted(set(config.required_evidence_modes) - set(evidence_mode_counts))
    if missing_evidence_modes:
        report.error("missing_evidence_mode_coverage", "Missing required evidence modes: " + ", ".join(missing_evidence_modes) + ".")

    report.metrics.update(
        {
            "branch_pool_count": branch_pool_count,
            "total_branches": total_branches,
            "all_tie_branch_pools": all_tie_count,
            "all_tie_rate": all_tie_rate,
            "top_tie_branch_pools": top_tie_count,
            "top_tie_rate": top_tie_rate,
            "generator_error_count": generator_error_count,
            "generator_error_rate": generator_error_rate,
            "heuristic_fallback_branch_count": fallback_branch_count,
            "task_type_counts": _counter_to_sorted_dict(task_type_counts),
            "evidence_mode_counts": _counter_to_sorted_dict(evidence_mode_counts),
            "selected_tool_counts": _counter_to_sorted_dict(selected_tool_counts),
            "selected_relationship_counts": _counter_to_sorted_dict(relationship_counts),
        }
    )
    return branch_pool_lookup


def _audit_turns(rows: list[dict[str, Any]], report: AuditReport) -> None:
    selected_false = 0
    for row_index, row in enumerate(rows, start=1):
        try:
            turn = TrajectoryTurn.from_dict(row)
        except Exception as exc:
            report.error("turn_schema_parse", f"TrajectoryTurn parse failed: {exc}", artifact="trajectory_turns.jsonl", row_index=row_index)
            continue
        if not turn.selected:
            selected_false += 1
        validation = validate_candidate_branch(turn.branch)
        if not validation.valid:
            report.error("turn_branch_validation", "; ".join(validation.errors), artifact="trajectory_turns.jsonl", row_index=row_index)
    if selected_false:
        report.error("unselected_turn", f"{selected_false} trajectory turns have selected=false.", artifact="trajectory_turns.jsonl")
    report.metrics["trajectory_turn_count"] = len(rows)


def _audit_final_summaries(rows: list[dict[str, Any]], report: AuditReport, manifest: dict[str, Any] | None) -> None:
    task_type_counts: Counter[str] = Counter()
    evidence_mode_counts: Counter[str] = Counter()
    step_counts: Counter[int] = Counter()
    rewards: list[float] = []
    for row_index, row in enumerate(rows, start=1):
        task_type = row.get("task_type")
        evidence_mode = row.get("evidence_mode")
        if isinstance(task_type, str):
            task_type_counts[task_type] += 1
        if isinstance(evidence_mode, str):
            evidence_mode_counts[evidence_mode] += 1
        step_count = row.get("step_count")
        if isinstance(step_count, int):
            step_counts[step_count] += 1
        else:
            report.error("summary_missing_step_count", "final_summaries row is missing integer step_count.", artifact="final_summaries.jsonl", row_index=row_index)
        for key in ("terminal_schema_score", "terminal_reward"):
            if not isinstance(row.get(key), (int, float)):
                report.error("summary_missing_score", f"final_summaries row is missing numeric {key}.", artifact="final_summaries.jsonl", row_index=row_index)
        if isinstance(row.get("terminal_reward"), (int, float)):
            rewards.append(float(row["terminal_reward"]))

    expected = (manifest or {}).get("num_trajectories")
    if isinstance(expected, int) and expected != len(rows):
        report.error("summary_count_mismatch", f"manifest num_trajectories={expected} but final_summaries has {len(rows)} rows.", artifact="final_summaries.jsonl")

    report.metrics.update(
        {
            "final_summary_count": len(rows),
            "final_summary_task_type_counts": _counter_to_sorted_dict(task_type_counts),
            "final_summary_evidence_mode_counts": _counter_to_sorted_dict(evidence_mode_counts),
            "step_count_distribution": _counter_to_sorted_dict(step_counts),
            "terminal_reward_min": min(rewards) if rewards else None,
            "terminal_reward_mean": (sum(rewards) / len(rewards)) if rewards else None,
            "terminal_reward_max": max(rewards) if rewards else None,
        }
    )


def _audit_preference_pairs(
    rows: list[dict[str, Any]],
    report: AuditReport,
    config: AuditConfig,
    *,
    artifact_name: str,
    branch_pool_lookup: dict[tuple[str, int], dict[str, Any]],
) -> set[tuple[str, int]]:
    pair_bins: Counter[tuple[str, str]] = Counter()
    pair_step_keys: set[tuple[str, int]] = set()
    margins: list[float] = []
    for row_index, row in enumerate(rows, start=1):
        try:
            pair = PreferencePair.from_dict(row)
        except Exception as exc:
            report.error("pair_schema_parse", f"PreferencePair parse failed: {exc}", artifact=artifact_name, row_index=row_index)
            continue

        validation = validate_preference_pair(pair)
        if not validation.valid:
            report.error("pair_validation", "; ".join(validation.errors), artifact=artifact_name, row_index=row_index)
        if pair.score_margin < config.preference_pair_margin:
            report.error("pair_margin_too_small", f"score_margin={pair.score_margin:.6f} is below {config.preference_pair_margin:.6f}.", artifact=artifact_name, row_index=row_index)
        if pair.normalized_score_chosen < pair.normalized_score_rejected:
            report.error("pair_reversed_normalized_score", "chosen normalized score is lower than rejected.", artifact=artifact_name, row_index=row_index)
        if pair.raw_score_chosen < pair.raw_score_rejected:
            report.error("pair_reversed_raw_score", "chosen raw score is lower than rejected.", artifact=artifact_name, row_index=row_index)

        key = (pair.trajectory_id, pair.decision_step)
        pair_step_keys.add(key)
        pool = branch_pool_lookup.get(key)
        if pool is not None:
            pool_branch_ids = {
                branch.get("branch_id")
                for branch in pool.get("branches", [])
                if isinstance(branch, dict)
            }
            if pair.chosen.branch_id != pool.get("selected_branch_id"):
                report.error("pair_chosen_not_selected", "Pair chosen branch does not match branch-pool selected_branch_id.", artifact=artifact_name, row_index=row_index)
            if pair.chosen.branch_id not in pool_branch_ids or pair.rejected.branch_id not in pool_branch_ids:
                report.error("pair_branch_not_in_pool", "Pair chosen/rejected branch id is not present in the source branch pool.", artifact=artifact_name, row_index=row_index)

        pair_bins[(pair.task_type.value, pair.difficulty_bin.value)] += 1
        margins.append(pair.score_margin)

    if artifact_name == "preference_pairs.jsonl":
        if config.require_pairs and not rows:
            report.error("no_preference_pairs", "No balanced preference pairs were produced.", artifact=artifact_name)
        if len(pair_bins) < config.min_balanced_pair_bins:
            report.error(
                "balanced_pair_bins_starved",
                f"Balanced preference pairs cover {len(pair_bins)} bins, below required minimum {config.min_balanced_pair_bins}.",
                artifact=artifact_name,
            )

        report.metrics.update(
            {
                "preference_pair_count": len(rows),
                "preference_pair_bins": {f"{task_type}/{difficulty}": count for (task_type, difficulty), count in sorted(pair_bins.items())},
                "preference_pair_margin_min": min(margins) if margins else None,
                "preference_pair_margin_mean": (sum(margins) / len(margins)) if margins else None,
                "preference_pair_margin_max": max(margins) if margins else None,
            }
        )
    else:
        report.metrics.update(
            {
                "raw_preference_pair_count": len(rows),
                "raw_preference_pair_bins": {f"{task_type}/{difficulty}": count for (task_type, difficulty), count in sorted(pair_bins.items())},
            }
        )
    return pair_step_keys


def audit_run(run_dir: Path, config: AuditConfig) -> AuditReport:
    report = AuditReport(run_dir=str(run_dir))
    if not run_dir.exists():
        report.error("missing_run_dir", f"Run directory does not exist: {run_dir}")
        return report

    progress = _read_json(run_dir / "progress.json", report)
    manifest = _read_json(run_dir / "manifest.json", report)
    report.freeze = _build_freeze(run_dir, manifest, progress)

    if config.require_completed and progress is not None and progress.get("status") != "completed":
        report.error("run_not_completed", f"progress.json status is {progress.get('status')!r}; expected 'completed'.", artifact="progress.json")
    if progress and "error" in progress:
        report.error("progress_contains_error", f"progress.json contains an error: {progress['error']}", artifact="progress.json")

    rows_by_file = {name: _read_jsonl(run_dir / name, report) for name in ARTIFACT_JSONL_FILES}
    for artifact_name, rows in rows_by_file.items():
        _audit_blocked_keys(artifact_name, rows, report)

    _audit_logs(run_dir, report)
    branch_pool_lookup = _audit_branch_pools(rows_by_file["branch_pools.jsonl"], report, config)
    _audit_turns(rows_by_file["trajectory_turns.jsonl"], report)
    _audit_final_summaries(rows_by_file["final_summaries.jsonl"], report, manifest)
    raw_pair_steps = _audit_preference_pairs(
        rows_by_file["preference_pairs_raw.jsonl"],
        report,
        config,
        artifact_name="preference_pairs_raw.jsonl",
        branch_pool_lookup=branch_pool_lookup,
    )
    balanced_pair_steps = _audit_preference_pairs(
        rows_by_file["preference_pairs.jsonl"],
        report,
        config,
        artifact_name="preference_pairs.jsonl",
        branch_pool_lookup=branch_pool_lookup,
    )

    branch_step_keys = set(branch_pool_lookup)
    report.metrics["branch_pool_steps_without_raw_pairs"] = len(branch_step_keys - raw_pair_steps)
    report.metrics["branch_pool_steps_without_balanced_pairs"] = len(branch_step_keys - balanced_pair_steps)

    if manifest is not None:
        artifacts = manifest.get("artifacts", {})
        if isinstance(artifacts, dict):
            expected_pairs = artifacts.get("preference_pair_count")
            actual_pairs = report.metrics.get("preference_pair_count")
            if isinstance(expected_pairs, int) and actual_pairs is not None and expected_pairs != actual_pairs:
                report.error("manifest_pair_count_mismatch", f"manifest preference_pair_count={expected_pairs} but file has {actual_pairs}.", artifact="manifest.json")
            expected_raw = artifacts.get("preference_pair_raw_count")
            actual_raw = report.metrics.get("raw_preference_pair_count")
            if isinstance(expected_raw, int) and actual_raw is not None and expected_raw != actual_raw:
                report.error("manifest_raw_pair_count_mismatch", f"manifest preference_pair_raw_count={expected_raw} but file has {actual_raw}.", artifact="manifest.json")

    return report


def _split_csv(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one trajectory-generation run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Trajectory run directory to audit.")
    parser.add_argument("--preference-pair-margin", type=float, default=0.10)
    parser.add_argument("--max-all-tie-rate", type=float, default=0.20)
    parser.add_argument("--max-top-tie-rate", type=float, default=1.0)
    parser.add_argument("--max-generator-error-rate", type=float, default=0.0)
    parser.add_argument("--min-balanced-pair-bins", type=int, default=6)
    parser.add_argument(
        "--required-task-types",
        type=str,
        default=",".join(EXPECTED_TASK_TYPES),
        help="Comma-separated task types required for a production-gate pass. Use '' to disable.",
    )
    parser.add_argument(
        "--required-evidence-modes",
        type=str,
        default=",".join(EXPECTED_EVIDENCE_MODES),
        help="Comma-separated evidence modes required for a production-gate pass. Use '' to disable.",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="Do not fail solely because progress.json is not completed.")
    parser.add_argument("--allow-empty-pairs", action="store_true", help="Do not fail solely because preference_pairs.jsonl is empty.")
    parser.add_argument("--json", action="store_true", help="Emit the full audit report as JSON.")
    return parser


def _print_human(report: AuditReport) -> None:
    status = "PASS" if report.ok else "FAIL"
    print(f"Trajectory audit: {status}")
    print(f"Run directory: {report.run_dir}")
    print("Key metrics:")
    for key in (
        "branch_pool_count",
        "total_branches",
        "all_tie_rate",
        "top_tie_rate",
        "preference_pair_count",
        "raw_preference_pair_count",
        "preference_pair_margin_min",
        "branch_pool_steps_without_raw_pairs",
        "vllm_400_count",
        "observed_context_limit",
        "max_rejected_input_tokens",
    ):
        print(f"  {key}: {report.metrics.get(key)}")
    print("Coverage:")
    print(f"  task_type_counts: {report.metrics.get('task_type_counts')}")
    print(f"  evidence_mode_counts: {report.metrics.get('evidence_mode_counts')}")
    print(f"  preference_pair_bins: {report.metrics.get('preference_pair_bins')}")
    print("Freeze:")
    print(f"  current_git_sha: {report.freeze.get('current_git_sha')}")
    print(f"  current_git_status_short: {report.freeze.get('current_git_status_short') or '<clean>'}")
    print(f"  served_model_id: {report.freeze.get('served_model_id')}")
    generator = report.freeze.get("manifest_generator") or {}
    if generator:
        print(f"  manifest_model_name: {generator.get('model_name')}")
        print(f"  manifest_api_mode: {generator.get('resolved_api_mode')}")
    if report.findings:
        print("Findings:")
        for finding in report.findings:
            location = ""
            if finding.artifact:
                location = f" [{finding.artifact}"
                if finding.row_index is not None:
                    location += f":{finding.row_index}"
                location += "]"
            print(f"  {finding.severity.upper()} {finding.code}{location}: {finding.message}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = AuditConfig(
        preference_pair_margin=args.preference_pair_margin,
        max_all_tie_rate=args.max_all_tie_rate,
        max_top_tie_rate=args.max_top_tie_rate,
        max_generator_error_rate=args.max_generator_error_rate,
        min_balanced_pair_bins=args.min_balanced_pair_bins,
        require_completed=not args.allow_incomplete,
        require_pairs=not args.allow_empty_pairs,
        required_task_types=_split_csv(args.required_task_types),
        required_evidence_modes=_split_csv(args.required_evidence_modes),
    )
    report = audit_run(args.run_dir, config)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_human(report)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
