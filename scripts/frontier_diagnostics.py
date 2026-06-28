#!/usr/bin/env python3
"""Diagnose exact-membership frontier coverage in trajectory branch pools.

This module is intentionally read-only. It uses hidden targets only for
post-hoc scoring diagnostics and writes a separate report artifact; it must not
mutate branch pools, trajectory turns, prompts, or training exports.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import json
import statistics


DEFAULT_OUTPUT_NAME = "frontier_diagnostics.json"
DEFAULT_PROMPT_PREVIEW_LIMIT = 40
EXACT_PAIR_CATEGORIES = frozenset(
    {"exact_recovery", "exact_refinement", "exact_over_partial"}
)


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def read_jsonl_tolerant(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read JSONL rows, skipping malformed rows and returning parse errors."""

    if path is None or not path.exists():
        return [], [{"row_index": None, "error": "missing_file", "path": str(path)}]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                errors.append({"row_index": row_index, "error": str(exc)})
                continue
            if not isinstance(payload, dict):
                errors.append({"row_index": row_index, "error": f"expected object, got {type(payload).__name__}"})
                continue
            rows.append(payload)
    return rows, errors


def infer_tasks_path(run_dir: Path) -> Path | None:
    """Infer the task JSONL path from run_freeze.json or manifest.json."""

    for filename in ("run_freeze.json", "manifest.json", "completed_review_manifest.json"):
        payload = _read_json(run_dir / filename)
        if not payload:
            continue
        for path_value in _walk_task_paths(payload):
            path = Path(path_value)
            if path.exists():
                return path
    return None


def _walk_task_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"tasks_path", "task_path"} and isinstance(item, str) and item:
                paths.append(item)
            else:
                paths.extend(_walk_task_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_walk_task_paths(item))
    return paths


def load_task_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    rows, _errors = read_jsonl_tolerant(path)
    return {
        str(row["task_id"]): row
        for row in rows
        if isinstance(row.get("task_id"), str) and row.get("task_id")
    }


def _unique_strings(values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    if not isinstance(values, list):
        return out
    for value in values:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _predicted_gene_ids_from_state(state: Any) -> list[str]:
    if not isinstance(state, dict):
        return []
    groups = state.get("predicted_groups")
    if not isinstance(groups, list):
        return []
    gene_ids: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            continue
        for gene_id in group.get("gene_ids", []):
            if isinstance(gene_id, str) and gene_id and gene_id not in seen:
                seen.add(gene_id)
                gene_ids.append(gene_id)
    return gene_ids


def _branch_updated_state(branch: dict[str, Any]) -> dict[str, Any]:
    verifier = branch.get("verifier_step")
    if not isinstance(verifier, dict):
        return {}
    state = verifier.get("updated_state")
    return state if isinstance(state, dict) else {}


def _branch_predicted_gene_ids(branch: dict[str, Any]) -> list[str]:
    return _predicted_gene_ids_from_state(_branch_updated_state(branch))


def _branch_relationship_status(branch: dict[str, Any]) -> str:
    value = _branch_updated_state(branch).get("relationship_status")
    return value if isinstance(value, str) else ""


def _branch_task_success_level(branch: dict[str, Any]) -> str:
    local_score = branch.get("local_score")
    if not isinstance(local_score, dict):
        return "unknown"
    score_metadata = local_score.get("score_metadata")
    if not isinstance(score_metadata, dict):
        return "unknown"
    task_success = score_metadata.get("task_success")
    if not isinstance(task_success, dict):
        return "unknown"
    value = task_success.get("task_success_level")
    return value if isinstance(value, str) else "unknown"


def _branch_membership_metrics(branch: dict[str, Any]) -> dict[str, float]:
    local_score = branch.get("local_score")
    if not isinstance(local_score, dict):
        return {}
    score_metadata = local_score.get("score_metadata")
    if not isinstance(score_metadata, dict):
        return {}
    complex_meta = score_metadata.get("complex")
    if not isinstance(complex_meta, dict):
        return {}
    best_group = complex_meta.get("best_group_post")
    if not isinstance(best_group, dict):
        return {}
    metrics = best_group.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("jaccard", "precision", "recall"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _branch_exact_membership(branch: dict[str, Any]) -> bool:
    if (
        _branch_task_success_level(branch) == "positive"
        and _branch_relationship_status(branch) == "validated_group"
    ):
        return True
    metrics = _branch_membership_metrics(branch)
    return (
        _branch_relationship_status(branch) == "validated_group"
        and all(metrics.get(key, 0.0) >= 1.0 for key in ("jaccard", "precision", "recall"))
    )


def _branch_observation(branch: dict[str, Any]) -> dict[str, Any] | None:
    observation = branch.get("observation")
    return observation if isinstance(observation, dict) else None


def _observation_payload(branch: dict[str, Any]) -> dict[str, Any]:
    observation = _branch_observation(branch)
    if observation is None:
        return {}
    payload = observation.get("payload")
    return payload if isinstance(payload, dict) else {}


def _observation_provenance(branch: dict[str, Any]) -> dict[str, Any]:
    observation = _branch_observation(branch)
    if observation is None:
        return {}
    provenance = observation.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def _gene_id_from_ranked_item(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("gene_id", "gene", "node_id", "id", "name"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _rank_from_ranked_item(item: Any, fallback: int) -> float:
    if isinstance(item, dict):
        value = item.get("rank")
        if isinstance(value, (int, float)):
            return float(value)
    return float(fallback)


def _ranked_rows(branch: dict[str, Any]) -> list[dict[str, Any]]:
    payload = _observation_payload(branch)
    rows = payload.get("ranked_genes")
    if not isinstance(rows, list):
        rows = payload.get("results")
    return rows if isinstance(rows, list) else []


def _ranked_gene_ranks(branch: dict[str, Any]) -> dict[str, float]:
    ranks: dict[str, float] = {}
    for index, item in enumerate(_ranked_rows(branch), start=1):
        gene_id = _gene_id_from_ranked_item(item)
        if gene_id is None:
            continue
        rank = _rank_from_ranked_item(item, index)
        if gene_id not in ranks or rank < ranks[gene_id]:
            ranks[gene_id] = rank
    return ranks


def _prompt_preview_gene_ids(
    branch: dict[str, Any],
    *,
    task_type: str,
    current_gene_ids: set[str],
    prompt_preview_limit: int,
) -> set[str]:
    """Approximate the model-visible RWR non-seed preview from persisted rows."""

    if prompt_preview_limit <= 0:
        return set()
    preview: list[str] = []
    for item in _ranked_rows(branch):
        gene_id = _gene_id_from_ranked_item(item)
        if gene_id is None:
            continue
        if task_type == "recovery" and gene_id in current_gene_ids:
            continue
        if task_type == "refinement" and gene_id not in current_gene_ids:
            continue
        if gene_id not in preview:
            preview.append(gene_id)
        if len(preview) >= prompt_preview_limit:
            break
    return set(preview)


def _frontier_rows_from_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _frontier_gene_ids(rows: list[dict[str, Any]]) -> set[str]:
    gene_ids: set[str] = set()
    for row in rows:
        gene_id = _gene_id_from_ranked_item(row)
        if gene_id is not None:
            gene_ids.add(gene_id)
    return gene_ids


def _candidate_frontier_gene_ids(branch: dict[str, Any]) -> set[str]:
    gene_ids: set[str] = set()
    metadata = branch.get("metadata")
    provenance = _observation_provenance(branch)
    containers = [metadata if isinstance(metadata, dict) else {}, provenance]
    for container in containers:
        gene_ids.update(_frontier_gene_ids(_frontier_rows_from_value(container.get("candidate_frontier"))))
        edit = container.get("deterministic_membership_edit")
        if isinstance(edit, dict):
            gene_ids.update(_frontier_gene_ids(_frontier_rows_from_value(edit.get("candidate_frontier"))))
            gene_ids.update(_unique_strings(edit.get("candidate_gene_ids")))
    return gene_ids


def _deterministic_edit(branch: dict[str, Any]) -> dict[str, Any]:
    metadata = branch.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("deterministic_membership_edit"), dict):
        return metadata["deterministic_membership_edit"]
    provenance = _observation_provenance(branch)
    if isinstance(provenance.get("deterministic_membership_edit"), dict):
        return provenance["deterministic_membership_edit"]
    return {}


def _source_task_id(row: dict[str, Any]) -> str:
    value = row.get("source_task_id")
    if isinstance(value, str) and value:
        return value
    context = row.get("context")
    if isinstance(context, dict):
        value = context.get("source_task_id")
        if isinstance(value, str) and value:
            return value
    trajectory_id = row.get("trajectory_id")
    return str(trajectory_id or "")


def _row_current_gene_ids(row: dict[str, Any]) -> set[str]:
    context = row.get("context")
    state = context.get("state") if isinstance(context, dict) else {}
    return set(_predicted_gene_ids_from_state(state if isinstance(state, dict) else {}))


def _target_gene_ids(task_row: dict[str, Any] | None) -> set[str]:
    hidden = task_row.get("hidden_target") if isinstance(task_row, dict) else None
    if not isinstance(hidden, dict):
        return set()
    return set(_unique_strings(hidden.get("target_gene_ids")))


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _fmt_rate(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


class _TaskAccumulator:
    def __init__(
        self,
        *,
        source_task_id: str,
        trajectory_id: str,
        task_type: str,
        difficulty: str,
        evidence_mode: str,
        target_gene_ids: set[str],
    ) -> None:
        self.source_task_id = source_task_id
        self.trajectory_id = trajectory_id
        self.task_type = task_type
        self.difficulty = difficulty
        self.evidence_mode = evidence_mode
        self.target_gene_ids = set(target_gene_ids)
        self.initial_step_index: int | None = None
        self.initial_gene_ids: set[str] = set()
        self.branch_pool_count = 0
        self.branch_count = 0
        self.selected_branch_count = 0
        self.exact_branch_count = 0
        self.selected_exact_branch_count = 0
        self.rwr_ranks: dict[str, float] = {}
        self.prompt_preview_gene_ids: set[str] = set()
        self.edit_frontier_gene_ids: set[str] = set()
        self.any_branch_predicted_gene_ids: set[str] = set()
        self.selected_branch_predicted_gene_ids: set[str] = set()
        self.removed_gene_ids: set[str] = set()
        self.selected_removed_gene_ids: set[str] = set()
        self.edit_branch_count = 0
        self.selected_edit_branch_count = 0
        self.pair_count_raw = 0
        self.exact_pair_count_raw = 0
        self.pair_count_balanced = 0
        self.exact_pair_count_balanced = 0

    def note_initial(self, step_index: int, current_gene_ids: set[str]) -> None:
        if self.initial_step_index is None or step_index < self.initial_step_index:
            self.initial_step_index = step_index
            self.initial_gene_ids = set(current_gene_ids)

    def add_branch(self, branch: dict[str, Any], *, current_gene_ids: set[str], selected: bool, prompt_preview_limit: int) -> None:
        self.branch_count += 1
        if selected:
            self.selected_branch_count += 1
        predicted = set(_branch_predicted_gene_ids(branch))
        self.any_branch_predicted_gene_ids.update(predicted)
        if selected:
            self.selected_branch_predicted_gene_ids.update(predicted)
        if _branch_exact_membership(branch):
            self.exact_branch_count += 1
            if selected:
                self.selected_exact_branch_count += 1

        for gene_id, rank in _ranked_gene_ranks(branch).items():
            if gene_id not in self.rwr_ranks or rank < self.rwr_ranks[gene_id]:
                self.rwr_ranks[gene_id] = rank
        self.prompt_preview_gene_ids.update(
            _prompt_preview_gene_ids(
                branch,
                task_type=self.task_type,
                current_gene_ids=current_gene_ids,
                prompt_preview_limit=prompt_preview_limit,
            )
        )
        self.edit_frontier_gene_ids.update(_candidate_frontier_gene_ids(branch))

        edit = _deterministic_edit(branch)
        if edit:
            self.edit_branch_count += 1
            if selected:
                self.selected_edit_branch_count += 1
            removed = set(_unique_strings(edit.get("removed_gene_ids")))
            self.removed_gene_ids.update(removed)
            if selected:
                self.selected_removed_gene_ids.update(removed)

    def to_dict(self, *, max_gene_details: int) -> dict[str, Any]:
        if self.task_type == "recovery":
            target_candidates = self.target_gene_ids - self.initial_gene_ids
            details = [
                self._gene_detail(gene_id, target_role="missing_target")
                for gene_id in sorted(target_candidates)
            ]
            details.sort(key=lambda item: (item["rwr_best_rank"] is None, item["rwr_best_rank"] or 10**12, item["gene_id"]))
            return {
                **self._base_payload(),
                "target_gene_count": len(self.target_gene_ids),
                "initial_gene_count": len(self.initial_gene_ids),
                "recovery_missing_target_gene_count": len(target_candidates),
                "frontier_recalled_gene_count": len(target_candidates & set(self.rwr_ranks)),
                "frontier_surfaced_at_preview_gene_count": len(target_candidates & self.prompt_preview_gene_ids),
                "edit_frontier_gene_count": len(target_candidates & self.edit_frontier_gene_ids),
                "added_to_any_branch_gene_count": len(target_candidates & self.any_branch_predicted_gene_ids),
                "added_to_selected_branch_gene_count": len(target_candidates & self.selected_branch_predicted_gene_ids),
                "visible_but_not_added_gene_count": len((target_candidates & self.prompt_preview_gene_ids) - self.any_branch_predicted_gene_ids),
                "frontier_recall_at_topk": _fmt_rate(_rate(len(target_candidates & set(self.rwr_ranks)), len(target_candidates))),
                "frontier_surfaced_at_preview": _fmt_rate(_rate(len(target_candidates & self.prompt_preview_gene_ids), len(target_candidates))),
                "edit_frontier_coverage": _fmt_rate(_rate(len(target_candidates & self.edit_frontier_gene_ids), len(target_candidates))),
                "added_to_any_branch_rate": _fmt_rate(_rate(len(target_candidates & self.any_branch_predicted_gene_ids), len(target_candidates))),
                "added_to_selected_branch_rate": _fmt_rate(_rate(len(target_candidates & self.selected_branch_predicted_gene_ids), len(target_candidates))),
                "visible_but_not_added_rate": _fmt_rate(
                    _rate(
                        len((target_candidates & self.prompt_preview_gene_ids) - self.any_branch_predicted_gene_ids),
                        len(target_candidates & self.prompt_preview_gene_ids),
                    )
                ),
                "missing_target_gene_details": details[:max_gene_details],
                "missing_target_gene_detail_count": len(details),
            }

        extras = self.initial_gene_ids - self.target_gene_ids
        details = [
            self._gene_detail(gene_id, target_role="extra_input")
            for gene_id in sorted(extras)
        ]
        details.sort(key=lambda item: (not item["removed_by_any_edit"], item["rwr_best_rank"] is None, item["rwr_best_rank"] or 10**12, item["gene_id"]))
        return {
            **self._base_payload(),
            "target_gene_count": len(self.target_gene_ids),
            "initial_gene_count": len(self.initial_gene_ids),
            "refinement_extra_gene_count": len(extras),
            "frontier_flagged_extra_gene_count": len(extras & (self.edit_frontier_gene_ids | self.removed_gene_ids)),
            "removed_by_any_edit_gene_count": len(extras & self.removed_gene_ids),
            "removed_by_selected_edit_gene_count": len(extras & self.selected_removed_gene_ids),
            "selected_branch_pruned_extra_gene_count": len(extras - self.selected_branch_predicted_gene_ids),
            "frontier_flagged_extra_rate": _fmt_rate(_rate(len(extras & (self.edit_frontier_gene_ids | self.removed_gene_ids)), len(extras))),
            "removed_by_any_edit_rate": _fmt_rate(_rate(len(extras & self.removed_gene_ids), len(extras))),
            "removed_by_selected_edit_rate": _fmt_rate(_rate(len(extras & self.selected_removed_gene_ids), len(extras))),
            "selected_branch_pruned_extra_rate": _fmt_rate(_rate(len(extras - self.selected_branch_predicted_gene_ids), len(extras))),
            "extra_gene_details": details[:max_gene_details],
            "extra_gene_detail_count": len(details),
        }

    def _base_payload(self) -> dict[str, Any]:
        return {
            "source_task_id": self.source_task_id,
            "trajectory_id": self.trajectory_id,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "evidence_mode": self.evidence_mode,
            "branch_pool_count": self.branch_pool_count,
            "branch_count": self.branch_count,
            "selected_branch_count": self.selected_branch_count,
            "deterministic_edit_branch_count": self.edit_branch_count,
            "selected_deterministic_edit_branch_count": self.selected_edit_branch_count,
            "exact_branch_count": self.exact_branch_count,
            "selected_exact_branch_count": self.selected_exact_branch_count,
            "pair_count_raw": self.pair_count_raw,
            "exact_pair_count_raw": self.exact_pair_count_raw,
            "pair_count_balanced": self.pair_count_balanced,
            "exact_pair_count_balanced": self.exact_pair_count_balanced,
        }

    def _gene_detail(self, gene_id: str, *, target_role: str) -> dict[str, Any]:
        return {
            "gene_id": gene_id,
            "target_role": target_role,
            "rwr_best_rank": self.rwr_ranks.get(gene_id),
            "in_prompt_preview": gene_id in self.prompt_preview_gene_ids,
            "in_edit_frontier": gene_id in self.edit_frontier_gene_ids,
            "added_to_any_branch": gene_id in self.any_branch_predicted_gene_ids,
            "added_to_selected_branch": gene_id in self.selected_branch_predicted_gene_ids,
            "removed_by_any_edit": gene_id in self.removed_gene_ids,
            "removed_by_selected_edit": gene_id in self.selected_removed_gene_ids,
        }


def diagnose_frontier_run(
    *,
    run_dir: Path,
    tasks_path: Path | None = None,
    branch_pools_path: Path | None = None,
    preference_pairs_raw_path: Path | None = None,
    preference_pairs_path: Path | None = None,
    prompt_preview_limit: int = DEFAULT_PROMPT_PREVIEW_LIMIT,
    max_gene_details_per_task: int = 20,
) -> dict[str, Any]:
    """Build a frontier diagnostic report for recovery/refinement branch pools."""

    branch_pools_path = branch_pools_path or (run_dir / "branch_pools.jsonl")
    preference_pairs_raw_path = preference_pairs_raw_path or (run_dir / "preference_pairs_raw.jsonl")
    preference_pairs_path = preference_pairs_path or (run_dir / "preference_pairs.jsonl")
    tasks_path = tasks_path or infer_tasks_path(run_dir)

    branch_rows, branch_errors = read_jsonl_tolerant(branch_pools_path)
    task_rows = load_task_rows(tasks_path)

    accumulators: dict[tuple[str, str], _TaskAccumulator] = {}
    missing_task_rows: Counter[str] = Counter()
    skipped_task_types: Counter[str] = Counter()

    for row in branch_rows:
        task_type = str(row.get("task_type") or "")
        if task_type not in {"recovery", "refinement"}:
            if task_type:
                skipped_task_types[task_type] += 1
            continue
        source_task_id = _source_task_id(row)
        task_row = task_rows.get(source_task_id)
        target_gene_ids = _target_gene_ids(task_row)
        if not target_gene_ids:
            missing_task_rows[source_task_id] += 1
            continue
        trajectory_id = str(row.get("trajectory_id") or source_task_id)
        key = (trajectory_id, source_task_id)
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = _TaskAccumulator(
                source_task_id=source_task_id,
                trajectory_id=trajectory_id,
                task_type=task_type,
                difficulty=str(row.get("difficulty") or (task_row or {}).get("difficulty") or "unknown"),
                evidence_mode=str(row.get("evidence_mode") or (task_row or {}).get("evidence_mode") or "unknown"),
                target_gene_ids=target_gene_ids,
            )
            accumulators[key] = accumulator
        current_gene_ids = _row_current_gene_ids(row)
        step_index = row.get("step_index")
        accumulator.branch_pool_count += 1
        accumulator.note_initial(step_index if isinstance(step_index, int) else 0, current_gene_ids)
        selected_branch_id = row.get("selected_branch_id")
        branches = row.get("branches")
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            accumulator.add_branch(
                branch,
                current_gene_ids=current_gene_ids,
                selected=branch.get("branch_id") == selected_branch_id,
                prompt_preview_limit=prompt_preview_limit,
            )

    raw_pair_rows, raw_pair_errors = read_jsonl_tolerant(preference_pairs_raw_path)
    balanced_pair_rows, balanced_pair_errors = read_jsonl_tolerant(preference_pairs_path)
    _count_pairs(raw_pair_rows, accumulators, raw=True)
    _count_pairs(balanced_pair_rows, accumulators, raw=False)

    by_task = [
        accumulator.to_dict(max_gene_details=max_gene_details_per_task)
        for accumulator in sorted(accumulators.values(), key=lambda item: (item.task_type, item.source_task_id, item.trajectory_id))
    ]
    aggregate = _aggregate(by_task)
    aggregate.update(
        {
            "branch_pool_parse_error_count": len(branch_errors),
            "preference_pairs_raw_parse_error_count": len(raw_pair_errors),
            "preference_pairs_parse_error_count": len(balanced_pair_errors),
            "missing_task_row_branch_pool_count": sum(missing_task_rows.values()),
            "missing_task_row_source_task_count": len(missing_task_rows),
            "skipped_task_type_counts": dict(sorted(skipped_task_types.items())),
        }
    )

    return {
        "schema_version": "frontier-diagnostics-v1",
        "artifact_role": "post_hoc_hidden_target_diagnostic_not_for_training_export",
        "run_dir": str(run_dir),
        "tasks_path": str(tasks_path) if tasks_path is not None else None,
        "branch_pools_path": str(branch_pools_path),
        "preference_pairs_raw_path": str(preference_pairs_raw_path),
        "preference_pairs_path": str(preference_pairs_path),
        "prompt_preview_limit": prompt_preview_limit,
        "max_gene_details_per_task": max_gene_details_per_task,
        "aggregate": aggregate,
        "by_task": by_task,
        "parse_errors": {
            "branch_pools": branch_errors[:20],
            "preference_pairs_raw": raw_pair_errors[:20],
            "preference_pairs": balanced_pair_errors[:20],
        },
    }


def _pair_category(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        value = provenance.get("pair_category")
        if isinstance(value, str):
            return value
    value = row.get("pair_category")
    return value if isinstance(value, str) else "unknown"


def _pair_is_exact(row: dict[str, Any]) -> bool:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    category = _pair_category(row)
    return (
        category in EXACT_PAIR_CATEGORIES
        or (
            provenance.get("chosen_exact_membership") is True
            and provenance.get("rejected_exact_membership") is not True
        )
    )


def _count_pairs(
    rows: list[dict[str, Any]],
    accumulators: dict[tuple[str, str], _TaskAccumulator],
    *,
    raw: bool,
) -> None:
    by_trajectory: dict[str, list[_TaskAccumulator]] = defaultdict(list)
    for accumulator in accumulators.values():
        by_trajectory[accumulator.trajectory_id].append(accumulator)
    for row in rows:
        trajectory_id = row.get("trajectory_id")
        if not isinstance(trajectory_id, str):
            continue
        for accumulator in by_trajectory.get(trajectory_id, []):
            if raw:
                accumulator.pair_count_raw += 1
                if _pair_is_exact(row):
                    accumulator.exact_pair_count_raw += 1
            else:
                accumulator.pair_count_balanced += 1
                if _pair_is_exact(row):
                    accumulator.exact_pair_count_balanced += 1


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key) or 0) for row in rows)


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    ranks: list[float] = []
    for row in rows:
        for key in ("missing_target_gene_details", "extra_gene_details"):
            details = row.get(key)
            if not isinstance(details, list):
                continue
            for detail in details:
                if isinstance(detail, dict) and isinstance(detail.get("rwr_best_rank"), (int, float)):
                    ranks.append(float(detail["rwr_best_rank"]))
    if not ranks:
        return {"mean": None, "median": None, "max": None}
    return {
        "mean": round(statistics.fmean(ranks), 6),
        "median": round(statistics.median(ranks), 6),
        "max": round(max(ranks), 6),
    }


def _aggregate(by_task: list[dict[str, Any]]) -> dict[str, Any]:
    recovery = [row for row in by_task if row.get("task_type") == "recovery"]
    refinement = [row for row in by_task if row.get("task_type") == "refinement"]
    recovery_missing = _sum(recovery, "recovery_missing_target_gene_count")
    recovery_preview = _sum(recovery, "frontier_surfaced_at_preview_gene_count")
    refinement_extra = _sum(refinement, "refinement_extra_gene_count")
    exact_pair_raw = _sum(by_task, "exact_pair_count_raw")
    exact_pair_balanced = _sum(by_task, "exact_pair_count_balanced")
    return {
        "task_count": len(by_task),
        "recovery_task_count": len(recovery),
        "refinement_task_count": len(refinement),
        "branch_pool_count": _sum(by_task, "branch_pool_count"),
        "branch_count": _sum(by_task, "branch_count"),
        "deterministic_edit_branch_count": _sum(by_task, "deterministic_edit_branch_count"),
        "selected_deterministic_edit_branch_count": _sum(by_task, "selected_deterministic_edit_branch_count"),
        "exact_branch_count": _sum(by_task, "exact_branch_count"),
        "selected_exact_branch_count": _sum(by_task, "selected_exact_branch_count"),
        "exact_pair_count_raw": exact_pair_raw,
        "exact_pair_count_balanced": exact_pair_balanced,
        "recovery_missing_target_gene_count": recovery_missing,
        "recovery_frontier_recalled_gene_count": _sum(recovery, "frontier_recalled_gene_count"),
        "recovery_frontier_surfaced_at_preview_gene_count": recovery_preview,
        "recovery_edit_frontier_gene_count": _sum(recovery, "edit_frontier_gene_count"),
        "recovery_added_to_any_branch_gene_count": _sum(recovery, "added_to_any_branch_gene_count"),
        "recovery_added_to_selected_branch_gene_count": _sum(recovery, "added_to_selected_branch_gene_count"),
        "recovery_visible_but_not_added_gene_count": _sum(recovery, "visible_but_not_added_gene_count"),
        "recovery_frontier_recall_at_topk": _fmt_rate(_rate(_sum(recovery, "frontier_recalled_gene_count"), recovery_missing)),
        "recovery_frontier_surfaced_at_preview": _fmt_rate(_rate(recovery_preview, recovery_missing)),
        "recovery_edit_frontier_coverage": _fmt_rate(_rate(_sum(recovery, "edit_frontier_gene_count"), recovery_missing)),
        "recovery_added_to_any_branch_rate": _fmt_rate(_rate(_sum(recovery, "added_to_any_branch_gene_count"), recovery_missing)),
        "recovery_added_to_selected_branch_rate": _fmt_rate(_rate(_sum(recovery, "added_to_selected_branch_gene_count"), recovery_missing)),
        "recovery_visible_but_not_added_rate": _fmt_rate(_rate(_sum(recovery, "visible_but_not_added_gene_count"), recovery_preview)),
        "refinement_extra_gene_count": refinement_extra,
        "refinement_frontier_flagged_extra_gene_count": _sum(refinement, "frontier_flagged_extra_gene_count"),
        "refinement_removed_by_any_edit_gene_count": _sum(refinement, "removed_by_any_edit_gene_count"),
        "refinement_removed_by_selected_edit_gene_count": _sum(refinement, "removed_by_selected_edit_gene_count"),
        "refinement_selected_branch_pruned_extra_gene_count": _sum(refinement, "selected_branch_pruned_extra_gene_count"),
        "refinement_frontier_flagged_extra_rate": _fmt_rate(_rate(_sum(refinement, "frontier_flagged_extra_gene_count"), refinement_extra)),
        "refinement_removed_by_any_edit_rate": _fmt_rate(_rate(_sum(refinement, "removed_by_any_edit_gene_count"), refinement_extra)),
        "refinement_removed_by_selected_edit_rate": _fmt_rate(_rate(_sum(refinement, "removed_by_selected_edit_gene_count"), refinement_extra)),
        "refinement_selected_branch_pruned_extra_rate": _fmt_rate(_rate(_sum(refinement, "selected_branch_pruned_extra_gene_count"), refinement_extra)),
        "observed_rwr_rank_summary": _rank_summary(by_task),
    }
