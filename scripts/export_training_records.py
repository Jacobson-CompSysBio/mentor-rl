#!/usr/bin/env python3
"""Export completed trajectory runs into DPO and SFT training records."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import PreferencePair
from runtime.validators import validate_preference_pair
from scripts.dpo_pair_loader_smoke import render_dpo_record


REQUIRED_RUN_ARTIFACTS = (
    "manifest.json",
    "progress.json",
    "final_summaries.jsonl",
    "branch_pools.jsonl",
    "preference_pairs_raw.jsonl",
    "preference_pairs.jsonl",
)
EXACT_SFT_TASK_TYPES = {"recovery", "refinement"}
SFT_SYSTEM_PROMPT = (
    "You are Mentor-RL. Produce a concise mechanistic trajectory summary from "
    "visible evidence only; do not infer hidden targets."
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path.name}:{line_number}.")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _default_export_dir(run_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return run_dir / "training_exports" / f"export_{stamp}"


def _check_completed_run(run_dir: Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run is missing required artifacts: {', '.join(missing)}")

    manifest = _read_json(run_dir / "manifest.json")
    progress = _read_json(run_dir / "progress.json")
    status = str(progress.get("status", "")).lower()
    if status != "completed" and not allow_incomplete:
        raise ValueError(
            f"Run progress status is {progress.get('status')!r}; pass --allow-incomplete to export anyway."
        )
    return {"manifest": manifest, "progress": progress}


def _load_valid_pairs(path: Path) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                pair = PreferencePair.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Failed to parse {path.name}:{line_number}: {exc}") from exc
            validation = validate_preference_pair(pair)
            if not validation.valid:
                raise ValueError(
                    f"Invalid pair at {path.name}:{line_number}: {'; '.join(validation.errors)}"
                )
            pairs.append(pair)
    return pairs


def _summary_question(row: dict[str, Any]) -> str:
    final_state = row.get("final_state")
    if isinstance(final_state, dict):
        user_anchors = final_state.get("user_anchors")
        if isinstance(user_anchors, dict):
            query_text = user_anchors.get("query_text")
            if isinstance(query_text, str) and query_text.strip():
                return query_text.strip()
    source_task_id = row.get("source_task_id")
    if isinstance(source_task_id, str) and source_task_id.strip():
        return source_task_id.strip()
    return "Summarize the completed Mentor-RL trajectory."


def _sft_record_from_summary(row: dict[str, Any]) -> dict[str, Any]:
    metadata_keys = (
        "trajectory_id",
        "source_task_id",
        "task_type",
        "difficulty",
        "evidence_mode",
        "trajectory_seed",
        "step_count",
        "selected_branch_ids",
        "task_success_level",
        "terminal_reward",
        "terminal_schema_score",
        "terminal_absolute_complex_score",
        "terminal_absolute_mechanistic_score",
        "terminal_mechanism_evidence_score",
    )
    return {
        "system": SFT_SYSTEM_PROMPT,
        "question": _summary_question(row),
        "answer": str(row.get("rendered_summary", "")).strip(),
        "metadata": {key: row.get(key) for key in metadata_keys if key in row},
    }


def _eligible_sft_summary(row: dict[str, Any], *, include_partial: bool) -> bool:
    if row.get("task_type") not in EXACT_SFT_TASK_TYPES:
        return False
    if str(row.get("task_success_level")) == "positive":
        return True
    return include_partial and str(row.get("task_success_level")) == "partial"


def _branch_metadata_from_turn(row: dict[str, Any]) -> tuple[str | None, str | None, dict[str, Any]]:
    trajectory_id = row.get("trajectory_id")
    branch = row.get("branch")
    if not isinstance(branch, dict):
        return (trajectory_id if isinstance(trajectory_id, str) else None, None, {})
    branch_id = branch.get("branch_id")
    metadata = branch.get("metadata")
    return (
        trajectory_id if isinstance(trajectory_id, str) else None,
        branch_id if isinstance(branch_id, str) else None,
        metadata if isinstance(metadata, dict) else {},
    )


def _metadata_is_scaffolded_membership_edit(metadata: dict[str, Any]) -> bool:
    edit = metadata.get("deterministic_membership_edit")
    return (
        isinstance(edit, dict)
        and (
            edit.get("requires_model_validation") is True
            or metadata.get("scaffolded_membership_edit_requires_validation") is True
        )
    )


def _scaffolded_selected_branch_ids_by_trajectory(run_dir: Path) -> dict[str, set[str]]:
    path = run_dir / "trajectory_turns.jsonl"
    if not path.exists():
        return {}
    branch_ids_by_trajectory: dict[str, set[str]] = {}
    for row in _iter_jsonl(path):
        trajectory_id, branch_id, metadata = _branch_metadata_from_turn(row)
        if trajectory_id is None or branch_id is None:
            continue
        if _metadata_is_scaffolded_membership_edit(metadata):
            branch_ids_by_trajectory.setdefault(trajectory_id, set()).add(branch_id)
    return branch_ids_by_trajectory


def _summary_ends_on_scaffolded_membership_edit(
    row: dict[str, Any],
    scaffolded_branch_ids_by_trajectory: dict[str, set[str]],
) -> bool:
    trajectory_id = row.get("trajectory_id")
    if not isinstance(trajectory_id, str):
        return False
    scaffolded_branch_ids = scaffolded_branch_ids_by_trajectory.get(trajectory_id, set())
    selected_branch_ids = row.get("selected_branch_ids")
    if not isinstance(selected_branch_ids, list) or not selected_branch_ids:
        return False
    terminal_branch_id = selected_branch_ids[-1]
    return isinstance(terminal_branch_id, str) and terminal_branch_id in scaffolded_branch_ids


def export_training_records(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    include_partial: bool = False,
    allow_incomplete: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export a completed trajectory run into DPO and exact-positive SFT JSONL files."""

    run_dir = run_dir.resolve()
    run_info = _check_completed_run(run_dir, allow_incomplete=allow_incomplete)
    out_dir = (out_dir or _default_export_dir(run_dir)).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Export directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_valid_pairs(run_dir / "preference_pairs.jsonl")
    dpo_records = [render_dpo_record(pair) for pair in pairs]
    summaries = _iter_jsonl(run_dir / "final_summaries.jsonl")
    scaffolded_branch_ids_by_trajectory = _scaffolded_selected_branch_ids_by_trajectory(run_dir)
    sft_eligible_summaries = [
        row
        for row in summaries
        if _eligible_sft_summary(row, include_partial=include_partial)
    ]
    sft_scaffolded_excluded_count = sum(
        1
        for row in sft_eligible_summaries
        if _summary_ends_on_scaffolded_membership_edit(row, scaffolded_branch_ids_by_trajectory)
    )
    sft_records = [
        _sft_record_from_summary(row)
        for row in sft_eligible_summaries
        if not _summary_ends_on_scaffolded_membership_edit(row, scaffolded_branch_ids_by_trajectory)
    ]

    dpo_path = out_dir / "dpo_records.jsonl"
    sft_path = out_dir / "sft_exact_trajectories.jsonl"
    _write_jsonl(dpo_path, dpo_records)
    _write_jsonl(sft_path, sft_records)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "source_manifest": str(run_dir / "manifest.json"),
        "source_progress_status": run_info["progress"].get("status"),
        "include_partial_sft": include_partial,
        "dpo_record_count": len(dpo_records),
        "sft_record_count": len(sft_records),
        "sft_scaffolded_membership_edit_excluded_count": sft_scaffolded_excluded_count,
        "outputs": {
            "dpo_records": str(dpo_path),
            "sft_exact_trajectories": str(sft_path),
        },
        "source_metrics": run_info["manifest"].get("artifacts", {}),
    }
    manifest_path = out_dir / "training_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Mentor-RL training records from a trajectory run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed trajectory run directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Fresh export directory. Defaults to RUN_DIR/training_exports/export_<utc timestamp>.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include partial recovery/refinement trajectories in the SFT export.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow export when progress.json status is not completed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty export directory.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = export_training_records(
        args.run_dir,
        args.out_dir,
        include_partial=args.include_partial,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
