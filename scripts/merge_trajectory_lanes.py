#!/usr/bin/env python3
"""Merge per-node trajectory lanes from one fat Frontier allocation."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

ARTIFACTS = [
    "branch_pools.jsonl",
    "trajectory_turns.jsonl",
    "finding_records.jsonl",
    "preference_pairs_raw.jsonl",
    "preference_pairs.jsonl",
    "final_summaries.jsonl",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def lane_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix), path.name
    return 10**12, path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-lanes", type=int, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root
    src = run_root / "trajectories"
    merged = run_root / "trajectories_merged"
    merged.mkdir(parents=True, exist_ok=True)

    completed: list[Path] = []
    incomplete: list[tuple[str, str]] = []
    for lane_dir in sorted(src.glob("node_*"), key=lane_sort_key):
        progress_path = lane_dir / "progress.json"
        status = "missing_progress"
        if progress_path.exists():
            try:
                status = str(read_json(progress_path).get("status"))
            except Exception as exc:  # pragma: no cover
                status = f"invalid_progress:{exc}"
        if status == "completed":
            completed.append(lane_dir)
        else:
            incomplete.append((lane_dir.name, status))

    if len(completed) < args.expected_lanes and not args.allow_partial:
        raise SystemExit(
            f"only {len(completed)}/{args.expected_lanes} completed lanes; incomplete={incomplete}"
        )
    if not completed:
        raise SystemExit("no completed lanes to merge")

    counts: dict[str, int] = {}
    for name in ARTIFACTS:
        out = merged / name
        with out.open("wb") as writer:
            for lane_dir in completed:
                path = lane_dir / name
                if path.exists() and path.stat().st_size:
                    data = path.read_bytes()
                    writer.write(data)
                    if data and not data.endswith(b"\n"):
                        writer.write(b"\n")
        counts[name] = count_lines(out)

    tasks_selected = run_root / "inputs" / "tasks_selected.jsonl"
    tasks_out = merged / "tasks_merged.jsonl"
    if tasks_selected.exists():
        tasks_out.write_bytes(tasks_selected.read_bytes())

    first_manifest = None
    for lane_dir in completed:
        path = lane_dir / "manifest.json"
        if path.exists():
            first_manifest = read_json(path)
            break
    if first_manifest is None:
        raise SystemExit("completed lanes have no manifest.json")

    total_branches = 0
    branch_path = merged / "branch_pools.jsonl"
    with branch_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                branches = row.get("branches")
                if isinstance(branches, list):
                    total_branches += len(branches)

    manifest = dict(first_manifest)
    manifest["num_trajectories"] = counts["final_summaries.jsonl"]
    manifest["task_count"] = counts["final_summaries.jsonl"]
    manifest["total_branch_pools"] = counts["branch_pools.jsonl"]
    manifest["total_steps"] = counts["trajectory_turns.jsonl"]
    manifest["total_branches"] = total_branches
    manifest["merged_fat_allocation"] = {
        "source_run_root": str(run_root),
        "source_lanes": [path.name for path in completed],
        "incomplete_lanes": [{"lane": name, "status": status} for name, status in incomplete],
        "merged_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    task_selection = dict(manifest.get("task_selection") or {})
    task_selection.update(
        {
            "tasks_path": str(tasks_out) if tasks_out.exists() else None,
            "num_tasks": counts["final_summaries.jsonl"],
            "source_task_file": str(tasks_selected) if tasks_selected.exists() else None,
            "source_lanes": [str(path) for path in completed],
        }
    )
    manifest["task_selection"] = task_selection
    artifacts = dict(manifest.get("artifacts") or {})
    artifacts.update(
        {
            "finding_record_count": counts["finding_records.jsonl"],
            "preference_pair_raw_count": counts["preference_pairs_raw.jsonl"],
            "preference_pair_count": counts["preference_pairs.jsonl"],
        }
    )
    manifest["artifacts"] = artifacts
    outputs = dict(manifest.get("outputs") or {})
    outputs.update({name.removesuffix(".jsonl"): str(merged / name) for name in ARTIFACTS})
    manifest["outputs"] = outputs
    (merged / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    started_values = []
    for path in completed:
        try:
            started_values.append(read_json(path / "progress.json").get("started_at", ""))
        except Exception:
            pass
    progress = {
        "status": "completed" if len(completed) >= args.expected_lanes else "partial",
        "current_stage": "completed" if len(completed) >= args.expected_lanes else "partial_merge",
        "current_stage_label": "Completed" if len(completed) >= args.expected_lanes else "Partial merge",
        "message": "Merged completed fat-allocation trajectory lanes.",
        "overall_progress": 1.0 if len(completed) >= args.expected_lanes else len(completed) / max(args.expected_lanes, 1),
        "stage_index": 4,
        "stage_count": 4,
        "started_at": min(started_values) if started_values else "",
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": {
            "num_trajectories": counts["final_summaries.jsonl"],
            "completed_lanes": len(completed),
            "expected_lanes": args.expected_lanes,
            "total_steps": counts["trajectory_turns.jsonl"],
            "total_branch_pools": counts["branch_pools.jsonl"],
            "total_branches": total_branches,
            "total_preference_pairs_raw": counts["preference_pairs_raw.jsonl"],
            "total_preference_pairs": counts["preference_pairs.jsonl"],
        },
    }
    (merged / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for extra in ["served_models.json", "run_freeze.json"]:
        for lane_dir in completed:
            path = lane_dir / extra
            if path.exists():
                (merged / extra).write_bytes(path.read_bytes())
                break

    lane_logs = merged / "lane_logs"
    lane_logs.mkdir(parents=True, exist_ok=True)
    copied_server_log = False
    for lane_dir in completed:
        log_dst = lane_logs / lane_dir.name
        log_dst.mkdir(parents=True, exist_ok=True)
        for relative in [
            Path("vllm_server.log"),
            Path("logs/lane.out"),
            Path("logs/lane.err"),
            Path("progress.json"),
        ]:
            src_path = lane_dir / relative
            if not src_path.exists():
                continue
            dst_path = log_dst / relative.name
            shutil.copy2(src_path, dst_path)
            if relative == Path("vllm_server.log") and not copied_server_log:
                shutil.copy2(src_path, merged / "vllm_server.log")
                copied_server_log = True

    with (merged / "merge_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write(f"run_root={run_root}\n")
        handle.write(f"completed_lanes\t{len(completed)}\n")
        handle.write(f"expected_lanes\t{args.expected_lanes}\n")
        for name in ARTIFACTS:
            handle.write(f"{name}\t{counts[name]}\n")
        for name, status in incomplete:
            handle.write(f"incomplete\t{name}\t{status}\n")

    print((merged / "merge_counts.tsv").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
