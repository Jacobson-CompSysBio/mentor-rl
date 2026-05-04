#!/usr/bin/env python3
"""Select stratified task subsets for trajectory-generation verification.

Use this before large model-backed runs to build a small pilot task file and a
representative set of smoke-test task ids.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "corum_corpus" / "tasks.train.jsonl"
TASK_TYPE_ORDER = ("explanation", "recovery", "refinement", "none")
EVIDENCE_MODE_ORDER = ("contextual", "full", "graph", "minimal")
COMPLEX_ID_RE = re.compile(r"corum_complex_\d+")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _complex_key(task_id: str) -> str:
    match = COMPLEX_ID_RE.search(task_id)
    if match:
        return match.group(0)
    return task_id.split(".", 1)[0]


def _bucket_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("task_type")), str(row.get("evidence_mode"))


def _ordered_bucket_keys(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    present = {_bucket_key(row) for row in rows}
    ordered = [
        (task_type, evidence_mode)
        for task_type in TASK_TYPE_ORDER
        for evidence_mode in EVIDENCE_MODE_ORDER
        if (task_type, evidence_mode) in present
    ]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_bucket_key(row)].append(row)
    for bucket_rows in grouped.values():
        bucket_rows.sort(key=lambda row: (_complex_key(str(row.get("task_id"))), str(row.get("task_id"))))
    return grouped


def select_smoke_task_ids(rows: list[dict[str, Any]], *, per_bucket: int = 1) -> list[str]:
    grouped = _group_rows(rows)
    selected: list[str] = []
    seen: set[str] = set()
    for key in _ordered_bucket_keys(rows):
        for row in grouped[key][:per_bucket]:
            task_id = str(row.get("task_id"))
            if task_id not in seen:
                seen.add(task_id)
                selected.append(task_id)
    return selected


def select_pilot_rows(rows: list[dict[str, Any]], *, pilot_size: int) -> list[dict[str, Any]]:
    if pilot_size <= 0:
        return []

    grouped = _group_rows(rows)
    ordered_keys = _ordered_bucket_keys(rows)
    selected: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    bucket_offsets = {key: 0 for key in ordered_keys}

    while len(selected) < pilot_size:
        made_progress = False
        for key in ordered_keys:
            bucket_rows = grouped[key]
            offset = bucket_offsets[key]
            while offset < len(bucket_rows) and str(bucket_rows[offset].get("task_id")) in seen_task_ids:
                offset += 1
            bucket_offsets[key] = offset
            if offset >= len(bucket_rows):
                continue
            row = bucket_rows[offset]
            selected.append(row)
            seen_task_ids.add(str(row.get("task_id")))
            bucket_offsets[key] += 1
            made_progress = True
            if len(selected) >= pilot_size:
                break
        if not made_progress:
            break
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select verification smoke and pilot task subsets.")
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--pilot-size", type=int, default=128)
    parser.add_argument("--smoke-per-bucket", type=int, default=1)
    parser.add_argument("--pilot-out", type=Path, default=None, help="Optional path for the selected pilot task JSONL.")
    parser.add_argument("--smoke-task-ids-out", type=Path, default=None, help="Optional path for selected smoke task ids.")
    parser.add_argument("--json", action="store_true", help="Print selection metadata as JSON.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    rows = _load_rows(args.tasks_path)
    smoke_ids = select_smoke_task_ids(rows, per_bucket=args.smoke_per_bucket)
    pilot_rows = select_pilot_rows(rows, pilot_size=args.pilot_size)

    if args.pilot_out is not None:
        _write_jsonl(args.pilot_out, pilot_rows)
    if args.smoke_task_ids_out is not None:
        _write_lines(args.smoke_task_ids_out, smoke_ids)

    summary = {
        "tasks_path": str(args.tasks_path),
        "total_input_tasks": len(rows),
        "smoke_task_count": len(smoke_ids),
        "smoke_task_ids": smoke_ids,
        "pilot_task_count": len(pilot_rows),
        "pilot_bucket_counts": {},
        "pilot_out": str(args.pilot_out) if args.pilot_out else None,
        "smoke_task_ids_out": str(args.smoke_task_ids_out) if args.smoke_task_ids_out else None,
    }
    bucket_counts: dict[str, int] = defaultdict(int)
    for row in pilot_rows:
        task_type, evidence_mode = _bucket_key(row)
        bucket_counts[f"{task_type}/{evidence_mode}"] += 1
    summary["pilot_bucket_counts"] = dict(sorted(bucket_counts.items()))

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Selected {len(smoke_ids)} smoke tasks and {len(pilot_rows)} pilot tasks.")
        print("Smoke task ids:")
        for task_id in smoke_ids:
            print(f"  {task_id}")
        print("Pilot bucket counts:")
        for key, count in summary["pilot_bucket_counts"].items():
            print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
