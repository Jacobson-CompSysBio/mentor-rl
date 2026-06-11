#!/usr/bin/env python3
"""Select stratified task subsets for trajectory-generation verification.

Use this before large model-backed runs to build a small pilot task file and a
representative set of smoke-test task ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "gw_dendrogram_corpus" / "tasks.train.jsonl"
DEFAULT_PILOT_DIR = REPO_ROOT / "data" / "gw_dendrogram_corpus" / "pilots"
DEFAULT_SELECTION_SEED = 42
TASK_TYPE_ORDER = ("explanation", "recovery", "refinement", "none")
EVIDENCE_MODE_ORDER = ("graph", "minimal", "contextual", "full")
DIFFICULTY_ORDER = ("complete", "easy", "medium", "hard")
SIZE_BIN_ORDER = ("small", "medium", "large", "other", "unknown")
SOURCE_ORDER = ("MENTOR_GW_DENDROGRAM", "RWR_LOE_FULL_BRAIN", "CORUM", "unknown")
COMPLEX_ID_RE = re.compile(r"(?:corum_complex|gw_dendrogram_module|rwr_loe_module)_\d+")


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


def size_bin_for_task_row(row: dict[str, Any]) -> str:
    """Infer the dendrogram size bin used for representative pilot sampling."""

    explicit_size_bin = row.get("size_bin")
    if isinstance(explicit_size_bin, str) and explicit_size_bin:
        return explicit_size_bin

    hidden_target = row.get("hidden_target")
    target_gene_ids = None
    if isinstance(hidden_target, dict):
        target_gene_ids = hidden_target.get("target_gene_ids")

    gene_count = None
    if isinstance(target_gene_ids, list) and target_gene_ids:
        gene_count = len({gene_id for gene_id in target_gene_ids if isinstance(gene_id, str)})
    else:
        visible_inputs = row.get("visible_inputs")
        if isinstance(visible_inputs, dict):
            seed_gene_ids = visible_inputs.get("seed_gene_ids")
            if isinstance(seed_gene_ids, list) and seed_gene_ids:
                gene_count = len({gene_id for gene_id in seed_gene_ids if isinstance(gene_id, str)})

    if gene_count is None:
        return "unknown"
    if 5 <= gene_count <= 10:
        return "small"
    if 11 <= gene_count <= 15:
        return "medium"
    if gene_count >= 16:
        return "large"
    return "other"


def source_for_task_row(row: dict[str, Any]) -> str:
    """Infer the corpus source used for optional mixed-corpus stratification."""

    explicit_source = row.get("source")
    if isinstance(explicit_source, str) and explicit_source:
        return explicit_source

    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        provenance_source = provenance.get("source")
        if isinstance(provenance_source, str) and provenance_source:
            return provenance_source

    task_id = str(row.get("task_id", ""))
    if task_id.startswith("gw_dendrogram_module_"):
        return "MENTOR_GW_DENDROGRAM"
    if task_id.startswith("rwr_loe_module_"):
        return "RWR_LOE_FULL_BRAIN"
    if task_id.startswith("corum_complex_"):
        return "CORUM"
    return "unknown"


def _ordered_sources(rows: list[dict[str, Any]]) -> list[str]:
    present = {source_for_task_row(row) for row in rows}
    ordered = [source for source in SOURCE_ORDER if source in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _bucket_key(
    row: dict[str, Any],
    *,
    stratify_by_size_bin: bool = True,
    stratify_by_source: bool = False,
) -> tuple[str, ...]:
    key = [str(row.get("task_type")), str(row.get("evidence_mode"))]
    if stratify_by_size_bin:
        key.insert(0, size_bin_for_task_row(row))
    if stratify_by_source:
        key.insert(0, source_for_task_row(row))
    return tuple(key)


def _pilot_bucket_key(
    row: dict[str, Any],
    *,
    stratify_by_difficulty: bool,
    stratify_by_size_bin: bool = True,
    stratify_by_source: bool = False,
) -> tuple[str, ...]:
    task_type = str(row.get("task_type"))
    evidence_mode = str(row.get("evidence_mode"))
    key = [task_type, evidence_mode]
    if stratify_by_size_bin:
        key.insert(0, size_bin_for_task_row(row))
    if stratify_by_source:
        key.insert(0, source_for_task_row(row))
    if not stratify_by_difficulty:
        return tuple(key)
    key.append(str(row.get("difficulty", "")))
    return tuple(key)


def _ordered_smoke_bucket_keys(
    rows: list[dict[str, Any]],
    *,
    stratify_by_size_bin: bool = True,
    stratify_by_source: bool = False,
) -> list[tuple[str, ...]]:
    present = {
        _bucket_key(
            row,
            stratify_by_size_bin=stratify_by_size_bin,
            stratify_by_source=stratify_by_source,
        )
        for row in rows
    }
    if stratify_by_source and stratify_by_size_bin:
        ordered = [
            (source, size_bin, task_type, evidence_mode)
            for source in _ordered_sources(rows)
            for size_bin in SIZE_BIN_ORDER
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            if (source, size_bin, task_type, evidence_mode) in present
        ]
    elif stratify_by_source:
        ordered = [
            (source, task_type, evidence_mode)
            for source in _ordered_sources(rows)
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            if (source, task_type, evidence_mode) in present
        ]
    elif stratify_by_size_bin:
        ordered = [
            (size_bin, task_type, evidence_mode)
            for size_bin in SIZE_BIN_ORDER
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            if (size_bin, task_type, evidence_mode) in present
        ]
    else:
        ordered = [
            (task_type, evidence_mode)
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            if (task_type, evidence_mode) in present
        ]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _ordered_pilot_bucket_keys(
    rows: list[dict[str, Any]],
    *,
    stratify_by_difficulty: bool,
    stratify_by_size_bin: bool = True,
    stratify_by_source: bool = False,
) -> list[tuple[str, ...]]:
    present = {
        _pilot_bucket_key(
            row,
            stratify_by_difficulty=stratify_by_difficulty,
            stratify_by_size_bin=stratify_by_size_bin,
            stratify_by_source=stratify_by_source,
        )
        for row in rows
    }
    if not stratify_by_difficulty:
        return list(
            _ordered_smoke_bucket_keys(
                rows,
                stratify_by_size_bin=stratify_by_size_bin,
                stratify_by_source=stratify_by_source,
            )
        )

    if stratify_by_source and stratify_by_size_bin:
        ordered = [
            (source, size_bin, task_type, evidence_mode, difficulty)
            for source in _ordered_sources(rows)
            for size_bin in SIZE_BIN_ORDER
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            for difficulty in DIFFICULTY_ORDER
            if (source, size_bin, task_type, evidence_mode, difficulty) in present
        ]
    elif stratify_by_source:
        ordered = [
            (source, task_type, evidence_mode, difficulty)
            for source in _ordered_sources(rows)
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            for difficulty in DIFFICULTY_ORDER
            if (source, task_type, evidence_mode, difficulty) in present
        ]
    elif stratify_by_size_bin:
        ordered = [
            (size_bin, task_type, evidence_mode, difficulty)
            for size_bin in SIZE_BIN_ORDER
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            for difficulty in DIFFICULTY_ORDER
            if (size_bin, task_type, evidence_mode, difficulty) in present
        ]
    else:
        ordered = [
            (task_type, evidence_mode, difficulty)
            for task_type in TASK_TYPE_ORDER
            for evidence_mode in EVIDENCE_MODE_ORDER
            for difficulty in DIFFICULTY_ORDER
            if (task_type, evidence_mode, difficulty) in present
        ]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _stable_selection_sort_key(row: dict[str, Any], *, seed: int) -> tuple[str, str]:
    task_id = str(row.get("task_id"))
    complex_id = _complex_key(task_id)
    digest = hashlib.sha256(f"{seed}|{complex_id}|{task_id}".encode("utf-8")).hexdigest()
    return digest, task_id


def _group_rows(
    rows: list[dict[str, Any]],
    *,
    bucket_by_difficulty: bool = False,
    bucket_by_size_bin: bool = True,
    bucket_by_source: bool = False,
    seed: int = DEFAULT_SELECTION_SEED,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bucket_by_difficulty:
            key = _pilot_bucket_key(
                row,
                stratify_by_difficulty=True,
                stratify_by_size_bin=bucket_by_size_bin,
                stratify_by_source=bucket_by_source,
            )
        else:
            key = _bucket_key(
                row,
                stratify_by_size_bin=bucket_by_size_bin,
                stratify_by_source=bucket_by_source,
            )
        grouped[key].append(row)
    for bucket_rows in grouped.values():
        bucket_rows.sort(key=lambda row: _stable_selection_sort_key(row, seed=seed))
    return grouped


def select_smoke_task_ids(
    rows: list[dict[str, Any]],
    *,
    per_bucket: int = 1,
    seed: int = DEFAULT_SELECTION_SEED,
    stratify_by_size_bin: bool = True,
    stratify_by_source: bool = False,
) -> list[str]:
    grouped = _group_rows(
        rows,
        seed=seed,
        bucket_by_size_bin=stratify_by_size_bin,
        bucket_by_source=stratify_by_source,
    )
    selected: list[str] = []
    seen: set[str] = set()
    for key in _ordered_smoke_bucket_keys(
        rows,
        stratify_by_size_bin=stratify_by_size_bin,
        stratify_by_source=stratify_by_source,
    ):
        for row in grouped[key][:per_bucket]:
            task_id = str(row.get("task_id"))
            if task_id not in seen:
                seen.add(task_id)
                selected.append(task_id)
    return selected


def select_pilot_rows(
    rows: list[dict[str, Any]],
    *,
    pilot_size: int,
    seed: int = DEFAULT_SELECTION_SEED,
    stratify_by_difficulty: bool = True,
    stratify_by_size_bin: bool = True,
    stratify_by_source: bool = False,
) -> list[dict[str, Any]]:
    if pilot_size <= 0:
        return []

    grouped = _group_rows(
        rows,
        bucket_by_difficulty=stratify_by_difficulty,
        bucket_by_size_bin=stratify_by_size_bin,
        bucket_by_source=stratify_by_source,
        seed=seed,
    )
    ordered_keys = _ordered_pilot_bucket_keys(
        rows,
        stratify_by_difficulty=stratify_by_difficulty,
        stratify_by_size_bin=stratify_by_size_bin,
        stratify_by_source=stratify_by_source,
    )
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


def write_standard_pilots(
    rows: list[dict[str, Any]],
    *,
    pilot_dir: Path = DEFAULT_PILOT_DIR,
    seed: int = DEFAULT_SELECTION_SEED,
    stratify_by_source: bool = False,
) -> dict[str, Any]:
    """Write the 24-task smoke and 60-task quality pilot JSONLs."""

    smoke_rows = select_pilot_rows(
        rows,
        pilot_size=24,
        seed=seed,
        stratify_by_difficulty=False,
        stratify_by_size_bin=True,
        stratify_by_source=stratify_by_source,
    )
    quality_rows = select_pilot_rows(
        rows,
        pilot_size=60,
        seed=seed,
        stratify_by_difficulty=True,
        stratify_by_size_bin=True,
        stratify_by_source=stratify_by_source,
    )
    smoke_path = pilot_dir / f"stratified_24_seed{seed}.tasks.jsonl"
    quality_path = pilot_dir / f"stratified_60_seed{seed}.tasks.jsonl"
    _write_jsonl(smoke_path, smoke_rows)
    _write_jsonl(quality_path, quality_rows)
    return {
        "smoke24_path": str(smoke_path),
        "quality60_path": str(quality_path),
        "smoke24_count": len(smoke_rows),
        "quality60_count": len(quality_rows),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select verification smoke and pilot task subsets.")
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--pilot-size", type=int, default=60)
    parser.add_argument("--smoke-per-bucket", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--no-pilot-difficulty-strata",
        action="store_true",
        help="Keep pilot stratification at task-type/evidence-mode only.",
    )
    parser.add_argument(
        "--no-size-bin-strata",
        action="store_true",
        help="Disable dendrogram size-bin stratification.",
    )
    parser.add_argument(
        "--source-strata",
        action="store_true",
        help="Stratify smoke and pilot selections by corpus source for mixed corpora.",
    )
    parser.add_argument(
        "--write-standard-pilots",
        action="store_true",
        help="Write deterministic 24-task smoke and 60-task quality pilot JSONLs.",
    )
    parser.add_argument("--standard-pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--pilot-out", type=Path, default=None, help="Optional path for the selected pilot task JSONL.")
    parser.add_argument("--smoke-task-ids-out", type=Path, default=None, help="Optional path for selected smoke task ids.")
    parser.add_argument("--json", action="store_true", help="Print selection metadata as JSON.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    rows = _load_rows(args.tasks_path)
    stratify_by_difficulty = not args.no_pilot_difficulty_strata
    stratify_by_size_bin = not args.no_size_bin_strata
    stratify_by_source = args.source_strata
    smoke_ids = select_smoke_task_ids(
        rows,
        per_bucket=args.smoke_per_bucket,
        seed=args.seed,
        stratify_by_size_bin=stratify_by_size_bin,
        stratify_by_source=stratify_by_source,
    )
    pilot_rows = select_pilot_rows(
        rows,
        pilot_size=args.pilot_size,
        seed=args.seed,
        stratify_by_difficulty=stratify_by_difficulty,
        stratify_by_size_bin=stratify_by_size_bin,
        stratify_by_source=stratify_by_source,
    )
    standard_pilots = None
    if args.write_standard_pilots:
        standard_pilots = write_standard_pilots(
            rows,
            pilot_dir=args.standard_pilot_dir,
            seed=args.seed,
            stratify_by_source=stratify_by_source,
        )

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
        "selection_seed": args.seed,
        "pilot_stratified_by_difficulty": stratify_by_difficulty,
        "stratified_by_size_bin": stratify_by_size_bin,
        "stratified_by_source": stratify_by_source,
        "pilot_bucket_counts": {},
        "standard_pilots": standard_pilots,
        "pilot_out": str(args.pilot_out) if args.pilot_out else None,
        "smoke_task_ids_out": str(args.smoke_task_ids_out) if args.smoke_task_ids_out else None,
    }
    bucket_counts: dict[str, int] = defaultdict(int)
    for row in pilot_rows:
        key = _pilot_bucket_key(
            row,
            stratify_by_difficulty=stratify_by_difficulty,
            stratify_by_size_bin=stratify_by_size_bin,
            stratify_by_source=stratify_by_source,
        )
        bucket_counts["/".join(key)] += 1
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
        if standard_pilots:
            print("Standard pilot files:")
            for key, value in standard_pilots.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
