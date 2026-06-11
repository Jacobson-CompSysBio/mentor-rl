#!/usr/bin/env python3
"""Mix module task corpora while preserving source/task strata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_gw_dendrogram_corpus import deterministic_select_subset, write_json, write_jsonl
from scripts.select_verification_tasks import size_bin_for_task_row, source_for_task_row


DEFAULT_OUT_DIR = REPO_ROOT / "data" / "module_corpus_full_brain_mixed"
DEFAULT_CORPUS_DIRS = (
    REPO_ROOT / "data" / "gw_dendrogram_corpus_full_brain",
    REPO_ROOT / "data" / "rwr_loe_corpus_full_brain",
)
SPLITS = ("train", "val", "test")
SCHEMA_VERSION = "module-corpus-mix-v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest(corpus_dir: Path) -> dict[str, Any]:
    path = corpus_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _row_with_source(row: dict[str, Any], *, corpus_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    provenance = copied.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
        copied["provenance"] = provenance
    source = source_for_task_row(copied)
    if source == "unknown":
        manifest_source = manifest.get("source")
        if isinstance(manifest_source, str) and manifest_source:
            source = manifest_source
    provenance.setdefault("source", source)
    provenance["source_corpus_dir"] = str(corpus_dir)
    return copied


def load_task_rows(corpus_dirs: list[Path]) -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for corpus_dir in corpus_dirs:
        manifest = load_manifest(corpus_dir)
        for split in SPLITS:
            for row in load_jsonl(corpus_dir / f"tasks.{split}.jsonl"):
                rows_by_split[split].append(_row_with_source(row, corpus_dir=corpus_dir, manifest=manifest))
    return rows_by_split


def _mix_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        size_bin_for_task_row(row),
        str(row.get("task_type")),
        str(row.get("evidence_mode")),
        str(row.get("difficulty", "")),
    )


def _source(row: dict[str, Any]) -> str:
    return source_for_task_row(row)


def balance_rows_by_source(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    balance_by_source: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not balance_by_source:
        return sorted(rows, key=lambda row: str(row.get("task_id"))), {"mode": "unbalanced", "input_count": len(rows)}

    grouped: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[_mix_key(row)][_source(row)].append(row)

    selected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for key in sorted(grouped):
        source_rows = grouped[key]
        if len(source_rows) < 2:
            summary["|".join(key)] = {
                "selected_per_source": 0,
                "input_counts_by_source": {source: len(items) for source, items in source_rows.items()},
            }
            continue
        target = min(len(items) for items in source_rows.values())
        summary["|".join(key)] = {
            "selected_per_source": target,
            "input_counts_by_source": {source: len(items) for source, items in source_rows.items()},
        }
        for source in sorted(source_rows):
            rows_for_source = sorted(source_rows[source], key=lambda row: str(row.get("task_id")))
            if len(rows_for_source) > target:
                selected_ids = set(
                    deterministic_select_subset(
                        [str(row.get("task_id")) for row in rows_for_source],
                        subset_size=target,
                        seed=seed,
                        salt=f"mix_module_corpora|{key}|{source}",
                    )
                )
                rows_for_source = [row for row in rows_for_source if str(row.get("task_id")) in selected_ids]
            selected.extend(rows_for_source)
    return sorted(selected, key=lambda row: str(row.get("task_id"))), summary


def copy_auxiliary_rows(
    *,
    corpus_dirs: list[Path],
    file_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_dir in corpus_dirs:
        manifest = load_manifest(corpus_dir)
        source = manifest.get("source")
        for row in load_jsonl(corpus_dir / file_name):
            copied = json.loads(json.dumps(row))
            copied.setdefault("source", source)
            copied["source_corpus_dir"] = str(corpus_dir)
            rows.append(copied)
    return rows


def build_mixed_corpus(
    *,
    corpus_dirs: list[Path],
    out_dir: Path,
    seed: int = 42,
    balance_by_source: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = load_task_rows(corpus_dirs)
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    balance_summary: dict[str, Any] = {}
    for split in SPLITS:
        selected, summary = balance_rows_by_source(
            rows_by_split[split],
            seed=seed,
            balance_by_source=balance_by_source,
        )
        selected_by_split[split] = selected
        balance_summary[split] = summary

    module_rows = copy_auxiliary_rows(corpus_dirs=corpus_dirs, file_name="modules.jsonl")
    prototype_rows = copy_auxiliary_rows(corpus_dirs=corpus_dirs, file_name="prototypes.jsonl")
    all_tasks = [row for split in SPLITS for row in selected_by_split[split]]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": "MIXED_MODULE_CORPUS_FULL_BRAIN",
        "corpus_dirs": [str(path) for path in corpus_dirs],
        "out_dir": str(out_dir),
        "seed": seed,
        "balance_by_source": balance_by_source,
        "task_count": len(all_tasks),
        "task_count_by_split": {split: len(selected_by_split[split]) for split in SPLITS},
        "task_count_by_source": dict(Counter(source_for_task_row(row) for row in all_tasks)),
        "task_count_by_type": dict(Counter(str(row.get("task_type")) for row in all_tasks)),
    }
    split_report = {
        "task_count_by_split_source_type_evidence_difficulty_size": {},
        "balance": balance_summary,
    }
    for split in SPLITS:
        counts: Counter[str] = Counter()
        for row in selected_by_split[split]:
            counts[
                "|".join(
                    [
                        source_for_task_row(row),
                        str(row.get("task_type")),
                        str(row.get("evidence_mode")),
                        str(row.get("difficulty", "")),
                        size_bin_for_task_row(row),
                    ]
                )
            ] += 1
        split_report["task_count_by_split_source_type_evidence_difficulty_size"][split] = dict(sorted(counts.items()))

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "split_report.json", split_report)
    write_jsonl(out_dir / "modules.jsonl", module_rows)
    write_jsonl(out_dir / "prototypes.jsonl", prototype_rows)
    for split in SPLITS:
        write_jsonl(out_dir / f"tasks.{split}.jsonl", selected_by_split[split])

    return {
        "manifest": manifest,
        "split_report": split_report,
        "tasks": all_tasks,
        "modules": module_rows,
        "prototypes": prototype_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix module corpora for full-brain training/testing.")
    parser.add_argument("--corpus-dir", type=Path, action="append", dest="corpus_dirs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balance-by-source", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dirs = args.corpus_dirs or list(DEFAULT_CORPUS_DIRS)
    result = build_mixed_corpus(
        corpus_dirs=corpus_dirs,
        out_dir=args.out_dir,
        seed=args.seed,
        balance_by_source=not args.no_balance_by_source,
    )
    summary = {
        "manifest_path": str(args.out_dir / "manifest.json"),
        "task_count": result["manifest"]["task_count"],
        "task_count_by_source": result["manifest"]["task_count_by_source"],
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
