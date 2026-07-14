#!/usr/bin/env python3
"""Check whether recovery-task hidden targets are surfaced by runtime tools.

This diagnostic is intentionally separate from trajectory generation. It uses
hidden targets only for offline validation and never writes trajectory artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import ToolAction
from runtime.environment import RuntimeEnvironment


DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "module_corpus_full_brain_mixed" / "tasks.train.jsonl"
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "runtime" / "full_brain_multiplex_store"
DEFAULT_TOP_KS = (50, 100, 250, 500, 1000)


def _load_rows(path: Path, *, max_tasks: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_type") == "recovery":
                rows.append(row)
                if max_tasks is not None and len(rows) >= max_tasks:
                    break
    return rows


def _rank_lookup(results: list[dict[str, Any]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, result in enumerate(results, start=1):
        gene_id = result.get("gene_id")
        if isinstance(gene_id, str) and gene_id and gene_id not in lookup:
            lookup[gene_id] = index
    return lookup


def _score_lookup(results: list[dict[str, Any]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for result in results:
        gene_id = result.get("gene_id")
        score = result.get("score")
        if isinstance(gene_id, str) and isinstance(score, (int, float)):
            lookup[gene_id] = float(score)
    return lookup


def audit_rows(
    rows: list[dict[str, Any]],
    *,
    environment: RuntimeEnvironment,
    top_ks: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    recoverable_by_top_k: Counter[int] = Counter()
    missing_count_distribution: Counter[int] = Counter()

    for row_index, row in enumerate(rows):
        visible_inputs = row.get("visible_inputs", {})
        hidden_target = row.get("hidden_target", {})
        seed_gene_ids = [
            gene_id for gene_id in visible_inputs.get("seed_gene_ids", []) if isinstance(gene_id, str)
        ]
        target_gene_ids = [
            gene_id for gene_id in hidden_target.get("target_gene_ids", []) if isinstance(gene_id, str)
        ]
        missing_target_ids = sorted(set(target_gene_ids) - set(seed_gene_ids))
        missing_count_distribution[len(missing_target_ids)] += 1

        per_top_k: dict[str, Any] = {}
        first_recovered_top_k: int | None = None
        for top_k in top_ks:
            action = ToolAction(
                tool_name="rwr_multiplex",
                arguments={"seeds": seed_gene_ids, "top_k": top_k},
                call_id=f"recoverability.{row_index}.rwr{top_k}",
            )
            observation = environment.execute(action)
            payload = observation.payload or {}
            results = payload.get("results", [])
            result_list = results if isinstance(results, list) else []
            ranks = _rank_lookup(result_list)
            scores = _score_lookup(result_list)
            recovered = [gene_id for gene_id in missing_target_ids if gene_id in ranks]
            if recovered and first_recovered_top_k is None:
                first_recovered_top_k = top_k
            per_top_k[str(top_k)] = {
                "status": observation.status.value,
                "recovered_missing_target_ids": recovered,
                "recovered_missing_target_ranks": {gene_id: ranks[gene_id] for gene_id in recovered},
                "recovered_missing_target_scores": {gene_id: scores.get(gene_id) for gene_id in recovered},
                "top_non_seed_gene_ids": [
                    result.get("gene_id")
                    for result in result_list
                    if isinstance(result, dict) and result.get("gene_id") not in set(seed_gene_ids)
                ][:25],
            }

        if first_recovered_top_k is not None:
            for top_k in top_ks:
                if top_k >= first_recovered_top_k:
                    recoverable_by_top_k[top_k] += 1

        records.append(
            {
                "task_id": row.get("task_id"),
                "evidence_mode": row.get("evidence_mode"),
                "difficulty": row.get("difficulty"),
                "seed_gene_ids": seed_gene_ids,
                "seed_gene_symbols": visible_inputs.get("seed_gene_symbols", []),
                "target_gene_ids": target_gene_ids,
                "target_gene_symbols": hidden_target.get("target_gene_symbols"),
                "missing_target_ids": missing_target_ids,
                "first_recovered_top_k": first_recovered_top_k,
                "rwr": per_top_k,
            }
        )

    total = len(records)
    summary = {
        "task_count": total,
        "top_ks": list(top_ks),
        "missing_count_distribution": {str(key): value for key, value in sorted(missing_count_distribution.items())},
        "recoverable_by_top_k": {
            str(top_k): {
                "count": recoverable_by_top_k[top_k],
                "rate": (recoverable_by_top_k[top_k] / total) if total else 0.0,
            }
            for top_k in top_ks
        },
    }
    return records, summary


def _parse_top_ks(value: str) -> tuple[int, ...]:
    top_ks = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not top_ks or any(top_k <= 0 for top_k in top_ks):
        raise argparse.ArgumentTypeError("top-k values must be positive integers.")
    return top_ks


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recovery-task RWR recoverability.")
    parser.add_argument("--tasks-path", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--compiled-library-path", type=Path, default=None)
    parser.add_argument("--top-ks", type=_parse_top_ks, default=DEFAULT_TOP_KS)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    args = parser.parse_args()

    rows = _load_rows(args.tasks_path, max_tasks=args.max_tasks)
    environment = RuntimeEnvironment(
        store_dir=str(args.store_dir),
        compiled_library_path=str(args.compiled_library_path) if args.compiled_library_path else None,
    )
    records, summary = audit_rows(rows, environment=environment, top_ks=args.top_ks)

    if args.out_jsonl is not None:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")

    print("Recovery recoverability audit")
    print(f"Tasks path: {args.tasks_path}")
    print(f"Recovery tasks: {summary['task_count']}")
    print(f"Missing target count distribution: {summary['missing_count_distribution']}")
    print("Recoverable by RWR top_k:")
    for top_k, payload in summary["recoverable_by_top_k"].items():
        print(f"  top_k={top_k}: {payload['count']} ({payload['rate']:.3f})")
    if args.out_jsonl is not None:
        print(f"Wrote per-task records: {args.out_jsonl}")


if __name__ == "__main__":
    main()
