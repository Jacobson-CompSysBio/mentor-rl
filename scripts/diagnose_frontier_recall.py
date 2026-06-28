#!/usr/bin/env python3
"""Write exact-membership frontier diagnostics for a trajectory run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.frontier_diagnostics import (
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROMPT_PREVIEW_LIMIT,
    diagnose_frontier_run,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether exact recovery/refinement targets reached the RWR "
            "frontier, model-visible preview, deterministic edit frontier, "
            "selected branch, and DPO pair export."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Trajectory run directory.")
    parser.add_argument("--tasks-path", type=Path, default=None, help="Optional task JSONL path. Defaults to run metadata inference.")
    parser.add_argument("--branch-pools-path", type=Path, default=None, help="Optional branch_pools.jsonl path.")
    parser.add_argument("--preference-pairs-raw-path", type=Path, default=None, help="Optional preference_pairs_raw.jsonl path.")
    parser.add_argument("--preference-pairs-path", type=Path, default=None, help="Optional balanced preference_pairs.jsonl path.")
    parser.add_argument("--out", type=Path, default=None, help=f"Output JSON path. Defaults to RUN_DIR/{DEFAULT_OUTPUT_NAME}.")
    parser.add_argument(
        "--prompt-preview-limit",
        type=int,
        default=DEFAULT_PROMPT_PREVIEW_LIMIT,
        help="Persisted-RWR rows treated as the model-visible non-seed preview.",
    )
    parser.add_argument(
        "--max-gene-details-per-task",
        type=int,
        default=20,
        help="Maximum hidden-target gene detail rows retained per task in the diagnostic artifact.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_path = args.out or (args.run_dir / DEFAULT_OUTPUT_NAME)
    report = diagnose_frontier_run(
        run_dir=args.run_dir,
        tasks_path=args.tasks_path,
        branch_pools_path=args.branch_pools_path,
        preference_pairs_raw_path=args.preference_pairs_raw_path,
        preference_pairs_path=args.preference_pairs_path,
        prompt_preview_limit=max(args.prompt_preview_limit, 0),
        max_gene_details_per_task=max(args.max_gene_details_per_task, 0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate = report.get("aggregate", {})
    print(
        "Wrote frontier diagnostics to "
        f"{out_path} "
        f"(tasks={aggregate.get('task_count', 0)}, "
        f"recovery_recall={aggregate.get('recovery_frontier_recall_at_topk')}, "
        f"preview={aggregate.get('recovery_frontier_surfaced_at_preview')}, "
        f"exact_branches={aggregate.get('exact_branch_count', 0)}, "
        f"exact_pairs_raw={aggregate.get('exact_pair_count_raw', 0)})"
    )


if __name__ == "__main__":
    main()
