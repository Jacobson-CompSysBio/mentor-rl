#!/usr/bin/env python3
"""Build the isolated S2.2 multiplex component with an explicit generation budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_2_corpus import build_corpus, SPLITS  # noqa: E402


def main() -> None:
    """Require budget selection or a clearly labeled, one-query-per-cell smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke', action='store_true', help='verification only; leaves generation budget unset')
    budget.add_argument('--queries-per-cell', type=int, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'),
                        help='unique queries per feasible scope/predicate/support-count cell in each split')
    args = parser.parse_args()
    counts = None if args.smoke else dict(zip(SPLITS, args.queries_per_cell))
    manifest = build_corpus(args.output_dir, counts=counts, smoke=args.smoke, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in
                      ('purpose', 'generation_budget', 'row_counts', 'panel_id', 'training_ready')}, indent=2))


if __name__ == '__main__':
    main()
