#!/usr/bin/env python3
"""Build untruncated S2.4 verification artifacts and audit the local context limit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_4_corpus import SPLITS, build_corpus  # noqa: E402


def main() -> None:
    """Require an explicit verification schedule; do not select a training budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke', action='store_true', help='extrema and semantic contrasts; no generation budget')
    budget.add_argument('--queries-per-count', type=int, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'),
                        help='verification queries per attainable count in each split; not a production export')
    args = parser.parse_args()
    manifest = build_corpus(args.output_dir, counts=None if args.smoke else dict(zip(SPLITS, args.queries_per_count)),
                            smoke=args.smoke, seed=args.seed)
    print(json.dumps({k: manifest[k] for k in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                              'training_ready', 'production_export_allowed')}, indent=2))


if __name__ == '__main__':
    main()
