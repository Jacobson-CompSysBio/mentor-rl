#!/usr/bin/env python3
"""Build S2.13 verification traces from an explicit source-pair budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_13_corpus import SPLITS, build_corpus  # noqa: E402


def main() -> None:
    """Require a verification budget without an implicit production budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke', action='store_true', help='use the pinned verification schedule and evidence contrasts')
    budget.add_argument('--pairs-per-stratum', type=int, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'),
                        help='source pairs per attained layer/category stratum in each split')
    args = parser.parse_args()
    if args.pairs_per_stratum is not None and any(count <= 0 for count in args.pairs_per_stratum):
        parser.error('--pairs-per-stratum requires positive integers')
    manifest = build_corpus(args.output_dir,
                            counts=None if args.smoke else dict(zip(SPLITS, args.pairs_per_stratum)),
                            smoke=args.smoke, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                                    'training_ready', 'production_export_allowed')}, indent=2))


if __name__ == '__main__':
    main()
