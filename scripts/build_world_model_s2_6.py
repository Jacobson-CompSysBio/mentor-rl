#!/usr/bin/env python3
"""Build complete S2.6 verification traces for an explicit finite input schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_6_corpus import SPLITS, build_corpus  # noqa: E402


def main() -> None:
    """Require selected sizes with a budget and retain every selected input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke', action='store_true', help='use the pinned verification sizes and semantic contrasts')
    budget.add_argument('--requests-per-stratum', type=int, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'),
                        help='distinct input sets per feasible (layer,size,category-count triple) in each split')
    parser.add_argument('--input-sizes', type=int, nargs='+', metavar='N',
                        help='explicit finite list sizes, including zero; required with --requests-per-stratum')
    args = parser.parse_args()
    if args.smoke and args.input_sizes is not None:
        parser.error('--smoke uses the pinned size schedule; omit --input-sizes')
    if not args.smoke and args.input_sizes is None:
        parser.error('--requests-per-stratum requires --input-sizes')
    if args.input_sizes is not None and any(size < 0 for size in args.input_sizes):
        parser.error('--input-sizes requires nonnegative integers')
    if args.requests_per_stratum is not None and any(count <= 0 for count in args.requests_per_stratum):
        parser.error('--requests-per-stratum requires positive integers')
    manifest = build_corpus(args.output_dir,
                            counts=None if args.smoke else dict(zip(SPLITS, args.requests_per_stratum)),
                            smoke=args.smoke, input_sizes=args.input_sizes, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                                    'training_ready', 'production_export_allowed')}, indent=2))


if __name__ == '__main__':
    main()
