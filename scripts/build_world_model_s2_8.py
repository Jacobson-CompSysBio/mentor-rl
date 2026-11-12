#!/usr/bin/env python3
"""Build S2.8 verification traces from an explicit source-gene budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_8_corpus import SPLITS, build_corpus  # noqa: E402


def main() -> None:
    """Require a verification budget and preserve complete requested sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--smoke', action='store_true', help='use the pinned verification schedule and semantic contrasts')
    budget.add_argument('--genes-per-stratum', type=int, nargs=3, metavar=('TRAIN', 'VALIDATION', 'TEST'),
                        help='distinct source genes per attainable stratum in each split; no degree or population cap')
    args = parser.parse_args()
    if args.genes_per_stratum is not None and any(count <= 0 for count in args.genes_per_stratum):
        parser.error('--genes-per-stratum requires positive integers')
    manifest = build_corpus(args.output_dir,
                            counts=None if args.smoke else dict(zip(SPLITS, args.genes_per_stratum)),
                            smoke=args.smoke, seed=args.seed)
    print(json.dumps({key: manifest[key] for key in ('purpose', 'generation_budget', 'row_counts', 'panel_id',
                                                    'training_ready', 'production_export_allowed')}, indent=2))


if __name__ == '__main__':
    main()
