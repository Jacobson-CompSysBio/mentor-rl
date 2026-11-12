#!/usr/bin/env python3
"""Build the isolated, source-backed S2.1 single-edge corpus component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.world_model.s2_corpus import DEFAULT_COUNTS, build_corpus  # noqa: E402


def main() -> None:
    """Build new immutable artifacts, or verify an identical deterministic rerun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261028)
    for split, count in DEFAULT_COUNTS.items():
        parser.add_argument(f'--{split}-queries-per-stratum', type=int, default=count)
    args = parser.parse_args()
    counts = {split: getattr(args, f'{split}_queries_per_stratum') for split in DEFAULT_COUNTS}
    manifest = build_corpus(args.output_dir, counts=counts, seed=args.seed)
    print(json.dumps({'output_dir': str(args.output_dir), 'row_counts': manifest['row_counts'],
                      'panel_id': manifest['panel_id'], 'training_ready': manifest['training_ready']}, indent=2))


if __name__ == '__main__':
    main()
