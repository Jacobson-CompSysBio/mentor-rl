#!/usr/bin/env python3
"""Merge complete native S1 prediction shards in panel order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_world_model_s1_exact_generation import (  # noqa: E402
    merge_prediction_shards,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-bundle", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shard-count", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = merge_prediction_shards(
        bundle_root=args.generation_bundle,
        shards_root=args.shards_root,
        output_dir=args.output_dir,
        expected_shard_count=args.expected_shard_count,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
