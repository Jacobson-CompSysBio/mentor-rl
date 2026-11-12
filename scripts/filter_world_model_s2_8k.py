#!/usr/bin/env python3
"""Materialize fitting S2 candidates without changing a trace or source split.

This writes a new candidate pool, not a balanced training export. All token
lengths must come from the verified complete-wire audit. Production case quotas,
coverage, replay, and the trainer adapter remain separate readiness checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_world_model_s2_context import SPLITS, load_verified_lengths, sha256


def filter_candidates(report_path: Path, output_dir: Path, *, root: Path = ROOT) -> dict:
    """Copy only fitting original JSONL lines into a new immutable candidate pool."""
    report_path, output_dir = Path(report_path), Path(output_dir)
    report = json.loads(report_path.read_text())
    if (report.get("max_length") != 8192 or report.get("truncation_applied") is not False
            or report.get("unmeasured_families")):
        raise ValueError("all thirteen families require verified untruncated 8192-token measurements")
    families = report["families"]
    if {entry["family"] for entry in families} != {f"S2.{n}" for n in range(1, 14)}:
        raise ValueError("the candidate audit must cover every S2 family")
    if output_dir.exists():
        raise FileExistsError(f"preserve existing candidate output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "mentor-rl-s2-8k-candidates-v1",
        "purpose": "length_filtered_candidates", "max_length": 8192,
        "truncation_applied": False, "original_row_bytes_preserved": True,
        "source_splits_preserved": True, "training_ready": False,
        "production_export_allowed": False,
        "audit_path": str(report_path), "audit_sha256": sha256(report_path),
        "families": [], "row_counts": dict.fromkeys(SPLITS, 0),
        "remaining_work": ["Map approved case IDs to generators and finish missing implementations.",
            "Generate and balance retained cases to 512/64/64 distinct rows.",
            "Assemble and fully tokenize the approved 80/15/5 training mixture.",
            "Verify production training and exact evaluation adapters."]}
    with tempfile.TemporaryDirectory(prefix=".s2_8k_", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "pool"
        staging.mkdir()
        for family in families:
            source = root / family["path"]
            if sha256(source / "manifest.json") != family["manifest_sha256"]:
                raise ValueError(f"source manifest changed after context audit: {source}")
            cache = family.get("separate_token_cache")
            cache_path = root / cache["path"] if cache else None
            if cache and sha256(cache_path) != cache["sha256"]:
                raise ValueError(f"token cache changed after context audit: {cache_path}")
            _, lengths = load_verified_lengths(source, root, cache_path)
            target = staging / family["family"].replace(".", "_").lower()
            target.mkdir()
            files, counts, excluded = {}, {}, []
            for split in SPLITS:
                count = 0
                with (source / f"{split}.jsonl").open("rb") as incoming, (target / f"{split}.jsonl").open("wb") as outgoing:
                    for line in incoming:
                        row = json.loads(line)
                        row_id = row["row_id"]
                        if row["metadata"]["split"] != split:
                            raise ValueError("candidate filtering cannot move a source split")
                        if row["metadata"]["question_family"].replace(".single_edge", "") != family["family"]:
                            raise ValueError("candidate source belongs to a different question family")
                        if lengths[row_id]["tokens"] <= 8192:
                            outgoing.write(line)
                            count += 1
                        else:
                            excluded.append(row_id)
                if count != family["splits"][split]["retained"]:
                    raise ValueError("candidate counts differ from the reviewed length audit")
                path = target / f"{split}.jsonl"
                files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size, "rows": count}
                counts[split] = count
                summary["row_counts"][split] += count
            if sorted(excluded) != sorted(family["excluded_row_ids"]):
                raise ValueError("candidate exclusions differ from the reviewed length audit")
            receipt = {"family": family["family"], "source_path": family["path"],
                "source_manifest_sha256": family["manifest_sha256"], "row_counts": counts,
                "excluded_rows": len(excluded), "files": files,
                "max_length": 8192, "training_ready": False, "truncation_applied": False}
            (target / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
            summary["families"].append(receipt)
        (staging / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        shutil.move(str(staging), str(output_dir))
    return summary


def main() -> None:
    """Require a measured audit and a new destination; never overwrite a corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = filter_candidates(args.audit, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "row_counts": summary["row_counts"],
                      "training_ready": summary["training_ready"]}, indent=2))


if __name__ == "__main__":
    main()
