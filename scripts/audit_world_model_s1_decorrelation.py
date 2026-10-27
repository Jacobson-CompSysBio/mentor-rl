#!/usr/bin/env python3
"""Verify a versioned S1 corpus and its actual tokenizer/supervision budget."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract, system_prompt_sha256
from runtime.world_model_training import (
    LOSS_FIELD_NAMES, LOSS_SEGMENT_NAMES, S1_V3_DATASET_ID,
    flatten_v3_trace_for_arrow, tokenize_v3_trace_for_sft,
)

TOKENIZER = None
SEGMENT_TELEMETRY = False
MAX_LENGTH = 4096
BASELINE = REPO_ROOT / "data/world_model_v4/corpora/s1_v6"
CANDIDATE = REPO_ROOT / "data/world_model_v4/corpora/s1_v8"
TOKENIZER_PATH = REPO_ROOT / (
    "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4/plain_base_tokenizer/tokenizer"
)


def sha256(path: Path) -> str:
    """Hash a corpus artifact without loading it into memory."""

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def initialize_worker(candidate: str, segment_telemetry: bool = False) -> None:
    """Load the existing local training tokenizer once per CPU worker."""

    from transformers import AutoTokenizer

    global TOKENIZER, CANDIDATE, SEGMENT_TELEMETRY
    CANDIDATE = Path(candidate)
    SEGMENT_TELEMETRY = segment_telemetry
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)


def audit_shard(job: tuple[str, int, int]) -> dict:
    """Exercise the real training serializer and completion masks on every row."""

    split, shard, workers = job
    contract = load_contract()
    result = dict(split=split, rows=0, maximum_tokens=0, supervised_tokens=0)
    if SEGMENT_TELEMETRY:
        result["segment_tokens"] = {name: 0 for name in LOSS_SEGMENT_NAMES.values()}
        result["field_tokens"] = {name: 0 for name in LOSS_FIELD_NAMES.values()}
    with (CANDIDATE / f"{split}.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if index % workers != shard:
                continue
            row = json.loads(line)
            assert row["messages"][0]["content"] == contract["runtime"]["system_prompt"]
            assert row["metadata"]["system_prompt_sha256"] == system_prompt_sha256(contract)
            if split != "test":
                flatten_v3_trace_for_arrow(row, expected_split=split, dataset_id=S1_V3_DATASET_ID)
            encoded = tokenize_v3_trace_for_sft(
                row, TOKENIZER, MAX_LENGTH, segment_telemetry=SEGMENT_TELEMETRY
            )
            assert len(encoded["input_ids"]) == len(encoded["completion_mask"])
            assert set(encoded["completion_mask"]) == {0, 1}
            result["rows"] += 1
            result["supervised_tokens"] += sum(encoded["completion_mask"])
            if SEGMENT_TELEMETRY:
                segments, fields = encoded["loss_segments"], encoded["loss_field_masks"]
                assert len(segments) == len(fields) == len(encoded["input_ids"])
                assert all(bool(segment) == bool(mask) for segment, mask in
                           zip(segments, encoded["completion_mask"], strict=True))
                assert all(not field or segment == 3 for field, segment in zip(fields, segments, strict=True))
                for code, name in LOSS_SEGMENT_NAMES.items():
                    result["segment_tokens"][name] += segments.count(code)
                for bit, name in LOSS_FIELD_NAMES.items():
                    result["field_tokens"][name] += sum(bool(field & bit) for field in fields)
            length = len(encoded["input_ids"])
            if length > result["maximum_tokens"]:
                result["maximum_tokens"] = length
                result["maximum_row_id"] = row["row_id"]
    return result


def check_frozen_artifacts(candidate: Path, pins_path: Path) -> dict:
    """Verify v6 is untouched and all evaluation rows/keys remain byte-identical."""

    pins = json.loads(pins_path.read_text())
    for name, expected in pins.items():
        assert sha256(REPO_ROOT / name) == expected, f"Frozen artifact changed: {name}"
    comparisons = {}
    for section in ("", "s1_1", "s1_2", "s1_3", "s1_4", "s1_6", "s1_7", "s1_8", "s1_9"):
        for split in ("validation", "test"):
            path = Path(section) / f"{split}.jsonl"
            digest = sha256(candidate / path)
            assert digest == sha256(BASELINE / path), f"Held-out rows changed: {path}"
            comparisons[str(path)] = digest
    for split in ("validation", "test"):
        for kind in ("questions", "answer_key"):
            name = f"{split}_{kind}.jsonl"
            root = REPO_ROOT / "data/world_model_v4/eval"
            digest = sha256(root / candidate.name / name)
            assert digest == sha256(root / "s1_v6" / name), f"Evaluation changed: {name}"
            comparisons[name] = digest
    return {"baseline_files_verified": len(pins), "identical_heldout_files": comparisons}


def check_candidate_identity(candidate: Path, overlay: Path) -> dict:
    """Bind the audit to the corpus's actual published overlay and row hashes."""
    manifest = json.loads((candidate / "manifest.json").read_text())
    pin = manifest["contract"]["s1_overlay"]
    if overlay.resolve() != (REPO_ROOT / pin["path"]).resolve() or sha256(overlay) != pin["sha256"]:
        raise ValueError("The audit overlay differs from the candidate manifest pin")
    if manifest["contract"]["sha256"] != sha256(REPO_ROOT / "runtime/world_model/contract.yaml"):
        raise ValueError("The candidate manifest names a different shared contract")
    for split in ("train", "validation", "test"):
        name = f"{split}.jsonl"
        if sha256(candidate / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"The candidate manifest hash changed for {name}")
    return manifest


def main() -> None:
    """Write a pinned audit only after every component and tokenizer check passes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--overlay", type=Path, default=REPO_ROOT / "runtime/world_model/s1_v8.yaml")
    parser.add_argument("--baseline-pins", type=Path,
                        default=REPO_ROOT / "diagnostics/s1_v8_decorrelation/baseline_pins.json")
    parser.add_argument("--segment-loss-telemetry", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.candidate = args.candidate.resolve()
    check_candidate_identity(args.candidate, args.overlay)
    report = {
        "max_length": MAX_LENGTH, "frozen": check_frozen_artifacts(args.candidate, args.baseline_pins),
        "segment_loss_telemetry": args.segment_loss_telemetry,
        "splits": {}, "files": {},
        "tokenizer_files": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in sorted(TOKENIZER_PATH.iterdir()) if path.is_file()
        },
        "serializer_sha256": sha256(REPO_ROOT / "runtime/world_model_training.py"),
        "overlay_sha256": sha256(args.overlay),
        "implementation_files": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in [
                *(REPO_ROOT / "runtime/world_model" / f"s1_{section}_corpus.py"
                  for section in (1, 2, 3, 4, 6, 7, 8, 9)),
                REPO_ROOT / "runtime/world_model/s1_overlay.py",
                REPO_ROOT / "runtime/world_model/s1_prompt_banks.py",
                REPO_ROOT / "scripts/build_world_model_s1_bundle.py",
            ]
        },
    }
    jobs = [(split, shard, args.workers) for split in ("train", "validation", "test")
            for shard in range(args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                             initargs=(str(args.candidate.resolve()), args.segment_loss_telemetry)) as pool:
        for result in pool.map(audit_shard, jobs):
            split = result.pop("split")
            entry = report["splits"].setdefault(split, dict(rows=0, maximum_tokens=0, supervised_tokens=0))
            for key in ("rows", "supervised_tokens"):
                entry[key] += result[key]
            if args.segment_loss_telemetry:
                for kind in ("segment_tokens", "field_tokens"):
                    target = entry.setdefault(kind, {name: 0 for name in result[kind]})
                    for name, count in result[kind].items():
                        target[name] += count
            if result["maximum_tokens"] > entry["maximum_tokens"]:
                entry["maximum_tokens"] = result["maximum_tokens"]
                entry["maximum_row_id"] = result["maximum_row_id"]
            print(json.dumps({"split": split, **entry}), flush=True)
    for split, expected in (("train", 55276), ("validation", 6875), ("test", 6875)):
        assert report["splits"][split]["rows"] == expected
        path = args.candidate / f"{split}.jsonl"
        report["files"][str(path.relative_to(REPO_ROOT))] = sha256(path)
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
