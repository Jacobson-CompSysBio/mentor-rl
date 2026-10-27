#!/usr/bin/env python3
"""Build a deterministic, behavior-stratified smoke slice of aggregate S1."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import S1_V3_TRACE_FAMILIES  # noqa: E402
from scripts.build_world_model_s1_bundle import (  # noqa: E402
    AtomicBatch,
    BundleBuildError,
    CORPUS_SCHEMA_VERSION,
    DATASET_ID,
    EVALUATION_CONTRACT,
    EVALUATOR_SCHEMA_VERSION,
    LOSS_CONTRACT,
    MANIFEST_SCHEMA_VERSION,
    canonical_json,
    project_row,
    read_json,
    sha256_file,
    stable_sha256,
    write_json,
)


SPLITS = ("train", "validation", "test")
EVALUATION_SPLITS = ("validation", "test")
DEFAULT_SEED = 20261006
DEFAULT_TRAIN_PER_SIGNATURE = 2
DEFAULT_EVAL_PER_SIGNATURE = 1
SELECTION_ALGORITHM = "longest_plus_sha256_rank_per_behavior_signature_v1"
SIGNATURE_FIELDS = (
    "step",
    "question_family",
    "source_trace",
    "book_mode",
    "blocked_reason_code",
    "final_status",
    "continuation_decision",
    "result_keys",
    "tool_call_path",
)


@dataclass(frozen=True, slots=True)
class SelectedRow:
    """One source row selected for the smoke corpus."""

    rank: str
    row_id: str
    raw_line: bytes
    row: dict[str, Any]
    signature_id: str
    signature: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SplitSelection:
    """The complete deterministic selection for one split."""

    rows: tuple[SelectedRow, ...]
    source_row_count: int
    source_signature_count: int
    selected_row_ids_sha256: str
    family_counts: dict[str, int]
    step_counts: dict[str, int]
    signature_counts: dict[str, dict[str, Any]]


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return one repository-relative path."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise BundleBuildError(f"path escapes repository root: {path}") from error


def behavior_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the training behavior exercised by one canonical S0/S1 row."""

    metadata = row.get("metadata")
    messages = row.get("messages")
    if not isinstance(metadata, Mapping) or not isinstance(messages, list):
        raise BundleBuildError("a smoke source row lacks metadata or messages")
    step = metadata.get("step")
    family = metadata.get("question_family")
    source_trace = metadata.get("source_trace")
    book_mode = metadata.get("book_mode")
    blocked_reason_code = metadata.get("blocked_reason_code")
    if step not in {"S0", "S1"}:
        raise BundleBuildError(f"unsupported smoke source step: {step!r}")
    if family not in S1_V3_TRACE_FAMILIES:
        raise BundleBuildError(f"unsupported smoke question family: {family!r}")
    if (
        not isinstance(source_trace, str)
        or not source_trace.startswith(f"{step}.")
    ):
        raise BundleBuildError(
            f"smoke source trace {source_trace!r} does not match step {step}"
        )
    if not isinstance(book_mode, str) or not book_mode:
        raise BundleBuildError("a smoke source row lacks book_mode")
    if blocked_reason_code is not None and not isinstance(
        blocked_reason_code, str
    ):
        raise BundleBuildError("a smoke blocked_reason_code is invalid")
    if not messages or not isinstance(messages[-1], Mapping):
        raise BundleBuildError("a smoke source row lacks a final message")
    final_content = messages[-1].get("content")
    if (
        messages[-1].get("role") != "assistant"
        or not isinstance(final_content, Mapping)
    ):
        raise BundleBuildError("a smoke source row lacks a final assistant object")
    final_status = final_content.get("status")
    continuation_decision = final_content.get("continuation_decision")
    if not isinstance(final_status, str) or not final_status:
        raise BundleBuildError("a smoke final answer lacks status")
    if not isinstance(continuation_decision, str) or not continuation_decision:
        raise BundleBuildError("a smoke final answer lacks continuation_decision")
    result = final_content.get("result")
    if result is None:
        result_keys: list[str] = []
    elif isinstance(result, Mapping):
        if any(not isinstance(key, str) for key in result):
            raise BundleBuildError("a smoke final result has a non-string key")
        result_keys = sorted(result)
    else:
        raise BundleBuildError("a smoke final result is not an object or null")
    tool_call_path: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise BundleBuildError("a smoke source row has a non-object message")
        if message.get("role") != "assistant" or "tool_call" not in message:
            continue
        tool_call = message.get("tool_call")
        if not isinstance(tool_call, Mapping) or not isinstance(
            tool_call.get("name"), str
        ):
            raise BundleBuildError("a smoke source row has an invalid tool call")
        tool_call_path.append(str(tool_call["name"]))
    return {
        "step": step,
        "question_family": family,
        "source_trace": source_trace,
        "book_mode": book_mode,
        "blocked_reason_code": blocked_reason_code,
        "final_status": final_status,
        "continuation_decision": continuation_decision,
        "result_keys": result_keys,
        "tool_call_path": tool_call_path,
    }


def select_split(
    source_path: Path,
    *,
    split: str,
    rows_per_signature: int,
    seed: int,
) -> SplitSelection:
    """Select the lowest hash-ranked rows from every observed behavior."""

    if split not in SPLITS:
        raise BundleBuildError(f"unsupported smoke split: {split}")
    if rows_per_signature < 1:
        raise BundleBuildError("rows_per_signature must be positive")
    ranked_candidates: dict[str, list[SelectedRow]] = {}
    longest_candidates: dict[str, SelectedRow] = {}
    signature_source_counts: Counter[str] = Counter()
    source_ids: set[str] = set()
    source_row_count = 0
    with source_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            source_row_count += 1
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BundleBuildError(
                    f"invalid smoke source JSON at {source_path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise BundleBuildError("a smoke source row is not an object")
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id:
                raise BundleBuildError("a smoke source row lacks row_id")
            if row_id in source_ids:
                raise BundleBuildError(f"duplicate smoke source row ID {row_id}")
            source_ids.add(row_id)
            signature = behavior_signature(row)
            signature_id = stable_sha256(signature)
            rank = hashlib.sha256(
                f"{seed}:{split}:{signature_id}:{row_id}".encode("utf-8")
            ).hexdigest()
            selected = SelectedRow(
                rank=rank,
                row_id=row_id,
                raw_line=raw_line.rstrip(b"\r\n") + b"\n",
                row=row,
                signature_id=signature_id,
                signature=signature,
            )
            signature_source_counts[signature_id] += 1
            group = ranked_candidates.setdefault(signature_id, [])
            group.append(selected)
            group.sort(key=lambda value: (value.rank, value.row_id))
            del group[rows_per_signature:]

            longest = longest_candidates.get(signature_id)
            if longest is None or len(selected.raw_line) > len(
                longest.raw_line
            ) or (
                len(selected.raw_line) == len(longest.raw_line)
                and selected.row_id < longest.row_id
            ):
                longest_candidates[signature_id] = selected

    insufficient = sorted(
        signature_id
        for signature_id, count in signature_source_counts.items()
        if count < rows_per_signature
    )
    if insufficient:
        raise BundleBuildError(
            "smoke source has too few rows for behavior signatures: "
            + ", ".join(insufficient)
        )
    selected_by_signature: dict[str, list[SelectedRow]] = {}
    for signature_id in sorted(ranked_candidates):
        longest = longest_candidates[signature_id]
        selected = [longest]
        selected.extend(
            row
            for row in ranked_candidates[signature_id]
            if row.row_id != longest.row_id
        )
        selected_by_signature[signature_id] = selected[:rows_per_signature]
    rows = tuple(
        row
        for signature_id in sorted(selected_by_signature)
        for row in selected_by_signature[signature_id]
    )
    family_counts = Counter(
        str(row.signature["question_family"]) for row in rows
    )
    step_counts = Counter(str(row.signature["step"]) for row in rows)
    if set(family_counts) != set(S1_V3_TRACE_FAMILIES):
        raise BundleBuildError(
            f"smoke {split} does not cover every S1 family: "
            f"{sorted(family_counts)}"
        )
    signature_counts = {
        signature_id: {
            "signature": selected_by_signature[signature_id][0].signature,
            "selected_row_count": len(selected_by_signature[signature_id]),
            "longest_row_id": longest_candidates[signature_id].row_id,
        }
        for signature_id in sorted(selected_by_signature)
    }
    return SplitSelection(
        rows=rows,
        source_row_count=source_row_count,
        source_signature_count=len(selected_by_signature),
        selected_row_ids_sha256=stable_sha256([row.row_id for row in rows]),
        family_counts=dict(sorted(family_counts.items())),
        step_counts=dict(sorted(step_counts.items())),
        signature_counts=signature_counts,
    )


def _validate_source_manifest(
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify the complete aggregate bundle used as the smoke source."""

    manifest = read_json(source_root / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION
        or manifest.get("dataset_id") != DATASET_ID
        or manifest.get("loss_contract") != LOSS_CONTRACT
        or manifest.get("evaluation_contract") != EVALUATION_CONTRACT
    ):
        raise BundleBuildError("the aggregate S1 smoke source contract changed")
    files = manifest.get("files")
    row_counts = manifest.get("row_counts")
    if not isinstance(files, Mapping) or not isinstance(row_counts, Mapping):
        raise BundleBuildError("the aggregate S1 source lacks file metadata")
    verified: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        name = f"{split}.jsonl"
        spec = files.get(name)
        if not isinstance(spec, Mapping):
            raise BundleBuildError(f"the aggregate S1 source lacks {name}")
        path = source_root / name
        digest = sha256_file(path)
        count = spec.get("row_count")
        if (
            digest != spec.get("sha256")
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            or count != row_counts.get(split)
        ):
            raise BundleBuildError(f"the aggregate S1 source {name} changed")
        verified[name] = {"row_count": count, "sha256": digest}
    return manifest, verified


def build(
    *,
    repo_root: Path = REPO_ROOT,
    source_root: Path | None = None,
    corpus_root: Path | None = None,
    evaluator_root: Path | None = None,
    seed: int = DEFAULT_SEED,
    train_per_signature: int = DEFAULT_TRAIN_PER_SIGNATURE,
    eval_per_signature: int = DEFAULT_EVAL_PER_SIGNATURE,
) -> dict[str, Any]:
    """Build and atomically publish one stratified smoke bundle."""

    repo_root = repo_root.resolve()
    source_root = (
        source_root or repo_root / "data/world_model_v3/corpora/s1"
    ).resolve()
    corpus_root = (
        corpus_root or repo_root / "data/world_model_v3/corpora/s1_smoke"
    ).resolve()
    evaluator_root = (
        evaluator_root or repo_root / "data/world_model_v3/eval/s1_smoke"
    ).resolve()
    source_manifest, source_files = _validate_source_manifest(source_root)
    source_manifest_path = source_root / "manifest.json"
    source_manifest_sha256 = sha256_file(source_manifest_path)
    limits = {
        "train": train_per_signature,
        "validation": eval_per_signature,
        "test": eval_per_signature,
    }
    selections = {
        split: select_split(
            source_root / f"{split}.jsonl",
            split=split,
            rows_per_signature=limits[split],
            seed=seed,
        )
        for split in SPLITS
    }
    for split in SPLITS:
        expected = source_files[f"{split}.jsonl"]["row_count"]
        if selections[split].source_row_count != expected:
            raise BundleBuildError(
                f"smoke {split} read {selections[split].source_row_count} rows, "
                f"expected {expected}"
            )

    batch = AtomicBatch()
    try:
        corpus_temporary = {
            split: batch.path_for(corpus_root / f"{split}.jsonl")
            for split in SPLITS
        }
        question_temporary = {
            split: batch.path_for(evaluator_root / f"{split}_questions.jsonl")
            for split in EVALUATION_SPLITS
        }
        answer_temporary = {
            split: batch.path_for(evaluator_root / f"{split}_answer_key.jsonl")
            for split in EVALUATION_SPLITS
        }
        published_ids: set[str] = set()
        for split in SPLITS:
            questions = None
            answers = None
            with corpus_temporary[split].open("wb") as corpus_output:
                try:
                    if split in EVALUATION_SPLITS:
                        questions = question_temporary[split].open("wb")
                        answers = answer_temporary[split].open("wb")
                    for selected in selections[split].rows:
                        if selected.row_id in published_ids:
                            raise BundleBuildError(
                                f"duplicate published smoke row ID {selected.row_id}"
                            )
                        published_ids.add(selected.row_id)
                        corpus_output.write(selected.raw_line)
                        if split in EVALUATION_SPLITS:
                            assert questions is not None and answers is not None
                            question, answer = project_row(
                                selected.row_id,
                                selected.row["messages"],
                                selected.row["tools"],
                                selected.row["metadata"],
                                split,
                            )
                            questions.write(
                                (canonical_json(question) + "\n").encode("utf-8")
                            )
                            answers.write(
                                (canonical_json(answer) + "\n").encode("utf-8")
                            )
                finally:
                    if questions is not None:
                        questions.close()
                    if answers is not None:
                        answers.close()

        row_counts = {
            split: len(selections[split].rows) for split in SPLITS
        }
        aggregate_files = {
            f"{split}.jsonl": {
                "row_count": row_counts[split],
                "sha256": sha256_file(corpus_temporary[split]),
            }
            for split in SPLITS
        }
        selection_manifest = {
            "kind": "behavior_stratified_smoke_v1",
            "algorithm": SELECTION_ALGORITHM,
            "rank_input": "{seed}:{split}:{signature_id}:{row_id}",
            "seed": seed,
            "behavior_signature_fields": list(SIGNATURE_FIELDS),
            "source_manifest": {
                "path": _repo_relative(source_manifest_path, repo_root),
                "sha256": source_manifest_sha256,
            },
            "source_files": {
                name: {
                    "path": _repo_relative(source_root / name, repo_root),
                    **spec,
                }
                for name, spec in sorted(source_files.items())
            },
            "splits": {
                split: {
                    "rows_per_behavior_signature": limits[split],
                    "source_row_count": selections[split].source_row_count,
                    "source_signature_count": selections[
                        split
                    ].source_signature_count,
                    "selected_row_count": row_counts[split],
                    "selected_row_ids_sha256": selections[
                        split
                    ].selected_row_ids_sha256,
                    "all_source_signatures_covered": True,
                    "step_counts": selections[split].step_counts,
                    "signatures": selections[split].signature_counts,
                }
                for split in SPLITS
            },
        }
        corpus_manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "training_contract": LOSS_CONTRACT,
            "loss_contract": LOSS_CONTRACT,
            "evaluation_contract": EVALUATION_CONTRACT,
            "contract": source_manifest["contract"],
            "stage_order": source_manifest["stage_order"],
            "train_block_order": source_manifest["train_block_order"],
            "disabled_stages": source_manifest["disabled_stages"],
            "subset": selection_manifest,
            "row_counts": row_counts,
            "total_row_count": sum(row_counts.values()),
            "family_counts": {
                split: selections[split].family_counts for split in SPLITS
            },
            "files": aggregate_files,
        }
        corpus_manifest_path = corpus_root / "manifest.json"
        corpus_manifest_temporary = batch.path_for(corpus_manifest_path)
        write_json(corpus_manifest_temporary, corpus_manifest)
        corpus_manifest_sha256 = sha256_file(corpus_manifest_temporary)

        panels: dict[str, Any] = {}
        for split in EVALUATION_SPLITS:
            questions_path = evaluator_root / f"{split}_questions.jsonl"
            answer_key_path = evaluator_root / f"{split}_answer_key.jsonl"
            questions_sha256 = sha256_file(question_temporary[split])
            panels[split] = {
                "questions_path": _repo_relative(questions_path, repo_root),
                "questions_sha256": questions_sha256,
                "answer_key_path": _repo_relative(answer_key_path, repo_root),
                "answer_key_sha256": sha256_file(answer_temporary[split]),
                "row_count": row_counts[split],
                "family_counts": selections[split].family_counts,
                "behavior_signature_count": selections[
                    split
                ].source_signature_count,
                f"{split}_panel_id": questions_sha256,
            }
        evaluator_manifest = {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "evaluation_contract": EVALUATION_CONTRACT,
            "loss_contract": LOSS_CONTRACT,
            "corpus_manifest": _repo_relative(
                corpus_manifest_path, repo_root
            ),
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "contract": source_manifest["contract"],
            "stage_order": source_manifest["stage_order"],
            "validation": panels["validation"],
            "test": panels["test"],
        }
        evaluator_manifest_path = evaluator_root / "manifest.json"
        evaluator_manifest_temporary = batch.path_for(evaluator_manifest_path)
        write_json(evaluator_manifest_temporary, evaluator_manifest)
        evaluator_manifest_sha256 = sha256_file(evaluator_manifest_temporary)
        batch.commit()
    except BaseException:
        batch.cleanup()
        raise

    return {
        "corpus_manifest": _repo_relative(corpus_manifest_path, repo_root),
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "evaluator_manifest": _repo_relative(evaluator_manifest_path, repo_root),
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "row_counts": row_counts,
        "corpus_files": aggregate_files,
        "validation": panels["validation"],
        "test": panels["test"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--train-per-signature",
        type=int,
        default=DEFAULT_TRAIN_PER_SIGNATURE,
    )
    parser.add_argument(
        "--eval-per-signature",
        type=int,
        default=DEFAULT_EVAL_PER_SIGNATURE,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        repo_root=args.repo_root,
        source_root=args.source_root,
        corpus_root=args.corpus_root,
        evaluator_root=args.evaluator_root,
        seed=args.seed,
        train_per_signature=args.train_per_signature,
        eval_per_signature=args.eval_per_signature,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
