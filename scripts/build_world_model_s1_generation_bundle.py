#!/usr/bin/env python3
"""Build one verified, answer-key-free native S1 generation bundle."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.s1_overlay import S1_OVERLAY_V9_SCHEMA_VERSION
from runtime.world_model.contract import (  # noqa: E402
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from runtime.world_model.source_index import (  # noqa: E402
    check_source_index,
    content_sha256,
)
from scripts.world_model_s1_eval_common import (  # noqa: E402
    DATASET_ID,
    EVALUATION_CONTRACT,
    EVALUATOR_SCHEMA_VERSION,
    GENERATION_BUNDLE_SCHEMA_VERSION,
    LOSS_CONTRACT,
    S1EvaluationError,
    canonical_json,
    read_json,
    read_jsonl,
    require_sha256,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    validate_question,
    write_json,
    write_jsonl,
)


DEFAULT_SOURCE_INDEX = (
    REPO_ROOT
    / "data/world_model_v3/source_indexes/"
    "world_model_source_index_"
    "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
    ".json.gz"
)
DEFAULT_SOURCE_INDEX_SHA256 = (
    "08d093d802562c0e0620e00b96ff94b4779a9d2118c48b4d741e9feabc7b0a9e"
)
S1_V10_OVERLAY_PATH = Path("runtime/world_model/s1_v10.yaml")
# The canonical S1 overlays an evaluator may name, with their schemas.
S1_OVERLAY_SCHEMAS = {Path("runtime/world_model/s1_v10.yaml"): S1_OVERLAY_V9_SCHEMA_VERSION}


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise S1EvaluationError(f"path escapes repository root: {path}") from error


def _contract_pin(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    repo_root: Path,
    s1_overlay_path: Path | None = None,
) -> dict[str, Any]:
    """Return the exact current contract identity stored by the S1 builders."""

    pin = {
        "path": _repo_relative(contract_path, repo_root),
        "sha256": sha256_file(contract_path),
        "row_schema_version": contract["runtime"]["row_schema_version"],
        "system_prompt_sha256": system_prompt_sha256(contract),
        "tool_contract_version": contract["runtime"]["tool_contract_version"],
        "tool_definitions_sha256": stable_sha256(tool_definitions(contract)),
    }
    if s1_overlay_path is not None:
        overlay_path = (repo_root / s1_overlay_path).resolve()
        pin["s1_overlay"] = {
            "path": _repo_relative(overlay_path, repo_root),
            "sha256": sha256_file(overlay_path),
            "schema_version": S1_OVERLAY_SCHEMAS[s1_overlay_path],
        }
    return pin


def _load_evaluator(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    repo_root: Path,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    """Verify the public S1 evaluator and return its native test questions."""

    if sha256_file(evaluator_manifest_path) != evaluator_manifest_sha256:
        raise S1EvaluationError("the evaluator manifest identity changed")
    evaluator = read_json(evaluator_manifest_path)
    if (
        evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != DATASET_ID
        or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
        or evaluator.get("loss_contract") != LOSS_CONTRACT
    ):
        raise S1EvaluationError("the S1 evaluator contract changed")
    evaluator_pin = evaluator.get("contract")
    # Verify the complete contract pin, including the current training overlay.
    declared = (
        evaluator_pin.get("s1_overlay") if isinstance(evaluator_pin, Mapping) else None
    )
    s1_overlay_path = None
    if declared is not None:
        s1_overlay_path = S1_V10_OVERLAY_PATH
        if isinstance(declared, Mapping) and isinstance(declared.get("path"), str):
            candidate = Path(declared["path"])
            if candidate in S1_OVERLAY_SCHEMAS:
                s1_overlay_path = candidate
    expected_pin = _contract_pin(
        contract,
        contract_path=contract_path,
        repo_root=repo_root,
        s1_overlay_path=s1_overlay_path,
    )
    if evaluator_pin != expected_pin:
        raise S1EvaluationError("the evaluator's runtime contract pin changed")

    corpus_manifest_path = resolve_repo_path(
        repo_root,
        evaluator.get("corpus_manifest"),
        "S1 corpus manifest",
    )
    corpus_manifest_sha256 = require_sha256(
        evaluator.get("corpus_manifest_sha256"),
        "S1 corpus manifest identity",
    )
    if sha256_file(corpus_manifest_path) != corpus_manifest_sha256:
        raise S1EvaluationError("the S1 corpus manifest identity changed")
    corpus_manifest = read_json(corpus_manifest_path)
    if (
        corpus_manifest.get("dataset_id") != DATASET_ID
        or corpus_manifest.get("evaluation_contract") != EVALUATION_CONTRACT
        or corpus_manifest.get("loss_contract") != LOSS_CONTRACT
        or corpus_manifest.get("contract") != expected_pin
    ):
        raise S1EvaluationError("the evaluator and corpus contracts disagree")

    panel = evaluator.get("test")
    if not isinstance(panel, Mapping):
        raise S1EvaluationError("the S1 evaluator has no test panel")
    questions_path = resolve_repo_path(
        repo_root,
        panel.get("questions_path"),
        "S1 test questions",
    )
    questions_sha256 = require_sha256(
        panel.get("questions_sha256"),
        "S1 test questions identity",
    )
    if sha256_file(questions_path) != questions_sha256:
        raise S1EvaluationError("the S1 test question identity changed")
    if panel.get("test_panel_id") != questions_sha256:
        raise S1EvaluationError("the S1 test panel ID changed")
    rows = [validate_question(row, contract) for row in read_jsonl(questions_path)]
    if not rows or len(rows) != panel.get("row_count"):
        raise S1EvaluationError("the S1 test row count changed")
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise S1EvaluationError("the S1 test record IDs are not unique")
    family_counts = Counter(
        str(row["metadata"]["question_family"]) for row in rows
    )
    if dict(sorted(family_counts.items())) != dict(
        sorted(panel.get("family_counts", {}).items())
    ):
        raise S1EvaluationError("the S1 test family counts changed")
    return evaluator, questions_path, rows


def build_generation_bundle(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    source_index_path: Path,
    source_index_sha256: str,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    contract_path: Path | None = None,
) -> dict[str, Any]:
    """Validate S1 public inputs and publish a native answer-free bundle."""

    repo_root = repo_root.resolve()
    evaluator_manifest_path = evaluator_manifest_path.resolve()
    source_index_path = source_index_path.resolve()
    contract_path = (
        contract_path or repo_root / "runtime/world_model/contract.yaml"
    ).resolve()
    try:
        evaluator_manifest_path.relative_to(repo_root)
        source_index_path.relative_to(repo_root)
        contract_path.relative_to(repo_root)
    except ValueError as error:
        raise S1EvaluationError("an S1 bundle input escapes the repository") from error
    evaluator_manifest_sha256 = require_sha256(
        evaluator_manifest_sha256,
        "evaluator manifest identity",
    )
    source_index_sha256 = require_sha256(
        source_index_sha256,
        "source-index identity",
    )
    contract = load_contract(contract_path)
    evaluator, questions_path, rows = _load_evaluator(
        evaluator_manifest_path=evaluator_manifest_path,
        evaluator_manifest_sha256=evaluator_manifest_sha256,
        repo_root=repo_root,
        contract=contract,
        contract_path=contract_path,
    )
    source_index = check_source_index(source_index_path, contract)
    if content_sha256(source_index) != source_index_sha256:
        raise S1EvaluationError("the S1 source-index identity changed")

    panel = evaluator["test"]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    questions_output = output_dir / "questions.jsonl"
    write_jsonl(questions_output, rows, private=True)
    record_ids = [str(row["record_id"]) for row in rows]
    book_mode_counts = Counter(
        str(row["metadata"]["book_mode"]) for row in rows
    )
    failure_row_count = sum(
        row["metadata"].get("blocked_reason_code") == "tool_error"
        for row in rows
    )
    bundle = {
        "schema_version": GENERATION_BUNDLE_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "evaluation_contract": EVALUATION_CONTRACT,
        "loss_contract": LOSS_CONTRACT,
        "test_panel_id": panel["test_panel_id"],
        "record_count": len(rows),
        "record_ids_sha256": stable_sha256(record_ids),
        "family_counts": dict(sorted(panel["family_counts"].items())),
        "book_mode_counts": dict(sorted(book_mode_counts.items())),
        "failure_row_count": failure_row_count,
        "source_evaluator_manifest_sha256": evaluator_manifest_sha256,
        "source_questions_path": _repo_relative(questions_path, repo_root),
        "source_questions_sha256": panel["questions_sha256"],
        "questions_sha256": sha256_file(questions_output),
        "contract": dict(evaluator["contract"]),
        "source_index": {
            "path": _repo_relative(source_index_path, repo_root),
            "id": f"sha256:{source_index_sha256}",
            "sha256": source_index_sha256,
        },
        "failure_harness": {
            "policy_source": "runtime/world_model/contract.yaml",
            "selector": "blocked_reason_code=tool_error",
            "metadata_visible_to_model": False,
        },
        "native_transcript": True,
        "reads_private_answer_keys": False,
    }
    bundle["bundle_sha256"] = stable_sha256(bundle)
    write_json(output_dir / "manifest.json", bundle, private=True)
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument(
        "--source-index", type=Path, default=DEFAULT_SOURCE_INDEX
    )
    parser.add_argument(
        "--source-index-sha256", default=DEFAULT_SOURCE_INDEX_SHA256
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--contract", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_generation_bundle(
        evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256,
        source_index_path=args.source_index,
        source_index_sha256=args.source_index_sha256,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        contract_path=args.contract,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
