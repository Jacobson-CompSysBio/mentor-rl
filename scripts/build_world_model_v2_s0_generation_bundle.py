#!/usr/bin/env python3
"""Build one answer-free S0 test bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import (  # noqa: E402
    build_world_model_prompt_messages,
    validate_s0_record_metadata,
)


BUNDLE_SCHEMA_VERSION = "mentor-rl-world-model-s0-generation-bundle-v4"
EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-manifest-v4"
EVALUATION_CONTRACT = "seen_fact_closed_book_recall_v1"
DATASET_ID = "world_model_v2_s0_human_identifiers_v4"
QUESTION_KEYS = frozenset(
    {
        "context",
        "input",
        "metadata",
        "provenance",
        "question",
        "record_id",
        "split",
        "system",
        "validators",
    }
)


class S0GenerationBundleError(RuntimeError):
    """Report one invalid generation bundle input."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S0GenerationBundleError(
            f"Could not read one JSON object from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise S0GenerationBundleError(f"Expected one JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise S0GenerationBundleError(
                        f"Expected one JSON object at {path}:{line_number}"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise S0GenerationBundleError(
            f"Could not read JSONL from {path}: {error}"
        ) from error
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write stable JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_declared_path(root: Path, value: Any, label: str) -> Path:
    """Resolve one declared path below its root."""

    if not isinstance(value, str) or not value:
        raise S0GenerationBundleError(f"{label} is absent")
    declared = Path(value)
    path = declared if declared.is_absolute() else REPO_ROOT / declared
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise S0GenerationBundleError(
            f"{label} escapes the repository"
        ) from error
    return path


def _validate_question(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return one valid answer-free test question."""

    if set(row) != QUESTION_KEYS or "answer" in row:
        raise S0GenerationBundleError(
            "A test question has an unsafe field set"
        )
    record_id = row.get("record_id")
    provenance = row.get("provenance")
    metadata = row.get("metadata")
    if not isinstance(record_id, str) or not record_id:
        raise S0GenerationBundleError("A test record ID is invalid")
    if row.get("split") != "test":
        raise S0GenerationBundleError("A generation row is not a test row")
    if not isinstance(provenance, Mapping):
        raise S0GenerationBundleError("A test row has no provenance")
    if (
        provenance.get("fact_role") != "seen"
        or provenance.get("prompt_form_id") != "test"
    ):
        raise S0GenerationBundleError("A test row has invalid provenance")
    validate_s0_record_metadata(metadata)
    build_world_model_prompt_messages(
        system=str(row.get("system", "")),
        question=str(row.get("question", "")),
        metadata=metadata,
        context=row.get("context"),
    )
    return {str(key): value for key, value in row.items()}


def build_generation_bundle(
    *,
    evaluator_manifest_path: Path,
    evaluator_manifest_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate the test panel and write one answer-free bundle."""

    evaluator_manifest_path = evaluator_manifest_path.resolve()
    if sha256_file(evaluator_manifest_path) != evaluator_manifest_sha256:
        raise S0GenerationBundleError(
            "The evaluator manifest identity changed"
        )
    evaluator = read_json(evaluator_manifest_path)
    if (
        evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != DATASET_ID
        or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
    ):
        raise S0GenerationBundleError("The evaluator contract changed")
    test = evaluator.get("test")
    if not isinstance(test, Mapping):
        raise S0GenerationBundleError("The evaluator has no test panel")
    questions_path = _resolve_declared_path(
        evaluator_manifest_path.parent,
        test.get("questions_path"),
        "test questions path",
    )
    expected_questions_hash = test.get("questions_sha256")
    if (
        not isinstance(expected_questions_hash, str)
        or sha256_file(questions_path) != expected_questions_hash
    ):
        raise S0GenerationBundleError("The test question identity changed")
    rows = [_validate_question(row) for row in read_jsonl(questions_path)]
    expected_rows = test.get("row_count")
    if expected_rows != len(rows) or len(rows) < 1:
        raise S0GenerationBundleError("The test row count changed")
    record_ids = [str(row["record_id"]) for row in rows]
    if len(set(record_ids)) != len(record_ids):
        raise S0GenerationBundleError("The test record IDs are not unique")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    questions_output = output_dir / "questions.jsonl"
    write_jsonl(questions_output, rows)
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "evaluation_contract": EVALUATION_CONTRACT,
        "test_panel_id": test.get("test_panel_id"),
        "record_count": len(rows),
        "record_ids_sha256": stable_sha256(record_ids),
        "source_evaluator_manifest_sha256": evaluator_manifest_sha256,
        "source_questions_sha256": expected_questions_hash,
        "questions_sha256": sha256_file(questions_output),
        "reads_private_answer_keys": False,
    }
    bundle["bundle_sha256"] = stable_sha256(bundle)
    write_json(output_dir / "manifest.json", bundle)
    return bundle


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Build one generation bundle."""

    args = parse_args()
    result = build_generation_bundle(
        evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
