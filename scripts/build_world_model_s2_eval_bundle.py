#!/usr/bin/env python3
"""Build the frozen, answer-separated S2 structured-rollout evaluator panel."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, BinaryIO


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.s2_production_adapter import (  # noqa: E402
    evaluation_target,
    flatten_row,
)


EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s2-evaluator-v1"
EVALUATION_CONTRACT = "held_out_s2_structured_rollout_v1"
CORPUS_SCHEMA_VERSION = "mentor-rl-s2-production-corpus-v1"
PRODUCTION_ID = "s2_production_v1_8k"
EXPECTED_FAMILIES = tuple(f"S2.{index}" for index in range(1, 14))
PUBLIC_METADATA_KEYS = (
    "schema_version",
    "tool_contract_version",
    "question_family",
    "split",
)
TARGET_KEYS = (
    "status",
    "result",
    "continuation_decision",
    "provenance",
    "blocked_reason_required",
    "expected_tool_calls",
    "expected_tool_responses",
)
SCORED_TARGET_KEYS = (
    "status",
    "result",
    "continuation_decision",
    "provenance",
    "blocked_reason_required",
    "expected_tool_calls",
)
DEFAULT_CORPUS_MANIFEST = (
    REPO_ROOT / "data/world_model_v4/corpora/s2_production_v1_8k/manifest.json"
)
DEFAULT_CORPUS_MANIFEST_SHA256 = (
    "2d4e0613f3327e0e98fef908b453327cefea69350bd8d16c3f96f22acdfb1dfc"
)
DEFAULT_QUESTIONS_SHA256 = (
    "0b60dc7ea96031baa5b87a000a70e9de5eb20a10b5febb39ac078ac00c9471c9"
)
DEFAULT_ANSWER_KEY_SHA256 = (
    "e27144117ec11e2e65710ccc37dda3fefb5dc92f2c11a2cc13f9b6b05f7ed811"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class S2EvalBuildError(RuntimeError):
    """Report one invalid S2 evaluator input or immutable output."""


def canonical_json(value: Any) -> str:
    """Serialize one identity or JSONL record deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def ordered_json(value: Any) -> str:
    """Serialize one wire record without changing contract-significant order."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    """Return one required lowercase SHA-256 value."""

    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S2EvalBuildError(f"{label} must be a lowercase SHA-256 value")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one required JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S2EvalBuildError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise S2EvalBuildError(f"{label} must be one JSON object")
    return value


def _repo_relative(path: Path, repo_root: Path, label: str) -> str:
    """Return one repository-relative input path without allowing escape."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as error:
        raise S2EvalBuildError(f"{label} escapes the repository: {path}") from error


def _validate_route(route: Any, family: str) -> None:
    """Require one exact prompt/tool route from the production manifest."""

    if not isinstance(route, Mapping) or set(route) != {
        "contract_version",
        "system_prompt_sha256",
        "tools_sha256",
    }:
        raise S2EvalBuildError(f"{family} has an invalid contract route")
    if not isinstance(route["contract_version"], str) or not route["contract_version"]:
        raise S2EvalBuildError(f"{family} has no contract version")
    _require_sha256(route["system_prompt_sha256"], f"{family} system prompt")
    _require_sha256(route["tools_sha256"], f"{family} tool definitions")


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Validate the immutable production scope needed by the evaluator."""

    if (
        manifest.get("schema_version") != CORPUS_SCHEMA_VERSION
        or manifest.get("production_id") != PRODUCTION_ID
        or manifest.get("data_ready") is not True
        or manifest.get("case_coverage_ready") is not True
        or manifest.get("production_export_allowed") is not True
    ):
        raise S2EvalBuildError("the S2 production scope is not evaluation-ready")

    routes = manifest.get("contract_routes")
    if not isinstance(routes, Mapping) or set(routes) != set(EXPECTED_FAMILIES):
        raise S2EvalBuildError("the S2 evaluator requires all 13 contract routes")
    for family in EXPECTED_FAMILIES:
        _validate_route(routes[family], family)

    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    }:
        raise S2EvalBuildError("the S2 production manifest has no complete split inventory")
    for name in ("validation.jsonl", "test.jsonl"):
        item = files.get(name)
        if not isinstance(item, Mapping):
            raise S2EvalBuildError(f"the {name} file declaration is invalid")
        _require_sha256(item.get("sha256"), f"{name} identity")
        if (
            type(item.get("row_count")) is not int
            or item["row_count"] < 1
            or type(item.get("bytes")) is not int
            or item["bytes"] < 1
        ):
            raise S2EvalBuildError(f"the {name} size declaration is invalid")

    expected_panel_id = stable_sha256(
        {
            split: files[f"{split}.jsonl"]["sha256"]
            for split in ("validation", "test")
        }
    )
    if manifest.get("panel_id") != expected_panel_id:
        raise S2EvalBuildError("the frozen S2 panel identity changed")

    test_rows = files["test.jsonl"]["row_count"]
    budget = manifest.get("budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("test") != test_rows
        or not isinstance(budget.get("s2"), Mapping)
        or budget["s2"].get("test") != test_rows
    ):
        raise S2EvalBuildError("the S2 test budget disagrees with its file declaration")

    entries = manifest.get("case_entries")
    retained_cases = manifest.get("retained_cases")
    quota = manifest.get("per_case_quotas")
    if (
        not isinstance(entries, list)
        or type(retained_cases) is not int
        or retained_cases < 1
        or len(entries) != retained_cases
        or not isinstance(quota, Mapping)
        or type(quota.get("test")) is not int
        or quota["test"] < 1
        or retained_cases * quota["test"] != test_rows
    ):
        raise S2EvalBuildError("the S2 retained-case test quota changed")
    expected_case_counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise S2EvalBuildError("an S2 case entry is invalid")
        case_id = entry.get("case_id")
        counts = entry.get("counts")
        if (
            not isinstance(case_id, str)
            or case_id in expected_case_counts
            or not isinstance(counts, Mapping)
            or counts.get("test") != quota["test"]
        ):
            raise S2EvalBuildError("an S2 case entry has an invalid test count")
        expected_case_counts[case_id] = quota["test"]
    return dict(routes), expected_case_counts


def project_row(
    row: Mapping[str, Any], contract_routes: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split one validated S2 row into a public prompt and private target."""

    flattened = flatten_row(
        row,
        expected_split="test",
        contract_routes=contract_routes,
    )
    messages = row["messages"]
    user_indexes = [
        index for index, message in enumerate(messages) if message["role"] == "user"
    ]
    if len(user_indexes) != 1:
        raise S2EvalBuildError("an S2 evaluation row must have one current user turn")
    current_user = user_indexes[0]
    if [message["role"] for message in messages[: current_user + 1]] != [
        "system",
        "harness",
        "user",
    ]:
        raise S2EvalBuildError("an S2 public question must be system, harness, then user")
    if current_user == len(messages) - 1:
        raise S2EvalBuildError("an S2 evaluation row has no private expected suffix")

    metadata = row["metadata"]
    question = {
        "row_id": flattened["record_id"],
        "messages": deepcopy(messages[: current_user + 1]),
        "tools": deepcopy(row["tools"]),
        "metadata": {key: deepcopy(metadata[key]) for key in PUBLIC_METADATA_KEYS},
    }
    target = evaluation_target(row)
    if tuple(target) != TARGET_KEYS:
        raise S2EvalBuildError("the S2 structured evaluation target changed")
    answer = {
        "row_id": flattened["record_id"],
        "scoring_metadata": {
            "question_family": metadata["question_family"],
            "catalog_case_id": metadata["catalog_case_id"],
        },
        "target": target,
    }
    return question, answer


def _write_record(
    destination: BinaryIO,
    digest: Any,
    value: Mapping[str, Any],
) -> None:
    """Write and hash one canonical JSONL record."""

    encoded = (ordered_json(value) + "\n").encode("utf-8")
    destination.write(encoded)
    digest.update(encoded)


def _implementation_pins() -> dict[str, str]:
    """Return the code identities that define answer separation and targets."""

    adapter = REPO_ROOT / "runtime/world_model/s2_production_adapter.py"
    return {
        "scripts/build_world_model_s2_eval_bundle.py": sha256_file(Path(__file__)),
        "runtime/world_model/s2_production_adapter.py": sha256_file(adapter),
    }


def _publish_immutable(temporary: Path, output_dir: Path) -> None:
    """Publish new output atomically or accept only byte-identical artifacts."""

    if not output_dir.exists():
        os.replace(temporary, output_dir)
        return
    if not output_dir.is_dir():
        raise S2EvalBuildError(f"the output path is not a directory: {output_dir}")
    filenames = {"manifest.json", "test_questions.jsonl", "test_answer_key.jsonl"}
    existing = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if existing != filenames:
        raise S2EvalBuildError("the immutable S2 evaluator file set changed")
    for filename in filenames:
        if sha256_file(output_dir / filename) != sha256_file(temporary / filename):
            raise S2EvalBuildError("the immutable S2 evaluator output changed")
    if ((output_dir / "test_answer_key.jsonl").stat().st_mode & 0o777) != 0o600:
        raise S2EvalBuildError("the immutable S2 answer key is not private")


def build_eval_bundle(
    *,
    corpus_manifest_path: Path = DEFAULT_CORPUS_MANIFEST,
    corpus_manifest_sha256: str = DEFAULT_CORPUS_MANIFEST_SHA256,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate the production test corpus and publish its sealed evaluator."""

    repo_root = repo_root.resolve()
    corpus_manifest_path = corpus_manifest_path.resolve()
    _repo_relative(corpus_manifest_path, repo_root, "corpus manifest")
    corpus_manifest_sha256 = _require_sha256(
        corpus_manifest_sha256,
        "corpus manifest identity",
    )
    if sha256_file(corpus_manifest_path) != corpus_manifest_sha256:
        raise S2EvalBuildError("the S2 production manifest identity changed")
    corpus_manifest = _read_json(corpus_manifest_path, "corpus manifest")
    routes, expected_case_counts = _validate_manifest(corpus_manifest)

    test_path = corpus_manifest_path.parent / "test.jsonl"
    _repo_relative(test_path, repo_root, "test corpus")
    test_spec = corpus_manifest["files"]["test.jsonl"]
    output_dir = output_dir.resolve()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.building-",
        dir=output_dir.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        temporary.chmod(0o700)
        questions_path = temporary / "test_questions.jsonl"
        answers_path = temporary / "test_answer_key.jsonl"
        source_digest = hashlib.sha256()
        question_digest = hashlib.sha256()
        answer_digest = hashlib.sha256()
        source_bytes = 0
        record_ids: list[str] = []
        seen_record_ids: set[str] = set()
        family_counts: Counter[str] = Counter()
        case_counts: Counter[str] = Counter()

        try:
            with test_path.open("rb") as source, questions_path.open(
                "wb"
            ) as questions, answers_path.open("wb") as answers:
                for line_number, raw_line in enumerate(source, start=1):
                    source_digest.update(raw_line)
                    source_bytes += len(raw_line)
                    if not raw_line.strip():
                        raise S2EvalBuildError(
                            f"the test corpus has a blank record at line {line_number}"
                        )
                    try:
                        row = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise S2EvalBuildError(
                            f"invalid test JSON at line {line_number}: {error}"
                        ) from error
                    if not isinstance(row, Mapping):
                        raise S2EvalBuildError(
                            f"the test record at line {line_number} is not an object"
                        )
                    try:
                        question, answer = project_row(row, routes)
                    except (KeyError, TypeError, ValueError) as error:
                        raise S2EvalBuildError(
                            f"invalid S2 test record at line {line_number}: {error}"
                        ) from error
                    record_id = question["row_id"]
                    if record_id in seen_record_ids:
                        raise S2EvalBuildError(f"duplicate S2 row ID: {record_id}")
                    seen_record_ids.add(record_id)
                    record_ids.append(record_id)
                    family_counts[question["metadata"]["question_family"]] += 1
                    case_counts[answer["scoring_metadata"]["catalog_case_id"]] += 1
                    _write_record(questions, question_digest, question)
                    _write_record(answers, answer_digest, answer)
        except OSError as error:
            raise S2EvalBuildError(f"cannot build the S2 evaluator: {error}") from error

        if (
            source_digest.hexdigest() != test_spec["sha256"]
            or source_bytes != test_spec["bytes"]
            or len(record_ids) != test_spec["row_count"]
        ):
            raise S2EvalBuildError("the S2 test file differs from its production manifest")
        if set(family_counts) != set(EXPECTED_FAMILIES):
            raise S2EvalBuildError("the S2 test panel does not cover all 13 families")
        if dict(case_counts) != expected_case_counts:
            raise S2EvalBuildError("the S2 test panel changed its exact case quotas")
        if corpus_manifest_sha256 == DEFAULT_CORPUS_MANIFEST_SHA256 and (
            question_digest.hexdigest() != DEFAULT_QUESTIONS_SHA256
            or answer_digest.hexdigest() != DEFAULT_ANSWER_KEY_SHA256
        ):
            raise S2EvalBuildError("the canonical S2 evaluator artifacts changed")

        questions_path.chmod(0o640)
        answers_path.chmod(0o600)
        manifest = {
            "schema_version": EVALUATOR_SCHEMA_VERSION,
            "dataset_id": PRODUCTION_ID,
            "evaluation_contract": EVALUATION_CONTRACT,
            "source_corpus": {
                "manifest_path": _repo_relative(
                    corpus_manifest_path,
                    repo_root,
                    "corpus manifest",
                ),
                "manifest_sha256": corpus_manifest_sha256,
                "schema_version": CORPUS_SCHEMA_VERSION,
                "production_id": PRODUCTION_ID,
                "test_path": _repo_relative(test_path, repo_root, "test corpus"),
                "test_sha256": test_spec["sha256"],
                "panel_id": corpus_manifest["panel_id"],
            },
            "contract_routes": routes,
            "test": {
                "questions_path": "test_questions.jsonl",
                "questions_sha256": question_digest.hexdigest(),
                "answer_key_path": "test_answer_key.jsonl",
                "answer_key_sha256": answer_digest.hexdigest(),
                "test_panel_id": corpus_manifest["panel_id"],
                "row_count": len(record_ids),
                "record_ids_sha256": stable_sha256(record_ids),
                "family_counts": dict(sorted(family_counts.items())),
                "case_counts": dict(sorted(case_counts.items())),
            },
            "scoring": {
                "target_fields": list(SCORED_TARGET_KEYS),
                "headline": "structured_accuracy",
                "scores_summary_text": False,
                "scores_blocked_reason_text": False,
                "scores_blocked_reason_presence": True,
            },
            "answer_separation": {
                "public_metadata_fields": list(PUBLIC_METADATA_KEYS),
                "public_questions_end_at_current_user": True,
                "private_targets_contain_expected_tool_responses": True,
                "generation_reads_private_answer_keys": False,
            },
            "implementation_sha256": _implementation_pins(),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o640)
        _publish_immutable(temporary, output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the pinned evaluator build arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
    )
    parser.add_argument(
        "--corpus-manifest-sha256",
        default=DEFAULT_CORPUS_MANIFEST_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    """Build the evaluator and print its deterministic manifest."""

    args = parse_args()
    result = build_eval_bundle(
        corpus_manifest_path=args.corpus_manifest,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
