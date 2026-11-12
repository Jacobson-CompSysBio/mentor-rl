#!/usr/bin/env python3
"""Build an answer-free S2 generation bundle with isolated tool fixtures."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, BinaryIO

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.s2_production_adapter import tools_sha256  # noqa: E402
from scripts.build_world_model_s2_eval_bundle import (  # noqa: E402
    EVALUATION_CONTRACT, EVALUATOR_SCHEMA_VERSION, EXPECTED_FAMILIES,
    PUBLIC_METADATA_KEYS, TARGET_KEYS, ordered_json, sha256_file, stable_sha256,
)

GENERATION_BUNDLE_SCHEMA_VERSION = "mentor-rl-world-model-s2-generation-bundle-v1"
TOOL_FIXTURE_SCHEMA_VERSION = "mentor-rl-world-model-s2-tool-fixtures-v1"


class S2GenerationBundleError(RuntimeError):
    """Report an unsafe evaluator input or generation-bundle output."""


def _write_record(destination: BinaryIO, digest: Any, value: Mapping[str, Any]) -> None:
    """Write one finite ordered JSONL record while computing its identity."""
    encoded = (ordered_json(value) + "\n").encode("utf-8")
    destination.write(encoded)
    digest.update(encoded)


def _validate_question(row: Any, routes: Mapping[str, Any]) -> dict[str, Any]:
    """Return one safe public S2 question bound to its family route."""
    if not isinstance(row, Mapping) or set(row) != {"row_id", "messages", "tools", "metadata"}:
        raise S2GenerationBundleError("an S2 question has an unsafe field set")
    row_id, messages, tools, metadata = (row.get(key) for key in ("row_id", "messages", "tools", "metadata"))
    if not isinstance(row_id, str) or not row_id:
        raise S2GenerationBundleError("an S2 question has no row ID")
    if (not isinstance(messages, list)
            or any(not isinstance(message, Mapping) for message in messages)
            or [message.get("role") for message in messages] != ["system", "harness", "user"]
            or any(set(message) != {"role", "content"} or not isinstance(message.get("content"), str)
                   for message in messages)
            or not isinstance(tools, list) or not isinstance(metadata, Mapping)
            or set(metadata) != set(PUBLIC_METADATA_KEYS) or metadata.get("split") != "test"):
        raise S2GenerationBundleError(f"{row_id} has an unsafe public prompt")
    family = metadata.get("question_family")
    route = routes.get(family) if isinstance(family, str) else None
    if not isinstance(route, Mapping):
        raise S2GenerationBundleError(f"{row_id} has no contract route")
    prompt_sha256 = hashlib.sha256(str(messages[0].get("content", "")).encode()).hexdigest()
    if (metadata.get("tool_contract_version") != route.get("contract_version")
            or prompt_sha256 != route.get("system_prompt_sha256")
            or tools_sha256(tools) != route.get("tools_sha256")):
        raise S2GenerationBundleError(f"{row_id} differs from its family route")
    return {str(key): value for key, value in row.items()}


def _tool_fixture(answer: Any, question: Mapping[str, Any]) -> dict[str, Any]:
    """Extract only ordered calls and responses needed by the offline harness."""
    if not isinstance(answer, Mapping) or set(answer) != {"row_id", "scoring_metadata", "target"}:
        raise S2GenerationBundleError("an S2 private answer has an invalid shape")
    if answer.get("row_id") != question.get("row_id"):
        raise S2GenerationBundleError("an S2 answer does not match its public question")
    target = answer.get("target")
    if not isinstance(target, Mapping) or set(target) != set(TARGET_KEYS):
        raise S2GenerationBundleError("an S2 private structured target changed")
    calls, responses = target.get("expected_tool_calls"), target.get("expected_tool_responses")
    if not isinstance(calls, list) or not isinstance(responses, list) or len(calls) != len(responses):
        raise S2GenerationBundleError("an S2 tool trajectory is incomplete")
    steps = []
    for call, response in zip(calls, responses, strict=True):
        if (not isinstance(call, Mapping) or set(call) != {"name", "arguments"}
                or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), Mapping)
                or not isinstance(response, Mapping) or set(response) != {"name", "content"}
                or response.get("name") != call.get("name") or not isinstance(response.get("content"), Mapping)):
            raise S2GenerationBundleError("an S2 tool fixture has an invalid pair")
        steps.append({"tool_call": dict(call), "tool_response": dict(response)})
    return {"schema_version": TOOL_FIXTURE_SCHEMA_VERSION,
            "row_id": question["row_id"], "steps": steps}


def _publish_immutable(temporary: Path, output_dir: Path) -> None:
    """Publish atomically, accepting an existing output only when identical."""
    filenames = {"manifest.json", "questions.jsonl", "tool_fixtures.jsonl"}
    if not output_dir.exists():
        os.replace(temporary, output_dir)
        return
    if not output_dir.is_dir():
        raise S2GenerationBundleError("the generation output is not a directory")
    actual = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()}
    if actual != filenames:
        raise S2GenerationBundleError("the immutable generation file set changed")
    for filename in filenames:
        if sha256_file(output_dir / filename) != sha256_file(temporary / filename):
            raise S2GenerationBundleError("the immutable generation bundle changed")


def build_generation_bundle(*, evaluator_manifest_path: Path,
                            evaluator_manifest_sha256: str, output_dir: Path) -> dict[str, Any]:
    """Validate an S2 evaluator and publish public prompts plus tool fixtures."""
    evaluator_manifest_path = evaluator_manifest_path.resolve()
    if sha256_file(evaluator_manifest_path) != evaluator_manifest_sha256:
        raise S2GenerationBundleError("the S2 evaluator manifest identity changed")
    try:
        evaluator = json.loads(evaluator_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise S2GenerationBundleError(f"cannot read the S2 evaluator manifest: {error}") from error
    test = evaluator.get("test") if isinstance(evaluator, Mapping) else None
    routes = evaluator.get("contract_routes") if isinstance(evaluator, Mapping) else None
    if (evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
            or evaluator.get("evaluation_contract") != EVALUATION_CONTRACT
            or not isinstance(test, Mapping) or not isinstance(routes, Mapping)
            or set(routes) != set(EXPECTED_FAMILIES)):
        raise S2GenerationBundleError("the S2 evaluator contract changed")
    root = evaluator_manifest_path.parent
    questions_source = root / str(test.get("questions_path", ""))
    answers_source = root / str(test.get("answer_key_path", ""))
    try:
        questions_source.resolve().relative_to(root.resolve())
        answers_source.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise S2GenerationBundleError("an S2 evaluator file escapes its root") from error
    if (sha256_file(questions_source) != test.get("questions_sha256")
            or sha256_file(answers_source) != test.get("answer_key_sha256")):
        raise S2GenerationBundleError("an S2 evaluator panel file changed")

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.building-", dir=output_dir.parent) as name:
        temporary = Path(name)
        temporary.chmod(0o700)
        questions_output, fixtures_output = temporary / "questions.jsonl", temporary / "tool_fixtures.jsonl"
        question_digest, fixture_digest = hashlib.sha256(), hashlib.sha256()
        record_ids, seen, family_counts = [], set(), Counter()
        maximum_tool_calls = 0
        with questions_source.open() as questions, answers_source.open() as answers, \
                questions_output.open("wb") as public, fixtures_output.open("wb") as fixtures:
            for line_number, pair in enumerate(zip_longest(questions, answers), start=1):
                question_line, answer_line = pair
                if question_line is None or answer_line is None:
                    raise S2GenerationBundleError("the S2 question and answer counts differ")
                try:
                    question = _validate_question(json.loads(question_line), routes)
                    fixture = _tool_fixture(json.loads(answer_line), question)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise S2GenerationBundleError(f"invalid S2 evaluator row {line_number}: {error}") from error
                row_id = str(question["row_id"])
                if row_id in seen:
                    raise S2GenerationBundleError(f"duplicate S2 generation row {row_id}")
                seen.add(row_id); record_ids.append(row_id)
                family_counts[str(question["metadata"]["question_family"])] += 1
                maximum_tool_calls = max(maximum_tool_calls, len(fixture["steps"]))
                _write_record(public, question_digest, question)
                _write_record(fixtures, fixture_digest, fixture)
        if (len(record_ids) != test.get("row_count")
                or stable_sha256(record_ids) != test.get("record_ids_sha256")
                or dict(sorted(family_counts.items())) != test.get("family_counts")):
            raise S2GenerationBundleError("the S2 generation record set changed")
        questions_output.chmod(0o640); fixtures_output.chmod(0o600)
        manifest = {"schema_version": GENERATION_BUNDLE_SCHEMA_VERSION,
            "dataset_id": evaluator.get("dataset_id"), "evaluation_contract": EVALUATION_CONTRACT,
            "test_panel_id": test.get("test_panel_id"), "record_count": len(record_ids),
            "record_ids_sha256": stable_sha256(record_ids),
            "family_counts": dict(sorted(family_counts.items())),
            "source_evaluator_manifest_sha256": evaluator_manifest_sha256,
            "source_questions_sha256": test.get("questions_sha256"),
            "source_answer_key_sha256": test.get("answer_key_sha256"),
            "questions_sha256": question_digest.hexdigest(),
            "tool_fixtures_sha256": fixture_digest.hexdigest(),
            "tool_fixture_schema_version": TOOL_FIXTURE_SCHEMA_VERSION,
            "maximum_tool_calls": maximum_tool_calls, "contract_routes": dict(routes),
            "native_transcript": True, "reads_private_answer_keys": False,
            "tool_fixtures_contain_final_targets": False}
        manifest["bundle_sha256"] = stable_sha256(manifest)
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (temporary / "manifest.json").chmod(0o640)
        _publish_immutable(temporary, output_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse evaluator-to-generation build inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Build one generation bundle and print its identity."""
    args = parse_args()
    result = build_generation_bundle(evaluator_manifest_path=args.evaluator_manifest,
        evaluator_manifest_sha256=args.evaluator_manifest_sha256, output_dir=args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
