#!/usr/bin/env python3
"""Build the canonical S0 trajectory panel for the named registry."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_world_model_v2_s0_tool_sft import (  # noqa: E402
    CONFIG_SCHEMA_VERSION,
    FAMILIES,
    answer_key_row,
    build_components,
    make_test_question,
    protect_evaluator_dir,
    read_json,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    trajectory_fields,
    write_json,
    write_jsonl,
)
from runtime.world_model_s0_tools import S0IdentifierToolRuntime  # noqa: E402


CONFIG_SCHEMA_VERSION_FULL = (
    "mentor-rl-world-model-s0-full-tool-trajectory-eval-build-v6"
)
EVALUATOR_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-full-tool-trajectory-evaluator-manifest-v6"
)
FULL_EVALUATION_CONTRACT = (
    "full_registry_tool_trajectory_v1"
)
FULL_DATASET_ID = (
    "world_model_v2_s0_human_identifier_trajectories_full_registry_v6"
)


def _answer_key(
    question: Mapping[str, Any],
    fact: Mapping[str, Any],
    runtime: S0IdentifierToolRuntime,
) -> dict[str, Any]:
    """Create one answer row for the trajectory contract."""

    answer = answer_key_row(
        question,
        fact,
        runtime,
    )
    trajectory_targets = trajectory_fields(fact)
    answer.update(
        {
            "expected_assistant_thinking": trajectory_targets[
                "assistant_thinking"
            ],
            "expected_final_answer": trajectory_targets[
                "assistant_final"
            ],
        }
    )
    return answer


def _require_object(value: Any, label: str) -> dict[str, Any]:
    """Return one required object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one object")
    return {str(key): item for key, item in value.items()}


def _population_filter(
    config: Mapping[str, Any],
) -> tuple[int | None, frozenset[str]]:
    """Return the optional ambiguous-row filter contract."""

    value = config.get("population_filter")
    if value is None:
        return None, frozenset()
    declaration = _require_object(value, "population_filter")
    maximum = declaration.get("maximum_ambiguous_candidate_gene_ids")
    expected = declaration.get("expected_excluded_fact_ids")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 1
    ):
        raise ValueError(
            "maximum_ambiguous_candidate_gene_ids must be positive"
        )
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(item, str) or not item for item in expected)
        or len(set(expected)) != len(expected)
    ):
        raise ValueError(
            "expected_excluded_fact_ids must contain unique fact IDs"
        )
    return maximum, frozenset(expected)


def build_full_evaluator(config_path: Path) -> dict[str, Any]:
    """Build the full registry evaluation panel."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION_FULL:
        raise ValueError("The full evaluation config schema changed")

    source_decl = _require_object(
        config.get("source_build_config"),
        "source_build_config",
    )
    source_path = resolve_repo_path(source_decl.get("path"))
    expected_source_hash = str(source_decl.get("sha256", ""))
    if sha256_file(source_path) != expected_source_hash:
        raise ValueError("The source build config identity changed")
    source = read_json(source_path)
    if source.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("The source build config schema changed")

    registry_decl = _require_object(
        source.get("identifier_registry"),
        "identifier_registry",
    )
    registry_path = resolve_repo_path(registry_decl.get("path"))
    registry_sha256 = str(registry_decl.get("expected_sha256", ""))
    if sha256_file(registry_path) != registry_sha256:
        raise ValueError("The identifier registry identity changed")
    registry = read_json(registry_path)
    mappings = registry.get("gene_symbols_by_id")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The identifier registry has no mappings")

    runtime = S0IdentifierToolRuntime.from_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )
    all_facts = sorted(
        (
            fact
            for component in build_components(mappings)
            for fact in component["facts"]
        ),
        key=lambda fact: str(fact["fact_id"]),
    )
    registry_id = f"sha256:{registry_sha256}"
    maximum_candidates, expected_excluded = _population_filter(config)
    question_facts = [
        (
            make_test_question(
                fact,
                config=source,
                registry_id=registry_id,
                fact_role="registry_population",
            ),
            fact,
        )
        for fact in all_facts
    ]
    excluded_pairs = []
    if maximum_candidates is not None:
        excluded_pairs = [
            (question, fact)
            for question, fact in question_facts
            if fact["family"] == FAMILIES[2]
            and len(fact["answer"]["candidate_gene_ids"])
            > maximum_candidates
        ]
    actual_excluded = frozenset(
        str(fact["fact_id"]) for _, fact in excluded_pairs
    )
    if actual_excluded != expected_excluded:
        raise ValueError(
            "The excluded full-panel facts changed: "
            f"expected={sorted(expected_excluded)} "
            f"actual={sorted(actual_excluded)}"
        )
    excluded_fact_ids = actual_excluded
    kept_pairs = [
        (question, fact)
        for question, fact in question_facts
        if fact["fact_id"] not in excluded_fact_ids
    ]
    questions = [question for question, _ in kept_pairs]
    answers = [
        _answer_key(
            question,
            fact,
            runtime,
        )
        for question, fact in kept_pairs
    ]
    excluded_records = sorted(
        (
            {
                "record_id": str(question["record_id"]),
                "fact_id": str(fact["fact_id"]),
                "family": str(fact["family"]),
                "gene_symbol": str(fact["input"]["gene_symbol"]),
                "candidate_gene_id_count": len(
                    fact["answer"]["candidate_gene_ids"]
                ),
            }
            for question, fact in excluded_pairs
        ),
        key=lambda row: str(row["record_id"]),
    )
    record_ids = [str(row["record_id"]) for row in questions]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("The full panel record IDs are not unique")

    outputs = _require_object(config.get("outputs"), "outputs")
    evaluator_dir = resolve_repo_path(outputs.get("evaluator_dir"))
    questions_path = evaluator_dir / "test_questions.jsonl"
    answers_path = evaluator_dir / "test_answer_key.jsonl"
    manifest_path = evaluator_dir / "manifest.json"
    write_jsonl(questions_path, questions)
    write_jsonl(answers_path, answers)

    family_counts = {
        family: sum(
            row["metadata"]["question_family"] == family
            for row in questions
        )
        for family in FAMILIES
    }
    population = {
        "scope": "all_named_gene_ids_and_all_gene_symbols",
        "row_count": len(questions),
        "family_counts": family_counts,
        "record_ids_sha256": stable_sha256(record_ids),
        "excluded_gene_ids_without_name": registry.get("counts", {}).get(
            "excluded_gene_ids_without_name"
        ),
    }
    if maximum_candidates is not None:
        population.update(
            {
                "scope": (
                    "all_named_gene_ids_and_all_gene_symbols_"
                    "except_declared_long_ambiguities"
                ),
                "population_filter": {
                    "maximum_ambiguous_candidate_gene_ids": (
                        maximum_candidates
                    ),
                },
                "excluded_records": excluded_records,
            }
        )
    manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "dataset_id": FULL_DATASET_ID,
        "source_dataset_id": source["dataset_id"],
        "evaluation_id": config["evaluation_id"],
        "evaluation_contract": FULL_EVALUATION_CONTRACT,
        "source_build_config": {
            "path": str(source_path.relative_to(REPO_ROOT)),
            "sha256": expected_source_hash,
        },
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
        },
        "population": population,
        "test": {
            "questions_path": str(questions_path.relative_to(REPO_ROOT)),
            "questions_sha256": sha256_file(questions_path),
            "answer_key_path": str(answers_path.relative_to(REPO_ROOT)),
            "answer_key_sha256": sha256_file(answers_path),
            "row_count": len(questions),
            "test_panel_id": sha256_file(questions_path),
        },
    }
    write_json(manifest_path, manifest)
    protect_evaluator_dir(evaluator_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "config/"
            "world_model_v2_s0_full_registry_tool_trajectory_"
            "eval_build_v6_sans_3_long_rows.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured full evaluation panel."""

    manifest = build_full_evaluator(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
