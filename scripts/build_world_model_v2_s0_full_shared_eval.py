#!/usr/bin/env python3
"""Build the full S0 shared trace evaluation panel."""

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

from runtime.world_model_contracts import (  # noqa: E402
    SUPERVISED_TRACE_SCHEMA_VERSION,
    WORLD_MODEL_ANSWER_SCHEMA_VERSION,
)
from runtime.world_model_s0_tools import (  # noqa: E402
    S0IdentifierToolRuntime,
)
from scripts.build_world_model_v2_s0_shared_sft import (  # noqa: E402
    CONFIG_SCHEMA_VERSION,
    DATASET_ID,
    answer_key_row,
    assign_not_found_facts,
    load_previous_symbol_pool,
    make_trace,
    public_question_row,
)
from scripts.build_world_model_v2_s0_tool_sft import (  # noqa: E402
    FAMILIES,
    build_components,
    protect_evaluator_dir,
    read_json,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    write_json,
    write_jsonl,
)


CONFIG_SCHEMA_VERSION_FULL = (
    "mentor-rl-world-model-s0-full-shared-eval-build-v7"
)
EVALUATOR_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-full-shared-evaluator-manifest-v7"
)
FULL_EVALUATION_CONTRACT = "full_registry_supervised_trace_v1"
FULL_DATASET_ID = (
    "world_model_v2_s0_human_identifier_supervised_traces_full_registry_v7"
)


def _require_object(value: Any, label: str) -> dict[str, Any]:
    """Return one required object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one object")
    return {str(key): item for key, item in value.items()}


def _population_filter(
    config: Mapping[str, Any],
) -> tuple[int, frozenset[str]]:
    """Return the required ambiguous-row filter."""

    declaration = _require_object(
        config.get("population_filter"),
        "population_filter",
    )
    maximum = declaration.get("maximum_ambiguous_candidate_gene_ids")
    expected = declaration.get("expected_excluded_fact_ids")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
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
    """Build all registry rows and the fixed absent-symbol panel."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION_FULL:
        raise ValueError("The full shared evaluation config schema changed")

    source_decl = _require_object(
        config.get("source_build_config"),
        "source_build_config",
    )
    source_path = resolve_repo_path(source_decl.get("path"))
    source_sha256 = str(source_decl.get("sha256", ""))
    if sha256_file(source_path) != source_sha256:
        raise ValueError("The source build config identity changed")
    source = read_json(source_path)
    if (
        source.get("schema_version") != CONFIG_SCHEMA_VERSION
        or source.get("dataset_id") != DATASET_ID
    ):
        raise ValueError("The source build config contract changed")

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

    natural_facts = sorted(
        (
            fact
            for component in build_components(mappings)
            for fact in component["facts"]
        ),
        key=lambda fact: str(fact["fact_id"]),
    )
    maximum_candidates, expected_excluded = _population_filter(config)
    excluded_facts = [
        fact
        for fact in natural_facts
        if fact["family"] == FAMILIES[2]
        and len(fact["answer"]["candidate_gene_ids"]) > maximum_candidates
    ]
    actual_excluded = frozenset(
        str(fact["fact_id"]) for fact in excluded_facts
    )
    if actual_excluded != expected_excluded:
        raise ValueError(
            "The excluded full-panel facts changed: "
            f"expected={sorted(expected_excluded)} "
            f"actual={sorted(actual_excluded)}"
        )
    natural_facts = [
        fact for fact in natural_facts if fact["fact_id"] not in actual_excluded
    ]

    missing_config = _require_object(
        source.get("not_found_symbols"),
        "not_found_symbols",
    )
    hgnc_path = resolve_repo_path(missing_config.get("path"))
    hgnc_sha256 = str(missing_config.get("expected_sha256", ""))
    current_symbols = {
        str(symbol)
        for symbols in mappings.values()
        for symbol in symbols
    }
    previous_symbols = load_previous_symbol_pool(
        hgnc_path,
        expected_sha256=hgnc_sha256,
        current_symbols=current_symbols,
    )
    absent_facts = assign_not_found_facts(previous_symbols, missing_config)[
        "test"
    ]
    facts = sorted(
        natural_facts + absent_facts,
        key=lambda fact: str(fact["fact_id"]),
    )

    registry_id = f"sha256:{registry_sha256}"
    traces = [
        make_trace(
            fact,
            config=source,
            registry_id=registry_id,
            prompt_form="test",
            split="test",
            fact_role="registry_population",
            runtime=runtime,
        )
        for fact in facts
    ]
    questions = [
        public_question_row(trace, fact, split="test")
        for trace, fact in zip(traces, facts, strict=True)
    ]
    answers = [
        answer_key_row(trace, fact)
        for trace, fact in zip(traces, facts, strict=True)
    ]
    record_ids = [trace.record_id for trace in traces]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("The full panel record IDs are not unique")

    excluded_records = sorted(
        (
            {
                "record_id": make_trace(
                    fact,
                    config=source,
                    registry_id=registry_id,
                    prompt_form="test",
                    split="test",
                    fact_role="registry_population",
                    runtime=runtime,
                ).record_id,
                "fact_id": str(fact["fact_id"]),
                "family": str(fact["family"]),
                "gene_symbol": str(fact["input"]["gene_symbol"]),
                "candidate_gene_id_count": len(
                    fact["answer"]["candidate_gene_ids"]
                ),
            }
            for fact in excluded_facts
        ),
        key=lambda row: str(row["record_id"]),
    )

    outputs = _require_object(config.get("outputs"), "outputs")
    evaluator_dir = resolve_repo_path(outputs.get("evaluator_dir"))
    questions_path = evaluator_dir / "test_questions.jsonl"
    answers_path = evaluator_dir / "test_answer_key.jsonl"
    manifest_path = evaluator_dir / "manifest.json"
    write_jsonl(questions_path, questions)
    write_jsonl(answers_path, answers)

    family_counts = {
        family: sum(
            trace.metadata["question_family"] == family for trace in traces
        )
        for family in FAMILIES
    }
    status_counts = {"resolved": 0, "ambiguous": 0, "not_found": 0}
    for trace in traces:
        status_counts[trace.final_answer.status] += 1
    population = {
        "scope": (
            "all_registry_gene_ids_and_symbols_except_declared_long_"
            "ambiguities_plus_fixed_absent_symbol_panel"
        ),
        "row_count": len(traces),
        "family_counts": family_counts,
        "status_counts": status_counts,
        "record_ids_sha256": stable_sha256(record_ids),
        "registry_gene_id_count": len(mappings),
        "registry_symbol_count": int(
            registry["counts"]["unique_gene_names"]
        ),
        "unnamed_gene_id_count": int(
            registry["counts"]["gene_ids_without_name"]
        ),
        "fixed_not_found_symbol_count": len(absent_facts),
        "population_filter": {
            "maximum_ambiguous_candidate_gene_ids": maximum_candidates,
        },
        "excluded_records": excluded_records,
    }
    manifest = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "dataset_id": FULL_DATASET_ID,
        "source_dataset_id": DATASET_ID,
        "evaluation_id": config["evaluation_id"],
        "evaluation_contract": FULL_EVALUATION_CONTRACT,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "answer_schema_version": WORLD_MODEL_ANSWER_SCHEMA_VERSION,
        "source_build_config": {
            "path": str(source_path.relative_to(REPO_ROOT)),
            "sha256": source_sha256,
        },
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
        },
        "not_found_symbol_source": {
            "path": str(hgnc_path.relative_to(REPO_ROOT)),
            "sha256": hgnc_sha256,
        },
        "population": population,
        "test": {
            "questions_path": str(questions_path.relative_to(REPO_ROOT)),
            "questions_sha256": sha256_file(questions_path),
            "answer_key_path": str(answers_path.relative_to(REPO_ROOT)),
            "answer_key_sha256": sha256_file(answers_path),
            "row_count": len(traces),
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
            / "config/world_model_v2_s0_full_shared_eval_build_v7.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured full shared panel."""

    manifest = build_full_evaluator(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
