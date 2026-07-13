#!/usr/bin/env python3
"""Load and validate the declarative pre-trajectory SFT curriculum plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "pretrajectory-sft-curriculum-plan-v1"
EXPECTED_BOOK_MODES = {"closed_book", "open_book", "tool_call"}
EXPECTED_STAGE_NAMES = [
    "stage1_entity_schema",
    "stage2_topology_priors",
    "stage3_open_book_vectors",
    "stage4_module_world_model",
    "stage5_structured_tools",
    "stage6_blend",
]
EXPECTED_MIXTURE_BUCKETS = {
    "entity_normalization_schema",
    "layer_metadata_membership",
    "edge_neighbor_topology",
    "paths_layer_counts",
    "subgraph_components_hubness",
    "rwr_distance_vectors",
    "module_set_algebra",
    "global_cohesion_calibration",
    "structured_context_and_tools",
}
EXPECTED_CROSS_CELL_DIMENSIONS = {
    "stage",
    "question_family",
    "book_mode",
    "difficulty_source",
    "context_budget_profile",
    "module_source",
    "layer_family",
}
EXPECTED_COUNT_LIFECYCLE_FIELDS = {
    "requested",
    "generated",
    "compacted",
    "filtered",
    "selected",
}
EXPECTED_ARTIFACTS = {
    "curriculum_plan.json",
    "manifest.json",
    "coverage_report.json",
    "leakage_report.json",
    "audit_report.json",
}
EXPECTED_VARIANTS = {"closed_book", "open_book", "tool_call", "paraphrase", "page"}
EXPECTED_SPLITS = {"train", "val", "test"}
EXPECTED_METADATA_FIELDS = {
    "schema_version",
    "record_id",
    "oracle_fact_id",
    "book_mode",
    "question_family",
    "multiplex_id",
    "store_id",
    "flist_id",
    "layer_scope",
    "layer_ids",
    "layer_families",
    "entity_namespace",
    "module_source",
    "answer_format",
    "difficulty_source",
    "context_budget_profile",
    "evidence_handles",
    "provenance",
}
SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class CurriculumPlanValidationError(ValueError):
    """Raised when a curriculum plan violates one or more contract rules."""

    def __init__(self, path: Path, errors: Sequence[str]) -> None:
        self.path = path
        self.errors = tuple(errors)
        details = "\n".join(f"  - {error}" for error in self.errors)
        super().__init__(f"Invalid curriculum plan {path}:\n{details}")


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _list(value: Any) -> list[Any] | None:
    return value if isinstance(value, list) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _string_set(value: Any) -> set[str] | None:
    items = _list(value)
    if items is None or any(not isinstance(item, str) for item in items):
        return None
    return set(items)


def _require_root_sections(plan: Mapping[str, Any], errors: list[str]) -> None:
    required = {
        "contract_version",
        "plan_id",
        "dataset_schema_version",
        "graph_contract",
        "book_modes",
        "record_contract",
        "numeric_policy",
        "context_budget_profiles",
        "mixture",
        "build_profiles",
        "question_families",
        "stages",
        "coverage_contract",
        "split_contract",
        "artifact_contract",
        "promotion_contract",
    }
    for key in sorted(required - set(plan)):
        errors.append(f"root: missing required field '{key}'")


def _validate_graph_contract(plan: Mapping[str, Any], errors: list[str]) -> None:
    graph = _mapping(plan.get("graph_contract"))
    if graph is None:
        errors.append("graph_contract: expected an object")
        return
    for key in (
        "multiplex_id",
        "store_path",
        "flist_path",
        "entity_namespace",
        "mentor_ev_source",
        "rwr_loe_source",
    ):
        if not isinstance(graph.get(key), str) or not graph[key].strip():
            errors.append(f"graph_contract.{key}: expected a non-empty string")
    required_true = ("full_store_is_oracle", "exact_claims_require_store_identity")
    required_false = ("candidate_sampling_may_define_truth", "file_prefix_sampling_allowed")
    for key in required_true:
        if graph.get(key) is not True:
            errors.append(f"graph_contract.{key}: must be true")
    for key in required_false:
        if graph.get(key) is not False:
            errors.append(f"graph_contract.{key}: must be false")


def _validate_record_and_numeric_contracts(plan: Mapping[str, Any], errors: list[str]) -> None:
    record = _mapping(plan.get("record_contract"))
    if record is None:
        errors.append("record_contract: expected an object")
    else:
        fields = _string_set(record.get("required_metadata_fields"))
        if fields != EXPECTED_METADATA_FIELDS:
            errors.append(
                "record_contract.required_metadata_fields: must exactly cover "
                f"{sorted(EXPECTED_METADATA_FIELDS)}"
            )
        if record.get("answer_format") != "json":
            errors.append("record_contract.answer_format: must be 'json'")
        if record.get("canonical_entity_namespace") != "ensembl_gene_id_primary":
            errors.append(
                "record_contract.canonical_entity_namespace: must be 'ensembl_gene_id_primary'"
            )
        for key in (
            "raw_paths_forbidden_in_question_context_answer",
            "oracle_fact_id_assigned_before_rendering",
        ):
            if record.get(key) is not True:
                errors.append(f"record_contract.{key}: must be true")
    numeric = _mapping(plan.get("numeric_policy"))
    if numeric is None:
        errors.append("numeric_policy: expected an object")
        return
    for key in (
        "float_answers_require_declared_absolute_or_relative_tolerance",
        "derived_metrics_recomputed_by_validator",
    ):
        if numeric.get(key) is not True:
            errors.append(f"numeric_policy.{key}: must be true")
    if numeric.get("closed_book_arbitrary_floats_allowed") is not False:
        errors.append("numeric_policy.closed_book_arbitrary_floats_allowed: must be false")
    significant_figures = _number(numeric.get("default_significant_figures"))
    if significant_figures is None or not 3 <= significant_figures <= 5 or not significant_figures.is_integer():
        errors.append("numeric_policy.default_significant_figures: expected an integer from 3 to 5")
    for key in ("exact_types", "tolerance_types", "top_k_metrics"):
        if not _string_set(numeric.get(key)):
            errors.append(f"numeric_policy.{key}: expected a non-empty list of strings")


def _validate_budget_profiles(plan: Mapping[str, Any], errors: list[str]) -> set[str]:
    profiles = _mapping(plan.get("context_budget_profiles"))
    if profiles is None or not profiles:
        errors.append("context_budget_profiles: expected a non-empty object")
        return set()
    profile_names = set(profiles)
    for name, raw_profile in profiles.items():
        profile = _mapping(raw_profile)
        prefix = f"context_budget_profiles.{name}"
        if profile is None:
            errors.append(f"{prefix}: expected an object")
            continue
        values: dict[str, float] = {}
        for field in (
            "max_prompt_tokens",
            "max_answer_tokens",
            "max_total_tokens",
            "max_answer_characters",
        ):
            number = _number(profile.get(field))
            if number is None or number <= 0 or not number.is_integer():
                errors.append(f"{prefix}.{field}: expected a positive integer")
            else:
                values[field] = number
        if all(key in values for key in ("max_prompt_tokens", "max_answer_tokens", "max_total_tokens")):
            if values["max_prompt_tokens"] + values["max_answer_tokens"] > values["max_total_tokens"]:
                errors.append(
                    f"{prefix}: max_prompt_tokens + max_answer_tokens exceeds max_total_tokens"
                )
        if profile.get("paging_required_for_oversized_objects") is not True:
            errors.append(f"{prefix}.paging_required_for_oversized_objects: must be true")
    return profile_names


def _validate_mixture(plan: Mapping[str, Any], errors: list[str]) -> set[str]:
    mixture = _mapping(plan.get("mixture"))
    if mixture is None:
        errors.append("mixture: expected an object")
        return set()
    buckets = _mapping(mixture.get("content_buckets"))
    if buckets is None:
        errors.append("mixture.content_buckets: expected an object")
        bucket_names: set[str] = set()
    else:
        bucket_names = set(buckets)
        missing = EXPECTED_MIXTURE_BUCKETS - bucket_names
        extra = bucket_names - EXPECTED_MIXTURE_BUCKETS
        if missing:
            errors.append(f"mixture.content_buckets: missing buckets {sorted(missing)}")
        if extra:
            errors.append(f"mixture.content_buckets: unknown buckets {sorted(extra)}")
        weights: list[float] = []
        for name, value in buckets.items():
            number = _number(value)
            if number is None or number <= 0:
                errors.append(f"mixture.content_buckets.{name}: expected a positive fraction")
            else:
                weights.append(number)
        if len(weights) == len(buckets) and abs(sum(weights) - 1.0) > 1e-9:
            errors.append(
                f"mixture.content_buckets: fractions must sum to 1.0 (observed {sum(weights):.12g})"
            )
    dimensions = _string_set(mixture.get("orthogonal_dimensions"))
    if dimensions is None:
        errors.append("mixture.orthogonal_dimensions: expected a list of strings")
    elif dimensions != EXPECTED_CROSS_CELL_DIMENSIONS:
        errors.append(
            "mixture.orthogonal_dimensions: must exactly cover "
            f"{sorted(EXPECTED_CROSS_CELL_DIMENSIONS)}"
        )
    tolerance = _number(mixture.get("fraction_tolerance"))
    if tolerance is None or tolerance < 0 or tolerance > 0.05:
        errors.append("mixture.fraction_tolerance: expected a number between 0 and 0.05")
    return bucket_names


def _validate_build_profiles(plan: Mapping[str, Any], errors: list[str]) -> None:
    profiles = _mapping(plan.get("build_profiles"))
    if profiles is None or not profiles:
        errors.append("build_profiles: expected a non-empty object")
        return
    for name, raw_profile in profiles.items():
        prefix = f"build_profiles.{name}"
        profile = _mapping(raw_profile)
        if profile is None:
            errors.append(f"{prefix}: expected an object")
            continue
        counts = _mapping(profile.get("split_counts"))
        if counts is None or set(counts) != EXPECTED_SPLITS:
            errors.append(f"{prefix}.split_counts: must contain exactly train, val, and test")
        elif any(_number(value) is None or float(value) <= 0 or not float(value).is_integer() for value in counts.values()):
            errors.append(f"{prefix}.split_counts: every count must be a positive integer")
        for field in (
            "minimum_selected_per_required_family",
            "minimum_selected_per_material_cross_cell",
        ):
            value = _number(profile.get(field))
            if value is None or value <= 0 or not value.is_integer():
                errors.append(f"{prefix}.{field}: expected a positive integer")


def _validate_families(
    plan: Mapping[str, Any],
    bucket_names: set[str],
    errors: list[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]]]:
    raw_families = _list(plan.get("question_families"))
    if raw_families is None:
        errors.append("question_families: expected a list")
        return {}, {}
    by_name: dict[str, Mapping[str, Any]] = {}
    ids: list[int] = []
    primary_by_stage: dict[str, list[str]] = {}
    for index, raw_family in enumerate(raw_families):
        prefix = f"question_families[{index}]"
        family = _mapping(raw_family)
        if family is None:
            errors.append(f"{prefix}: expected an object")
            continue
        family_id = family.get("id")
        if isinstance(family_id, bool) or not isinstance(family_id, int):
            errors.append(f"{prefix}.id: expected an integer")
        else:
            ids.append(family_id)
        name = family.get("name")
        if not isinstance(name, str) or not SLUG_RE.fullmatch(name):
            errors.append(f"{prefix}.name: expected a snake_case slug")
            continue
        if name in by_name:
            errors.append(f"question_families: duplicate family name '{name}'")
        else:
            by_name[name] = family
        primary_stage = family.get("primary_stage")
        if primary_stage not in EXPECTED_STAGE_NAMES[:5]:
            errors.append(f"{prefix}.primary_stage: must name one of stages 1-5")
        else:
            primary_by_stage.setdefault(str(primary_stage), []).append(name)
        bucket = family.get("mixture_bucket")
        if bucket not in bucket_names:
            errors.append(f"{prefix}.mixture_bucket: unknown bucket '{bucket}'")
        modes = _string_set(family.get("allowed_book_modes"))
        if not modes:
            errors.append(f"{prefix}.allowed_book_modes: expected at least one mode")
        elif not modes <= EXPECTED_BOOK_MODES:
            errors.append(f"{prefix}.allowed_book_modes: unknown modes {sorted(modes - EXPECTED_BOOK_MODES)}")
        if not isinstance(family.get("difficulty_source"), str) or not str(
            family.get("difficulty_source", "")
        ).strip():
            errors.append(f"{prefix}.difficulty_source: expected a non-empty string")
    if len(ids) != len(set(ids)):
        errors.append("question_families: duplicate numeric family ids")
    expected_ids = set(range(1, 82))
    observed_ids = set(ids)
    if observed_ids != expected_ids:
        errors.append(
            "question_families: ids must cover 1..81 exactly; "
            f"missing={sorted(expected_ids - observed_ids)}, extra={sorted(observed_ids - expected_ids)}"
        )
    return by_name, primary_by_stage


def _validate_stages(
    plan: Mapping[str, Any],
    families: Mapping[str, Mapping[str, Any]],
    primary_by_stage: Mapping[str, list[str]],
    profile_names: set[str],
    errors: list[str],
) -> None:
    raw_stages = _list(plan.get("stages"))
    if raw_stages is None:
        errors.append("stages: expected a list")
        return
    observed_names: list[str] = []
    stages_by_name: dict[str, Mapping[str, Any]] = {}
    for offset, raw_stage in enumerate(raw_stages, start=1):
        prefix = f"stages[{offset - 1}]"
        stage = _mapping(raw_stage)
        if stage is None:
            errors.append(f"{prefix}: expected an object")
            continue
        name = stage.get("name")
        if not isinstance(name, str):
            errors.append(f"{prefix}.name: expected a string")
            continue
        observed_names.append(name)
        stages_by_name[name] = stage
        if stage.get("index") != offset:
            errors.append(f"{prefix}.index: expected {offset}")
        modes = _string_set(stage.get("allowed_book_modes"))
        if not modes:
            errors.append(f"{prefix}.allowed_book_modes: expected at least one mode")
            modes = set()
        elif not modes <= EXPECTED_BOOK_MODES:
            errors.append(f"{prefix}.allowed_book_modes: unknown modes {sorted(modes - EXPECTED_BOOK_MODES)}")
        budgets = _string_set(stage.get("allowed_budget_profiles"))
        if not budgets:
            errors.append(f"{prefix}.allowed_budget_profiles: expected at least one profile")
        elif not budgets <= profile_names:
            errors.append(f"{prefix}.allowed_budget_profiles: unknown profiles {sorted(budgets - profile_names)}")
        replay = _number(stage.get("earlier_stage_replay_fraction"))
        if replay is None or not 0 <= replay <= 1:
            errors.append(f"{prefix}.earlier_stage_replay_fraction: expected a number between 0 and 1")
        required = _string_set(stage.get("required_question_families"))
        if required is None:
            errors.append(f"{prefix}.required_question_families: expected a list of strings")
            required = set()
        if len(required) != len(stage.get("required_question_families", [])):
            errors.append(f"{prefix}.required_question_families: contains duplicates")
        unknown = required - set(families)
        if unknown:
            errors.append(f"{prefix}.required_question_families: unknown families {sorted(unknown)}")
        if name in EXPECTED_STAGE_NAMES[:5]:
            expected = set(primary_by_stage.get(name, []))
            if required != expected:
                errors.append(
                    f"{prefix}.required_question_families: must equal families with primary_stage={name}; "
                    f"missing={sorted(expected - required)}, extra={sorted(required - expected)}"
                )
            for family_name in sorted(required & set(families)):
                family_modes = _string_set(families[family_name].get("allowed_book_modes")) or set()
                if modes and not (family_modes & modes):
                    errors.append(
                        f"{prefix}: family '{family_name}' has no allowed book mode in its primary stage"
                    )
        elif name == "stage6_blend" and required:
            errors.append("stage6_blend.required_question_families: must be empty; stage6 is replay only")
    if observed_names != EXPECTED_STAGE_NAMES:
        errors.append(
            f"stages: expected ordered stages {EXPECTED_STAGE_NAMES}, observed {observed_names}"
        )
    blend = stages_by_name.get("stage6_blend")
    if blend is not None:
        weights = _mapping(blend.get("source_stage_weights"))
        expected_sources = set(EXPECTED_STAGE_NAMES[:5])
        if weights is None or set(weights) != expected_sources:
            errors.append("stage6_blend.source_stage_weights: must contain exactly stages 1-5")
        else:
            numeric_weights = [_number(value) for value in weights.values()]
            if any(value is None or value <= 0 for value in numeric_weights):
                errors.append("stage6_blend.source_stage_weights: every weight must be positive")
            elif abs(sum(value for value in numeric_weights if value is not None) - 1.0) > 1e-9:
                errors.append("stage6_blend.source_stage_weights: weights must sum to 1.0")


def _validate_coverage(plan: Mapping[str, Any], errors: list[str]) -> None:
    coverage = _mapping(plan.get("coverage_contract"))
    if coverage is None:
        errors.append("coverage_contract: expected an object")
        return
    if coverage.get("underfill_policy") != "fatal":
        errors.append("coverage_contract.underfill_policy: must be 'fatal'")
    if coverage.get("deterministic_sampling") is not True:
        errors.append("coverage_contract.deterministic_sampling: must be true")
    if coverage.get("sampling_independent_of_file_order") is not True:
        errors.append("coverage_contract.sampling_independent_of_file_order: must be true")
    dimensions = _string_set(coverage.get("required_cross_cell_dimensions"))
    if dimensions != EXPECTED_CROSS_CELL_DIMENSIONS:
        errors.append(
            "coverage_contract.required_cross_cell_dimensions: must exactly match the mixture dimensions"
        )
    lifecycle = _string_set(coverage.get("count_lifecycle_fields"))
    if lifecycle != EXPECTED_COUNT_LIFECYCLE_FIELDS:
        errors.append(
            f"coverage_contract.count_lifecycle_fields: must exactly cover {sorted(EXPECTED_COUNT_LIFECYCLE_FIELDS)}"
        )
    metrics = _string_set(coverage.get("required_unique_object_metrics"))
    if not metrics:
        errors.append("coverage_contract.required_unique_object_metrics: expected a non-empty list")
    polarities = _string_set(coverage.get("required_polarities"))
    if polarities != {"positive", "negative", "insufficient_context"}:
        errors.append(
            "coverage_contract.required_polarities: must contain positive, negative, and insufficient_context"
        )


def _validate_split(plan: Mapping[str, Any], errors: list[str]) -> None:
    split = _mapping(plan.get("split_contract"))
    if split is None:
        errors.append("split_contract: expected an object")
        return
    if split.get("split_unit") != "oracle_fact_group":
        errors.append("split_contract.split_unit: must be 'oracle_fact_group'")
    if split.get("primary_group_key") != "oracle_fact_id":
        errors.append("split_contract.primary_group_key: must be 'oracle_fact_id'")
    if split.get("render_variants_same_split") is not True:
        errors.append("split_contract.render_variants_same_split: must be true")
    variants = _string_set(split.get("variants"))
    if variants != EXPECTED_VARIANTS:
        errors.append(f"split_contract.variants: must exactly cover {sorted(EXPECTED_VARIANTS)}")
    fractions = _mapping(split.get("assignment_fractions"))
    if fractions is None or set(fractions) != EXPECTED_SPLITS:
        errors.append("split_contract.assignment_fractions: must contain exactly train, val, and test")
    else:
        values = [_number(value) for value in fractions.values()]
        if any(value is None or value <= 0 for value in values):
            errors.append("split_contract.assignment_fractions: every fraction must be positive")
        elif abs(sum(value for value in values if value is not None) - 1.0) > 1e-9:
            errors.append("split_contract.assignment_fractions: fractions must sum to 1.0")
    leakage = _string_set(split.get("fatal_leakage_checks"))
    required_leakage = {"oracle_fact_cross_split", "exact_duplicate_cross_split", "near_duplicate_cross_split"}
    if leakage is None or not required_leakage <= leakage:
        errors.append(
            f"split_contract.fatal_leakage_checks: missing required checks {sorted(required_leakage - (leakage or set()))}"
        )


def _validate_artifacts(plan: Mapping[str, Any], errors: list[str]) -> None:
    artifact = _mapping(plan.get("artifact_contract"))
    if artifact is None:
        errors.append("artifact_contract: expected an object")
        return
    files = _string_set(artifact.get("required_dataset_files"))
    if files != EXPECTED_ARTIFACTS:
        errors.append(f"artifact_contract.required_dataset_files: must exactly cover {sorted(EXPECTED_ARTIFACTS)}")
    for key in (
        "manifest_must_embed_plan_hash",
        "reports_must_embed_plan_hash",
        "raw_filesystem_paths_forbidden_in_model_text",
        "tool_rows_require_live_schema_validation",
    ):
        if artifact.get(key) is not True:
            errors.append(f"artifact_contract.{key}: must be true")
    if artifact.get("tool_schema_validity_required") != 1.0:
        errors.append("artifact_contract.tool_schema_validity_required: must equal 1.0")


def _validate_promotion(plan: Mapping[str, Any], errors: list[str]) -> None:
    promotion = _mapping(plan.get("promotion_contract"))
    if promotion is None:
        errors.append("promotion_contract: expected an object")
        return
    minimum = _number(promotion.get("minimum_examples_per_reported_metric"))
    if minimum is None or minimum < 100 or not minimum.is_integer():
        errors.append(
            "promotion_contract.minimum_examples_per_reported_metric: expected an integer >= 100"
        )
    regression = _number(promotion.get("maximum_absolute_regression_on_any_earlier_stage"))
    if regression is None or regression < 0 or regression > 0.05:
        errors.append(
            "promotion_contract.maximum_absolute_regression_on_any_earlier_stage: expected 0..0.05"
        )
    for key in ("gold_self_evaluation_required", "zero_fatal_dataset_audit_errors_required"):
        if promotion.get(key) is not True:
            errors.append(f"promotion_contract.{key}: must be true")
    gates = _mapping(promotion.get("stages"))
    if gates is None or set(gates) != set(EXPECTED_STAGE_NAMES):
        errors.append("promotion_contract.stages: must define gates for every stage including blend")
        return
    for stage_name, raw_stage_gates in gates.items():
        stage_gates = _mapping(raw_stage_gates)
        if stage_gates is None or not stage_gates:
            errors.append(f"promotion_contract.stages.{stage_name}: expected a non-empty gate object")
            continue
        for gate_name, threshold in stage_gates.items():
            if gate_name == "all_prior_stage_gates_still_pass":
                if threshold is not True:
                    errors.append(
                        "promotion_contract.stages.stage6_blend.all_prior_stage_gates_still_pass: must be true"
                    )
                continue
            value = _number(threshold)
            if value is None or not 0 <= value <= 1:
                errors.append(
                    f"promotion_contract.stages.{stage_name}.{gate_name}: expected a threshold in [0, 1]"
                )


def validate_curriculum_plan(plan: Any) -> list[str]:
    """Return all contract errors without mutating or partially accepting the plan."""

    if not isinstance(plan, Mapping):
        return ["root: expected a JSON object"]
    errors: list[str] = []
    _require_root_sections(plan, errors)
    if plan.get("contract_version") != CONTRACT_VERSION:
        errors.append(
            f"contract_version: expected '{CONTRACT_VERSION}', observed {plan.get('contract_version')!r}"
        )
    for key in ("plan_id", "dataset_schema_version"):
        if not isinstance(plan.get(key), str) or not str(plan.get(key, "")).strip():
            errors.append(f"{key}: expected a non-empty string")
    book_modes = _string_set(plan.get("book_modes"))
    if book_modes != EXPECTED_BOOK_MODES:
        errors.append(f"book_modes: must exactly cover {sorted(EXPECTED_BOOK_MODES)}")
    _validate_graph_contract(plan, errors)
    _validate_record_and_numeric_contracts(plan, errors)
    profile_names = _validate_budget_profiles(plan, errors)
    bucket_names = _validate_mixture(plan, errors)
    _validate_build_profiles(plan, errors)
    families, primary_by_stage = _validate_families(plan, bucket_names, errors)
    _validate_stages(plan, families, primary_by_stage, profile_names, errors)
    _validate_coverage(plan, errors)
    _validate_split(plan, errors)
    _validate_artifacts(plan, errors)
    _validate_promotion(plan, errors)
    return errors


def curriculum_plan_hash(plan: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 to embed in generated dataset artifacts."""

    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_curriculum_plan(path: Path | str) -> dict[str, Any]:
    """Load a plan and raise one aggregate, actionable error on contract failure."""

    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CurriculumPlanValidationError(plan_path, ["file does not exist"]) from exc
    except json.JSONDecodeError as exc:
        raise CurriculumPlanValidationError(
            plan_path,
            [f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"],
        ) from exc
    errors = validate_curriculum_plan(payload)
    if errors:
        raise CurriculumPlanValidationError(plan_path, errors)
    return dict(payload)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan",
        nargs="?",
        type=Path,
        default=repo_root / "config" / "pretrajectory_sft_curriculum_v1.json",
        help="Curriculum plan JSON (defaults to the source-controlled v1 plan).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the validation summary as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        plan = load_curriculum_plan(args.plan)
    except CurriculumPlanValidationError as exc:
        if args.json:
            print(
                json.dumps(
                    {"valid": False, "plan_path": str(exc.path), "errors": list(exc.errors)},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(str(exc), file=sys.stderr)
        return 2
    summary = {
        "valid": True,
        "plan_path": str(args.plan),
        "plan_id": plan["plan_id"],
        "contract_version": plan["contract_version"],
        "plan_hash": curriculum_plan_hash(plan),
        "question_family_count": len(plan["question_families"]),
        "stage_count": len(plan["stages"]),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "Valid curriculum plan "
            f"{summary['plan_id']} ({summary['question_family_count']} families, "
            f"{summary['stage_count']} stages, sha256={summary['plan_hash']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
