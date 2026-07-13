#!/usr/bin/env python3
"""Evaluate pre-trajectory SFT predictions with exact graph-fact checks."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


GENE_RE = re.compile(r"\bENSG\d{11}\b")
MODULE_RE = re.compile(r"\b(?:rwr_loe_module|gw_dendrogram_module|mentor_ev_module)_\d+\b")
_LAYER_COMPONENT = r"[A-Za-z0-9_-](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?"
LAYER_RE = re.compile(
    rf"(?<![A-Za-z0-9_.:-])({_LAYER_COMPONENT}(?::{_LAYER_COMPONENT})+)"
    rf"(?![A-Za-z0-9_:-]|\.(?=[A-Za-z0-9_-]))"
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z0-9_]|\.(?=[A-Za-z0-9_]))"
)
FINAL_MARKER_RE = re.compile(r"(?im)(?:^|\n)\s*final\b[:\s]*")
SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")

EVALUATOR_CONTRACT_VERSION = "pretrajectory-sft-exact-v3"
PLAN_DATASET_SCHEMA_VERSION = "pretrajectory-sft-v3"
NUMBER_ABSOLUTE_TOLERANCE = 1e-15
NUMBER_RELATIVE_TOLERANCE = 1e-12

BOOK_MODE_TO_CONTEXT_MODE = {
    "closed_book": "no_context",
    "open_book": "open_book_context",
    "tool_call": "tool_observation",
}

TOPOLOGY_VIEW_TYPES = {
    "monoplex_edge_existence",
    "multiplex_edge_existence",
    "direct_neighbors_by_layer",
    "unique_multiplex_neighbors",
    "gene_layer_membership",
    "nodes_present_by_layer",
    "monoplex_shortest_path",
    "aggregate_multiplex_shortest_path",
    "path_layer_decomposition",
    "monoplex_vs_multiplex_path_comparison",
    "induced_subgraph",
    "connected_components",
    "shared_common_neighbors",
    "degree_hub_bias",
}
ENTITY_SCHEMA_VIEW_TYPES = {
    "entity_id_normalization",
    "gene_alias_disambiguation",
    "graph_schema_provenance",
    "layer_tag_metadata",
    "layer_family_membership",
}
MODULE_SET_VIEW_TYPES = {
    "mentor_ev_module_membership",
    "module_overlap_set_algebra",
    "module_containment_set_algebra",
    "module_source_distinction",
    "module_cohesion_summary",
}
RWR_VIEW_TYPES = {
    "rwr_loe_rank_lookup",
    "rwr_loe_rank_comparison",
    "rwr_loe_topk_membership",
    "rwr_neighborhood_interpretation",
}
STRUCTURED_TOOL_VIEW_TYPES = {
    "tool_call_choice",
    "structured_state_update",
    "provenance_refusal_raw_cli",
}
EXACT_VIEW_TYPES = TOPOLOGY_VIEW_TYPES | ENTITY_SCHEMA_VIEW_TYPES | MODULE_SET_VIEW_TYPES | RWR_VIEW_TYPES | STRUCTURED_TOOL_VIEW_TYPES

UNSUPPORTED_PATTERNS = (
    "definitely causal",
    "confirmed causal",
    "proves caus",
    "causally validated",
    "direct physical interaction",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_canonical_objects(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    objects: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        object_id = row.get("object_id")
        if isinstance(object_id, str):
            objects[object_id] = row
    return objects


def _is_plan_v3_metadata(metadata: dict[str, Any]) -> bool:
    return metadata.get("schema_version") == PLAN_DATASET_SCHEMA_VERSION


def _effective_view_type(
    metadata: dict[str, Any],
    canonical: dict[str, Any] | None = None,
) -> str:
    """Return a stable reporting family without changing legacy scoring rules."""

    for value in (
        metadata.get("view_type"),
        metadata.get("question_family"),
        (canonical or {}).get("object_type"),
    ):
        if isinstance(value, str) and value:
            return value
    return ""


def _effective_context_mode(metadata: dict[str, Any]) -> str | None:
    context_mode = metadata.get("context_mode")
    if isinstance(context_mode, str) and context_mode:
        return context_mode
    book_mode = metadata.get("book_mode")
    if isinstance(book_mode, str):
        return BOOK_MODE_TO_CONTEXT_MODE.get(book_mode)
    return None


def _resolve_canonical_object(
    metadata: dict[str, Any],
    canonical_objects_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve either v3 reference handle and validate the canonical envelope.

    Legacy rows historically used only ``canonical_object_id`` and legacy test
    fixtures often omit the envelope fields.  Envelope validation therefore
    applies only to plan-driven ``pretrajectory-sft-v3`` rows.
    """

    canonical_object_id = metadata.get("canonical_object_id")
    oracle_fact_id = metadata.get("oracle_fact_id")
    canonical_object_id = canonical_object_id if isinstance(canonical_object_id, str) else ""
    oracle_fact_id = oracle_fact_id if isinstance(oracle_fact_id, str) else ""
    reference_id = canonical_object_id or oracle_fact_id
    canonical = canonical_objects_by_id.get(reference_id, {}) if reference_id else {}
    if not _is_plan_v3_metadata(metadata):
        return canonical, []

    issues: list[str] = []
    if not reference_id:
        issues.append("missing_canonical_reference")
    if canonical_object_id and oracle_fact_id and canonical_object_id != oracle_fact_id:
        issues.append("canonical_oracle_reference_mismatch")
    question_family = metadata.get("question_family")
    if not isinstance(question_family, str) or not question_family:
        issues.append("missing_question_family")
    if reference_id and not canonical:
        issues.append(f"missing_canonical_object:{reference_id}")
        return canonical, issues
    if canonical:
        if canonical.get("object_id") != reference_id:
            issues.append("canonical_object_id_mismatch")
        if isinstance(question_family, str) and question_family and canonical.get("object_type") != question_family:
            issues.append("canonical_object_type_mismatch")
        if not isinstance(canonical.get("payload"), dict):
            issues.append("canonical_payload_not_object")
    return canonical, issues


def clean_prediction_text(text: str) -> str:
    cleaned = SPECIAL_TOKEN_RE.sub("", text or "").strip()
    matches = list(FINAL_MARKER_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()
    return cleaned


def ids_from_text(text: str) -> set[str]:
    return set(GENE_RE.findall(text or "")) | set(MODULE_RE.findall(text or ""))


def layers_from_text(text: str) -> set[str]:
    # The token expression is deliberately quote-agnostic: it captures the same
    # arbitrary-depth identifier when it is bare, backticked, or JSON-quoted.
    return set(LAYER_RE.findall(text or ""))


def numbers_from_text(text: str) -> set[str]:
    return set(NUMBER_RE.findall(text or ""))


def _number_value(token: str) -> float | None:
    try:
        value = float(token)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rendered_number_tolerance(token: str, value: float) -> float:
    """Return the rounding interval implied by an answer-rendered number."""

    mantissa, separator, exponent_text = token.lower().partition("e")
    exponent = int(exponent_text) if separator else 0
    unsigned_mantissa = mantissa.lstrip("+-")
    decimals = len(unsigned_mantissa.partition(".")[2]) if "." in unsigned_mantissa else None
    rendered_tolerance = 0.0
    if separator:
        # Scientific notation makes even an integer mantissa's rendered
        # precision explicit: ``2e-7`` denotes one significant digit.
        rendered_tolerance = 0.5 * (10.0 ** (exponent - (decimals or 0)))
    elif decimals is not None:
        rendered_tolerance = 0.5 * (10.0 ** (exponent - decimals))
    return max(
        rendered_tolerance,
        abs(value) * NUMBER_RELATIVE_TOLERANCE,
        NUMBER_ABSOLUTE_TOLERANCE,
    )


def numbers_match(expected_token: str, predicted_token: str) -> bool:
    expected_value = _number_value(expected_token)
    predicted_value = _number_value(predicted_token)
    if expected_value is None or predicted_value is None:
        return False
    return abs(expected_value - predicted_value) <= _rendered_number_tolerance(expected_token, expected_value)


def _unique_numeric_tokens(tokens: set[str]) -> list[str]:
    unique: list[str] = []
    for token in sorted(tokens):
        if not any(numbers_match(existing, token) and numbers_match(token, existing) for existing in unique):
            unique.append(token)
    return unique


def _match_numbers(
    expected_tokens: set[str],
    predicted_tokens: set[str],
) -> tuple[list[str], list[str], int]:
    expected = _unique_numeric_tokens(expected_tokens)
    predicted = _unique_numeric_tokens(predicted_tokens)
    unmatched_prediction_indices = set(range(len(predicted)))
    missing: list[str] = []
    matched = 0
    for expected_token in expected:
        match_index = next(
            (
                index
                for index in unmatched_prediction_indices
                if numbers_match(expected_token, predicted[index])
            ),
            None,
        )
        if match_index is None:
            missing.append(expected_token)
        else:
            unmatched_prediction_indices.remove(match_index)
            matched += 1
    extra = sorted(predicted[index] for index in unmatched_prediction_indices)
    return missing, extra, matched


_NUMBER_BODY = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"


def _first_labeled_number(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match.group("number")
    return None


def _all_labeled_numbers(text: str, pattern: str) -> tuple[str, ...]:
    return tuple(match.group("number") for match in re.finditer(pattern, text, flags=re.IGNORECASE))


def _layer_count_map_from_text(text: str) -> dict[str, str] | None:
    payload = json_object_from_text(text)
    if isinstance(payload, dict):
        result = {
            str(layer): str(count)
            for layer, count in payload.items()
            if isinstance(layer, str)
            and ":" in layer
            and isinstance(count, (int, float))
            and not isinstance(count, bool)
        }
        if result:
            return result

    result: dict[str, str] = {}
    for layer in layers_from_text(text):
        escaped_layer = re.escape(layer)
        patterns = (
            rf"(?P<number>{_NUMBER_BODY})\s+(?:path\s+)?edges?.{{0,48}}?{escaped_layer}",
            rf"{escaped_layer}.{{0,48}}?(?P<number>{_NUMBER_BODY})\s+(?:path\s+)?edges?",
        )
        value = _first_labeled_number(text, patterns)
        if value is not None:
            result[layer] = value
    return result or None


def _neighbor_layer_map_from_text(text: str) -> dict[str, tuple[str, ...]] | None:
    payload = json_object_from_text(text)
    if not isinstance(payload, dict):
        return None
    nested_map = payload.get("neighbor_layer_map")
    if isinstance(nested_map, dict):
        payload = nested_map
    result: dict[str, tuple[str, ...]] = {}
    for gene_id, layer_values in payload.items():
        if not isinstance(gene_id, str) or not GENE_RE.fullmatch(gene_id):
            continue
        if not isinstance(layer_values, list) or not all(isinstance(value, str) for value in layer_values):
            continue
        result[gene_id] = tuple(sorted(str(value) for value in layer_values))
    return result or None


def _rank_order_from_text(text: str) -> tuple[tuple[str, str], tuple[str, str]] | None:
    pattern = re.compile(
        rf"(?P<left>{GENE_RE.pattern}).{{0,180}}?\brank\s*(?:is\s*)?[:=]?\s*"
        rf"(?P<left_rank>{_NUMBER_BODY}).{{0,80}}?(?:versus|than).{{0,40}}?"
        rf"(?P<right>{GENE_RE.pattern}).{{0,80}}?\brank\s*(?:is\s*)?[:=]?\s*"
        rf"(?P<right_rank>{_NUMBER_BODY})",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    return (
        (match.group("left"), match.group("left_rank")),
        (match.group("right"), match.group("right_rank")),
    )


def _seed_id_from_text(text: str) -> str | None:
    match = re.search(rf"\bseed(?:\s+gene)?\s+({GENE_RE.pattern})\b", text, flags=re.IGNORECASE)
    return match.group(1) if match is not None else None


def _typed_fields_from_text(view_type: str, text: str) -> dict[str, Any]:
    """Extract labeled or mapped target fields, preserving their semantic roles."""

    cleaned = clean_prediction_text(text)
    fields: dict[str, Any] = {}

    def add_number(field: str, *patterns: str) -> None:
        value = _first_labeled_number(cleaned, tuple(patterns))
        if value is not None:
            fields[field] = value

    if view_type == "monoplex_edge_existence":
        add_number("weight", rf"\bweight[\"']?\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})")
    elif view_type == "direct_neighbors_by_layer":
        add_number(
            "neighbor_count",
            rf"\bhas\s+(?P<number>{_NUMBER_BODY})\s+(?:direct\s+)?neighbors?",
        )
    elif view_type == "unique_multiplex_neighbors":
        add_number(
            "neighbor_count",
            rf"\bhas\s+(?P<number>{_NUMBER_BODY})\s+unique(?:\s+direct|\s+multiplex)?\s+neighbors?",
        )
        neighbor_map = _neighbor_layer_map_from_text(cleaned)
        if neighbor_map is not None:
            fields["neighbor_layer_map"] = neighbor_map
    elif view_type in {"monoplex_shortest_path", "aggregate_multiplex_shortest_path"}:
        add_number(
            "hop_count",
            rf"\bhop\s+count\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
            rf"\bhops?\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
            rf"(?P<number>{_NUMBER_BODY})\s*[- ]\s*hops?\b",
        )
    elif view_type == "path_layer_decomposition":
        layer_counts = _layer_count_map_from_text(cleaned)
        if layer_counts is not None:
            fields["layer_counts"] = layer_counts
    elif view_type == "induced_subgraph":
        add_number(
            "edge_count",
            rf"\bhas\s+(?P<number>{_NUMBER_BODY})\s+(?:recorded\s+)?edges?",
        )
        weights = _all_labeled_numbers(
            cleaned,
            rf"\bweight[\"']?\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
        )
        if weights:
            fields["edge_weights"] = weights
    elif view_type == "degree_hub_bias":
        add_number(
            "degree",
            rf"\bdegree(?!\s+percentile)\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
        )
        add_number(
            "degree_percentile",
            rf"\bdegree\s+percentile\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
            rf"\bpercentile\s*\(\s*(?P<number>{_NUMBER_BODY})\s*\)",
        )
    elif view_type == "layer_tag_metadata":
        add_number("node_count", rf"(?P<number>{_NUMBER_BODY})\s+(?:recorded\s+)?nodes?")
        add_number("edge_count", rf"(?P<number>{_NUMBER_BODY})\s+(?:recorded\s+)?edges?")
    elif view_type == "module_overlap_set_algebra":
        add_number(
            "intersection_size",
            rf"\bshare\s+(?P<number>{_NUMBER_BODY})\s+genes?",
            rf"\boverlap\s+by\s+(?P<number>{_NUMBER_BODY})\s+genes?",
            rf"\bis\s+(?P<number>{_NUMBER_BODY})\s+genes?",
        )
        add_number(
            "union_size",
            rf"\bunion\s+size\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
        )
        add_number(
            "overlap_jaccard",
            rf"\bjaccard(?:\s+overlap)?\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
        )
    elif view_type == "module_cohesion_summary":
        add_number(
            "node_count",
            rf"(?P<number>{_NUMBER_BODY})\s+(?:sampled\s+)?(?:nodes?|genes?)",
        )
        add_number("edge_count", rf"(?P<number>{_NUMBER_BODY})\s+(?:recorded\s+)?edges?")
        add_number(
            "density",
            rf"\bdensity\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})",
        )
    elif view_type == "rwr_loe_rank_lookup":
        add_number("rank", rf"\brank\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})")
        add_number("score", rf"\bscore\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})")
        seed_id = _seed_id_from_text(cleaned)
        if seed_id is not None:
            fields["seed_gene_id"] = seed_id
    elif view_type == "rwr_loe_rank_comparison":
        rank_order = _rank_order_from_text(cleaned)
        if rank_order is not None:
            fields["rank_order"] = rank_order
        seed_id = _seed_id_from_text(cleaned)
        if seed_id is not None:
            fields["seed_gene_id"] = seed_id
    elif view_type == "rwr_loe_topk_membership":
        add_number("top_k", rf"\btop[-\s]+(?P<number>{_NUMBER_BODY})")
        add_number("rank", rf"\brank\s*(?:is\s*)?[:=]?\s*(?P<number>{_NUMBER_BODY})")
        seed_id = _seed_id_from_text(cleaned)
        if seed_id is not None:
            fields["seed_gene_id"] = seed_id

    return fields


def _typed_values_match(expected: Any, predicted: Any) -> bool:
    if isinstance(expected, str) and _number_value(expected) is not None:
        return isinstance(predicted, str) and numbers_match(expected, predicted)
    if isinstance(expected, tuple):
        if not isinstance(predicted, tuple) or len(expected) != len(predicted):
            return False
        if all(isinstance(value, str) and _number_value(value) is not None for value in expected):
            missing, extra, matched = _match_numbers(set(expected), set(predicted))
            return (
                not missing
                and not extra
                and matched == len(_unique_numeric_tokens(set(expected)))
            )
        return all(_typed_values_match(left, right) for left, right in zip(expected, predicted))
    if isinstance(expected, dict):
        return (
            isinstance(predicted, dict)
            and set(expected) == set(predicted)
            and all(_typed_values_match(value, predicted[key]) for key, value in expected.items())
        )
    return expected == predicted


def _typed_field_failures(view_type: str, answer: str, prediction: str) -> tuple[list[str], dict[str, Any]]:
    expected = _typed_fields_from_text(view_type, answer)
    predicted = _typed_fields_from_text(view_type, prediction)
    failures = [
        field
        for field, expected_value in expected.items()
        if field not in predicted or not _typed_values_match(expected_value, predicted[field])
    ]
    return sorted(failures), expected


def _required_typed_fields(view_type: str, answer: str) -> set[str]:
    lowered = answer.lower()
    required: set[str] = set()
    if view_type == "monoplex_edge_existence" and "weight" in lowered:
        required.add("weight")
    elif view_type == "direct_neighbors_by_layer" and "direct neighbor" in lowered:
        required.add("neighbor_count")
    elif view_type == "unique_multiplex_neighbors":
        if "unique multiplex neighbor" in lowered:
            required.add("neighbor_count")
        if "neighbor-to-layer support" in lowered:
            required.add("neighbor_layer_map")
    elif view_type in {"monoplex_shortest_path", "aggregate_multiplex_shortest_path"}:
        if "hop count" in lowered:
            required.add("hop_count")
    elif view_type == "path_layer_decomposition" and "path layer count" in lowered:
        required.add("layer_counts")
    elif view_type == "induced_subgraph":
        if re.search(r"\bhas\s+[-+\d.]", lowered):
            required.add("edge_count")
        if "weight" in lowered:
            required.add("edge_weights")
    elif view_type == "degree_hub_bias":
        if "degree" in lowered:
            required.add("degree")
        if "degree percentile" in lowered:
            required.add("degree_percentile")
    elif view_type == "layer_tag_metadata":
        if "node" in lowered:
            required.add("node_count")
        if "edge" in lowered:
            required.add("edge_count")
    elif view_type == "module_overlap_set_algebra":
        if "share" in lowered or "overlap" in lowered:
            required.add("intersection_size")
        if "union size" in lowered:
            required.add("union_size")
        if "jaccard" in lowered:
            required.add("overlap_jaccard")
    elif view_type == "module_cohesion_summary":
        if "node" in lowered or "sampled genes" in lowered:
            required.add("node_count")
        if "edge" in lowered:
            required.add("edge_count")
        if "density" in lowered:
            required.add("density")
    elif view_type == "rwr_loe_rank_lookup":
        if "rank" in lowered:
            required.add("rank")
        if "score" in lowered:
            required.add("score")
        if "seed" in lowered:
            required.add("seed_gene_id")
    elif view_type == "rwr_loe_rank_comparison":
        if "rank" in lowered:
            required.add("rank_order")
        if "seed" in lowered:
            required.add("seed_gene_id")
    elif view_type == "rwr_loe_topk_membership":
        if "top" in lowered:
            required.add("top_k")
        if "rank" in lowered:
            required.add("rank")
        if "seed" in lowered:
            required.add("seed_gene_id")
    return required


def _typed_numeric_tokens(value: Any) -> set[str]:
    tokens: set[str] = set()
    if isinstance(value, str) and _number_value(value) is not None:
        tokens.add(value)
    elif isinstance(value, dict):
        for nested_value in value.values():
            tokens.update(_typed_numeric_tokens(nested_value))
    elif isinstance(value, (tuple, list)):
        for nested_value in value:
            tokens.update(_typed_numeric_tokens(nested_value))
    return tokens


def _typed_identifier_tokens(value: Any) -> tuple[set[str], set[str]]:
    identifiers: set[str] = set()
    layers: set[str] = set()
    if isinstance(value, str):
        identifiers.update(ids_from_text(value))
        layers.update(layers_from_text(value))
    elif isinstance(value, dict):
        for key, nested_value in value.items():
            identifiers.update(ids_from_text(str(key)))
            layers.update(layers_from_text(str(key)))
            nested_ids, nested_layers = _typed_identifier_tokens(nested_value)
            identifiers.update(nested_ids)
            layers.update(nested_layers)
    elif isinstance(value, (tuple, list)):
        for nested_value in value:
            nested_ids, nested_layers = _typed_identifier_tokens(nested_value)
            identifiers.update(nested_ids)
            layers.update(nested_layers)
    return identifiers, layers


def _extraction_coverage_issues(view_type: str, answer: str) -> list[str]:
    """Cross-check generic extraction against independent labeled target fields."""

    typed_fields = _typed_fields_from_text(view_type, answer)
    required_fields = _required_typed_fields(view_type, answer)
    issues = [f"missing_typed_target:{field}" for field in sorted(required_fields - set(typed_fields))]
    generic_numbers = numbers_from_text(answer)
    generic_ids = ids_from_text(answer)
    generic_layers = layers_from_text(answer)
    for field, value in typed_fields.items():
        for token in _typed_numeric_tokens(value):
            if not any(numbers_match(token, generic_token) for generic_token in generic_numbers):
                issues.append(f"number_not_extracted:{field}:{token}")
        typed_ids, typed_layers = _typed_identifier_tokens(value)
        for identifier in sorted(typed_ids - generic_ids):
            issues.append(f"id_not_extracted:{field}:{identifier}")
        for layer in sorted(typed_layers - generic_layers):
            issues.append(f"layer_not_extracted:{field}:{layer}")
    return sorted(set(issues))


def evaluate_extraction_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    exact_row_count = 0
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        legacy_view_type = str(metadata.get("view_type", ""))
        plan_v3 = _is_plan_v3_metadata(metadata)
        if legacy_view_type not in EXACT_VIEW_TYPES and not plan_v3:
            continue
        exact_row_count += 1
        if plan_v3 and not legacy_view_type:
            row_issues: list[str] = []
            if metadata.get("answer_format") != "json":
                row_issues.append("answer_format_not_json")
            try:
                answer_payload = json.loads(clean_prediction_text(str(row.get("answer", ""))))
            except json.JSONDecodeError:
                answer_payload = None
            if not isinstance(answer_payload, dict):
                row_issues.append("invalid_exact_json_target")
        else:
            row_issues = _extraction_coverage_issues(
                legacy_view_type,
                str(row.get("answer", "")),
            )
        if row_issues:
            issues.append(
                {
                    "idx": row.get("idx"),
                    "record_id": metadata.get("record_id"),
                    "view_type": _effective_view_type(metadata),
                    "issues": row_issues,
                }
            )
    passed = None if exact_row_count == 0 else not issues
    return {
        "status": "not_applicable" if passed is None else ("passed" if passed else "failed"),
        "passed": passed,
        "exact_row_count": exact_row_count,
        "failure_count": len(issues),
        "failures": issues,
    }


def _id_is_explicitly_absent(text: str, identifier: str) -> bool:
    escaped = re.escape(identifier)
    patterns = (
        rf"{escaped}\s+(?:(?:is|was|are)\s+)?(?:not\s+present|absent|not\s+recorded|not\s+in\s+(?:the\s+)?layer)\b",
        rf"\b(?:not\s+present|absent|not\s+recorded)\s*[:,-]?\s*{escaped}\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _context_allowances(
    row: dict[str, Any],
    view_type: str,
    payload: dict[str, Any],
    prediction: str,
) -> tuple[set[str], set[str]]:
    question = str(row.get("question", ""))
    system = str(row.get("system", ""))
    context_text = f"{system}\n{question}"
    context_ids = ids_from_text(context_text)
    context_layers = layers_from_text(context_text)
    allowed_ids: set[str] = set()
    allowed_layers: set[str] = set()

    if view_type == "connected_components":
        allowed_ids.update(context_ids)
        allowed_layers.update(context_layers)
    elif view_type == "path_layer_decomposition":
        allowed_ids.update(context_ids)
    elif view_type == "nodes_present_by_layer":
        allowed_ids.update(
            identifier
            for identifier in context_ids
            if _id_is_explicitly_absent(prediction, identifier)
        )
    elif view_type == "rwr_neighborhood_interpretation":
        seed_gene_id = payload.get("seed_gene_id")
        if isinstance(seed_gene_id, str) and seed_gene_id in context_ids:
            allowed_ids.add(seed_gene_id)
        else:
            seed_gene_id = _seed_id_from_text(question)
            if seed_gene_id is not None:
                allowed_ids.add(seed_gene_id)
    return allowed_ids, allowed_layers


def prediction_contains_term(prediction: str, term: str) -> bool:
    if not term:
        return True
    return term in prediction


def json_object_from_text(text: str) -> dict[str, Any] | None:
    cleaned = clean_prediction_text(text)
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        for end in range(len(cleaned), start, -1):
            snippet = cleaned[start:end]
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            return payload if isinstance(payload, dict) else None
    return None


def whole_json_object_from_text(text: str) -> dict[str, Any] | None:
    """Parse a contract only when the entire cleaned answer is a JSON object."""

    try:
        payload = json.loads(clean_prediction_text(text))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _json_contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _json_contains_subset(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _json_contains_subset(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(
                float(actual),
                float(expected),
                rel_tol=NUMBER_RELATIVE_TOLERANCE,
                abs_tol=NUMBER_ABSOLUTE_TOLERANCE,
            )
        )
    return actual == expected


def _json_exact_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_json_exact_match(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _json_exact_match(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(
                float(actual),
                float(expected),
                rel_tol=NUMBER_RELATIVE_TOLERANCE,
                abs_tol=NUMBER_ABSOLUTE_TOLERANCE,
            )
        )
    return actual == expected


def yes_no_label(text: str) -> str | None:
    stripped = clean_prediction_text(text).strip().lower()
    if re.match(r"^yes\b", stripped):
        return "yes"
    if re.match(r"^no\b", stripped):
        return "no"
    first_clause = stripped[:240]
    if any(phrase in first_clause for phrase in ("not directly connected", "no edge", "no recorded edge", "no path")):
        return "no"
    if "directly connected" in first_clause or "has an edge" in first_clause:
        return "yes"
    return None


def _unsupported_match_is_negated(text: str, match_start: int) -> bool:
    prefix = text[:match_start]
    # Punctuation and explicit contrast markers end the scope of an earlier
    # negation (for example, "not merely associated; definitely causal").
    clause = re.split(r"[.;!?\n]|\b(?:but|however|instead|rather)\b", prefix)[-1]
    words = re.findall(r"[a-z]+(?:['’][a-z]+)?", clause.lower())[-24:]
    if not words:
        return False
    joined = " ".join(words)
    return bool(
        re.search(
            r"\b(?:not|never|without|cannot|can't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b",
            joined,
        )
        or re.search(r"\bno (?:evidence|support|basis|reason|claim|proof)\b", joined)
    )


def has_unsupported_language(text: str) -> bool:
    lowered = clean_prediction_text(text).lower()
    for pattern in (*UNSUPPORTED_PATTERNS, "no biological relationship"):
        for match in re.finditer(re.escape(pattern), lowered):
            if not _unsupported_match_is_negated(lowered, match.start()):
                return True
    return False


def _rendered_answer_facts(payload: dict[str, Any], answer: str) -> dict[str, Any]:
    """Extract the contract from the rendered target, not hidden payload facts.

    Canonical payloads often contain query IDs, full module members, rank-cache
    context, or higher-precision floats that the target intentionally does not
    render.  Those fields cannot be requirements for reproducing that target.
    The payload is used only to identify rendered categorical terms that the
    generic ID/layer/number extractors cannot classify.
    """

    expected_terms: set[str] = set()
    categorical_keys = {
        "alias",
        "allowed_interface",
        "directionality",
        "forbidden_interface",
        "layer_family",
        "layer_namespace",
        "preferred_answer",
        "reason",
        "relationship_status",
        "size_bin",
        "source",
        "source_name",
        "task_id",
        "task_type",
        "tool_name",
    }

    def visit(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                visit(nested_value, str(nested_key))
        elif isinstance(value, list):
            for nested_value in value:
                visit(nested_value, key)
        elif key in categorical_keys and isinstance(value, str) and len(value) >= 1:
            if value in answer and not ids_from_text(value) and not layers_from_text(value):
                expected_terms.add(value)

    visit(payload)
    return {
        "expected_ids": sorted(ids_from_text(answer)),
        "allowed_extra_ids": [],
        "expected_layers": sorted(layers_from_text(answer)),
        "expected_numbers": sorted(numbers_from_text(answer)),
        "expected_terms": sorted(expected_terms),
        "expected_label": yes_no_label(answer),
    }


def _rate(numerator: int, denominator: int | None) -> float | None:
    if denominator is None:
        return None
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def evaluate_row(row: dict[str, Any], canonical_objects_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    legacy_view_type = str(metadata.get("view_type", ""))
    plan_v3 = _is_plan_v3_metadata(metadata)
    bucket = str(metadata.get("mixture_bucket", ""))
    answer = str(row.get("answer", ""))
    prediction_raw = str(row.get("prediction", ""))
    prediction = clean_prediction_text(prediction_raw)
    canonical, canonical_reference_issues = _resolve_canonical_object(
        metadata,
        canonical_objects_by_id,
    )
    dataset_contract_issues = list(canonical_reference_issues)
    view_type = _effective_view_type(metadata, canonical)
    payload = canonical.get("payload") if isinstance(canonical.get("payload"), dict) else {}
    facts = _rendered_answer_facts(payload, answer)

    expected_ids = set(facts["expected_ids"])
    expected_layers = set(facts["expected_layers"])
    expected_numbers = set(facts["expected_numbers"])
    expected_terms = set(facts["expected_terms"])
    allowed_extra_ids, allowed_extra_layers = _context_allowances(
        row,
        legacy_view_type,
        payload,
        prediction,
    )
    allowed_extra_ids.update(facts["allowed_extra_ids"])
    # These are allowances only for context facts absent from the rendered
    # target.  Removing expected facts keeps precision accounting and report
    # diagnostics semantically explicit.
    allowed_extra_ids.difference_update(expected_ids)
    allowed_extra_layers.difference_update(expected_layers)

    pred_ids = ids_from_text(prediction)
    pred_layers = layers_from_text(prediction)
    pred_numbers = numbers_from_text(prediction)
    expected_label = facts["expected_label"]
    pred_label = yes_no_label(prediction)

    missing_ids = sorted(expected_ids - pred_ids)
    extra_ids = sorted(pred_ids - expected_ids - allowed_extra_ids)
    missing_layers = sorted(expected_layers - pred_layers)
    extra_layers = sorted(pred_layers - expected_layers - allowed_extra_layers)
    missing_numbers, extra_numbers, matched_number_count = _match_numbers(
        expected_numbers,
        pred_numbers,
    )
    missing_terms = sorted(term for term in expected_terms if not prediction_contains_term(prediction, term))
    typed_field_failures, expected_typed_fields = _typed_field_failures(
        legacy_view_type,
        answer,
        prediction,
    )
    extraction_coverage_issues = (
        []
        if plan_v3 and not legacy_view_type
        else _extraction_coverage_issues(legacy_view_type, answer)
    )

    id_recall = _rate(len(expected_ids & pred_ids), len(expected_ids)) if expected_ids else None
    id_precision = _rate(len((pred_ids - allowed_extra_ids) & expected_ids), len(pred_ids - allowed_extra_ids)) if expected_ids or pred_ids else None
    layer_recall = _rate(len(expected_layers & pred_layers), len(expected_layers)) if expected_layers else None
    scored_pred_layers = pred_layers - allowed_extra_layers
    layer_precision = _rate(
        len(scored_pred_layers & expected_layers),
        len(scored_pred_layers),
    ) if expected_layers or scored_pred_layers else None
    unique_expected_numbers = _unique_numeric_tokens(expected_numbers)
    unique_predicted_numbers = _unique_numeric_tokens(pred_numbers)
    number_recall = _rate(matched_number_count, len(unique_expected_numbers)) if unique_expected_numbers else None
    number_precision = (
        _rate(matched_number_count, len(unique_predicted_numbers))
        if unique_expected_numbers or unique_predicted_numbers
        else None
    )
    label_correct = expected_label is None or pred_label == expected_label
    unsupported = has_unsupported_language(prediction)
    json_subset_match: bool | None = None
    json_exact_match: bool | None = None
    expected_payload = whole_json_object_from_text(answer)
    prediction_payload: dict[str, Any] | None = None
    if expected_payload is not None:
        prediction_payload = whole_json_object_from_text(prediction)
    elif legacy_view_type in {"tool_call_choice", "structured_state_update"}:
        # Structured views remain JSON contracts even if a future renderer
        # wraps their object in a short label.  Embedded JSON in ordinary prose
        # is graph evidence, not a formatting requirement.
        expected_payload = json_object_from_text(answer)
        prediction_payload = json_object_from_text(prediction)
    if expected_payload is not None:
        json_subset_match = (
            prediction_payload is not None
            and _json_contains_subset(prediction_payload, expected_payload)
        )
        if plan_v3:
            json_exact_match = (
                prediction_payload is not None
                and _json_exact_match(prediction_payload, expected_payload)
            )
    elif plan_v3:
        dataset_contract_issues.append("invalid_exact_json_target")
    if plan_v3 and metadata.get("answer_format") != "json":
        dataset_contract_issues.append("answer_format_not_json")
    if plan_v3:
        validator = metadata.get("validator")
        if (
            not isinstance(validator, dict)
            or not isinstance(validator.get("type"), str)
            or not validator["type"]
        ):
            dataset_contract_issues.append("missing_validator_contract")
    dataset_contract_issues = sorted(set(dataset_contract_issues))

    exact_graph_fact_pass = None
    if legacy_view_type in EXACT_VIEW_TYPES or plan_v3:
        exact_graph_fact_pass = (
            label_correct
            and not missing_ids
            and not extra_ids
            and not missing_layers
            and not extra_layers
            and not missing_numbers
            and not extra_numbers
            and not missing_terms
            and not typed_field_failures
            and not extraction_coverage_issues
            and (json_subset_match is not False)
            and (not plan_v3 or json_exact_match is True)
            and not unsupported
            and not dataset_contract_issues
        )

    return {
        "idx": row.get("idx"),
        "record_id": metadata.get("record_id"),
        "view_type": view_type,
        "legacy_view_type": legacy_view_type or None,
        "question_family": metadata.get("question_family"),
        "dataset_schema_version": metadata.get("schema_version"),
        "canonical_object_id": metadata.get("canonical_object_id"),
        "oracle_fact_id": metadata.get("oracle_fact_id"),
        "canonical_object_type": canonical.get("object_type"),
        "canonical_reference_valid": not canonical_reference_issues if plan_v3 else None,
        "canonical_reference_issues": canonical_reference_issues,
        "dataset_contract_valid": not dataset_contract_issues if plan_v3 else None,
        "dataset_contract_issues": dataset_contract_issues,
        "mixture_bucket": bucket,
        "context_mode": _effective_context_mode(metadata),
        "overlap_score": row.get("overlap_score"),
        "prediction_clean": prediction,
        "expected_label": expected_label,
        "predicted_label": pred_label,
        "label_correct": label_correct,
        "id_recall": id_recall,
        "id_precision": id_precision,
        "layer_recall": layer_recall,
        "layer_precision": layer_precision,
        "number_recall": number_recall,
        "number_precision": number_precision,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "allowed_context_ids": sorted(allowed_extra_ids),
        "missing_layers": missing_layers,
        "extra_layers": extra_layers,
        "allowed_context_layers": sorted(allowed_extra_layers),
        "missing_numbers": missing_numbers,
        "extra_numbers": extra_numbers,
        "missing_terms": missing_terms,
        "expected_typed_fields": expected_typed_fields,
        "typed_field_failures": typed_field_failures,
        "extraction_coverage_issues": extraction_coverage_issues,
        "json_subset_match": json_subset_match,
        "json_exact_match": json_exact_match,
        "unsupported_language": unsupported,
        "exact_graph_fact_pass": exact_graph_fact_pass,
    }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(mean(present)) if present else None


def _summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    exact_items = [item for item in items if item["exact_graph_fact_pass"] is not None]
    id_items = [item for item in items if item["id_recall"] is not None]
    layer_items = [item for item in items if item["layer_recall"] is not None]
    number_items = [item for item in items if item["number_recall"] is not None]
    number_precision_items = [item for item in items if item["number_precision"] is not None]
    yes_no_items = [item for item in items if item["expected_label"] is not None]
    return {
        "count": len(items),
        "exact_applicable_count": len(exact_items),
        "exact_pass_count": sum(item["exact_graph_fact_pass"] is True for item in exact_items),
        "id_applicable_count": len(id_items),
        "layer_applicable_count": len(layer_items),
        "number_applicable_count": len(number_items),
        "number_precision_applicable_count": len(number_precision_items),
        "yes_no_applicable_count": len(yes_no_items),
        "mean_overlap_score": _mean([item.get("overlap_score") for item in items]),
        "mean_id_recall": _mean([item["id_recall"] for item in items]),
        "mean_id_precision": _mean([item["id_precision"] for item in items]),
        "mean_layer_recall": _mean([item["layer_recall"] for item in items]),
        "mean_layer_precision": _mean([item["layer_precision"] for item in items]),
        "mean_number_recall": _mean([item["number_recall"] for item in items]),
        "mean_number_precision": _mean(
            [item["number_precision"] for item in number_precision_items]
        ),
        "yes_no_accuracy": _mean([1.0 if item["label_correct"] else 0.0 for item in yes_no_items]),
        "exact_graph_fact_pass_rate": _mean([1.0 if item["exact_graph_fact_pass"] else 0.0 for item in exact_items]),
        "unsupported_language_rate": _mean([1.0 if item["unsupported_language"] else 0.0 for item in items]),
    }


def evaluate_gold_self_contract(
    rows: list[dict[str, Any]],
    *,
    canonical_objects_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify that rendered gold answers satisfy the evaluator contract."""

    canonical_objects_by_id = canonical_objects_by_id or {}
    gold_metrics = [
        evaluate_row({**row, "prediction": row.get("answer", "")}, canonical_objects_by_id)
        for row in rows
    ]
    exact_items = [item for item in gold_metrics if item["exact_graph_fact_pass"] is not None]
    extraction_coverage = evaluate_extraction_coverage(rows)

    def all_applicable(metric: str, expected: float) -> bool | None:
        values = [item[metric] for item in exact_items if item[metric] is not None]
        return all(math.isclose(float(value), expected) for value in values) if values else None

    checks: dict[str, bool | None] = {
        "exact_graph_fact_pass_rate_is_one": (
            all(item["exact_graph_fact_pass"] is True for item in exact_items) if exact_items else None
        ),
        "id_recall_is_one": all_applicable("id_recall", 1.0),
        "id_precision_is_one": all_applicable("id_precision", 1.0),
        "layer_recall_is_one": all_applicable("layer_recall", 1.0),
        "layer_precision_is_one": all_applicable("layer_precision", 1.0),
        "number_recall_is_one": all_applicable("number_recall", 1.0),
        "number_precision_is_one": all_applicable("number_precision", 1.0),
        "yes_no_accuracy_is_one": (
            all(item["label_correct"] for item in exact_items if item["expected_label"] is not None)
            if any(item["expected_label"] is not None for item in exact_items)
            else None
        ),
        "unsupported_language_rate_is_zero": (
            all(not item["unsupported_language"] for item in exact_items) if exact_items else None
        ),
        "extraction_coverage_passed": extraction_coverage["passed"],
        "dataset_contract_passed": (
            all(not item["dataset_contract_issues"] for item in exact_items)
            if any(item["dataset_schema_version"] == PLAN_DATASET_SCHEMA_VERSION for item in exact_items)
            else None
        ),
    }
    failing_items = [
        item
        for item in exact_items
        if item["exact_graph_fact_pass"] is not True
        or item["unsupported_language"]
        or any(
            item[metric] is not None and not math.isclose(float(item[metric]), 1.0)
            for metric in (
                "id_recall",
                "id_precision",
                "layer_recall",
                "layer_precision",
                "number_recall",
                "number_precision",
            )
        )
        or (item["expected_label"] is not None and not item["label_correct"])
    ]
    passed = None if not exact_items else all(value is not False for value in checks.values())
    return {
        "contract_version": EVALUATOR_CONTRACT_VERSION,
        "status": "not_applicable" if passed is None else ("passed" if passed else "failed"),
        "passed": passed,
        "exact_row_count": len(exact_items),
        "checks": checks,
        "extraction_coverage": extraction_coverage,
        "summary": _summarize_group(exact_items),
        "failure_count": len(failing_items),
        "failing_indices": [item.get("idx") for item in failing_items],
        "failing_record_ids": [item.get("record_id") for item in failing_items],
    }


def evaluate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    canonical_objects_by_id: dict[str, dict[str, Any]] | None = None,
    max_examples: int = 80,
) -> dict[str, Any]:
    canonical_objects_by_id = canonical_objects_by_id or {}
    row_metrics = [evaluate_row(row, canonical_objects_by_id) for row in rows]
    by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_context_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[str] = Counter()
    exact_failure_counts: Counter[str] = Counter()

    def collect_failures(counter: Counter[str], item: dict[str, Any]) -> None:
        if item["missing_ids"]:
            counter["missing_ids"] += 1
        if item["extra_ids"]:
            counter["extra_ids"] += 1
        if item["missing_layers"]:
            counter["missing_layers"] += 1
        if item["extra_layers"]:
            counter["extra_layers"] += 1
        if item["missing_numbers"]:
            counter["missing_numbers"] += 1
        if item["extra_numbers"]:
            counter["extra_numbers"] += 1
        if item["missing_terms"]:
            counter["missing_terms"] += 1
        if item["typed_field_failures"]:
            counter["typed_field_mismatch"] += 1
        if item["extraction_coverage_issues"]:
            counter["extraction_coverage_failure"] += 1
        if item["canonical_reference_issues"]:
            counter["canonical_reference_failure"] += 1
        if item["dataset_contract_issues"]:
            counter["dataset_contract_failure"] += 1
        if item["json_subset_match"] is False:
            counter["json_subset_mismatch"] += 1
        if item["json_exact_match"] is False:
            counter["json_exact_mismatch"] += 1
        if item["expected_label"] is not None and not item["label_correct"]:
            counter["wrong_yes_no"] += 1
        if item["unsupported_language"]:
            counter["unsupported_language"] += 1
        if item["exact_graph_fact_pass"] is False:
            counter["exact_graph_fact_fail"] += 1

    for item in row_metrics:
        by_view[item["view_type"]].append(item)
        by_bucket[item["mixture_bucket"]].append(item)
        by_context_mode[str(item.get("context_mode") or "unspecified")].append(item)
        collect_failures(failure_counts, item)
        if item["exact_graph_fact_pass"] is not None:
            collect_failures(exact_failure_counts, item)

    examples = sorted(
        row_metrics,
        key=lambda item: (
            item["exact_graph_fact_pass"] is not False,
            item.get("overlap_score") if isinstance(item.get("overlap_score"), (int, float)) else 1.0,
        ),
    )[:max_examples]

    exact_row_metrics = [item for item in row_metrics if item["exact_graph_fact_pass"] is not None]
    summary = _summarize_group(row_metrics)
    summary["exact_only"] = _summarize_group(exact_row_metrics)
    gold_self_evaluation = evaluate_gold_self_contract(
        rows,
        canonical_objects_by_id=canonical_objects_by_id,
    )
    return {
        "evaluator_contract": {
            "version": EVALUATOR_CONTRACT_VERSION,
            "expected_fact_source": "rendered_answer",
            "layer_identifiers": "bare_backticked_or_json_quoted_arbitrary_depth",
            "numeric_comparison": "answer_rendering_rounding_interval",
            "numeric_requirement": "all_exact_view_rendered_numbers",
            "typed_field_comparison": "view_specific_labeled_and_mapped_fields",
            "context_policy": "view_specific_query_restatement_allowances",
            "structured_json_comparison": "whole_answer_or_structured_view_recursive_subset",
            "plan_structured_json_comparison": "whole_answer_recursive_exact",
            "plan_dataset_schema": PLAN_DATASET_SCHEMA_VERSION,
            "plan_family_source": "metadata.question_family_then_canonical_object.object_type",
            "plan_canonical_reference": "metadata.canonical_object_id_or_oracle_fact_id",
            "plan_context_mapping": BOOK_MODE_TO_CONTEXT_MODE,
        },
        "gold_self_evaluation": gold_self_evaluation,
        "extraction_coverage": gold_self_evaluation["extraction_coverage"],
        "sample_count": len(rows),
        "summary": summary,
        "by_view_type": {key: _summarize_group(value) for key, value in sorted(by_view.items())},
        "by_mixture_bucket": {key: _summarize_group(value) for key, value in sorted(by_bucket.items())},
        "by_context_mode": {key: _summarize_group(value) for key, value in sorted(by_context_mode.items())},
        "failure_counts": dict(sorted(failure_counts.items())),
        "exact_failure_counts": dict(sorted(exact_failure_counts.items())),
        "examples": examples,
    }


def render_html_report(report: dict[str, Any], rows_by_idx: dict[Any, dict[str, Any]], *, title: str = "Pre-Trajectory SFT Exact Evaluation") -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return html.escape(str(value))

    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;line-height:1.35}"
        "table{border-collapse:collapse;width:100%;margin:14px 0}td,th{border:1px solid #ddd;padding:6px;vertical-align:top}"
        "th{background:#f5f5f5}.fail{background:#fff0f0}.pass{background:#f0fff4}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:8px;border-radius:4px}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<h2>Summary</h2>",
        "<pre>" + html.escape(json.dumps(report.get("summary", {}), indent=2, sort_keys=True)) + "</pre>",
        "<h2>By View Type</h2><table><tr><th>View</th><th>n</th><th>Exact</th><th>ID R</th><th>ID P</th><th>Layer R</th><th>Layer P</th><th>Yes/No</th></tr>",
    ]
    for view, metrics in report.get("by_view_type", {}).items():
        sections.append(
            "<tr>"
            f"<td>{html.escape(view)}</td><td>{fmt(metrics.get('count'))}</td>"
            f"<td>{fmt(metrics.get('exact_graph_fact_pass_rate'))}</td>"
            f"<td>{fmt(metrics.get('mean_id_recall'))}</td>"
            f"<td>{fmt(metrics.get('mean_id_precision'))}</td>"
            f"<td>{fmt(metrics.get('mean_layer_recall'))}</td>"
            f"<td>{fmt(metrics.get('mean_layer_precision'))}</td>"
            f"<td>{fmt(metrics.get('yes_no_accuracy'))}</td>"
            "</tr>"
        )
    sections.append("</table><h2>Failure Examples</h2>")
    for item in report.get("examples", []):
        row = rows_by_idx.get(item.get("idx"), {})
        css = "pass" if item.get("exact_graph_fact_pass") else "fail"
        sections.append(
            f"<section class='{css}'><h3>idx {fmt(item.get('idx'))} | {fmt(item.get('view_type'))}</h3>"
            "<pre><b>Question</b>\n" + html.escape(str(row.get("question", ""))) + "</pre>"
            "<pre><b>Expected</b>\n" + html.escape(str(row.get("answer", ""))) + "</pre>"
            "<pre><b>Prediction</b>\n" + html.escape(str(row.get("prediction", ""))) + "</pre>"
            "<pre><b>Metrics</b>\n" + html.escape(json.dumps(item, indent=2, sort_keys=True)) + "</pre>"
            "</section>"
        )
    sections.append("</body></html>")
    return "\n".join(sections)


def evaluate_predictions_file(
    predictions_path: Path,
    *,
    canonical_objects_path: Path | None = None,
    json_out: Path | None = None,
    html_out: Path | None = None,
    max_examples: int = 80,
    gold_self_only: bool = False,
) -> dict[str, Any]:
    rows = read_jsonl(predictions_path)
    if gold_self_only:
        rows = [{**row, "prediction": row.get("answer", "")} for row in rows]
    canonical = load_canonical_objects(canonical_objects_path)
    report = evaluate_prediction_rows(rows, canonical_objects_by_id=canonical, max_examples=max_examples)
    if json_out is not None:
        write_json(json_out, report)
    if html_out is not None:
        rows_by_idx = {row.get("idx"): row for row in rows}
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(render_html_report(report, rows_by_idx), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre-trajectory SFT holdout predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Dataset directory containing canonical_objects.jsonl.")
    parser.add_argument("--canonical-objects", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--html-out", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=80)
    parser.add_argument(
        "--gold-self-only",
        action="store_true",
        help="Copy each rendered answer into prediction before scoring the file.",
    )
    parser.add_argument(
        "--allow-invalid-gold-contract",
        action="store_true",
        help="Write the report but do not exit nonzero when gold self-evaluation is invalid.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_path = args.canonical_objects
    if canonical_path is None and args.dataset_dir is not None:
        canonical_path = args.dataset_dir / "canonical_objects.jsonl"
    report = evaluate_predictions_file(
        args.predictions,
        canonical_objects_path=canonical_path,
        json_out=args.json_out,
        html_out=args.html_out,
        max_examples=args.max_examples,
        gold_self_only=args.gold_self_only,
    )
    print(
        json.dumps(
            {
                "sample_count": report["sample_count"],
                "evaluator_contract": report["evaluator_contract"],
                "gold_self_evaluation": report["gold_self_evaluation"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["gold_self_evaluation"]["passed"] is not True and not args.allow_invalid_gold_contract:
        print(
            "Gold self-evaluation contract is not valid; refusing to accept the exact-evaluation report.",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
