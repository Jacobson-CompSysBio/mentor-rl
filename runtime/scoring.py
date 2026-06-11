"""Simple deterministic scoring for MENTOR-RL runtime branches.

This module scores one candidate branch against the hidden CORUM target for the
current task. The goal is to keep the logic easy to read:

- schema score asks whether the branch is structurally valid
- complex delta asks whether the predicted gene group moved toward the target
- mechanism delta asks whether predicted labels became more accurate
- efficiency penalty discourages long or invalid tool use

The scorer only uses hidden CORUM targets at score time. They never enter the
visible runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable

from .schemas import (
    CandidateBranch,
    ContinuationState,
    LabelSource,
    LocalScoreBreakdown,
    MechanisticLabel,
    RelationshipStatus,
    SchemaValidationError,
    StructuredState,
    TaskType,
    ToolAction,
    ToolObservationStatus,
)
from .validators import (
    is_duplicate_tool_action,
    validate_candidate_branch,
    validate_tool_action_semantics,
)


@dataclass(frozen=True)
class ComplexMetricWeights:
    """Weights for Jaccard, precision, and recall in complex scoring."""

    jaccard: float
    precision: float
    recall: float

    def __post_init__(self) -> None:
        if self.jaccard < 0 or self.precision < 0 or self.recall < 0:
            raise ValueError("Complex metric weights must be non-negative.")
        if (self.jaccard + self.precision + self.recall) <= 0:
            raise ValueError("At least one complex metric weight must be positive.")


@dataclass(frozen=True)
class LocalScoringConfig:
    """Small set of tunable scoring constants used by the local scorer."""

    schema_weight: float = 1.0
    complex_delta_weight: float = 1.0
    mechanism_delta_weight: float = 1.0
    efficiency_weight: float = 1.0
    step_penalty_lambda: float = 0.05
    invalid_call_penalty_lambda: float = 0.95
    none_relationship_weight: float = 0.6
    none_abstention_weight: float = 0.4
    recovery_complex_weights: ComplexMetricWeights = field(
        default_factory=lambda: ComplexMetricWeights(jaccard=0.2, precision=0.2, recall=0.6)
    )
    refinement_complex_weights: ComplexMetricWeights = field(
        default_factory=lambda: ComplexMetricWeights(jaccard=0.2, precision=0.6, recall=0.2)
    )
    explanation_complex_weights: ComplexMetricWeights = field(
        default_factory=lambda: ComplexMetricWeights(jaccard=1.0 / 3.0, precision=1.0 / 3.0, recall=1.0 / 3.0)
    )

    def __post_init__(self) -> None:
        numeric_fields = (
            "schema_weight",
            "complex_delta_weight",
            "mechanism_delta_weight",
            "efficiency_weight",
            "step_penalty_lambda",
            "invalid_call_penalty_lambda",
            "none_relationship_weight",
            "none_abstention_weight",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if (self.none_relationship_weight + self.none_abstention_weight) <= 0:
            raise ValueError("At least one none-task abstention weight must be positive.")

    def complex_weights_for_task(self, task_type: TaskType) -> ComplexMetricWeights:
        """Return the J/P/R weights for one task type."""

        if task_type == TaskType.RECOVERY:
            return self.recovery_complex_weights
        if task_type == TaskType.REFINEMENT:
            return self.refinement_complex_weights
        if task_type == TaskType.EXPLANATION:
            return self.explanation_complex_weights
        raise ValueError(f"Complex metric weights are not defined for task type: {task_type}.")


DEFAULT_LOCAL_SCORING_CONFIG = LocalScoringConfig()


@dataclass(frozen=True)
class TerminalScoringConfig:
    """Weights for scoring a completed trajectory against hidden supervision."""

    schema_weight: float = 1.0
    absolute_complex_weight: float = 1.0
    complex_delta_weight: float = 1.0
    absolute_mechanism_weight: float = 1.0
    mechanism_delta_weight: float = 1.0
    efficiency_weight: float = 1.0
    local_config: LocalScoringConfig = field(default_factory=LocalScoringConfig)

    def __post_init__(self) -> None:
        numeric_fields = (
            "schema_weight",
            "absolute_complex_weight",
            "complex_delta_weight",
            "absolute_mechanism_weight",
            "mechanism_delta_weight",
            "efficiency_weight",
        )
        for field_name in numeric_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")

    def complex_weights_for_task(self, task_type: TaskType) -> ComplexMetricWeights:
        return self.local_config.complex_weights_for_task(task_type)


DEFAULT_TERMINAL_SCORING_CONFIG = TerminalScoringConfig()

TASK_SUCCESS_THRESHOLDS = {
    "explanation": {
        "positive_recall": 0.85,
        "positive_jaccard": 0.75,
        "partial_recall": 0.60,
        "partial_jaccard": 0.50,
    },
    "recovery": {
        "positive_recall": 0.80,
        "positive_jaccard": 0.80,
        "partial_recall": 0.60,
        "partial_jaccard": 0.60,
    },
    "refinement": {
        "positive_recall": 0.80,
        "positive_jaccard": 0.80,
        "positive_precision": 0.80,
        "partial_recall": 0.60,
        "partial_jaccard": 0.60,
        "partial_precision": 0.60,
    },
    "none": {
        "positive_relationship": RelationshipStatus.INSUFFICIENT_SUPPORT.value,
        "positive_predicted_gene_count": 0,
    },
}


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be a dict, got {type(value).__name__}.")
    return value


def _require_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        _fail(f"{name} must be an int, got {type(value).__name__}.")
    if value <= 0:
        _fail(f"{name} must be positive.")
    return value


def _normalize_label_name(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _flatten_predicted_gene_ids(state: StructuredState) -> list[str]:
    gene_ids: list[str] = []
    seen: set[str] = set()
    for group in state.predicted_groups:
        for gene_id in group.gene_ids:
            if gene_id not in seen:
                seen.add(gene_id)
                gene_ids.append(gene_id)
    return gene_ids


def _set_metrics(predicted_gene_ids: list[str], target_gene_ids: list[str]) -> dict[str, float]:
    predicted = set(predicted_gene_ids)
    target = set(target_gene_ids)
    overlap = predicted & target
    union = predicted | target
    return {
        "jaccard": 1.0 if not union else len(overlap) / len(union),
        "precision": _safe_divide(len(overlap), len(predicted)),
        "recall": _safe_divide(len(overlap), len(target)),
    }


def _weighted_complex_score(
    metrics: dict[str, float],
    weights: ComplexMetricWeights,
) -> float:
    return (
        weights.jaccard * metrics["jaccard"]
        + weights.precision * metrics["precision"]
        + weights.recall * metrics["recall"]
    )


def _best_group_match(
    state: StructuredState,
    target_gene_ids: list[str],
    weights: ComplexMetricWeights,
) -> dict[str, Any]:
    if not state.predicted_groups:
        empty_metrics = _set_metrics([], target_gene_ids)
        return {
            "group_id": None,
            "gene_ids": [],
            "metrics": empty_metrics,
            "score": _weighted_complex_score(empty_metrics, weights),
        }

    best_score = None
    best_payload: dict[str, Any] | None = None
    target = set(target_gene_ids)
    for group in state.predicted_groups:
        metrics = _set_metrics(group.gene_ids, target_gene_ids)
        score = _weighted_complex_score(metrics, weights)
        false_positives = len(set(group.gene_ids) - target)
        candidate_payload = {
            "group_id": group.group_id,
            "gene_ids": list(group.gene_ids),
            "metrics": metrics,
            "score": score,
            "false_positives": false_positives,
        }
        if best_payload is None:
            best_payload = candidate_payload
            best_score = score
            continue
        assert best_score is not None  # pragma: no cover - typing guard
        if score > best_score:
            best_payload = candidate_payload
            best_score = score
            continue
        if score == best_score and false_positives < best_payload["false_positives"]:
            best_payload = candidate_payload
            best_score = score
            continue
        if (
            score == best_score
            and false_positives == best_payload["false_positives"]
            and group.group_id < str(best_payload["group_id"])
        ):
            best_payload = candidate_payload
            best_score = score

    assert best_payload is not None  # pragma: no cover - impossible after non-empty loop
    best_payload.pop("false_positives", None)
    return best_payload


def _visible_seed_retention(state: StructuredState, task_row: dict[str, Any]) -> dict[str, Any]:
    visible_inputs = task_row.get("visible_inputs")
    seed_gene_ids = []
    if isinstance(visible_inputs, dict):
        seed_gene_ids = [
            str(gene_id)
            for gene_id in visible_inputs.get("seed_gene_ids", [])
            if isinstance(gene_id, str) and gene_id
        ]
    predicted_gene_ids = set(_flatten_predicted_gene_ids(state))
    retained = [gene_id for gene_id in seed_gene_ids if gene_id in predicted_gene_ids]
    return {
        "seed_gene_count": len(seed_gene_ids),
        "retained_seed_gene_count": len(retained),
        "retained_seed_gene_ids": retained,
        "seed_retention": 1.0 if not seed_gene_ids else len(retained) / len(seed_gene_ids),
    }


def _positive_task_success(
    *,
    task_type: TaskType,
    best_group: dict[str, Any],
    state: StructuredState,
    task_row: dict[str, Any],
) -> dict[str, Any]:
    metrics = best_group.get("metrics", {})
    jaccard = float(metrics.get("jaccard") or 0.0)
    precision = float(metrics.get("precision") or 0.0)
    recall = float(metrics.get("recall") or 0.0)
    thresholds = TASK_SUCCESS_THRESHOLDS[task_type.value]
    failure_reasons: list[str] = []

    if recall < thresholds["positive_recall"]:
        failure_reasons.append("target_recall_below_positive_threshold")
    if jaccard < thresholds["positive_jaccard"]:
        failure_reasons.append("target_jaccard_below_positive_threshold")
    if task_type == TaskType.REFINEMENT and precision < thresholds["positive_precision"]:
        failure_reasons.append("target_precision_below_positive_threshold")
    if task_type == TaskType.EXPLANATION:
        seed_retention = _visible_seed_retention(state, task_row)
        if seed_retention["seed_retention"] < thresholds["positive_recall"]:
            failure_reasons.append("visible_module_genes_were_dropped")
    else:
        seed_retention = _visible_seed_retention(state, task_row)

    relationship_ok = state.relationship_status in {
        RelationshipStatus.VALIDATED_GROUP,
        RelationshipStatus.PARTIALLY_OBSERVED_GROUP,
    }
    if not relationship_ok:
        failure_reasons.append("relationship_status_not_group")

    positive = not failure_reasons
    partial = (
        not positive
        and relationship_ok
        and (
            recall >= thresholds["partial_recall"]
            or jaccard >= thresholds["partial_jaccard"]
        )
    )
    if task_type == TaskType.REFINEMENT and partial and precision < thresholds["partial_precision"]:
        partial = False
    if task_type == TaskType.EXPLANATION and partial and seed_retention["seed_retention"] < thresholds["partial_recall"]:
        partial = False

    if positive:
        level = "positive"
    elif partial:
        level = "partial"
    else:
        level = "negative"

    return {
        "task_success": positive,
        "task_success_level": level,
        "task_quality_failure_reasons": failure_reasons,
        "thresholds": thresholds,
        "best_group": best_group,
        "metrics": {
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
        },
        "relationship_status": state.relationship_status.value,
        "seed_retention": seed_retention,
    }


def _none_task_success(
    *,
    state: StructuredState,
    expected_relationship: RelationshipStatus,
) -> dict[str, Any]:
    predicted_gene_ids = _flatten_predicted_gene_ids(state)
    failure_reasons: list[str] = []
    if state.relationship_status != expected_relationship:
        failure_reasons.append("relationship_status_not_insufficient_support")
    if predicted_gene_ids:
        failure_reasons.append("predicted_genes_not_empty")

    positive = not failure_reasons
    partial = (
        not positive
        and (
            state.relationship_status == expected_relationship
            or not predicted_gene_ids
        )
    )
    return {
        "task_success": positive,
        "task_success_level": "positive" if positive else ("partial" if partial else "negative"),
        "task_quality_failure_reasons": failure_reasons,
        "thresholds": TASK_SUCCESS_THRESHOLDS["none"],
        "expected_relationship": expected_relationship.value,
        "relationship_status": state.relationship_status.value,
        "predicted_gene_count": len(predicted_gene_ids),
        "predicted_gene_ids": predicted_gene_ids,
    }


def _mechanism_cap_for_task_success(task_success: dict[str, Any]) -> float:
    level = task_success.get("task_success_level")
    if level == "positive":
        return 1.0
    if level == "partial":
        return 0.50
    return 0.15


def _calibrate_task_success_with_evidence(
    task_success: dict[str, Any],
    *,
    task_type: TaskType,
    state: StructuredState,
    mechanism_evidence: dict[str, Any],
) -> dict[str, Any]:
    if task_type == TaskType.NONE:
        return task_success
    enrichment = _safe_dict(_safe_dict(mechanism_evidence.get("consensus")).get("enrichment"))
    if not enrichment.get("weak_group_support"):
        return task_success
    if state.relationship_status != RelationshipStatus.VALIDATED_GROUP:
        return task_success

    calibrated = dict(task_success)
    failure_reasons = list(calibrated.get("task_quality_failure_reasons") or [])
    reason = "validated_group_weak_enrichment_support"
    if reason not in failure_reasons:
        failure_reasons.append(reason)
    calibrated["task_quality_failure_reasons"] = failure_reasons
    calibrated["task_success"] = False
    if calibrated.get("task_success_level") == "positive":
        calibrated["task_success_level"] = "partial"
    return calibrated


def _canonical_label_targets(mechanism_labels: dict[str, Any] | None) -> dict[str, set[str]]:
    if mechanism_labels is None:
        return {"ids": set(), "names": set()}

    mechanism_labels = _require_mapping("mechanism_labels", mechanism_labels)
    canonical_ids = {
        str(label_id)
        for label_id in mechanism_labels.get("go_ids", []) + mechanism_labels.get("fcgs_ids", [])
        if label_id
    }
    canonical_names = {
        _normalize_label_name(name)
        for name in (
            mechanism_labels.get("go_names", [])
            + mechanism_labels.get("fcgs_names", [])
            + ([mechanism_labels.get("primary_label")] if mechanism_labels.get("primary_label") else [])
        )
        if name
    }
    return {"ids": canonical_ids, "names": canonical_names}


def _has_canonical_label_targets(canonical_targets: dict[str, set[str]]) -> bool:
    return bool(canonical_targets["ids"] or canonical_targets["names"])


def _mechanistic_accuracy(
    mechanistic_labels: list[MechanisticLabel],
    canonical_targets: dict[str, set[str]],
) -> dict[str, Any]:
    unique_labels: list[MechanisticLabel] = []
    seen: set[tuple[str, str]] = set()
    for label in mechanistic_labels:
        fingerprint = (label.label_id or "", _normalize_label_name(label.label_name))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_labels.append(label)

    matched_labels: list[dict[str, str | None]] = []
    for label in unique_labels:
        label_id = label.label_id or ""
        normalized_name = _normalize_label_name(label.label_name)
        if label_id in canonical_targets["ids"] or normalized_name in canonical_targets["names"]:
            matched_labels.append(
                {
                    "label_id": label.label_id,
                    "label_name": label.label_name,
                }
            )

    accuracy = _safe_divide(len(matched_labels), len(unique_labels))
    return {
        "accuracy": accuracy,
        "predicted_count": len(unique_labels),
        "matched_count": len(matched_labels),
        "matched_labels": matched_labels,
    }


_GENERIC_MECHANISM_TERMS = {
    "network module",
    "connected subgraph",
    "co-expression",
    "coexpressed",
    "co-expressed",
    "protein binding",
    "shared group",
    "functional module",
    "coherent module",
    "biological process",
    "molecular function",
    "cellular component",
}

_ABSTENTION_TERMS = (
    "insufficient",
    "inconclusive",
    "unresolved",
    "does not support",
    "no specific",
    "unknown",
)

MECHANISM_EVIDENCE_SCORE_WEIGHTS = {
    "specific_claim": {
        "grounding": 0.25,
        "consensus": 0.25,
        "specificity": 0.18,
        "network_agreement": 0.12,
        "stability": 0.10,
        "cross_tool_agreement": 0.10,
        "unsupported_claim_multiplier": 0.35,
    },
    "abstention": {
        "abstention": 0.60,
        "network_agreement": 0.20,
        "grounding": 0.20,
    },
    "evidence_strength": {
        "mygene_scale": 0.75,
        "network_scale": 0.50,
        "network_without_consensus_scale": 0.35,
        "unsupported_threshold": 0.05,
    },
}
WEAK_GROUP_ENRICHMENT_MAX_INTERSECTION = 1
WEAK_GROUP_ENRICHMENT_MAX_PRECISION = 0.25
WEAK_GROUP_ENRICHMENT_MIN_QUERY_SIZE = 4
WEAK_GROUP_ENRICHMENT_SCORE_CAP = 0.18
WEAK_VALIDATED_GROUP_MECHANISM_SCORE_CAP = 0.35


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _enrichment_query_size(result: dict[str, Any], payload: dict[str, Any] | None = None) -> int:
    query_size = result.get("query_size")
    if isinstance(query_size, (int, float)) and query_size >= 0:
        return int(query_size)
    if isinstance(payload, dict):
        query_gene_ids = payload.get("query_gene_ids")
        if isinstance(query_gene_ids, list):
            return len([gene_id for gene_id in query_gene_ids if gene_id])
    return 0


def _is_weak_group_enrichment(result: dict[str, Any], *, query_size: int) -> bool:
    intersection_size = int(result.get("intersection_size") or 0)
    precision = float(result.get("precision") or 0.0)
    return (
        intersection_size <= WEAK_GROUP_ENRICHMENT_MAX_INTERSECTION
        and (
            query_size >= WEAK_GROUP_ENRICHMENT_MIN_QUERY_SIZE
            or precision <= WEAK_GROUP_ENRICHMENT_MAX_PRECISION
        )
    )


def _text_contains(haystack: str, needle: str) -> bool:
    return bool(needle and needle.lower() in haystack.lower())


def _payload_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True).lower()
    except TypeError:
        return str(value).lower()


def _observed_mechanism_terms(state: StructuredState) -> dict[str, Any]:
    ids: set[str] = set()
    names: set[str] = set()
    text_parts: list[str] = []
    for record in state.evidence_log:
        provenance = record.provenance
        payload = provenance.get("payload")
        text_parts.append(record.summary)
        text_parts.append(_payload_text(payload))
        if not isinstance(payload, dict):
            continue
        if provenance.get("tool_name") == "enrich_gene_set":
            for result in payload.get("results", []):
                if not isinstance(result, dict):
                    continue
                native = result.get("native")
                name = result.get("name")
                if native:
                    ids.add(str(native))
                if name:
                    names.add(_normalize_label_name(str(name)))
        elif provenance.get("tool_name") == "query_mygene":
            for hit in payload.get("results", []):
                if not isinstance(hit, dict):
                    continue
                name = hit.get("name")
                if name:
                    names.add(_normalize_label_name(str(name)))
                go_payload = hit.get("go")
                if isinstance(go_payload, dict):
                    for branch_payload in go_payload.values():
                        entries = branch_payload if isinstance(branch_payload, list) else [branch_payload]
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            go_id = entry.get("id")
                            term = entry.get("term")
                            if go_id:
                                ids.add(str(go_id))
                            if term:
                                names.add(_normalize_label_name(str(term)))
                pathway_payload = hit.get("pathway")
                if isinstance(pathway_payload, dict):
                    for source_payload in pathway_payload.values():
                        entries = source_payload if isinstance(source_payload, list) else [source_payload]
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            pathway_id = entry.get("id")
                            pathway_name = entry.get("name")
                            if pathway_id:
                                ids.add(str(pathway_id))
                            if pathway_name:
                                names.add(_normalize_label_name(str(pathway_name)))
    return {"ids": ids, "names": names, "text": " ".join(text_parts).lower()}


def _evidence_by_tool(state: StructuredState) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in state.evidence_log:
        tool_name = record.provenance.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            grouped.setdefault(tool_name, []).append(
                {
                    "record": record,
                    "payload": record.provenance.get("payload"),
                }
            )
    return grouped


def _best_enrichment_support(state: StructuredState) -> dict[str, Any]:
    best: dict[str, Any] = {
        "score": 0.0,
        "term": None,
        "intersection_size": 0,
        "precision": 0.0,
        "p_value": None,
        "query_size": 0,
        "weak_group_support": False,
    }
    for item in _evidence_by_tool(state).get("enrich_gene_set", []):
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            intersection_size = int(result.get("intersection_size") or 0)
            precision = float(result.get("precision") or 0.0)
            query_size = _enrichment_query_size(result, payload)
            p_value = result.get("p_value")
            significance_score = 1.0 if result.get("significant") else 0.5
            if isinstance(p_value, (int, float)):
                if p_value <= 1e-6:
                    significance_score = 1.0
                elif p_value <= 1e-3:
                    significance_score = max(significance_score, 0.8)
                elif p_value <= 0.05:
                    significance_score = max(significance_score, 0.6)
            consensus_score = min(intersection_size / 3.0, 1.0)
            score = _clamp01(0.45 * significance_score + 0.35 * consensus_score + 0.20 * precision)
            weak_group_support = _is_weak_group_enrichment(result, query_size=query_size)
            if weak_group_support:
                score = min(score, WEAK_GROUP_ENRICHMENT_SCORE_CAP)
            if score > best["score"]:
                best = {
                    "score": score,
                    "term": result.get("name"),
                    "term_id": result.get("native"),
                    "source": result.get("source"),
                    "intersection_size": intersection_size,
                    "precision": precision,
                    "p_value": p_value,
                    "query_size": query_size,
                    "weak_group_support": weak_group_support,
                }
    return best


def _mygene_support(state: StructuredState) -> dict[str, Any]:
    queried_genes: set[str] = set()
    hit_genes: set[str] = set()
    informative_hits = 0
    for item in _evidence_by_tool(state).get("query_mygene", []):
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        query = payload.get("query")
        if query:
            queried_genes.add(str(query))
        for hit in payload.get("results", []):
            if not isinstance(hit, dict):
                continue
            if hit.get("symbol") or hit.get("name") or hit.get("summary") or hit.get("go") or hit.get("pathway"):
                informative_hits += 1
                if query:
                    hit_genes.add(str(query))
    return {
        "queried_gene_count": len(queried_genes),
        "informative_gene_count": len(hit_genes),
        "informative_hit_count": informative_hits,
        "score": _clamp01(len(hit_genes) / 3.0),
    }


def _network_support(state: StructuredState) -> dict[str, Any]:
    score = 0.0
    signals: list[str] = []
    for item in _evidence_by_tool(state).get("induce_subgraph", []):
        payload = item.get("payload")
        if isinstance(payload, dict) and int(payload.get("combined_edge_count") or 0) > 0:
            score = max(score, 0.8)
            signals.append("induce_subgraph_edges")
    for tool_name in ("shortest_paths", "shortest_path"):
        for item in _evidence_by_tool(state).get(tool_name, []):
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue
            if tool_name == "shortest_paths":
                paths = payload.get("paths")
                if isinstance(paths, list) and paths:
                    first_path = paths[0] if isinstance(paths[0], dict) else {}
                    path_length = first_path.get("path_length")
                    score = max(score, 1.0 if path_length is not None and int(path_length) <= 2 else 0.6)
                    signals.append(tool_name)
            elif payload.get("hop_count") is not None:
                hop_count = int(payload.get("hop_count"))
                score = max(score, 1.0 if hop_count <= 2 else 0.6)
                signals.append(tool_name)
    for tool_name in ("rwr", "rwr_loe", "rwr_multiplex", "rwr_monoplex"):
        for item in _evidence_by_tool(state).get(tool_name, []):
            payload = item.get("payload")
            if isinstance(payload, dict) and (payload.get("results") or payload.get("ranked_genes")):
                score = max(score, 0.7)
                signals.append(tool_name)
    return {"score": score, "signals": sorted(set(signals))}


def _enrichment_result_score(result: dict[str, Any]) -> float:
    intersection_size = int(result.get("intersection_size") or 0)
    precision = float(result.get("precision") or 0.0)
    query_size = _enrichment_query_size(result)
    p_value = result.get("p_value")
    significance_score = 1.0 if result.get("significant") else 0.5
    if isinstance(p_value, (int, float)):
        if p_value <= 1e-6:
            significance_score = 1.0
        elif p_value <= 1e-3:
            significance_score = max(significance_score, 0.8)
        elif p_value <= 0.05:
            significance_score = max(significance_score, 0.6)
    consensus_score = min(intersection_size / 3.0, 1.0)
    score = _clamp01(0.45 * significance_score + 0.35 * consensus_score + 0.20 * precision)
    if _is_weak_group_enrichment(result, query_size=query_size):
        score = min(score, WEAK_GROUP_ENRICHMENT_SCORE_CAP)
    return score


def _stability_support(state: StructuredState) -> dict[str, Any]:
    by_term: dict[str, list[dict[str, Any]]] = {}
    for item in _evidence_by_tool(state).get("enrich_gene_set", []):
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        query_gene_ids = sorted(str(gene_id) for gene_id in (payload.get("query_gene_ids") or []) if gene_id)
        query_key = "|".join(query_gene_ids)
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            term_key = str(result.get("native") or _normalize_label_name(str(result.get("name") or "")))
            if not term_key:
                continue
            by_term.setdefault(term_key, []).append(
                {
                    "term": result.get("name"),
                    "term_id": result.get("native"),
                    "query_key": query_key,
                    "query_gene_count": len(query_gene_ids),
                    "support_score": _enrichment_result_score(result),
                }
            )

    best: dict[str, Any] = {
        "score": 0.0,
        "available": False,
        "term": None,
        "term_id": None,
        "query_gene_set_count": 0,
        "negative_control_weakened": False,
    }
    for observations in by_term.values():
        unique_query_keys = {item["query_key"] for item in observations if item["query_key"]}
        if len(unique_query_keys) < 2:
            continue
        support_scores = [float(item["support_score"]) for item in observations]
        base_score = sum(support_scores) / len(support_scores)
        min_size = min(int(item["query_gene_count"]) for item in observations)
        max_size = max(int(item["query_gene_count"]) for item in observations)
        smaller_best = max(
            float(item["support_score"]) for item in observations if int(item["query_gene_count"]) == min_size
        )
        larger_best = max(
            float(item["support_score"]) for item in observations if int(item["query_gene_count"]) == max_size
        )
        negative_control_weakened = max_size > min_size and larger_best + 0.05 < smaller_best
        score = _clamp01(0.85 * base_score + (0.15 if negative_control_weakened else 0.0))
        if score > best["score"]:
            representative = observations[0]
            best = {
                "score": score,
                "available": True,
                "term": representative["term"],
                "term_id": representative["term_id"],
                "query_gene_set_count": len(unique_query_keys),
                "negative_control_weakened": negative_control_weakened,
            }
    return best


def _specificity_score(labels: list[MechanisticLabel], claim_text: str) -> dict[str, Any]:
    if not labels and not claim_text:
        return {"score": 0.0, "generic_labels": [], "label_count": 0}
    generic_labels: list[str] = []
    label_scores: list[float] = []
    for label in labels:
        normalized = _normalize_label_name(label.label_name)
        is_generic = any(term in normalized for term in _GENERIC_MECHANISM_TERMS)
        if is_generic:
            generic_labels.append(label.label_name)
        if label.label_id:
            label_scores.append(1.0 if not is_generic else 0.35)
        elif label.label_source in (
            LabelSource.GO,
            LabelSource.REACTOME,
            LabelSource.FCGS,
            LabelSource.COMPLEX_NAME,
        ):
            label_scores.append(0.8 if not is_generic else 0.3)
        else:
            label_scores.append(0.55 if not is_generic and len(normalized.split()) >= 2 else 0.2)
    if not label_scores:
        claim_generic = any(term in _normalize_label_name(claim_text) for term in _GENERIC_MECHANISM_TERMS)
        label_scores.append(0.35 if claim_generic else 0.5)
    return {
        "score": _clamp01(sum(label_scores) / len(label_scores)),
        "generic_labels": generic_labels,
        "label_count": len(labels),
    }


def _grounding_score(
    state: StructuredState,
    labels: list[MechanisticLabel],
    claim_text: str,
) -> dict[str, Any]:
    if not labels and not claim_text:
        return {"score": 0.0, "grounded_label_count": 0, "label_count": 0}
    evidence_ids = {record.evidence_id for record in state.evidence_log}
    observed = _observed_mechanism_terms(state)
    observed_text = str(observed.get("text") or "")
    grounded = 0
    for label in labels:
        normalized_name = _normalize_label_name(label.label_name)
        id_grounded = bool(label.label_id and label.label_id in observed["ids"])
        name_grounded = normalized_name in observed["names"] or _text_contains(observed_text, label.label_name)
        citation_grounded = bool(set(label.evidence_ids) & evidence_ids)
        if id_grounded or name_grounded or citation_grounded:
            grounded += 1
    if labels:
        score = grounded / len(labels)
    else:
        score = 0.4 if claim_text and observed_text and _text_contains(observed_text, claim_text[:48]) else 0.0
    return {
        "score": _clamp01(score),
        "grounded_label_count": grounded,
        "label_count": len(labels),
    }


def _abstention_score(
    state: StructuredState,
    *,
    claim_text: str,
    evidence_strength: float,
) -> dict[str, Any]:
    normalized_claim = _normalize_label_name(claim_text)
    abstains = (
        state.relationship_status == RelationshipStatus.INSUFFICIENT_SUPPORT
        or any(term in normalized_claim for term in _ABSTENTION_TERMS)
    )
    if abstains:
        score = 1.0 - evidence_strength
    elif evidence_strength <= 0.05 and claim_text:
        score = 0.0
    else:
        score = 0.5
    return {"score": _clamp01(score), "abstains": abstains}


def _mechanism_evidence_quality(
    state: StructuredState,
    *,
    claim_text: str = "",
    task_type: TaskType | None = None,
) -> dict[str, Any]:
    labels = list(state.mechanistic_labels)
    enrichment = _best_enrichment_support(state)
    mygene = _mygene_support(state)
    network = _network_support(state)
    stability = _stability_support(state)
    evidence_weights = MECHANISM_EVIDENCE_SCORE_WEIGHTS["evidence_strength"]
    claim_weights = MECHANISM_EVIDENCE_SCORE_WEIGHTS["specific_claim"]
    abstention_weights = MECHANISM_EVIDENCE_SCORE_WEIGHTS["abstention"]
    annotation_evidence_strength = max(
        enrichment["score"],
        mygene["score"] * evidence_weights["mygene_scale"],
    )
    evidence_strength = max(
        enrichment["score"],
        mygene["score"] * evidence_weights["mygene_scale"],
        network["score"] * evidence_weights["network_scale"],
    )
    grounding = _grounding_score(state, labels, claim_text)
    specificity = _specificity_score(labels, claim_text)
    consensus_score = max(enrichment["score"], mygene["score"])
    network_score = (
        network["score"]
        if consensus_score > 0
        else network["score"] * evidence_weights["network_without_consensus_scale"]
    )
    abstention = _abstention_score(state, claim_text=claim_text, evidence_strength=evidence_strength)

    if abstention["abstains"]:
        if task_type is not None and task_type != TaskType.NONE and annotation_evidence_strength <= 0.05:
            total = 0.0
        else:
            total = (
                abstention_weights["abstention"] * abstention["score"]
                + abstention_weights["network_agreement"] * network_score
                + abstention_weights["grounding"] * grounding["score"]
            )
    else:
        total = (
            claim_weights["grounding"] * grounding["score"]
            + claim_weights["consensus"] * consensus_score
            + claim_weights["specificity"] * specificity["score"]
            + claim_weights["network_agreement"] * network_score
            + claim_weights["stability"] * stability["score"]
            + claim_weights["cross_tool_agreement"] * min(enrichment["score"], mygene["score"])
        )
        if evidence_strength <= evidence_weights["unsupported_threshold"] and (labels or claim_text):
            total *= claim_weights["unsupported_claim_multiplier"]
        if (
            task_type is not None
            and task_type != TaskType.NONE
            and enrichment.get("weak_group_support")
            and state.relationship_status == RelationshipStatus.VALIDATED_GROUP
        ):
            total = min(total, WEAK_VALIDATED_GROUP_MECHANISM_SCORE_CAP)

    return {
        "score": _clamp01(total),
        "grounding": grounding,
        "consensus": {
            "score": _clamp01(consensus_score),
            "enrichment": enrichment,
            "mygene": mygene,
        },
        "specificity": specificity,
        "network_agreement": network,
        "stability": stability,
        "abstention": abstention,
        "cross_tool_agreement": min(enrichment["score"], mygene["score"]),
        "evidence_strength": evidence_strength,
        "annotation_evidence_strength": annotation_evidence_strength,
        "weak_group_evidence": {
            "weak": bool(enrichment.get("weak_group_support")),
            "term": enrichment.get("term"),
            "term_id": enrichment.get("term_id"),
            "intersection_size": enrichment.get("intersection_size"),
            "precision": enrichment.get("precision"),
            "query_size": enrichment.get("query_size"),
        },
        "weights": MECHANISM_EVIDENCE_SCORE_WEIGHTS,
    }


def _score_positive_complex_transition(
    prior_state: StructuredState,
    post_state: StructuredState,
    *,
    target_gene_ids: list[str],
    weights: ComplexMetricWeights,
) -> tuple[float, dict[str, Any]]:
    pre_match = _best_group_match(prior_state, target_gene_ids, weights)
    post_match = _best_group_match(post_state, target_gene_ids, weights)
    return (
        post_match["score"] - pre_match["score"],
        {
            "complex_score_pre": pre_match["score"],
            "complex_score_post": post_match["score"],
            "best_group_pre": pre_match,
            "best_group_post": post_match,
        },
    )


def _none_state_score(
    state: StructuredState,
    *,
    expected_relationship: RelationshipStatus,
    config: LocalScoringConfig,
) -> tuple[float, dict[str, Any]]:
    predicted_gene_ids = _flatten_predicted_gene_ids(state)
    relationship_score = 1.0 if state.relationship_status == expected_relationship else 0.0
    abstention_score = 1.0 if not predicted_gene_ids else 0.0
    score = (
        config.none_relationship_weight * relationship_score
        + config.none_abstention_weight * abstention_score
    )
    return (
        score,
        {
            "relationship_score": relationship_score,
            "abstention_score": abstention_score,
            "predicted_gene_count": len(predicted_gene_ids),
            "predicted_gene_ids": predicted_gene_ids,
        },
    )


def _score_none_transition(
    prior_state: StructuredState,
    post_state: StructuredState,
    *,
    expected_relationship: RelationshipStatus,
    config: LocalScoringConfig,
) -> tuple[float, dict[str, Any]]:
    pre_score, pre_metadata = _none_state_score(
        prior_state,
        expected_relationship=expected_relationship,
        config=config,
    )
    post_score, post_metadata = _none_state_score(
        post_state,
        expected_relationship=expected_relationship,
        config=config,
    )
    return (
        post_score - pre_score,
        {
            "complex_score_pre": pre_score,
            "complex_score_post": post_score,
            "none_score_pre": pre_metadata,
            "none_score_post": post_metadata,
        },
    )


def _schema_validation_errors(
    branch: CandidateBranch,
    *,
    prior_state: StructuredState,
    available_gene_ids: set[str] | None,
    available_layers: set[str] | None,
) -> list[str]:
    validation = validate_candidate_branch(branch)
    generator_errors = branch.metadata.get("generator_errors", [])
    if isinstance(generator_errors, list):
        for error in generator_errors:
            if isinstance(error, str) and error:
                validation.add_error(error)

    tool_action = branch.actor_step.tool_action
    if tool_action is not None:
        if branch.observation is None:
            validation.add_error("A branch with a tool_action must include an observation.")
        validation.extend(
            validate_tool_action_semantics(
                tool_action,
                state=prior_state,
                available_gene_ids=available_gene_ids,
                available_layers=available_layers,
            )
        )

    return list(validation.errors)


def score_candidate_branch(
    task_row: dict[str, Any],
    prior_state: StructuredState,
    branch: CandidateBranch,
    *,
    step_index: int,
    max_steps: int,
    prior_actions: Iterable[ToolAction] | None = None,
    available_gene_ids: set[str] | None = None,
    available_layers: set[str] | None = None,
    config: LocalScoringConfig = DEFAULT_LOCAL_SCORING_CONFIG,
) -> LocalScoreBreakdown:
    """Score one candidate branch using hidden CORUM supervision.

    Parameters are intentionally direct:
    - `task_row` is one canonical CORUM task row
    - `prior_state` is the shared-prefix state before the branch
    - `branch` is the proposed continuation to score
    - `step_index` and `max_steps` define the length penalty term
    """

    task_row = _require_mapping("task_row", task_row)
    if not isinstance(prior_state, StructuredState):
        _fail("prior_state must be a StructuredState instance.")
    if not isinstance(branch, CandidateBranch):
        _fail("branch must be a CandidateBranch instance.")
    _require_positive_int("max_steps", max_steps)
    if not isinstance(step_index, int) or step_index < 0:
        _fail("step_index must be a non-negative int.")

    task_type = TaskType(task_row["task_type"])
    hidden_target = _require_mapping("hidden_target", task_row["hidden_target"])
    post_state = branch.verifier_step.updated_state
    schema_errors = _schema_validation_errors(
        branch,
        prior_state=prior_state,
        available_gene_ids=available_gene_ids,
        available_layers=available_layers,
    )
    schema_score = 0.0 if schema_errors else 1.0

    if task_type == TaskType.NONE:
        expected_relationship = RelationshipStatus(hidden_target["relationship_status"])
        complex_delta, complex_metadata = _score_none_transition(
            prior_state,
            post_state,
            expected_relationship=expected_relationship,
            config=config,
        )
        task_profile_metadata: dict[str, Any] = {
            "none_relationship_weight": config.none_relationship_weight,
            "none_abstention_weight": config.none_abstention_weight,
        }
        task_success = _none_task_success(
            state=post_state,
            expected_relationship=expected_relationship,
        )
    else:
        target_gene_ids = hidden_target.get("target_gene_ids")
        if not isinstance(target_gene_ids, list) or not target_gene_ids:
            _fail("Positive tasks must include a non-empty hidden_target.target_gene_ids list.")
        weights = config.complex_weights_for_task(task_type)
        complex_delta, complex_metadata = _score_positive_complex_transition(
            prior_state,
            post_state,
            target_gene_ids=target_gene_ids,
            weights=weights,
        )
        task_profile_metadata = {
            "jaccard_weight": weights.jaccard,
            "precision_weight": weights.precision,
            "recall_weight": weights.recall,
        }
        task_success = _positive_task_success(
            task_type=task_type,
            best_group=complex_metadata["best_group_post"],
            state=post_state,
            task_row=task_row,
        )

    canonical_targets = _canonical_label_targets(task_row.get("mechanism_labels"))
    pre_mechanistic = _mechanistic_accuracy(prior_state.mechanistic_labels, canonical_targets)
    post_mechanistic = _mechanistic_accuracy(post_state.mechanistic_labels, canonical_targets)
    pre_mechanism_evidence = _mechanism_evidence_quality(
        prior_state,
        claim_text="",
        task_type=task_type,
    )
    post_mechanism_evidence = _mechanism_evidence_quality(
        post_state,
        claim_text=branch.verifier_step.updated_interpretation.mechanistic_claim,
        task_type=task_type,
    )
    task_success = _calibrate_task_success_with_evidence(
        task_success,
        task_type=task_type,
        state=post_state,
        mechanism_evidence=post_mechanism_evidence,
    )
    mechanism_evidence_delta = (
        post_mechanism_evidence["score"] - pre_mechanism_evidence["score"]
    )
    if _has_canonical_label_targets(canonical_targets):
        mechanistic_delta = post_mechanistic["accuracy"] - pre_mechanistic["accuracy"]
        mechanism_score_source = "hidden_label_targets"
    else:
        mechanistic_delta = mechanism_evidence_delta
        mechanism_score_source = "evidence_grounded_unsupervised"

    task_mismatch_penalty = 0.0
    effective_mechanistic_delta = mechanistic_delta
    if task_type == TaskType.EXPLANATION:
        pre_metrics = _safe_dict(_safe_dict(complex_metadata.get("best_group_pre")).get("metrics"))
        post_metrics = _safe_dict(_safe_dict(complex_metadata.get("best_group_post")).get("metrics"))
        pre_recall = float(pre_metrics.get("recall") or 0.0)
        post_recall = float(post_metrics.get("recall") or 0.0)
        if post_recall + 1e-9 < pre_recall:
            task_mismatch_penalty = pre_recall - post_recall
            effective_mechanistic_delta = min(effective_mechanistic_delta, 0.0)
    elif task_type != TaskType.NONE and task_success["task_success_level"] == "negative" and complex_delta < 0:
        effective_mechanistic_delta = min(effective_mechanistic_delta, 0.0)

    tool_action = branch.actor_step.tool_action
    observation = branch.observation
    duplicate_tool_call = bool(
        tool_action is not None and prior_actions is not None and is_duplicate_tool_action(tool_action, prior_actions)
    )
    empty_annotation_observation = bool(
        observation is not None
        and observation.status == ToolObservationStatus.EMPTY
        and observation.provenance.get("tool_name") in {"query_mygene", "enrich_gene_set"}
    )
    invalid_observation = bool(
        observation is not None
        and not empty_annotation_observation
        and observation.status
        in (ToolObservationStatus.EMPTY, ToolObservationStatus.INVALID, ToolObservationStatus.ERROR)
    )
    missing_observation = tool_action is not None and observation is None
    invalid_tool_call_increment = 0
    invalid_reasons: list[str] = []
    if tool_action is not None:
        semantic_validation = validate_tool_action_semantics(
            tool_action,
            state=prior_state,
            available_gene_ids=available_gene_ids,
            available_layers=available_layers,
        )
        if not semantic_validation.valid:
            invalid_reasons.extend(semantic_validation.errors)
        if duplicate_tool_call:
            invalid_reasons.append("duplicate_tool_call")
        if invalid_observation:
            invalid_reasons.append(f"observation_status={observation.status.value}")
        if missing_observation:
            invalid_reasons.append("missing_observation")
        if invalid_reasons:
            invalid_tool_call_increment = 1

    total_tool_calls = prior_state.total_tool_call_count + (1 if tool_action is not None else 0)
    invalid_tool_calls = prior_state.invalid_tool_call_count + invalid_tool_call_increment
    invalid_ratio = _safe_divide(invalid_tool_calls, total_tool_calls)
    step_fraction = min(step_index / max_steps, 1.0)
    efficiency_penalty = (
        config.step_penalty_lambda * step_fraction
        + config.invalid_call_penalty_lambda * invalid_ratio
    )

    total_score = (
        config.schema_weight * schema_score
        + config.complex_delta_weight * complex_delta
        + config.mechanism_delta_weight * effective_mechanistic_delta
        - config.efficiency_weight * efficiency_penalty
        - task_mismatch_penalty
    )

    score_metadata = {
        "task_type": task_type.value,
        "task_profile": task_profile_metadata,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "relationship_status_pre": prior_state.relationship_status.value,
        "relationship_status_post": post_state.relationship_status.value,
        "relationship_status_expected": hidden_target.get("relationship_status"),
        "complex": complex_metadata,
        "mechanistic": {
            "score_source": mechanism_score_source,
            "accuracy_pre": pre_mechanistic["accuracy"],
            "accuracy_post": post_mechanistic["accuracy"],
            "matched_count_pre": pre_mechanistic["matched_count"],
            "matched_count_post": post_mechanistic["matched_count"],
            "predicted_count_pre": pre_mechanistic["predicted_count"],
            "predicted_count_post": post_mechanistic["predicted_count"],
            "matched_labels_pre": pre_mechanistic["matched_labels"],
            "matched_labels_post": post_mechanistic["matched_labels"],
            "mechanism_evidence_pre": pre_mechanism_evidence,
            "mechanism_evidence_post": post_mechanism_evidence,
            "mechanism_evidence_delta": mechanism_evidence_delta,
            "effective_delta_for_score": effective_mechanistic_delta,
        },
        "task_success": task_success,
        "task_mismatch_penalty": task_mismatch_penalty,
        "efficiency": {
            "step_fraction": step_fraction,
            "total_tool_calls": total_tool_calls,
            "invalid_tool_calls": invalid_tool_calls,
            "invalid_ratio": invalid_ratio,
            "duplicate_tool_call": duplicate_tool_call,
            "invalid_observation": invalid_observation,
            "missing_observation": missing_observation,
            "invalid_tool_call_increment": invalid_tool_call_increment,
            "invalid_reasons": invalid_reasons,
        },
    }

    return LocalScoreBreakdown(
        schema_score=schema_score,
        complex_membership_delta=complex_delta,
        mechanistic_label_delta=mechanistic_delta,
        efficiency_penalty=efficiency_penalty,
        total_score=total_score,
        normalized_score=None,
        mechanism_evidence_delta=mechanism_evidence_delta,
        mechanism_evidence_score=post_mechanism_evidence["score"],
        score_metadata=score_metadata,
    )


def _terminal_schema_score(final_state: StructuredState) -> tuple[float, dict[str, Any]]:
    schema_valid = True
    schema_errors: list[str] = []
    try:
        StructuredState.from_dict(final_state.to_dict())
    except SchemaValidationError as error:
        schema_valid = False
        schema_errors.append(str(error))
    termination_recorded = (
        final_state.continuation_state == ContinuationState.STOP
        and final_state.termination_reason is not None
    )
    return (
        1.0 if schema_valid and termination_recorded else 0.0,
        {
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "termination_recorded": termination_recorded,
            "termination_reason": (
                final_state.termination_reason.value
                if final_state.termination_reason is not None
                else None
            ),
        },
    )


def score_terminal_trajectory(
    task_row: dict[str, Any],
    initial_state: StructuredState,
    final_state: StructuredState,
    *,
    step_count: int,
    max_steps: int,
    config: TerminalScoringConfig = DEFAULT_TERMINAL_SCORING_CONFIG,
) -> dict[str, Any]:
    """Score one completed trajectory using the proposal's terminal reward."""

    task_row = _require_mapping("task_row", task_row)
    if not isinstance(initial_state, StructuredState):
        _fail("initial_state must be a StructuredState instance.")
    if not isinstance(final_state, StructuredState):
        _fail("final_state must be a StructuredState instance.")
    _require_positive_int("max_steps", max_steps)
    if not isinstance(step_count, int) or step_count < 0:
        _fail("step_count must be a non-negative int.")

    task_type = TaskType(task_row["task_type"])
    hidden_target = _require_mapping("hidden_target", task_row["hidden_target"])
    if task_type == TaskType.NONE:
        expected_relationship = RelationshipStatus(hidden_target["relationship_status"])
        initial_complex_score, initial_complex_metadata = _none_state_score(
            initial_state,
            expected_relationship=expected_relationship,
            config=config.local_config,
        )
        final_complex_score, final_complex_metadata = _none_state_score(
            final_state,
            expected_relationship=expected_relationship,
            config=config.local_config,
        )
        complex_metadata: dict[str, Any] = {
            "expected_relationship": expected_relationship.value,
            "initial": initial_complex_metadata,
            "final": final_complex_metadata,
            "none_relationship_weight": config.local_config.none_relationship_weight,
            "none_abstention_weight": config.local_config.none_abstention_weight,
        }
        task_success = _none_task_success(
            state=final_state,
            expected_relationship=expected_relationship,
        )
    else:
        target_gene_ids = hidden_target.get("target_gene_ids")
        if not isinstance(target_gene_ids, list) or not target_gene_ids:
            _fail("Positive tasks must include a non-empty hidden_target.target_gene_ids list.")
        weights = config.complex_weights_for_task(task_type)
        initial_match = _best_group_match(initial_state, target_gene_ids, weights)
        final_match = _best_group_match(final_state, target_gene_ids, weights)
        initial_complex_score = float(initial_match["score"])
        final_complex_score = float(final_match["score"])
        complex_metadata = {
            "target_gene_ids": list(target_gene_ids),
            "initial_best_group": initial_match,
            "final_best_group": final_match,
            "jaccard_weight": weights.jaccard,
            "precision_weight": weights.precision,
            "recall_weight": weights.recall,
        }
        task_success = _positive_task_success(
            task_type=task_type,
            best_group=final_match,
            state=final_state,
            task_row=task_row,
        )

    complex_delta = final_complex_score - initial_complex_score

    canonical_targets = _canonical_label_targets(task_row.get("mechanism_labels"))
    initial_mechanistic = _mechanistic_accuracy(initial_state.mechanistic_labels, canonical_targets)
    final_mechanistic = _mechanistic_accuracy(final_state.mechanistic_labels, canonical_targets)
    initial_mechanism_evidence = _mechanism_evidence_quality(
        initial_state,
        claim_text="",
        task_type=task_type,
    )
    final_mechanism_evidence = _mechanism_evidence_quality(
        final_state,
        claim_text="",
        task_type=task_type,
    )
    task_success = _calibrate_task_success_with_evidence(
        task_success,
        task_type=task_type,
        state=final_state,
        mechanism_evidence=final_mechanism_evidence,
    )
    initial_mechanism_evidence_score = float(initial_mechanism_evidence["score"])
    final_mechanism_evidence_score = float(final_mechanism_evidence["score"])
    if _has_canonical_label_targets(canonical_targets):
        initial_mechanistic_score = float(initial_mechanistic["accuracy"])
        final_mechanistic_score = float(final_mechanistic["accuracy"])
        mechanism_score_source = "hidden_label_targets"
    else:
        initial_mechanistic_score = initial_mechanism_evidence_score
        final_mechanistic_score = final_mechanism_evidence_score
        mechanism_score_source = "evidence_grounded_unsupervised"
    mechanistic_delta = final_mechanistic_score - initial_mechanistic_score
    mechanism_cap = _mechanism_cap_for_task_success(task_success)
    effective_final_mechanistic_score = min(final_mechanistic_score, mechanism_cap)
    effective_mechanistic_delta = effective_final_mechanistic_score - min(
        initial_mechanistic_score,
        mechanism_cap,
    )

    schema_score, schema_metadata = _terminal_schema_score(final_state)
    total_tool_calls = max(0, final_state.total_tool_call_count - initial_state.total_tool_call_count)
    invalid_tool_calls = max(0, final_state.invalid_tool_call_count - initial_state.invalid_tool_call_count)
    invalid_ratio = _safe_divide(invalid_tool_calls, total_tool_calls)
    step_fraction = min(_safe_divide(step_count, max_steps), 1.0)
    efficiency_penalty = (
        config.local_config.step_penalty_lambda * step_fraction
        + config.local_config.invalid_call_penalty_lambda * invalid_ratio
    )

    terminal_reward = (
        config.schema_weight * schema_score
        + config.absolute_complex_weight * final_complex_score
        + config.complex_delta_weight * complex_delta
        + config.absolute_mechanism_weight * effective_final_mechanistic_score
        + config.mechanism_delta_weight * effective_mechanistic_delta
        - config.efficiency_weight * efficiency_penalty
    )

    return {
        "schema_score": schema_score,
        "absolute_complex_score": final_complex_score,
        "complex_delta": complex_delta,
        "absolute_mechanistic_score": final_mechanistic_score,
        "mechanistic_delta": mechanistic_delta,
        "mechanism_evidence_score": final_mechanism_evidence_score,
        "mechanism_evidence_delta": final_mechanism_evidence_score - initial_mechanism_evidence_score,
        "effective_absolute_mechanistic_score": effective_final_mechanistic_score,
        "effective_mechanistic_delta": effective_mechanistic_delta,
        "mechanism_reward_cap": mechanism_cap,
        "task_success": task_success["task_success"],
        "task_success_level": task_success["task_success_level"],
        "task_quality_failure_reasons": task_success["task_quality_failure_reasons"],
        "efficiency_penalty": efficiency_penalty,
        "terminal_reward": terminal_reward,
        "metadata": {
            "task_type": task_type.value,
            "step_count": step_count,
            "max_steps": max_steps,
            "task_success": task_success,
            "schema": schema_metadata,
            "complex": complex_metadata,
            "mechanistic": {
                "score_source": mechanism_score_source,
                "initial": initial_mechanistic,
                "final": final_mechanistic,
                "mechanism_evidence_initial": initial_mechanism_evidence,
                "mechanism_evidence_final": final_mechanism_evidence,
                "effective_final_score_for_reward": effective_final_mechanistic_score,
                "effective_delta_for_reward": effective_mechanistic_delta,
                "reward_cap": mechanism_cap,
            },
            "efficiency": {
                "step_fraction": step_fraction,
                "total_tool_calls": total_tool_calls,
                "invalid_tool_calls": invalid_tool_calls,
                "invalid_ratio": invalid_ratio,
                "step_penalty_lambda": config.local_config.step_penalty_lambda,
                "invalid_call_penalty_lambda": config.local_config.invalid_call_penalty_lambda,
            },
        },
    }


__all__ = [
    "ComplexMetricWeights",
    "DEFAULT_LOCAL_SCORING_CONFIG",
    "DEFAULT_TERMINAL_SCORING_CONFIG",
    "LocalScoringConfig",
    "TerminalScoringConfig",
    "score_candidate_branch",
    "score_terminal_trajectory",
]
