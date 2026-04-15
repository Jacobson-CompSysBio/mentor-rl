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
from typing import Any, Iterable

from .schemas import (
    CandidateBranch,
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

    canonical_targets = _canonical_label_targets(task_row.get("mechanism_labels"))
    pre_mechanistic = _mechanistic_accuracy(prior_state.mechanistic_labels, canonical_targets)
    post_mechanistic = _mechanistic_accuracy(post_state.mechanistic_labels, canonical_targets)
    mechanistic_delta = post_mechanistic["accuracy"] - pre_mechanistic["accuracy"]

    tool_action = branch.actor_step.tool_action
    observation = branch.observation
    duplicate_tool_call = bool(
        tool_action is not None and prior_actions is not None and is_duplicate_tool_action(tool_action, prior_actions)
    )
    invalid_observation = bool(
        observation is not None
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
        + config.mechanism_delta_weight * mechanistic_delta
        - config.efficiency_weight * efficiency_penalty
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
            "accuracy_pre": pre_mechanistic["accuracy"],
            "accuracy_post": post_mechanistic["accuracy"],
            "matched_count_pre": pre_mechanistic["matched_count"],
            "matched_count_post": post_mechanistic["matched_count"],
            "predicted_count_pre": pre_mechanistic["predicted_count"],
            "predicted_count_post": post_mechanistic["predicted_count"],
            "matched_labels_pre": pre_mechanistic["matched_labels"],
            "matched_labels_post": post_mechanistic["matched_labels"],
        },
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
        score_metadata=score_metadata,
    )


__all__ = [
    "ComplexMetricWeights",
    "DEFAULT_LOCAL_SCORING_CONFIG",
    "LocalScoringConfig",
    "score_candidate_branch",
]
