"""Simple state helpers for the MENTOR-RL runtime.

This file does two jobs:

1. build the initial interpretation and structured state from one CORUM task
2. provide small helper functions for updating state without mutating objects

The helpers are intentionally simple. They return new schema objects rather
than mutating the old ones in place, which makes trajectory generation easier
to reason about and replay.
"""

from __future__ import annotations

from typing import Any

from .schemas import (
    ContinuationState,
    EvidenceRecord,
    GeneGroup,
    Interpretation,
    MechanisticLabel,
    RelationshipStatus,
    SchemaValidationError,
    StructuredState,
    TerminationReason,
    UserAnchors,
)


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be a dict, got {type(value).__name__}.")
    return value


def _require_task_field(task_row: dict[str, Any], field_name: str) -> Any:
    if field_name not in task_row:
        _fail(f"Task row is missing required field: {field_name}.")
    return task_row[field_name]


def clone_interpretation(interpretation: Interpretation) -> Interpretation:
    """Return a deep copy of an interpretation object."""

    return Interpretation.from_dict(interpretation.to_dict())


def clone_state(state: StructuredState) -> StructuredState:
    """Return a deep copy of a structured state object."""

    return StructuredState.from_dict(state.to_dict())


def summarize_user_evidence(visible_inputs: dict[str, Any]) -> str:
    """Create a short, human-readable summary of the user-provided evidence."""

    visible_inputs = _require_mapping("visible_inputs", visible_inputs)
    summaries: list[str] = []

    seed_gene_symbols = visible_inputs.get("seed_gene_symbols", [])
    if seed_gene_symbols:
        summaries.append(f"Seed genes: {', '.join(seed_gene_symbols)}.")
    if visible_inputs.get("context_text"):
        summaries.append("User provided free-text context.")
    if visible_inputs.get("graph_query_spec"):
        summaries.append("User provided a graph query specification.")
    if visible_inputs.get("structured_annotations"):
        summaries.append("User provided structured annotations.")

    if not summaries:
        return "User provided the initial query only."
    return " ".join(summaries)


def build_user_anchors_from_task(task_row: dict[str, Any]) -> UserAnchors:
    """Convert one canonical CORUM task row into runtime user anchors."""

    task_row = _require_mapping("task_row", task_row)
    query_text = _require_task_field(task_row, "query_text")
    visible_inputs = _require_mapping("visible_inputs", _require_task_field(task_row, "visible_inputs"))

    return UserAnchors(
        query_text=query_text,
        evidence=visible_inputs,
        evidence_mode=task_row.get("evidence_mode"),
        source_task_id=task_row.get("task_id"),
    )


def build_seed_gene_group(
    visible_inputs: dict[str, Any],
    *,
    group_id: str = "group_0",
    rationale: str = "Initial seed group from the user inputs.",
) -> GeneGroup:
    """Create the starting predicted group from the seed genes in a task row."""

    visible_inputs = _require_mapping("visible_inputs", visible_inputs)
    return GeneGroup(
        group_id=group_id,
        gene_ids=visible_inputs.get("seed_gene_ids", []),
        gene_symbols=visible_inputs.get("seed_gene_symbols", []),
        rationale=rationale,
    )


def make_initial_interpretation(task_row: dict[str, Any]) -> Interpretation:
    """Build the initial working interpretation described in the proposal."""

    task_row = _require_mapping("task_row", task_row)
    query_text = _require_task_field(task_row, "query_text")
    visible_inputs = _require_mapping("visible_inputs", _require_task_field(task_row, "visible_inputs"))

    return Interpretation(
        mechanistic_claim="",
        main_evidence=summarize_user_evidence(visible_inputs),
        uncertainty="",
        next_subgoal=query_text,
    )


def initialize_state_from_corum_task(
    task_row: dict[str, Any],
    *,
    max_budget: int,
) -> tuple[Interpretation, StructuredState]:
    """Create the initial interpretation and structured state from one task row.

    This follows the proposal closely:
    - relationship status starts as `unknown`
    - predicted group starts as the seed genes
    - evidence log starts empty
    - mechanistic labels start empty
    - remaining budget starts at `max_budget`
    - continuation state starts as `continue`
    """

    if max_budget < 0:
        _fail("max_budget must be non-negative.")

    task_row = _require_mapping("task_row", task_row)
    visible_inputs = _require_mapping("visible_inputs", _require_task_field(task_row, "visible_inputs"))

    interpretation = make_initial_interpretation(task_row)
    state = StructuredState(
        user_anchors=build_user_anchors_from_task(task_row),
        relationship_status=RelationshipStatus.UNKNOWN,
        predicted_groups=[build_seed_gene_group(visible_inputs)],
        evidence_log=[],
        mechanistic_labels=[],
        remaining_budget=max_budget,
        continuation_state=ContinuationState.CONTINUE,
        invalid_tool_call_count=0,
        total_tool_call_count=0,
        termination_reason=None,
    )
    return interpretation, state


def append_evidence_record(state: StructuredState, record: EvidenceRecord) -> StructuredState:
    """Return a new state with one more evidence record appended."""

    new_state = clone_state(state)
    new_state.evidence_log.append(record)
    return new_state


def replace_predicted_groups(
    state: StructuredState,
    predicted_groups: list[GeneGroup],
    *,
    relationship_status: RelationshipStatus | None = None,
) -> StructuredState:
    """Return a new state with updated predicted groups."""

    new_state = clone_state(state)
    new_state.predicted_groups = predicted_groups
    if relationship_status is not None:
        new_state.relationship_status = relationship_status
    return new_state


def replace_mechanistic_labels(
    state: StructuredState,
    mechanistic_labels: list[MechanisticLabel],
) -> StructuredState:
    """Return a new state with updated mechanistic labels."""

    new_state = clone_state(state)
    new_state.mechanistic_labels = mechanistic_labels
    return new_state


def record_tool_call(state: StructuredState, *, invalid: bool = False) -> StructuredState:
    """Return a new state with tool-call counters updated."""

    new_state = clone_state(state)
    new_state.total_tool_call_count += 1
    if invalid:
        new_state.invalid_tool_call_count += 1
    return new_state


def decrement_budget(state: StructuredState, amount: int = 1) -> StructuredState:
    """Return a new state with a smaller remaining budget."""

    if amount < 0:
        _fail("Budget decrement amount must be non-negative.")
    new_state = clone_state(state)
    new_state.remaining_budget = max(0, new_state.remaining_budget - amount)
    return new_state


def set_continuation_state(
    state: StructuredState,
    continuation_state: ContinuationState,
    *,
    termination_reason: TerminationReason | None = None,
) -> StructuredState:
    """Return a new state with an updated continuation decision."""

    new_state = clone_state(state)
    new_state.continuation_state = continuation_state
    new_state.termination_reason = termination_reason
    return new_state


def finalize_budget_exhausted(
    interpretation: Interpretation,
    state: StructuredState,
) -> tuple[Interpretation, StructuredState]:
    """Mark a trajectory as ended because the step budget ran out."""

    new_interpretation = clone_interpretation(interpretation)
    new_state = set_continuation_state(
        state,
        ContinuationState.STOP,
        termination_reason=TerminationReason.BUDGET_EXHAUSTED,
    )
    new_state.remaining_budget = 0
    return new_interpretation, new_state


def finalize_model_stop(
    interpretation: Interpretation,
    state: StructuredState,
) -> tuple[Interpretation, StructuredState]:
    """Mark a trajectory as ended because the model chose to stop."""

    new_interpretation = clone_interpretation(interpretation)
    new_interpretation.next_subgoal = ""
    new_state = set_continuation_state(
        state,
        ContinuationState.STOP,
        termination_reason=TerminationReason.MODEL_STOP,
    )
    return new_interpretation, new_state


__all__ = [
    "append_evidence_record",
    "build_seed_gene_group",
    "build_user_anchors_from_task",
    "clone_interpretation",
    "clone_state",
    "decrement_budget",
    "finalize_budget_exhausted",
    "finalize_model_stop",
    "initialize_state_from_corum_task",
    "make_initial_interpretation",
    "record_tool_call",
    "replace_mechanistic_labels",
    "replace_predicted_groups",
    "set_continuation_state",
    "summarize_user_evidence",
]
