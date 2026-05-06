"""Simple runtime schemas for MENTOR-RL.

This module defines the plain Python objects that will be passed between the
tool runtime, the trajectory generator, the scorer, and the DPO dataset
builder. The main design goal is clarity:

- every important runtime object has an explicit dataclass
- each object validates itself on creation
- every object can be converted to and from JSON-friendly dictionaries

These schemas only describe visible runtime state and logged artifacts. Hidden
CORUM supervision stays outside these objects and is only used later for
scoring and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from numbers import Real
from typing import Any


RUNTIME_SCHEMA_VERSION = "mentor-rl-runtime-v1"
KNOWN_TOOL_NAMES = (
    "query_mygene",
    "get_neighbors",
    "shortest_path",
    "rwr_multiplex",
    "rwr_monoplex",
    "induce_subgraph",
)


class SchemaValidationError(ValueError):
    """Raised when a runtime schema instance is malformed."""


class StrEnum(str, Enum):
    """Enum that behaves like a plain string when serialized or printed."""

    def __str__(self) -> str:
        return self.value


class TaskType(StrEnum):
    """The four task families defined by the proposal."""

    RECOVERY = "recovery"
    REFINEMENT = "refinement"
    EXPLANATION = "explanation"
    NONE = "none"


class PreferenceDifficulty(StrEnum):
    """Difficulty bins used when mining DPO preference pairs."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ContinuationState(StrEnum):
    """What the verifier wants the agent to do next."""

    CONTINUE = "continue"
    REVISE = "revise"
    STOP = "stop"


class RelationshipStatus(StrEnum):
    """The model's current view of how the seed genes relate to each other."""

    UNKNOWN = "unknown"
    PARTIALLY_OBSERVED_GROUP = "partially_observed_group"
    VALIDATED_GROUP = "validated_group"
    MULTIPLE_GROUPS = "multiple_groups"
    INSUFFICIENT_SUPPORT = "insufficient_support"


class ToolObservationStatus(StrEnum):
    """High-level status for a tool execution result."""

    SUCCESS = "success"
    EMPTY = "empty"
    INVALID = "invalid"
    ERROR = "error"


class LabelSource(StrEnum):
    """Where a mechanistic label came from."""

    GO = "go"
    FCGS = "fcgs"
    COMPLEX_NAME = "complex_name"
    FREE_TEXT = "free_text"
    OTHER = "other"


class EvidenceSourceType(StrEnum):
    """Where a logged evidence record came from."""

    USER_INPUT = "user_input"
    TOOL_OBSERVATION = "tool_observation"
    INTERMEDIATE_FINDING = "intermediate_finding"
    MODEL_SUMMARY = "model_summary"


class TerminationReason(StrEnum):
    """Why the trajectory ended."""

    MODEL_STOP = "model_stop"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RUNTIME_ERROR = "runtime_error"


def _fail(message: str) -> None:
    raise SchemaValidationError(message)


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be a dict, got {type(value).__name__}.")
    return value


def _require_str(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        _fail(f"{name} must be a string, got {type(value).__name__}.")
    if not allow_empty and not value:
        _fail(f"{name} must be a non-empty string.")
    return value


def _require_optional_str(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _require_str(name, value, allow_empty=True)


def _require_non_negative_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        _fail(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0:
        _fail(f"{name} must be non-negative.")
    return value


def _require_real(name: str, value: Any) -> float:
    if not isinstance(value, Real):
        _fail(f"{name} must be numeric, got {type(value).__name__}.")
    return float(value)


def _coerce_enum(name: str, value: Any, enum_type: type[StrEnum]) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in enum_type)
            _fail(f"{name} must be one of: {allowed}.")
            raise exc  # pragma: no cover
    _fail(f"{name} must be a string or {enum_type.__name__}.")
    raise AssertionError("unreachable")  # pragma: no cover


def _require_unique_str_list(name: str, values: Any) -> list[str]:
    if not isinstance(values, list):
        _fail(f"{name} must be a list.")
    parsed: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item = _require_str(f"{name}[{index}]", value)
        if item in seen:
            _fail(f"{name} must not contain duplicates: {item}.")
        seen.add(item)
        parsed.append(item)
    return parsed


def _validate_gene_fields(
    gene_ids: list[str],
    gene_symbols: list[str],
    *,
    field_name: str,
    allow_empty_ids: bool = False,
) -> None:
    if not allow_empty_ids and not gene_ids:
        _fail(f"{field_name}.gene_ids must not be empty.")
    if gene_symbols and len(gene_symbols) != len(gene_ids):
        _fail(
            f"{field_name}.gene_symbols must either be empty or match the length of "
            f"{field_name}.gene_ids."
        )


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field_name: _serialize(getattr(value, field_name))
            for field_name in value.__dataclass_fields__
        }
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


class SchemaMixin:
    """Small helper that gives schema objects JSON-friendly serializers."""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass
class UserAnchors(SchemaMixin):
    """The user-facing inputs that start a task."""

    query_text: str
    evidence: dict[str, Any]
    evidence_mode: str | None = None
    source_task_id: str | None = None

    def __post_init__(self) -> None:
        self.query_text = _require_str("query_text", self.query_text)
        self.evidence = _require_mapping("evidence", self.evidence)
        self.evidence_mode = _require_optional_str("evidence_mode", self.evidence_mode)
        self.source_task_id = _require_optional_str("source_task_id", self.source_task_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserAnchors":
        payload = _require_mapping("UserAnchors", payload)
        return cls(
            query_text=payload["query_text"],
            evidence=payload.get("evidence", {}),
            evidence_mode=payload.get("evidence_mode"),
            source_task_id=payload.get("source_task_id"),
        )


@dataclass
class Interpretation(SchemaMixin):
    """Human-readable hypothesis state used in the agent loop."""

    mechanistic_claim: str
    main_evidence: str
    uncertainty: str
    next_subgoal: str

    def __post_init__(self) -> None:
        self.mechanistic_claim = _require_str(
            "mechanistic_claim", self.mechanistic_claim, allow_empty=True
        )
        self.main_evidence = _require_str("main_evidence", self.main_evidence, allow_empty=True)
        self.uncertainty = _require_str("uncertainty", self.uncertainty, allow_empty=True)
        self.next_subgoal = _require_str("next_subgoal", self.next_subgoal, allow_empty=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Interpretation":
        payload = _require_mapping("Interpretation", payload)
        return cls(
            mechanistic_claim=payload.get("mechanistic_claim", ""),
            main_evidence=payload.get("main_evidence", ""),
            uncertainty=payload.get("uncertainty", ""),
            next_subgoal=payload.get("next_subgoal", ""),
        )


@dataclass
class ToolAction(SchemaMixin):
    """One tool call proposed by the actor."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str

    def __post_init__(self) -> None:
        self.tool_name = _require_str("tool_name", self.tool_name)
        self.arguments = _require_mapping("arguments", self.arguments)
        self.call_id = _require_str("call_id", self.call_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolAction":
        payload = _require_mapping("ToolAction", payload)
        return cls(
            tool_name=payload["tool_name"],
            arguments=payload.get("arguments", {}),
            call_id=payload["call_id"],
        )


@dataclass
class ToolObservation(SchemaMixin):
    """The structured result returned by the deterministic runtime."""

    status: ToolObservationStatus
    provenance: dict[str, Any]
    call_id: str
    payload: dict[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        self.status = _coerce_enum("status", self.status, ToolObservationStatus)
        self.provenance = _require_mapping("provenance", self.provenance)
        self.call_id = _require_str("call_id", self.call_id)
        if self.payload is not None:
            self.payload = _require_mapping("payload", self.payload)
        self.error = _require_optional_str("error", self.error)

        if self.status == ToolObservationStatus.SUCCESS and self.payload is None:
            _fail("Successful tool observations must include a payload.")
        if self.status in (ToolObservationStatus.INVALID, ToolObservationStatus.ERROR) and not self.error:
            _fail("Invalid or error tool observations must include an error message.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolObservation":
        payload = _require_mapping("ToolObservation", payload)
        return cls(
            status=payload["status"],
            provenance=payload.get("provenance", {}),
            call_id=payload["call_id"],
            payload=payload.get("payload"),
            error=payload.get("error"),
        )


@dataclass
class ActorStep(SchemaMixin):
    """What the actor said and which tool it decided to call."""

    reasoning_text: str
    tool_action: ToolAction | None = None

    def __post_init__(self) -> None:
        self.reasoning_text = _require_str("reasoning_text", self.reasoning_text, allow_empty=True)
        if self.tool_action is not None and not isinstance(self.tool_action, ToolAction):
            _fail("tool_action must be a ToolAction or None.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActorStep":
        payload = _require_mapping("ActorStep", payload)
        tool_action = payload.get("tool_action")
        return cls(
            reasoning_text=payload.get("reasoning_text", ""),
            tool_action=ToolAction.from_dict(tool_action) if tool_action is not None else None,
        )


@dataclass
class GeneGroup(SchemaMixin):
    """A predicted mechanistic group in canonical gene ID space."""

    group_id: str
    gene_ids: list[str]
    gene_symbols: list[str] = field(default_factory=list)
    rationale: str = ""

    def __post_init__(self) -> None:
        self.group_id = _require_str("group_id", self.group_id)
        self.gene_ids = _require_unique_str_list("gene_ids", self.gene_ids)
        self.gene_symbols = _require_unique_str_list("gene_symbols", self.gene_symbols)
        _validate_gene_fields(self.gene_ids, self.gene_symbols, field_name="GeneGroup")
        self.rationale = _require_str("rationale", self.rationale, allow_empty=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GeneGroup":
        payload = _require_mapping("GeneGroup", payload)
        return cls(
            group_id=payload["group_id"],
            gene_ids=payload.get("gene_ids", []),
            gene_symbols=payload.get("gene_symbols", []),
            rationale=payload.get("rationale", ""),
        )


@dataclass
class MechanisticLabel(SchemaMixin):
    """A structured label attached to the current hypothesis."""

    label_source: LabelSource
    label_name: str
    label_id: str | None = None
    evidence_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.label_source = _coerce_enum("label_source", self.label_source, LabelSource)
        self.label_name = _require_str("label_name", self.label_name)
        self.label_id = _require_optional_str("label_id", self.label_id)
        self.evidence_ids = _require_unique_str_list("evidence_ids", self.evidence_ids)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MechanisticLabel":
        payload = _require_mapping("MechanisticLabel", payload)
        return cls(
            label_source=payload["label_source"],
            label_name=payload["label_name"],
            label_id=payload.get("label_id"),
            evidence_ids=payload.get("evidence_ids", []),
        )


@dataclass
class EvidenceRecord(SchemaMixin):
    """One piece of evidence collected during exploration."""

    evidence_id: str
    source_type: EvidenceSourceType
    summary: str
    provenance: dict[str, Any]
    supporting_gene_ids: list[str] = field(default_factory=list)
    supporting_gene_symbols: list[str] = field(default_factory=list)
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        self.evidence_id = _require_str("evidence_id", self.evidence_id)
        self.source_type = _coerce_enum("source_type", self.source_type, EvidenceSourceType)
        self.summary = _require_str("summary", self.summary)
        self.provenance = _require_mapping("provenance", self.provenance)
        self.supporting_gene_ids = _require_unique_str_list(
            "supporting_gene_ids", self.supporting_gene_ids
        )
        self.supporting_gene_symbols = _require_unique_str_list(
            "supporting_gene_symbols", self.supporting_gene_symbols
        )
        _validate_gene_fields(
            self.supporting_gene_ids,
            self.supporting_gene_symbols,
            field_name="EvidenceRecord",
            allow_empty_ids=True,
        )
        self.tool_call_id = _require_optional_str("tool_call_id", self.tool_call_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        payload = _require_mapping("EvidenceRecord", payload)
        return cls(
            evidence_id=payload["evidence_id"],
            source_type=payload["source_type"],
            summary=payload["summary"],
            provenance=payload.get("provenance", {}),
            supporting_gene_ids=payload.get("supporting_gene_ids", []),
            supporting_gene_symbols=payload.get("supporting_gene_symbols", []),
            tool_call_id=payload.get("tool_call_id"),
        )


@dataclass
class StructuredState(SchemaMixin):
    """Machine-readable state used for validation, scoring, and provenance."""

    user_anchors: UserAnchors
    relationship_status: RelationshipStatus
    predicted_groups: list[GeneGroup]
    evidence_log: list[EvidenceRecord]
    mechanistic_labels: list[MechanisticLabel]
    remaining_budget: int
    continuation_state: ContinuationState
    invalid_tool_call_count: int = 0
    total_tool_call_count: int = 0
    termination_reason: TerminationReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_anchors, UserAnchors):
            _fail("user_anchors must be a UserAnchors instance.")
        self.relationship_status = _coerce_enum(
            "relationship_status", self.relationship_status, RelationshipStatus
        )
        if not isinstance(self.predicted_groups, list):
            _fail("predicted_groups must be a list.")
        for index, group in enumerate(self.predicted_groups):
            if not isinstance(group, GeneGroup):
                _fail(f"predicted_groups[{index}] must be a GeneGroup.")
        if not isinstance(self.evidence_log, list):
            _fail("evidence_log must be a list.")
        for index, record in enumerate(self.evidence_log):
            if not isinstance(record, EvidenceRecord):
                _fail(f"evidence_log[{index}] must be an EvidenceRecord.")
        if not isinstance(self.mechanistic_labels, list):
            _fail("mechanistic_labels must be a list.")
        for index, label in enumerate(self.mechanistic_labels):
            if not isinstance(label, MechanisticLabel):
                _fail(f"mechanistic_labels[{index}] must be a MechanisticLabel.")
        self.remaining_budget = _require_non_negative_int("remaining_budget", self.remaining_budget)
        self.continuation_state = _coerce_enum(
            "continuation_state", self.continuation_state, ContinuationState
        )
        self.invalid_tool_call_count = _require_non_negative_int(
            "invalid_tool_call_count", self.invalid_tool_call_count
        )
        self.total_tool_call_count = _require_non_negative_int(
            "total_tool_call_count", self.total_tool_call_count
        )
        if self.invalid_tool_call_count > self.total_tool_call_count:
            _fail("invalid_tool_call_count cannot exceed total_tool_call_count.")
        if self.termination_reason is not None:
            self.termination_reason = _coerce_enum(
                "termination_reason", self.termination_reason, TerminationReason
            )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructuredState":
        payload = _require_mapping("StructuredState", payload)
        return cls(
            user_anchors=UserAnchors.from_dict(payload["user_anchors"]),
            relationship_status=payload["relationship_status"],
            predicted_groups=[GeneGroup.from_dict(item) for item in payload.get("predicted_groups", [])],
            evidence_log=[EvidenceRecord.from_dict(item) for item in payload.get("evidence_log", [])],
            mechanistic_labels=[
                MechanisticLabel.from_dict(item) for item in payload.get("mechanistic_labels", [])
            ],
            remaining_budget=payload["remaining_budget"],
            continuation_state=payload["continuation_state"],
            invalid_tool_call_count=payload.get("invalid_tool_call_count", 0),
            total_tool_call_count=payload.get("total_tool_call_count", 0),
            termination_reason=payload.get("termination_reason"),
        )


@dataclass
class VerifierStep(SchemaMixin):
    """The verifier's updated interpretation, state, and continuation choice."""

    updated_interpretation: Interpretation
    updated_state: StructuredState
    continuation_decision: ContinuationState
    verifier_notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.updated_interpretation, Interpretation):
            _fail("updated_interpretation must be an Interpretation instance.")
        if not isinstance(self.updated_state, StructuredState):
            _fail("updated_state must be a StructuredState instance.")
        self.continuation_decision = _coerce_enum(
            "continuation_decision", self.continuation_decision, ContinuationState
        )
        self.verifier_notes = _require_str("verifier_notes", self.verifier_notes, allow_empty=True)
        if self.updated_state.continuation_state != self.continuation_decision:
            _fail("updated_state.continuation_state must match continuation_decision.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VerifierStep":
        payload = _require_mapping("VerifierStep", payload)
        return cls(
            updated_interpretation=Interpretation.from_dict(payload["updated_interpretation"]),
            updated_state=StructuredState.from_dict(payload["updated_state"]),
            continuation_decision=payload["continuation_decision"],
            verifier_notes=payload.get("verifier_notes", ""),
        )


@dataclass
class LocalScoreBreakdown(SchemaMixin):
    """Deterministic score components for one candidate branch."""

    schema_score: float
    complex_membership_delta: float
    mechanistic_label_delta: float
    efficiency_penalty: float
    total_score: float
    normalized_score: float | None = None
    score_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.schema_score = _require_real("schema_score", self.schema_score)
        self.complex_membership_delta = _require_real(
            "complex_membership_delta", self.complex_membership_delta
        )
        self.mechanistic_label_delta = _require_real(
            "mechanistic_label_delta", self.mechanistic_label_delta
        )
        self.efficiency_penalty = _require_real("efficiency_penalty", self.efficiency_penalty)
        self.total_score = _require_real("total_score", self.total_score)
        if self.normalized_score is not None:
            self.normalized_score = _require_real("normalized_score", self.normalized_score)
            if not 0.0 <= self.normalized_score <= 1.0:
                _fail("normalized_score must be in [0, 1].")
        self.score_metadata = _require_mapping("score_metadata", self.score_metadata)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalScoreBreakdown":
        payload = _require_mapping("LocalScoreBreakdown", payload)
        return cls(
            schema_score=payload["schema_score"],
            complex_membership_delta=payload["complex_membership_delta"],
            mechanistic_label_delta=payload["mechanistic_label_delta"],
            efficiency_penalty=payload["efficiency_penalty"],
            total_score=payload["total_score"],
            normalized_score=payload.get("normalized_score"),
            score_metadata=payload.get("score_metadata", {}),
        )


@dataclass
class CandidateBranch(SchemaMixin):
    """One scored continuation candidate from a shared-prefix branch pool."""

    branch_id: str
    actor_step: ActorStep
    verifier_step: VerifierStep
    local_score: LocalScoreBreakdown
    observation: ToolObservation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.branch_id = _require_str("branch_id", self.branch_id)
        if not isinstance(self.actor_step, ActorStep):
            _fail("actor_step must be an ActorStep instance.")
        if not isinstance(self.verifier_step, VerifierStep):
            _fail("verifier_step must be a VerifierStep instance.")
        if not isinstance(self.local_score, LocalScoreBreakdown):
            _fail("local_score must be a LocalScoreBreakdown instance.")
        if self.observation is not None and not isinstance(self.observation, ToolObservation):
            _fail("observation must be a ToolObservation or None.")
        self.metadata = _require_mapping("metadata", self.metadata)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateBranch":
        payload = _require_mapping("CandidateBranch", payload)
        observation = payload.get("observation")
        return cls(
            branch_id=payload["branch_id"],
            actor_step=ActorStep.from_dict(payload["actor_step"]),
            verifier_step=VerifierStep.from_dict(payload["verifier_step"]),
            local_score=LocalScoreBreakdown.from_dict(payload["local_score"]),
            observation=ToolObservation.from_dict(observation) if observation is not None else None,
            metadata=payload.get("metadata", {}),
        )


@dataclass
class TrajectoryTurn(SchemaMixin):
    """The selected branch at one step of a logged trajectory."""

    trajectory_id: str
    step_index: int
    prior_interpretation: Interpretation
    prior_state: StructuredState
    branch: CandidateBranch
    selected: bool
    finding_text: str = ""

    def __post_init__(self) -> None:
        self.trajectory_id = _require_str("trajectory_id", self.trajectory_id)
        self.step_index = _require_non_negative_int("step_index", self.step_index)
        if not isinstance(self.prior_interpretation, Interpretation):
            _fail("prior_interpretation must be an Interpretation instance.")
        if not isinstance(self.prior_state, StructuredState):
            _fail("prior_state must be a StructuredState instance.")
        if not isinstance(self.branch, CandidateBranch):
            _fail("branch must be a CandidateBranch instance.")
        if not isinstance(self.selected, bool):
            _fail("selected must be a bool.")
        self.finding_text = _require_str("finding_text", self.finding_text, allow_empty=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryTurn":
        payload = _require_mapping("TrajectoryTurn", payload)
        return cls(
            trajectory_id=payload["trajectory_id"],
            step_index=payload["step_index"],
            prior_interpretation=Interpretation.from_dict(payload["prior_interpretation"]),
            prior_state=StructuredState.from_dict(payload["prior_state"]),
            branch=CandidateBranch.from_dict(payload["branch"]),
            selected=payload["selected"],
            finding_text=payload.get("finding_text", ""),
        )


@dataclass
class SharedPrefixContext(SchemaMixin):
    """The shared context seen by all candidates at one decision step."""

    query_text: str
    user_evidence: dict[str, Any]
    interpretation: Interpretation
    state: StructuredState
    source_task_id: str | None = None

    def __post_init__(self) -> None:
        self.query_text = _require_str("query_text", self.query_text)
        self.user_evidence = _require_mapping("user_evidence", self.user_evidence)
        if not isinstance(self.interpretation, Interpretation):
            _fail("interpretation must be an Interpretation instance.")
        if not isinstance(self.state, StructuredState):
            _fail("state must be a StructuredState instance.")
        self.source_task_id = _require_optional_str("source_task_id", self.source_task_id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SharedPrefixContext":
        payload = _require_mapping("SharedPrefixContext", payload)
        return cls(
            query_text=payload["query_text"],
            user_evidence=payload.get("user_evidence", {}),
            interpretation=Interpretation.from_dict(payload["interpretation"]),
            state=StructuredState.from_dict(payload["state"]),
            source_task_id=payload.get("source_task_id"),
        )


@dataclass
class PreferencePair(SchemaMixin):
    """A single DPO training pair mined from one shared-prefix branch pool."""

    pair_id: str
    context: SharedPrefixContext
    chosen: CandidateBranch
    rejected: CandidateBranch
    task_type: TaskType
    difficulty_bin: PreferenceDifficulty
    decision_step: int
    raw_score_chosen: float
    raw_score_rejected: float
    normalized_score_chosen: float
    normalized_score_rejected: float
    score_margin: float
    source_task_id: str
    trajectory_id: str
    trajectory_seed: int
    runtime_version: str = RUNTIME_SCHEMA_VERSION
    evidence_mode: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pair_id = _require_str("pair_id", self.pair_id)
        if not isinstance(self.context, SharedPrefixContext):
            _fail("context must be a SharedPrefixContext instance.")
        if not isinstance(self.chosen, CandidateBranch):
            _fail("chosen must be a CandidateBranch instance.")
        if not isinstance(self.rejected, CandidateBranch):
            _fail("rejected must be a CandidateBranch instance.")
        if self.chosen.branch_id == self.rejected.branch_id:
            _fail("chosen and rejected must reference different branch_ids.")
        self.task_type = _coerce_enum("task_type", self.task_type, TaskType)
        self.difficulty_bin = _coerce_enum(
            "difficulty_bin", self.difficulty_bin, PreferenceDifficulty
        )
        self.decision_step = _require_non_negative_int("decision_step", self.decision_step)
        self.raw_score_chosen = _require_real("raw_score_chosen", self.raw_score_chosen)
        self.raw_score_rejected = _require_real("raw_score_rejected", self.raw_score_rejected)
        self.normalized_score_chosen = _require_real(
            "normalized_score_chosen", self.normalized_score_chosen
        )
        self.normalized_score_rejected = _require_real(
            "normalized_score_rejected", self.normalized_score_rejected
        )
        self.score_margin = _require_real("score_margin", self.score_margin)
        self.source_task_id = _require_str("source_task_id", self.source_task_id)
        self.trajectory_id = _require_str("trajectory_id", self.trajectory_id)
        self.trajectory_seed = _require_non_negative_int("trajectory_seed", self.trajectory_seed)
        self.runtime_version = _require_str("runtime_version", self.runtime_version)
        self.evidence_mode = _require_optional_str("evidence_mode", self.evidence_mode)
        self.provenance = _require_mapping("provenance", self.provenance)

        for name, score in (
            ("normalized_score_chosen", self.normalized_score_chosen),
            ("normalized_score_rejected", self.normalized_score_rejected),
        ):
            if not 0.0 <= score <= 1.0:
                _fail(f"{name} must be in [0, 1].")
        if self.score_margin < 0:
            _fail("score_margin must be non-negative.")
        if self.raw_score_chosen < self.raw_score_rejected:
            _fail("raw_score_chosen must be >= raw_score_rejected.")
        if self.normalized_score_chosen < self.normalized_score_rejected:
            _fail("normalized_score_chosen must be >= normalized_score_rejected.")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreferencePair":
        payload = _require_mapping("PreferencePair", payload)
        return cls(
            pair_id=payload["pair_id"],
            context=SharedPrefixContext.from_dict(payload["context"]),
            chosen=CandidateBranch.from_dict(payload["chosen"]),
            rejected=CandidateBranch.from_dict(payload["rejected"]),
            task_type=payload["task_type"],
            difficulty_bin=payload["difficulty_bin"],
            decision_step=payload["decision_step"],
            raw_score_chosen=payload["raw_score_chosen"],
            raw_score_rejected=payload["raw_score_rejected"],
            normalized_score_chosen=payload["normalized_score_chosen"],
            normalized_score_rejected=payload["normalized_score_rejected"],
            score_margin=payload["score_margin"],
            source_task_id=payload["source_task_id"],
            trajectory_id=payload["trajectory_id"],
            trajectory_seed=payload["trajectory_seed"],
            runtime_version=payload.get("runtime_version", RUNTIME_SCHEMA_VERSION),
            evidence_mode=payload.get("evidence_mode"),
            provenance=payload.get("provenance", {}),
        )


__all__ = [
    "KNOWN_TOOL_NAMES",
    "RUNTIME_SCHEMA_VERSION",
    "ActorStep",
    "CandidateBranch",
    "ContinuationState",
    "EvidenceRecord",
    "EvidenceSourceType",
    "GeneGroup",
    "Interpretation",
    "LabelSource",
    "LocalScoreBreakdown",
    "MechanisticLabel",
    "PreferenceDifficulty",
    "PreferencePair",
    "RelationshipStatus",
    "SchemaValidationError",
    "SharedPrefixContext",
    "StructuredState",
    "TaskType",
    "TerminationReason",
    "ToolAction",
    "ToolObservation",
    "ToolObservationStatus",
    "TrajectoryTurn",
    "UserAnchors",
    "VerifierStep",
]
