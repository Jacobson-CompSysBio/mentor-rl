"""Simple validation helpers for runtime objects and tool calls.

These checks are meant to be easy to read and easy to reuse. They do not run
the tools. They only answer questions like:

- is this tool name allowed?
- do the arguments have the right shape?
- do the requested genes and layers exist?
- is this tool call a duplicate of an earlier one?

The result object is intentionally small so later pipeline stages can log
validation failures without having to catch exceptions for every case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .schemas import (
    KNOWN_TOOL_NAMES,
    CandidateBranch,
    PreferencePair,
    SchemaValidationError,
    SharedPrefixContext,
    StructuredState,
    ToolAction,
)


@dataclass
class ValidationResult:
    """Small container for validation success plus any error messages."""

    valid: bool
    errors: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True, errors=[])

    @classmethod
    def fail(cls, *errors: str) -> "ValidationResult":
        return cls(valid=False, errors=[error for error in errors if error])

    def add_error(self, error: str) -> None:
        self.valid = False
        self.errors.append(error)

    def extend(self, other: "ValidationResult") -> None:
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)


def _ensure_tool_action(tool_action: ToolAction) -> ValidationResult:
    if not isinstance(tool_action, ToolAction):
        return ValidationResult.fail("tool_action must be a ToolAction instance.")
    return ValidationResult.ok()


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_string_list(value: Any, *, allow_empty: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    if not allow_empty and not value:
        return False
    return all(_is_non_empty_string(item) for item in value)


_ALL_LAYER_ALIASES = {"all", "*", "all_layers", "all layers", "multiplex"}


def _is_all_layer_alias(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _ALL_LAYER_ALIASES


def _is_all_layer_list(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return not value or all(_is_all_layer_alias(item) for item in value)
    return _is_all_layer_alias(value)


def normalize_tool_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return canonical tool arguments for common all-layer spellings.

    Graph tools use an omitted layer field to mean "query all layers". Model
    outputs often spell that as `["all"]`, `[]`, or `null`; normalize those
    aliases before validation, duplicate detection, execution, and scoring.
    """

    normalized = dict(arguments)
    if tool_name in {"get_neighbors", "induce_subgraph"}:
        if "layers" in normalized and _is_all_layer_list(normalized["layers"]):
            normalized.pop("layers")
    elif tool_name == "shortest_path":
        if "layer" in normalized and _is_all_layer_alias(normalized["layer"]):
            normalized.pop("layer")
    return normalized


def normalize_tool_action(tool_action: ToolAction) -> ToolAction:
    """Return a tool action with canonicalized arguments."""

    if not isinstance(tool_action, ToolAction):
        raise SchemaValidationError("tool_action must be a ToolAction instance.")
    return ToolAction(
        tool_name=tool_action.tool_name,
        arguments=normalize_tool_arguments(tool_action.tool_name, tool_action.arguments),
        call_id=tool_action.call_id,
    )


def _reject_unknown_arguments(
    arguments: dict[str, Any],
    *,
    allowed: set[str],
) -> ValidationResult:
    result = ValidationResult.ok()
    for key in arguments:
        if key not in allowed:
            result.add_error(f"Unexpected argument for tool call: {key}.")
    return result


def validate_tool_name(tool_name: str) -> ValidationResult:
    """Check whether a tool name belongs to the supported runtime vocabulary."""

    if tool_name not in KNOWN_TOOL_NAMES:
        allowed = ", ".join(KNOWN_TOOL_NAMES)
        return ValidationResult.fail(f"Unknown tool name: {tool_name}. Allowed tools: {allowed}.")
    return ValidationResult.ok()


def validate_tool_action_schema(tool_action: ToolAction) -> ValidationResult:
    """Check that a tool call has the right argument names and basic types."""

    base = _ensure_tool_action(tool_action)
    if not base.valid:
        return base

    result = validate_tool_name(tool_action.tool_name)
    arguments = normalize_tool_arguments(tool_action.tool_name, tool_action.arguments)

    if tool_action.tool_name == "query_mygene":
        result.extend(_reject_unknown_arguments(arguments, allowed={"query", "fields"}))
        if not _is_non_empty_string(arguments.get("query")):
            result.add_error("query_mygene requires a non-empty string field named 'query'.")
        if "fields" in arguments and not _is_string_list(arguments["fields"], allow_empty=False):
            result.add_error("query_mygene 'fields' must be a non-empty list of strings.")

    elif tool_action.tool_name == "get_neighbors":
        result.extend(_reject_unknown_arguments(arguments, allowed={"gene", "layers"}))
        if not _is_non_empty_string(arguments.get("gene")):
            result.add_error("get_neighbors requires a non-empty string field named 'gene'.")
        if "layers" in arguments and not _is_string_list(arguments["layers"], allow_empty=False):
            result.add_error("get_neighbors 'layers' must be a non-empty list of strings.")

    elif tool_action.tool_name == "shortest_path":
        result.extend(_reject_unknown_arguments(arguments, allowed={"source", "target", "layer"}))
        if not _is_non_empty_string(arguments.get("source")):
            result.add_error("shortest_path requires a non-empty string field named 'source'.")
        if not _is_non_empty_string(arguments.get("target")):
            result.add_error("shortest_path requires a non-empty string field named 'target'.")
        if "layer" in arguments and not _is_non_empty_string(arguments["layer"]):
            result.add_error("shortest_path 'layer' must be a non-empty string when provided.")

    elif tool_action.tool_name == "rwr_multiplex":
        result.extend(_reject_unknown_arguments(arguments, allowed={"seeds", "top_k"}))
        if not _is_string_list(arguments.get("seeds"), allow_empty=False):
            result.add_error("rwr_multiplex requires a non-empty list field named 'seeds'.")
        if "top_k" in arguments and (not isinstance(arguments["top_k"], int) or arguments["top_k"] <= 0):
            result.add_error("rwr_multiplex 'top_k' must be a positive integer.")

    elif tool_action.tool_name == "rwr_monoplex":
        result.extend(_reject_unknown_arguments(arguments, allowed={"seeds", "layer", "top_k"}))
        if not _is_string_list(arguments.get("seeds"), allow_empty=False):
            result.add_error("rwr_monoplex requires a non-empty list field named 'seeds'.")
        if not _is_non_empty_string(arguments.get("layer")):
            result.add_error("rwr_monoplex requires a non-empty string field named 'layer'.")
        if "top_k" in arguments and (not isinstance(arguments["top_k"], int) or arguments["top_k"] <= 0):
            result.add_error("rwr_monoplex 'top_k' must be a positive integer.")

    elif tool_action.tool_name == "induce_subgraph":
        result.extend(_reject_unknown_arguments(arguments, allowed={"genes", "layers"}))
        if not _is_string_list(arguments.get("genes"), allow_empty=False):
            result.add_error("induce_subgraph requires a non-empty list field named 'genes'.")
        if "layers" in arguments and not _is_string_list(arguments["layers"], allow_empty=False):
            result.add_error("induce_subgraph 'layers' must be a non-empty list of strings.")

    return result


def validate_tool_action_semantics(
    tool_action: ToolAction,
    *,
    state: StructuredState | None = None,
    available_gene_ids: set[str] | None = None,
    available_layers: set[str] | None = None,
) -> ValidationResult:
    """Check that a tool call makes sense for the current runtime environment."""

    result = validate_tool_action_schema(tool_action)
    if not result.valid:
        return result

    if state is not None:
        if not isinstance(state, StructuredState):
            return ValidationResult.fail("state must be a StructuredState instance.")
        if state.continuation_state == "stop":
            result.add_error("Tool calls are not valid after the state has entered stop mode.")
        if state.remaining_budget <= 0:
            result.add_error("Tool calls are not valid when the remaining budget is 0.")

    arguments = normalize_tool_arguments(tool_action.tool_name, tool_action.arguments)

    gene_fields = []
    layer_fields = []
    if tool_action.tool_name == "get_neighbors":
        gene_fields.append(arguments["gene"])
        layer_fields.extend(arguments.get("layers", []))
    elif tool_action.tool_name == "shortest_path":
        gene_fields.extend([arguments["source"], arguments["target"]])
        if "layer" in arguments:
            layer_fields.append(arguments["layer"])
    elif tool_action.tool_name == "rwr_multiplex":
        gene_fields.extend(arguments["seeds"])
    elif tool_action.tool_name == "rwr_monoplex":
        gene_fields.extend(arguments["seeds"])
        layer_fields.append(arguments["layer"])
    elif tool_action.tool_name == "induce_subgraph":
        gene_fields.extend(arguments["genes"])
        layer_fields.extend(arguments.get("layers", []))

    if available_gene_ids is not None:
        missing_genes = sorted(gene for gene in gene_fields if gene not in available_gene_ids)
        if missing_genes:
            result.add_error(
                "Tool call references genes that are not present in the runtime graph: "
                + ", ".join(missing_genes)
                + "."
            )

    if available_layers is not None:
        invalid_layers = sorted(layer for layer in layer_fields if layer not in available_layers)
        if invalid_layers:
            result.add_error(
                "Tool call references unknown layers: " + ", ".join(invalid_layers) + "."
            )

    return result


def validate_tool_action(
    tool_action: ToolAction,
    *,
    state: StructuredState | None = None,
    available_gene_ids: set[str] | None = None,
    available_layers: set[str] | None = None,
) -> ValidationResult:
    """Run both schema checks and semantic checks for one tool call."""

    return validate_tool_action_semantics(
        tool_action,
        state=state,
        available_gene_ids=available_gene_ids,
        available_layers=available_layers,
    )


def tool_action_fingerprint(tool_action: ToolAction) -> str:
    """Return a stable fingerprint for duplicate-call detection."""

    if not isinstance(tool_action, ToolAction):
        raise SchemaValidationError("tool_action must be a ToolAction instance.")
    return json.dumps(
        {
            "tool_name": tool_action.tool_name,
            "arguments": normalize_tool_arguments(tool_action.tool_name, tool_action.arguments),
        },
        sort_keys=True,
    )


def is_duplicate_tool_action(
    tool_action: ToolAction,
    prior_actions: Iterable[ToolAction],
) -> bool:
    """Check whether a tool call exactly matches an earlier call."""

    fingerprint = tool_action_fingerprint(tool_action)
    return any(tool_action_fingerprint(prior) == fingerprint for prior in prior_actions)


def validate_shared_prefix_context(context: SharedPrefixContext) -> ValidationResult:
    """Check that a shared-prefix context object is well formed."""

    if not isinstance(context, SharedPrefixContext):
        return ValidationResult.fail("context must be a SharedPrefixContext instance.")
    try:
        context.to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult.fail(f"SharedPrefixContext failed to serialize: {exc}")
    return ValidationResult.ok()


def validate_candidate_branch(branch: CandidateBranch) -> ValidationResult:
    """Check that a branch is internally consistent."""

    if not isinstance(branch, CandidateBranch):
        return ValidationResult.fail("branch must be a CandidateBranch instance.")

    result = ValidationResult.ok()
    try:
        branch.to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        result.add_error(f"CandidateBranch failed to serialize: {exc}")
        return result

    tool_action = branch.actor_step.tool_action
    observation = branch.observation
    if tool_action is None and observation is not None:
        result.add_error("A branch cannot contain an observation without a tool action.")
    if tool_action is not None and observation is not None:
        if tool_action.call_id != observation.call_id:
            result.add_error("tool_action.call_id must match observation.call_id.")
        result.extend(validate_tool_action_schema(tool_action))

    return result


def validate_preference_pair(pair: PreferencePair) -> ValidationResult:
    """Check that a mined preference pair is structurally usable for DPO."""

    if not isinstance(pair, PreferencePair):
        return ValidationResult.fail("pair must be a PreferencePair instance.")

    result = validate_shared_prefix_context(pair.context)
    result.extend(validate_candidate_branch(pair.chosen))
    result.extend(validate_candidate_branch(pair.rejected))

    if pair.context.source_task_id and pair.context.source_task_id != pair.source_task_id:
        result.add_error("context.source_task_id must match pair.source_task_id when both are set.")

    return result


__all__ = [
    "ValidationResult",
    "is_duplicate_tool_action",
    "normalize_tool_action",
    "normalize_tool_arguments",
    "tool_action_fingerprint",
    "validate_candidate_branch",
    "validate_preference_pair",
    "validate_shared_prefix_context",
    "validate_tool_action",
    "validate_tool_action_schema",
    "validate_tool_action_semantics",
    "validate_tool_name",
]
