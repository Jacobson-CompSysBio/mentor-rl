"""Contract helpers for model-facing tool-curriculum examples.

The pre-trajectory curriculum must teach the tool surface that the runtime
actually accepts.  This module deliberately imports the runtime vocabulary and
validator instead of maintaining a second, hand-written schema.  It also keeps
machine-local execution details out of training text and records whether a
tool choice follows the cheapest-specific-tool policy.

The low-level ``rwr_hpc_app`` escape hatch is a live runtime tool, but it is not
a model-facing curriculum tool: it accepts command-line arguments and a local
working directory.  Curriculum examples should use the structured tools that
wrap it instead.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import (
    KNOWN_TOOL_NAMES,
    RUNTIME_SCHEMA_VERSION,
    SchemaValidationError,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
)
from .validators import normalize_tool_arguments, validate_tool_action


TOOL_CURRICULUM_CONTRACT_VERSION = "mentor-rl-tool-curriculum-v1"
CHEAPEST_SPECIFIC_POLICY_VERSION = "mentor-rl-cheapest-specific-tool-v1"
REDACTED_LOCAL_VALUE = "[redacted-local-provenance]"

# This is intentionally derived from the runtime vocabulary.  Fabricated names
# such as ``get_graph_schema`` and ``query_module_oracle`` can never enter the
# model-facing contract through this list.
_NON_CURRICULUM_RUNTIME_TOOLS = frozenset({"rwr_hpc_app"})
CURRICULUM_TOOL_NAMES = tuple(
    tool_name for tool_name in KNOWN_TOOL_NAMES if tool_name not in _NON_CURRICULUM_RUNTIME_TOOLS
)


class ToolCurriculumContractError(ValueError):
    """Raised when a tool-curriculum record violates the public contract."""


@dataclass(frozen=True)
class ToolIntentPolicy:
    """Preferred tool and cost metadata for one evidence-gathering intent."""

    intent: str
    preferred_tool: str
    cost_rank: int
    evidence_scope: str
    discouraged_alternatives: tuple[str, ...] = ()
    rationale: str = ""

    def metadata(self, *, selected_tool: str | None = None) -> dict[str, Any]:
        selected = selected_tool or self.preferred_tool
        return {
            "policy": "cheapest_specific_tool",
            "policy_version": CHEAPEST_SPECIFIC_POLICY_VERSION,
            "intent": self.intent,
            "selected_tool": selected,
            "preferred_tool": self.preferred_tool,
            "selected_is_preferred": selected == self.preferred_tool,
            "cost_rank": self.cost_rank,
            "evidence_scope": self.evidence_scope,
            "discouraged_alternatives": list(self.discouraged_alternatives),
            "rationale": self.rationale,
        }


def _policy(
    intent: str,
    preferred_tool: str,
    cost_rank: int,
    evidence_scope: str,
    *discouraged_alternatives: str,
    rationale: str,
) -> ToolIntentPolicy:
    if preferred_tool not in CURRICULUM_TOOL_NAMES:  # pragma: no cover - import-time guard
        raise RuntimeError(f"Policy {intent!r} names a non-curriculum tool: {preferred_tool}.")
    unknown_alternatives = set(discouraged_alternatives).difference(CURRICULUM_TOOL_NAMES)
    if unknown_alternatives:  # pragma: no cover - import-time guard
        raise RuntimeError(
            f"Policy {intent!r} names unknown alternatives: {sorted(unknown_alternatives)}."
        )
    return ToolIntentPolicy(
        intent=intent,
        preferred_tool=preferred_tool,
        cost_rank=cost_rank,
        evidence_scope=evidence_scope,
        discouraged_alternatives=tuple(discouraged_alternatives),
        rationale=rationale,
    )


# Cost rank is ordinal within this curriculum: 1 is a point lookup, 2 is a
# bounded local/subset query, 3 is a vector/global calculation, and 4 is an
# external annotation request.  The preferred tool should return the smallest
# result sufficient for the stated intent.
TOOL_INTENT_POLICIES: dict[str, ToolIntentPolicy] = {
    item.intent: item
    for item in (
        _policy(
            "gene_annotation",
            "query_mygene",
            4,
            "one gene annotation",
            rationale="Use the annotation lookup only when graph evidence cannot answer the question.",
        ),
        _policy(
            "gene_set_enrichment",
            "enrich_gene_set",
            4,
            "one bounded gene set",
            rationale="Request enrichment for the supplied set instead of exposing raw annotation data.",
        ),
        _policy(
            "direct_neighbors",
            "get_neighbors",
            1,
            "one gene in selected layers",
            "induce_subgraph",
            rationale="A direct-neighbor lookup is cheaper and more specific than a subgraph query.",
        ),
        _policy(
            "single_shortest_path",
            "shortest_path",
            2,
            "one source-target pair",
            "shortest_paths",
            rationale="Use the single-pair tool unless a many-to-many path set is required.",
        ),
        _policy(
            "batched_shortest_paths",
            "shortest_paths",
            3,
            "bounded source and target sets",
            rationale="The structured batch tool is appropriate only for multi-pair path evidence.",
        ),
        _policy(
            "induced_subgraph",
            "induce_subgraph",
            2,
            "one bounded gene set",
            rationale="Return only edges induced by the explicitly supplied genes.",
        ),
        _policy(
            "multiplex_ranking",
            "rwr",
            3,
            "one seed set across selected layers",
            "rwr_multiplex",
            rationale="Use the canonical structured RWR interface instead of its legacy alias.",
        ),
        _policy(
            "monoplex_ranking",
            "rwr",
            3,
            "one seed set in one layer",
            "rwr_monoplex",
            rationale="Use structured RWR with a layer argument instead of its legacy alias.",
        ),
        _policy(
            "query_filtered_ranking",
            "rwr_loe",
            3,
            "one seed set and an optional query set",
            "rwr",
            rationale="Use RWR-LOE when query filtering is part of the requested evidence.",
        ),
        _policy(
            "pair_rank",
            "get_rank",
            1,
            "one ordered gene pair",
            "rwr",
            "rwr_loe",
            rationale="A scalar rank lookup avoids materializing a full ranking vector.",
        ),
        _policy(
            "pair_distance",
            "get_distance",
            1,
            "one gene pair",
            "rwr",
            rationale="A scalar distance lookup avoids a full-vector response.",
        ),
        _policy(
            "pair_spearman",
            "get_spearman",
            1,
            "one gene pair",
            "get_distance",
            rationale="Request the named correlation directly when its semantics matter.",
        ),
        _policy(
            "pair_pearson",
            "get_pearson",
            1,
            "one gene pair",
            "get_distance",
            rationale="Request the named correlation directly when its semantics matter.",
        ),
        _policy(
            "pair_dot_similarity",
            "get_dot_similarity",
            1,
            "one gene pair",
            "get_distance",
            rationale="Request the named similarity directly when its semantics matter.",
        ),
        _policy(
            "rank_vector_summary",
            "get_rank_vector_summary",
            2,
            "one seed set summary",
            "rwr",
            rationale="Use a summary tool when exact full-vector values are not needed.",
        ),
        _policy(
            "encoding_summary",
            "get_encoding_summary",
            2,
            "one seed set summary",
            "rwr",
            rationale="Use the bounded encoding summary instead of returning a full ranking.",
        ),
        _policy(
            "gene_layer_membership",
            "get_gene_layers",
            1,
            "one gene",
            "get_layer_stats",
            rationale="Query membership for the gene rather than scanning layer statistics.",
        ),
        _policy(
            "nodes_by_layer_for_gene",
            "get_nodes_by_layer",
            1,
            "one gene",
            rationale="Use the runtime's structured per-gene layer-node view.",
        ),
        _policy(
            "layer_statistics",
            "get_layer_stats",
            2,
            "bounded layer summary",
            rationale="Use the aggregate layer summary rather than raw graph files.",
        ),
        _policy(
            "path_layer_counts",
            "get_path_layer_counts",
            2,
            "bounded source and target sets",
            "shortest_paths",
            rationale="Use the layer-count projection when individual paths are not required.",
        ),
        _policy(
            "component_summary",
            "get_component_summary",
            2,
            "one bounded gene set",
            "induce_subgraph",
            rationale="Use the component summary when the edge list itself is unnecessary.",
        ),
        _policy(
            "seed_essentiality",
            "get_seed_essentiality",
            3,
            "one seed set",
            "rwr",
            rationale="Use the dedicated perturbation statistic instead of manual repeated RWR calls.",
        ),
        _policy(
            "layer_ablation",
            "get_layer_ablation",
            3,
            "one seed set across layers",
            "rwr",
            rationale="Use the dedicated layer-ablation summary instead of manual repeated RWR calls.",
        ),
        _policy(
            "node_perturbation",
            "get_node_perturbation",
            3,
            "one seed set and bounded perturbation set",
            "rwr",
            rationale="Use the dedicated perturbation summary instead of manual repeated RWR calls.",
        ),
    )
}


_SENSITIVE_KEYS = frozenset(
    {
        "argv",
        "binary_path",
        "build_dir",
        "cache_dir",
        "cache_path",
        "command",
        "command_line",
        "cwd",
        "executable",
        "flist",
        "flist_path",
        "home",
        "host",
        "hostname",
        "job_id",
        "manifest_path",
        "output_dir",
        "output_path",
        "pbs_job_id",
        "pid",
        "repo_root",
        "raw_stderr",
        "raw_stdout",
        "scratch_dir",
        "slurm_job_id",
        "source_path",
        "stderr",
        "stdout",
        "store_path",
        "user",
        "username",
        "working_directory",
    }
)

_PUBLIC_PROVENANCE_KEYS = frozenset(
    {
        "active_layers",
        "algorithm",
        "app_returncode",
        "backend",
        "cache_hit",
        "distance_type",
        "flist_id",
        "graph_version",
        "implementation",
        "layer_count",
        "layer_name",
        "multiplex_id",
        "network_flist_sha256",
        "network_used",
        "queried_layers",
        "restart_probability",
        "runtime_schema_version",
        "schema_version",
        "search_mode",
        "source",
        "store_id",
        "tool_contract_version",
        "tool_name",
    }
)

_FILE_URI_RE = re.compile(r"file://[^\s\"'`]+", re.IGNORECASE)
_WINDOWS_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])[^\s\"'`]+")
_POSIX_ABSOLUTE_RE = re.compile(r"(?<![A-Za-z0-9:/])/(?:[^\s\"'`,;]+)")
_LOCAL_RELATIVE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:\.\.?[/\\]|data[/\\]|runtime[/\\]|scripts[/\\]|checkpoints[/\\]|"
    r"outputs?[/\\]|logs?[/\\]|cache[/\\])"
    r"[^\s\"'`,;]*",
    re.IGNORECASE,
)


def _sanitize_string(value: str) -> str:
    sanitized = _FILE_URI_RE.sub(REDACTED_LOCAL_VALUE, value)
    sanitized = _WINDOWS_ABSOLUTE_RE.sub(REDACTED_LOCAL_VALUE, sanitized)
    sanitized = _POSIX_ABSOLUTE_RE.sub(REDACTED_LOCAL_VALUE, sanitized)
    sanitized = _LOCAL_RELATIVE_RE.sub(REDACTED_LOCAL_VALUE, sanitized)
    return sanitized


def sanitize_tool_payload(value: Any) -> Any:
    """Return JSON-like tool data with machine-local details removed.

    Explicit execution/logging fields are omitted.  Absolute paths and common
    repository-relative paths embedded in otherwise useful strings are replaced
    with a stable marker.  Biological ``path_gene_ids`` fields are preserved.
    """

    if isinstance(value, Mapping):
        return {
            str(key): sanitize_tool_payload(item)
            for key, item in value.items()
            if str(key).lower() not in _SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [sanitize_tool_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_tool_payload(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def sanitize_public_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    """Allowlist stable public provenance and remove execution-host details."""

    clean = {
        str(key): sanitize_tool_payload(value)
        for key, value in (provenance or {}).items()
        if str(key) in _PUBLIC_PROVENANCE_KEYS
    }
    clean["runtime_schema_version"] = RUNTIME_SCHEMA_VERSION
    clean["tool_contract_version"] = TOOL_CURRICULUM_CONTRACT_VERSION
    return clean


def find_provenance_leaks(value: Any, *, _location: str = "$") -> list[str]:
    """Return locations of path, host, command, or scheduler details."""

    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            location = f"{_location}.{key_text}"
            if key_text.lower() in _SENSITIVE_KEYS:
                leaks.append(location)
            leaks.extend(find_provenance_leaks(item, _location=location))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            leaks.extend(find_provenance_leaks(item, _location=f"{_location}[{index}]"))
    elif isinstance(value, str) and _sanitize_string(value) != value:
        leaks.append(_location)
    return leaks


def assert_no_provenance_leakage(value: Any) -> None:
    """Raise when a curriculum-visible value contains machine-local details."""

    leaks = find_provenance_leaks(value)
    if leaks:
        raise ToolCurriculumContractError(
            "Curriculum-visible tool data contains local provenance at: " + ", ".join(leaks)
        )


def is_curriculum_tool(tool_name: str) -> bool:
    """Return whether ``tool_name`` is a live, model-facing runtime tool."""

    return tool_name in CURRICULUM_TOOL_NAMES


def select_tool_for_intent(intent: str) -> str:
    """Return the cheapest specific live tool for a declared intent."""

    try:
        return TOOL_INTENT_POLICIES[intent].preferred_tool
    except KeyError as exc:
        raise ToolCurriculumContractError(f"Unknown tool intent: {intent}.") from exc


def tool_policy_metadata(
    intent: str,
    *,
    selected_tool: str | None = None,
    require_preferred: bool = True,
) -> dict[str, Any]:
    """Build auditable cheapest-specific-tool policy metadata."""

    try:
        policy = TOOL_INTENT_POLICIES[intent]
    except KeyError as exc:
        raise ToolCurriculumContractError(f"Unknown tool intent: {intent}.") from exc
    selected = selected_tool or policy.preferred_tool
    if selected not in CURRICULUM_TOOL_NAMES:
        raise ToolCurriculumContractError(f"Tool is not model-facing and live: {selected}.")
    if require_preferred and selected != policy.preferred_tool:
        raise ToolCurriculumContractError(
            f"Intent {intent!r} requires cheapest specific tool {policy.preferred_tool!r}; "
            f"got {selected!r}."
        )
    return policy.metadata(selected_tool=selected)


def _default_call_id(tool_name: str, arguments: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "curriculum_" + hashlib.sha256(encoded).hexdigest()[:20]


def build_tool_action(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    call_id: str | None = None,
    available_gene_ids: set[str] | None = None,
    available_layers: set[str] | None = None,
) -> ToolAction:
    """Construct and validate one live model-facing tool action."""

    if tool_name not in CURRICULUM_TOOL_NAMES:
        raise ToolCurriculumContractError(f"Tool is not model-facing and live: {tool_name}.")
    if not isinstance(arguments, Mapping):
        raise ToolCurriculumContractError("Tool arguments must be a mapping.")
    raw_arguments = dict(arguments)
    assert_no_provenance_leakage(raw_arguments)
    normalized_arguments = normalize_tool_arguments(tool_name, raw_arguments)
    action = ToolAction(
        tool_name=tool_name,
        arguments=normalized_arguments,
        call_id=call_id or _default_call_id(tool_name, normalized_arguments),
    )
    result = validate_tool_action(
        action,
        available_gene_ids=available_gene_ids,
        available_layers=available_layers,
    )
    if not result.valid:
        raise ToolCurriculumContractError("Invalid tool action: " + "; ".join(result.errors))
    return action


def validate_tool_observation(
    observation: ToolObservation | Mapping[str, Any],
    *,
    action: ToolAction | None = None,
    require_sanitized: bool = True,
) -> ToolObservation:
    """Validate a supplied observation and its linkage to an optional action."""

    try:
        parsed = (
            observation
            if isinstance(observation, ToolObservation)
            else ToolObservation.from_dict(dict(observation))
        )
    except (KeyError, TypeError, SchemaValidationError, ValueError) as exc:
        raise ToolCurriculumContractError(f"Invalid tool observation: {exc}") from exc

    if action is not None:
        if parsed.call_id != action.call_id:
            raise ToolCurriculumContractError(
                f"Observation call_id {parsed.call_id!r} does not match action {action.call_id!r}."
            )
        provenance_tool = parsed.provenance.get("tool_name")
        if provenance_tool is not None and provenance_tool != action.tool_name:
            raise ToolCurriculumContractError(
                f"Observation provenance names {provenance_tool!r}, expected {action.tool_name!r}."
            )
        if parsed.payload is not None:
            payload_tool = parsed.payload.get("tool_name")
            if payload_tool is not None and payload_tool != action.tool_name:
                raise ToolCurriculumContractError(
                    f"Observation payload names {payload_tool!r}, expected {action.tool_name!r}."
                )
    if require_sanitized:
        assert_no_provenance_leakage(parsed.to_dict())
    return parsed


def build_tool_observation(
    action: ToolAction,
    *,
    payload: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    status: ToolObservationStatus | str = ToolObservationStatus.SUCCESS,
    error: str | None = None,
) -> ToolObservation:
    """Build a schema-valid, public observation from runtime-like values."""

    if action.tool_name not in CURRICULUM_TOOL_NAMES:
        raise ToolCurriculumContractError(f"Tool is not model-facing and live: {action.tool_name}.")
    try:
        normalized_status = ToolObservationStatus(status)
    except ValueError as exc:
        raise ToolCurriculumContractError(f"Unknown observation status: {status}.") from exc

    clean_payload = None if payload is None else sanitize_tool_payload(dict(payload))
    clean_provenance = sanitize_public_provenance(provenance)
    clean_provenance["tool_name"] = action.tool_name
    clean_error = None if error is None else _sanitize_string(error)
    try:
        observation = ToolObservation(
            status=normalized_status,
            provenance=clean_provenance,
            call_id=action.call_id,
            payload=clean_payload,
            error=clean_error,
        )
    except SchemaValidationError as exc:
        raise ToolCurriculumContractError(f"Invalid tool observation: {exc}") from exc
    return validate_tool_observation(observation, action=action, require_sanitized=True)


def build_tool_exchange(
    intent: str,
    arguments: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    call_id: str | None = None,
    available_gene_ids: set[str] | None = None,
    available_layers: set[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready action/observation/policy curriculum context."""

    tool_name = select_tool_for_intent(intent)
    action = build_tool_action(
        tool_name,
        arguments,
        call_id=call_id,
        available_gene_ids=available_gene_ids,
        available_layers=available_layers,
    )
    observation = build_tool_observation(
        action,
        payload=payload,
        provenance=provenance,
    )
    exchange = {
        "tool_action": action.to_dict(),
        "tool_observation": observation.to_dict(),
        "tool_policy": tool_policy_metadata(intent, selected_tool=tool_name),
    }
    assert_no_provenance_leakage(exchange)
    return exchange


__all__ = [
    "CHEAPEST_SPECIFIC_POLICY_VERSION",
    "CURRICULUM_TOOL_NAMES",
    "REDACTED_LOCAL_VALUE",
    "TOOL_CURRICULUM_CONTRACT_VERSION",
    "TOOL_INTENT_POLICIES",
    "ToolCurriculumContractError",
    "ToolIntentPolicy",
    "assert_no_provenance_leakage",
    "build_tool_action",
    "build_tool_exchange",
    "build_tool_observation",
    "find_provenance_leaks",
    "is_curriculum_tool",
    "sanitize_public_provenance",
    "sanitize_tool_payload",
    "select_tool_for_intent",
    "tool_policy_metadata",
    "validate_tool_observation",
]
