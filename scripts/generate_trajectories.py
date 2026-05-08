#!/usr/bin/env python3
"""Generate shared-prefix trajectories from canonical CORUM tasks.

This script is the first end-to-end trajectory generator for the DPO pipeline.
It uses the existing runtime, state, validation, and scoring layers to:

1. load canonical CORUM tasks
2. initialize the visible runtime state
3. build deterministic actor candidates
4. execute tool calls in the runtime
5. build deterministic verifier updates
6. score every branch
7. select one branch per step and log the whole branch pool

The generator now supports two candidate sources:

- `heuristic` for tests and small local debugging
- `model_vllm` for real model-backed generation through an OpenAI-compatible
  vLLM endpoint such as a served `gpt-oss-120b` runtime

Both paths write the same branch-pool and trajectory artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import (
    ActorStep,
    CandidateBranch,
    ContinuationState,
    EvidenceRecord,
    EvidenceSourceType,
    GeneGroup,
    Interpretation,
    LabelSource,
    LocalScoreBreakdown,
    MechanisticLabel,
    PreferenceDifficulty,
    PreferencePair,
    RelationshipStatus,
    RuntimeEnvironment,
    SharedPrefixContext,
    TaskType,
    TerminationReason,
    ToolAction,
    ToolObservation,
    ToolObservationStatus,
    TrajectoryTurn,
    VerifierStep,
    append_evidence_record,
    clone_interpretation,
    clone_state,
    decrement_budget,
    initialize_state_from_corum_task,
    normalize_tool_arguments,
    record_tool_call,
    replace_mechanistic_labels,
    replace_predicted_groups,
    score_candidate_branch,
    score_terminal_trajectory,
    set_continuation_state,
    validate_candidate_branch,
    validate_tool_action_semantics,
)


DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "corum_corpus" / "tasks.train.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "corum_trajectories"
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "humannet_multiplex_store"
DEFAULT_PROGRESS_FILENAME = "progress.json"
DEFAULT_GENERATOR_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_GENERATOR_API_KEY_ENV = "OPENAI_API_KEY"
STRUCTURED_OUTPUT_MAX_TOKENS = 2048
DEFAULT_ACTOR_RATIONALE_MAX_TOKENS = 2048
DEFAULT_PREFERENCE_PAIR_MARGIN = 0.10
DEFAULT_SELECTION_SCORE_EPSILON = 0.02
DEFAULT_RECOVERY_RWR_TOP_K = 50
GPT_OSS_FINAL_CHANNEL_PREFIX = "<|start|>assistant<|channel|>final<|message|>"
PROMPT_TEXT_MAX_CHARS = 700
PROMPT_ACTOR_REASONING_MAX_CHARS = 1200
PROMPT_LIST_PREVIEW_LIMIT = 20
PROMPT_RWR_RESULT_PREVIEW_LIMIT = 30
PROMPT_RWR_NON_SEED_PREVIEW_LIMIT = 25
PROMPT_LAYER_PREVIEW_LIMIT = 12
PROMPT_EDGE_PREVIEW_LIMIT = 32
PROMPT_MYGENE_PREVIEW_LIMIT = 5
PROMPT_TOOL_REFERENCE_GENE_LIMIT = 40
PROMPT_TOOL_REFERENCE_LAYER_LIMIT = 40
ACTOR_SAMPLING_STRATEGIES = ("batch", "verbalized")
SELECTION_POLICIES = ("score", "task_quality")
PAIR_MINING_STRATEGIES = ("score_margin", "quality_balanced")
ACTOR_DIVERSITY_DIRECTIVES = (
    {
        "name": "best_direct_decision",
        "instruction": (
            "Choose the best direct next action. If visible evidence is already enough, "
            "prefer no tool call and explain the decision."
        ),
        "preferred_tools": [],
    },
    {
        "name": "subgraph_coherence_probe",
        "instruction": (
            "Explore candidate-group coherence. Prefer induce_subgraph over the current "
            "candidate gene set when at least two valid genes are available."
        ),
        "preferred_tools": ["induce_subgraph"],
    },
    {
        "name": "pair_connectivity_probe",
        "instruction": (
            "Explore pairwise connectivity. Prefer shortest_path between two informative "
            "valid genes when at least two valid genes are available."
        ),
        "preferred_tools": ["shortest_path"],
    },
    {
        "name": "neighborhood_or_expansion_probe",
        "instruction": (
            "Explore local neighborhood or expansion evidence. Prefer get_neighbors for "
            "explanation/none checks, and rwr_multiplex for recovery/refinement expansion."
        ),
        "preferred_tools": ["get_neighbors", "rwr_multiplex"],
    },
)
TASK_ACTOR_DIVERSITY_DIRECTIVES = {
    "recovery": (
        {
            "name": "recovery_rwr_expansion",
            "instruction": (
                "Explore recovery expansion. Prefer rwr_multiplex from the current "
                "seed/candidate group with top_k at least 50 to identify plausible "
                "missing complex members beyond the seeds."
            ),
            "preferred_tools": ["rwr_multiplex"],
        },
        {
            "name": "recovery_neighbor_expansion",
            "instruction": (
                "Explore local expansion evidence. Prefer get_neighbors on one informative "
                "anchor gene and look for candidate genes that could extend the group."
            ),
            "preferred_tools": ["get_neighbors"],
        },
        {
            "name": "recovery_subgraph_validation",
            "instruction": (
                "Validate whether the current candidate group is coherent before stopping. "
                "Prefer induce_subgraph when at least two valid genes are available."
            ),
            "preferred_tools": ["induce_subgraph"],
        },
        {
            "name": "recovery_pair_connectivity",
            "instruction": (
                "Check pairwise graph support among informative candidate genes. Prefer "
                "shortest_path when at least two valid genes are available."
            ),
            "preferred_tools": ["shortest_path"],
        },
        {
            "name": "recovery_direct_decision",
            "instruction": (
                "Make the best direct recovery decision only if visible evidence is already "
                "sufficient; otherwise prefer a valid expansion or validation tool."
            ),
            "preferred_tools": [],
        },
    ),
    "refinement": (
        {
            "name": "refinement_subgraph_pruning",
            "instruction": (
                "Explore refinement evidence. Prefer induce_subgraph to identify which "
                "members of the current candidate group are graph-supported."
            ),
            "preferred_tools": ["induce_subgraph"],
        },
        {
            "name": "refinement_rwr_support",
            "instruction": (
                "Use restart-walk support to distinguish coherent members from weaker "
                "ones. Prefer rwr_multiplex from the current candidate group."
            ),
            "preferred_tools": ["rwr_multiplex"],
        },
        {
            "name": "refinement_pair_connectivity",
            "instruction": (
                "Probe whether questionable gene pairs are connected. Prefer shortest_path "
                "between two informative valid genes."
            ),
            "preferred_tools": ["shortest_path"],
        },
        {
            "name": "refinement_direct_decision",
            "instruction": (
                "Make the best direct refinement decision only if visible evidence already "
                "supports a coherent subset."
            ),
            "preferred_tools": [],
        },
    ),
    "none": (
        {
            "name": "none_subgraph_disconfirmation",
            "instruction": (
                "Look for disconfirming graph evidence before abstaining. Prefer "
                "induce_subgraph over the current candidate group."
            ),
            "preferred_tools": ["induce_subgraph"],
        },
        {
            "name": "none_pair_disconfirmation",
            "instruction": (
                "Test whether seed genes are connected. Prefer shortest_path when at least "
                "two valid genes are available."
            ),
            "preferred_tools": ["shortest_path"],
        },
        {
            "name": "none_neighbor_check",
            "instruction": (
                "Inspect one seed neighborhood to see whether any local support exists. "
                "Prefer get_neighbors on one valid anchor gene."
            ),
            "preferred_tools": ["get_neighbors"],
        },
        {
            "name": "none_abstain_if_supported",
            "instruction": (
                "Abstain only when the visible and observed evidence supports no single "
                "shared mechanism."
            ),
            "preferred_tools": [],
        },
    ),
    "explanation": (
        {
            "name": "explanation_annotation_decision",
            "instruction": (
                "Use visible annotations and context to decide the strongest shared "
                "mechanism when they are already sufficient."
            ),
            "preferred_tools": [],
        },
        {
            "name": "explanation_subgraph_validation",
            "instruction": (
                "Validate candidate-group coherence. Prefer induce_subgraph when at least "
                "two valid genes are available."
            ),
            "preferred_tools": ["induce_subgraph"],
        },
        {
            "name": "explanation_pair_connectivity",
            "instruction": (
                "Probe pairwise connectivity when graph evidence would clarify the "
                "mechanism. Prefer shortest_path."
            ),
            "preferred_tools": ["shortest_path"],
        },
        {
            "name": "explanation_neighbor_context",
            "instruction": (
                "Inspect one anchor neighborhood when local graph context could strengthen "
                "or weaken the explanation. Prefer get_neighbors."
            ),
            "preferred_tools": ["get_neighbors"],
        },
    ),
}
TOOL_COVERAGE_DIRECTIVES = {
    "recovery": {
        "name": "tool_coverage_recovery_expansion",
        "instruction": (
            "This retry exists because no usable tool-backed actor candidate was observed. "
            "Choose a valid runtime tool if any valid graph argument can be formed; prefer "
            "rwr_multiplex with top_k at least 50 for recovery expansion, then "
            "get_neighbors or induce_subgraph."
        ),
        "preferred_tools": ["rwr_multiplex", "get_neighbors", "induce_subgraph"],
    },
    "refinement": {
        "name": "tool_coverage_refinement_probe",
        "instruction": (
            "This retry exists because no usable tool-backed actor candidate was observed. "
            "Choose a valid runtime tool if any valid graph argument can be formed; prefer "
            "induce_subgraph for pruning evidence, then rwr_multiplex or shortest_path."
        ),
        "preferred_tools": ["induce_subgraph", "rwr_multiplex", "shortest_path"],
    },
}
PAIR_CATEGORY_PRIORITIES = {
    "recovery_expansion": 0,
    "tool_supported_improvement": 1,
    "refinement_precision": 2,
    "none_abstention": 3,
    "score_margin": 4,
    "mechanism_label_only": 5,
    "conservative_stop": 6,
}
TRAJECTORY_STAGES = (
    ("load_tasks", "Load canonical CORUM tasks"),
    ("initialize_runtime", "Initialize deterministic runtime"),
    ("generate_trajectories", "Generate shared-prefix trajectories"),
    ("write_manifest", "Write run manifest"),
)

ACTOR_SYSTEM_PROMPT = """You are the actor policy for MENTOR-RL.
Your job is to pick the single best next action using only the visible task inputs,
the current interpretation, and the current visible state.

Task types:
- explanation: decide the strongest shared mechanism already supported by the seed set
- recovery: the seed set is incomplete; expand to a coherent related module
- refinement: the seed set may contain unrelated genes; isolate the coherent subset
- none: decide whether the set fails to support one shared mechanism

Evidence modes:
- minimal: only seed genes are visible
- graph: seed genes plus a graph query specification are visible
- contextual: seed genes plus context text are visible
- full: seed genes, graph query specification, context text, and structured annotations are visible

Actor rules:
- Write a concise, visible ReAct-style reasoning step.
- Choose at most one next runtime tool call.
- Use only visible evidence and deterministic runtime observations.
- Never assume access to hidden targets or labels that were not shown.
- Use canonical Ensembl gene ids when you reference genes in tool arguments.
- For graph tool arguments, choose gene ids from the provided
  tool_argument_reference candidate_gene_ids or from exact ids returned by a
  previous successful tool observation. Do not invent new ENSG ids.
- Prefer the cheapest action that is most likely to reduce uncertainty.
- If current visible evidence is already enough, do not call a tool.
- To query all graph layers, omit the `layers` or `layer` argument entirely.
  Never write "all", [], or null for layer selection.
- For shortest_path, source and target must each be one string id, never a
  list, tuple, comma-separated string, or missing value.
- Do not update relationship status, predicted groups, mechanistic labels, or
  other structured state fields. The verifier owns that structured update.

Tool guidance:
- query_mygene: look up identifiers or metadata for one gene or alias string
- get_neighbors: inspect one seed gene's neighborhood
- shortest_path: test whether two genes are closely connected
- induce_subgraph: inspect coherence inside a candidate group
- rwr_monoplex: rank candidates on one named layer
- rwr_multiplex: rank candidates across the multiplex; prefer for recovery or refinement

Allowed tools:
- query_mygene: {"query": str, "fields": [str] optional}
- get_neighbors: {"gene": str, "layers": [real layer name] optional; omit for all layers}
- shortest_path: {"source": str, "target": str, "layer": real layer name optional; omit for all layers}
- rwr_multiplex: {"seeds": [str], "top_k": int optional}
- rwr_monoplex: {"seeds": [str], "layer": real layer name required, "top_k": int optional}
- induce_subgraph: {"genes": [str], "layers": [real layer name] optional; omit for all layers}

If the serving backend supports tool calls, use a native tool call for the
action. If it does not, write the reasoning normally and optionally end with one
machine-readable line:
TOOL_ACTION: {"tool_name": "...", "arguments": {...}}
"""

ACTOR_RATIONALE_SYSTEM_PROMPT = """You are writing the visible actor rationale for the next MENTOR-RL step before the tool is chosen.
Use only the visible task inputs, current interpretation, and current visible state.
Explain what uncertainty matters most right now and what kind of next action would best reduce it.
Do not mention hidden chain-of-thought, invisible labels, or tool results that have not been observed yet.
If the visible evidence is already enough, explain why no further tool call is needed.
Write 2 to 5 grounded sentences.
Return exactly one JSON object that matches the provided schema.
"""

VERIFIER_SYSTEM_PROMPT = """You are the verifier policy for MENTOR-RL.
Your job is to update the current interpretation and visible hypothesis state after
the actor step and the deterministic observation.

Verifier rules:
- Return exactly one JSON object and nothing else.
- Use only the visible task inputs, prior interpretation, prior state, actor output,
  and deterministic observation.
- Never assume access to hidden targets or labels that were not shown.
- Use canonical Ensembl gene ids when you reference genes.
- Keep the update conservative and grounded in the visible evidence.

Relationship status meanings:
- unknown: evidence is still too weak to classify the group
- partially_observed_group: there appears to be one coherent group, but it is still incomplete or uncertain
- validated_group: there is enough evidence for one coherent shared group
- multiple_groups: the evidence suggests more than one unrelated module
- insufficient_support: the evidence does not support one coherent mechanism

Continuation decision meanings:
- continue: one more step could improve the hypothesis
- revise: the last action was invalid, noisy, or unhelpful; try a different direction
- stop: current evidence is sufficient for the best current conclusion

Label source meanings:
- go: Gene Ontology label
- fcgs: FCGS label
- complex_name: complex-name-derived label
- free_text: grounded free-text label
- other: any other grounded label source

State update guidance:
- predicted_gene_ids should contain the best current coherent group.
- For explanation, the predicted group often matches the visible seed set.
- For recovery, add genes only when the observation supports them.
  When an RWR observation is available, explicitly evaluate the top non-seed
  candidates before declaring the seed group complete. Add only candidates that
  have credible visible support; otherwise explain why the non-seed candidates
  are too weak and continue if another check could help.
- For refinement, remove genes that look unsupported or off-module.
- For none tasks, prefer "insufficient_support" or "multiple_groups" when one coherent mechanism is not supported.
- If relationship_status is insufficient_support, an empty predicted_gene_ids list is acceptable.
- Prefer GO or FCGS labels when visible annotations support them.

Output schema:
{
  "updated_interpretation": {
    "mechanistic_claim": str,
    "main_evidence": str,
    "uncertainty": str,
    "next_subgoal": str
  },
  "updated_state": {
    "relationship_status": str,
    "predicted_gene_ids": [str],
    "mechanistic_labels": [
      {"label_source": str, "label_name": str, "label_id": str or null}
    ],
    "continuation_decision": str,
    "verifier_notes": str
  }
}
"""

ACTOR_OUTPUT_TOOL_NAME = "emit_actor_step"
VERIFIER_OUTPUT_TOOL_NAME = "emit_verifier_update"
RUNTIME_TOOL_NAMES = (
    "query_mygene",
    "get_neighbors",
    "shortest_path",
    "rwr_multiplex",
    "rwr_monoplex",
    "induce_subgraph",
)
TOOL_ACTION_LINE_RE = re.compile(
    r"(?im)^\s*(?:TOOL_ACTION|ACTION)\s*:\s*(?P<payload>\{.*\})\s*$"
)
RAW_GENERATION_PAYLOAD_KEYS = {"token_ids", "prompt_token_ids"}


def utc_now_iso() -> str:
    """Return the current UTC time in a JSON-friendly ISO format."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from a model response."""

    if not isinstance(text, str):
        raise ValueError("Model response text must be a string.")

    fence_start = text.find("```json")
    if fence_start != -1:
        fence_end = text.find("```", fence_start + 7)
        if fence_end != -1:
            text = text[fence_start + 7 : fence_end].strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("Model response did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Model response contained an incomplete JSON object.")


def _parse_model_json(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response."""

    return json.loads(_extract_first_json_object(text))


def _safe_list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            items.append(item)
    return _unique(items)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _strip_raw_generation_payload(value: Any) -> Any:
    """Drop endpoint token payloads before retaining model response diagnostics."""

    if isinstance(value, dict):
        return {
            key: _strip_raw_generation_payload(item)
            for key, item in value.items()
            if key not in RAW_GENERATION_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_strip_raw_generation_payload(item) for item in value]
    return value


def _truncate_prompt_text(value: Any, *, max_chars: int = PROMPT_TEXT_MAX_CHARS) -> str:
    text = _safe_text(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _preview_list(values: Any, *, limit: int = PROMPT_LIST_PREVIEW_LIMIT) -> list[Any]:
    if not isinstance(values, list):
        return []
    return values[:limit]


def _rwr_result_gene_id(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    gene_id = result.get("gene_id")
    return gene_id if isinstance(gene_id, str) and gene_id else None


def _compact_layer_list_payload(
    values: Any,
    *,
    count_key: str,
    sample_key: str,
    limit: int = PROMPT_LAYER_PREVIEW_LIMIT,
) -> dict[str, Any]:
    if not isinstance(values, list):
        return {count_key: 0, sample_key: []}
    return {count_key: len(values), sample_key: values[:limit]}


def _compact_provenance_for_prompt(provenance: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in provenance.items():
        if key in {"queried_layers", "active_layers"}:
            compact.update(
                _compact_layer_list_payload(
                    value,
                    count_key=f"{key}_count",
                    sample_key=f"{key}_sample",
                )
            )
        elif isinstance(value, str):
            compact[key] = _truncate_prompt_text(value, max_chars=240)
        elif isinstance(value, list):
            compact[f"{key}_count"] = len(value)
            compact[f"{key}_sample"] = value[:PROMPT_LIST_PREVIEW_LIMIT]
        else:
            compact[key] = value
    return compact


def _state_payload_for_model_prompt(state: Any) -> dict[str, Any]:
    state_payload = state.to_dict()
    state_payload.pop("user_anchors", None)

    for group in state_payload.get("predicted_groups", []):
        if isinstance(group, dict):
            group["rationale"] = _truncate_prompt_text(group.get("rationale"), max_chars=280)

    for record in state_payload.get("evidence_log", []):
        if not isinstance(record, dict):
            continue
        record["summary"] = _truncate_prompt_text(record.get("summary"), max_chars=280)
        provenance = record.get("provenance")
        if isinstance(provenance, dict):
            record["provenance"] = _compact_provenance_for_prompt(provenance)

    return state_payload


def _interpretation_payload_for_model_prompt(interpretation: Interpretation) -> dict[str, Any]:
    payload = interpretation.to_dict()
    for key in ("mechanistic_claim", "main_evidence", "uncertainty", "next_subgoal"):
        payload[key] = _truncate_prompt_text(payload.get(key))
    return payload


def _append_unique_strings(target: list[str], values: Iterable[Any]) -> None:
    seen = set(target)
    for value in values:
        if isinstance(value, str) and value and value not in seen:
            target.append(value)
            seen.add(value)


def _candidate_gene_ids_for_tool_reference(context: SharedPrefixContext) -> list[str]:
    gene_ids: list[str] = []
    visible_inputs = context.user_evidence if isinstance(context.user_evidence, dict) else {}
    _append_unique_strings(gene_ids, _safe_list_of_strings(visible_inputs.get("seed_gene_ids")))
    _append_unique_strings(gene_ids, _flatten_predicted_gene_ids(context.state))

    for evidence_record in context.state.evidence_log:
        _append_unique_strings(gene_ids, evidence_record.supporting_gene_ids)
    return gene_ids


def _tool_argument_reference_payload(
    context: SharedPrefixContext,
    *,
    environment: RuntimeEnvironment | None = None,
) -> dict[str, Any]:
    candidate_gene_ids = _candidate_gene_ids_for_tool_reference(context)
    if environment is not None:
        graph_candidate_gene_ids = [
            gene_id for gene_id in candidate_gene_ids if gene_id in environment.available_gene_ids
        ]
        unavailable_candidate_gene_ids = [
            gene_id for gene_id in candidate_gene_ids if gene_id not in environment.available_gene_ids
        ]
        available_layers = sorted(environment.available_layers)
    else:
        graph_candidate_gene_ids = candidate_gene_ids
        unavailable_candidate_gene_ids = []
        available_layers = []

    return {
        "rules": [
            "Graph tools require canonical Ensembl gene id strings, not symbols.",
            "Use candidate_gene_ids for gene/source/target/seeds/genes unless a prior successful tool observation returned another exact id.",
            "Use query_mygene first when only a gene symbol, alias, or non-Ensembl id is available.",
            "For all graph layers, omit layer/layers entirely; do not pass null, [], 'all', or '*' values.",
            "shortest_path source and target must each be one non-empty string id.",
            "get_neighbors gene must be one non-empty string id.",
            "induce_subgraph genes and RWR seeds must be non-empty arrays of string ids.",
        ],
        "argument_shapes": {
            "query_mygene": {"query": "string", "fields": "optional non-empty string array"},
            "get_neighbors": {"gene": "string", "layers": "optional non-empty layer-name array"},
            "shortest_path": {
                "source": "string",
                "target": "string",
                "layer": "optional layer-name string",
            },
            "rwr_multiplex": {"seeds": "non-empty string array", "top_k": "optional positive integer"},
            "rwr_monoplex": {
                "seeds": "non-empty string array",
                "layer": "required layer-name string",
                "top_k": "optional positive integer",
            },
            "induce_subgraph": {"genes": "non-empty string array", "layers": "optional non-empty layer-name array"},
        },
        "candidate_gene_ids": graph_candidate_gene_ids[:PROMPT_TOOL_REFERENCE_GENE_LIMIT],
        "candidate_gene_id_count": len(graph_candidate_gene_ids),
        "unavailable_candidate_gene_ids": unavailable_candidate_gene_ids[:PROMPT_TOOL_REFERENCE_GENE_LIMIT],
        "available_layer_names": available_layers[:PROMPT_TOOL_REFERENCE_LAYER_LIMIT],
        "available_layer_count": len(available_layers),
    }


def _actor_sampling_directive_payload(
    *,
    sample_index: int,
    task_type: str | None,
) -> dict[str, Any]:
    directives = TASK_ACTOR_DIVERSITY_DIRECTIVES.get(str(task_type), ACTOR_DIVERSITY_DIRECTIVES)
    directive = directives[sample_index % len(directives)]
    return {
        "strategy": "verbalized",
        "sample_index": sample_index,
        "task_type": task_type,
        "directive_name": directive["name"],
        "instruction": directive["instruction"],
        "preferred_tools": directive["preferred_tools"],
        "rules": [
            "Follow this directive only when it is valid for the current state and available candidate_gene_ids.",
            "If the preferred tool is invalid or unhelpful, choose the best valid alternative.",
            "Do not copy another sample's action merely for consistency; this sample is meant to explore a distinct plausible branch.",
        ],
    }


def _actor_tool_coverage_directive_payload(
    *,
    sample_index: int,
    task_type: str | None,
) -> dict[str, Any]:
    directive = TOOL_COVERAGE_DIRECTIVES.get(str(task_type))
    if directive is None:
        directive = {
            "name": "tool_coverage_probe",
            "instruction": (
                "Choose a valid runtime tool if any valid graph argument can be formed; "
                "otherwise explain why no tool is currently useful."
            ),
            "preferred_tools": ["induce_subgraph", "shortest_path", "get_neighbors"],
        }
    return {
        "strategy": "tool_coverage_retry",
        "sample_index": sample_index,
        "task_type": task_type,
        "directive_name": directive["name"],
        "instruction": directive["instruction"],
        "preferred_tools": directive["preferred_tools"],
        "rules": [
            "Prefer a valid runtime tool over no_tool for this retry.",
            "Use only provided candidate_gene_ids or exact ids returned by prior successful tool observations.",
            "If no valid tool argument can be formed, choose no_tool and explain the blocker.",
        ],
    }


def _actor_prompt_payload(
    context: SharedPrefixContext,
    *,
    step_index: int,
    environment: RuntimeEnvironment | None = None,
    actor_sampling_directive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_payload = _state_payload_for_model_prompt(context.state)
    payload = {
        "query_text": context.query_text,
        "visible_inputs": context.user_evidence,
        "interpretation": _interpretation_payload_for_model_prompt(context.interpretation),
        "state": state_payload,
        "step_index": step_index,
        "tool_argument_reference": _tool_argument_reference_payload(
            context,
            environment=environment,
        ),
    }
    if actor_sampling_directive is not None:
        payload["actor_sampling_directive"] = actor_sampling_directive
    return payload


def _observation_for_verifier_prompt(observation: ToolObservation | None) -> dict[str, Any] | None:
    if observation is None or observation.status != ToolObservationStatus.SUCCESS:
        return None

    payload = observation.payload or {}
    provenance = observation.provenance or {}
    tool_name = _safe_text(provenance.get("tool_name"))
    summary, supporting_gene_ids = _summarize_observation(observation)
    compact_payload: dict[str, Any]

    if tool_name == "get_neighbors":
        layers = payload.get("layers", [])
        layer_summaries: list[dict[str, Any]] = []
        for layer_payload in (layers if isinstance(layers, list) else []):
            if not isinstance(layer_payload, dict):
                continue
            neighbor_count = int(layer_payload.get("neighbor_count", 0) or 0)
            if neighbor_count <= 0:
                continue
            layer_summaries.append(
                {
                    "layer_name": layer_payload.get("layer_name"),
                    "neighbor_count": neighbor_count,
                    "neighbors_sample": _preview_list(layer_payload.get("neighbors"), limit=8),
                }
            )
            if len(layer_summaries) >= PROMPT_LAYER_PREVIEW_LIMIT:
                break
        compact_payload = {
            "query_gene_id": payload.get("query_gene_id"),
            "unique_neighbor_count": payload.get("unique_neighbor_count", 0),
            "unique_neighbors_sample": _preview_list(payload.get("unique_neighbors")),
            "layers_with_neighbors_sample": layer_summaries,
        }
    elif tool_name == "induce_subgraph":
        edge_samples: list[dict[str, Any]] = []
        layers_with_edges: list[dict[str, Any]] = []
        layers = payload.get("layers", [])
        for layer_payload in (layers if isinstance(layers, list) else []):
            if not isinstance(layer_payload, dict):
                continue
            edge_count = int(layer_payload.get("edge_count", 0) or 0)
            if edge_count <= 0:
                continue
            edges = _preview_list(
                layer_payload.get("edges"),
                limit=max(0, PROMPT_EDGE_PREVIEW_LIMIT - len(edge_samples)),
            )
            edge_samples.extend(edge for edge in edges if isinstance(edge, dict))
            layers_with_edges.append(
                {
                    "layer_name": layer_payload.get("layer_name"),
                    "edge_count": edge_count,
                    "present_gene_ids": _preview_list(layer_payload.get("present_gene_ids")),
                }
            )
            if len(layers_with_edges) >= PROMPT_LAYER_PREVIEW_LIMIT or len(edge_samples) >= PROMPT_EDGE_PREVIEW_LIMIT:
                break
        compact_payload = {
            "query_gene_ids": _preview_list(payload.get("query_gene_ids")),
            "present_gene_ids": _preview_list(payload.get("present_gene_ids")),
            "missing_gene_ids": _preview_list(payload.get("missing_gene_ids")),
            "combined_edge_count": payload.get("combined_edge_count", 0),
            "layers_with_edges_sample": layers_with_edges,
            "edge_sample": edge_samples,
        }
    elif tool_name == "shortest_path":
        compact_payload = {
            "source_gene_id": payload.get("source_gene_id"),
            "target_gene_id": payload.get("target_gene_id"),
            "path_gene_ids": _preview_list(payload.get("path_gene_ids")),
            "hop_count": payload.get("hop_count"),
            "layer_name": payload.get("layer_name"),
        }
    elif tool_name in {"rwr_multiplex", "rwr_monoplex"}:
        seed_gene_ids = _safe_list_of_strings(payload.get("seed_gene_ids"))
        seed_gene_set = set(seed_gene_ids)
        results = payload.get("results", [])
        result_list = results if isinstance(results, list) else []
        non_seed_results = [
            result
            for result in result_list
            if (gene_id := _rwr_result_gene_id(result)) is not None and gene_id not in seed_gene_set
        ]
        seed_results = [
            result
            for result in result_list
            if (gene_id := _rwr_result_gene_id(result)) is not None and gene_id in seed_gene_set
        ]
        compact_payload = {
            "seed_gene_ids": _preview_list(seed_gene_ids),
            "active_seed_gene_ids": _preview_list(payload.get("active_seed_gene_ids")),
            "top_k": payload.get("top_k"),
            "result_count": len(result_list),
            "seed_results_sample": _preview_list(seed_results, limit=8),
            "non_seed_result_count": len(non_seed_results),
            "top_non_seed_results": _preview_list(
                non_seed_results,
                limit=PROMPT_RWR_NON_SEED_PREVIEW_LIMIT,
            ),
            "results": _preview_list(
                result_list,
                limit=PROMPT_RWR_RESULT_PREVIEW_LIMIT,
            ),
            "recovery_interpretation_hint": (
                "For recovery tasks, evaluate top_non_seed_results as possible "
                "missing complex members before deciding that the seed group is complete."
            ),
        }
        if "layer_name" in payload:
            compact_payload["layer_name"] = payload.get("layer_name")
        if "active_layers" in payload:
            compact_payload.update(
                _compact_layer_list_payload(
                    payload.get("active_layers"),
                    count_key="active_layer_count",
                    sample_key="active_layers_sample",
                )
            )
    elif tool_name == "query_mygene":
        compact_payload = {
            "query": payload.get("query"),
            "requested_fields": _preview_list(payload.get("requested_fields")),
            "result_count": payload.get("result_count", 0),
            "results": _preview_list(payload.get("results"), limit=PROMPT_MYGENE_PREVIEW_LIMIT),
        }
    else:
        compact_payload = {}

    return {
        "status": observation.status.value,
        "tool_name": tool_name,
        "call_id": observation.call_id,
        "summary": summary,
        "supporting_gene_ids_sample": supporting_gene_ids[:PROMPT_LIST_PREVIEW_LIMIT],
        "provenance": _compact_provenance_for_prompt(provenance),
        "payload": compact_payload,
    }


def _verifier_prompt_payload(
    context: SharedPrefixContext,
    *,
    actor_step: ActorStep,
    observation: ToolObservation | None,
    step_index: int,
    task_type: str | None = None,
) -> dict[str, Any]:
    prior_state_payload = _state_payload_for_model_prompt(context.state)
    payload = {
        "query_text": context.query_text,
        "visible_inputs": context.user_evidence,
        "prior_interpretation": _interpretation_payload_for_model_prompt(context.interpretation),
        "prior_state": prior_state_payload,
        "actor_output": {
            "reasoning_text": _truncate_prompt_text(
                actor_step.reasoning_text,
                max_chars=PROMPT_ACTOR_REASONING_MAX_CHARS,
            ),
            "tool_action": actor_step.tool_action.to_dict() if actor_step.tool_action else None,
        },
        "deterministic_observation": _observation_for_verifier_prompt(observation),
        "step_index": step_index,
    }
    if task_type == "recovery":
        payload["task_guidance"] = {
            "objective": "Recover missing coherent complex members beyond the current seed/candidate group.",
            "candidate_policy": [
                "Inspect top non-seed tool candidates before marking the seed group complete.",
                "Add a non-seed candidate only when the visible observation gives credible support.",
                "If no non-seed candidate is supported, explain that limitation and choose continue when another query could help.",
            ],
        }
    return payload


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _request_error_raw_text(error_kind: str, error: Exception) -> str:
    return _json_dumps_compact(
        {
            "error": error_kind,
            "message": _truncate_prompt_text(str(error), max_chars=1200),
        }
    )


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = ""
            for key in ("text", "content", "input_text"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    text = value.strip()
                    break
        else:
            text = ""
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _response_choice_to_text(choice: dict[str, Any]) -> str:
    message = _safe_dict(choice.get("message"))
    content_text = _message_content_to_text(message.get("content"))
    if content_text:
        return content_text

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return _json_dumps_compact({"tool_calls": tool_calls})

    reasoning_text = _message_content_to_text(message.get("reasoning_content"))
    if reasoning_text:
        return _json_dumps_compact({"reasoning_content": reasoning_text})

    return _json_dumps_compact(
        {
            "choice_index": choice.get("index"),
            "finish_reason": choice.get("finish_reason"),
            "message": message,
        }
    )


def _message_has_visible_output(message: dict[str, Any]) -> bool:
    if _message_content_to_text(message.get("content")):
        return True
    if _message_content_to_text(message.get("reasoning_content")):
        return True
    if _safe_text(message.get("refusal")):
        return True
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _choice_has_visible_output(choice: dict[str, Any]) -> bool:
    return _message_has_visible_output(_safe_dict(choice.get("message")))


def _response_truncated_before_visible_output(
    payload: dict[str, Any],
    *,
    prefix: str,
) -> str | None:
    if payload.get("finish_reason") != "length":
        return None
    message = _safe_dict(payload.get("message"))
    if _message_has_visible_output(message):
        return None
    return f"{prefix}_response_truncated_before_visible_output"


def _function_call_from_tool_calls(
    tool_calls: Any,
    *,
    prefix: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(tool_calls, list) or not tool_calls:
        return None, errors

    if len(tool_calls) > 1:
        errors.append(f"{prefix}_multiple_tool_calls_returned")

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None, errors + [f"{prefix}_tool_call_not_a_dict"]

    function_payload = _safe_dict(first_call.get("function"))
    tool_name = function_payload.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None, errors + [f"{prefix}_tool_call_name_missing_or_invalid"]

    raw_arguments = function_payload.get("arguments")
    arguments: dict[str, Any] = {}
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        if raw_arguments.strip():
            try:
                parsed_arguments = json.loads(raw_arguments)
            except Exception as error:
                errors.append(f"{prefix}_tool_call_arguments_json_parse_error: {error}")
            else:
                if isinstance(parsed_arguments, dict):
                    arguments = parsed_arguments
                else:
                    errors.append(f"{prefix}_tool_call_arguments_not_a_dict")
    elif raw_arguments is not None:
        errors.append(f"{prefix}_tool_call_arguments_invalid")

    return {"tool_name": tool_name, "arguments": arguments}, errors


def _named_tool_arguments_from_choice(
    choice: dict[str, Any],
    *,
    expected_tool_name: str,
    prefix: str,
    require_tool_call: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    message = _safe_dict(choice.get("message"))
    errors: list[str] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        function_call, tool_call_errors = _function_call_from_tool_calls(
            tool_calls,
            prefix=prefix,
        )
        errors.extend(tool_call_errors)
        if function_call is not None and function_call.get("tool_name") != expected_tool_name:
            errors.append(f"{prefix}_tool_call_name_unexpected")
        payload = (
            _safe_dict(function_call.get("arguments"))
            if function_call is not None
            else {}
        )
    else:
        if require_tool_call:
            errors.append(f"{prefix}_tool_call_missing")
        content_text = _message_content_to_text(message.get("content"))
        payload = _parse_model_json(content_text) if content_text else {}

    payload.setdefault("finish_reason", choice.get("finish_reason"))
    payload.setdefault("message", message)
    reasoning_text = _message_content_to_text(message.get("reasoning_content"))
    if reasoning_text and "reasoning_content" not in payload:
        payload["reasoning_content"] = reasoning_text
    return payload, errors


def _named_function_tool(
    *,
    function_name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": description,
            "parameters": parameters,
        },
    }


def _named_function_tool_choice(function_name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": function_name},
    }


def _actor_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning_text": {"type": "string"},
            "tool_action": _actor_tool_action_schema(),
        },
        "required": ["reasoning_text", "tool_action"],
        "additionalProperties": False,
    }


def _actor_tool_action_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "null"},
            {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "enum": list(RUNTIME_TOOL_NAMES),
                    },
                    "arguments": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["tool_name", "arguments"],
                "additionalProperties": False,
            },
        ]
    }


def _actor_action_only_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool_action": _actor_tool_action_schema(),
        },
        "required": ["tool_action"],
        "additionalProperties": False,
    }


def _actor_reasoning_text_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reasoning_text": {"type": "string"},
        },
        "required": ["reasoning_text"],
        "additionalProperties": False,
    }


def _verifier_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "updated_interpretation": {
                "type": "object",
                "properties": {
                    "mechanistic_claim": {"type": "string"},
                    "main_evidence": {"type": "string"},
                    "uncertainty": {"type": "string"},
                    "next_subgoal": {"type": "string"},
                },
                "required": [
                    "mechanistic_claim",
                    "main_evidence",
                    "uncertainty",
                    "next_subgoal",
                ],
                "additionalProperties": False,
            },
            "updated_state": {
                "type": "object",
                "properties": {
                    "relationship_status": {
                        "type": "string",
                        "enum": [status.value for status in RelationshipStatus],
                    },
                    "predicted_gene_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mechanistic_labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label_source": {
                                    "type": "string",
                                    "enum": [source.value for source in LabelSource],
                                },
                                "label_name": {"type": "string"},
                                "label_id": {"type": ["string", "null"]},
                            },
                            "required": ["label_source", "label_name", "label_id"],
                            "additionalProperties": False,
                        },
                    },
                    "continuation_decision": {
                        "type": "string",
                        "enum": [state.value for state in ContinuationState],
                    },
                    "verifier_notes": {"type": "string"},
                },
                "required": [
                    "relationship_status",
                    "predicted_gene_ids",
                    "mechanistic_labels",
                    "continuation_decision",
                    "verifier_notes",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["updated_interpretation", "updated_state"],
        "additionalProperties": False,
    }


def _responses_function_tool_from_chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function_payload = _safe_dict(tool.get("function"))
    return {
        "type": "function",
        "name": _safe_text(function_payload.get("name")),
        "description": _safe_text(function_payload.get("description")),
        "parameters": _safe_dict(function_payload.get("parameters")),
        "strict": True,
    }


def _responses_json_schema_text_config(
    *,
    name: str,
    description: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "description": description,
            "schema": schema,
            "strict": True,
        }
    }


def _actor_output_tool(parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return _named_function_tool(
        function_name=ACTOR_OUTPUT_TOOL_NAME,
        description="Emit the actor policy decision for the next MENTOR-RL step.",
        parameters=parameters if parameters is not None else _actor_output_schema(),
    )


def _verifier_output_tool() -> dict[str, Any]:
    return _named_function_tool(
        function_name=VERIFIER_OUTPUT_TOOL_NAME,
        description="Emit the verifier policy update for the current MENTOR-RL branch.",
        parameters=_verifier_output_schema(),
    )


def _runtime_tool_parameters(tool_name: str) -> dict[str, Any]:
    if tool_name == "query_mygene":
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One gene symbol, alias, or Ensembl id to resolve.",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Optional metadata fields to request.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    if tool_name == "get_neighbors":
        return {
            "type": "object",
            "properties": {
                "gene": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One canonical Ensembl gene id from the visible candidate_gene_ids or a prior successful tool observation.",
                },
                "layers": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Optional concrete layer names. Omit this field to query all layers; do not use 'all'.",
                },
            },
            "required": ["gene"],
            "additionalProperties": False,
        }
    if tool_name == "shortest_path":
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One canonical Ensembl gene id string. Never pass an array or comma-separated list.",
                },
                "target": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One canonical Ensembl gene id string. Never pass an array or comma-separated list.",
                },
                "layer": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Optional concrete layer name. Omit this field to query all layers; do not use 'all'.",
                },
            },
            "required": ["source", "target"],
            "additionalProperties": False,
        }
    if tool_name == "rwr_multiplex":
        return {
            "type": "object",
            "properties": {
                "seeds": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Canonical Ensembl gene id strings from candidate_gene_ids or prior successful tool observations.",
                },
                "top_k": {"type": "integer", "minimum": 1},
            },
            "required": ["seeds"],
            "additionalProperties": False,
        }
    if tool_name == "rwr_monoplex":
        return {
            "type": "object",
            "properties": {
                "seeds": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Canonical Ensembl gene id strings from candidate_gene_ids or prior successful tool observations.",
                },
                "layer": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A concrete layer name. Use rwr_multiplex instead when querying across all layers.",
                },
                "top_k": {"type": "integer", "minimum": 1},
            },
            "required": ["seeds", "layer"],
            "additionalProperties": False,
        }
    if tool_name == "induce_subgraph":
        return {
            "type": "object",
            "properties": {
                "genes": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Canonical Ensembl gene id strings from candidate_gene_ids or prior successful tool observations.",
                },
                "layers": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "description": "Optional concrete layer names. Omit this field to query all layers; do not use 'all'.",
                },
            },
            "required": ["genes"],
            "additionalProperties": False,
        }
    raise ValueError(f"Unknown runtime tool: {tool_name}")


def _runtime_tool_description(tool_name: str) -> str:
    descriptions = {
        "query_mygene": "Retrieve gene identifier and metadata information for one query string.",
        "get_neighbors": "Retrieve direct graph neighbors for one gene.",
        "shortest_path": "Compute a shortest path between two genes.",
        "rwr_multiplex": "Rank genes by random walk with restart across the multiplex.",
        "rwr_monoplex": "Rank genes by random walk with restart on one named layer.",
        "induce_subgraph": "Inspect the subgraph induced by a queried gene set.",
    }
    return descriptions[tool_name]


def _runtime_tools() -> list[dict[str, Any]]:
    return [
        _named_function_tool(
            function_name=tool_name,
            description=_runtime_tool_description(tool_name),
            parameters=_runtime_tool_parameters(tool_name),
        )
        for tool_name in RUNTIME_TOOL_NAMES
    ]


def _normalize_runtime_tool_action(
    payload: dict[str, Any] | None,
    *,
    prefix: str = "actor",
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if payload is None:
        return None, errors
    if not isinstance(payload, dict):
        return None, [f"{prefix}_tool_action_not_a_dict"]

    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not tool_name:
        return None, [f"{prefix}_tool_name_missing_or_invalid"]
    if tool_name not in RUNTIME_TOOL_NAMES:
        return None, [f"{prefix}_tool_name_unknown"]
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        errors.append(f"{prefix}_tool_arguments_not_a_dict")
        arguments = {}
    arguments = normalize_tool_arguments(tool_name, arguments)
    return {"tool_name": tool_name, "arguments": arguments}, errors


def _extract_tool_action_from_text(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(text, str) or not text.strip():
        return None, []

    for match in TOOL_ACTION_LINE_RE.finditer(text):
        raw_payload = match.group("payload")
        try:
            payload = json.loads(raw_payload)
        except requests.RequestException as error:
            return None, [f"actor_tool_action_json_parse_error: {error}"]
        return _normalize_runtime_tool_action(payload)

    try:
        payload = _parse_model_json(text)
    except Exception:
        return None, []

    if "tool_action" in payload:
        return _normalize_runtime_tool_action(payload.get("tool_action"))
    if "tool_name" in payload:
        return _normalize_runtime_tool_action(payload)
    return None, []


def _normalize_actor_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []

    response_error = _response_truncated_before_visible_output(payload, prefix="actor")
    if response_error:
        errors.append(response_error)

    reasoning_text = ""
    raw_reasoning_text = payload.get("reasoning_text")
    if isinstance(raw_reasoning_text, str):
        reasoning_text = raw_reasoning_text
    elif raw_reasoning_text is not None:
        errors.append("actor_reasoning_text_invalid")

    if not reasoning_text:
        reasoning_content = payload.get("reasoning_content")
        if isinstance(reasoning_content, str):
            reasoning_text = reasoning_content

    tool_action = payload.get("tool_action")
    if "tool_calls" in payload:
        tool_action, tool_call_errors = _function_call_from_tool_calls(
            payload.get("tool_calls"),
            prefix="actor",
        )
        errors.extend(tool_call_errors)
    tool_action, tool_action_errors = _normalize_runtime_tool_action(tool_action)
    errors.extend(tool_action_errors)

    if not reasoning_text and tool_action is None:
        errors.append("actor_reasoning_and_tool_action_blank")

    return {"reasoning_text": reasoning_text, "tool_action": tool_action}, errors


def _actor_candidate_from_choice(choice: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    message = _safe_dict(choice.get("message"))
    errors: list[str] = []

    response_error = _response_truncated_before_visible_output(choice, prefix="actor")
    if response_error:
        errors.append(response_error)

    content_text = _message_content_to_text(message.get("content"))
    reasoning_content = _message_content_to_text(message.get("reasoning_content"))
    reasoning_text = content_text or reasoning_content

    tool_action = None
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        function_call, tool_call_errors = _function_call_from_tool_calls(
            tool_calls,
            prefix="actor",
        )
        errors.extend(tool_call_errors)
        if function_call and function_call.get("tool_name") == ACTOR_OUTPUT_TOOL_NAME:
            legacy_payload = _safe_dict(function_call.get("arguments"))
            normalized, normalize_errors = _normalize_actor_payload(legacy_payload)
            errors.extend(normalize_errors)
            if not reasoning_text:
                reasoning_text = normalized["reasoning_text"]
            tool_action = normalized["tool_action"]
        else:
            tool_action, normalize_errors = _normalize_runtime_tool_action(function_call)
            errors.extend(normalize_errors)

    if tool_action is None and content_text:
        parsed_tool_action, parse_errors = _extract_tool_action_from_text(content_text)
        errors.extend(parse_errors)
        if parsed_tool_action is not None:
            tool_action = parsed_tool_action
            try:
                legacy_payload = _parse_model_json(content_text)
            except Exception:
                legacy_payload = {}
            if (
                isinstance(legacy_payload, dict)
                and isinstance(legacy_payload.get("reasoning_text"), str)
                and content_text.strip().startswith("{")
            ):
                reasoning_text = legacy_payload["reasoning_text"]

    if not reasoning_text and tool_action is None:
        errors.append("actor_reasoning_and_tool_action_blank")

    return {"reasoning_text": reasoning_text, "tool_action": tool_action}, errors


def _validate_verifier_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    response_error = _response_truncated_before_visible_output(payload, prefix="verifier")
    if response_error:
        errors.append(response_error)
    if not isinstance(payload.get("updated_interpretation"), dict):
        errors.append("verifier_updated_interpretation_missing_or_invalid")
    if not isinstance(payload.get("updated_state"), dict):
        errors.append("verifier_updated_state_missing_or_invalid")
    updated_interpretation = _safe_dict(payload.get("updated_interpretation"))
    updated_state = _safe_dict(payload.get("updated_state"))
    if (
        isinstance(payload.get("updated_interpretation"), dict)
        and isinstance(payload.get("updated_state"), dict)
        and not any(
            value not in (None, "", [], {})
            for value in list(updated_interpretation.values()) + list(updated_state.values())
        )
    ):
        errors.append("verifier_payload_blank")
    return errors


def _has_error_prefix(errors: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(
        isinstance(error, str) and error.startswith(prefix)
        for error in errors
        for prefix in prefixes
    )


def _actor_candidate_is_usable(actor_step: ActorStep, errors: Iterable[str]) -> bool:
    fatal_errors = {
        "actor_response_truncated_before_visible_output",
        "actor_tool_action_not_a_dict",
        "actor_tool_name_missing_or_invalid",
        "actor_tool_name_unknown",
        "actor_reasoning_and_tool_action_blank",
        "repeated_character_collapse",
        "repeated_token_collapse",
        "whitespace_only_output",
    }
    if any(error in fatal_errors for error in errors):
        return False
    if _has_error_prefix(
        errors,
        (
            "actor_json_parse_error:",
            "actor_tool_semantics_invalid:",
            "actor_tool_action_json_parse_error:",
            "actor_tool_call_",
        ),
    ):
        return False
    return bool(actor_step.reasoning_text.strip()) or actor_step.tool_action is not None


def _verifier_candidate_is_usable(candidate: dict[str, Any], errors: Iterable[str]) -> bool:
    fatal_errors = {
        "verifier_response_truncated_before_visible_output",
        "verifier_updated_interpretation_missing_or_invalid",
        "verifier_updated_state_missing_or_invalid",
        "verifier_payload_blank",
    }
    if any(error in fatal_errors for error in errors):
        return False
    if _has_error_prefix(errors, ("verifier_json_parse_error:", "verifier_tool_call_")):
        return False

    payload = _safe_dict(candidate.get("payload"))
    return (
        isinstance(payload.get("updated_interpretation"), dict)
        and isinstance(payload.get("updated_state"), dict)
    )


@dataclass
class ModelGeneratorConfig:
    """Configuration for the model-backed trajectory candidate generator."""

    api_base: str = DEFAULT_GENERATOR_API_BASE
    api_mode: str = "auto"
    model_name: str | None = None
    api_key: str | None = None
    api_key_env: str = DEFAULT_GENERATOR_API_KEY_ENV
    request_timeout_seconds: int = 3600
    temperature: float = 0.8
    top_p: float = 0.95
    max_completion_tokens: int = 4096
    actor_rationale_max_completion_tokens: int = DEFAULT_ACTOR_RATIONALE_MAX_TOKENS
    reasoning_effort: str = "low"
    actor_sampling_strategy: str = "batch"

    def __post_init__(self) -> None:
        if self.api_mode not in {"auto", "chat_completions", "responses", "completions"}:
            raise ValueError(
                "api_mode must be one of: auto, chat_completions, responses, completions."
            )
        if self.actor_sampling_strategy not in ACTOR_SAMPLING_STRATEGIES:
            allowed = ", ".join(ACTOR_SAMPLING_STRATEGIES)
            raise ValueError(f"actor_sampling_strategy must be one of: {allowed}.")
        if self.max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive.")
        if self.actor_rationale_max_completion_tokens <= 0:
            raise ValueError("actor_rationale_max_completion_tokens must be positive.")

    def resolved_api_key(self) -> str:
        """Return the API key, falling back to an environment variable."""

        if self.api_key is not None:
            return self.api_key
        return os.getenv(self.api_key_env, "EMPTY")


class OpenAICompatibleCandidateGenerator:
    """Generate actor and verifier candidates through OpenAI-compatible APIs."""

    def __init__(self, config: ModelGeneratorConfig) -> None:
        self.config = config
        self.session: requests.Session | Any | None = None
        self._thread_local = threading.local()
        self.model_name = config.model_name or self._discover_model_name()
        self.api_mode = self._resolve_api_mode()

    def _session(self) -> requests.Session | Any:
        if self.session is not None:
            return self.session

        thread_session = getattr(self._thread_local, "session", None)
        if thread_session is None:
            thread_session = requests.Session()
            self._thread_local.session = thread_session
        return thread_session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.resolved_api_key()}",
            "Content-Type": "application/json",
        }

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            body = _safe_text(getattr(response, "text", "")).strip()
            if len(body) > 2000:
                body = body[:2000].rstrip() + "... [truncated]"
            message = str(error)
            if body:
                message = f"{message}; response_body={body}"
            raise requests.HTTPError(message, response=response) from error

    def _discover_model_name(self) -> str:
        response = self._session().get(
            f"{self.config.api_base.rstrip('/')}/models",
            headers=self._headers(),
            timeout=self.config.request_timeout_seconds,
        )
        self._raise_for_status(response)
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("The generator API did not report any served models.")
        model_name = data[0].get("id")
        if not isinstance(model_name, str) or not model_name:
            raise RuntimeError("The generator API returned an invalid model id.")
        return model_name

    def _resolve_api_mode(self) -> str:
        if self.config.api_mode != "auto":
            return self.config.api_mode
        model_name = self.model_name.lower()
        if "gpt-oss" in model_name:
            return "completions"
        return "chat_completions"

    def _model_is_gpt_oss(self) -> bool:
        return "gpt-oss" in self.model_name.lower()

    def _prefers_named_output_tools(self) -> bool:
        return self.api_mode == "chat_completions" and self._model_is_gpt_oss()

    def _should_disable_hidden_thinking(self) -> bool:
        return self.api_mode in {"chat_completions", "completions"} and self._model_is_gpt_oss()

    def _should_generate_actor_rationale(self) -> bool:
        return False

    def _prompt_tokenizer(self) -> Any:
        thread_tokenizer = getattr(self._thread_local, "tokenizer", None)
        if thread_tokenizer is None:
            from transformers import AutoTokenizer

            thread_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._thread_local.tokenizer = thread_tokenizer
        return thread_tokenizer

    def _render_completion_prompt(
        self,
        messages: list[dict[str, str]],
        *,
        disable_hidden_thinking: bool,
    ) -> str:
        tokenizer = self._prompt_tokenizer()
        chat_template_kwargs: dict[str, Any] = {}
        if disable_hidden_thinking and self._model_is_gpt_oss():
            chat_template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("The completion prompt renderer returned an empty prompt.")
        return prompt

    def _assert_gpt_oss_completion_prompt_contract(
        self,
        prompt: str,
        *,
        disable_hidden_thinking: bool,
    ) -> None:
        if not (disable_hidden_thinking and self._model_is_gpt_oss()):
            return
        if prompt.rstrip().endswith(GPT_OSS_FINAL_CHANNEL_PREFIX):
            return
        raise RuntimeError(
            "The gpt-oss chat template for "
            f"{self.model_name!r} does not appear to honor enable_thinking=False. "
            "Expected the rendered completions prompt to end with "
            f"{GPT_OSS_FINAL_CHANNEL_PREFIX!r}."
        )

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        n: int,
        seed: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        guided_json: dict[str, Any] | None = None,
        use_reasoning: bool = True,
        max_completion_tokens: int | None = None,
        disable_hidden_thinking: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "n": n,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "max_completion_tokens": (
                max_completion_tokens
                if max_completion_tokens is not None
                else self.config.max_completion_tokens
            ),
            "include_reasoning": False,
            "seed": seed,
        }
        if use_reasoning and self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if disable_hidden_thinking and self._model_is_gpt_oss():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if guided_json is not None:
            payload["guided_json"] = guided_json
        response = self._session().post(
            f"{self.config.api_base.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        self._raise_for_status(response)
        response_payload = response.json()
        choices: list[dict[str, Any]] = []
        for choice in response_payload.get("choices", []):
            if not isinstance(choice, dict):
                choices.append(
                    {
                        "index": None,
                        "finish_reason": None,
                        "message": {"content": _json_dumps_compact({"choice": choice})},
                    }
                )
                continue
            choices.append(choice)
        if not choices:
            choices.append(
                {
                    "index": None,
                    "finish_reason": None,
                    "message": {
                        "content": _json_dumps_compact(
                            {
                                "error": "chat_completion_returned_no_choices",
                                "response_keys": sorted(response_payload),
                            }
                        )
                    }
                }
            )
        return choices

    def _responses_finish_reason(
        self,
        response_payload: dict[str, Any],
        *,
        tool_calls_present: bool,
    ) -> str | None:
        if tool_calls_present:
            return "tool_calls"

        incomplete_details = _safe_dict(response_payload.get("incomplete_details"))
        incomplete_reason = _safe_text(incomplete_details.get("reason")).lower()
        if incomplete_reason in {"length", "max_output_tokens", "max_completion_tokens"}:
            return "length"

        status = _safe_text(response_payload.get("status")).lower()
        if status == "completed":
            return "stop"
        if status == "incomplete" and incomplete_reason:
            return "length" if "max" in incomplete_reason else incomplete_reason
        return None

    def _responses_choice_from_payload(
        self,
        response_payload: dict[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "tool_calls": []}
        content_parts: list[dict[str, Any]] = []
        reasoning_parts: list[dict[str, Any]] = []

        for item in response_payload.get("output", []):
            if not isinstance(item, dict):
                continue
            item_type = _safe_text(item.get("type"))
            if item_type == "message":
                for part in item.get("content", []):
                    if isinstance(part, dict):
                        content_parts.append(part)
            elif item_type == "reasoning":
                for part in item.get("content", []):
                    if isinstance(part, dict):
                        reasoning_parts.append(part)
            elif item_type == "function_call":
                function_name = _safe_text(item.get("name"))
                raw_arguments = item.get("arguments")
                arguments = raw_arguments if isinstance(raw_arguments, dict) else _safe_text(raw_arguments)
                if function_name:
                    message["tool_calls"].append(
                        {
                            "id": _safe_text(item.get("call_id")) or _safe_text(item.get("id")),
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": arguments,
                            },
                        }
                    )

        if content_parts:
            message["content"] = content_parts
        if reasoning_parts:
            message["reasoning_content"] = reasoning_parts

        return {
            "index": index,
            "finish_reason": self._responses_finish_reason(
                response_payload,
                tool_calls_present=bool(message["tool_calls"]),
            ),
            "message": message,
            "response_id": response_payload.get("id"),
            "status": response_payload.get("status"),
        }

    def _responses(
        self,
        *,
        messages: list[dict[str, str]],
        n: int,
        tools: list[dict[str, Any]] | None = None,
        text_config: dict[str, Any] | None = None,
        use_reasoning: bool = True,
        max_output_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "input": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_output_tokens": (
                max_output_tokens
                if max_output_tokens is not None
                else self.config.max_completion_tokens
            ),
            "parallel_tool_calls": False,
            "store": False,
        }
        if use_reasoning and self.config.reasoning_effort:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        if tools:
            payload["tools"] = [
                _responses_function_tool_from_chat_tool(tool)
                for tool in tools
            ]
        if text_config is not None:
            payload["text"] = text_config

        choices: list[dict[str, Any]] = []
        for index in range(n):
            # vLLM Harmony only supports auto tool selection on /responses.
            response = self._session().post(
                f"{self.config.api_base.rstrip('/')}/responses",
                headers=self._headers(),
                json=payload,
                timeout=self.config.request_timeout_seconds,
            )
            self._raise_for_status(response)
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                response_payload = {"output": [], "status": None, "raw_response": response_payload}
            choices.append(self._responses_choice_from_payload(response_payload, index=index))

        if not choices:
            choices.append(
                {
                    "index": None,
                    "finish_reason": None,
                    "message": {
                        "content": _json_dumps_compact({"error": "responses_api_returned_no_choices"}),
                    },
                }
            )
        return choices

    def _completions(
        self,
        *,
        messages: list[dict[str, str]],
        n: int,
        seed: int,
        guided_json: dict[str, Any] | None = None,
        max_completion_tokens: int | None = None,
        disable_hidden_thinking: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[dict[str, Any]]:
        prompt = self._render_completion_prompt(
            messages,
            disable_hidden_thinking=disable_hidden_thinking,
        )
        self._assert_gpt_oss_completion_prompt_contract(
            prompt,
            disable_hidden_thinking=disable_hidden_thinking,
        )
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "n": n,
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "max_tokens": (
                max_completion_tokens
                if max_completion_tokens is not None
                else self.config.max_completion_tokens
            ),
            "seed": seed,
            "add_special_tokens": False,
        }
        if guided_json is not None:
            payload["guided_json"] = guided_json
        response = self._session().post(
            f"{self.config.api_base.rstrip('/')}/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        self._raise_for_status(response)
        response_payload = response.json()
        choices: list[dict[str, Any]] = []
        for choice in response_payload.get("choices", []):
            if not isinstance(choice, dict):
                choices.append(
                    {
                        "index": None,
                        "finish_reason": None,
                        "message": {"content": _json_dumps_compact({"choice": choice})},
                    }
                )
                continue
            choices.append(
                {
                    "index": choice.get("index"),
                    "finish_reason": choice.get("finish_reason"),
                    "message": {
                        "role": "assistant",
                        "content": _safe_text(choice.get("text")),
                    },
                }
            )
        if not choices:
            choices.append(
                {
                    "index": None,
                    "finish_reason": None,
                    "message": {
                        "content": _json_dumps_compact(
                            {
                                "error": "completions_api_returned_no_choices",
                                "response_keys": sorted(response_payload),
                            }
                        )
                    },
                }
            )
        return choices

    def _generate_choices(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        n: int,
        seed: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        guided_json: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        disable_hidden_thinking: bool = False,
    ) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        structured_max_tokens = min(self.config.max_completion_tokens, STRUCTURED_OUTPUT_MAX_TOKENS)
        if self.api_mode == "responses":
            choices = self._responses(
                messages=messages,
                n=n,
                tools=tools,
                text_config=text_config,
                use_reasoning=False,
                max_output_tokens=structured_max_tokens,
            )
            fallback_guided_json = guided_json
            if fallback_guided_json is None and text_config is not None:
                fallback_guided_json = _safe_dict(_safe_dict(text_config.get("format")).get("schema"))
            for index, choice in enumerate(choices):
                if _choice_has_visible_output(choice):
                    continue
                fallback_choices = self._chat(
                    messages,
                    n=1,
                    seed=seed + index,
                    tools=tools,
                    tool_choice=tool_choice,
                    guided_json=fallback_guided_json,
                    use_reasoning=False,
                    max_completion_tokens=structured_max_tokens,
                    disable_hidden_thinking=disable_hidden_thinking,
                )
                fallback_choice = fallback_choices[0]
                if _choice_has_visible_output(fallback_choice):
                    fallback_choice["fallback_backend"] = "chat_completions"
                    fallback_choice["fallback_trigger"] = "responses_blank_visible_output"
                    choices[index] = fallback_choice
            return choices

        if self.api_mode == "completions":
            if tools or tool_choice:
                raise ValueError("The completions backend does not support tool calls.")
            completion_guided_json = guided_json
            if completion_guided_json is None and text_config is not None:
                completion_guided_json = _safe_dict(
                    _safe_dict(text_config.get("format")).get("schema")
                )
            return self._completions(
                messages=messages,
                n=n,
                seed=seed,
                guided_json=completion_guided_json,
                max_completion_tokens=structured_max_tokens,
                disable_hidden_thinking=disable_hidden_thinking,
            )

        choices = self._chat(
            messages,
            n=n,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            guided_json=guided_json,
            use_reasoning=False,
            max_completion_tokens=structured_max_tokens,
            disable_hidden_thinking=disable_hidden_thinking,
        )
        if self._model_is_gpt_oss():
            fallback_guided_json = guided_json
            if fallback_guided_json is None and tools:
                first_tool = tools[0] if len(tools) == 1 else {}
                fallback_guided_json = _safe_dict(
                    _safe_dict(first_tool.get("function")).get("parameters")
                )
            if tools and len(tools) != 1:
                fallback_guided_json = guided_json
            for index, choice in enumerate(choices):
                if _choice_has_visible_output(choice):
                    continue
                try:
                    fallback_choices = self._completions(
                        messages=messages,
                        n=1,
                        seed=seed + index,
                        guided_json=fallback_guided_json,
                        max_completion_tokens=structured_max_tokens,
                        disable_hidden_thinking=disable_hidden_thinking,
                    )
                except Exception as error:
                    choice["fallback_error"] = f"completions_fallback_failed: {error}"
                    continue
                fallback_choice = fallback_choices[0]
                if _choice_has_visible_output(fallback_choice):
                    fallback_choice["fallback_backend"] = "completions"
                    fallback_choice["fallback_trigger"] = "chat_completions_blank_visible_output"
                    choices[index] = fallback_choice
        return choices

    def _generate_actor_reasoning(
        self,
        context: SharedPrefixContext,
        *,
        task_row: dict[str, Any],
        step_index: int,
        seed: int,
        environment: RuntimeEnvironment | None = None,
    ) -> tuple[str, list[str]]:
        del task_row
        user_prompt = json.dumps(
            _actor_prompt_payload(
                context,
                step_index=step_index,
                environment=environment,
            ),
            indent=2,
            sort_keys=True,
        )
        choice = self._chat(
            messages=[
                {"role": "system", "content": ACTOR_RATIONALE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            n=1,
            seed=seed,
            guided_json=_actor_reasoning_text_schema(),
            use_reasoning=True,
            max_completion_tokens=self.config.actor_rationale_max_completion_tokens,
            disable_hidden_thinking=self._should_disable_hidden_thinking(),
        )[0]
        try:
            payload, payload_errors = _named_tool_arguments_from_choice(
                choice,
                expected_tool_name="",
                prefix="actor_rationale",
            )
        except Exception as error:
            return "", [f"actor_rationale_json_parse_error: {error}"]

        errors = list(payload_errors)
        response_error = _response_truncated_before_visible_output(
            payload,
            prefix="actor_rationale",
        )
        if response_error:
            errors.append(response_error)

        reasoning_text = _safe_text(payload.get("reasoning_text")).strip()
        if not reasoning_text:
            reasoning_text = _safe_text(payload.get("reasoning_content")).strip()
        if not reasoning_text:
            errors.append("actor_rationale_text_missing")
        return reasoning_text, _unique(errors)

    def generate_actor_candidates(
        self,
        context: SharedPrefixContext,
        *,
        task_row: dict[str, Any],
        step_index: int,
        n_act: int,
        seed: int,
        environment: RuntimeEnvironment | None = None,
        force_tool_coverage: bool = False,
    ) -> list[dict[str, Any]]:
        if self.api_mode == "responses":
            system_prompt = (
                ACTOR_SYSTEM_PROMPT
                + "\nKeep the visible reasoning concise. Use at most one native runtime tool call."
            )
            tools = _runtime_tools()
            tool_choice = None
            guided_json = None
            text_config = None
        elif self.api_mode == "chat_completions":
            system_prompt = (
                ACTOR_SYSTEM_PROMPT
                + "\nKeep the visible reasoning concise. Use at most one native runtime tool call."
            )
            tools = _runtime_tools()
            tool_choice = "auto"
            guided_json = None
            text_config = None
        else:
            system_prompt = (
                ACTOR_SYSTEM_PROMPT
                + "\nWrite free-form reasoning. If a tool is needed, end with one TOOL_ACTION line."
            )
            tools = None
            tool_choice = None
            guided_json = None
            text_config = None

        def build_user_prompt(sample_index: int | None = None) -> str:
            directive = None
            if force_tool_coverage and sample_index is not None:
                directive = _actor_tool_coverage_directive_payload(
                    sample_index=sample_index,
                    task_type=task_row.get("task_type"),
                )
            elif sample_index is not None:
                directive = _actor_sampling_directive_payload(
                    sample_index=sample_index,
                    task_type=task_row.get("task_type"),
                )
            return json.dumps(
                _actor_prompt_payload(
                    context,
                    step_index=step_index,
                    environment=environment,
                    actor_sampling_directive=directive,
                ),
                indent=2,
                sort_keys=True,
            )

        candidates: list[dict[str, Any]] = []
        if force_tool_coverage or (self.config.actor_sampling_strategy == "verbalized" and n_act > 1):
            choices: list[dict[str, Any]] = []
            for sample_index in range(n_act):
                directive_payload = (
                    _actor_tool_coverage_directive_payload(
                        sample_index=sample_index,
                        task_type=task_row.get("task_type"),
                    )
                    if force_tool_coverage
                    else _actor_sampling_directive_payload(
                        sample_index=sample_index,
                        task_type=task_row.get("task_type"),
                    )
                )
                try:
                    sample_choices = self._generate_choices(
                        system_prompt=system_prompt,
                        user_prompt=build_user_prompt(sample_index),
                        n=1,
                        seed=seed + sample_index,
                        tools=tools,
                        tool_choice=tool_choice,
                        guided_json=guided_json,
                        text_config=text_config,
                        disable_hidden_thinking=self._should_disable_hidden_thinking(),
                    )
                except requests.RequestException as error:
                    choices.append(
                        {
                            "finish_reason": None,
                            "message": {
                                "content": _request_error_raw_text("actor_request_failed", error),
                            },
                            "actor_sampling_directive": directive_payload,
                            "actor_request_error": f"actor_request_failed: {error}",
                        }
                    )
                    continue
                for choice in sample_choices:
                    choice["actor_sampling_directive"] = directive_payload
                choices.extend(sample_choices[:1])
        else:
            try:
                choices = self._generate_choices(
                    system_prompt=system_prompt,
                    user_prompt=build_user_prompt(),
                    n=n_act,
                    seed=seed,
                    tools=tools,
                    tool_choice=tool_choice,
                    guided_json=guided_json,
                    text_config=text_config,
                    disable_hidden_thinking=self._should_disable_hidden_thinking(),
                )
            except requests.RequestException as error:
                return [
                    {
                        "reasoning_text": "",
                        "tool_action": None,
                        "raw_text": _request_error_raw_text("actor_request_failed", error),
                        "generator_errors": [f"actor_request_failed: {error}"],
                    }
                ]

        for choice in choices:
            raw_text = _json_dumps_compact(_strip_raw_generation_payload(choice))
            try:
                normalized_payload, payload_errors = _actor_candidate_from_choice(choice)
                candidate_reasoning_text = normalized_payload["reasoning_text"]
                request_error = choice.get("actor_request_error")
                if isinstance(request_error, str) and request_error:
                    payload_errors.append(request_error)
                candidates.append(
                    {
                        "reasoning_text": candidate_reasoning_text,
                        "tool_action": normalized_payload["tool_action"],
                        "raw_text": raw_text,
                        "generator_errors": _unique(payload_errors),
                        "actor_sampling_directive": choice.get("actor_sampling_directive"),
                    }
                )
            except Exception as error:
                candidates.append(
                    {
                        "reasoning_text": "",
                        "tool_action": None,
                        "raw_text": raw_text,
                        "generator_errors": [f"actor_json_parse_error: {error}"],
                        "actor_sampling_directive": choice.get("actor_sampling_directive"),
                    }
                )
        return candidates

    def generate_verifier_candidates(
        self,
        context: SharedPrefixContext,
        *,
        task_row: dict[str, Any],
        actor_candidate: dict[str, Any],
        actor_step: ActorStep,
        observation: ToolObservation | None,
        step_index: int,
        n_ver: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        user_prompt = json.dumps(
            _verifier_prompt_payload(
                context,
                actor_step=actor_step,
                observation=observation,
                step_index=step_index,
                task_type=task_row.get("task_type"),
            ),
            indent=2,
            sort_keys=True,
        )
        if self.api_mode == "responses":
            system_prompt = (
                VERIFIER_SYSTEM_PROMPT
                + "\nDo not emit analysis or commentary."
                + "\nPut the entire reply in the final answer as exactly one JSON object that matches the provided schema."
            )
            tools = None
            tool_choice = None
            guided_json = None
            text_config = _responses_json_schema_text_config(
                name=VERIFIER_OUTPUT_TOOL_NAME,
                description="Emit the verifier policy update for the current MENTOR-RL branch.",
                schema=_verifier_output_schema(),
            )
        elif self._prefers_named_output_tools():
            system_prompt = (
                VERIFIER_SYSTEM_PROMPT
                + "\nDo not emit analysis, commentary, or hidden chain-of-thought."
                + f"\nCall `{VERIFIER_OUTPUT_TOOL_NAME}` immediately with arguments that match its schema."
            )
            tools = [_verifier_output_tool()]
            tool_choice = _named_function_tool_choice(VERIFIER_OUTPUT_TOOL_NAME)
            guided_json = None
            text_config = None
        else:
            system_prompt = (
                VERIFIER_SYSTEM_PROMPT
                + "\nReturn exactly one JSON object that matches the provided schema."
            )
            tools = None
            tool_choice = None
            guided_json = _verifier_output_schema()
            text_config = None
        candidates: list[dict[str, Any]] = []
        try:
            choices = self._generate_choices(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                n=n_ver,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                guided_json=guided_json,
                text_config=text_config,
                disable_hidden_thinking=self._should_disable_hidden_thinking(),
            )
        except requests.RequestException as error:
            return [
                {
                    "payload": {},
                    "raw_text": _request_error_raw_text("verifier_request_failed", error),
                    "generator_errors": list(actor_candidate.get("generator_errors", []))
                    + [f"verifier_request_failed: {error}"],
                }
            ]

        for choice in choices:
            raw_text = _json_dumps_compact(_strip_raw_generation_payload(choice))
            try:
                payload, payload_errors = _named_tool_arguments_from_choice(
                    choice,
                    expected_tool_name=VERIFIER_OUTPUT_TOOL_NAME,
                    prefix="verifier",
                    require_tool_call=(
                        self._prefers_named_output_tools()
                        and choice.get("fallback_backend") != "completions"
                    ),
                )
                candidates.append(
                    {
                        "payload": payload,
                        "raw_text": raw_text,
                        "generator_errors": list(actor_candidate.get("generator_errors", []))
                        + payload_errors
                        + _validate_verifier_payload(payload),
                    }
                )
            except Exception as error:
                candidates.append(
                    {
                        "payload": {},
                        "raw_text": raw_text,
                        "generator_errors": list(actor_candidate.get("generator_errors", []))
                        + [f"verifier_json_parse_error: {error}"],
                    }
                )
        return candidates


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _write_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")


def _load_task_rows(
    tasks_path: Path,
    *,
    max_tasks: int | None = None,
    task_shard_index: int = 0,
    task_shard_count: int = 1,
) -> list[dict[str, Any]]:
    if task_shard_count <= 0:
        raise ValueError("task_shard_count must be positive.")
    if task_shard_index < 0 or task_shard_index >= task_shard_count:
        raise ValueError("task_shard_index must be between 0 and task_shard_count - 1.")

    rows: list[dict[str, Any]] = []
    input_task_index = 0
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if input_task_index % task_shard_count != task_shard_index:
                input_task_index += 1
                continue
            rows.append(json.loads(line))
            input_task_index += 1
            if max_tasks is not None and len(rows) >= max_tasks:
                break
    return rows


def _gene_symbol_lookup(task_row: dict[str, Any]) -> dict[str, str]:
    visible_inputs = task_row["visible_inputs"]
    return {
        gene_id: symbol
        for gene_id, symbol in zip(
            visible_inputs.get("seed_gene_ids", []),
            visible_inputs.get("seed_gene_symbols", []),
            strict=False,
        )
    }


def _symbol_for_gene(gene_id: str, symbol_lookup: dict[str, str]) -> str:
    return symbol_lookup.get(gene_id, gene_id)


def _build_gene_group(
    gene_ids: list[str],
    *,
    symbol_lookup: dict[str, str],
    group_id: str = "group_0",
    rationale: str = "",
) -> GeneGroup:
    return GeneGroup(
        group_id=group_id,
        gene_ids=gene_ids,
        gene_symbols=[_symbol_for_gene(gene_id, symbol_lookup) for gene_id in gene_ids],
        rationale=rationale,
    )


def _flatten_predicted_gene_ids(state: Any) -> list[str]:
    gene_ids: list[str] = []
    for group in state.predicted_groups:
        gene_ids.extend(group.gene_ids)
    return _unique(gene_ids)


def _current_gene_ids(task_row: dict[str, Any], state: Any) -> list[str]:
    current = _flatten_predicted_gene_ids(state)
    if current:
        return current
    return list(task_row["visible_inputs"].get("seed_gene_ids", []))


def _build_visible_mechanistic_labels(task_row: dict[str, Any], *, limit: int = 2) -> list[MechanisticLabel]:
    visible_inputs = task_row["visible_inputs"]
    structured_annotations = visible_inputs.get("structured_annotations")
    if not isinstance(structured_annotations, dict):
        return []

    labels: list[MechanisticLabel] = []
    for go_term in structured_annotations.get("go_terms", [])[:limit]:
        if not isinstance(go_term, dict):
            continue
        go_id = go_term.get("go_id")
        go_name = go_term.get("go_name")
        if go_name:
            labels.append(
                MechanisticLabel(
                    label_source=LabelSource.GO,
                    label_id=go_id,
                    label_name=go_name,
                    evidence_ids=[],
                )
            )

    remaining = max(0, limit - len(labels))
    if remaining:
        fcgs_ids = structured_annotations.get("fcgs_ids", [])
        fcgs_names = structured_annotations.get("fcgs_names", [])
        for fcgs_id, fcgs_name in zip(fcgs_ids, fcgs_names, strict=False):
            if not fcgs_name:
                continue
            labels.append(
                MechanisticLabel(
                    label_source=LabelSource.FCGS,
                    label_id=fcgs_id,
                    label_name=fcgs_name,
                    evidence_ids=[],
                )
            )
            if len(labels) >= limit:
                break

    return labels


def _summarize_observation(observation: ToolObservation | None) -> tuple[str, list[str]]:
    if observation is None:
        return "No tool observation was used for this branch.", []

    tool_name = observation.provenance.get("tool_name")
    payload = observation.payload or {}
    if observation.status == ToolObservationStatus.EMPTY:
        return f"{tool_name} returned no supporting evidence.", []
    if observation.status in (ToolObservationStatus.INVALID, ToolObservationStatus.ERROR):
        return f"{tool_name} failed: {observation.error}", []

    if tool_name == "get_neighbors":
        neighbors = payload.get("unique_neighbors", [])
        return (
            f"Found {len(neighbors)} unique neighbors for {payload.get('query_gene_id')}.",
            neighbors[:10],
        )
    if tool_name == "induce_subgraph":
        present_gene_ids = payload.get("present_gene_ids", [])
        return (
            f"Observed {payload.get('combined_edge_count', 0)} edges among {len(present_gene_ids)} queried genes.",
            present_gene_ids[:10],
        )
    if tool_name == "shortest_path":
        path_gene_ids = payload.get("path_gene_ids", [])
        if not path_gene_ids:
            return ("No path was found between the queried genes.", [])
        return (
            f"Found a path of {payload.get('hop_count')} hops between the queried genes.",
            path_gene_ids[:10],
        )
    if tool_name in {"rwr_multiplex", "rwr_monoplex"}:
        ranked_gene_ids = [item.get("gene_id") for item in payload.get("results", []) if item.get("gene_id")]
        return (
            f"Ranked {len(ranked_gene_ids)} genes from the seed set with restart walk.",
            ranked_gene_ids[:10],
        )
    if tool_name == "query_mygene":
        return (
            f"Retrieved {payload.get('result_count', 0)} MyGene hits for {payload.get('query')}.",
            [],
        )

    return (f"Recorded a {tool_name} observation.", [])


def _build_evidence_record(
    observation: ToolObservation | None,
    *,
    step_index: int,
    branch_id: str,
    symbol_lookup: dict[str, str],
) -> EvidenceRecord | None:
    if observation is None:
        return None
    if observation.status in (ToolObservationStatus.INVALID, ToolObservationStatus.ERROR):
        return None

    summary, supporting_gene_ids = _summarize_observation(observation)
    return EvidenceRecord(
        evidence_id=f"{branch_id}.evidence",
        source_type=EvidenceSourceType.TOOL_OBSERVATION,
        summary=summary,
        provenance={
            "step_index": step_index,
            **observation.provenance,
        },
        supporting_gene_ids=supporting_gene_ids,
        supporting_gene_symbols=[_symbol_for_gene(gene_id, symbol_lookup) for gene_id in supporting_gene_ids],
        tool_call_id=observation.call_id,
    )


def _degree_supported_gene_ids(payload: dict[str, Any]) -> list[str]:
    degree_counts: dict[str, int] = {}
    for layer_payload in payload.get("layers", []):
        if not isinstance(layer_payload, dict):
            continue
        for edge in layer_payload.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source = edge.get("source_gene_id")
            target = edge.get("target_gene_id")
            if source:
                degree_counts[source] = degree_counts.get(source, 0) + 1
            if target:
                degree_counts[target] = degree_counts.get(target, 0) + 1
    supported = [gene_id for gene_id, degree in degree_counts.items() if degree > 0]
    supported.sort()
    return supported


def _positive_group_update(
    task_type: str,
    current_gene_ids: list[str],
    visible_seed_ids: list[str],
    observation: ToolObservation | None,
    *,
    conservative: bool,
) -> tuple[list[str], RelationshipStatus]:
    if observation is None or observation.status in (
        ToolObservationStatus.INVALID,
        ToolObservationStatus.ERROR,
    ):
        return current_gene_ids, RelationshipStatus.UNKNOWN

    payload = observation.payload or {}
    tool_name = observation.provenance.get("tool_name")

    if tool_name == "rwr_multiplex" or tool_name == "rwr_monoplex":
        ranked_gene_ids = [item.get("gene_id") for item in payload.get("results", []) if item.get("gene_id")]
        if task_type == "recovery":
            add_limit = 1 if conservative else 2
            additions = [gene_id for gene_id in ranked_gene_ids if gene_id not in current_gene_ids][:add_limit]
            new_gene_ids = _unique(current_gene_ids + additions)
            status = (
                RelationshipStatus.PARTIALLY_OBSERVED_GROUP
                if additions
                else RelationshipStatus.VALIDATED_GROUP
            )
            return new_gene_ids, status

        if task_type == "refinement":
            top_window = set(ranked_gene_ids[: max(2, len(current_gene_ids))])
            retained = [gene_id for gene_id in current_gene_ids if gene_id in top_window]
            if not retained:
                retained = current_gene_ids[:]
            if conservative and len(retained) < 2:
                retained = current_gene_ids[:]
            status = (
                RelationshipStatus.VALIDATED_GROUP
                if retained != current_gene_ids
                else RelationshipStatus.PARTIALLY_OBSERVED_GROUP
            )
            return retained, status

        return current_gene_ids, RelationshipStatus.VALIDATED_GROUP

    if tool_name == "get_neighbors":
        neighbors = [gene_id for gene_id in payload.get("unique_neighbors", []) if gene_id]
        if task_type == "recovery":
            additions = [gene_id for gene_id in neighbors if gene_id not in current_gene_ids][: 1 if conservative else 2]
            new_gene_ids = _unique(current_gene_ids + additions)
            status = (
                RelationshipStatus.PARTIALLY_OBSERVED_GROUP
                if additions
                else RelationshipStatus.UNKNOWN
            )
            return new_gene_ids, status

        if task_type == "refinement":
            retained = [
                gene_id
                for gene_id in current_gene_ids
                if gene_id == current_gene_ids[0] or gene_id in neighbors
            ]
            if len(retained) < 2 or conservative:
                retained = current_gene_ids[:]
            status = (
                RelationshipStatus.VALIDATED_GROUP
                if retained != current_gene_ids
                else RelationshipStatus.UNKNOWN
            )
            return retained, status

        overlap = set(neighbors) & set(current_gene_ids)
        status = RelationshipStatus.VALIDATED_GROUP if overlap else RelationshipStatus.UNKNOWN
        return current_gene_ids, status

    if tool_name == "shortest_path":
        path_gene_ids = [gene_id for gene_id in payload.get("path_gene_ids", []) if gene_id]
        if not path_gene_ids:
            return current_gene_ids, RelationshipStatus.UNKNOWN
        additions = [gene_id for gene_id in path_gene_ids if gene_id not in current_gene_ids]
        if task_type == "recovery" and additions and not conservative:
            current_gene_ids = _unique(current_gene_ids + additions[:1])
        status = (
            RelationshipStatus.VALIDATED_GROUP
            if payload.get("hop_count") is not None and payload.get("hop_count") <= 2
            else RelationshipStatus.PARTIALLY_OBSERVED_GROUP
        )
        return current_gene_ids, status

    if tool_name == "induce_subgraph":
        edge_count = int(payload.get("combined_edge_count", 0))
        if task_type == "refinement":
            supported_gene_ids = _degree_supported_gene_ids(payload)
            if supported_gene_ids and len(supported_gene_ids) >= 2 and not conservative:
                return supported_gene_ids, RelationshipStatus.VALIDATED_GROUP
        if edge_count > 0 and len(current_gene_ids) >= 2:
            return current_gene_ids, RelationshipStatus.VALIDATED_GROUP
        return current_gene_ids, RelationshipStatus.UNKNOWN

    if tool_name == "query_mygene":
        return visible_seed_ids or current_gene_ids, RelationshipStatus.PARTIALLY_OBSERVED_GROUP

    return current_gene_ids, RelationshipStatus.UNKNOWN


def _none_group_update(
    current_gene_ids: list[str],
    observation: ToolObservation | None,
    *,
    abstain: bool,
) -> tuple[list[str], RelationshipStatus]:
    if abstain:
        return [], RelationshipStatus.INSUFFICIENT_SUPPORT

    if observation is None or observation.status in (
        ToolObservationStatus.INVALID,
        ToolObservationStatus.ERROR,
    ):
        return current_gene_ids, RelationshipStatus.UNKNOWN

    payload = observation.payload or {}
    tool_name = observation.provenance.get("tool_name")
    if observation.status == ToolObservationStatus.EMPTY:
        return [], RelationshipStatus.INSUFFICIENT_SUPPORT

    if tool_name == "induce_subgraph":
        if int(payload.get("combined_edge_count", 0)) == 0:
            return [], RelationshipStatus.INSUFFICIENT_SUPPORT
        return current_gene_ids, RelationshipStatus.PARTIALLY_OBSERVED_GROUP

    if tool_name == "shortest_path":
        if payload.get("hop_count") is None:
            return [], RelationshipStatus.INSUFFICIENT_SUPPORT
        return current_gene_ids, RelationshipStatus.PARTIALLY_OBSERVED_GROUP

    if tool_name == "get_neighbors":
        neighbors = payload.get("unique_neighbors", [])
        if not neighbors:
            return [], RelationshipStatus.INSUFFICIENT_SUPPORT
        return current_gene_ids, RelationshipStatus.PARTIALLY_OBSERVED_GROUP

    return current_gene_ids, RelationshipStatus.UNKNOWN


def _continuation_for_branch(
    task_type: str,
    *,
    actor_template_id: str,
    verifier_template_id: str,
    prior_gene_ids: list[str],
    updated_gene_ids: list[str],
    relationship_status: RelationshipStatus,
    visible_labels: list[MechanisticLabel],
    observation: ToolObservation | None,
    remaining_budget: int,
) -> tuple[ContinuationState, TerminationReason | None]:
    if remaining_budget <= 0:
        return ContinuationState.STOP, TerminationReason.BUDGET_EXHAUSTED

    if actor_template_id == "stop_with_current_evidence":
        return ContinuationState.STOP, TerminationReason.MODEL_STOP

    if task_type == "none" and relationship_status == RelationshipStatus.INSUFFICIENT_SUPPORT:
        return ContinuationState.STOP, TerminationReason.MODEL_STOP

    if task_type == "explanation" and (
        visible_labels or relationship_status == RelationshipStatus.VALIDATED_GROUP
    ):
        return ContinuationState.STOP, TerminationReason.MODEL_STOP

    if task_type in {"recovery", "refinement"} and relationship_status == RelationshipStatus.VALIDATED_GROUP:
        if updated_gene_ids == prior_gene_ids or verifier_template_id == "conservative":
            return ContinuationState.STOP, TerminationReason.MODEL_STOP

    if observation is not None and observation.status in (
        ToolObservationStatus.INVALID,
        ToolObservationStatus.ERROR,
    ):
        return ContinuationState.REVISE, None

    return ContinuationState.CONTINUE, None


def _render_interpretation(
    query_text: str,
    updated_gene_ids: list[str],
    relationship_status: RelationshipStatus,
    mechanistic_labels: list[MechanisticLabel],
    observation_summary: str,
    continuation_state: ContinuationState,
    *,
    symbol_lookup: dict[str, str],
) -> Interpretation:
    gene_symbols = [_symbol_for_gene(gene_id, symbol_lookup) for gene_id in updated_gene_ids[:6]]
    if relationship_status == RelationshipStatus.INSUFFICIENT_SUPPORT:
        mechanistic_claim = "Current evidence does not support a single shared mechanism."
    elif mechanistic_labels:
        mechanistic_claim = f"Current evidence is consistent with {mechanistic_labels[0].label_name}."
    elif gene_symbols:
        mechanistic_claim = "Current evidence supports a shared group containing " + ", ".join(gene_symbols) + "."
    else:
        mechanistic_claim = "Current evidence remains inconclusive."

    if continuation_state == ContinuationState.STOP:
        uncertainty = ""
        next_subgoal = ""
    elif continuation_state == ContinuationState.REVISE:
        uncertainty = "The last action did not provide enough clean evidence."
        next_subgoal = query_text
    else:
        uncertainty = "Additional graph evidence could still change the current hypothesis."
        next_subgoal = query_text

    return Interpretation(
        mechanistic_claim=mechanistic_claim,
        main_evidence=observation_summary,
        uncertainty=uncertainty,
        next_subgoal=next_subgoal,
    )


def _render_finding_text(branch: CandidateBranch) -> str:
    score = branch.local_score.total_score
    relationship = branch.verifier_step.updated_state.relationship_status.value
    summary = branch.verifier_step.updated_interpretation.main_evidence
    return f"[{relationship}] score={score:.4f} {summary}"


def _build_finding_record(
    *,
    task_row: dict[str, Any],
    trajectory_id: str,
    trajectory_seed: int,
    step_index: int,
    context: SharedPrefixContext,
    branch: CandidateBranch,
    finding_text: str,
) -> dict[str, Any]:
    tool_action = branch.actor_step.tool_action
    observation = branch.observation
    return {
        "finding_id": f"{trajectory_id}.step{step_index}.finding",
        "trajectory_id": trajectory_id,
        "trajectory_seed": trajectory_seed,
        "source_task_id": task_row["task_id"],
        "task_type": task_row["task_type"],
        "difficulty": task_row.get("difficulty"),
        "evidence_mode": task_row.get("evidence_mode"),
        "step_index": step_index,
        "shared_prefix_context": context.to_dict(),
        "chosen_branch_id": branch.branch_id,
        "finding_text": finding_text,
        "actor_reasoning_text": branch.actor_step.reasoning_text,
        "tool_provenance": {
            "tool_name": tool_action.tool_name if tool_action is not None else None,
            "arguments": tool_action.arguments if tool_action is not None else None,
            "call_id": tool_action.call_id if tool_action is not None else None,
            "observation_status": observation.status.value if observation is not None else None,
            "observation": observation.to_dict() if observation is not None else None,
        },
        "updated_interpretation": branch.verifier_step.updated_interpretation.to_dict(),
        "updated_state": branch.verifier_step.updated_state.to_dict(),
    }


def _pair_category(
    *,
    task_type: str,
    context: SharedPrefixContext,
    chosen_branch: CandidateBranch,
    rejected_branch: CandidateBranch,
) -> str:
    chosen_features = _branch_quality_features(chosen_branch, prior_state=context.state)
    rejected_features = _branch_quality_features(rejected_branch, prior_state=context.state)
    complex_diff = (
        chosen_branch.local_score.complex_membership_delta
        - rejected_branch.local_score.complex_membership_delta
    )
    mechanism_diff = (
        chosen_branch.local_score.mechanistic_label_delta
        - rejected_branch.local_score.mechanistic_label_delta
    )
    chosen_group_delta = int(chosen_features["group_size_delta"])
    rejected_group_delta = int(rejected_features["group_size_delta"])

    if (
        task_type == "recovery"
        and chosen_group_delta > 0
        and chosen_group_delta >= rejected_group_delta
        and complex_diff >= -1e-9
    ):
        return "recovery_expansion"
    if (
        task_type == "refinement"
        and float(chosen_features["precision_delta"]) > float(rejected_features["precision_delta"]) + 1e-9
    ):
        return "refinement_precision"
    if (
        task_type == "none"
        and chosen_branch.verifier_step.updated_state.relationship_status
        == RelationshipStatus.INSUFFICIENT_SUPPORT
    ):
        return "none_abstention"
    if bool(chosen_features["has_successful_tool"]) and (
        not bool(rejected_features["has_successful_tool"]) or complex_diff > 1e-9
    ):
        return "tool_supported_improvement"
    if abs(complex_diff) <= 1e-9 and mechanism_diff > 1e-9:
        return "mechanism_label_only"
    if (
        task_type in {"recovery", "refinement"}
        and chosen_features["tool_name"] == "no_tool"
        and chosen_group_delta <= 0
    ):
        return "conservative_stop"
    return "score_margin"


def _pair_quality_provenance(
    *,
    task_row: dict[str, Any],
    context: SharedPrefixContext,
    branches: list[CandidateBranch],
    chosen_branch: CandidateBranch,
    rejected_branch: CandidateBranch,
    score_margin: float,
    pair_mining_strategy: str,
) -> dict[str, Any]:
    task_type = str(task_row["task_type"])
    chosen_features = _branch_quality_features(chosen_branch, prior_state=context.state)
    rejected_features = _branch_quality_features(rejected_branch, prior_state=context.state)
    return {
        "candidate_count": len(branches),
        "valid_candidate_count": sum(1 for branch in branches if _branch_is_usable_for_selection(branch)),
        "selected_branch_id": chosen_branch.branch_id,
        "score_margin_threshold": score_margin,
        "difficulty": task_row.get("difficulty"),
        "pair_mining_strategy": pair_mining_strategy,
        "pair_category": _pair_category(
            task_type=task_type,
            context=context,
            chosen_branch=chosen_branch,
            rejected_branch=rejected_branch,
        ),
        "chosen_tool_name": chosen_features["tool_name"],
        "rejected_tool_name": rejected_features["tool_name"],
        "chosen_has_successful_tool": chosen_features["has_successful_tool"],
        "rejected_has_successful_tool": rejected_features["has_successful_tool"],
        "chosen_gene_count": chosen_features["post_gene_count"],
        "rejected_gene_count": rejected_features["post_gene_count"],
        "chosen_group_size_delta": chosen_features["group_size_delta"],
        "rejected_group_size_delta": rejected_features["group_size_delta"],
        "chosen_recall_delta": chosen_features["recall_delta"],
        "rejected_recall_delta": rejected_features["recall_delta"],
        "chosen_precision_delta": chosen_features["precision_delta"],
        "rejected_precision_delta": rejected_features["precision_delta"],
        "complex_delta_diff": (
            chosen_branch.local_score.complex_membership_delta
            - rejected_branch.local_score.complex_membership_delta
        ),
        "mechanistic_delta_diff": (
            chosen_branch.local_score.mechanistic_label_delta
            - rejected_branch.local_score.mechanistic_label_delta
        ),
        "efficiency_penalty_diff": (
            chosen_branch.local_score.efficiency_penalty
            - rejected_branch.local_score.efficiency_penalty
        ),
    }


def _preference_difficulty_for_rank(index: int, total: int) -> PreferenceDifficulty:
    if total <= 1:
        return PreferenceDifficulty.MEDIUM
    if index == 0:
        return PreferenceDifficulty.EASY
    if index == total - 1:
        return PreferenceDifficulty.HARD
    return PreferenceDifficulty.MEDIUM


def _mine_preference_pairs(
    *,
    task_row: dict[str, Any],
    trajectory_id: str,
    trajectory_seed: int,
    step_index: int,
    context: SharedPrefixContext,
    branches: list[CandidateBranch],
    chosen_branch: CandidateBranch,
    score_margin: float,
    pair_mining_strategy: str = "score_margin",
) -> list[PreferencePair]:
    if not branches:
        return []
    if not _branch_is_usable_for_selection(chosen_branch):
        return []

    chosen_normalized = float(chosen_branch.local_score.normalized_score or 0.0)
    ordered_rejected = sorted(
        (
            branch
            for branch in branches
            if branch.branch_id != chosen_branch.branch_id
            and _branch_is_usable_for_selection(branch)
            and (chosen_normalized - float(branch.local_score.normalized_score or 0.0)) >= score_margin
        ),
        key=lambda branch: (
            float(branch.local_score.normalized_score or 0.0),
            branch.local_score.total_score,
            branch.branch_id,
        ),
    )
    if not ordered_rejected:
        return []

    if pair_mining_strategy == "quality_balanced":
        difficulty_targets = [
            (_preference_difficulty_for_rank(index, len(ordered_rejected)), branch)
            for index, branch in enumerate(ordered_rejected)
        ]
    else:
        difficulty_targets = [
            (PreferenceDifficulty.EASY, ordered_rejected[0]),
            (PreferenceDifficulty.MEDIUM, ordered_rejected[len(ordered_rejected) // 2]),
            (PreferenceDifficulty.HARD, ordered_rejected[-1]),
        ]
    pairs: list[PreferencePair] = []
    seen_branch_ids: set[str] = set()
    for difficulty_bin, rejected_branch in difficulty_targets:
        if rejected_branch.branch_id in seen_branch_ids:
            continue
        seen_branch_ids.add(rejected_branch.branch_id)
        rejected_normalized = float(rejected_branch.local_score.normalized_score or 0.0)
        pair = PreferencePair(
            pair_id=(
                f"{trajectory_id}.step{step_index}.pref."
                f"{difficulty_bin.value}.{rejected_branch.branch_id}"
            ),
            context=SharedPrefixContext.from_dict(context.to_dict()),
            chosen=CandidateBranch.from_dict(chosen_branch.to_dict()),
            rejected=CandidateBranch.from_dict(rejected_branch.to_dict()),
            task_type=TaskType(task_row["task_type"]),
            difficulty_bin=difficulty_bin,
            decision_step=step_index,
            raw_score_chosen=chosen_branch.local_score.total_score,
            raw_score_rejected=rejected_branch.local_score.total_score,
            normalized_score_chosen=chosen_normalized,
            normalized_score_rejected=rejected_normalized,
            score_margin=chosen_normalized - rejected_normalized,
            source_task_id=task_row["task_id"],
            trajectory_id=trajectory_id,
            trajectory_seed=trajectory_seed,
            evidence_mode=task_row.get("evidence_mode"),
            provenance=_pair_quality_provenance(
                task_row=task_row,
                context=context,
                branches=branches,
                chosen_branch=chosen_branch,
                rejected_branch=rejected_branch,
                score_margin=score_margin,
                pair_mining_strategy=pair_mining_strategy,
            ),
        )
        pairs.append(pair)
    return pairs


def _pair_category_from_provenance(pair: PreferencePair) -> str:
    category = pair.provenance.get("pair_category")
    return category if isinstance(category, str) and category else "score_margin"


def _round_robin_quality_pairs(pairs: list[PreferencePair]) -> list[PreferencePair]:
    buckets: dict[tuple[int, str, str], list[PreferencePair]] = {}
    for pair in sorted(
        pairs,
        key=lambda item: (
            item.source_task_id,
            item.trajectory_seed,
            item.decision_step,
            -item.score_margin,
            item.pair_id,
        ),
    ):
        category = _pair_category_from_provenance(pair)
        key = (
            PAIR_CATEGORY_PRIORITIES.get(category, PAIR_CATEGORY_PRIORITIES["score_margin"]),
            category,
            str(pair.evidence_mode),
        )
        buckets.setdefault(key, []).append(pair)

    ordered: list[PreferencePair] = []
    keys = sorted(buckets)
    while keys:
        next_keys: list[tuple[int, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _select_quality_balanced_bucket(bucket: list[PreferencePair], target_size: int) -> list[PreferencePair]:
    if target_size <= 0:
        return []
    ordered = _round_robin_quality_pairs(bucket)
    selected: list[PreferencePair] = []
    deferred: list[PreferencePair] = []
    weak_limit = max(1, int(math.ceil(target_size * 0.20)))
    weak_counts: Counter[str] = Counter()
    weak_categories = {"mechanism_label_only", "conservative_stop"}

    for pair in ordered:
        category = _pair_category_from_provenance(pair)
        if category in weak_categories and weak_counts[category] >= weak_limit:
            deferred.append(pair)
            continue
        selected.append(pair)
        if category in weak_categories:
            weak_counts[category] += 1
        if len(selected) >= target_size:
            return selected

    for pair in deferred:
        selected.append(pair)
        if len(selected) >= target_size:
            break
    return selected


def _balance_preference_pairs(
    pairs: list[PreferencePair],
    *,
    pair_mining_strategy: str = "score_margin",
) -> list[PreferencePair]:
    if not pairs:
        return []

    buckets: dict[tuple[str, str], list[PreferencePair]] = {}
    for pair in pairs:
        key = (pair.task_type.value, pair.difficulty_bin.value)
        buckets.setdefault(key, []).append(pair)

    min_bucket_size = min(len(bucket) for bucket in buckets.values())
    balanced: list[PreferencePair] = []
    for key in sorted(buckets):
        if pair_mining_strategy == "quality_balanced":
            balanced.extend(_select_quality_balanced_bucket(buckets[key], min_bucket_size))
        else:
            bucket = sorted(
                buckets[key],
                key=lambda pair: (
                    pair.source_task_id,
                    pair.trajectory_seed,
                    pair.decision_step,
                    pair.pair_id,
                ),
            )
            balanced.extend(bucket[:min_bucket_size])

    return sorted(
        balanced,
        key=lambda pair: (
            pair.source_task_id,
            pair.trajectory_seed,
            pair.decision_step,
            pair.difficulty_bin.value,
            pair.pair_id,
        ),
    )


def _render_final_summary(
    task_row: dict[str, Any],
    interpretation: Interpretation,
    state: Any,
    findings: list[str],
) -> str:
    predicted_gene_ids = _flatten_predicted_gene_ids(state)
    claim = interpretation.mechanistic_claim.strip()
    if not claim:
        if state.relationship_status == RelationshipStatus.INSUFFICIENT_SUPPORT:
            claim = "No single shared mechanism is supported."
        elif state.mechanistic_labels:
            claim = f"The best supported mechanism is {state.mechanistic_labels[0].label_name}."
        else:
            claim = (
                f"The current best hypothesis is {state.relationship_status.value} "
                f"for genes {predicted_gene_ids}."
            )

    supporting_findings = "; ".join(finding.strip() for finding in findings if finding.strip())
    if not supporting_findings:
        supporting_findings = interpretation.main_evidence.strip() or "No intermediate findings were recorded."

    stopping_condition = (
        state.termination_reason.value
        if state.termination_reason is not None
        else state.continuation_state.value
    )
    return (
        f"Claim: {claim} "
        f"Supporting findings: {supporting_findings} "
        f"Stopping condition: {stopping_condition}. "
        f"Query: {task_row['query_text']}"
    )


def _actor_templates_for_step(
    task_row: dict[str, Any],
    state: Any,
    *,
    trajectory_id: str,
    step_index: int,
    n_act: int,
    recovery_rwr_top_k: int = DEFAULT_RECOVERY_RWR_TOP_K,
) -> list[dict[str, Any]]:
    task_type = task_row["task_type"]
    visible_inputs = task_row["visible_inputs"]
    current_gene_ids = _current_gene_ids(task_row, state)
    templates: list[dict[str, Any]] = []

    def add_template(
        template_id: str,
        reasoning_text: str,
        *,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        if any(existing["template_id"] == template_id for existing in templates):
            return
        tool_action = None
        if tool_name is not None:
            tool_action = ToolAction(
                tool_name=tool_name,
                arguments=arguments or {},
                call_id=f"{trajectory_id}.step{step_index}.{template_id}",
            )
        templates.append(
            {
                "template_id": template_id,
                "actor_step": ActorStep(reasoning_text=reasoning_text, tool_action=tool_action),
            }
        )

    if visible_inputs.get("structured_annotations"):
        add_template(
            "use_visible_annotations",
            "Use the visible annotations to update the mechanistic interpretation before another tool call.",
        )

    if current_gene_ids:
        add_template(
            "induce_subgraph_current_group",
            "Inspect the induced subgraph on the current candidate group.",
            tool_name="induce_subgraph",
            arguments={"genes": current_gene_ids},
        )
        add_template(
            "neighbors_first_seed",
            "Inspect the neighborhood around the first current anchor gene.",
            tool_name="get_neighbors",
            arguments={"gene": current_gene_ids[0]},
        )
        if task_type in {"recovery", "refinement"}:
            add_template(
                "rwr_expand_group",
                "Rank candidate genes from the current seed set with multiplex restart walk.",
                tool_name="rwr_multiplex",
                arguments={
                    "seeds": current_gene_ids,
                    "top_k": recovery_rwr_top_k if task_type == "recovery" else 10,
                },
            )

    if len(current_gene_ids) >= 2:
        add_template(
            "shortest_path_seed_pair",
            "Check whether the first two genes are connected by a short path.",
            tool_name="shortest_path",
            arguments={"source": current_gene_ids[0], "target": current_gene_ids[1]},
        )

    add_template(
        "stop_with_current_evidence",
        "Stop and commit to the current interpretation.",
    )

    ordered_ids_by_task = {
        "recovery": [
            "rwr_expand_group",
            "induce_subgraph_current_group",
            "neighbors_first_seed",
            "shortest_path_seed_pair",
            "stop_with_current_evidence",
            "use_visible_annotations",
        ],
        "refinement": [
            "induce_subgraph_current_group",
            "rwr_expand_group",
            "neighbors_first_seed",
            "shortest_path_seed_pair",
            "stop_with_current_evidence",
            "use_visible_annotations",
        ],
        "explanation": [
            "use_visible_annotations",
            "induce_subgraph_current_group",
            "shortest_path_seed_pair",
            "neighbors_first_seed",
            "stop_with_current_evidence",
        ],
        "none": [
            "induce_subgraph_current_group",
            "shortest_path_seed_pair",
            "neighbors_first_seed",
            "stop_with_current_evidence",
        ],
    }
    ordered_ids = ordered_ids_by_task.get(task_type, [])
    order_lookup = {template_id: index for index, template_id in enumerate(ordered_ids)}
    templates.sort(key=lambda item: order_lookup.get(item["template_id"], len(order_lookup)))
    return templates[: max(1, n_act)]


def _verifier_template_ids(task_type: str, actor_template_id: str, *, n_ver: int) -> list[str]:
    if actor_template_id == "use_visible_annotations":
        return ["standard"][: max(1, n_ver)]
    if actor_template_id == "stop_with_current_evidence":
        return ["standard"][: max(1, n_ver)]
    if task_type == "none":
        ordered = ["abstain", "probe"]
    else:
        ordered = ["standard", "conservative"]
    return ordered[: max(1, n_ver)]


def _build_actor_step_from_model_candidate(
    candidate: dict[str, Any],
    *,
    trajectory_id: str,
    step_index: int,
    actor_index: int,
) -> tuple[ActorStep, list[str]]:
    errors: list[str] = []
    reasoning_text = candidate.get("reasoning_text")
    if not isinstance(reasoning_text, str):
        reasoning_text = ""
        errors.append("actor_reasoning_text_missing_or_invalid")

    tool_action_payload = candidate.get("tool_action")
    tool_action = None
    if tool_action_payload is not None:
        if not isinstance(tool_action_payload, dict):
            errors.append("actor_tool_action_not_a_dict")
        else:
            tool_name = tool_action_payload.get("tool_name")
            arguments = tool_action_payload.get("arguments")
            if not isinstance(tool_name, str) or not tool_name:
                errors.append("actor_tool_name_missing_or_invalid")
            else:
                arguments = normalize_tool_arguments(tool_name, _safe_dict(arguments))
                tool_action = ToolAction(
                    tool_name=tool_name,
                    arguments=arguments,
                    call_id=f"{trajectory_id}.step{step_index}.model_actor{actor_index}",
                )

    return ActorStep(reasoning_text=reasoning_text, tool_action=tool_action), errors


def _apply_task_tool_defaults(
    actor_step: ActorStep,
    *,
    task_type: str,
    recovery_rwr_top_k: int,
) -> dict[str, Any]:
    action = actor_step.tool_action
    if action is None:
        return {}
    if task_type == "recovery" and action.tool_name == "rwr_multiplex":
        old_top_k = action.arguments.get("top_k")
        if not isinstance(old_top_k, int) or old_top_k < recovery_rwr_top_k:
            action.arguments["top_k"] = recovery_rwr_top_k
            return {
                "tool_name": action.tool_name,
                "argument": "top_k",
                "old_value": old_top_k,
                "new_value": recovery_rwr_top_k,
                "reason": "recovery_expansion_requires_broad_non_seed_candidate_search",
            }
    return {}


def _build_labels_from_model_payload(payload: dict[str, Any]) -> tuple[list[MechanisticLabel], list[str]]:
    labels: list[MechanisticLabel] = []
    errors: list[str] = []
    for index, raw_label in enumerate(payload.get("mechanistic_labels", [])):
        if not isinstance(raw_label, dict):
            errors.append(f"verifier_label_{index}_not_a_dict")
            continue
        label_source = raw_label.get("label_source")
        label_name = raw_label.get("label_name")
        label_id = raw_label.get("label_id")
        if not isinstance(label_name, str) or not label_name:
            errors.append(f"verifier_label_{index}_missing_name")
            continue
        if not isinstance(label_source, str) or not label_source:
            label_source = LabelSource.FREE_TEXT.value
            errors.append(f"verifier_label_{index}_missing_source")
        try:
            labels.append(
                MechanisticLabel(
                    label_source=label_source,
                    label_name=label_name,
                    label_id=label_id if isinstance(label_id, str) else None,
                    evidence_ids=[],
                )
            )
        except Exception as error:
            errors.append(f"verifier_label_{index}_invalid: {error}")
    return labels, errors


def _build_branch_from_model_output(
    task_row: dict[str, Any],
    prior_state: Any,
    actor_step: ActorStep,
    observation: ToolObservation | None,
    verifier_candidate: dict[str, Any],
    *,
    branch_id: str,
    step_index: int,
    symbol_lookup: dict[str, str],
    generator_errors: list[str],
) -> CandidateBranch:
    payload = _safe_dict(verifier_candidate.get("payload"))
    updated_interpretation_payload = _safe_dict(payload.get("updated_interpretation"))
    updated_state_payload = _safe_dict(payload.get("updated_state"))
    errors = _unique(list(generator_errors) + list(verifier_candidate.get("generator_errors", [])))

    updated_state = clone_state(prior_state)
    updated_state = decrement_budget(updated_state)
    if actor_step.tool_action is not None:
        invalid_tool = observation is None or observation.status != ToolObservationStatus.SUCCESS
        updated_state = record_tool_call(updated_state, invalid=invalid_tool)

    evidence_record = _build_evidence_record(
        observation,
        step_index=step_index,
        branch_id=branch_id,
        symbol_lookup=symbol_lookup,
    )
    if evidence_record is not None:
        updated_state = append_evidence_record(updated_state, evidence_record)

    relationship_status_raw = updated_state_payload.get("relationship_status", RelationshipStatus.UNKNOWN.value)
    try:
        relationship_status = RelationshipStatus(relationship_status_raw)
    except Exception:
        relationship_status = RelationshipStatus.UNKNOWN
        errors.append("verifier_relationship_status_invalid")

    predicted_gene_ids = _safe_list_of_strings(updated_state_payload.get("predicted_gene_ids"))
    if task_row["task_type"] == "none" and relationship_status == RelationshipStatus.INSUFFICIENT_SUPPORT:
        predicted_gene_ids = []

    if predicted_gene_ids:
        updated_state = replace_predicted_groups(
            updated_state,
            [
                _build_gene_group(
                    predicted_gene_ids,
                    symbol_lookup=symbol_lookup,
                    rationale=actor_step.reasoning_text,
                )
            ],
            relationship_status=relationship_status,
        )
    else:
        updated_state = replace_predicted_groups(
            updated_state,
            [],
            relationship_status=relationship_status,
        )

    mechanistic_labels, label_errors = _build_labels_from_model_payload(updated_state_payload)
    errors.extend(label_errors)
    updated_state = replace_mechanistic_labels(updated_state, mechanistic_labels)

    continuation_raw = updated_state_payload.get(
        "continuation_decision",
        ContinuationState.REVISE.value if errors else ContinuationState.CONTINUE.value,
    )
    try:
        continuation_state = ContinuationState(continuation_raw)
    except Exception:
        continuation_state = ContinuationState.REVISE
        errors.append("verifier_continuation_decision_invalid")

    termination_reason = None
    if updated_state.remaining_budget <= 0:
        continuation_state = ContinuationState.STOP
        termination_reason = TerminationReason.BUDGET_EXHAUSTED
    elif continuation_state == ContinuationState.STOP:
        termination_reason = TerminationReason.MODEL_STOP
    updated_state = set_continuation_state(
        updated_state,
        continuation_state,
        termination_reason=termination_reason,
    )

    updated_interpretation = Interpretation(
        mechanistic_claim=_safe_text(updated_interpretation_payload.get("mechanistic_claim")),
        main_evidence=_safe_text(updated_interpretation_payload.get("main_evidence")),
        uncertainty=_safe_text(updated_interpretation_payload.get("uncertainty")),
        next_subgoal=_safe_text(updated_interpretation_payload.get("next_subgoal")),
    )

    verifier_notes = _safe_text(updated_state_payload.get("verifier_notes"))
    if not verifier_notes and "verifier_notes" in updated_state_payload and not isinstance(
        updated_state_payload.get("verifier_notes"),
        str,
    ):
        errors.append("verifier_notes_invalid")

    return CandidateBranch(
        branch_id=branch_id,
        actor_step=actor_step,
        observation=observation,
        verifier_step=VerifierStep(
            updated_interpretation=updated_interpretation,
            updated_state=updated_state,
            continuation_decision=continuation_state,
            verifier_notes=verifier_notes,
        ),
        local_score=LocalScoreBreakdown(
            schema_score=0.0,
            complex_membership_delta=0.0,
            mechanistic_label_delta=0.0,
            efficiency_penalty=0.0,
            total_score=0.0,
        ),
        metadata={
            "generator_backend": "model_vllm",
            "step_index": step_index,
            "task_type": task_row["task_type"],
            "generator_errors": errors,
        },
    )


def _build_heuristic_branch_for_templates(
    task_row: dict[str, Any],
    prior_interpretation: Interpretation,
    prior_state: Any,
    actor_template_id: str,
    actor_step: ActorStep,
    verifier_template_id: str,
    observation: ToolObservation | None,
    *,
    branch_id: str,
    step_index: int,
    max_steps: int,
    symbol_lookup: dict[str, str],
) -> CandidateBranch:
    del prior_interpretation
    current_gene_ids = _current_gene_ids(task_row, prior_state)
    visible_seed_ids = list(task_row["visible_inputs"].get("seed_gene_ids", []))

    updated_state = clone_state(prior_state)
    updated_state = decrement_budget(updated_state)
    if actor_step.tool_action is not None:
        invalid_tool = observation is None or observation.status != ToolObservationStatus.SUCCESS
        updated_state = record_tool_call(updated_state, invalid=invalid_tool)

    evidence_record = _build_evidence_record(
        observation,
        step_index=step_index,
        branch_id=branch_id,
        symbol_lookup=symbol_lookup,
    )
    if evidence_record is not None:
        updated_state = append_evidence_record(updated_state, evidence_record)

    visible_labels = _build_visible_mechanistic_labels(
        task_row,
        limit=1 if verifier_template_id in {"conservative", "probe"} else 2,
    )
    if visible_labels:
        updated_state = replace_mechanistic_labels(updated_state, visible_labels)

    if task_row["task_type"] == "none":
        updated_gene_ids, relationship_status = _none_group_update(
            current_gene_ids,
            observation,
            abstain=verifier_template_id == "abstain" or actor_template_id == "stop_with_current_evidence",
        )
    else:
        updated_gene_ids, relationship_status = _positive_group_update(
            task_row["task_type"],
            current_gene_ids,
            visible_seed_ids,
            observation,
            conservative=verifier_template_id == "conservative",
        )

    if actor_template_id == "use_visible_annotations" and task_row["task_type"] == "explanation":
        relationship_status = RelationshipStatus.VALIDATED_GROUP
    if actor_template_id == "stop_with_current_evidence" and task_row["task_type"] == "none":
        updated_gene_ids = []
        relationship_status = RelationshipStatus.INSUFFICIENT_SUPPORT

    if updated_gene_ids:
        updated_state = replace_predicted_groups(
            updated_state,
            [_build_gene_group(updated_gene_ids, symbol_lookup=symbol_lookup, rationale=actor_step.reasoning_text)],
            relationship_status=relationship_status,
        )
    else:
        updated_state = replace_predicted_groups(
            updated_state,
            [],
            relationship_status=relationship_status,
        )

    continuation_state, termination_reason = _continuation_for_branch(
        task_row["task_type"],
        actor_template_id=actor_template_id,
        verifier_template_id=verifier_template_id,
        prior_gene_ids=current_gene_ids,
        updated_gene_ids=updated_gene_ids,
        relationship_status=relationship_status,
        visible_labels=visible_labels,
        observation=observation,
        remaining_budget=updated_state.remaining_budget,
    )
    updated_state = set_continuation_state(
        updated_state,
        continuation_state,
        termination_reason=termination_reason,
    )

    observation_summary, _ = _summarize_observation(observation)
    updated_interpretation = _render_interpretation(
        task_row["query_text"],
        updated_gene_ids,
        relationship_status,
        updated_state.mechanistic_labels,
        observation_summary,
        continuation_state,
        symbol_lookup=symbol_lookup,
    )

    return CandidateBranch(
        branch_id=branch_id,
        actor_step=actor_step,
        observation=observation,
        verifier_step=VerifierStep(
            updated_interpretation=updated_interpretation,
            updated_state=updated_state,
            continuation_decision=continuation_state,
            verifier_notes=f"template={verifier_template_id}",
        ),
        local_score=LocalScoreBreakdown(
            schema_score=0.0,
            complex_membership_delta=0.0,
            mechanistic_label_delta=0.0,
            efficiency_penalty=0.0,
            total_score=0.0,
        ),
        metadata={
            "generator_backend": "heuristic",
            "actor_template_id": actor_template_id,
            "verifier_template_id": verifier_template_id,
            "step_index": step_index,
            "task_type": task_row["task_type"],
        },
    )


def _score_branch(
    task_row: dict[str, Any],
    prior_state: Any,
    branch: CandidateBranch,
    *,
    step_index: int,
    max_steps: int,
    prior_actions: list[ToolAction],
    environment: RuntimeEnvironment,
) -> CandidateBranch:
    branch.local_score = score_candidate_branch(
        task_row,
        prior_state,
        branch,
        step_index=step_index,
        max_steps=max_steps,
        prior_actions=prior_actions,
        available_gene_ids=environment.available_gene_ids,
        available_layers=environment.available_layers,
    )
    return branch


def _branch_selection_errors(branch: CandidateBranch) -> list[str]:
    """Return errors that make a branch unsafe for selection or DPO mining."""

    errors: list[str] = []
    validation = validate_candidate_branch(branch)
    errors.extend(validation.errors)

    score_metadata = branch.local_score.score_metadata
    if isinstance(score_metadata, dict) and score_metadata.get("schema_valid") is False:
        schema_errors = score_metadata.get("schema_errors")
        if isinstance(schema_errors, list):
            errors.extend(str(error) for error in schema_errors if error)
        else:
            errors.append("local_score_schema_invalid")

    return _unique(errors)


def _branch_is_usable_for_selection(branch: CandidateBranch) -> bool:
    return not _branch_selection_errors(branch)


def _branch_tool_name(branch: CandidateBranch) -> str:
    action = branch.actor_step.tool_action
    return action.tool_name if action is not None else "no_tool"


def _branch_has_successful_tool(branch: CandidateBranch) -> bool:
    return (
        branch.actor_step.tool_action is not None
        and branch.observation is not None
        and branch.observation.status == ToolObservationStatus.SUCCESS
    )


def _predicted_gene_ids_from_state(state: Any) -> list[str]:
    gene_ids: list[str] = []
    for group in getattr(state, "predicted_groups", []):
        gene_ids.extend(getattr(group, "gene_ids", []))
    return _unique(gene_ids)


def _branch_predicted_gene_ids(branch: CandidateBranch) -> list[str]:
    return _predicted_gene_ids_from_state(branch.verifier_step.updated_state)


def _complex_metric_delta(branch: CandidateBranch, metric_name: str) -> float:
    complex_metadata = branch.local_score.score_metadata.get("complex", {})
    if not isinstance(complex_metadata, dict):
        return 0.0
    pre_metrics = _safe_dict(_safe_dict(complex_metadata.get("best_group_pre")).get("metrics"))
    post_metrics = _safe_dict(_safe_dict(complex_metadata.get("best_group_post")).get("metrics"))
    pre_value = pre_metrics.get(metric_name, 0.0)
    post_value = post_metrics.get(metric_name, 0.0)
    if not isinstance(pre_value, (int, float)) or not isinstance(post_value, (int, float)):
        return 0.0
    return float(post_value) - float(pre_value)


def _branch_quality_features(
    branch: CandidateBranch,
    *,
    prior_state: Any,
) -> dict[str, Any]:
    prior_gene_ids = _predicted_gene_ids_from_state(prior_state)
    post_gene_ids = _branch_predicted_gene_ids(branch)
    relationship_status = branch.verifier_step.updated_state.relationship_status.value
    tool_name = _branch_tool_name(branch)
    return {
        "tool_name": tool_name,
        "has_successful_tool": _branch_has_successful_tool(branch),
        "prior_gene_count": len(prior_gene_ids),
        "post_gene_count": len(post_gene_ids),
        "group_size_delta": len(post_gene_ids) - len(prior_gene_ids),
        "complex_delta": branch.local_score.complex_membership_delta,
        "mechanistic_delta": branch.local_score.mechanistic_label_delta,
        "efficiency_penalty": branch.local_score.efficiency_penalty,
        "recall_delta": _complex_metric_delta(branch, "recall"),
        "precision_delta": _complex_metric_delta(branch, "precision"),
        "jaccard_delta": _complex_metric_delta(branch, "jaccard"),
        "relationship_status": relationship_status,
        "continuation_state": branch.verifier_step.updated_state.continuation_state.value,
    }


def _tool_preference_rank(task_type: str, tool_name: str) -> int:
    task_preferences = {
        "recovery": {
            "rwr_multiplex": 4,
            "get_neighbors": 3,
            "induce_subgraph": 2,
            "shortest_path": 1,
        },
        "refinement": {
            "induce_subgraph": 4,
            "rwr_multiplex": 3,
            "shortest_path": 2,
            "get_neighbors": 1,
        },
        "none": {
            "induce_subgraph": 3,
            "shortest_path": 2,
            "get_neighbors": 1,
        },
        "explanation": {
            "induce_subgraph": 3,
            "shortest_path": 2,
            "get_neighbors": 1,
        },
    }
    return task_preferences.get(task_type, {}).get(tool_name, 0)


def _branch_task_quality_tuple(
    branch: CandidateBranch,
    *,
    task_type: str,
    prior_state: Any,
) -> tuple[Any, ...]:
    features = _branch_quality_features(branch, prior_state=prior_state)
    normalized = float(branch.local_score.normalized_score or 0.0)
    total = branch.local_score.total_score
    status = features["relationship_status"]
    tool_rank = _tool_preference_rank(task_type, str(features["tool_name"]))
    successful_tool = int(bool(features["has_successful_tool"]))
    no_tool = int(features["tool_name"] == "no_tool")
    group_size_delta = int(features["group_size_delta"])
    complex_delta = float(features["complex_delta"])
    mechanism_delta = float(features["mechanistic_delta"])
    recall_delta = float(features["recall_delta"])
    precision_delta = float(features["precision_delta"])

    if task_type == "recovery":
        quality = (
            complex_delta,
            recall_delta,
            int(group_size_delta > 0),
            group_size_delta,
            successful_tool,
            tool_rank,
            mechanism_delta,
            -float(features["efficiency_penalty"]),
            normalized,
            total,
        )
    elif task_type == "refinement":
        quality = (
            precision_delta,
            complex_delta,
            int(group_size_delta < 0),
            successful_tool,
            tool_rank,
            mechanism_delta,
            -abs(group_size_delta),
            -float(features["efficiency_penalty"]),
            normalized,
            total,
        )
    elif task_type == "none":
        quality = (
            int(status == RelationshipStatus.INSUFFICIENT_SUPPORT.value),
            complex_delta,
            successful_tool,
            tool_rank,
            -len(_branch_predicted_gene_ids(branch)),
            -float(features["efficiency_penalty"]),
            normalized,
            total,
        )
    else:
        quality = (
            mechanism_delta,
            complex_delta,
            int(status == RelationshipStatus.VALIDATED_GROUP.value),
            no_tool,
            successful_tool,
            tool_rank,
            -float(features["efficiency_penalty"]),
            normalized,
            total,
        )
    return quality + (branch.branch_id,)


def _normalize_branch_pool(branches: list[CandidateBranch]) -> None:
    if not branches:
        return

    scores = [branch.local_score.total_score for branch in branches]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        normalized = 1.0 if len(branches) == 1 else 0.5
        for branch in branches:
            branch.local_score.normalized_score = normalized
        return

    for branch in branches:
        normalized_score = (branch.local_score.total_score - min_score) / (max_score - min_score)
        branch.local_score.normalized_score = float(normalized_score)


def _select_best_branch(
    branches: list[CandidateBranch],
    *,
    require_valid: bool = False,
    task_row: dict[str, Any] | None = None,
    prior_state: Any | None = None,
    selection_policy: str = "score",
    selection_score_epsilon: float = 0.0,
) -> CandidateBranch:
    if not branches:
        raise RuntimeError("No branches are available for selection.")

    valid_branches = [branch for branch in branches if _branch_is_usable_for_selection(branch)]
    if require_valid and not valid_branches:
        raise RuntimeError("No schema-valid branches are available for selection.")

    selectable_branches = valid_branches or branches
    if selection_policy == "task_quality" and task_row is not None and prior_state is not None:
        max_normalized = max(float(branch.local_score.normalized_score or 0.0) for branch in selectable_branches)
        near_top = [
            branch
            for branch in selectable_branches
            if max_normalized - float(branch.local_score.normalized_score or 0.0)
            <= selection_score_epsilon
        ]
        task_type = str(task_row.get("task_type", ""))
        selected = max(
            near_top,
            key=lambda branch: _branch_task_quality_tuple(
                branch,
                task_type=task_type,
                prior_state=prior_state,
            ),
        )
        selected.metadata["selection_policy"] = selection_policy
        selected.metadata["selection_score_epsilon"] = selection_score_epsilon
        selected.metadata["selection_quality"] = _branch_quality_features(
            selected,
            prior_state=prior_state,
        )
        return selected

    return sorted(
        selectable_branches,
        key=lambda branch: (
            -(branch.local_score.normalized_score or 0.0),
            -branch.local_score.total_score,
            branch.branch_id,
        ),
    )[0]


@dataclass
class TrajectoryGenerationConfig:
    """Small configuration object for trajectory generation."""

    max_steps: int = 4
    n_act: int = 4
    n_ver: int = 2
    task_concurrency: int = 1
    seed: int = 0
    candidate_source: str = "heuristic"
    allow_model_fallback: bool = False
    preference_pair_margin: float = DEFAULT_PREFERENCE_PAIR_MARGIN
    selection_policy: str = "score"
    selection_score_epsilon: float = DEFAULT_SELECTION_SCORE_EPSILON
    pair_mining_strategy: str = "score_margin"
    tool_coverage_retry_count: int = 0
    recovery_rwr_top_k: int = DEFAULT_RECOVERY_RWR_TOP_K

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.n_act <= 0:
            raise ValueError("n_act must be positive.")
        if self.n_ver <= 0:
            raise ValueError("n_ver must be positive.")
        if self.task_concurrency <= 0:
            raise ValueError("task_concurrency must be positive.")
        if self.candidate_source not in {"heuristic", "model_vllm"}:
            raise ValueError("candidate_source must be one of: heuristic, model_vllm.")
        if self.preference_pair_margin < 0:
            raise ValueError("preference_pair_margin must be non-negative.")
        if self.selection_policy not in SELECTION_POLICIES:
            allowed = ", ".join(SELECTION_POLICIES)
            raise ValueError(f"selection_policy must be one of: {allowed}.")
        if self.selection_score_epsilon < 0:
            raise ValueError("selection_score_epsilon must be non-negative.")
        if self.pair_mining_strategy not in PAIR_MINING_STRATEGIES:
            allowed = ", ".join(PAIR_MINING_STRATEGIES)
            raise ValueError(f"pair_mining_strategy must be one of: {allowed}.")
        if self.tool_coverage_retry_count < 0:
            raise ValueError("tool_coverage_retry_count must be non-negative.")
        if self.recovery_rwr_top_k <= 0:
            raise ValueError("recovery_rwr_top_k must be positive.")


class ProgressTracker:
    """Persist progress for long trajectory-generation runs."""

    def __init__(self, path: Path, stage_defs: tuple[tuple[str, str], ...]) -> None:
        self.path = path
        self.stage_defs = stage_defs
        self.stage_index_lookup = {
            stage_name: index for index, (stage_name, _) in enumerate(stage_defs, start=1)
        }
        self.state: dict[str, Any] = {
            "status": "running",
            "current_stage": None,
            "current_stage_label": None,
            "stage_index": 0,
            "stage_count": len(stage_defs),
            "overall_progress": 0.0,
            "message": "Initialized trajectory generation.",
            "metrics": {},
            "started_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def start_stage(self, stage_name: str, *, message: str, metrics: dict[str, Any] | None = None) -> None:
        self.state["current_stage"] = stage_name
        self.state["current_stage_label"] = dict(self.stage_defs)[stage_name]
        self.state["stage_index"] = self.stage_index_lookup[stage_name]
        self.state["message"] = message
        self.state["metrics"] = metrics or {}
        self.state["updated_at"] = utc_now_iso()
        self.state["overall_progress"] = round(
            (self.state["stage_index"] - 1) / max(1, self.state["stage_count"]),
            6,
        )
        self._write()

    def update(self, *, message: str | None = None, metrics: dict[str, Any] | None = None) -> None:
        if message is not None:
            self.state["message"] = message
        if metrics:
            self.state["metrics"].update(metrics)
        if self.state["current_stage"] == "generate_trajectories":
            completed = self.state["metrics"].get("completed_tasks", 0)
            total = self.state["metrics"].get("total_tasks", 0)
            stage_progress = float(completed) / float(total) if total else 0.0
            self.state["overall_progress"] = round(
                (self.state["stage_index"] - 1 + stage_progress) / max(1, self.state["stage_count"]),
                6,
            )
        self.state["updated_at"] = utc_now_iso()
        self._write()

    def complete(self, *, message: str, metrics: dict[str, Any] | None = None) -> None:
        self.state["status"] = "completed"
        self.state["current_stage"] = "completed"
        self.state["current_stage_label"] = "Completed"
        self.state["stage_index"] = self.state["stage_count"]
        self.state["overall_progress"] = 1.0
        self.state["message"] = message
        if metrics:
            self.state["metrics"].update(metrics)
        self.state["updated_at"] = utc_now_iso()
        self._write()

    def fail(self, error: Exception) -> None:
        self.state["status"] = "failed"
        self.state["message"] = f"Trajectory generation failed: {error}"
        self.state["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
        self.state["updated_at"] = utc_now_iso()
        self._write()


def generate_task_trajectory(
    task_row: dict[str, Any],
    *,
    trajectory_id: str,
    trajectory_seed: int,
    environment: RuntimeEnvironment,
    config: TrajectoryGenerationConfig,
    candidate_generator: OpenAICompatibleCandidateGenerator | None = None,
) -> dict[str, Any]:
    """Generate one deterministic trajectory and all of its branch pools."""

    symbol_lookup = _gene_symbol_lookup(task_row)
    interpretation, state = initialize_state_from_corum_task(task_row, max_budget=config.max_steps)
    initial_state = clone_state(state)
    prior_actions: list[ToolAction] = []

    branch_pools: list[dict[str, Any]] = []
    trajectory_turns: list[TrajectoryTurn] = []
    finding_records: list[dict[str, Any]] = []
    preference_pairs_raw: list[PreferencePair] = []
    selected_branch_ids: list[str] = []

    for step_index in range(config.max_steps):
        if state.continuation_state == ContinuationState.STOP or state.remaining_budget <= 0:
            break

        context = SharedPrefixContext(
            query_text=task_row["query_text"],
            user_evidence=task_row["visible_inputs"],
            interpretation=clone_interpretation(interpretation),
            state=clone_state(state),
            source_task_id=task_row.get("task_id"),
        )
        branches: list[CandidateBranch] = []
        rejected_model_errors: list[str] = []
        rejected_model_candidates: list[dict[str, Any]] = []
        if config.candidate_source == "model_vllm":
            if candidate_generator is None:
                raise ValueError("candidate_generator is required for model_vllm generation.")

            actor_candidates = candidate_generator.generate_actor_candidates(
                context,
                task_row=task_row,
                step_index=step_index,
                n_act=config.n_act,
                seed=trajectory_seed + step_index,
                environment=environment,
            )
            if (
                config.tool_coverage_retry_count > 0
                and task_row["task_type"] in {"recovery", "refinement"}
                and not any(
                    isinstance(candidate.get("tool_action"), dict)
                    and candidate["tool_action"].get("tool_name")
                    for candidate in actor_candidates
                )
            ):
                actor_candidates.extend(
                    candidate_generator.generate_actor_candidates(
                        context,
                        task_row=task_row,
                        step_index=step_index,
                        n_act=config.tool_coverage_retry_count,
                        seed=trajectory_seed + step_index + 7919,
                        environment=environment,
                        force_tool_coverage=True,
                    )
                )
            for actor_index, actor_candidate in enumerate(actor_candidates):
                actor_step, actor_errors = _build_actor_step_from_model_candidate(
                    actor_candidate,
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    actor_index=actor_index,
                )
                actor_generation_errors = _unique(
                    actor_errors + list(actor_candidate.get("generator_errors", []))
                )
                tool_default_metadata = _apply_task_tool_defaults(
                    actor_step,
                    task_type=task_row["task_type"],
                    recovery_rwr_top_k=config.recovery_rwr_top_k,
                )
                if actor_step.tool_action is not None:
                    actor_tool_validation = validate_tool_action_semantics(
                        actor_step.tool_action,
                        state=state,
                        available_gene_ids=environment.available_gene_ids,
                        available_layers=environment.available_layers,
                    )
                    if not actor_tool_validation.valid:
                        actor_generation_errors = _unique(
                            actor_generation_errors
                            + [
                                f"actor_tool_semantics_invalid: {error}"
                                for error in actor_tool_validation.errors
                            ]
                        )
                if not _actor_candidate_is_usable(actor_step, actor_generation_errors):
                    rejected_model_errors.extend(actor_generation_errors)
                    rejected_model_candidates.append(
                        {
                            "phase": "actor",
                            "actor_index": actor_index,
                            "generator_errors": actor_generation_errors,
                        }
                    )
                    continue
                observation = None
                if actor_step.tool_action is not None:
                    observation = environment.execute(
                        actor_step.tool_action,
                        state=state,
                        prior_actions=prior_actions,
                    )

                verifier_candidates = candidate_generator.generate_verifier_candidates(
                    context,
                    task_row=task_row,
                    actor_candidate=actor_candidate,
                    actor_step=actor_step,
                    observation=observation,
                    step_index=step_index,
                    n_ver=config.n_ver,
                    seed=trajectory_seed + (step_index * 100) + actor_index,
                )
                for verifier_index, verifier_candidate in enumerate(verifier_candidates):
                    branch_generation_errors = _unique(
                        actor_generation_errors + list(verifier_candidate.get("generator_errors", []))
                    )
                    if not _verifier_candidate_is_usable(verifier_candidate, branch_generation_errors):
                        rejected_model_errors.extend(branch_generation_errors)
                        rejected_model_candidates.append(
                            {
                                "phase": "verifier",
                                "actor_index": actor_index,
                                "verifier_index": verifier_index,
                                "generator_errors": branch_generation_errors,
                            }
                        )
                        continue
                    branch_id = f"{trajectory_id}.step{step_index}.a{actor_index}.v{verifier_index}"
                    branch = _build_branch_from_model_output(
                        task_row,
                        state,
                        actor_step,
                        observation,
                        verifier_candidate,
                        branch_id=branch_id,
                        step_index=step_index,
                        symbol_lookup=symbol_lookup,
                        generator_errors=actor_generation_errors,
                    )
                    if actor_candidate.get("actor_sampling_directive") is not None:
                        branch.metadata["actor_sampling_directive"] = actor_candidate[
                            "actor_sampling_directive"
                        ]
                    if tool_default_metadata:
                        branch.metadata["tool_argument_defaults"] = tool_default_metadata
                    branch = _score_branch(
                        task_row,
                        state,
                        branch,
                        step_index=step_index,
                        max_steps=config.max_steps,
                        prior_actions=prior_actions,
                        environment=environment,
                    )
                    branch_validation_errors = _branch_selection_errors(branch)
                    if branch_validation_errors and not config.allow_model_fallback:
                        rejected_model_errors.extend(branch_validation_errors)
                        rejected_model_candidates.append(
                            {
                                "phase": "branch_validation",
                                "actor_index": actor_index,
                                "verifier_index": verifier_index,
                                "branch_id": branch_id,
                                "generator_errors": branch_validation_errors,
                            }
                        )
                        continue
                    branches.append(branch)
            if not branches:
                rejected_model_errors = _unique(
                    rejected_model_errors or ["model_generator_returned_no_candidates"]
                )
                if not config.allow_model_fallback:
                    diagnostic_preview = json.dumps(
                        rejected_model_candidates[:2],
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    raise RuntimeError(
                        "Model-backed generation produced no usable candidates "
                        f"for {trajectory_id} step {step_index}. "
                        f"errors={rejected_model_errors}. "
                        f"rejected_preview={diagnostic_preview}"
                    )
                templates = _actor_templates_for_step(
                    task_row,
                    state,
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    n_act=config.n_act,
                    recovery_rwr_top_k=config.recovery_rwr_top_k,
                )
                for actor_index, template in enumerate(templates):
                    actor_template_id = template["template_id"]
                    actor_step = template["actor_step"]
                    observation = None
                    if actor_step.tool_action is not None:
                        observation = environment.execute(
                            actor_step.tool_action,
                            state=state,
                            prior_actions=prior_actions,
                        )
                    verifier_template_ids = _verifier_template_ids(
                        task_row["task_type"],
                        actor_template_id,
                        n_ver=config.n_ver,
                    )
                    for verifier_index, verifier_template_id in enumerate(verifier_template_ids):
                        branch_id = (
                            f"{trajectory_id}.step{step_index}.fallback_a{actor_index}.v{verifier_index}"
                        )
                        branch = _build_heuristic_branch_for_templates(
                            task_row,
                            interpretation,
                            state,
                            actor_template_id,
                            actor_step,
                            verifier_template_id,
                            observation,
                            branch_id=branch_id,
                            step_index=step_index,
                            max_steps=config.max_steps,
                            symbol_lookup=symbol_lookup,
                        )
                        branch.metadata["generator_backend"] = "heuristic_fallback"
                        branch.metadata["generator_errors"] = rejected_model_errors
                        branch = _score_branch(
                            task_row,
                            state,
                            branch,
                            step_index=step_index,
                            max_steps=config.max_steps,
                            prior_actions=prior_actions,
                            environment=environment,
                        )
                        branches.append(branch)
        else:
            templates = _actor_templates_for_step(
                task_row,
                state,
                trajectory_id=trajectory_id,
                step_index=step_index,
                n_act=config.n_act,
                recovery_rwr_top_k=config.recovery_rwr_top_k,
            )

            for actor_index, template in enumerate(templates):
                actor_template_id = template["template_id"]
                actor_step = template["actor_step"]
                observation = None
                if actor_step.tool_action is not None:
                    observation = environment.execute(
                        actor_step.tool_action,
                        state=state,
                        prior_actions=prior_actions,
                    )

                verifier_template_ids = _verifier_template_ids(
                    task_row["task_type"],
                    actor_template_id,
                    n_ver=config.n_ver,
                )
                for verifier_index, verifier_template_id in enumerate(verifier_template_ids):
                    branch_id = (
                        f"{trajectory_id}.step{step_index}.a{actor_index}.v{verifier_index}"
                    )
                    branch = _build_heuristic_branch_for_templates(
                        task_row,
                        interpretation,
                        state,
                        actor_template_id,
                        actor_step,
                        verifier_template_id,
                        observation,
                        branch_id=branch_id,
                        step_index=step_index,
                        max_steps=config.max_steps,
                        symbol_lookup=symbol_lookup,
                    )
                    branch = _score_branch(
                        task_row,
                        state,
                        branch,
                        step_index=step_index,
                        max_steps=config.max_steps,
                        prior_actions=prior_actions,
                        environment=environment,
                    )
                    branches.append(branch)

        _normalize_branch_pool(branches)
        selected_branch = _select_best_branch(
            branches,
            require_valid=config.candidate_source == "model_vllm" and not config.allow_model_fallback,
            task_row=task_row,
            prior_state=state,
            selection_policy=config.selection_policy,
            selection_score_epsilon=config.selection_score_epsilon,
        )
        preference_pairs_raw.extend(
            _mine_preference_pairs(
                task_row=task_row,
                trajectory_id=trajectory_id,
                trajectory_seed=trajectory_seed,
                step_index=step_index,
                context=context,
                branches=branches,
                chosen_branch=selected_branch,
                score_margin=config.preference_pair_margin,
                pair_mining_strategy=config.pair_mining_strategy,
            )
        )
        selected_branch_ids.append(selected_branch.branch_id)

        branch_pools.append(
            {
                "trajectory_id": trajectory_id,
                "source_task_id": task_row["task_id"],
                "task_type": task_row["task_type"],
                "difficulty": task_row.get("difficulty"),
                "evidence_mode": task_row.get("evidence_mode"),
                "step_index": step_index,
                "context": context.to_dict(),
                "branches": [branch.to_dict() for branch in branches],
                "selected_branch_id": selected_branch.branch_id,
            }
        )

        finding_text = _render_finding_text(selected_branch)
        turn = TrajectoryTurn(
            trajectory_id=trajectory_id,
            step_index=step_index,
            prior_interpretation=clone_interpretation(interpretation),
            prior_state=clone_state(state),
            branch=selected_branch,
            selected=True,
            finding_text=finding_text,
        )
        trajectory_turns.append(turn)
        finding_records.append(
            _build_finding_record(
                task_row=task_row,
                trajectory_id=trajectory_id,
                trajectory_seed=trajectory_seed,
                step_index=step_index,
                context=context,
                branch=selected_branch,
                finding_text=finding_text,
            )
        )

        interpretation = clone_interpretation(selected_branch.verifier_step.updated_interpretation)
        state = clone_state(selected_branch.verifier_step.updated_state)
        if selected_branch.actor_step.tool_action is not None:
            prior_actions.append(selected_branch.actor_step.tool_action)

    terminal_score = score_terminal_trajectory(
        task_row,
        initial_state,
        state,
        step_count=len(trajectory_turns),
        max_steps=config.max_steps,
    )
    return {
        "trajectory_id": trajectory_id,
        "task_row": task_row,
        "turns": trajectory_turns,
        "branch_pools": branch_pools,
        "finding_records": finding_records,
        "preference_pairs_raw": preference_pairs_raw,
        "final_interpretation": interpretation,
        "final_state": state,
        "selected_branch_ids": selected_branch_ids,
        "terminal_score": terminal_score,
        "rendered_summary": _render_final_summary(
            task_row,
            interpretation,
            state,
            [record["finding_text"] for record in finding_records],
        ),
    }


def generate_trajectories(
    *,
    task_rows: list[dict[str, Any]],
    out_dir: Path,
    environment: RuntimeEnvironment,
    config: TrajectoryGenerationConfig,
    model_generator_config: ModelGeneratorConfig | None = None,
    candidate_generator: OpenAICompatibleCandidateGenerator | None = None,
    progress_path: Path | None = None,
    task_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate trajectories for a task list and write the run artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_tracker = ProgressTracker(progress_path or (out_dir / DEFAULT_PROGRESS_FILENAME), TRAJECTORY_STAGES)

    branch_pool_path = out_dir / "branch_pools.jsonl"
    trajectory_turns_path = out_dir / "trajectory_turns.jsonl"
    finding_records_path = out_dir / "finding_records.jsonl"
    preference_pairs_raw_path = out_dir / "preference_pairs_raw.jsonl"
    preference_pairs_path = out_dir / "preference_pairs.jsonl"
    final_summaries_path = out_dir / "final_summaries.jsonl"
    manifest_path = out_dir / "manifest.json"

    total_steps = 0
    total_branch_pools = 0
    total_branches = 0
    raw_preference_pairs: list[PreferencePair] = []
    if config.candidate_source == "model_vllm" and candidate_generator is None:
        if model_generator_config is None:
            raise ValueError("model_generator_config is required when candidate_source=model_vllm.")
        candidate_generator = OpenAICompatibleCandidateGenerator(model_generator_config)

    try:
        progress_tracker.start_stage(
            "load_tasks",
            message="Loaded canonical CORUM tasks.",
            metrics={
                "total_tasks": len(task_rows),
                **(task_selection or {}),
            },
        )

        progress_tracker.start_stage(
            "initialize_runtime",
            message="Initialized deterministic runtime.",
            metrics=environment.describe(),
        )

        progress_tracker.start_stage(
            "generate_trajectories",
            message="Generating shared-prefix trajectories.",
            metrics={
                "completed_tasks": 0,
                "total_tasks": len(task_rows),
                "completed_steps": 0,
                "completed_branch_pools": 0,
                "completed_branches": 0,
            },
        )

        with branch_pool_path.open("w", encoding="utf-8") as branch_pool_handle, \
            trajectory_turns_path.open("w", encoding="utf-8") as turns_handle, \
            finding_records_path.open("w", encoding="utf-8") as findings_handle, \
            preference_pairs_raw_path.open("w", encoding="utf-8") as raw_pairs_handle, \
            final_summaries_path.open("w", encoding="utf-8") as summaries_handle:

            def _generate_single_task(task_entry: tuple[int, dict[str, Any]]) -> tuple[int, int, dict[str, Any]]:
                task_index, task_row = task_entry
                trajectory_seed = config.seed + task_index
                trajectory_id = f"{task_row['task_id']}.seed{trajectory_seed}"
                generated = generate_task_trajectory(
                    task_row,
                    trajectory_id=trajectory_id,
                    trajectory_seed=trajectory_seed,
                    environment=environment,
                    config=config,
                    candidate_generator=candidate_generator,
                )
                return task_index, trajectory_seed, generated

            task_entries = list(enumerate(task_rows))
            if config.task_concurrency == 1:
                generation_iter = map(_generate_single_task, task_entries)
                executor_context = None
            else:
                executor_context = ThreadPoolExecutor(max_workers=config.task_concurrency)
                generation_iter = executor_context.map(_generate_single_task, task_entries)

            completed_tasks = 0
            try:
                for task_index, trajectory_seed, generated in generation_iter:
                    task_row = task_rows[task_index]

                    for branch_pool in generated["branch_pools"]:
                        _write_jsonl_line(branch_pool_handle, branch_pool)
                        total_branch_pools += 1
                        total_branches += len(branch_pool["branches"])

                    for turn in generated["turns"]:
                        row = turn.to_dict()
                        row["source_task_id"] = task_row["task_id"]
                        row["task_type"] = task_row["task_type"]
                        row["difficulty"] = task_row.get("difficulty")
                        row["evidence_mode"] = task_row.get("evidence_mode")
                        _write_jsonl_line(turns_handle, row)
                        total_steps += 1

                    for finding_record in generated["finding_records"]:
                        _write_jsonl_line(findings_handle, finding_record)

                    for preference_pair in generated["preference_pairs_raw"]:
                        raw_preference_pairs.append(preference_pair)
                        _write_jsonl_line(raw_pairs_handle, preference_pair.to_dict())

                    terminal_score = generated["terminal_score"]

                    _write_jsonl_line(
                        summaries_handle,
                        {
                            "trajectory_id": generated["trajectory_id"],
                            "source_task_id": task_row["task_id"],
                            "task_type": task_row["task_type"],
                            "difficulty": task_row.get("difficulty"),
                            "evidence_mode": task_row.get("evidence_mode"),
                            "trajectory_seed": trajectory_seed,
                            "step_count": len(generated["turns"]),
                            "selected_branch_ids": generated["selected_branch_ids"],
                            "rendered_summary": generated["rendered_summary"],
                            "final_interpretation": generated["final_interpretation"].to_dict(),
                            "final_state": generated["final_state"].to_dict(),
                            "finding_count": len(generated["finding_records"]),
                            "terminal_schema_score": terminal_score["schema_score"],
                            "terminal_absolute_complex_score": terminal_score["absolute_complex_score"],
                            "terminal_complex_delta_score": terminal_score["complex_delta"],
                            "terminal_absolute_mechanistic_score": terminal_score["absolute_mechanistic_score"],
                            "terminal_mechanistic_delta_score": terminal_score["mechanistic_delta"],
                            "terminal_efficiency_penalty": terminal_score["efficiency_penalty"],
                            "terminal_reward": terminal_score["terminal_reward"],
                        },
                    )

                    completed_tasks += 1
                    progress_tracker.update(
                        message=f"Generated trajectory for {task_row['task_id']}.",
                        metrics={
                            "completed_tasks": completed_tasks,
                            "total_tasks": len(task_rows),
                            "current_task_id": task_row["task_id"],
                            "completed_steps": total_steps,
                            "completed_branch_pools": total_branch_pools,
                            "completed_branches": total_branches,
                        },
                    )
            finally:
                if executor_context is not None:
                    executor_context.shutdown(wait=True)

        balanced_preference_pairs = _balance_preference_pairs(
            raw_preference_pairs,
            pair_mining_strategy=config.pair_mining_strategy,
        )
        with preference_pairs_path.open("w", encoding="utf-8") as preference_pairs_handle:
            for preference_pair in balanced_preference_pairs:
                _write_jsonl_line(preference_pairs_handle, preference_pair.to_dict())

        progress_tracker.start_stage(
            "write_manifest",
            message="Writing trajectory manifest.",
            metrics={
                "num_trajectories": len(task_rows),
                "total_steps": total_steps,
                "total_branch_pools": total_branch_pools,
                "total_branches": total_branches,
                "total_preference_pairs_raw": len(raw_preference_pairs),
                "total_preference_pairs": len(balanced_preference_pairs),
            },
        )

        manifest = {
            "generated_at": utc_now_iso(),
            "task_count": len(task_rows),
            "num_trajectories": len(task_rows),
            "total_steps": total_steps,
            "total_branch_pools": total_branch_pools,
            "total_branches": total_branches,
            "candidate_source": config.candidate_source,
            "config": {
                "max_steps": config.max_steps,
                "n_act": config.n_act,
                "n_ver": config.n_ver,
                "task_concurrency": config.task_concurrency,
                "seed": config.seed,
                "allow_model_fallback": config.allow_model_fallback,
                "preference_pair_margin": config.preference_pair_margin,
                "selection_policy": config.selection_policy,
                "selection_score_epsilon": config.selection_score_epsilon,
                "pair_mining_strategy": config.pair_mining_strategy,
                "tool_coverage_retry_count": config.tool_coverage_retry_count,
                "recovery_rwr_top_k": config.recovery_rwr_top_k,
            },
            "generator": {
                "candidate_source": config.candidate_source,
                "api_base": model_generator_config.api_base if model_generator_config else None,
                "configured_api_mode": (
                    model_generator_config.api_mode if model_generator_config else None
                ),
                "resolved_api_mode": (
                    getattr(candidate_generator, "api_mode", None) if candidate_generator else None
                ),
                "model_name": getattr(candidate_generator, "model_name", None) if candidate_generator else None,
                "max_completion_tokens": (
                    model_generator_config.max_completion_tokens if model_generator_config else None
                ),
                "actor_rationale_max_completion_tokens": (
                    model_generator_config.actor_rationale_max_completion_tokens
                    if model_generator_config
                    else None
                ),
                "request_timeout_seconds": (
                    model_generator_config.request_timeout_seconds if model_generator_config else None
                ),
                "reasoning_effort": (
                    model_generator_config.reasoning_effort if model_generator_config else None
                ),
                "actor_sampling_strategy": (
                    model_generator_config.actor_sampling_strategy if model_generator_config else None
                ),
            },
            "task_selection": task_selection or {},
            "runtime": environment.describe(),
            "artifacts": {
                "finding_record_count": total_steps,
                "preference_pair_raw_count": len(raw_preference_pairs),
                "preference_pair_count": len(balanced_preference_pairs),
            },
            "outputs": {
                "branch_pools": str(branch_pool_path),
                "trajectory_turns": str(trajectory_turns_path),
                "finding_records": str(finding_records_path),
                "preference_pairs_raw": str(preference_pairs_raw_path),
                "preference_pairs": str(preference_pairs_path),
                "final_summaries": str(final_summaries_path),
            },
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        progress_tracker.complete(
            message="Completed trajectory generation.",
            metrics={
                "num_trajectories": len(task_rows),
                "total_steps": total_steps,
                "total_branch_pools": total_branch_pools,
                "total_branches": total_branches,
                "total_preference_pairs_raw": len(raw_preference_pairs),
                "total_preference_pairs": len(balanced_preference_pairs),
            },
        )
        return manifest

    except Exception as error:
        progress_tracker.fail(error)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for trajectory generation."""

    parser = argparse.ArgumentParser(description="Generate shared-prefix CORUM trajectories.")
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help="Path to a canonical CORUM task JSONL file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where trajectory artifacts will be written.",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=DEFAULT_STORE_DIR,
        help="Compiled multiplex store directory. Use this for the full HumanNet runtime.",
    )
    parser.add_argument(
        "--compiled-library-path",
        type=Path,
        default=None,
        help="Optional path to the compiled C++ runtime library.",
    )
    parser.add_argument(
        "--multiplex-flist",
        type=Path,
        default=None,
        help="Optional text flist for the Python reference backend.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional number of task rows to process from the input JSONL.",
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used to partition the input task list.",
    )
    parser.add_argument(
        "--task-shard-count",
        type=int,
        default=1,
        help="Total number of disjoint task shards to partition the input task list into.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum number of decision steps per trajectory.",
    )
    parser.add_argument(
        "--n-act",
        type=int,
        default=4,
        help="Maximum number of actor candidates per step.",
    )
    parser.add_argument(
        "--n-ver",
        type=int,
        default=2,
        help="Maximum number of verifier variants per actor candidate.",
    )
    parser.add_argument(
        "--task-concurrency",
        type=int,
        default=1,
        help="Number of trajectories to generate concurrently. Higher values improve vLLM batching at the cost of more runtime pressure.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed recorded in trajectory ids and the output manifest.",
    )
    parser.add_argument(
        "--candidate-source",
        choices=("heuristic", "model_vllm"),
        default="heuristic",
        help="Candidate generator backend.",
    )
    parser.add_argument(
        "--allow-model-fallback",
        action="store_true",
        help="Allow model_vllm runs to fall back to heuristic templates when no usable model candidate is produced.",
    )
    parser.add_argument(
        "--preference-pair-margin",
        type=float,
        default=DEFAULT_PREFERENCE_PAIR_MARGIN,
        help="Minimum normalized-score margin required to keep a dispreferred branch in DPO pair mining.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=SELECTION_POLICIES,
        default="score",
        help=(
            "Branch selection policy. 'score' picks the highest local score; "
            "'task_quality' uses task-aware tie breakers within --selection-score-epsilon."
        ),
    )
    parser.add_argument(
        "--selection-score-epsilon",
        type=float,
        default=DEFAULT_SELECTION_SCORE_EPSILON,
        help="Normalized-score window for task_quality selection tie breakers.",
    )
    parser.add_argument(
        "--pair-mining-strategy",
        choices=PAIR_MINING_STRATEGIES,
        default="score_margin",
        help=(
            "Preference-pair mining mode. 'quality_balanced' keeps score margins but "
            "adds pair categories and category/evidence-aware balancing."
        ),
    )
    parser.add_argument(
        "--tool-coverage-retry-count",
        type=int,
        default=0,
        help=(
            "For recovery/refinement model runs, request this many extra tool-directed "
            "actor candidates when the initial actor batch contains no tool action."
        ),
    )
    parser.add_argument(
        "--recovery-rwr-top-k",
        type=int,
        default=DEFAULT_RECOVERY_RWR_TOP_K,
        help=(
            "Minimum rwr_multiplex top_k used for recovery actor branches so the "
            "verifier can inspect a broader non-seed candidate list."
        ),
    )
    parser.add_argument(
        "--generator-api-base",
        type=str,
        default=DEFAULT_GENERATOR_API_BASE,
        help="OpenAI-compatible API base for model-backed candidate generation.",
    )
    parser.add_argument(
        "--generator-api-mode",
        choices=("auto", "chat_completions", "responses", "completions"),
        default="auto",
        help="Generator API style. 'auto' uses completions for gpt-oss and chat completions otherwise.",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        help="Optional model id to send to the generator API. If omitted, discover the first served model.",
    )
    parser.add_argument(
        "--generator-api-key-env",
        type=str,
        default=DEFAULT_GENERATOR_API_KEY_ENV,
        help="Environment variable name that holds the generator API key.",
    )
    parser.add_argument(
        "--generator-api-key",
        type=str,
        default=None,
        help="Optional API key for the generator endpoint.",
    )
    parser.add_argument(
        "--generator-temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for model-backed candidate generation.",
    )
    parser.add_argument(
        "--generator-top-p",
        type=float,
        default=0.95,
        help="Top-p sampling value for model-backed candidate generation.",
    )
    parser.add_argument(
        "--generator-max-completion-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for actor and verifier generation calls.",
    )
    parser.add_argument(
        "--generator-actor-rationale-max-completion-tokens",
        type=int,
        default=DEFAULT_ACTOR_RATIONALE_MAX_TOKENS,
        help="Maximum completion tokens for the actor rationale follow-up pass.",
    )
    parser.add_argument(
        "--generator-reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
        help="Reasoning effort sent to model-backed generation requests.",
    )
    parser.add_argument(
        "--actor-sampling-strategy",
        choices=ACTOR_SAMPLING_STRATEGIES,
        default="batch",
        help=(
            "Actor sampling mode. 'batch' requests n_act samples from one prompt; "
            "'verbalized' requests each actor sample with a distinct exploration directive."
        ),
    )
    parser.add_argument(
        "--generator-timeout-seconds",
        type=int,
        default=3600,
        help="Request timeout for model-backed generation calls.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help="Optional path for the progress JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run trajectory generation from the command line."""

    args = parse_args()
    task_rows = _load_task_rows(
        args.tasks_path,
        max_tasks=args.max_tasks,
        task_shard_index=args.task_shard_index,
        task_shard_count=args.task_shard_count,
    )
    config = TrajectoryGenerationConfig(
        max_steps=args.max_steps,
        n_act=args.n_act,
        n_ver=args.n_ver,
        task_concurrency=args.task_concurrency,
        seed=args.seed,
        candidate_source=args.candidate_source,
        allow_model_fallback=args.allow_model_fallback,
        preference_pair_margin=args.preference_pair_margin,
        selection_policy=args.selection_policy,
        selection_score_epsilon=args.selection_score_epsilon,
        pair_mining_strategy=args.pair_mining_strategy,
        tool_coverage_retry_count=args.tool_coverage_retry_count,
        recovery_rwr_top_k=args.recovery_rwr_top_k,
    )
    model_generator_config = None
    if args.candidate_source == "model_vllm":
        model_generator_config = ModelGeneratorConfig(
            api_base=args.generator_api_base,
            api_mode=args.generator_api_mode,
            model_name=args.generator_model,
            api_key=args.generator_api_key,
            api_key_env=args.generator_api_key_env,
            request_timeout_seconds=args.generator_timeout_seconds,
            temperature=args.generator_temperature,
            top_p=args.generator_top_p,
            max_completion_tokens=args.generator_max_completion_tokens,
            actor_rationale_max_completion_tokens=args.generator_actor_rationale_max_completion_tokens,
            reasoning_effort=args.generator_reasoning_effort,
            actor_sampling_strategy=args.actor_sampling_strategy,
        )

    if args.store_dir is not None:
        environment = RuntimeEnvironment(
            store_dir=str(args.store_dir),
            compiled_library_path=str(args.compiled_library_path) if args.compiled_library_path else None,
        )
    elif args.multiplex_flist is not None:
        environment = RuntimeEnvironment(multiplex_flist=str(args.multiplex_flist))
    else:
        raise ValueError("Provide either --store-dir or --multiplex-flist.")

    task_selection = {
        "tasks_path": str(args.tasks_path),
        "max_tasks": args.max_tasks,
        "task_shard_index": args.task_shard_index,
        "task_shard_count": args.task_shard_count,
    }

    generate_trajectories(
        task_rows=task_rows,
        out_dir=args.out_dir,
        environment=environment,
        config=config,
        model_generator_config=model_generator_config,
        progress_path=args.progress_path,
        task_selection=task_selection,
    )


if __name__ == "__main__":
    main()
