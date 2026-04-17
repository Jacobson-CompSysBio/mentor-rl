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
import json
import os
import sys
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
    RelationshipStatus,
    RuntimeEnvironment,
    SharedPrefixContext,
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
    record_tool_call,
    replace_mechanistic_labels,
    replace_predicted_groups,
    score_candidate_branch,
    set_continuation_state,
)


DEFAULT_TASKS_PATH = REPO_ROOT / "data" / "corum_corpus" / "tasks.train.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "corum_trajectories"
DEFAULT_STORE_DIR = REPO_ROOT / "data" / "humannet_multiplex_store"
DEFAULT_PROGRESS_FILENAME = "progress.json"
DEFAULT_GENERATOR_API_BASE = "http://127.0.0.1:8000/v1"
DEFAULT_GENERATOR_API_KEY_ENV = "OPENAI_API_KEY"
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
- Return exactly one JSON object and nothing else.
- Choose at most one next tool call.
- Use only visible evidence and deterministic runtime observations.
- Never assume access to hidden targets or labels that were not shown.
- Use canonical Ensembl gene ids when you reference genes in tool arguments.
- Prefer the cheapest action that is most likely to reduce uncertainty.
- If current visible evidence is already enough, return "tool_action": null.

Tool guidance:
- query_mygene: look up identifiers or metadata for one gene or alias string
- get_neighbors: inspect one seed gene's neighborhood
- shortest_path: test whether two genes are closely connected
- induce_subgraph: inspect coherence inside a candidate group
- rwr_monoplex: rank candidates on one named layer
- rwr_multiplex: rank candidates across the multiplex; prefer for recovery or refinement

Allowed tools:
- query_mygene: {"query": str, "fields": [str] optional}
- get_neighbors: {"gene": str, "layers": [str] optional}
- shortest_path: {"source": str, "target": str, "layer": str optional}
- rwr_multiplex: {"seeds": [str], "top_k": int optional}
- rwr_monoplex: {"seeds": [str], "layer": str, "top_k": int optional}
- induce_subgraph: {"genes": [str], "layers": [str] optional}

Output schema:
{"reasoning_text": "...", "tool_action": null or {"tool_name": "...", "arguments": {...}}}
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


def _json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


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


def _tool_action_from_tool_calls(tool_calls: Any) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(tool_calls, list) or not tool_calls:
        return None, errors

    if len(tool_calls) > 1:
        errors.append("actor_multiple_tool_calls_returned")

    first_call = tool_calls[0]
    if not isinstance(first_call, dict):
        return None, errors + ["actor_tool_call_not_a_dict"]

    function_payload = _safe_dict(first_call.get("function"))
    tool_name = function_payload.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return None, errors + ["actor_tool_call_name_missing_or_invalid"]

    raw_arguments = function_payload.get("arguments")
    arguments: dict[str, Any] = {}
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        if raw_arguments.strip():
            try:
                parsed_arguments = json.loads(raw_arguments)
            except Exception as error:
                errors.append(f"actor_tool_call_arguments_json_parse_error: {error}")
            else:
                if isinstance(parsed_arguments, dict):
                    arguments = parsed_arguments
                else:
                    errors.append("actor_tool_call_arguments_not_a_dict")
    elif raw_arguments is not None:
        errors.append("actor_tool_call_arguments_invalid")

    return {"tool_name": tool_name, "arguments": arguments}, errors


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

    saw_tool_action = "tool_action" in payload or "tool_calls" in payload
    tool_action = payload.get("tool_action")
    if "tool_calls" in payload:
        tool_action, tool_call_errors = _tool_action_from_tool_calls(payload.get("tool_calls"))
        errors.extend(tool_call_errors)
    elif tool_action is not None and not isinstance(tool_action, dict):
        tool_action = None
        errors.append("actor_tool_action_not_a_dict")

    if not saw_tool_action:
        errors.append("actor_tool_action_missing")

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
        "actor_tool_action_missing",
        "actor_tool_action_not_a_dict",
        "actor_tool_name_missing_or_invalid",
        "actor_reasoning_and_tool_action_blank",
    }
    if any(error in fatal_errors for error in errors):
        return False
    if _has_error_prefix(errors, ("actor_json_parse_error:", "actor_tool_call_")):
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
    if _has_error_prefix(errors, ("verifier_json_parse_error:",)):
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
    model_name: str | None = None
    api_key: str | None = None
    api_key_env: str = DEFAULT_GENERATOR_API_KEY_ENV
    request_timeout_seconds: int = 180
    temperature: float = 0.8
    top_p: float = 0.95
    max_completion_tokens: int = 700
    reasoning_effort: str = "low"

    def resolved_api_key(self) -> str:
        """Return the API key, falling back to an environment variable."""

        if self.api_key is not None:
            return self.api_key
        return os.getenv(self.api_key_env, "EMPTY")


class OpenAICompatibleCandidateGenerator:
    """Generate actor and verifier candidates through a chat-completions API."""

    def __init__(self, config: ModelGeneratorConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.model_name = config.model_name or self._discover_model_name()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.resolved_api_key()}",
            "Content-Type": "application/json",
        }

    def _discover_model_name(self) -> str:
        response = self.session.get(
            f"{self.config.api_base.rstrip('/')}/models",
            headers=self._headers(),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])
        if not data:
            raise RuntimeError("The generator API did not report any served models.")
        model_name = data[0].get("id")
        if not isinstance(model_name, str) or not model_name:
            raise RuntimeError("The generator API returned an invalid model id.")
        return model_name

    def _chat(self, messages: list[dict[str, str]], *, n: int, seed: int) -> list[str]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "n": n,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_completion_tokens": self.config.max_completion_tokens,
            "reasoning_effort": self.config.reasoning_effort,
            "include_reasoning": True,
            "seed": seed,
        }
        response = self.session.post(
            f"{self.config.api_base.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        texts: list[str] = []
        for choice in payload.get("choices", []):
            if not isinstance(choice, dict):
                texts.append(_json_dumps_compact({"choice": choice}))
                continue
            texts.append(_response_choice_to_text(choice))
        if not texts:
            texts.append(
                _json_dumps_compact(
                    {
                        "error": "chat_completion_returned_no_choices",
                        "response_keys": sorted(payload),
                    }
                )
            )
        return texts

    def generate_actor_candidates(
        self,
        context: SharedPrefixContext,
        *,
        task_row: dict[str, Any],
        step_index: int,
        n_act: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        user_prompt = json.dumps(
            {
                "task_row": {
                    "task_id": task_row["task_id"],
                    "task_type": task_row["task_type"],
                    "difficulty": task_row.get("difficulty"),
                    "evidence_mode": task_row.get("evidence_mode"),
                },
                "query_text": context.query_text,
                "visible_inputs": context.user_evidence,
                "interpretation": context.interpretation.to_dict(),
                "state": context.state.to_dict(),
                "step_index": step_index,
            },
            indent=2,
            sort_keys=True,
        )
        messages = [
            {"role": "system", "content": ACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        candidates: list[dict[str, Any]] = []
        for raw_text in self._chat(messages, n=n_act, seed=seed):
            try:
                payload = _parse_model_json(raw_text)
                normalized_payload, payload_errors = _normalize_actor_payload(payload)
                candidates.append(
                    {
                        "reasoning_text": normalized_payload["reasoning_text"],
                        "tool_action": normalized_payload["tool_action"],
                        "raw_text": raw_text,
                        "generator_errors": payload_errors,
                    }
                )
            except Exception as error:
                candidates.append(
                    {
                        "reasoning_text": "",
                        "tool_action": None,
                        "raw_text": raw_text,
                        "generator_errors": [f"actor_json_parse_error: {error}"],
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
            {
                "task_row": {
                    "task_id": task_row["task_id"],
                    "task_type": task_row["task_type"],
                    "difficulty": task_row.get("difficulty"),
                    "evidence_mode": task_row.get("evidence_mode"),
                },
                "query_text": context.query_text,
                "visible_inputs": context.user_evidence,
                "prior_interpretation": context.interpretation.to_dict(),
                "prior_state": context.state.to_dict(),
                "actor_output": {
                    "reasoning_text": actor_step.reasoning_text,
                    "tool_action": actor_step.tool_action.to_dict() if actor_step.tool_action else None,
                },
                "deterministic_observation": observation.to_dict() if observation else None,
                "step_index": step_index,
            },
            indent=2,
            sort_keys=True,
        )
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        candidates: list[dict[str, Any]] = []
        for raw_text in self._chat(messages, n=n_ver, seed=seed):
            try:
                payload = _parse_model_json(raw_text)
                candidates.append(
                    {
                        "payload": payload,
                        "raw_text": raw_text,
                        "generator_errors": list(actor_candidate.get("generator_errors", []))
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


def _render_final_summary(
    task_row: dict[str, Any],
    interpretation: Interpretation,
    state: Any,
) -> str:
    predicted_gene_ids = _flatten_predicted_gene_ids(state)
    if state.relationship_status == RelationshipStatus.INSUFFICIENT_SUPPORT:
        return "Final decision: no single shared mechanism is supported."
    if state.mechanistic_labels:
        label_name = state.mechanistic_labels[0].label_name
        return (
            f"Final decision: {state.relationship_status.value}. "
            f"Predicted genes={predicted_gene_ids}. "
            f"Top label={label_name}."
        )
    return (
        f"Final decision: {state.relationship_status.value}. "
        f"Predicted genes={predicted_gene_ids}. "
        f"Query={task_row['query_text']}"
    )


def _actor_templates_for_step(
    task_row: dict[str, Any],
    state: Any,
    *,
    trajectory_id: str,
    step_index: int,
    n_act: int,
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
                arguments={"seeds": current_gene_ids, "top_k": 10},
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
                tool_action = ToolAction(
                    tool_name=tool_name,
                    arguments=_safe_dict(arguments),
                    call_id=f"{trajectory_id}.step{step_index}.model_actor{actor_index}",
                )

    return ActorStep(reasoning_text=reasoning_text, tool_action=tool_action), errors


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
            "raw_verifier_response": verifier_candidate.get("raw_text", ""),
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


def _select_best_branch(branches: list[CandidateBranch]) -> CandidateBranch:
    return sorted(
        branches,
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
    seed: int = 0
    candidate_source: str = "heuristic"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if self.n_act <= 0:
            raise ValueError("n_act must be positive.")
        if self.n_ver <= 0:
            raise ValueError("n_ver must be positive.")
        if self.candidate_source not in {"heuristic", "model_vllm"}:
            raise ValueError("candidate_source must be one of: heuristic, model_vllm.")


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
    prior_actions: list[ToolAction] = []

    branch_pools: list[dict[str, Any]] = []
    trajectory_turns: list[TrajectoryTurn] = []
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
        if config.candidate_source == "model_vllm":
            if candidate_generator is None:
                raise ValueError("candidate_generator is required for model_vllm generation.")

            actor_candidates = candidate_generator.generate_actor_candidates(
                context,
                task_row=task_row,
                step_index=step_index,
                n_act=config.n_act,
                seed=trajectory_seed + step_index,
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
                if not _actor_candidate_is_usable(actor_step, actor_generation_errors):
                    rejected_model_errors.extend(actor_generation_errors)
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
                    branch.metadata["raw_actor_response"] = actor_candidate.get("raw_text", "")
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
            if not branches:
                templates = _actor_templates_for_step(
                    task_row,
                    state,
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    n_act=config.n_act,
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
                        branch.metadata["generator_errors"] = _unique(
                            rejected_model_errors or ["model_generator_returned_no_candidates"]
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
        else:
            templates = _actor_templates_for_step(
                task_row,
                state,
                trajectory_id=trajectory_id,
                step_index=step_index,
                n_act=config.n_act,
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
        selected_branch = _select_best_branch(branches)
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

        turn = TrajectoryTurn(
            trajectory_id=trajectory_id,
            step_index=step_index,
            prior_interpretation=clone_interpretation(interpretation),
            prior_state=clone_state(state),
            branch=selected_branch,
            selected=True,
            finding_text=_render_finding_text(selected_branch),
        )
        trajectory_turns.append(turn)

        interpretation = clone_interpretation(selected_branch.verifier_step.updated_interpretation)
        state = clone_state(selected_branch.verifier_step.updated_state)
        if selected_branch.actor_step.tool_action is not None:
            prior_actions.append(selected_branch.actor_step.tool_action)

    return {
        "trajectory_id": trajectory_id,
        "task_row": task_row,
        "turns": trajectory_turns,
        "branch_pools": branch_pools,
        "final_interpretation": interpretation,
        "final_state": state,
        "selected_branch_ids": selected_branch_ids,
        "rendered_summary": _render_final_summary(task_row, interpretation, state),
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
    final_summaries_path = out_dir / "final_summaries.jsonl"
    manifest_path = out_dir / "manifest.json"

    total_steps = 0
    total_branch_pools = 0
    total_branches = 0
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
            final_summaries_path.open("w", encoding="utf-8") as summaries_handle:

            for task_index, task_row in enumerate(task_rows):
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
                    },
                )

                progress_tracker.update(
                    message=f"Generated trajectory for {task_row['task_id']}.",
                    metrics={
                        "completed_tasks": task_index + 1,
                        "total_tasks": len(task_rows),
                        "current_task_id": task_row["task_id"],
                        "completed_steps": total_steps,
                        "completed_branch_pools": total_branch_pools,
                        "completed_branches": total_branches,
                    },
                )

        progress_tracker.start_stage(
            "write_manifest",
            message="Writing trajectory manifest.",
            metrics={
                "num_trajectories": len(task_rows),
                "total_steps": total_steps,
                "total_branch_pools": total_branch_pools,
                "total_branches": total_branches,
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
                "seed": config.seed,
            },
            "generator": {
                "candidate_source": config.candidate_source,
                "api_base": model_generator_config.api_base if model_generator_config else None,
                "model_name": candidate_generator.model_name if candidate_generator else None,
                "max_completion_tokens": (
                    model_generator_config.max_completion_tokens if model_generator_config else None
                ),
                "reasoning_effort": (
                    model_generator_config.reasoning_effort if model_generator_config else None
                ),
            },
            "task_selection": task_selection or {},
            "runtime": environment.describe(),
            "outputs": {
                "branch_pools": str(branch_pool_path),
                "trajectory_turns": str(trajectory_turns_path),
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
        "--generator-api-base",
        type=str,
        default=DEFAULT_GENERATOR_API_BASE,
        help="OpenAI-compatible API base for model-backed candidate generation.",
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
        default=700,
        help="Maximum completion tokens for actor and verifier generation calls.",
    )
    parser.add_argument(
        "--generator-reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
        help="Reasoning effort sent to GPT-OSS-compatible chat completions.",
    )
    parser.add_argument(
        "--generator-timeout-seconds",
        type=int,
        default=180,
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
        seed=args.seed,
        candidate_source=args.candidate_source,
    )
    model_generator_config = None
    if args.candidate_source == "model_vllm":
        model_generator_config = ModelGeneratorConfig(
            api_base=args.generator_api_base,
            model_name=args.generator_model,
            api_key=args.generator_api_key,
            api_key_env=args.generator_api_key_env,
            request_timeout_seconds=args.generator_timeout_seconds,
            temperature=args.generator_temperature,
            top_p=args.generator_top_p,
            max_completion_tokens=args.generator_max_completion_tokens,
            reasoning_effort=args.generator_reasoning_effort,
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
