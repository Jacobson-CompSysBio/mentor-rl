#!/usr/bin/env python3
"""Compile generated SFT candidates into plan-governed curriculum artifacts.

This module intentionally does not generate oracle facts.  It is the strict
post-generation boundary between any candidate generator and a publishable
pre-trajectory curriculum: render, deduplicate, audit, select, and write.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.validate_pretrajectory_sft_curriculum_plan import (
    CurriculumPlanValidationError,
    curriculum_plan_hash,
    load_curriculum_plan,
    validate_curriculum_plan,
)


ARTIFACT_SCHEMA_VERSION = "pretrajectory-sft-curriculum-artifacts-v1"
SPLITS = ("train", "val", "test")
STAGE_NAMES = (
    "stage1_entity_schema",
    "stage2_topology_priors",
    "stage3_open_book_vectors",
    "stage4_module_world_model",
    "stage5_structured_tools",
    "stage6_blend",
)
MODEL_TEXT_FIELDS = ("system", "question", "context", "answer")
ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?:\.[A-Za-z0-9_.-]+)?"
)
RELATIVE_REPO_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:data|config|scripts|runtime|checkpoints|logs)/"
    r"[A-Za-z0-9_./-]+"
)
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\]+\\)+[^\s\\]+")
RAW_CLI_RE = re.compile(r"(?<![A-Za-z0-9_])--[a-z][a-z0-9_-]*(?:\s|=)")
ENSEMBL_RE = re.compile(r"\bENSG[0-9]+(?:\.[0-9]+)?\b")


RenderCallback = Callable[[Any], Any]
ToolValidator = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class RenderedCandidate:
    record: dict[str, Any]
    canonical_object: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        value = self.record.get("metadata")
        return value if isinstance(value, dict) else {}

    @property
    def record_id(self) -> str:
        return str(self.metadata.get("record_id", ""))

    @property
    def split(self) -> str:
        return str(self.metadata.get("split", ""))

    @property
    def bucket(self) -> str:
        return str(self.metadata.get("mixture_bucket", ""))

    @property
    def family(self) -> str:
        return str(self.metadata.get("question_family", ""))

    @property
    def stage(self) -> str:
        return str(self.metadata.get("curriculum_stage", ""))

    @property
    def oracle_fact_id(self) -> str:
        return str(self.metadata.get("oracle_fact_id", ""))


class CurriculumArtifactError(ValueError):
    """Aggregate fatal compilation errors rather than failing one row at a time."""

    def __init__(self, errors: Sequence[Mapping[str, Any]]) -> None:
        self.errors = tuple(dict(error) for error in errors)
        details = "\n".join(
            f"  - {error.get('code', 'error')}: {error.get('message', '')}"
            for error in self.errors[:25]
        )
        if len(self.errors) > 25:
            details += f"\n  - ... {len(self.errors) - 25} additional errors"
        super().__init__(f"Curriculum artifact compilation failed with {len(self.errors)} error(s):\n{details}")


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    initial_capacity: int
    cost: int


class _Dinic:
    """Small deterministic integer max-flow implementation for coverage repair."""

    def __init__(self) -> None:
        self.graph: list[list[_FlowEdge]] = []

    def add_node(self) -> int:
        self.graph.append([])
        return len(self.graph) - 1

    def add_edge(self, source: int, target: int, capacity: int, *, cost: int = 0) -> _FlowEdge:
        if capacity < 0:
            raise ValueError("flow capacity must be nonnegative")
        forward = _FlowEdge(target, len(self.graph[target]), capacity, capacity, cost)
        reverse = _FlowEdge(source, len(self.graph[source]), 0, 0, -cost)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def max_flow(self, source: int, target: int) -> int:
        total = 0
        node_count = len(self.graph)
        while True:
            levels = [-1] * node_count
            levels[source] = 0
            queue = [source]
            for node in queue:
                for edge in self.graph[node]:
                    if edge.capacity > 0 and levels[edge.to] < 0:
                        levels[edge.to] = levels[node] + 1
                        queue.append(edge.to)
            if levels[target] < 0:
                return total
            offsets = [0] * node_count

            def send(node: int, amount: int) -> int:
                if node == target:
                    return amount
                while offsets[node] < len(self.graph[node]):
                    edge = self.graph[node][offsets[node]]
                    if edge.capacity > 0 and levels[edge.to] == levels[node] + 1:
                        pushed = send(edge.to, min(amount, edge.capacity))
                        if pushed:
                            edge.capacity -= pushed
                            self.graph[edge.to][edge.reverse].capacity += pushed
                            return pushed
                    offsets[node] += 1
                return 0

            while True:
                pushed = send(source, 1 << 60)
                if not pushed:
                    break
                total += pushed

    def min_cost_flow(self, source: int, target: int, required: int) -> tuple[int, int]:
        """Send up to ``required`` units with deterministic shortest augmenting paths."""

        sent = 0
        total_cost = 0
        node_count = len(self.graph)
        infinity = 1 << 60
        while sent < required:
            distance = [infinity] * node_count
            predecessor: list[tuple[int, int] | None] = [None] * node_count
            distance[source] = 0
            # Bellman-Ford is deliberate: reverse residual edges can have cost
            # -1, and this graph is small relative to candidate generation.
            for _ in range(node_count - 1):
                changed = False
                for node, edges in enumerate(self.graph):
                    if distance[node] == infinity:
                        continue
                    for edge_index, edge in enumerate(edges):
                        candidate_distance = distance[node] + edge.cost
                        if edge.capacity > 0 and candidate_distance < distance[edge.to]:
                            distance[edge.to] = candidate_distance
                            predecessor[edge.to] = (node, edge_index)
                            changed = True
                if not changed:
                    break
            if predecessor[target] is None:
                break
            amount = required - sent
            cursor = target
            while cursor != source:
                previous, edge_index = predecessor[cursor]  # type: ignore[misc]
                amount = min(amount, self.graph[previous][edge_index].capacity)
                cursor = previous
            cursor = target
            while cursor != source:
                previous, edge_index = predecessor[cursor]  # type: ignore[misc]
                edge = self.graph[previous][edge_index]
                edge.capacity -= amount
                self.graph[edge.to][edge.reverse].capacity += amount
                cursor = previous
            sent += amount
            total_cost += amount * distance[target]
        return sent, total_cost


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_order(value: Any, *, seed: int, namespace: str) -> str:
    return _stable_hash({"seed": seed, "namespace": namespace, "value": value})


def _estimate_tokens(text: str) -> int:
    """Conservative deterministic estimator used without a model tokenizer."""

    encoded_length = len(text.encode("utf-8"))
    byte_chunks = math.ceil(encoded_length / 4) if encoded_length else 0
    lexical = 0
    for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        token = match.group(0)
        if re.fullmatch(r"\w+", token, flags=re.UNICODE):
            lexical += max(1, math.ceil(len(token.encode("utf-8")) / 4))
        else:
            lexical += 1
    return max(byte_chunks, lexical)


def largest_remainder_quotas(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate an integer total exactly, using deterministic Hamilton quotas."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a nonnegative integer")
    if not weights:
        if total:
            raise ValueError("weights may not be empty when total is positive")
        return {}
    normalized: dict[str, float] = {}
    for name, raw_weight in weights.items():
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise ValueError(f"weight for {name!r} must be numeric")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError(f"weight for {name!r} must be nonnegative")
        normalized[str(name)] = weight
    weight_sum = sum(normalized.values())
    if weight_sum <= 0:
        raise ValueError("weights must have a positive sum")
    exact = {name: total * weight / weight_sum for name, weight in normalized.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(exact, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    assert sum(quotas.values()) == total
    return quotas


def _load_plan(plan: Path | str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(plan, (str, Path)):
        return load_curriculum_plan(Path(plan))
    payload = copy.deepcopy(dict(plan))
    errors = validate_curriculum_plan(payload)
    if errors:
        raise CurriculumPlanValidationError(Path("<in-memory-plan>"), errors)
    return payload


def _extract_record_and_object(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        if "record" in value:
            return value.get("record"), value.get("canonical_object")
        return value, value.get("canonical_object")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    record = getattr(value, "record", None)
    canonical_object = getattr(value, "canonical_object", None)
    return record, canonical_object


def _render_candidate(raw: Any, render_candidate: RenderCallback | None) -> RenderedCandidate:
    rendered = render_candidate(raw) if render_candidate is not None else raw
    if render_candidate is None and callable(getattr(raw, "render", None)):
        rendered = raw.render()
    record, canonical_object = _extract_record_and_object(rendered)
    if canonical_object is None:
        _, canonical_object = _extract_record_and_object(raw)
    if not isinstance(record, Mapping):
        raise ValueError("renderer did not produce a record object")
    if not isinstance(canonical_object, Mapping):
        raise ValueError("candidate did not provide a canonical_object")
    record_copy = copy.deepcopy(dict(record))
    object_copy = copy.deepcopy(dict(canonical_object))
    answer = record_copy.get("answer")
    if isinstance(answer, (Mapping, list)):
        record_copy["answer"] = _canonical_json(answer)
    metadata = record_copy.get("metadata")
    object_id = object_copy.get("object_id")
    if isinstance(metadata, dict) and isinstance(object_id, str) and object_id:
        metadata.setdefault("canonical_object_id", object_id)
    return RenderedCandidate(record=record_copy, canonical_object=object_copy)


def _model_text(record: Mapping[str, Any]) -> str:
    return "\n".join(str(record.get(field, "")) for field in MODEL_TEXT_FIELDS)


def _text_fingerprint(record: Mapping[str, Any]) -> str:
    return _stable_hash([str(record.get(field, "")) for field in MODEL_TEXT_FIELDS])


def _near_text_fingerprint(record: Mapping[str, Any]) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", _model_text(record).lower()).strip()
    return _stable_hash(normalized)


def _tool_validation_result(result: Any) -> tuple[bool, str | None]:
    if isinstance(result, bool):
        return result, None
    if isinstance(result, tuple) and len(result) == 2:
        return bool(result[0]), str(result[1]) if result[1] else None
    if isinstance(result, Mapping):
        valid = result.get("valid") is True or result.get("passed") is True
        reason = result.get("reason") or result.get("message")
        return valid, str(reason) if reason else None
    return False, "tool validator returned an unsupported result"


def _budget_measurement(record: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    prompt_text = "\n".join(
        str(record.get(field, "")) for field in ("system", "question", "context")
    )
    answer = str(record.get("answer", ""))
    prompt_tokens = _estimate_tokens(prompt_text)
    answer_tokens = _estimate_tokens(answer)
    total_tokens = prompt_tokens + answer_tokens
    answer_characters = len(answer)
    violations: list[str] = []
    if prompt_tokens > int(profile["max_prompt_tokens"]):
        violations.append("max_prompt_tokens")
    if answer_tokens > int(profile["max_answer_tokens"]):
        violations.append("max_answer_tokens")
    if total_tokens > int(profile["max_total_tokens"]):
        violations.append("max_total_tokens")
    if answer_characters > int(profile["max_answer_characters"]):
        violations.append("max_answer_characters")
    return {
        "prompt_token_estimate": prompt_tokens,
        "answer_token_estimate": answer_tokens,
        "total_token_estimate": total_tokens,
        "answer_character_count": answer_characters,
        "violations": violations,
        "passed": not violations,
    }


def _candidate_audit_issues(
    candidate: RenderedCandidate,
    *,
    plan: Mapping[str, Any],
    family_by_name: Mapping[str, Mapping[str, Any]],
    stage_by_name: Mapping[str, Mapping[str, Any]],
    tool_validator: ToolValidator | None,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    record = candidate.record
    metadata = candidate.metadata
    record_id = candidate.record_id or None

    def add(code: str, message: str, **context: Any) -> None:
        issue: dict[str, Any] = {"code": code, "message": message}
        if record_id:
            issue["record_id"] = record_id
        if context:
            issue["context"] = context
        issues.append(issue)

    for field in ("system", "question", "answer"):
        if not isinstance(record.get(field), str) or not str(record.get(field, "")).strip():
            add("missing_model_text", f"record field {field!r} must be a non-empty string", field=field)
    if not isinstance(record.get("metadata"), dict):
        add("missing_metadata", "record metadata must be an object")
        return issues
    required_metadata = plan["record_contract"]["required_metadata_fields"]
    for field in required_metadata:
        if field not in metadata or metadata[field] is None:
            add("missing_metadata_field", f"required metadata field {field!r} is missing", field=field)
    if not isinstance(metadata.get("record_id"), str) or not metadata.get("record_id"):
        add("invalid_record_id", "metadata.record_id must be a non-empty string")
    if metadata.get("schema_version") != plan["dataset_schema_version"]:
        add(
            "wrong_dataset_schema",
            "metadata.schema_version does not match the curriculum plan",
            observed=metadata.get("schema_version"),
            expected=plan["dataset_schema_version"],
        )
    if candidate.split not in SPLITS:
        add("invalid_split", f"metadata.split must be one of {SPLITS}", observed=candidate.split)
    if not candidate.oracle_fact_id:
        add("invalid_oracle_fact_id", "metadata.oracle_fact_id must be a non-empty string")
    family = family_by_name.get(candidate.family)
    if family is None:
        add("unknown_question_family", "metadata.question_family is not in the curriculum plan")
    stage = stage_by_name.get(candidate.stage)
    if stage is None or candidate.stage == "stage6_blend":
        add("invalid_primary_stage", "generated records must have a primary curriculum stage from 1 through 5")
    if family is not None and candidate.stage != family.get("primary_stage"):
        add(
            "family_stage_mismatch",
            "record stage does not equal the family's plan-defined primary stage",
            expected=family.get("primary_stage"),
            observed=candidate.stage,
        )
    book_mode = metadata.get("book_mode")
    if family is not None and book_mode not in family.get("allowed_book_modes", []):
        add("family_book_mode_mismatch", "book mode is not allowed for this question family")
    if stage is not None and book_mode not in stage.get("allowed_book_modes", []):
        add("stage_book_mode_mismatch", "book mode is not allowed in this primary stage")
    if candidate.bucket not in plan["mixture"]["content_buckets"]:
        add("unknown_mixture_bucket", "metadata.mixture_bucket is not in the curriculum plan")
    if family is not None and candidate.bucket != family.get("mixture_bucket"):
        add(
            "family_bucket_mismatch",
            "record mixture bucket does not match its question-family contract",
            expected=family.get("mixture_bucket"),
            observed=candidate.bucket,
        )
    budget_name = metadata.get("context_budget_profile")
    profile = plan["context_budget_profiles"].get(budget_name)
    if not isinstance(profile, Mapping):
        add("unknown_budget_profile", "context_budget_profile is not defined by the plan")
    else:
        if stage is not None and budget_name not in stage.get("allowed_budget_profiles", []):
            add("stage_budget_profile_mismatch", "budget profile is not allowed in this primary stage")
        measurement = _budget_measurement(record, profile)
        metadata["budget_measurement"] = measurement
        if measurement["violations"]:
            add(
                "budget_exceeded",
                "record exceeds its declared context-budget profile",
                profile=budget_name,
                violations=measurement["violations"],
            )
    if metadata.get("answer_format") == "json":
        try:
            json.loads(str(record.get("answer", "")))
        except json.JSONDecodeError:
            add("invalid_json_answer", "answer_format=json requires a valid JSON answer")
    for field in ("layer_ids", "layer_families", "evidence_handles"):
        if field in metadata and not isinstance(metadata[field], list):
            add("invalid_metadata_type", f"metadata.{field} must be a list", field=field)
    if "provenance" in metadata and not isinstance(metadata["provenance"], (dict, list)):
        add("invalid_metadata_type", "metadata.provenance must be an object or list", field="provenance")
    expected_multiplex = plan["graph_contract"]["multiplex_id"]
    if metadata.get("multiplex_id") != expected_multiplex:
        add(
            "multiplex_identity_mismatch",
            "record multiplex_id differs from the plan oracle",
            expected=expected_multiplex,
            observed=metadata.get("multiplex_id"),
        )
    model_text = _model_text(record)
    if "file://" in model_text or ABSOLUTE_PATH_RE.search(model_text) or RELATIVE_REPO_PATH_RE.search(model_text) or WINDOWS_PATH_RE.search(model_text):
        add("raw_path_in_model_text", "model-facing text contains a raw filesystem path")
    if RAW_CLI_RE.search(model_text):
        add("raw_cli_in_model_text", "model-facing text contains a raw CLI option")
    object_id = candidate.canonical_object.get("object_id")
    if not isinstance(object_id, str) or not object_id:
        add("invalid_canonical_object", "canonical object requires a non-empty object_id")
    elif metadata.get("canonical_object_id") != object_id:
        add("canonical_object_reference_mismatch", "record canonical_object_id does not match its object")
    if not isinstance(candidate.canonical_object.get("payload"), Mapping):
        add("invalid_canonical_payload", "canonical object payload must be an object")
    if book_mode == "tool_call":
        if tool_validator is None:
            tool_valid = False
            reason = "a live tool_validator callback is required for every tool-call row"
        else:
            try:
                tool_valid, reason = _tool_validation_result(tool_validator(record))
            except Exception as exc:  # validation failures belong in the report
                tool_valid, reason = False, f"tool validator raised {type(exc).__name__}: {exc}"
            metadata["tool_schema_valid"] = tool_valid
        if not tool_valid:
            add("invalid_tool_schema", "tool-call record did not pass live schema validation", reason=reason)
    return issues


def _deduplicate_and_check_leakage(
    candidates: Sequence[RenderedCandidate],
) -> tuple[list[RenderedCandidate], dict[str, Any], list[dict[str, Any]]]:
    fatal: list[dict[str, Any]] = []
    by_record_id: dict[str, RenderedCandidate] = {}
    duplicate_record_ids = 0
    for candidate in sorted(candidates, key=lambda item: (item.record_id, _stable_hash(item.record))):
        record_id = candidate.record_id
        if not record_id:
            # Missing IDs are filtered by the row audit; retain them so they are counted.
            by_record_id[f"<missing>:{len(by_record_id)}"] = candidate
            continue
        previous = by_record_id.get(record_id)
        if previous is None:
            by_record_id[record_id] = candidate
            continue
        duplicate_record_ids += 1
        if _stable_hash([previous.record, previous.canonical_object]) != _stable_hash(
            [candidate.record, candidate.canonical_object]
        ):
            fatal.append(
                {
                    "code": "conflicting_duplicate_record_id",
                    "message": f"record_id {record_id!r} has non-identical definitions",
                    "record_id": record_id,
                }
            )
    record_deduped = list(by_record_id.values())
    fact_splits: dict[str, set[str]] = defaultdict(set)
    exact_splits: dict[str, set[str]] = defaultdict(set)
    near_splits: dict[str, set[str]] = defaultdict(set)
    for candidate in record_deduped:
        if candidate.oracle_fact_id and candidate.split:
            fact_splits[candidate.oracle_fact_id].add(candidate.split)
        if candidate.split:
            exact_splits[_text_fingerprint(candidate.record)].add(candidate.split)
            near_splits[_near_text_fingerprint(candidate.record)].add(candidate.split)
    fact_leaks = {
        key: sorted(splits) for key, splits in fact_splits.items() if len(splits) > 1
    }
    exact_leaks = {
        key: sorted(splits) for key, splits in exact_splits.items() if len(splits) > 1
    }
    near_leaks = {
        key: sorted(splits) for key, splits in near_splits.items() if len(splits) > 1
    }
    for fact_id, splits in sorted(fact_leaks.items()):
        fatal.append(
            {
                "code": "oracle_fact_cross_split",
                "message": f"oracle fact {fact_id!r} occurs in multiple splits: {splits}",
                "oracle_fact_id": fact_id,
                "splits": splits,
            }
        )
    for fingerprint, splits in sorted(exact_leaks.items()):
        fatal.append(
            {
                "code": "exact_duplicate_cross_split",
                "message": f"exact model-text duplicate occurs in multiple splits: {splits}",
                "fingerprint": fingerprint,
                "splits": splits,
            }
        )
    # Exact duplicates are also near duplicates; report only novel near leaks.
    for fingerprint, splits in sorted(near_leaks.items()):
        if fingerprint not in exact_leaks:
            fatal.append(
                {
                    "code": "near_duplicate_cross_split",
                    "message": f"normalized model-text duplicate occurs in multiple splits: {splits}",
                    "fingerprint": fingerprint,
                    "splits": splits,
                }
            )
    by_text: dict[tuple[str, str], RenderedCandidate] = {}
    duplicate_text_rows = 0
    for candidate in sorted(record_deduped, key=lambda item: item.record_id):
        key = (candidate.split, _text_fingerprint(candidate.record))
        if key in by_text:
            duplicate_text_rows += 1
            continue
        by_text[key] = candidate
    deduped = list(by_text.values())
    report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "passed": not fatal,
        "duplicate_record_id_count": duplicate_record_ids,
        "duplicate_model_text_count": duplicate_text_rows,
        "oracle_fact_cross_split_count": len(fact_leaks),
        "exact_duplicate_cross_split_count": len(exact_leaks),
        "near_duplicate_cross_split_count": len(near_leaks),
        "oracle_fact_cross_split": fact_leaks,
        "exact_duplicate_cross_split": exact_leaks,
        "near_duplicate_cross_split": near_leaks,
    }
    return deduped, report, fatal


def _optional_group_leakage(
    candidates: Sequence[RenderedCandidate],
    keys: Sequence[str],
) -> tuple[dict[str, dict[str, list[str]]], list[dict[str, Any]]]:
    leaks: dict[str, dict[str, list[str]]] = {}
    fatal: list[dict[str, Any]] = []
    for key in keys:
        values: dict[str, set[str]] = defaultdict(set)
        for candidate in candidates:
            value = candidate.metadata.get(key)
            if isinstance(value, str) and value and candidate.split:
                values[value].add(candidate.split)
        key_leaks = {value: sorted(splits) for value, splits in values.items() if len(splits) > 1}
        if key_leaks:
            leaks[key] = key_leaks
        for value, splits in sorted(key_leaks.items()):
            fatal.append(
                {
                    "code": "group_cross_split",
                    "message": f"{key}={value!r} occurs in multiple splits: {splits}",
                    "group_key": key,
                    "group_id": value,
                    "splits": splits,
                }
            )
    return leaks, fatal


def _round_robin_select(
    candidates: Sequence[RenderedCandidate],
    target: int,
    *,
    seed: int,
    namespace: str,
    group_key: Callable[[RenderedCandidate], str] = lambda candidate: candidate.family,
) -> list[RenderedCandidate]:
    if target < 0:
        raise ValueError("round-robin target must be nonnegative")
    groups: dict[str, list[RenderedCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[group_key(candidate)].append(candidate)
    for name in groups:
        groups[name].sort(
            key=lambda candidate: _stable_order(
                candidate.record_id, seed=seed, namespace=f"{namespace}:{name}"
            )
        )
    group_order = sorted(
        groups,
        key=lambda name: _stable_order(name, seed=seed, namespace=f"{namespace}:groups"),
    )
    offsets = {name: 0 for name in group_order}
    chosen: list[RenderedCandidate] = []
    while len(chosen) < target:
        progressed = False
        for name in group_order:
            offset = offsets[name]
            if offset >= len(groups[name]):
                continue
            chosen.append(groups[name][offset])
            offsets[name] += 1
            progressed = True
            if len(chosen) == target:
                break
        if not progressed:
            break
    return chosen


def _select_main_records(
    candidates: Sequence[RenderedCandidate],
    *,
    target_counts: Mapping[str, int],
    bucket_weights: Mapping[str, float],
    seed: int,
) -> tuple[list[RenderedCandidate], dict[str, Any], list[dict[str, Any]]]:
    selected: list[RenderedCandidate] = []
    report: dict[str, Any] = {"splits": {}}
    fatal: list[dict[str, Any]] = []
    for split in SPLITS:
        target = int(target_counts[split])
        quotas = largest_remainder_quotas(target, bucket_weights)
        split_report: dict[str, Any] = {
            "target": target,
            "available": sum(candidate.split == split for candidate in candidates),
            "bucket_quotas": quotas,
            "buckets": {},
        }
        split_selected: list[RenderedCandidate] = []
        for bucket, quota in quotas.items():
            pool = [
                candidate
                for candidate in candidates
                if candidate.split == split and candidate.bucket == bucket
            ]
            chosen = _round_robin_select(
                pool,
                quota,
                seed=seed,
                namespace=f"main:{split}:{bucket}",
            )
            split_report["buckets"][bucket] = {
                "requested": quota,
                "available": len(pool),
                "selected": len(chosen),
                "selected_by_family": dict(sorted(Counter(item.family for item in chosen).items())),
            }
            if len(chosen) != quota:
                fatal.append(
                    {
                        "code": "mixture_bucket_underfill",
                        "message": (
                            f"split {split!r} bucket {bucket!r} requested {quota} "
                            f"records but only {len(chosen)} valid deduplicated candidates were available"
                        ),
                        "split": split,
                        "bucket": bucket,
                        "requested": quota,
                        "available": len(pool),
                    }
                )
            split_selected.extend(chosen)
        if len(split_selected) != target:
            fatal.append(
                {
                    "code": "split_count_underfill",
                    "message": f"split {split!r} requested {target} records but selected {len(split_selected)}",
                    "split": split,
                    "requested": target,
                    "selected": len(split_selected),
                }
            )
        split_report["selected"] = len(split_selected)
        report["splits"][split] = split_report
        selected.extend(split_selected)
    return selected, report, fatal


def _refresh_selection_report(
    report: dict[str, Any],
    selected: Sequence[RenderedCandidate],
) -> None:
    for split in SPLITS:
        split_selected = [candidate for candidate in selected if candidate.split == split]
        report["splits"][split]["selected"] = len(split_selected)
        for bucket, payload in report["splits"][split]["buckets"].items():
            bucket_selected = [
                candidate for candidate in split_selected if candidate.bucket == bucket
            ]
            payload["selected"] = len(bucket_selected)
            payload["selected_by_family"] = dict(
                sorted(Counter(candidate.family for candidate in bucket_selected).items())
            )


def _cell_as_dict(cell: tuple[str, ...], fields: Sequence[str]) -> dict[str, str]:
    return dict(zip(fields, cell, strict=True))


def _material_cell_state(
    valid_pool: Sequence[RenderedCandidate],
    selected: Sequence[RenderedCandidate],
    minimum: int,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], set[tuple[str, ...]]]:
    valid_counts: Counter[tuple[str, ...]] = Counter(_cross_cell_key(item) for item in valid_pool)
    selected_counts: Counter[tuple[str, ...]] = Counter(_cross_cell_key(item) for item in selected)
    material = {cell for cell, count in valid_counts.items() if count >= minimum}
    return valid_counts, selected_counts, material


def _repair_material_cross_cells(
    valid_pool: Sequence[RenderedCandidate],
    selected: Sequence[RenderedCandidate],
    selection_report: dict[str, Any],
    *,
    minimum_cell_count: int,
    minimum_family_count: int,
    family_names: Sequence[str],
    cross_cell_fields: Sequence[str],
    seed: int,
) -> tuple[list[RenderedCandidate], dict[str, Any], list[dict[str, Any]]]:
    """Repair lower-bound coverage without changing a split/bucket quota.

    The feasibility model is an integer circulation:
    ``split+bucket -> cross-cell -> question-family``.  Exact partition quotas,
    material-cell lower bounds, and family lower bounds are solved together.
    Reconstructing rows retains already-selected candidates wherever the
    resulting partition/cell allocation permits, so every change is a literal
    within-partition swap.
    """

    if minimum_cell_count <= 0 or minimum_family_count <= 0:
        raise ValueError("coverage-repair minima must be positive")
    valid_counts, before_cell_counts, material_cells = _material_cell_state(
        valid_pool, selected, minimum_cell_count
    )
    before_family_counts = Counter(candidate.family for candidate in selected)
    underfilled_before = {
        cell for cell in material_cells if before_cell_counts[cell] < minimum_cell_count
    }
    family_underfilled_before = {
        family
        for family in family_names
        if before_family_counts[family] < minimum_family_count
    }
    base_report: dict[str, Any] = {
        "minimum_selected_per_material_cross_cell": minimum_cell_count,
        "minimum_selected_per_required_family": minimum_family_count,
        "valid_material_cell_count": len(material_cells),
        "underfilled_material_cell_count_before": len(underfilled_before),
        "underfilled_family_count_before": len(family_underfilled_before),
        "applied": bool(underfilled_before or family_underfilled_before),
        "swap_count": 0,
        "swaps_by_split_bucket": {},
        "swap_examples": [],
    }
    if not underfilled_before and not family_underfilled_before:
        base_report["underfilled_material_cell_count_after"] = 0
        base_report["underfilled_family_count_after"] = 0
        base_report["material_cells"] = [
            {
                **_cell_as_dict(cell, cross_cell_fields),
                "valid": valid_counts[cell],
                "selected_before": before_cell_counts[cell],
                "selected_after": before_cell_counts[cell],
                "minimum": minimum_cell_count,
                "material": True,
                "underfilled_before": False,
                "underfilled_after": False,
            }
            for cell in sorted(material_cells)
        ]
        selection_report["post_selection_repair"] = base_report
        return list(selected), base_report, []

    partition_pools: dict[
        tuple[str, str], dict[tuple[str, ...], list[RenderedCandidate]]
    ] = defaultdict(lambda: defaultdict(list))
    for candidate in valid_pool:
        partition_pools[(candidate.split, candidate.bucket)][_cross_cell_key(candidate)].append(
            candidate
        )
    selected_ids = {candidate.record_id for candidate in selected}
    selected_partition_cell_counts: Counter[
        tuple[tuple[str, str], tuple[str, ...]]
    ] = Counter(
        ((candidate.split, candidate.bucket), _cross_cell_key(candidate))
        for candidate in selected
    )
    partition_quotas = Counter((candidate.split, candidate.bucket) for candidate in selected)
    family_availability = Counter(candidate.family for candidate in valid_pool)
    impossible: list[dict[str, Any]] = []
    for family in family_names:
        if family_availability[family] < minimum_family_count:
            impossible.append(
                {
                    "code": "material_cross_cell_repair_infeasible",
                    "message": (
                        f"family {family!r} has {family_availability[family]} valid candidates, "
                        f"below required minimum {minimum_family_count}"
                    ),
                    "question_family": family,
                    "available": family_availability[family],
                    "required": minimum_family_count,
                }
            )
    if impossible:
        selection_report["post_selection_repair"] = base_report
        return list(selected), base_report, impossible

    flow = _Dinic()
    node_ids: dict[Any, int] = {}

    def node(label: Any) -> int:
        if label not in node_ids:
            node_ids[label] = flow.add_node()
        return node_ids[label]

    source = node(("source",))
    sink = node(("sink",))
    demand: Counter[int] = Counter()

    def add_lower_edge(
        source_node: int,
        target_node: int,
        lower: int,
        upper: int,
        *,
        cost: int = 0,
    ) -> _FlowEdge:
        if lower < 0 or upper < lower:
            raise ValueError(f"invalid lower-bound edge {lower}..{upper}")
        edge = flow.add_edge(source_node, target_node, upper - lower, cost=cost)
        demand[source_node] -= lower
        demand[target_node] += lower
        return edge

    for partition, quota in sorted(partition_quotas.items()):
        add_lower_edge(source, node(("partition", *partition)), quota, quota)

    partition_cell_edges: dict[
        tuple[tuple[str, str], tuple[str, ...]], list[_FlowEdge]
    ] = defaultdict(list)
    for partition in sorted(partition_pools):
        partition_node = node(("partition", *partition))
        for cell, pool in sorted(partition_pools[partition].items()):
            cell_node = node(("cell", *cell))
            current = selected_partition_cell_counts[(partition, cell)]
            available = len(pool)
            # Insertion order is intentional: Dinic consumes the zero-change
            # capacity before capacity that requires replacing a selected row.
            if current:
                partition_cell_edges[(partition, cell)].append(
                    add_lower_edge(partition_node, cell_node, 0, current, cost=0)
                )
            if available > current:
                partition_cell_edges[(partition, cell)].append(
                    add_lower_edge(
                        partition_node,
                        cell_node,
                        0,
                        available - current,
                        cost=1,
                    )
                )

    for cell, available in sorted(valid_counts.items()):
        family = cell[1]
        lower = minimum_cell_count if cell in material_cells else 0
        add_lower_edge(
            node(("cell", *cell)),
            node(("family", family)),
            lower,
            available,
        )
    for family in sorted(family_names):
        add_lower_edge(
            node(("family", family)),
            sink,
            minimum_family_count,
            family_availability[family],
        )

    total_selected = len(selected)
    add_lower_edge(sink, source, 0, total_selected)
    super_source = node(("super_source",))
    super_sink = node(("super_sink",))
    required_flow = 0
    for node_id, balance in sorted(demand.items()):
        if balance > 0:
            flow.add_edge(super_source, node_id, balance)
            required_flow += balance
        elif balance < 0:
            flow.add_edge(node_id, super_sink, -balance)
    realized_flow, repair_cost = flow.min_cost_flow(
        super_source, super_sink, required_flow
    )
    if realized_flow != required_flow:
        errors = [
            {
                "code": "material_cross_cell_repair_infeasible",
                "message": (
                    "No quota-preserving selection satisfies every material cross-cell and "
                    f"family minimum (circulation {realized_flow}/{required_flow})."
                ),
                "required_flow": required_flow,
                "realized_flow": realized_flow,
                "underfilled_material_cells_before": [
                    _cell_as_dict(cell, cross_cell_fields) for cell in sorted(underfilled_before)
                ],
                "underfilled_families_before": sorted(family_underfilled_before),
            }
        ]
        selection_report["post_selection_repair"] = base_report
        return list(selected), base_report, errors

    desired_counts: dict[
        tuple[tuple[str, str], tuple[str, ...]], int
    ] = {}
    for key, edges in partition_cell_edges.items():
        desired_counts[key] = sum(edge.initial_capacity - edge.capacity for edge in edges)

    repaired: list[RenderedCandidate] = []
    for partition in sorted(partition_pools):
        for cell, pool in sorted(partition_pools[partition].items()):
            target = desired_counts.get((partition, cell), 0)
            current = sorted(
                (candidate for candidate in pool if candidate.record_id in selected_ids),
                key=lambda candidate: _stable_order(
                    candidate.record_id,
                    seed=seed,
                    namespace=f"coverage-repair:retain:{partition}:{cell}",
                ),
            )
            incoming = sorted(
                (candidate for candidate in pool if candidate.record_id not in selected_ids),
                key=lambda candidate: _stable_order(
                    candidate.record_id,
                    seed=seed,
                    namespace=f"coverage-repair:add:{partition}:{cell}",
                ),
            )
            chosen = current[:target]
            if len(chosen) < target:
                chosen.extend(incoming[: target - len(chosen)])
            if len(chosen) != target:
                impossible.append(
                    {
                        "code": "material_cross_cell_repair_internal_underfill",
                        "message": (
                            f"repair allocation requested {target} candidates for {partition}/{cell} "
                            f"but reconstructed {len(chosen)}"
                        ),
                    }
                )
            repaired.extend(chosen)
    if impossible:
        selection_report["post_selection_repair"] = base_report
        return list(selected), base_report, impossible

    repaired.sort(key=lambda candidate: (candidate.split, candidate.bucket, candidate.record_id))
    after_valid_counts, after_cell_counts, after_material = _material_cell_state(
        valid_pool, repaired, minimum_cell_count
    )
    assert after_valid_counts == valid_counts
    assert after_material == material_cells
    after_family_counts = Counter(candidate.family for candidate in repaired)
    underfilled_after = {
        cell for cell in material_cells if after_cell_counts[cell] < minimum_cell_count
    }
    family_underfilled_after = {
        family
        for family in family_names
        if after_family_counts[family] < minimum_family_count
    }
    final_errors: list[dict[str, Any]] = []
    if underfilled_after:
        final_errors.append(
            {
                "code": "material_cross_cell_underfill",
                "message": (
                    f"coverage repair left {len(underfilled_after)} material cross-cells underfilled"
                ),
                "underfilled_cells": [
                    _cell_as_dict(cell, cross_cell_fields) for cell in sorted(underfilled_after)
                ],
            }
        )
    if family_underfilled_after:
        final_errors.append(
            {
                "code": "question_family_underfill_after_repair",
                "message": (
                    f"coverage repair left {len(family_underfilled_after)} families underfilled"
                ),
                "underfilled_families": sorted(family_underfilled_after),
            }
        )
    repaired_partition_counts = Counter((item.split, item.bucket) for item in repaired)
    if repaired_partition_counts != partition_quotas:
        final_errors.append(
            {
                "code": "coverage_repair_changed_partition_quota",
                "message": "coverage repair changed one or more split/mixture-bucket quotas",
            }
        )

    repaired_ids = {candidate.record_id for candidate in repaired}
    swap_examples: list[dict[str, Any]] = []
    swaps_by_partition: dict[str, int] = {}
    for partition in sorted(partition_quotas):
        before_ids = {
            candidate.record_id
            for candidate in selected
            if (candidate.split, candidate.bucket) == partition
        }
        after_ids = {
            candidate.record_id
            for candidate in repaired
            if (candidate.split, candidate.bucket) == partition
        }
        outgoing = sorted(before_ids - after_ids)
        incoming = sorted(after_ids - before_ids)
        if len(outgoing) != len(incoming):
            final_errors.append(
                {
                    "code": "coverage_repair_unpaired_swap",
                    "message": f"repair produced unpaired replacements in partition {partition}",
                }
            )
            continue
        if incoming:
            partition_name = f"{partition[0]}|{partition[1]}"
            swaps_by_partition[partition_name] = len(incoming)
            for removed, added in zip(outgoing, incoming, strict=True):
                if len(swap_examples) < 100:
                    swap_examples.append(
                        {
                            "split": partition[0],
                            "mixture_bucket": partition[1],
                            "removed_record_id": removed,
                            "added_record_id": added,
                        }
                    )
    base_report.update(
        {
            "swap_count": len(repaired_ids - selected_ids),
            "swaps_by_split_bucket": swaps_by_partition,
            "swap_examples": swap_examples,
            "underfilled_material_cell_count_after": len(underfilled_after),
            "underfilled_family_count_after": len(family_underfilled_after),
            "minimum_cost_repair_units": repair_cost,
            "material_cells": [
                {
                    **_cell_as_dict(cell, cross_cell_fields),
                    "valid": valid_counts[cell],
                    "selected_before": before_cell_counts[cell],
                    "selected_after": after_cell_counts[cell],
                    "minimum": minimum_cell_count,
                    "material": True,
                    "underfilled_before": cell in underfilled_before,
                    "underfilled_after": cell in underfilled_after,
                }
                for cell in sorted(material_cells)
            ],
        }
    )
    _refresh_selection_report(selection_report, repaired)
    selection_report["post_selection_repair"] = base_report
    return repaired, base_report, final_errors


def _family_underfill_errors(
    selected: Sequence[RenderedCandidate],
    *,
    family_names: Sequence[str],
    minimum: int,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = Counter(candidate.family for candidate in selected)
    errors: list[dict[str, Any]] = []
    for family in family_names:
        observed = counts.get(family, 0)
        if observed < minimum:
            errors.append(
                {
                    "code": "question_family_underfill",
                    "message": (
                        f"question family {family!r} requires at least {minimum} selected records "
                        f"but has {observed}"
                    ),
                    "question_family": family,
                    "required": minimum,
                    "observed": observed,
                }
            )
    return dict(sorted(counts.items())), errors


def _annotate_training_row(
    candidate: RenderedCandidate,
    *,
    training_stage: str,
    role: str,
) -> dict[str, Any]:
    record = copy.deepcopy(candidate.record)
    metadata = record["metadata"]
    metadata["training_stage"] = training_stage
    metadata["curriculum_role"] = role
    metadata["source_curriculum_stage"] = candidate.stage
    return record


def _build_stage_files(
    selected: Sequence[RenderedCandidate],
    valid_pool: Sequence[RenderedCandidate],
    *,
    plan: Mapping[str, Any],
    target_counts: Mapping[str, int],
    seed: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any], list[dict[str, Any]], set[str]]:
    stage_plan = {stage["name"]: stage for stage in plan["stages"]}
    stage_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        stage: {split: [] for split in SPLITS} for stage in STAGE_NAMES
    }
    report: dict[str, Any] = {"stages": {stage: {} for stage in STAGE_NAMES}}
    fatal: list[dict[str, Any]] = []
    referenced_ids: set[str] = set()
    stage_index = {stage: index for index, stage in enumerate(STAGE_NAMES, start=1)}
    for stage in STAGE_NAMES[:5]:
        replay_fraction = float(stage_plan[stage]["earlier_stage_replay_fraction"])
        for split in SPLITS:
            primary = [
                candidate for candidate in selected if candidate.split == split and candidate.stage == stage
            ]
            earlier = [
                candidate
                for candidate in selected
                if candidate.split == split and stage_index.get(candidate.stage, 99) < stage_index[stage]
            ]
            if replay_fraction <= 0 or not primary:
                replay_target = 0
            else:
                replay_target = int(round(len(primary) * replay_fraction / (1.0 - replay_fraction)))
            replay = _round_robin_select(
                earlier,
                replay_target,
                seed=seed,
                namespace=f"replay:{stage}:{split}",
                group_key=lambda candidate: f"{candidate.stage}:{candidate.family}",
            )
            if len(replay) != replay_target:
                fatal.append(
                    {
                        "code": "stage_replay_underfill",
                        "message": (
                            f"{stage}/{split} requested {replay_target} replay rows but found {len(replay)}"
                        ),
                        "stage": stage,
                        "split": split,
                        "requested": replay_target,
                        "selected": len(replay),
                    }
                )
            rows = [
                _annotate_training_row(candidate, training_stage=stage, role="primary")
                for candidate in primary
            ] + [
                _annotate_training_row(candidate, training_stage=stage, role="replay")
                for candidate in replay
            ]
            rows.sort(key=lambda row: str(row["metadata"]["record_id"]))
            stage_rows[stage][split] = rows
            referenced_ids.update(str(row["metadata"]["canonical_object_id"]) for row in rows)
            report["stages"][stage][split] = {
                "primary_count": len(primary),
                "replay_count": len(replay),
                "record_count": len(rows),
                "realized_replay_fraction": len(replay) / len(rows) if rows else 0.0,
            }
    blend_weights = stage_plan["stage6_blend"]["source_stage_weights"]
    for split in SPLITS:
        quotas = largest_remainder_quotas(int(target_counts[split]), blend_weights)
        blend: list[RenderedCandidate] = []
        counts: dict[str, int] = {}
        for source_stage, quota in quotas.items():
            pool = [
                candidate
                for candidate in valid_pool
                if candidate.split == split and candidate.stage == source_stage
            ]
            chosen = _round_robin_select(
                pool,
                quota,
                seed=seed,
                namespace=f"blend:{split}:{source_stage}",
            )
            counts[source_stage] = len(chosen)
            if len(chosen) != quota:
                fatal.append(
                    {
                        "code": "stage6_source_underfill",
                        "message": (
                            f"stage6/{split} requested {quota} rows from {source_stage} "
                            f"but found {len(chosen)}"
                        ),
                        "split": split,
                        "source_stage": source_stage,
                        "requested": quota,
                        "selected": len(chosen),
                    }
                )
            blend.extend(chosen)
        rows = [
            _annotate_training_row(candidate, training_stage="stage6_blend", role="consolidation")
            for candidate in blend
        ]
        rows.sort(key=lambda row: str(row["metadata"]["record_id"]))
        stage_rows["stage6_blend"][split] = rows
        referenced_ids.update(str(row["metadata"]["canonical_object_id"]) for row in rows)
        report["stages"]["stage6_blend"][split] = {
            "target": int(target_counts[split]),
            "source_stage_quotas": quotas,
            "source_stage_counts": counts,
            "record_count": len(rows),
        }
    return stage_rows, report, fatal, referenced_ids


def _cross_cell_key(candidate: RenderedCandidate) -> tuple[str, ...]:
    metadata = candidate.metadata
    layer_families = metadata.get("layer_families")
    layer_value = "|".join(sorted(str(value) for value in layer_families)) if isinstance(layer_families, list) and layer_families else "none"
    return (
        candidate.stage or "missing",
        candidate.family or "missing",
        str(metadata.get("book_mode", "missing")),
        str(metadata.get("difficulty_source", "missing")),
        str(metadata.get("context_budget_profile", "missing")),
        str(metadata.get("module_source", "missing")),
        layer_value,
    )


def _coverage_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, (Mapping, list, tuple, set)):
        normalized = sorted(value) if isinstance(value, set) else value
        return _canonical_json(normalized)
    return str(value)


def _unique_coverage(
    candidates: Sequence[RenderedCandidate], plan: Mapping[str, Any]
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    requested = plan["coverage_contract"]["required_unique_object_metrics"]
    values: dict[str, set[str]] = {str(metric): set() for metric in requested}
    declared_values: dict[str, set[str]] = {str(metric): set() for metric in requested}
    derived_values: dict[str, set[str]] = {str(metric): set() for metric in requested}
    for candidate in candidates:
        coverage_objects = candidate.metadata.get("coverage_objects")
        if isinstance(coverage_objects, Mapping):
            for metric, raw_values in coverage_objects.items():
                if metric not in values:
                    continue
                items = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
                tokens = {
                    token for item in items if (token := _coverage_token(item)) is not None
                }
                values[metric].update(tokens)
                declared_values[metric].update(tokens)
        payload_text = _canonical_json(candidate.canonical_object.get("payload", {}))
        if "canonical_genes" in values:
            genes = set(ENSEMBL_RE.findall(payload_text))
            values["canonical_genes"].update(genes)
            derived_values["canonical_genes"].update(genes)
        layer_ids = candidate.metadata.get("layer_ids")
        if "layers" in values and isinstance(layer_ids, list):
            layers = {str(value) for value in layer_ids if value is not None and str(value)}
            values["layers"].update(layers)
            derived_values["layers"].update(layers)
        if "live_tool_schemas" in values and candidate.metadata.get("book_mode") == "tool_call":
            tool_name = candidate.metadata.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                exchange = candidate.metadata.get("tool_exchange")
                action = exchange.get("tool_action") if isinstance(exchange, Mapping) else None
                tool_name = action.get("tool_name") if isinstance(action, Mapping) else None
            if isinstance(tool_name, str) and tool_name:
                values["live_tool_schemas"].add(tool_name)
                derived_values["live_tool_schemas"].add(tool_name)
    counts = {metric: len(items) for metric, items in sorted(values.items())}
    details = {
        metric: {
            "count": len(items),
            "declared_count": len(declared_values[metric]),
            "derived_count": len(derived_values[metric]),
            "values_hash": _stable_hash(sorted(items)),
            "sample": sorted(items)[:20],
        }
        for metric, items in sorted(values.items())
    }
    return counts, details


def _build_coverage_report(
    generated: Sequence[RenderedCandidate],
    filtered_ids: set[int],
    selected: Sequence[RenderedCandidate],
    *,
    plan: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    family_counts: Mapping[str, int],
    family_minimum: int,
    material_cell_minimum: int,
) -> dict[str, Any]:
    cell_fields = plan["coverage_contract"]["required_cross_cell_dimensions"]
    cells: dict[tuple[str, ...], dict[str, int]] = defaultdict(
        lambda: {"requested": 0, "generated": 0, "compacted": 0, "filtered": 0, "selected": 0}
    )
    for candidate in generated:
        cell = cells[_cross_cell_key(candidate)]
        cell["generated"] += 1
        cell["compacted"] += 1
        if id(candidate) in filtered_ids:
            cell["filtered"] += 1
    for candidate in selected:
        cell = cells[_cross_cell_key(candidate)]
        cell["requested"] += 1
        cell["selected"] += 1
    cross_cells = []
    material_cell_count = 0
    underfilled_material_cell_count = 0
    for key, counts in sorted(cells.items()):
        valid_count = counts["compacted"] - counts["filtered"]
        material = valid_count >= material_cell_minimum
        underfilled = material and counts["selected"] < material_cell_minimum
        material_cell_count += int(material)
        underfilled_material_cell_count += int(underfilled)
        cross_cells.append(
            {
                **dict(zip(cell_fields, key, strict=True)),
                **counts,
                "valid": valid_count,
                "minimum_selected": material_cell_minimum,
                "material": material,
                "underfilled": underfilled,
            }
        )
    unique_counts, unique_details = _unique_coverage(selected, plan)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "passed": underfilled_material_cell_count == 0,
        "underfill_policy": "fatal",
        "minimum_selected_per_material_cross_cell": material_cell_minimum,
        "material_cross_cell_count": material_cell_count,
        "underfilled_material_cross_cell_count": underfilled_material_cell_count,
        "selection": selection_report,
        "question_families": {
            family: {
                "selected": int(family_counts.get(family, 0)),
                "minimum": family_minimum,
                "underfilled": int(family_counts.get(family, 0)) < family_minimum,
            }
            for family in sorted(family_counts)
        },
        "record_count_by_stage": dict(sorted(Counter(item.stage for item in selected).items())),
        "record_count_by_book_mode": dict(
            sorted(Counter(str(item.metadata.get("book_mode")) for item in selected).items())
        ),
        "record_count_by_budget_profile": dict(
            sorted(
                Counter(str(item.metadata.get("context_budget_profile")) for item in selected).items()
            )
        ),
        "unique_object_coverage": unique_counts,
        "unique_object_coverage_details": unique_details,
        "cross_cell_fields": list(cell_fields),
        "cross_cells": cross_cells,
        "requested_semantics": (
            "Cross-cell requested counts are the exact plan-driven selections assigned to that cell; "
            "bucket-level preselection quotas are recorded under selection.splits."
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_artifacts(
    out_dir: Path,
    *,
    plan: Mapping[str, Any],
    selected: Sequence[RenderedCandidate],
    canonical_objects: Sequence[Mapping[str, Any]],
    stage_rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    coverage_report: Mapping[str, Any],
    leakage_report: Mapping[str, Any],
    audit_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    overwrite: bool,
) -> None:
    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=str(out_dir.parent))
    )
    try:
        _write_json(staging / "curriculum_plan.json", plan)
        for split in SPLITS:
            rows = [candidate.record for candidate in selected if candidate.split == split]
            rows.sort(key=lambda row: str(row["metadata"]["record_id"]))
            _write_jsonl(staging / f"{split}.jsonl", rows)
        _write_jsonl(staging / "canonical_objects.jsonl", canonical_objects)
        for stage in STAGE_NAMES:
            for split in SPLITS:
                _write_jsonl(
                    staging / "curriculum" / stage / f"{split}.jsonl",
                    stage_rows[stage][split],
                )
        _write_json(staging / "coverage_report.json", coverage_report)
        _write_json(staging / "leakage_report.json", leakage_report)
        _write_json(staging / "audit_report.json", audit_report)
        _write_json(staging / "manifest.json", manifest)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(staging, out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compile_pretrajectory_curriculum_artifacts(
    *,
    candidates: Iterable[Any],
    out_dir: Path | str,
    plan: Path | str | Mapping[str, Any],
    build_profile: str = "patchcheck",
    seed: int = 42,
    render_candidate: RenderCallback | None = None,
    tool_validator: ToolValidator | None = None,
    source_identities: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compile all generated candidates and atomically publish curriculum files.

    Candidate inputs may be existing builder objects with ``record`` and
    ``canonical_object`` attributes, two-tuples, mappings containing those two
    keys, or arbitrary objects handled by ``render_candidate``.  Invalid rows
    are reported and filtered before selection; any resulting quota, family,
    leakage, replay, or blend underfill is fatal and publishes no dataset.
    """

    plan_payload = _load_plan(plan)
    if build_profile not in plan_payload["build_profiles"]:
        raise ValueError(f"Unknown build profile {build_profile!r}")
    profile = plan_payload["build_profiles"][build_profile]
    target_counts = {split: int(profile["split_counts"][split]) for split in SPLITS}
    family_by_name = {
        str(family["name"]): family for family in plan_payload["question_families"]
    }
    stage_by_name = {str(stage["name"]): stage for stage in plan_payload["stages"]}

    rendered: list[RenderedCandidate] = []
    render_errors: list[dict[str, Any]] = []
    for index, raw in enumerate(candidates):
        try:
            rendered.append(_render_candidate(raw, render_candidate))
        except Exception as exc:
            render_errors.append(
                {
                    "code": "candidate_render_failed",
                    "message": f"candidate {index} could not be rendered: {type(exc).__name__}: {exc}",
                    "candidate_index": index,
                }
            )

    deduped, leakage_report, leakage_fatal = _deduplicate_and_check_leakage(rendered)
    group_keys = [
        "oracle_fact_group_id",
        *plan_payload["split_contract"]["co_grouping_keys_when_present"],
    ]
    optional_leaks, optional_fatal = _optional_group_leakage(
        deduped,
        group_keys,
    )
    leakage_report["optional_group_cross_split"] = optional_leaks
    leakage_report["optional_group_cross_split_count"] = sum(
        len(values) for values in optional_leaks.values()
    )
    leakage_report["passed"] = not leakage_fatal and not optional_fatal
    structural_fatal = [*leakage_fatal, *optional_fatal]
    if structural_fatal:
        raise CurriculumArtifactError(structural_fatal)

    valid_pool: list[RenderedCandidate] = []
    filtered_issues: list[dict[str, Any]] = list(render_errors)
    filtered_ids: set[int] = set()
    for candidate in deduped:
        issues = _candidate_audit_issues(
            candidate,
            plan=plan_payload,
            family_by_name=family_by_name,
            stage_by_name=stage_by_name,
            tool_validator=tool_validator,
        )
        if issues:
            filtered_issues.extend(issues)
            filtered_ids.add(id(candidate))
        else:
            valid_pool.append(candidate)

    selected, selection_report, selection_errors = _select_main_records(
        valid_pool,
        target_counts=target_counts,
        bucket_weights=plan_payload["mixture"]["content_buckets"],
        seed=seed,
    )
    family_minimum = int(profile["minimum_selected_per_required_family"])
    material_cell_minimum = int(profile["minimum_selected_per_material_cross_cell"])
    repair_errors: list[dict[str, Any]] = []
    if not selection_errors:
        selected, _repair_report, repair_errors = _repair_material_cross_cells(
            valid_pool,
            selected,
            selection_report,
            minimum_cell_count=material_cell_minimum,
            minimum_family_count=family_minimum,
            family_names=list(family_by_name),
            cross_cell_fields=plan_payload["coverage_contract"][
                "required_cross_cell_dimensions"
            ],
            seed=seed,
        )
    family_counts, family_errors = _family_underfill_errors(
        selected,
        family_names=list(family_by_name),
        minimum=family_minimum,
    )
    stage_rows, stage_report, stage_errors, stage_object_ids = _build_stage_files(
        selected,
        valid_pool,
        plan=plan_payload,
        target_counts=target_counts,
        seed=seed,
    )
    fatal = [*selection_errors, *repair_errors, *family_errors, *stage_errors]
    if fatal:
        raise CurriculumArtifactError(fatal)

    selected_object_ids = {
        str(candidate.metadata["canonical_object_id"]) for candidate in selected
    }
    all_output_object_ids = selected_object_ids | stage_object_ids
    canonical_by_id: dict[str, dict[str, Any]] = {}
    canonical_conflicts: list[dict[str, Any]] = []
    for candidate in valid_pool:
        object_id = str(candidate.canonical_object["object_id"])
        if object_id not in all_output_object_ids:
            continue
        previous = canonical_by_id.get(object_id)
        if previous is not None and _stable_hash(previous) != _stable_hash(candidate.canonical_object):
            canonical_conflicts.append(
                {
                    "code": "conflicting_canonical_object",
                    "message": f"canonical object {object_id!r} has non-identical definitions",
                    "canonical_object_id": object_id,
                }
            )
        else:
            canonical_by_id[object_id] = candidate.canonical_object
    missing_objects = all_output_object_ids - set(canonical_by_id)
    for object_id in sorted(missing_objects):
        canonical_conflicts.append(
            {
                "code": "missing_canonical_object",
                "message": f"output rows reference missing canonical object {object_id!r}",
                "canonical_object_id": object_id,
            }
        )
    if canonical_conflicts:
        raise CurriculumArtifactError(canonical_conflicts)
    canonical_objects = [canonical_by_id[key] for key in sorted(canonical_by_id)]

    split_counts = Counter(candidate.split for candidate in selected)
    bucket_counts = Counter(candidate.bucket for candidate in selected)
    final_errors: list[dict[str, Any]] = []
    for split in SPLITS:
        if split_counts[split] != target_counts[split]:
            final_errors.append(
                {
                    "code": "final_split_count_mismatch",
                    "message": (
                        f"final {split} count is {split_counts[split]}, expected {target_counts[split]}"
                    ),
                }
            )
    if final_errors:
        raise CurriculumArtifactError(final_errors)

    coverage_report = _build_coverage_report(
        deduped,
        filtered_ids,
        selected,
        plan=plan_payload,
        selection_report=selection_report,
        family_counts=family_counts,
        family_minimum=family_minimum,
        material_cell_minimum=material_cell_minimum,
    )
    if not coverage_report["passed"]:
        raise CurriculumArtifactError(
            [
                {
                    "code": "material_cross_cell_underfill",
                    "message": (
                        "Final coverage report contains "
                        f"{coverage_report['underfilled_material_cross_cell_count']} "
                        "underfilled material cross-cells."
                    ),
                }
            ]
        )
    plan_hash = curriculum_plan_hash(plan_payload)
    coverage_report.update({"plan_id": plan_payload["plan_id"], "plan_hash": plan_hash})
    leakage_report.update(
        {
            "plan_id": plan_payload["plan_id"],
            "plan_hash": plan_hash,
            "selected_record_count": len(selected),
        }
    )
    audit_report = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "passed": True,
        "fatal_error_count": 0,
        "warning_count": len(filtered_issues),
        "filtered_candidate_count": len(filtered_ids) + len(render_errors),
        "filtered_issue_count": len(filtered_issues),
        "filtered_issues": filtered_issues[:1000],
        "truncated_filtered_issue_count": max(0, len(filtered_issues) - 1000),
        "valid_deduplicated_candidate_count": len(valid_pool),
        "selected_record_count": len(selected),
        "record_count_by_split": dict(sorted(split_counts.items())),
        "record_count_by_mixture_bucket": dict(sorted(bucket_counts.items())),
        "question_family_minimum": family_minimum,
        "question_family_counts": family_counts,
        "material_cross_cell_minimum": material_cell_minimum,
        "material_cross_cell_count": coverage_report["material_cross_cell_count"],
        "underfilled_material_cross_cell_count": coverage_report[
            "underfilled_material_cross_cell_count"
        ],
        "post_selection_swap_count": selection_report.get(
            "post_selection_repair", {}
        ).get("swap_count", 0),
        "budget_violation_count_in_selected": 0,
        "raw_path_violation_count_in_selected": 0,
        "metadata_violation_count_in_selected": 0,
        "tool_schema_violation_count_in_selected": 0,
        "leakage_passed": True,
        "plan_id": plan_payload["plan_id"],
        "plan_hash": plan_hash,
    }
    output_paths = {
        "curriculum_plan": "curriculum_plan.json",
        "train": "train.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
        "canonical_objects": "canonical_objects.jsonl",
        "coverage_report": "coverage_report.json",
        "leakage_report": "leakage_report.json",
        "audit_report": "audit_report.json",
        "curriculum": {
            stage: {split: f"curriculum/{stage}/{split}.jsonl" for split in SPLITS}
            for stage in STAGE_NAMES
        },
    }
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "plan_id": plan_payload["plan_id"],
        "plan_hash": plan_hash,
        "dataset_schema_version": plan_payload["dataset_schema_version"],
        "build_profile": build_profile,
        "seed": seed,
        "raw_candidate_count": len(rendered) + len(render_errors),
        "rendered_candidate_count": len(rendered),
        "deduplicated_candidate_count": len(deduped),
        "filtered_candidate_count": len(filtered_ids) + len(render_errors),
        "valid_candidate_count": len(valid_pool),
        "selected_record_count": len(selected),
        "record_count_by_split": dict(sorted(split_counts.items())),
        "record_count_by_mixture_bucket": dict(sorted(bucket_counts.items())),
        "record_count_by_question_family": family_counts,
        "canonical_object_count": len(canonical_objects),
        "selection": selection_report,
        "curriculum_stage_files": stage_report,
        "content_hash": _stable_hash(
            [candidate.record for candidate in sorted(selected, key=lambda item: item.record_id)]
        ),
        "outputs": output_paths,
        "contracts": {
            "exact_split_counts": True,
            "largest_remainder_mixture_quotas": True,
            "family_round_robin": True,
            "fatal_underfill": True,
            "oracle_fact_group_split_isolation": True,
            "raw_path_free_selected_rows": True,
            "tool_schema_validity": 1.0,
        },
        "source_identities": copy.deepcopy(dict(source_identities or {})),
    }

    _write_artifacts(
        Path(out_dir),
        plan=plan_payload,
        selected=selected,
        canonical_objects=canonical_objects,
        stage_rows=stage_rows,
        coverage_report=coverage_report,
        leakage_report=leakage_report,
        audit_report=audit_report,
        manifest=manifest,
        overwrite=overwrite,
    )
    return {
        "manifest": manifest,
        "coverage_report": coverage_report,
        "leakage_report": leakage_report,
        "audit_report": audit_report,
        "records": [candidate.record for candidate in selected],
        "canonical_objects": canonical_objects,
    }


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CurriculumArtifactError",
    "compile_pretrajectory_curriculum_artifacts",
    "largest_remainder_quotas",
]
