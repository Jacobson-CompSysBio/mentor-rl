"""Build the canonical S1.9 module-ID parsing corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from . import s1_6_corpus as s1_6
from .contract import (
    CONTRACT_PATH,
    REPOSITORY_ROOT,
    load_contract,
    system_prompt_sha256,
    tool_definitions,
)
from .s1_overlay import apply_s1_overlay, overlay_pin
from .s1_prompt_banks import prompt_bank_manifest, wrapper_prompts
from .source_index import check_source_index
from .sources import file_sha256


CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-9-corpus-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-9-manifest-v1"
CONTRACT_S1_9_SHA256 = (
    "98badcd4b359a5c94025732cdc20f2773812cdc716a66d7078caf4fa82f872ba"
)
SPLITS = ("train", "validation", "test")

CorpusBuildError = s1_6.CorpusBuildError
ModuleFact = s1_6.ModuleFact
split_row_counts = s1_6.split_row_counts

FORM_BY_FAMILY = {"rwr_loe": 1, "mentor_ev": 2}
FAMILY_BY_FORM = {value: key for key, value in FORM_BY_FAMILY.items()}
TRACE_BY_BUCKET = {"rwr_loe_parse": 1, "mentor_ev_parse": 2}
BUCKET_BY_TRACE = {value: key for key, value in TRACE_BY_BUCKET.items()}
MODULE_FAMILIES = tuple(FORM_BY_FAMILY)

MODULE_GRAMMARS = {
    "rwr_loe": {
        "pattern": (
            "^rwr_loe:full_brain_multiplex_v1:"
            "seed_(ENSG[0-9]{11}):geometric_elbow_v1$"
        ),
        "field_count": 4,
        "multiplex_id": "full_brain_multiplex_v1",
        "threshold_recipe": "geometric_elbow_v1",
    },
    "mentor_ev": {
        "pattern": "^mentor_ev:full_brain_all_genes_v1:clade_[0-9]{6}$",
        "field_count": 3,
        "dendrogram_id": "full_brain_all_genes_v1",
        "clade_digits": 6,
    },
}
RWR_LOE_PATTERN = re.compile(str(MODULE_GRAMMARS["rwr_loe"]["pattern"]))
MENTOR_EV_PATTERN = re.compile(str(MODULE_GRAMMARS["mentor_ev"]["pattern"]))

MASK_TOKEN = "<SOURCE_VALUE>"
MASKED_WORDING_TOLERANCE = 0.05

MODULE_PROMPTS = {
    "train": {
        "direct": (
            "Answer this module question.\n{request}",
            "Review this module identifier.\n{request}",
            "Give the exact module parse.\n{request}",
            "Resolve this module request.\n{request}",
            "Read the supplied module ID.\n{request}",
            "Return the module fields.\n{request}",
            "Check this module identifier.\n{request}",
            "Interpret the module ID.\n{request}",
        ),
        "step": (
            "Answer this before the next step.\n{request}",
            "Check this before the workflow continues.\n{request}",
            "The next step needs this module parse.\n{request}",
            "Resolve this before the next action.\n{request}",
            "Prepare these module fields for later work.\n{request}",
            "Read this ID before analysis continues.\n{request}",
            "The workflow needs this module result.\n{request}",
            "Verify this module before work advances.\n{request}",
        ),
    },
    "validation": {
        "direct": (
            "Supply the exact module parse.\n{request}",
            "Read and answer this module question.\n{request}",
            "Return the requested module result.\n{request}",
            "Evaluate the supplied module ID.\n{request}",
            "State the encoded module fields.\n{request}",
            "Assess this module identifier.\n{request}",
            "Give the result for this module.\n{request}",
            "Resolve the supplied module question.\n{request}",
        ),
        "step": (
            "Answer this before analysis continues.\n{request}",
            "Prepare this module result for later work.\n{request}",
            "The workflow needs this module parse.\n{request}",
            "Check this identifier before work advances.\n{request}",
            "Resolve the module before the next task.\n{request}",
            "Supply these fields for the next action.\n{request}",
            "Verify this module before analysis advances.\n{request}",
            "Prepare the exact parse for the next step.\n{request}",
        ),
    },
    "test": {
        "direct": (
            "Find the exact module parse.\n{request}",
            "Give the result for this module question.\n{request}",
            "Read this module request and answer it.\n{request}",
            "State the requested module fields.\n{request}",
            "Use the module grammar.\n{request}",
            "Decide this module question.\n{request}",
            "Return the applicable module parse.\n{request}",
            "Assess the stated module ID.\n{request}",
        ),
        "step": (
            "Resolve this before analysis advances.\n{request}",
            "This module result is needed for the next action.\n{request}",
            "Prepare this module parse for later analysis.\n{request}",
            "The workflow first needs these module fields.\n{request}",
            "Check the module before the next step.\n{request}",
            "Supply this parse before work continues.\n{request}",
            "Verify the module ID for later use.\n{request}",
            "Resolve the identifier before the workflow advances.\n{request}",
        ),
    },
}

PARSE_REQUESTS = {
    "train": (
        "Parse module ID {module_id}.",
        "Return every field encoded in {module_id}.",
        "Decode the module identifier {module_id}.",
        "Read the fields from module ID {module_id}.",
        "Extract the exact parse of {module_id}.",
        "Give the parsed module record for {module_id}.",
        "Interpret the identifier {module_id}.",
        "Identify the parts of module ID {module_id}.",
    ),
    "validation": (
        "Parse the supplied module ID {module_id}.",
        "List the fields encoded by {module_id}.",
        "Decode this module identifier: {module_id}.",
        "Read the exact module record from {module_id}.",
        "Extract all fields in {module_id}.",
        "Give the module parse for {module_id}.",
        "Interpret this identifier: {module_id}.",
        "State the parts of the module ID {module_id}.",
    ),
    "test": (
        "Parse this module identifier: {module_id}.",
        "Return the encoded fields for {module_id}.",
        "Decode the supplied module ID {module_id}.",
        "Read every module field in {module_id}.",
        "Extract the module record from {module_id}.",
        "Give an exact parse of {module_id}.",
        "Interpret the module ID {module_id}.",
        "Name the parts encoded in {module_id}.",
    ),
}

S1_9_SETTING_KEYS = (
    "dataset_id",
    "question_family",
    "book_mode",
    "output_directory",
    "assignment_seed",
    "train_rows",
    "allocations",
    "evaluation_allocations",
    "semantic_buckets",
    "source_trace_by_allocation",
    "grammars",
    "result_shapes",
    "source_pools",
    "split_policy",
    "nonpositive_policy",
    "status_counts",
    "framing_counts",
)
TRAIN_SETTING_KEYS = (
    "train_module_reasoning", "train_balanced_requests", "train_final_reasoning",
    "train_decorrelate",
)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Store one exact S1.9 module parse scenario."""

    split: str
    form: int
    module_id: str
    module_family: str
    split_unit: str
    framing: str
    template_index: int
    request_index: int
    occurrence: int
    scenario_id: str
    train_reasoning: bool = False

    @property
    def source_trace(self) -> str:
        """Return the source trace label."""

        return f"S1.9.{self.form}"

    @property
    def semantic_bucket(self) -> str:
        """Return the semantic bucket."""

        return self.module_family


FactPools = dict[str, dict[str, list[ModuleFact]]]


def _canonical_json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys
    )


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def parse_module_id(module_id: str) -> dict[str, Any]:
    """Parse one S1.9 ID without an existence claim."""

    if not isinstance(module_id, str) or not module_id:
        raise CorpusBuildError("the module ID must be a nonempty string")
    rwr_match = RWR_LOE_PATTERN.fullmatch(module_id)
    if rwr_match:
        return {
            "module_id": module_id,
            "module_family": "rwr_loe",
            "multiplex_id": "full_brain_multiplex_v1",
            "seed_gene_id": rwr_match.group(1),
            "threshold_recipe": "geometric_elbow_v1",
            "existence_asserted": False,
        }
    mentor_match = MENTOR_EV_PATTERN.fullmatch(module_id)
    if mentor_match:
        return {
            "module_id": module_id,
            "module_family": "mentor_ev",
            "dendrogram_id": "full_brain_all_genes_v1",
            "clade_id": module_id.rsplit(":", 1)[1],
            "existence_asserted": False,
        }
    raise CorpusBuildError(
        f"the module ID does not match an S1.9 grammar: {module_id!r}"
    )


def _frozen_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the settings with the v1 build location.

    Ignore an overlay's output location and training-only switches;
    the frozen hash still protects every grammar and source setting.
    """

    return {
        **{key: value for key, value in settings.items() if key not in TRAIN_SETTING_KEYS},
        "output_directory": "data/world_model_v3/corpora/s1/s1_9",
    }


def _assert_exact_settings(settings: Mapping[str, Any]) -> None:
    """Fail if a frozen S1.9 decision changes."""

    source_keys = tuple(key for key in settings if key not in TRAIN_SETTING_KEYS)
    if source_keys != S1_9_SETTING_KEYS:
        raise CorpusBuildError("the S1.9 setting fields changed")
    if _stable_sha256(_frozen_settings(settings)) != CONTRACT_S1_9_SHA256:
        raise CorpusBuildError("the frozen S1.9 contract section changed")
    if _stable_sha256(settings["grammars"]) != _stable_sha256(MODULE_GRAMMARS):
        raise CorpusBuildError("the S1.9 grammars changed")
    if tuple(settings["allocations"]) != tuple(TRACE_BY_BUCKET):
        raise CorpusBuildError("the S1.9 training allocation fields changed")
    if tuple(settings["evaluation_allocations"]) != tuple(TRACE_BY_BUCKET):
        raise CorpusBuildError("the S1.9 evaluation allocation fields changed")
    if settings["nonpositive_policy"] != {
        "included": False,
        "invalid": "excluded",
        "ambiguous": "excluded",
        "unrecognized": "excluded",
        "syntactically_valid_unobserved": "excluded",
    }:
        raise CorpusBuildError("the S1.9 nonpositive policy changed")


def trace_counts(total: int, settings: Mapping[str, Any]) -> dict[int, int]:
    """Return the exact S1.9 trace allocation for one split."""

    _assert_exact_settings(settings)
    allocations = settings["allocations"]
    evaluation = settings["evaluation_allocations"]
    train_total = int(settings["train_rows"])
    evaluation_total = sum(int(value) for value in evaluation.values())
    if total == train_total:
        selected = allocations
    elif total == evaluation_total:
        selected = evaluation
    else:
        selected = s1_6.largest_remainder_counts(
            total,
            {
                name: Fraction(int(value), train_total)
                for name, value in allocations.items()
            },
        )
    return {
        TRACE_BY_BUCKET[name]: int(selected[name])
        for name in TRACE_BY_BUCKET
    }


def build_fact_pools(
    payload: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    seed: int,
    contract: Mapping[str, Any],
) -> FactPools:
    """Reuse the S1.6 split-safe module facts."""

    del contract
    _assert_exact_settings(settings)
    inherited = s1_6._module_facts(payload, seed=seed)
    pools: FactPools = {
        split: {family: [] for family in MODULE_FAMILIES} for split in SPLITS
    }
    for split in SPLITS:
        for fact in inherited[split]:
            if fact.namespace not in MODULE_FAMILIES:
                raise CorpusBuildError(
                    f"unknown source module family {fact.namespace!r}"
                )
            parsed = parse_module_id(fact.module_id)
            if parsed["module_family"] != fact.namespace:
                raise CorpusBuildError("a source module family does not match its ID")
            pools[split][fact.namespace].append(fact)
        for family in MODULE_FAMILIES:
            pools[split][family].sort(key=lambda item: item.fact_id)
            expected = int(
                settings["source_pools"][family]["split_counts"][split]
            )
            if len(pools[split][family]) != expected:
                raise CorpusBuildError(
                    f"the {split} {family} source pool count changed"
                )
    for family in MODULE_FAMILIES:
        total = sum(len(pools[split][family]) for split in SPLITS)
        if total != int(settings["source_pools"][family]["record_count"]):
            raise CorpusBuildError(f"the {family} source record count changed")
    split_ids = {
        split: {
            fact.module_id
            for family in MODULE_FAMILIES
            for fact in pools[split][family]
        }
        for split in SPLITS
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if split_ids[left] & split_ids[right]:
                raise CorpusBuildError(
                    f"source module IDs overlap in {left} and {right}"
                )
    return pools


def _mode(split: str) -> str:
    return "train" if split == "train" else "evaluation"


def _balanced_labels(counts: Mapping[str, int]) -> list[str]:
    remaining = {key: int(value) for key, value in counts.items()}
    labels: list[str] = []
    while any(remaining.values()):
        for key in counts:
            if remaining[key] > 0:
                labels.append(key)
                remaining[key] -= 1
    return labels


def select_scenarios(
    pools: FactPools,
    counts_by_split: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
    framing_shares: Mapping[str, Any],
    settings: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = MODULE_PROMPTS,
) -> dict[str, list[Scenario]]:
    """Select exact source modules without replacement."""

    del framing_shares
    _assert_exact_settings(settings)
    scenarios_by_split: dict[str, list[Scenario]] = {}
    for split in SPLITS:
        mode = _mode(split)
        frame_config = settings["framing_counts"][f"{mode}_form_counts"]
        scenarios: list[Scenario] = []
        for form, family in FAMILY_BY_FORM.items():
            count = int(counts_by_split[split][form])
            frame_counts = {
                key: int(value)
                for key, value in frame_config[BUCKET_BY_TRACE[form]].items()
            }
            if sum(frame_counts.values()) != count:
                raise CorpusBuildError(
                    f"the {split} form {form} frame count changed"
                )
            available = sorted(
                pools[split][family],
                key=lambda fact: _stable_sha256(
                    {
                        "seed": seed,
                        "split": split,
                        "family": family,
                        "module_id": fact.module_id,
                    }
                ),
            )
            if len(available) < count:
                raise CorpusBuildError(
                    f"the {split} {family} source pool is too small"
                )
            selected = available[:count]
            frames = _balanced_labels(frame_counts)
            frame_positions: Counter[str] = Counter()
            for occurrence, (fact, framing) in enumerate(
                zip(selected, frames, strict=True)
            ):
                template_index = frame_positions[framing] % len(
                    prompts[split][framing]
                )
                frame_positions[framing] += 1
                # Train has alternating frames. Repeat each request for two
                # positions so its verb cannot act as a continuation cue.
                request_position = (
                    occurrence // 2
                    if split == "train" and settings.get("train_balanced_requests")
                    else occurrence
                )
                request_index = request_position % len(PARSE_REQUESTS[split])
                if split == "train" and settings.get("train_decorrelate"):
                    # Request wording is independent of the wrapper-bank size.
                    # Shift the wrapper cycle each request round so each wrapper
                    # sees every request while both marginal counts stay balanced.
                    position = frame_positions[framing] - 1
                    request_index = position % len(PARSE_REQUESTS[split])
                    template_index = (
                        position // len(PARSE_REQUESTS[split]) + request_index
                    ) % len(prompts[split][framing])
                identity = {
                    "split": split,
                    "form": form,
                    "module_id": fact.module_id,
                    "module_family": family,
                    "split_unit": fact.split_unit,
                    "framing": framing,
                    "template_index": template_index,
                    "request_index": request_index,
                    "occurrence": occurrence,
                }
                scenarios.append(
                    Scenario(
                        **identity,
                        scenario_id=f"s1_9_{split}_"
                        + _stable_sha256(identity)[:24],
                        train_reasoning=split == "train" and bool(
                            settings.get("train_module_reasoning")
                        ),
                    )
                )
        scenarios.sort(
            key=lambda scenario: _stable_sha256(
                {"seed": seed, "split": split, "id": scenario.scenario_id}
            )
        )
        scenarios_by_split[split] = scenarios
    return scenarios_by_split


def module_decorrelation_audit(scenarios: Sequence[Scenario]) -> dict[str, Any]:
    """Require every training wrapper to cover every module request phrasing."""

    cells: dict[tuple[int, str, int], Counter[int]] = defaultdict(Counter)
    for item in scenarios:
        if item.split == "train":
            cells[(item.form, item.framing, item.template_index)][item.request_index] += 1
    expected = set(range(len(PARSE_REQUESTS["train"])))
    if not cells or any(set(counts) != expected for counts in cells.values()):
        raise CorpusBuildError("module wrappers lack independent request coverage")
    return {
        "pass": True,
        "wrapper_cells": len(cells),
        "realized_wrapper_request_cells": sum(len(counts) for counts in cells.values()),
        "request_variants_per_wrapper": len(expected),
        "counts": {f"{form}:{frame}:{index}": dict(counts) for (form, frame, index), counts in sorted(cells.items())},
    }


def _request(scenario: Scenario) -> str:
    return PARSE_REQUESTS[scenario.split][scenario.request_index].format(
        module_id=scenario.module_id
    )


def _prompt(
    scenario: Scenario,
    *,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = MODULE_PROMPTS,
) -> str:
    template = prompts[scenario.split][scenario.framing][
        scenario.template_index
    ]
    return template.format(request=_request(scenario))


def _legacy_thinking(scenario: Scenario) -> str:
    if scenario.module_family == "rwr_loe":
        return (
            "The rwr_loe grammar encodes the multiplex, seed gene, and "
            "threshold recipe. Parsing does not assert existence."
        )
    return (
        "The mentor_ev grammar encodes a frozen all-gene dendrogram and a "
        "six-digit clade. Parsing does not assert existence."
    )


def _thinking(
    scenario: Scenario, *,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> str:
    """Bind encoded components to the closed result schema without claiming existence."""

    if scenario.split == "train" and final_reasoning and scenario.form in final_reasoning:
        result = parse_module_id(scenario.module_id)
        if scenario.form == 1:
            result["seed_component"] = scenario.module_id.split(":")[2]
        return final_reasoning[scenario.form][scenario.framing].format(**result)
    if not scenario.train_reasoning:
        return _legacy_thinking(scenario)
    parts = scenario.module_id.split(":")
    result = parse_module_id(scenario.module_id)
    copy = f'Copy the full input "{scenario.module_id}" into module_id.'
    if scenario.module_family == "rwr_loe":
        mapping = (
            "The four colon-separated components match the rwr_loe grammar, so status is resolved. "
            "The first component sets module_family to rwr_loe. "
            "The second component sets multiplex_id to full_brain_multiplex_v1. "
            f'Remove only seed_ from "{parts[2]}" to set seed_gene_id to {result["seed_gene_id"]}. '
            "The fourth component sets threshold_recipe to geometric_elbow_v1. "
            "Emit exactly module_id, module_family, multiplex_id, seed_gene_id, threshold_recipe, and existence_asserted, in that order."
        )
    else:
        mapping = (
            "The three colon-separated components match the mentor_ev grammar, so status is resolved. "
            "The first component sets module_family to mentor_ev. "
            "The second component full_brain_all_genes_v1 sets dendrogram_id. "
            f'The third component "{parts[2]}" sets clade_id; preserve clade_ and all six digits, including leading zeros. '
            "Emit exactly module_id, module_family, dendrogram_id, clade_id, and existence_asserted, in that order."
        )
    limit = (
        "Only the encoded components belong in this result. "
        "Parsing the encoded fields does not establish existence, so existence_asserted is false."
    )
    scope = (
        "The enclosing request is complete after this answer, so continuation_decision is stop."
        if scenario.framing == "direct" else
        "Subsequent work uses this parse, so continuation_decision is continue."
    )
    return f"{copy} {mapping} {limit} {scope}"


def _summary(scenario: Scenario, result: Mapping[str, Any]) -> str:
    if scenario.module_family == "rwr_loe":
        return (
            "The RWR module is rooted at "
            f"<GENE:{result['seed_gene_id']}> in "
            f"<MULTIPLEX:{result['multiplex_id']}>; parsing alone does not "
            "assert existence."
        )
    return (
        f"The identifier parses as MENTOR-EV {result['clade_id']} in the "
        f"{result['dendrogram_id']} dendrogram; parsing alone does not "
        "assert existence."
    )


def _answer(scenario: Scenario) -> dict[str, Any]:
    result = parse_module_id(scenario.module_id)
    if result["module_family"] != scenario.module_family:
        raise CorpusBuildError("the scenario family does not match its module ID")
    return {
        "status": "resolved",
        "provenance": None,
        "result": result,
        "continuation_decision": (
            "stop" if scenario.framing == "direct" else "continue"
        ),
        "blocked_reason": None,
        "summary": _summary(scenario, result),
    }


def _row_id(scenario: Scenario, schema_version: str) -> str:
    return "wm3_s1_9_" + _stable_sha256(
        {"schema_version": schema_version, "scenario": scenario.scenario_id}
    )[:24]


def build_row(
    scenario: Scenario,
    *,
    contract: Mapping[str, Any],
    tools: list[dict[str, Any]] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = MODULE_PROMPTS,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Render one closed-book S1.9 row without a tool call."""

    schema_version = str(contract["runtime"]["row_schema_version"])
    row = {
        "row_id": _row_id(scenario, schema_version),
        "messages": [
            {"role": "system", "content": contract["runtime"]["system_prompt"]},
            {"role": "user", "content": _prompt(scenario, prompts=prompts)},
            {
                "role": "assistant",
                "thinking": _thinking(scenario, final_reasoning=final_reasoning),
                "content": _answer(scenario),
            },
        ],
        "tools": tool_definitions(contract) if tools is None else tools,
        "metadata": {
            "book_mode": "closed_book",
            "step": "S1",
            "question_family": "module_id_parsing",
            "form": scenario.form,
            "source_trace": scenario.source_trace,
            "split_unit": scenario.split_unit,
            "blocked_reason_code": None,
            "schema_version": schema_version,
            "system_prompt_sha256": system_prompt_sha256(contract),
            "tool_contract_version": contract["runtime"][
                "tool_contract_version"
            ],
            "ensembl_release": contract["coordinate_system"][
                "ensembl_release"
            ],
            "identifier_registry_id": contract["coordinate_system"][
                "identifier_registry_id"
            ],
        },
    }
    return row, []


THINKING_FILTER_RULES = (
    "one thinking step per row",
    "thinking uses only the module grammar",
    "parsing never asserts module existence",
    "no tool call or tool-result claim",
    "status and decision agree with the answer",
)


def _final_answer(row: Mapping[str, Any]) -> Mapping[str, Any]:
    assistants = [
        message
        for message in row["messages"]
        if message.get("role") == "assistant" and "content" in message
    ]
    if len(assistants) != 1 or not isinstance(assistants[0]["content"], Mapping):
        raise CorpusBuildError(f"row {row['row_id']} lacks one final answer")
    return assistants[0]["content"]


def _tool_calls(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        message["tool_call"]
        for message in row["messages"]
        if message.get("role") == "assistant" and "tool_call" in message
    ]


def _current_user_message(row: Mapping[str, Any]) -> str:
    users = [message for message in row["messages"] if message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise CorpusBuildError(f"row {row['row_id']} lacks one user message")
    return str(users[0]["content"])


def filter_row_thinking(
    row: Mapping[str, Any], scenario: Scenario, *,
    final_reasoning: Mapping[int, Mapping[str, str]] | None = None,
) -> int:
    """Reject a thought that violates an S1.9 runtime rule."""

    if _tool_calls(row) or any(
        message.get("role") == "tool" for message in row["messages"]
    ):
        raise CorpusBuildError(f"row {row['row_id']} contains a tool action")
    assistants = [
        message for message in row["messages"] if message.get("role") == "assistant"
    ]
    if len(assistants) != 1 or assistants[0].get("thinking") != _thinking(
        scenario, final_reasoning=final_reasoning,
    ):
        raise CorpusBuildError(f"row {row['row_id']} changes its exact thinking")
    thought = str(assistants[0]["thinking"])
    lowered = thought.casefold()
    if scenario.split == "train" and final_reasoning and scenario.form in final_reasoning:
        # Only the reviewed null-provenance explanation is allowed, after the
        # complete source-bound thought has passed the exact-template check.
        lowered = lowered.replace(
            "the answer uses null provenance because the fields come from the supplied identifier.",
            "", 1,
        )
    if any(value in lowered for value in ("provenance", "tool call", "tool result")):
        raise CorpusBuildError(f"row {row['row_id']} discusses runtime fields")
    if re.search(
        r"\b(?:module|clade|identifier|id) exists\b|\bexistence_asserted\s*(?:is|:|=)\s*true\b",
        lowered,
    ):
        raise CorpusBuildError(f"row {row['row_id']} asserts module existence")
    user = _current_user_message(row)
    for identifier in re.findall(
        r"(?:rwr_loe|mentor_ev):[A-Za-z0-9_:]+|\bENSG[0-9]{11}\b|\bclade_[0-9]+\b",
        thought,
    ):
        if identifier not in user:
            raise CorpusBuildError(f"row {row['row_id']} invents the identifier {identifier}")
    if _final_answer(row) != _answer(scenario):
        raise CorpusBuildError(f"row {row['row_id']} changes its exact answer")
    return 1


def _prompt_and_outcome_signatures(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    messages = row["messages"]
    last_user = max(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )
    prompt = messages[: last_user + 1]
    outcomes = [
        {"tool_name": message["name"], "result_status": message["content"]["status"]}
        for message in messages[last_user + 1 :]
        if message.get("role") == "tool"
    ]
    prompt_key = _canonical_json(prompt, sort_keys=False)
    outcome_key = _canonical_json(outcomes, sort_keys=False)
    combined = _canonical_json(
        {"prompt": prompt, "tool_outcomes": outcomes}, sort_keys=False
    )
    return prompt_key, outcome_key, combined


def duplicate_prompt_outcome_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Reject each repeated closed-book prompt and outcome."""

    global_prompts: set[str] = set()
    split_results = {}
    for split in SPLITS:
        prompts: set[str] = set()
        combined: set[str] = set()
        for row, _ in zip(
            rows_by_split[split], scenarios_by_split[split], strict=True
        ):
            prompt, _, signature = _prompt_and_outcome_signatures(row)
            if prompt in prompts or signature in combined:
                raise CorpusBuildError(
                    f"{split} repeats a prompt and closed-book outcome"
                )
            prompts.add(prompt)
            combined.add(signature)
        if global_prompts & prompts:
            raise CorpusBuildError(f"{split} repeats a prompt from another split")
        global_prompts.update(prompts)
        split_results[split] = {
            "row_count": len(rows_by_split[split]),
            "normalized_prompt_count": len(prompts),
            "normalized_prompt_tool_outcome_count": len(combined),
            "duplicate_prompt_tool_outcome_count": 0,
            "cross_split_duplicate_prompt_count": 0,
            "intended_success_failure_pair_count": 0,
            "pass": True,
        }
    return {
        "normalization": "retain the exact visible prompt and every tool outcome",
        "splits": split_results,
        "pass": True,
    }


def masked_visible_prompt(row: Mapping[str, Any], scenario: Scenario) -> str:
    """Mask the source module ID in one visible prompt."""

    return re.sub(
        re.escape(scenario.module_id),
        MASK_TOKEN,
        _current_user_message(row),
        flags=re.IGNORECASE,
    )


def _wording_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", value.casefold()))


def _wording_classifier(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    masker: Callable[[Mapping[str, Any], Scenario], str],
    scope: str,
    gated: bool,
    tolerance: float,
) -> dict[str, Any]:
    """Train and test one class-balanced word model."""

    labels = tuple(FAMILY_BY_FORM)
    token_counts: dict[int, Counter[str]] = defaultdict(Counter)
    token_totals: Counter[int] = Counter()
    vocabulary: set[str] = set()
    for row, scenario in zip(
        rows_by_split["train"], scenarios_by_split["train"], strict=True
    ):
        tokens = _wording_tokens(masker(row, scenario))
        token_counts[scenario.form].update(tokens)
        token_totals[scenario.form] += len(tokens)
        vocabulary.update(tokens)
    if not vocabulary:
        raise CorpusBuildError("the S1.9 wording classifier has no vocabulary")

    def predict(tokens: Sequence[str]) -> int:
        known = Counter(token for token in tokens if token in vocabulary)
        scores = []
        for label in labels:
            denominator = token_totals[label] + len(vocabulary)
            score = sum(
                count * math.log((token_counts[label][token] + 1) / denominator)
                for token, count in known.items()
            )
            scores.append((score, -label, label))
        return max(scores)[2]

    evaluations = {}
    passed_all = True
    for split in SPLITS:
        pairs = list(
            zip(rows_by_split[split], scenarios_by_split[split], strict=True)
        )
        distribution = Counter(scenario.form for _, scenario in pairs)
        correct = sum(
            predict(_wording_tokens(masker(row, scenario))) == scenario.form
            for row, scenario in pairs
        )
        accuracy = correct / len(pairs)
        baseline = max(distribution.values()) / len(pairs)
        apply_gate = gated and split != "train"
        threshold = baseline + tolerance if apply_gate else None
        passed = accuracy <= threshold if apply_gate else None
        if passed is False:
            passed_all = False
        evaluations[split] = {
            "row_count": len(pairs),
            "correct": correct,
            "accuracy": accuracy,
            "largest_form_baseline": baseline,
            "threshold": threshold,
            "gate_applied": apply_gate,
            "pass": passed,
        }
    result = {
        "algorithm": "class_balanced_multinomial_naive_bayes_v1",
        "scope": scope,
        "diagnostic_only": not gated,
        "mask_token": MASK_TOKEN,
        "absolute_tolerance": tolerance,
        "training_vocabulary_size": len(vocabulary),
        "splits": evaluations,
        "pass": passed_all if gated else None,
    }
    if gated and not passed_all:
        raise CorpusBuildError("masked user wording predicts the S1.9 form")
    return result


def masked_wording_classifier_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    *,
    tolerance: float = MASKED_WORDING_TOLERANCE,
) -> dict[str, Any]:
    """Gate form leakage after the source module ID is masked."""

    return _wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=masked_visible_prompt,
        scope="module_id_masked_visible_prompt",
        gated=True,
        tolerance=tolerance,
    )


def full_visible_prompt_classifier_diagnostic(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
) -> dict[str, Any]:
    """Measure full visible prompt cues without a pass gate."""

    return _wording_classifier(
        rows_by_split,
        scenarios_by_split,
        masker=lambda row, _: _current_user_message(row),
        scope="full_visible_module_prompt",
        gated=False,
        tolerance=MASKED_WORDING_TOLERANCE,
    )


def _sentence_word_counts(text: str) -> list[int]:
    normalized = re.sub(r"<[^>]+>", " CODE ", text)
    normalized = re.sub(r"\S*:\S+", " CODE ", normalized)
    normalized = re.sub(r"\b[A-Za-z0-9]+_[A-Za-z0-9_-]+\b", " CODE ", normalized)
    return [
        len(re.findall(r"[A-Za-z]+|CODE", sentence))
        for sentence in re.split(r"[.!?]+", normalized)
        if sentence.strip()
    ]


def _prose_length_audit(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Check generated prose against the repository sentence limits."""

    splits = {}
    for split in SPLITS:
        maximum_instruction = 0
        maximum_description = 0
        violations = 0
        for row in rows_by_split[split]:
            instructions = _sentence_word_counts(_current_user_message(row))
            maximum_instruction = max(
                maximum_instruction, max(instructions, default=0)
            )
            if any(count > 20 for count in instructions):
                violations += 1
            assistant = next(
                message
                for message in row["messages"]
                if message.get("role") == "assistant" and "content" in message
            )
            answer = assistant["content"]
            descriptions = [str(assistant["thinking"]), str(answer["summary"])]
            counts = [
                count
                for description in descriptions
                for count in _sentence_word_counts(description)
            ]
            maximum_description = max(
                maximum_description, max(counts, default=0)
            )
            # Reasoning length is reported without a hard cap. Summary prose
            # retains its existing limit, as do the user instructions above.
            if any(count > 25 for count in _sentence_word_counts(str(answer["summary"]))):
                violations += 1
        if violations:
            raise CorpusBuildError(f"{split} violates the prose length limits")
        splits[split] = {
            "row_count": len(rows_by_split[split]),
            "maximum_instruction_words": maximum_instruction,
            "maximum_description_words": maximum_description,
            "violation_count": 0,
            "pass": True,
        }
    return {"splits": splits, "pass": True}


def source_pool_coverage(pools: FactPools) -> dict[str, Any]:
    """Return exact source-pool counts and identity hashes."""

    return {
        "counts": {
            split: {
                family: len(pools[split][family]) for family in MODULE_FAMILIES
            }
            for split in SPLITS
        },
        "module_id_sha256": {
            split: {
                family: _stable_sha256(
                    sorted(fact.module_id for fact in pools[split][family])
                )
                for family in MODULE_FAMILIES
            }
            for split in SPLITS
        },
    }


def audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    scenarios_by_split: Mapping[str, Sequence[Scenario]],
    expected_counts: Mapping[str, Mapping[int, int]],
    *,
    source_pools: FactPools,
    framing_shares: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
    prompts: Mapping[str, Mapping[str, Sequence[str]]] = MODULE_PROMPTS,
) -> dict[str, Any]:
    """Audit rendered rows against all S1.9 invariants."""

    del framing_shares
    if contract is None:
        contract = load_contract()
    if settings is None:
        settings = contract["curriculum"]["families"]["S1.9"]
    _assert_exact_settings(settings)
    tools = tool_definitions(contract)
    source_ids = {
        split: {
            family: {fact.module_id: fact for fact in source_pools[split][family]}
            for family in MODULE_FAMILIES
        }
        for split in SPLITS
    }
    row_ids: set[str] = set()
    selected_ids: dict[str, set[str]] = {}
    selected_units: dict[str, set[str]] = {}
    status_counts: dict[str, dict[str, int]] = {}
    tool_call_counts: dict[str, dict[str, int]] = {}
    framing_counts: dict[str, dict[str, int]] = {}
    semantic_bucket_counts: dict[str, dict[str, int]] = {}
    module_family_counts: dict[str, dict[str, int]] = {}
    continuation_counts: dict[str, dict[str, int]] = {}
    prompt_template_counts: dict[str, Any] = {}
    request_template_counts: dict[str, Any] = {}
    selected_source_counts: dict[str, dict[str, int]] = {}
    selected_source_sha256: dict[str, dict[str, str]] = {}
    per_form: dict[str, Any] = {}
    thinking: dict[str, Any] = {}
    structures: dict[str, Any] = {}

    for split in SPLITS:
        rows = rows_by_split[split]
        scenarios = scenarios_by_split[split]
        if len(rows) != len(scenarios):
            raise CorpusBuildError(f"{split} row and scenario counts differ")
        expected_forms = {
            form: int(value) for form, value in expected_counts[split].items()
        }
        forms = Counter(scenario.form for scenario in scenarios)
        if {form: forms[form] for form in expected_forms} != expected_forms:
            raise CorpusBuildError(f"{split} trace counts changed")

        statuses: Counter[str] = Counter()
        calls: Counter[str] = Counter()
        frames: Counter[str] = Counter()
        semantics: Counter[str] = Counter()
        families: Counter[str] = Counter()
        decisions: Counter[str] = Counter()
        templates: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        requests: dict[str, Counter[str]] = defaultdict(Counter)
        gates: dict[str, Counter[str]] = defaultdict(Counter)
        chosen_by_family: dict[str, list[str]] = defaultdict(list)
        split_selected_ids: set[str] = set()
        split_selected_units: set[str] = set()
        no_calls = 0
        harnesses = 0
        thinking_steps = 0

        for row, scenario in zip(rows, scenarios, strict=True):
            row_id = str(row["row_id"])
            if row_id in row_ids:
                raise CorpusBuildError(f"duplicate row ID {row_id}")
            row_ids.add(row_id)
            if scenario.module_family != FAMILY_BY_FORM.get(scenario.form):
                raise CorpusBuildError(f"row {row_id} has a wrong module family")
            source_fact = source_ids[split][scenario.module_family].get(
                scenario.module_id
            )
            if source_fact is None:
                raise CorpusBuildError(f"row {row_id} uses a non-source module ID")
            if scenario.split_unit != source_fact.split_unit:
                raise CorpusBuildError(f"row {row_id} changes its source split unit")
            if scenario.module_id in split_selected_ids:
                raise CorpusBuildError(f"row {row_id} reuses a source module ID")
            split_selected_ids.add(scenario.module_id)
            split_selected_units.add(scenario.split_unit)
            chosen_by_family[scenario.module_family].append(scenario.module_id)

            expected_row = build_row(
                scenario, contract=contract, tools=tools, prompts=prompts,
                final_reasoning=settings.get("train_final_reasoning"),
            )[0]
            if row != expected_row:
                raise CorpusBuildError(f"row {row_id} differs from its exact scenario")
            answer = _final_answer(row)
            result = answer["result"]
            expected_keys = settings["result_shapes"][scenario.module_family]["keys"]
            if list(result) != list(expected_keys):
                raise CorpusBuildError(f"row {row_id} changes result key order")
            if answer["status"] != "resolved":
                raise CorpusBuildError(f"row {row_id} changes its resolved status")
            if answer["provenance"] is not None:
                raise CorpusBuildError(f"row {row_id} has closed-book provenance")
            if answer["blocked_reason"] is not None:
                raise CorpusBuildError(f"row {row_id} invents a blocked reason")
            if result.get("existence_asserted") is not False:
                raise CorpusBuildError(f"row {row_id} asserts module existence")
            if result != parse_module_id(scenario.module_id):
                raise CorpusBuildError(f"row {row_id} changes the module parse")
            if any(message.get("role") == "harness" for message in row["messages"]):
                harnesses += 1
            row_calls = _tool_calls(row)
            calls[str(len(row_calls))] += 1
            if row_calls or any(
                message.get("role") == "tool" for message in row["messages"]
            ):
                raise CorpusBuildError(f"row {row_id} makes a forbidden tool call")
            no_calls += 1
            thinking_steps += filter_row_thinking(
                row, scenario, final_reasoning=settings.get("train_final_reasoning"),
            )

            form_key = scenario.source_trace
            gate = gates[form_key]
            gate["row_count"] += 1
            for key in (
                "status",
                "result",
                "continuation_decision",
                "blocked_reason",
            ):
                gate[f"exact_{key}_count"] += 1
            gate["null_provenance_count"] += 1
            gate["false_existence_count"] += 1
            gate["exact_call_count"] += 1
            statuses[str(answer["status"])] += 1
            frames[scenario.framing] += 1
            semantics[scenario.semantic_bucket] += 1
            families[scenario.module_family] += 1
            decisions[str(answer["continuation_decision"])] += 1
            templates[form_key][scenario.framing][
                str(scenario.template_index)
            ] += 1
            requests[form_key][str(scenario.request_index)] += 1

        mode = _mode(split)
        expected_statuses = {
            key: int(value)
            for key, value in settings["status_counts"][mode].items()
        }
        if dict(statuses) != expected_statuses:
            raise CorpusBuildError(f"{split} status counts changed")
        expected_frames = {
            key: int(value)
            for key, value in settings["framing_counts"][mode].items()
            if key in ("direct", "step")
        }
        if dict(frames) != expected_frames:
            raise CorpusBuildError(f"{split} framing counts changed")
        form_frames = settings["framing_counts"][f"{mode}_form_counts"]
        for form, family in FAMILY_BY_FORM.items():
            form_key = f"S1.9.{form}"
            expected_family_frames = {
                key: int(value)
                for key, value in form_frames[BUCKET_BY_TRACE[form]].items()
            }
            actual_family_frames = Counter(
                scenario.framing
                for scenario in scenarios
                if scenario.form == form
            )
            if dict(actual_family_frames) != expected_family_frames:
                raise CorpusBuildError(
                    f"{split} {family} framing counts changed"
                )
            for frame in ("direct", "step"):
                if set(templates[form_key][frame]) != {
                    str(index) for index in range(len(prompts[split][frame]))
                }:
                    raise CorpusBuildError(
                        f"{split} form {form} {frame} lacks template coverage"
                    )
            if set(requests[form_key]) != {str(index) for index in range(8)}:
                raise CorpusBuildError(
                    f"{split} form {form} lacks request coverage"
                )

        selected_ids[split] = split_selected_ids
        selected_units[split] = split_selected_units
        status_counts[split] = dict(sorted(statuses.items()))
        tool_call_counts[split] = dict(sorted(calls.items()))
        framing_counts[split] = dict(sorted(frames.items()))
        semantic_bucket_counts[split] = dict(sorted(semantics.items()))
        module_family_counts[split] = {
            family: families[family] for family in MODULE_FAMILIES
        }
        continuation_counts[split] = dict(sorted(decisions.items()))
        prompt_template_counts[split] = {
            form: {
                frame: dict(sorted(values.items()))
                for frame, values in sorted(frame_values.items())
            }
            for form, frame_values in sorted(templates.items())
        }
        request_template_counts[split] = {
            form: dict(sorted(values.items()))
            for form, values in sorted(requests.items())
        }
        selected_source_counts[split] = {
            family: len(chosen_by_family[family]) for family in MODULE_FAMILIES
        }
        selected_source_sha256[split] = {
            family: _stable_sha256(sorted(chosen_by_family[family]))
            for family in MODULE_FAMILIES
        }
        per_form[split] = {}
        for form, count in expected_forms.items():
            form_key = f"S1.9.{form}"
            values = gates[form_key]
            form_gate = {
                "row_count": count,
                "exact_status_count": values["exact_status_count"],
                "exact_result_count": values["exact_result_count"],
                "exact_continuation_decision_count": values[
                    "exact_continuation_decision_count"
                ],
                "exact_blocked_reason_count": values[
                    "exact_blocked_reason_count"
                ],
                "null_provenance_count": values["null_provenance_count"],
                "false_existence_count": values["false_existence_count"],
                "exact_call_count": values["exact_call_count"],
            }
            form_gate["pass"] = all(
                value == count
                for key, value in form_gate.items()
                if key.endswith("_count") and key != "row_count"
            )
            per_form[split][form_key] = form_gate
        thinking[split] = {
            "row_count": len(rows),
            "thinking_step_count": thinking_steps,
            "pass": thinking_steps == len(rows),
        }
        structures[split] = {
            "row_count": len(rows),
            "harness_message_count": harnesses,
            "no_tool_call_count": no_calls,
            "unique_module_id_count": len(split_selected_ids),
            "unique_split_unit_count": len(split_selected_units),
            "pass": harnesses == 0
            and no_calls == len(rows)
            and len(split_selected_ids) == len(rows)
            and len(split_selected_units) == len(rows),
        }

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if selected_ids[left] & selected_ids[right]:
                raise CorpusBuildError(
                    f"selected module IDs overlap in {left} and {right}"
                )
            if selected_units[left] & selected_units[right]:
                raise CorpusBuildError(
                    f"selected split units overlap in {left} and {right}"
                )

    duplicate = duplicate_prompt_outcome_audit(rows_by_split, scenarios_by_split)
    wording = masked_wording_classifier_audit(rows_by_split, scenarios_by_split)
    full_prompt = full_visible_prompt_classifier_diagnostic(
        rows_by_split, scenarios_by_split
    )
    prose = _prose_length_audit(rows_by_split)
    all_form_gates = all(
        gate["pass"]
        for split_gates in per_form.values()
        for gate in split_gates.values()
    )
    return {
        "status_counts": status_counts,
        "tool_call_counts": tool_call_counts,
        "framing_counts": framing_counts,
        "semantic_bucket_counts": semantic_bucket_counts,
        "module_family_counts": module_family_counts,
        "continuation_counts": continuation_counts,
        "prompt_template_counts": prompt_template_counts,
        "request_template_counts": request_template_counts,
        "selected_source_counts": selected_source_counts,
        "selected_source_sha256": selected_source_sha256,
        "source_pool_coverage": source_pool_coverage(source_pools),
        "split_unit_sha256": {
            split: _stable_sha256(sorted(selected_units[split]))
            for split in SPLITS
        },
        "per_form_gates": {
            "splits": per_form,
            "parse_accuracy": 1.0,
            "existence_asserted_true_count": 0,
            "non_source_id_count": 0,
            "tool_call_rate": 0.0,
            "pass": all_form_gates,
        },
        "prompt_structure_audit": {
            "splits": structures,
            "full_visible_prompt_classifier": full_prompt,
            "prose_length_audit": prose,
            "pass": all(item["pass"] for item in structures.values())
            and prose["pass"],
        },
        "masked_wording_classifier": wording,
        "thinking_filter": {
            "rules": list(THINKING_FILTER_RULES),
            "splits": thinking,
            "pass": all(item["pass"] for item in thinking.values()),
        },
        "duplicate_prompt_tool_outcome": duplicate,
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row, sort_keys=False) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_corpus(path: Path) -> None:
    validator = (
        REPOSITORY_ROOT
        / "agents"
        / "world_model_v3_curriculum"
        / "s1_kit"
        / "validate.py"
    )
    result = subprocess.run(
        [sys.executable, str(validator), str(path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CorpusBuildError(
            f"semantic validation failed for {path}:\n"
            + result.stdout
            + result.stderr
        )


def _configured_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _source_record(path: Path) -> dict[str, str]:
    """Return a portable record for one source file."""

    resolved = path.resolve()
    try:
        stored_path = str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        stored_path = str(resolved)
    return {"path": stored_path, "sha256": file_sha256(resolved)}


def build_corpus(
    source_index_path: Path,
    output_directory: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path = CONTRACT_PATH,
    validate: bool = True,
    overlay: Mapping[str, str] | None = None,
    prompt_banks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate S1.9, then publish its manifest last.

    A pinned bank and the v5 switch change training wrappers and reasoning.
    Validation and test keep their inline wording and complete rows.
    """

    settings = contract["curriculum"]["families"]["S1.9"]
    _assert_exact_settings(settings)
    seed = int(settings["assignment_seed"])
    prompts = (
        wrapper_prompts(prompt_banks, MODULE_PROMPTS, bank_name="module")
        if prompt_banks is not None
        else MODULE_PROMPTS
    )
    source_index_path = source_index_path.resolve()
    payload = check_source_index(source_index_path, contract)
    pools = build_fact_pools(
        payload,
        settings=settings,
        seed=seed,
        contract=contract,
    )
    row_counts = split_row_counts(
        int(settings["train_rows"]), contract["curriculum"]["split_fractions"]
    )
    counts_by_split = {
        split: trace_counts(row_counts[split], settings) for split in SPLITS
    }
    scenarios_by_split = select_scenarios(
        pools,
        counts_by_split,
        seed=seed,
        framing_shares=contract["curriculum"]["framing"],
        settings=settings,
        prompts=prompts,
    )
    tools = tool_definitions(contract)
    rows_by_split = {
        split: [
            build_row(
                scenario, contract=contract, tools=tools, prompts=prompts,
                final_reasoning=settings.get("train_final_reasoning"),
            )[0]
            for scenario in scenarios_by_split[split]
        ]
        for split in SPLITS
    }
    audit = audit_rows(
        rows_by_split,
        scenarios_by_split,
        counts_by_split,
        source_pools=pools,
        framing_shares=contract["curriculum"]["framing"],
        settings=settings,
        contract=contract,
        prompts=prompts,
    )

    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".s1_9_build_", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        paths = {split: temporary / f"{split}.jsonl" for split in SPLITS}
        for split, path in paths.items():
            _write_jsonl(path, rows_by_split[split])
            if validate:
                _validate_corpus(path)

        golden_traces = REPOSITORY_ROOT / "runtime/world_model/golden_traces.jsonl"
        builder_path = Path(__file__).resolve()
        s1_6_builder = REPOSITORY_ROOT / "runtime/world_model/s1_6_corpus.py"
        source_index_record = _source_record(source_index_path)
        source_index_record["content_sha256"] = contract["source_index"][
            "content_sha256"
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "section": "S1.9",
            "dataset_id": settings["dataset_id"],
            "seed": seed,
            "contract_s1_9_sha256": _stable_sha256(settings),
            "prompt_templates_sha256": _stable_sha256(prompts),
            "request_templates_sha256": _stable_sha256(PARSE_REQUESTS),
            "module_grammars_sha256": _stable_sha256(MODULE_GRAMMARS),
            "result_shapes_sha256": _stable_sha256(settings["result_shapes"]),
            "scenario_assignment_sha256": {
                split: _stable_sha256(
                    [item.scenario_id for item in scenarios_by_split[split]]
                )
                for split in SPLITS
            },
            "row_counts": row_counts,
            "form_counts": {
                split: {
                    f"S1.9.{form}": int(count)
                    for form, count in counts_by_split[split].items()
                }
                for split in SPLITS
            },
            "semantic_bucket_counts": audit["semantic_bucket_counts"],
            "status_counts": audit["status_counts"],
            "tool_call_counts": audit["tool_call_counts"],
            "framing_counts": audit["framing_counts"],
            "prompt_template_counts": audit["prompt_template_counts"],
            "request_template_counts": audit["request_template_counts"],
            "module_family_counts": audit["module_family_counts"],
            "continuation_counts": audit["continuation_counts"],
            "selected_source_counts": audit["selected_source_counts"],
            "selected_source_sha256": audit["selected_source_sha256"],
            "sources": {
                "source_index": source_index_record,
                "contract": _source_record(contract_path),
                "builder": _source_record(builder_path),
                "s1_6_builder": _source_record(s1_6_builder),
                "golden_traces": _source_record(golden_traces),
                "multiplex_flist": dict(payload["sources"]["multiplex_flist"]),
                "mentor_ev_dendrogram": dict(
                    payload["sources"]["mentor_ev_dendrogram"]
                ),
                "records": payload["sources"],
                "module_policy": (
                    "parse_only_split_safe_source_ids_without_existence_claims"
                ),
            },
            "source_pool_coverage": audit["source_pool_coverage"],
            "split_unit_sha256": audit["split_unit_sha256"],
            "per_form_gates": audit["per_form_gates"],
            "prompt_structure_audit": audit["prompt_structure_audit"],
            "masked_wording_classifier": audit["masked_wording_classifier"],
            "thinking_filter": audit["thinking_filter"],
            "duplicate_prompt_tool_outcome": audit[
                "duplicate_prompt_tool_outcome"
            ],
            "evidence": {
                "call_fields": ["evidence_id", "row_id", "producer"],
                "calls": [],
                "call_count": 0,
            },
            "files": {
                path.name: {
                    "row_count": row_counts[split],
                    "sha256": file_sha256(path),
                }
                for split, path in paths.items()
            },
        }
        expected_manifest_keys = {
            "schema_version",
            "corpus_schema_version",
            "section",
            "dataset_id",
            "seed",
            "contract_s1_9_sha256",
            "prompt_templates_sha256",
            "request_templates_sha256",
            "module_grammars_sha256",
            "result_shapes_sha256",
            "scenario_assignment_sha256",
            "row_counts",
            "form_counts",
            "semantic_bucket_counts",
            "status_counts",
            "tool_call_counts",
            "framing_counts",
            "prompt_template_counts",
            "request_template_counts",
            "module_family_counts",
            "continuation_counts",
            "selected_source_counts",
            "selected_source_sha256",
            "sources",
            "source_pool_coverage",
            "split_unit_sha256",
            "per_form_gates",
            "prompt_structure_audit",
            "masked_wording_classifier",
            "thinking_filter",
            "duplicate_prompt_tool_outcome",
            "evidence",
            "files",
        }
        if set(manifest) != expected_manifest_keys:
            raise CorpusBuildError("the S1.9 manifest fields changed")
        if overlay is not None:
            manifest["s1_overlay"] = dict(overlay)
        if prompt_banks is not None:
            manifest["prompt_bank"] = prompt_bank_manifest(prompt_banks)
        if "train_final_reasoning" in settings:
            manifest["train_final_reasoning"] = dict(settings["train_final_reasoning"])
        if settings.get("train_decorrelate"):
            manifest["train_decorrelation"] = module_decorrelation_audit(scenarios_by_split["train"])
        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest)

        output_directory.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            (temporary / name).replace(output_directory / name)
        manifest_path.replace(output_directory / "manifest.json")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument(
        "--overlay",
        type=Path,
        help="apply an S1 overlay such as runtime/world_model/s1_v3.yaml",
    )
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip semantic validation for a focused test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    overlay = None
    prompt_banks = None
    if args.overlay is not None:
        contract, payload = apply_s1_overlay(
            contract, args.overlay, base_contract_path=args.contract
        )
        overlay = overlay_pin(args.overlay, payload)
        prompt_banks = payload.get("prompt_banks")
    digest = contract["source_index"]["content_sha256"]
    source_index = args.source_index or (
        _configured_path(contract["source_index"]["output_directory"])
        / f"world_model_source_index_{digest}.json.gz"
    )
    settings = contract["curriculum"]["families"]["S1.9"]
    output_directory = args.output_directory or _configured_path(
        settings["output_directory"]
    )
    canonical_output = _configured_path(settings["output_directory"]).resolve()
    if args.no_validate and output_directory.resolve() == canonical_output:
        parser.error(
            "--no-validate requires an explicit noncanonical output directory"
        )
    manifest = build_corpus(
        source_index,
        output_directory,
        contract=contract,
        contract_path=args.contract,
        validate=not args.no_validate,
        overlay=overlay,
        prompt_banks=prompt_banks,
    )
    print(
        json.dumps(
            {
                "output_directory": str(output_directory.resolve()),
                "row_counts": manifest["row_counts"],
                "form_counts": manifest["form_counts"],
                "files": manifest["files"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
