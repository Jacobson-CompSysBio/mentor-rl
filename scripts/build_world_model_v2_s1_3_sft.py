#!/usr/bin/env python3
"""Build the S1.3 ambiguous-symbol supervised trace corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_contracts import (  # noqa: E402
    SUPERVISED_EXPLANATION_LOSS_CONTRACT,
    SUPERVISED_EXPLANATION_SCHEMA_VERSION,
    SUPERVISED_TRACE_SCHEMA_VERSION,
    SupervisedTraceRecord,
    stable_json_sha256,
    text_sha256,
)
from runtime.world_model_prompts import (  # noqa: E402
    s1_shared_trace_prompt_contract,
)
from runtime.world_model_s0_tools import (  # noqa: E402
    REGISTRY_SCHEMA_VERSION_V2,
    SYMBOL_TOOL_NAME,
    TOOL_CONTRACT_VERSION_V2,
    S0IdentifierToolRuntime,
    identifier_tool_definitions,
)
from runtime.world_model_s1_shared import (  # noqa: E402
    S1_3_EXPLANATION_TEMPLATES,
    S1_3_FORMS,
    S1_3_QUESTION_FAMILY,
    S1_SOURCE_REFS,
    build_s1_3_trace,
)
from scripts.build_world_model_v2_s0_shared_sft import (  # noqa: E402
    DATASET_ID as S0_DATASET_ID,
)
from scripts.build_world_model_v2_s0_tool_sft import (  # noqa: E402
    FAMILIES,
    build_components,
    protect_evaluator_dir,
    read_json,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    write_json,
    write_jsonl,
)


CONFIG_SCHEMA_VERSION = "mentor-rl-world-model-s1-3-build-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-3-manifest-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-3-split-manifest-v1"
)
EVALUATOR_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-3-evaluator-manifest-v1"
)
SHARED_SELECTION_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-1-s1-3-selection-v1"
)
EXPLANATION_AUDIT_SCHEMA_VERSION = (
    "mentor-rl-world-model-explanation-audit-v1"
)
DATASET_ID = "world_model_v2_s1_3_ambiguous_symbol_v1"
TRAINING_CONTRACT = "supervised_trace_source_pool_v1"
EVALUATION_CONTRACT = "held_out_s0_ambiguous_components_v1"
QUESTION_SCHEMA_VERSION = "mentor-rl-world-model-s1-question-v1"


@dataclass(frozen=True)
class SelectedFact:
    """Store one S1.3 form assignment."""

    form: str
    fact: Mapping[str, Any]
    sample_index: int
    multiplex_candidate: bool


def _require_object(value: Any, label: str) -> dict[str, Any]:
    """Return one required JSON object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one object")
    return {str(key): item for key, item in value.items()}


def _require_count(value: Any, label: str) -> int:
    """Return one positive row count."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _candidate_gene_ids(fact: Mapping[str, Any]) -> list[str]:
    """Return the candidate IDs from one ambiguous fact."""

    answer = fact.get("answer")
    if not isinstance(answer, Mapping):
        raise ValueError("An S1.3 fact has no answer object")
    candidates = answer.get("candidate_gene_ids")
    if (
        answer.get("status") != "ambiguous"
        or not isinstance(candidates, list)
        or len(candidates) < 2
        or any(not isinstance(value, str) for value in candidates)
        or candidates != sorted(set(candidates))
    ):
        raise ValueError("An S1.3 fact has invalid candidates")
    return list(candidates)


def _gene_symbol(fact: Mapping[str, Any]) -> str:
    """Return the symbol from one ambiguous fact."""

    inputs = fact.get("input")
    if not isinstance(inputs, Mapping):
        raise ValueError("An S1.3 fact has no input object")
    symbol = inputs.get("gene_symbol")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("An S1.3 fact has no symbol")
    return symbol


def assign_ambiguous_facts(
    components: list[dict[str, Any]],
    s0_split_manifest: Mapping[str, Any],
    *,
    maximum_candidates: int,
    expected_excluded_fact_ids: frozenset[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Assign capped ambiguous facts with the fixed S0 components."""

    validation_groups = frozenset(
        str(value)
        for value in s0_split_manifest["validation_fact_group_ids"]
    )
    test_groups = frozenset(
        str(value) for value in s0_split_manifest["test_fact_group_ids"]
    )
    if validation_groups & test_groups:
        raise ValueError("The S0 evaluation component sets overlap")

    result = {"train": [], "validation": [], "test": []}
    excluded = []
    for component in components:
        group_id = str(component["fact_group_id"])
        split = (
            "validation"
            if group_id in validation_groups
            else "test"
            if group_id in test_groups
            else "train"
        )
        for fact in component["facts"]:
            if fact["family"] != FAMILIES[2]:
                continue
            candidates = _candidate_gene_ids(fact)
            if len(candidates) > maximum_candidates:
                excluded.append(dict(fact))
            else:
                result[split].append(dict(fact))

    observed_excluded = frozenset(
        str(fact["fact_id"]) for fact in excluded
    )
    if observed_excluded != expected_excluded_fact_ids:
        raise ValueError("The S1.3 capped ambiguous fact set changed")
    for values in result.values():
        values.sort(key=lambda fact: str(fact["fact_id"]))
    return result, sorted(excluded, key=lambda fact: str(fact["fact_id"]))


def _stable_pool_selection(
    facts: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    split: str,
    form: str,
    pool_name: str,
) -> list[dict[str, Any]]:
    """Select one stable pool and cycle only when necessary."""

    if not facts:
        raise ValueError(f"The {split} {form} {pool_name} pool is empty")
    selected: list[dict[str, Any]] = []
    cycle = 0
    while len(selected) < count:
        ordered = sorted(
            facts,
            key=lambda fact: stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "form": form,
                    "pool": pool_name,
                    "cycle": cycle,
                    "fact_id": fact["fact_id"],
                }
            ),
        )
        selected.extend(ordered[: count - len(selected)])
        cycle += 1
    return selected


def select_form_facts(
    ambiguous_facts: list[dict[str, Any]],
    multiplex_gene_ids: frozenset[str],
    *,
    counts: Mapping[str, Any],
    seed: int,
    split: str,
    target_multiplex_fraction: float,
) -> list[SelectedFact]:
    """Select S1.3 forms and prefer multiplex candidates in train."""

    if set(counts) != set(S1_3_FORMS):
        raise ValueError("The S1.3 form count fields changed")
    multiplex_facts = [
        fact
        for fact in ambiguous_facts
        if set(_candidate_gene_ids(fact)) & multiplex_gene_ids
    ]
    other_facts = [
        fact
        for fact in ambiguous_facts
        if not set(_candidate_gene_ids(fact)) & multiplex_gene_ids
    ]
    selected: list[SelectedFact] = []
    for form in S1_3_FORMS:
        count = _require_count(counts[form], f"{split}.{form}")
        if split == "train":
            multiplex_count = round(count * target_multiplex_fraction)
            other_count = count - multiplex_count
            chosen = [
                (fact, True)
                for fact in _stable_pool_selection(
                    multiplex_facts,
                    count=multiplex_count,
                    seed=seed,
                    split=split,
                    form=form,
                    pool_name="multiplex_candidate",
                )
            ]
            chosen.extend(
                (fact, False)
                for fact in _stable_pool_selection(
                    other_facts,
                    count=other_count,
                    seed=seed,
                    split=split,
                    form=form,
                    pool_name="other_candidate",
                )
            )
        else:
            chosen = [
                (
                    fact,
                    bool(
                        set(_candidate_gene_ids(fact))
                        & multiplex_gene_ids
                    ),
                )
                for fact in _stable_pool_selection(
                    ambiguous_facts,
                    count=count,
                    seed=seed,
                    split=split,
                    form=form,
                    pool_name="evaluation",
                )
            ]
        chosen.sort(
            key=lambda item: stable_sha256(
                {
                    "assignment_seed": seed,
                    "split": split,
                    "form": form,
                    "fact_id": item[0]["fact_id"],
                    "multiplex_candidate": item[1],
                }
            )
        )
        selected.extend(
            SelectedFact(
                form=form,
                fact=fact,
                sample_index=index,
                multiplex_candidate=multiplex_candidate,
            )
            for index, (fact, multiplex_candidate) in enumerate(chosen)
        )
    return sorted(
        selected,
        key=lambda item: stable_sha256(
            {
                "assignment_seed": seed,
                "split": split,
                "form": item.form,
                "fact_id": item.fact["fact_id"],
                "sample_index": item.sample_index,
            }
        ),
    )


def _record_id(selection: SelectedFact, split: str) -> str:
    """Return one stable S1.3 record ID."""

    return "wm2_s1_3_" + stable_sha256(
        {
            "form": selection.form,
            "fact_id": selection.fact["fact_id"],
            "split": split,
            "sample_index": selection.sample_index,
            "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        }
    )[:24]


def render_question(
    *,
    config: Mapping[str, Any],
    s1_1_config: Mapping[str, Any],
    split: str,
    selection: SelectedFact,
) -> tuple[str, int]:
    """Render one deterministic S1.3 question."""

    prompt_form = "validation" if split == "validation" else split
    if selection.form == "already_marked":
        prompt_forms = _require_object(
            config.get("already_marked_prompt_forms"),
            "already_marked_prompt_forms",
        )
        templates = prompt_forms.get(prompt_form)
    else:
        prompt_forms = _require_object(
            s1_1_config.get("prompt_forms"),
            "s1_1.prompt_forms",
        )
        split_forms = _require_object(
            prompt_forms.get(prompt_form),
            f"s1_1.prompt_forms.{prompt_form}",
        )
        templates = split_forms.get(selection.form)
    if (
        not isinstance(templates, list)
        or not templates
        or any(not isinstance(value, str) or not value for value in templates)
    ):
        raise ValueError("Each S1.3 prompt form needs text templates")
    template_index = int(
        stable_sha256(
            {
                "prompt_assignment_seed": config["prompt_assignment_seed"],
                "split": split,
                "form": selection.form,
                "fact_id": selection.fact["fact_id"],
                "sample_index": selection.sample_index,
            }
        )[:16],
        16,
    ) % len(templates)
    gene_symbol = _gene_symbol(selection.fact)
    registry_entry = json.dumps(
        {
            "status": "ambiguous",
            "gene_symbol": gene_symbol,
            "candidate_gene_ids": _candidate_gene_ids(selection.fact),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    question = templates[template_index].format(
        gene_symbol=gene_symbol,
        registry_entry_json=registry_entry,
    )
    return question, template_index


def make_trace(
    selection: SelectedFact,
    *,
    config: Mapping[str, Any],
    s1_1_config: Mapping[str, Any],
    split: str,
    registry_id: str,
    runtime: S0IdentifierToolRuntime,
) -> SupervisedTraceRecord:
    """Build one selected S1.3 trace."""

    question, rendering_index = render_question(
        config=config,
        s1_1_config=s1_1_config,
        split=split,
        selection=selection,
    )
    gene_symbol = _gene_symbol(selection.fact)
    candidates = _candidate_gene_ids(selection.fact)
    if selection.form == "already_marked":
        tool_result = None
        tool_provenance = None
    else:
        result = runtime.execute(
            SYMBOL_TOOL_NAME,
            {"gene_symbol": gene_symbol},
        )
        tool_result = result.payload
        tool_provenance = dict(result.provenance)
    trace_split = "train" if split == "train" else "val"
    provenance = {
        "fact_id": selection.fact["fact_id"],
        "fact_group_id": selection.fact["fact_group_id"],
        "fact_role": "train" if split == "train" else "unseen",
        "prompt_form_id": split,
        "question_form": selection.form,
        "rendering_index": rendering_index,
        "sample_index": selection.sample_index,
        "source_kind": "ensembl_registry",
        "s0_split_contract": "identifier_component_v7",
        "multiplex_candidate": selection.multiplex_candidate,
        "candidate_gene_id_count": len(candidates),
    }
    if tool_provenance is not None:
        provenance["tool_result_provenance"] = tool_provenance
    return build_s1_3_trace(
        record_id=_record_id(selection, split),
        question=question,
        form=selection.form,
        gene_symbol=gene_symbol,
        candidate_gene_ids=candidates,
        tool_result=tool_result,
        split=trace_split,
        provenance=provenance,
        identifier_registry_id=registry_id,
    )


def public_question_row(
    trace: SupervisedTraceRecord,
    selection: SelectedFact,
    *,
    split: str,
) -> dict[str, Any]:
    """Return one answer-free S1.3 question."""

    metadata = {
        str(key): value
        for key, value in trace.metadata.items()
        if key != "requires_user_input"
    }
    metadata["question_schema_version"] = QUESTION_SCHEMA_VERSION
    return {
        "record_id": trace.record_id,
        "metadata": metadata,
        "system": trace.messages[0]["content"],
        "question": trace.messages[1]["content"],
        "input": {"gene_symbol": _gene_symbol(selection.fact)},
        "tools": [dict(tool) for tool in trace.tools],
        "split": split,
        "provenance": {
            key: value
            for key, value in trace.provenance.items()
            if key
            not in {
                "tool_result_provenance",
                "candidate_gene_id_count",
                "multiplex_candidate",
            }
        },
    }


def answer_key_row(
    trace: SupervisedTraceRecord,
    selection: SelectedFact,
) -> dict[str, Any]:
    """Return one private S1.3 answer key."""

    actions = []
    for index in range(trace.tool_call_count):
        call_message = trace.messages[2 + index * 2]
        tool_message = trace.messages[3 + index * 2]
        function = call_message["tool_call"]["function"]
        actions.append(
            {
                "expected_tool_call": {
                    "name": function["name"],
                    "arguments": dict(function["arguments"]),
                },
                "expected_tool_payload": dict(tool_message["content"]),
                "expected_assistant_explanation": call_message[
                    "assistant_explanation"
                ],
                "expected_explanation_receipt": dict(
                    call_message["explanation_receipt"]
                ),
            }
        )
    final_message = trace.messages[-1]
    answer = trace.final_answer
    return {
        "record_id": trace.record_id,
        "fact_id": selection.fact["fact_id"],
        "fact_group_id": selection.fact["fact_group_id"],
        "family": S1_3_QUESTION_FAMILY,
        "form": trace.metadata["question_form"],
        "expected_status": answer.status,
        "expected_continuation_decision": answer.continuation_decision,
        "expected_requires_user_input": True,
        "expected_tool_call_count": trace.tool_call_count,
        "expected_tool_actions": actions,
        "expected_final_explanation": final_message[
            "assistant_explanation"
        ],
        "expected_final_explanation_receipt": dict(
            final_message["explanation_receipt"]
        ),
        "expected_final_answer": final_message["content"],
        "source_kind": "ensembl_registry",
    }


def build_explanation_audit(
    train_rows: list[SupervisedTraceRecord],
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Build the S1.3 template and fixed-sample audit."""

    samples: dict[str, list[dict[str, Any]]] = {}
    for form in S1_3_FORMS:
        form_rows = [
            row for row in train_rows if row.metadata["question_form"] == form
        ]
        if len(form_rows) < sample_size:
            raise ValueError(f"The {form} audit pool is too small")
        selected = sorted(
            form_rows,
            key=lambda row: stable_json_sha256(
                {
                    "sample_seed": seed,
                    "form": form,
                    "record_id": row.record_id,
                }
            ),
        )[:sample_size]
        samples[form] = [
            {
                "record_id": row.record_id,
                "actions": [
                    {
                        "action_index": index,
                        "assistant_explanation": message[
                            "assistant_explanation"
                        ],
                        "explanation_receipt": dict(
                            message["explanation_receipt"]
                        ),
                    }
                    for index, message in enumerate(
                        value
                        for value in row.messages
                        if value["role"] == "assistant"
                    )
                ],
            }
            for row in selected
        ]
    templates = {
        action_path: [
            {
                "explanation_template_id": template.template_id,
                "explanation_template": template.text,
                "explanation_template_sha256": text_sha256(template.text),
                "explanation_rule_ids": list(template.rule_ids),
                "explanation_source_refs": list(S1_SOURCE_REFS),
            }
            for template in action_templates
        ]
        for action_path, action_templates in sorted(
            S1_3_EXPLANATION_TEMPLATES.items()
        )
    }
    return {
        "schema_version": EXPLANATION_AUDIT_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "explanation_contract_version": (
            SUPERVISED_EXPLANATION_SCHEMA_VERSION
        ),
        "model_text_created": False,
        "model_sample_used": False,
        "template_review_required": True,
        "sample_review_required": True,
        "template_count": sum(len(value) for value in templates.values()),
        "templates": templates,
        "sample_contract": {
            "sample_seed": seed,
            "sample_size_per_form": sample_size,
            "selection": "stable_hash_of_form_and_record_id",
        },
        "samples": samples,
    }


def build_shared_pool_selection(
    train_rows: list[SupervisedTraceRecord],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select S1.3 rows for the shared S1.1 and S1.3 pool."""

    selection_config = _require_object(
        config.get("shared_pool_selection"),
        "shared_pool_selection",
    )
    form_counts = _require_object(
        selection_config.get("s1_3_form_counts"),
        "shared_pool_selection.s1_3_form_counts",
    )
    if set(form_counts) != set(S1_3_FORMS):
        raise ValueError("The shared S1.3 form counts changed")
    seed = int(selection_config["selection_seed"])
    selected_by_form = {}
    for form in S1_3_FORMS:
        count = _require_count(form_counts[form], f"shared.{form}")
        rows = [
            row for row in train_rows if row.metadata["question_form"] == form
        ]
        if len(rows) < count:
            raise ValueError("The shared S1.3 selection pool is too small")
        selected_by_form[form] = [
            row.record_id
            for row in sorted(
                rows,
                key=lambda row: stable_json_sha256(
                    {
                        "selection_seed": seed,
                        "form": form,
                        "record_id": row.record_id,
                    }
                ),
            )[:count]
        ]
    s1_1_count = _require_count(
        selection_config.get("s1_1_train_row_count"),
        "shared.s1_1_train_row_count",
    )
    s1_3_count = sum(len(values) for values in selected_by_form.values())
    ambiguous_fraction = s1_3_count / (s1_1_count + s1_3_count)
    minimum = float(selection_config["minimum_ambiguous_fraction"])
    maximum = float(selection_config["maximum_ambiguous_fraction"])
    if not minimum <= ambiguous_fraction <= maximum:
        raise ValueError("The shared ambiguous fraction is outside its range")
    return {
        "schema_version": SHARED_SELECTION_SCHEMA_VERSION,
        "s1_1_dataset_id": selection_config["s1_1_dataset_id"],
        "s1_3_dataset_id": DATASET_ID,
        "selection_seed": seed,
        "s1_1_train_row_count": s1_1_count,
        "s1_3_train_row_count": s1_3_count,
        "combined_train_row_count": s1_1_count + s1_3_count,
        "ambiguous_fraction": ambiguous_fraction,
        "allowed_ambiguous_fraction": {
            "minimum": minimum,
            "maximum": maximum,
        },
        "selected_record_ids_by_form": selected_by_form,
    }


def _counts(
    traces: Iterable[SupervisedTraceRecord],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return form, status, and tool-call counts."""

    traces = list(traces)
    return (
        dict(
            sorted(
                Counter(
                    row.metadata["question_form"] for row in traces
                ).items()
            )
        ),
        dict(sorted(Counter(row.final_answer.status for row in traces).items())),
        dict(
            sorted(
                Counter(str(row.tool_call_count) for row in traces).items()
            )
        ),
    )


def _population_stats(
    facts: Iterable[Mapping[str, Any]],
    multiplex_gene_ids: frozenset[str],
) -> dict[str, Any]:
    """Return counts for one ambiguous fact population."""

    facts = list(facts)
    candidate_counts = Counter(
        len(_candidate_gene_ids(fact)) for fact in facts
    )
    multiplex_count = sum(
        bool(set(_candidate_gene_ids(fact)) & multiplex_gene_ids)
        for fact in facts
    )
    return {
        "row_count": len(facts),
        "multiplex_candidate_row_count": multiplex_count,
        "candidate_count_histogram": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
    }


def build_corpus(config_path: Path) -> dict[str, Any]:
    """Build and write the S1.3 corpus."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("The config must use the S1.3 build schema")
    if config.get("dataset_id") != DATASET_ID:
        raise ValueError("The S1.3 dataset ID changed")

    registry_config = _require_object(
        config.get("identifier_registry"),
        "identifier_registry",
    )
    registry_path = resolve_repo_path(registry_config.get("path"))
    registry_sha256 = str(registry_config.get("expected_sha256", ""))
    if sha256_file(registry_path) != registry_sha256:
        raise ValueError("The identifier registry identity changed")
    registry = read_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION_V2:
        raise ValueError("S1.3 requires registry version 2")
    mappings = registry.get("gene_symbols_by_id")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The identifier registry has no mappings")
    runtime = S0IdentifierToolRuntime.from_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )

    multiplex_config = _require_object(
        config.get("multiplex_gene_universe"),
        "multiplex_gene_universe",
    )
    multiplex_path = resolve_repo_path(multiplex_config.get("path"))
    multiplex_sha256 = str(multiplex_config.get("expected_sha256", ""))
    if sha256_file(multiplex_path) != multiplex_sha256:
        raise ValueError("The multiplex gene universe identity changed")
    multiplex_payload = read_json(multiplex_path)
    raw_multiplex_gene_ids = multiplex_payload.get("gene_ids")
    if (
        not isinstance(raw_multiplex_gene_ids, list)
        or not raw_multiplex_gene_ids
        or any(not isinstance(value, str) for value in raw_multiplex_gene_ids)
        or raw_multiplex_gene_ids != sorted(set(raw_multiplex_gene_ids))
    ):
        raise ValueError("The multiplex gene universe is invalid")
    multiplex_gene_ids = frozenset(raw_multiplex_gene_ids)

    s0_contract = _require_object(
        config.get("s0_split_contract"),
        "s0_split_contract",
    )
    s0_config_path = resolve_repo_path(s0_contract.get("build_config_path"))
    s0_split_path = resolve_repo_path(s0_contract.get("split_manifest_path"))
    if sha256_file(s0_config_path) != s0_contract.get("build_config_sha256"):
        raise ValueError("The S0 build config identity changed")
    if sha256_file(s0_split_path) != s0_contract.get("split_manifest_sha256"):
        raise ValueError("The S0 split manifest identity changed")
    s0_config = read_json(s0_config_path)
    s0_split = read_json(s0_split_path)
    if (
        s0_config.get("dataset_id") != S0_DATASET_ID
        or s0_split.get("dataset_id") != S0_DATASET_ID
    ):
        raise ValueError("The S0 split dataset changed")

    s1_1_contract = _require_object(
        config.get("s1_1_question_pool"),
        "s1_1_question_pool",
    )
    s1_1_config_path = resolve_repo_path(s1_1_contract.get("config_path"))
    if sha256_file(s1_1_config_path) != s1_1_contract.get("config_sha256"):
        raise ValueError("The S1.1 question pool identity changed")
    s1_1_config = read_json(s1_1_config_path)
    if s1_1_config.get("dataset_id") != config["shared_pool_selection"][
        "s1_1_dataset_id"
    ]:
        raise ValueError("The S1.1 question pool dataset changed")

    population_filter = _require_object(
        config.get("population_filter"),
        "population_filter",
    )
    maximum_candidates = _require_count(
        population_filter.get("maximum_candidate_gene_ids"),
        "maximum_candidate_gene_ids",
    )
    raw_excluded = population_filter.get("expected_excluded_fact_ids")
    if (
        not isinstance(raw_excluded, list)
        or any(not isinstance(value, str) for value in raw_excluded)
        or len(set(raw_excluded)) != len(raw_excluded)
    ):
        raise ValueError("The expected S1.3 exclusions are invalid")
    ambiguous_facts, excluded_facts = assign_ambiguous_facts(
        build_components(mappings),
        s0_split,
        maximum_candidates=maximum_candidates,
        expected_excluded_fact_ids=frozenset(raw_excluded),
    )

    sampling = _require_object(config.get("sampling"), "sampling")
    seed = int(sampling["assignment_seed"])
    target_multiplex_fraction = float(
        sampling["target_multiplex_candidate_fraction"]
    )
    if not 0.0 < target_multiplex_fraction < 1.0:
        raise ValueError("The multiplex candidate fraction is invalid")
    allocations = _require_object(config.get("allocations"), "allocations")
    selections = {
        split: select_form_facts(
            ambiguous_facts[split],
            multiplex_gene_ids,
            counts=_require_object(allocations.get(split), split),
            seed=seed,
            split=split,
            target_multiplex_fraction=target_multiplex_fraction,
        )
        for split in ("train", "validation", "test")
    }
    registry_id = f"sha256:{registry_sha256}"
    traces = {
        split: [
            make_trace(
                selection,
                config=config,
                s1_1_config=s1_1_config,
                split=split,
                registry_id=registry_id,
                runtime=runtime,
            )
            for selection in selections[split]
        ]
        for split in ("train", "validation", "test")
    }

    group_sets = {
        split: {
            str(row.provenance["fact_group_id"])
            for row in split_rows
        }
        for split, split_rows in traces.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if group_sets[left] & group_sets[right]:
            raise ValueError(f"The {left} and {right} component sets overlap")

    outputs = _require_object(config.get("outputs"), "outputs")
    corpus_dir = resolve_repo_path(outputs.get("corpus_dir"))
    evaluator_dir = resolve_repo_path(outputs.get("evaluator_dir"))
    paths = {
        "train": corpus_dir / "train.jsonl",
        "validation": corpus_dir / "val.jsonl",
        "split_manifest": corpus_dir / "split_manifest.json",
        "shared_selection": corpus_dir / "shared_pool_train_selection.json",
        "explanation_audit": corpus_dir / "explanation_audit.json",
        "manifest": corpus_dir / "manifest.json",
        "validation_questions": evaluator_dir / "validation_questions.jsonl",
        "validation_answers": evaluator_dir / "validation_answer_key.jsonl",
        "test_questions": evaluator_dir / "test_questions.jsonl",
        "test_answers": evaluator_dir / "test_answer_key.jsonl",
        "evaluator_manifest": evaluator_dir / "manifest.json",
    }
    write_jsonl(paths["train"], [row.to_dict() for row in traces["train"]])
    write_jsonl(
        paths["validation"],
        [row.to_dict() for row in traces["validation"]],
    )
    for split in ("validation", "test"):
        write_jsonl(
            paths[f"{split}_questions"],
            [
                public_question_row(
                    row,
                    selection,
                    split="val" if split == "validation" else "test",
                )
                for row, selection in zip(
                    traces[split],
                    selections[split],
                    strict=True,
                )
            ],
        )
        write_jsonl(
            paths[f"{split}_answers"],
            [
                answer_key_row(row, selection)
                for row, selection in zip(
                    traces[split],
                    selections[split],
                    strict=True,
                )
            ],
        )

    audit_config = _require_object(
        config.get("explanation_audit"),
        "explanation_audit",
    )
    audit = build_explanation_audit(
        traces["train"],
        sample_size=_require_count(
            audit_config.get("sample_size_per_form"),
            "sample_size_per_form",
        ),
        seed=int(audit_config["sample_seed"]),
    )
    write_json(paths["explanation_audit"], audit)
    shared_selection = build_shared_pool_selection(traces["train"], config)
    write_json(paths["shared_selection"], shared_selection)

    row_counts = {split: len(value) for split, value in traces.items()}
    form_counts = {}
    status_counts = {}
    tool_call_counts = {}
    multiplex_candidate_counts = {}
    for split, split_rows in traces.items():
        forms, statuses, calls = _counts(split_rows)
        form_counts[split] = forms
        status_counts[split] = statuses
        tool_call_counts[split] = calls
        multiplex_candidate_counts[split] = sum(
            bool(row.provenance["multiplex_candidate"])
            for row in split_rows
        )
    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "assignment_seed": seed,
        "source_s0_split_manifest_sha256": sha256_file(s0_split_path),
        "row_counts": row_counts,
        "form_counts": form_counts,
        "status_counts": status_counts,
        "tool_call_counts": tool_call_counts,
        "multiplex_candidate_counts": multiplex_candidate_counts,
        "fact_group_ids": {
            "train_sha256": stable_sha256(sorted(group_sets["train"])),
            "validation": sorted(group_sets["validation"]),
            "test": sorted(group_sets["test"]),
        },
    }
    write_json(paths["split_manifest"], split_manifest)

    evaluator_manifest = {
        "schema_version": EVALUATOR_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "evaluation_contract": EVALUATION_CONTRACT,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "answer_schema_version": traces["train"][0].metadata[
            "answer_schema_version"
        ],
        "question_schema_version": QUESTION_SCHEMA_VERSION,
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
        },
    }
    for split in ("validation", "test"):
        evaluator_manifest[split] = {
            "questions_path": str(
                paths[f"{split}_questions"].relative_to(REPO_ROOT)
            ),
            "questions_sha256": sha256_file(paths[f"{split}_questions"]),
            "answer_key_path": str(
                paths[f"{split}_answers"].relative_to(REPO_ROOT)
            ),
            "answer_key_sha256": sha256_file(paths[f"{split}_answers"]),
            "row_count": len(traces[split]),
        }
    evaluator_manifest["test"]["test_panel_id"] = evaluator_manifest[
        "test"
    ]["questions_sha256"]
    write_json(paths["evaluator_manifest"], evaluator_manifest)

    population = {
        split: _population_stats(facts, multiplex_gene_ids)
        for split, facts in ambiguous_facts.items()
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "training_contract": TRAINING_CONTRACT,
        "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": evaluator_manifest["identifier_registry"],
        "multiplex_gene_universe": {
            "path": str(multiplex_path.relative_to(REPO_ROOT)),
            "sha256": multiplex_sha256,
            "gene_id_count": len(multiplex_gene_ids),
        },
        "population_filter": {
            "maximum_candidate_gene_ids": maximum_candidates,
            "excluded_records": [
                {
                    "fact_id": fact["fact_id"],
                    "gene_symbol": _gene_symbol(fact),
                    "candidate_gene_id_count": len(
                        _candidate_gene_ids(fact)
                    ),
                }
                for fact in excluded_facts
            ],
        },
        "eligible_population": population,
        "s0_split_contract": {
            "dataset_id": S0_DATASET_ID,
            "build_config_path": str(s0_config_path.relative_to(REPO_ROOT)),
            "build_config_sha256": sha256_file(s0_config_path),
            "split_manifest_path": str(s0_split_path.relative_to(REPO_ROOT)),
            "split_manifest_sha256": sha256_file(s0_split_path),
        },
        "s1_1_question_pool": {
            "config_path": str(s1_1_config_path.relative_to(REPO_ROOT)),
            "config_sha256": sha256_file(s1_1_config_path),
        },
        "system_prompt_contract": s1_shared_trace_prompt_contract().to_dict(),
        "tool_contract_version": TOOL_CONTRACT_VERSION_V2,
        "tool_definitions": identifier_tool_definitions(
            contract_version=TOOL_CONTRACT_VERSION_V2
        ),
        "config_sha256": sha256_file(config_path),
        "row_counts": row_counts,
        "form_counts": form_counts,
        "status_counts": status_counts,
        "tool_call_counts": tool_call_counts,
        "multiplex_candidate_counts": multiplex_candidate_counts,
        "shared_pool_selection": {
            "path": str(paths["shared_selection"].relative_to(REPO_ROOT)),
            "sha256": sha256_file(paths["shared_selection"]),
            "s1_3_train_row_count": shared_selection[
                "s1_3_train_row_count"
            ],
            "ambiguous_fraction": shared_selection["ambiguous_fraction"],
        },
        "explanation_audit": {
            "path": str(paths["explanation_audit"].relative_to(REPO_ROOT)),
            "sha256": sha256_file(paths["explanation_audit"]),
            "model_text_created": False,
            "model_sample_used": False,
        },
        "file_hashes": {
            "train.jsonl": sha256_file(paths["train"]),
            "val.jsonl": sha256_file(paths["validation"]),
            "split_manifest.json": sha256_file(paths["split_manifest"]),
            "shared_pool_train_selection.json": sha256_file(
                paths["shared_selection"]
            ),
            "explanation_audit.json": sha256_file(
                paths["explanation_audit"]
            ),
            "evaluator_manifest.json": sha256_file(
                paths["evaluator_manifest"]
            ),
        },
    }
    write_json(paths["manifest"], manifest)
    protect_evaluator_dir(evaluator_dir)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT / "config/world_model_v2_s1_3_ambiguous_symbol_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured S1.3 corpus."""

    manifest = build_corpus(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
