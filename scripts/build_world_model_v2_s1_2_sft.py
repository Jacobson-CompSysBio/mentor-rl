#!/usr/bin/env python3
"""Build the S1.2 Ensembl-to-symbol supervised trace corpus."""

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
    ENSEMBL_TOOL_NAME,
    REGISTRY_SCHEMA_VERSION_V2,
    TOOL_CONTRACT_VERSION_V2,
    S0IdentifierToolRuntime,
    identifier_tool_definitions,
)
from runtime.world_model_s1_shared import (  # noqa: E402
    S1_2_EXPLANATION_TEMPLATES,
    S1_2_FORMS,
    S1_2_INVALID_KINDS,
    S1_2_QUESTION_FAMILY,
    S1_SOURCE_REFS,
    build_s1_2_trace,
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


CONFIG_SCHEMA_VERSION = "mentor-rl-world-model-s1-2-build-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-2-manifest-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-2-split-manifest-v1"
)
EVALUATOR_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-2-evaluator-manifest-v1"
)
EXPLANATION_AUDIT_SCHEMA_VERSION = (
    "mentor-rl-world-model-explanation-audit-v1"
)
DATASET_ID = "world_model_v2_s1_2_ensembl_to_symbol_v1"
TRAINING_CONTRACT = "supervised_trace_v1"
EVALUATION_CONTRACT = "held_out_s0_components_v1"
QUESTION_SCHEMA_VERSION = "mentor-rl-world-model-s1-question-v1"


@dataclass(frozen=True)
class SelectedFact:
    """Store one S1.2 form assignment."""

    form: str
    fact: Mapping[str, Any]
    input_gene_id: str
    task_context: str
    invalid_kind: str | None = None


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


def _fact_gene_id(fact: Mapping[str, Any]) -> str:
    """Return the Ensembl ID from one S0 fact."""

    inputs = fact.get("input")
    if not isinstance(inputs, Mapping):
        raise ValueError("An S1.2 source fact has no input object")
    gene_id = inputs.get("gene_id")
    if not isinstance(gene_id, str):
        raise ValueError("An S1.2 source fact has no gene ID")
    return gene_id


def _fact_gene_symbol(fact: Mapping[str, Any]) -> str | None:
    """Return the one symbol or null from one S0 fact."""

    answer = fact.get("answer")
    if not isinstance(answer, Mapping):
        raise ValueError("An S1.2 source fact has no answer object")
    symbols = answer.get("gene_symbols")
    if not isinstance(symbols, list) or len(symbols) > 1:
        raise ValueError("An S1.2 source fact has an invalid symbol list")
    if not symbols:
        return None
    symbol = symbols[0]
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("An S1.2 source fact has an invalid symbol")
    return symbol


def _ordered_facts(
    facts: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    split: str,
    form: str,
) -> list[dict[str, Any]]:
    """Return one stable fact order for a form."""

    return sorted(
        (dict(fact) for fact in facts),
        key=lambda fact: stable_sha256(
            {
                "assignment_seed": seed,
                "split": split,
                "form": form,
                "fact_id": fact["fact_id"],
            }
        ),
    )


def assign_registry_facts(
    components: list[dict[str, Any]],
    s0_split_manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Assign Ensembl ID facts with the fixed S0 components."""

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
    seen_validation: set[str] = set()
    seen_test: set[str] = set()
    for component in components:
        group_id = str(component["fact_group_id"])
        if group_id in validation_groups:
            split = "validation"
            seen_validation.add(group_id)
        elif group_id in test_groups:
            split = "test"
            seen_test.add(group_id)
        else:
            split = "train"
        result[split].extend(
            dict(fact)
            for fact in component["facts"]
            if fact["family"] == FAMILIES[1]
        )

    expected_validation = {
        value
        for value in validation_groups
        if not value.startswith("group_not_found_")
    }
    expected_test = {
        value
        for value in test_groups
        if not value.startswith("group_not_found_")
    }
    if seen_validation != expected_validation or seen_test != expected_test:
        raise ValueError("The S0 registry component assignments changed")
    return result


def _versioned_gene_id(
    fact: Mapping[str, Any],
    *,
    seed: int,
    split: str,
) -> str:
    """Return one stable versioned Ensembl ID."""

    version = 1 + int(
        stable_sha256(
            {
                "assignment_seed": seed,
                "split": split,
                "form": "version_suffix",
                "fact_id": fact["fact_id"],
            }
        )[:8],
        16,
    ) % 25
    return f"{_fact_gene_id(fact)}.{version}"


def _invalid_gene_id(gene_id: str, invalid_kind: str) -> str:
    """Return one invalid ID with the selected form."""

    digits = gene_id.removeprefix("ENSG")
    if invalid_kind == "transcript":
        return f"ENST{digits}"
    if invalid_kind == "mouse_gene":
        return f"ENSMUSG{digits}"
    if invalid_kind == "malformed":
        return f"ENSG{digits[:-1]}"
    raise ValueError("The S1.2 invalid input kind is invalid")


def select_form_facts(
    registry_facts: list[dict[str, Any]],
    *,
    counts: Mapping[str, Any],
    seed: int,
    split: str,
) -> list[SelectedFact]:
    """Select the configured S1.2 forms for one split."""

    if set(counts) != set(S1_2_FORMS):
        raise ValueError("The S1.2 form count fields changed")
    named_facts = [
        fact for fact in registry_facts if _fact_gene_symbol(fact) is not None
    ]
    unnamed_facts = [
        fact for fact in registry_facts if _fact_gene_symbol(fact) is None
    ]
    selected: list[SelectedFact] = []
    for form in S1_2_FORMS:
        count = _require_count(counts[form], f"{split}.{form}")
        source = unnamed_facts if form == "no_symbol" else named_facts
        ordered = _ordered_facts(
            source,
            seed=seed,
            split=split,
            form=form,
        )
        if len(ordered) < count:
            raise ValueError(
                f"The {split} {form} pool has {len(ordered)} rows, "
                f"but the config needs {count}"
            )
        for index, fact in enumerate(ordered[:count]):
            gene_id = _fact_gene_id(fact)
            task_context = "inside_task" if form == "inside_task" else "direct"
            invalid_kind = None
            input_gene_id = gene_id
            if form == "version_suffix":
                input_gene_id = _versioned_gene_id(
                    fact,
                    seed=seed,
                    split=split,
                )
            elif form == "no_symbol":
                task_context = (
                    "direct" if index < (count + 1) // 2 else "inside_task"
                )
            elif form == "invalid_input":
                invalid_kind = S1_2_INVALID_KINDS[
                    index % len(S1_2_INVALID_KINDS)
                ]
                input_gene_id = _invalid_gene_id(gene_id, invalid_kind)
            selected.append(
                SelectedFact(
                    form=form,
                    fact=fact,
                    input_gene_id=input_gene_id,
                    task_context=task_context,
                    invalid_kind=invalid_kind,
                )
            )
    return sorted(
        selected,
        key=lambda item: stable_sha256(
            {
                "assignment_seed": seed,
                "split": split,
                "form": item.form,
                "fact_id": item.fact["fact_id"],
                "task_context": item.task_context,
                "invalid_kind": item.invalid_kind,
            }
        ),
    )


def _record_id(selection: SelectedFact, split: str) -> str:
    """Return one stable S1.2 record ID."""

    return "wm2_s1_2_" + stable_sha256(
        {
            "form": selection.form,
            "fact_id": selection.fact["fact_id"],
            "split": split,
            "input_gene_id": selection.input_gene_id,
            "task_context": selection.task_context,
            "invalid_kind": selection.invalid_kind,
            "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        }
    )[:24]


def render_question(
    *,
    config: Mapping[str, Any],
    split: str,
    selection: SelectedFact,
) -> tuple[str, int]:
    """Render one deterministic S1.2 question."""

    prompt_form = "validation" if split == "validation" else split
    template_form = selection.form
    if selection.form == "no_symbol":
        template_form = f"no_symbol_{selection.task_context}"
    prompt_forms = _require_object(config.get("prompt_forms"), "prompt_forms")
    split_forms = _require_object(
        prompt_forms.get(prompt_form),
        f"prompt_forms.{prompt_form}",
    )
    templates = split_forms.get(template_form)
    if (
        not isinstance(templates, list)
        or not templates
        or any(not isinstance(value, str) or not value for value in templates)
    ):
        raise ValueError("Each S1.2 prompt form needs text templates")
    template_index = int(
        stable_sha256(
            {
                "prompt_assignment_seed": config["prompt_assignment_seed"],
                "split": split,
                "form": selection.form,
                "task_context": selection.task_context,
                "fact_id": selection.fact["fact_id"],
            }
        )[:16],
        16,
    ) % len(templates)
    gene_id = _fact_gene_id(selection.fact)
    gene_symbol = _fact_gene_symbol(selection.fact)
    registry_entry = json.dumps(
        {
            "status": "resolved",
            "gene_id": gene_id,
            "gene_symbol": gene_symbol,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    question = templates[template_index].format(
        gene_id=gene_id,
        canonical_gene_id=gene_id,
        input_gene_id=selection.input_gene_id,
        gene_symbol=gene_symbol,
        registry_entry_json=registry_entry,
    )
    return question, template_index


def make_trace(
    selection: SelectedFact,
    *,
    config: Mapping[str, Any],
    split: str,
    registry_id: str,
    runtime: S0IdentifierToolRuntime,
) -> SupervisedTraceRecord:
    """Build one selected S1.2 trace."""

    question, rendering_index = render_question(
        config=config,
        split=split,
        selection=selection,
    )
    gene_id = _fact_gene_id(selection.fact)
    gene_symbol = _fact_gene_symbol(selection.fact)
    if selection.form in {"already_resolved", "invalid_input"}:
        tool_result = None
        tool_provenance = None
    else:
        result = runtime.execute(
            ENSEMBL_TOOL_NAME,
            {"gene_id": gene_id},
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
        "task_context": selection.task_context,
        "rendering_index": rendering_index,
        "source_kind": (
            "synthetic_invalid_identifier"
            if selection.form == "invalid_input"
            else "ensembl_registry"
        ),
        "s0_split_contract": "identifier_component_v7",
    }
    if selection.invalid_kind is not None:
        provenance["invalid_input_kind"] = selection.invalid_kind
    if tool_provenance is not None:
        provenance["tool_result_provenance"] = tool_provenance
    return build_s1_2_trace(
        record_id=_record_id(selection, split),
        question=question,
        form=selection.form,
        input_gene_id=selection.input_gene_id,
        canonical_gene_id=(
            None if selection.form == "invalid_input" else gene_id
        ),
        gene_symbol=(
            None if selection.form == "invalid_input" else gene_symbol
        ),
        tool_result=tool_result,
        task_context=selection.task_context,
        invalid_kind=selection.invalid_kind,
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
    """Return one answer-free S1.2 question."""

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
        "input": {"gene_id": selection.input_gene_id},
        "tools": [dict(tool) for tool in trace.tools],
        "split": split,
        "provenance": {
            key: value
            for key, value in trace.provenance.items()
            if key != "tool_result_provenance"
        },
    }


def answer_key_row(
    trace: SupervisedTraceRecord,
    selection: SelectedFact,
) -> dict[str, Any]:
    """Return one private S1.2 answer key."""

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
        "family": S1_2_QUESTION_FAMILY,
        "form": trace.metadata["question_form"],
        "task_context": trace.metadata["task_context"],
        "expected_status": answer.status,
        "expected_continuation_decision": answer.continuation_decision,
        "expected_requires_user_input": (
            answer.continuation_decision == "defer"
        ),
        "expected_tool_call_count": trace.tool_call_count,
        "expected_tool_actions": actions,
        "expected_final_explanation": final_message[
            "assistant_explanation"
        ],
        "expected_final_explanation_receipt": dict(
            final_message["explanation_receipt"]
        ),
        "expected_final_answer": final_message["content"],
        "source_kind": trace.provenance["source_kind"],
    }


def build_explanation_audit(
    train_rows: list[SupervisedTraceRecord],
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Build the S1.2 template and fixed-sample audit."""

    samples: dict[str, list[dict[str, Any]]] = {}
    for form in S1_2_FORMS:
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
            S1_2_EXPLANATION_TEMPLATES.items()
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


def _selection_variant_counts(
    selections: Iterable[SelectedFact],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return no-symbol context and invalid-kind counts."""

    selections = list(selections)
    no_symbol_contexts = Counter(
        item.task_context
        for item in selections
        if item.form == "no_symbol"
    )
    invalid_kinds = Counter(
        item.invalid_kind
        for item in selections
        if item.form == "invalid_input"
    )
    return (
        dict(sorted(no_symbol_contexts.items())),
        dict(sorted((str(key), value) for key, value in invalid_kinds.items())),
    )


def build_corpus(config_path: Path) -> dict[str, Any]:
    """Build and write the S1.2 corpus."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("The config must use the S1.2 build schema")
    if config.get("dataset_id") != DATASET_ID:
        raise ValueError("The S1.2 dataset ID changed")

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
        raise ValueError("S1.2 requires registry version 2")
    mappings = registry.get("gene_symbols_by_id")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The identifier registry has no mappings")
    runtime = S0IdentifierToolRuntime.from_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )

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

    components = build_components(mappings)
    registry_facts = assign_registry_facts(components, s0_split)
    seed = int(config["assignment_seed"])
    allocations = _require_object(config.get("allocations"), "allocations")
    selections = {
        split: select_form_facts(
            registry_facts[split],
            counts=_require_object(allocations.get(split), split),
            seed=seed,
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    registry_id = f"sha256:{registry_sha256}"
    traces = {
        split: [
            make_trace(
                selection,
                config=config,
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

    row_counts = {split: len(value) for split, value in traces.items()}
    form_counts = {}
    status_counts = {}
    tool_call_counts = {}
    no_symbol_context_counts = {}
    invalid_input_kind_counts = {}
    for split, split_rows in traces.items():
        forms, statuses, calls = _counts(split_rows)
        form_counts[split] = forms
        status_counts[split] = statuses
        tool_call_counts[split] = calls
        contexts, invalid_kinds = _selection_variant_counts(
            selections[split]
        )
        no_symbol_context_counts[split] = contexts
        invalid_input_kind_counts[split] = invalid_kinds
    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "assignment_seed": seed,
        "source_s0_split_manifest_sha256": sha256_file(s0_split_path),
        "row_counts": row_counts,
        "form_counts": form_counts,
        "status_counts": status_counts,
        "tool_call_counts": tool_call_counts,
        "no_symbol_context_counts": no_symbol_context_counts,
        "invalid_input_kind_counts": invalid_input_kind_counts,
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

    named_gene_id_count = sum(
        1 for symbols in mappings.values() if len(symbols) == 1
    )
    unnamed_gene_id_count = sum(
        1 for symbols in mappings.values() if not symbols
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "training_contract": TRAINING_CONTRACT,
        "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": evaluator_manifest["identifier_registry"],
        "registry_population": {
            "gene_id_count": len(mappings),
            "named_gene_id_count": named_gene_id_count,
            "unnamed_gene_id_count": unnamed_gene_id_count,
        },
        "s0_split_contract": {
            "dataset_id": S0_DATASET_ID,
            "build_config_path": str(s0_config_path.relative_to(REPO_ROOT)),
            "build_config_sha256": sha256_file(s0_config_path),
            "split_manifest_path": str(s0_split_path.relative_to(REPO_ROOT)),
            "split_manifest_sha256": sha256_file(s0_split_path),
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
        "no_symbol_context_counts": no_symbol_context_counts,
        "invalid_input_kind_counts": invalid_input_kind_counts,
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
            REPO_ROOT / "config/world_model_v2_s1_2_ensembl_to_symbol_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured S1.2 corpus."""

    manifest = build_corpus(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
