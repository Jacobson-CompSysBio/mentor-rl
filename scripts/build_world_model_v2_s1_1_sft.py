#!/usr/bin/env python3
"""Build the S1.1 symbol-to-Ensembl supervised trace corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
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
    TOOL_CONTRACT_VERSION_V2,
    S0IdentifierToolRuntime,
    identifier_tool_definitions,
)
from runtime.world_model_s1_shared import (  # noqa: E402
    S1_1_EXPLANATION_TEMPLATES,
    S1_1_FORMS,
    S1_1_QUESTION_FAMILY,
    S1_SOURCE_REFS,
    build_s1_1_trace,
)
from scripts.build_world_model_v2_s0_shared_sft import (  # noqa: E402
    DATASET_ID as S0_DATASET_ID,
    assign_not_found_facts,
    load_previous_symbol_pool,
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


CONFIG_SCHEMA_VERSION = "mentor-rl-world-model-s1-1-build-v1"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-1-manifest-v1"
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-1-split-manifest-v1"
)
EVALUATOR_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-1-evaluator-manifest-v1"
)
EXPLANATION_AUDIT_SCHEMA_VERSION = (
    "mentor-rl-world-model-explanation-audit-v1"
)
DATASET_ID = "world_model_v2_s1_1_symbol_to_ensembl_v1"
TRAINING_CONTRACT = "supervised_trace_v1"
EVALUATION_CONTRACT = "held_out_s0_components_v1"
QUESTION_SCHEMA_VERSION = "mentor-rl-world-model-s1-question-v1"


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
    """Assign resolved symbol facts with the fixed S0 components."""

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
            if fact["family"] == FAMILIES[0]
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


def select_form_facts(
    registry_facts: list[dict[str, Any]],
    absent_facts: list[dict[str, Any]],
    *,
    counts: Mapping[str, Any],
    seed: int,
    split: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Select the configured S1.1 forms for one split."""

    if set(counts) != set(S1_1_FORMS):
        raise ValueError("The S1.1 form count fields changed")
    selected: list[tuple[str, dict[str, Any]]] = []
    for form in S1_1_FORMS:
        count = _require_count(counts[form], f"{split}.{form}")
        source = absent_facts if form == "absent_symbol" else registry_facts
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
        selected.extend((form, fact) for fact in ordered[:count])
    return sorted(
        selected,
        key=lambda item: stable_sha256(
            {
                "assignment_seed": seed,
                "split": split,
                "form": item[0],
                "fact_id": item[1]["fact_id"],
            }
        ),
    )


def _record_id(form: str, fact: Mapping[str, Any], split: str) -> str:
    """Return one stable S1.1 record ID."""

    return "wm2_s1_1_" + stable_sha256(
        {
            "form": form,
            "fact_id": fact["fact_id"],
            "split": split,
            "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        }
    )[:24]


def render_question(
    *,
    config: Mapping[str, Any],
    split: str,
    form: str,
    fact: Mapping[str, Any],
) -> tuple[str, int]:
    """Render one deterministic S1.1 question."""

    prompt_form = "validation" if split == "validation" else split
    template_form = "direct" if form == "absent_symbol" else form
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
        raise ValueError("Each S1.1 prompt form needs text templates")
    template_index = int(
        stable_sha256(
            {
                "prompt_assignment_seed": config["prompt_assignment_seed"],
                "split": split,
                "form": form,
                "fact_id": fact["fact_id"],
            }
        )[:16],
        16,
    ) % len(templates)
    gene_symbol = str(fact["input"]["gene_symbol"])
    gene_id = fact["answer"].get("gene_id")
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
        gene_symbol=gene_symbol,
        gene_id=gene_id,
        registry_entry_json=registry_entry,
    )
    return question, template_index


def make_trace(
    form: str,
    fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    split: str,
    registry_id: str,
    runtime: S0IdentifierToolRuntime,
) -> SupervisedTraceRecord:
    """Build one selected S1.1 trace."""

    question, rendering_index = render_question(
        config=config,
        split=split,
        form=form,
        fact=fact,
    )
    gene_symbol = str(fact["input"]["gene_symbol"])
    gene_id = fact["answer"].get("gene_id")
    if form == "already_resolved":
        tool_result = None
        tool_provenance = None
    else:
        result = runtime.execute(
            "lookup_human_gene_symbol",
            {"gene_symbol": gene_symbol},
        )
        tool_result = result.payload
        tool_provenance = dict(result.provenance)
    trace_split = "train" if split == "train" else "val"
    provenance = {
        "fact_id": fact["fact_id"],
        "fact_group_id": fact["fact_group_id"],
        "fact_role": "train" if split == "train" else "unseen",
        "prompt_form_id": split,
        "question_form": form,
        "rendering_index": rendering_index,
        "source_kind": fact.get("source_kind", "ensembl_registry"),
        "s0_split_contract": "identifier_component_v7",
    }
    if tool_provenance is not None:
        provenance["tool_result_provenance"] = tool_provenance
    return build_s1_1_trace(
        record_id=_record_id(form, fact, split),
        question=question,
        form=form,
        gene_symbol=gene_symbol,
        gene_id=gene_id,
        tool_result=tool_result,
        split=trace_split,
        provenance=provenance,
        identifier_registry_id=registry_id,
    )


def public_question_row(
    trace: SupervisedTraceRecord,
    fact: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Return one answer-free S1 question."""

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
        "input": dict(fact["input"]),
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
    fact: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one private S1 answer key."""

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
        "fact_id": fact["fact_id"],
        "fact_group_id": fact["fact_group_id"],
        "family": S1_1_QUESTION_FAMILY,
        "form": trace.metadata["question_form"],
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
        "source_kind": fact.get("source_kind", "ensembl_registry"),
    }


def build_explanation_audit(
    train_rows: list[SupervisedTraceRecord],
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Build the S1.1 template and fixed-sample audit."""

    samples: dict[str, list[dict[str, Any]]] = {}
    for form in S1_1_FORMS:
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
            S1_1_EXPLANATION_TEMPLATES.items()
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
        dict(sorted(Counter(row.metadata["question_form"] for row in traces).items())),
        dict(sorted(Counter(row.final_answer.status for row in traces).items())),
        dict(sorted(Counter(str(row.tool_call_count) for row in traces).items())),
    )


def build_corpus(config_path: Path) -> dict[str, Any]:
    """Build and write the S1.1 corpus."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("The config must use the S1.1 build schema")
    if config.get("dataset_id") != DATASET_ID:
        raise ValueError("The S1.1 dataset ID changed")

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
        raise ValueError("S1.1 requires registry version 2")
    mappings = registry.get("gene_symbols_by_id")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The identifier registry has no mappings")
    runtime = S0IdentifierToolRuntime.from_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )

    s0_contract = _require_object(config.get("s0_split_contract"), "s0_split_contract")
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
    current_symbols = {
        str(symbol)
        for symbols in mappings.values()
        for symbol in symbols
    }
    missing_config = _require_object(
        s0_config.get("not_found_symbols"),
        "not_found_symbols",
    )
    hgnc_path = resolve_repo_path(missing_config.get("path"))
    hgnc_sha256 = str(missing_config.get("expected_sha256", ""))
    previous_symbols = load_previous_symbol_pool(
        hgnc_path,
        expected_sha256=hgnc_sha256,
        current_symbols=current_symbols,
    )
    absent_facts = assign_not_found_facts(previous_symbols, missing_config)

    seed = int(config["assignment_seed"])
    allocations = _require_object(config.get("allocations"), "allocations")
    selections = {
        split: select_form_facts(
            registry_facts[split],
            absent_facts[split],
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
                form,
                fact,
                config=config,
                split=split,
                registry_id=registry_id,
                runtime=runtime,
            )
            for form, fact in selections[split]
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
                    fact,
                    split="val" if split == "validation" else "test",
                )
                for row, (_, fact) in zip(
                    traces[split],
                    selections[split],
                    strict=True,
                )
            ],
        )
        write_jsonl(
            paths[f"{split}_answers"],
            [
                answer_key_row(row, fact)
                for row, (_, fact) in zip(
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
    for split, split_rows in traces.items():
        forms, statuses, calls = _counts(split_rows)
        form_counts[split] = forms
        status_counts[split] = statuses
        tool_call_counts[split] = calls
    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "assignment_seed": seed,
        "source_s0_split_manifest_sha256": sha256_file(s0_split_path),
        "row_counts": row_counts,
        "form_counts": form_counts,
        "status_counts": status_counts,
        "tool_call_counts": tool_call_counts,
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

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "training_contract": TRAINING_CONTRACT,
        "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": evaluator_manifest["identifier_registry"],
        "not_found_symbol_source": {
            "path": str(hgnc_path.relative_to(REPO_ROOT)),
            "sha256": hgnc_sha256,
            "eligible_previous_symbol_count": len(previous_symbols),
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
        "excluded_question_types": ["s1_3_ambiguous_symbol"],
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
            REPO_ROOT / "config/world_model_v2_s1_1_symbol_to_ensembl_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured S1.1 corpus."""

    manifest = build_corpus(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
