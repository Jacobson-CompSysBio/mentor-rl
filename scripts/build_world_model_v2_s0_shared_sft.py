#!/usr/bin/env python3
"""Build the version 7 S0 shared trace corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
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
    s0_shared_trace_prompt_contract,
)
from runtime.world_model_s0_shared import (  # noqa: E402
    S0_EXPLANATION_TEMPLATES,
    S0_SOURCE_REFS,
    build_s0_shared_trace,
)
from runtime.world_model_s0_tools import (  # noqa: E402
    REGISTRY_SCHEMA_VERSION_V2,
    TOOL_CONTRACT_VERSION_V2,
    S0IdentifierToolRuntime,
    identifier_tool_definitions,
)
from scripts.build_world_model_v2_s0_tool_sft import (  # noqa: E402
    FAMILIES,
    SYMBOL_FAMILIES,
    _tool_name,
    assign_evaluation_components,
    build_components,
    make_fact,
    protect_evaluator_dir,
    read_json,
    render_question,
    resolve_repo_path,
    sha256_file,
    stable_sha256,
    write_json,
    write_jsonl,
)


CONFIG_SCHEMA_VERSION = "mentor-rl-world-model-s0-shared-build-v7"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-shared-manifest-v7"
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-shared-split-manifest-v7"
)
EVALUATOR_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-shared-evaluator-manifest-v7"
)
EXPLANATION_AUDIT_SCHEMA_VERSION = (
    "mentor-rl-world-model-explanation-audit-v1"
)
DATASET_ID = "world_model_v2_s0_human_identifier_supervised_traces_v7"
TRAINING_CONTRACT = "supervised_trace_v1"
EVALUATION_CONTRACT = "unseen_component_supervised_trace_v1"
QUESTION_SCHEMA_VERSION = "mentor-rl-world-model-s0-question-v1"


def _require_object(value: Any, label: str) -> dict[str, Any]:
    """Return one required JSON object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be one object")
    return {str(key): item for key, item in value.items()}


def _require_nonnegative_int(value: Any, label: str) -> int:
    """Return one nonnegative integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def load_previous_symbol_pool(
    source_path: Path,
    *,
    expected_sha256: str,
    current_symbols: set[str],
) -> list[str]:
    """Load previous HGNC symbols that the registry does not contain."""

    if sha256_file(source_path) != expected_sha256:
        raise ValueError("The HGNC source SHA-256 value changed")
    previous_symbols: set[str] = set()
    with source_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames is None or "prev_symbol" not in reader.fieldnames:
            raise ValueError("The HGNC source has no prev_symbol column")
        for row in reader:
            raw = row.get("prev_symbol", "")
            if not isinstance(raw, str):
                raise ValueError("The HGNC previous symbol value is invalid")
            for symbol in raw.strip('"').split("|"):
                symbol = symbol.strip()
                if symbol and symbol not in current_symbols:
                    previous_symbols.add(symbol)
    if not previous_symbols:
        raise ValueError("The HGNC source has no absent previous symbols")
    return sorted(previous_symbols)


def _ordered_facts(
    facts: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    pool: str,
) -> list[dict[str, Any]]:
    """Return one stable fact order."""

    return sorted(
        (dict(fact) for fact in facts),
        key=lambda fact: stable_sha256(
            {
                "assignment_seed": seed,
                "pool": pool,
                "fact_id": fact["fact_id"],
            }
        ),
    )


def select_shared_train_facts(
    components: list[dict[str, Any]],
    assignments: Mapping[str, str],
    train_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select named, unnamed, and symbol facts for shared S0."""

    seed = int(train_config["assignment_seed"])
    excluded = frozenset(train_config.get("excluded_fact_ids", []))
    available = [
        fact
        for component in components
        if component["fact_group_id"] not in assignments
        for fact in component["facts"]
    ]
    available_ids = {str(fact["fact_id"]) for fact in available}
    missing_exclusions = excluded - available_ids
    if missing_exclusions:
        raise ValueError(
            "excluded_fact_ids contains unavailable train facts: "
            f"{sorted(missing_exclusions)}"
        )
    available = [
        fact for fact in available if fact["fact_id"] not in excluded
    ]

    ambiguous = [fact for fact in available if fact["family"] == FAMILIES[2]]
    resolved_symbols = [
        fact for fact in available if fact["family"] == FAMILIES[0]
    ]
    gene_ids = [fact for fact in available if fact["family"] == FAMILIES[1]]
    named_gene_ids = [
        fact for fact in gene_ids if fact["answer"]["gene_symbols"]
    ]
    unnamed_gene_ids = [
        fact for fact in gene_ids if not fact["answer"]["gene_symbols"]
    ]

    symbol_rows = _require_nonnegative_int(
        train_config.get("registry_symbol_rows"),
        "registry_symbol_rows",
    )
    named_rows = _require_nonnegative_int(
        train_config.get("named_gene_id_rows"),
        "named_gene_id_rows",
    )
    unnamed_rows = _require_nonnegative_int(
        train_config.get("unnamed_gene_id_rows"),
        "unnamed_gene_id_rows",
    )
    if len(ambiguous) > symbol_rows:
        raise ValueError("The symbol allocation cannot include all ambiguities")
    selected = list(ambiguous)
    selected.extend(
        _ordered_facts(
            resolved_symbols,
            seed=seed,
            pool="resolved_symbols",
        )[: symbol_rows - len(ambiguous)]
    )
    selected.extend(
        _ordered_facts(
            named_gene_ids,
            seed=seed,
            pool="named_gene_ids",
        )[:named_rows]
    )
    selected.extend(
        _ordered_facts(
            unnamed_gene_ids,
            seed=seed,
            pool="unnamed_gene_ids",
        )[:unnamed_rows]
    )
    expected = symbol_rows + named_rows + unnamed_rows
    if len(selected) != expected:
        raise ValueError("The registry cannot fill the shared train corpus")
    return _ordered_facts(selected, seed=seed, pool="train_output")


def make_not_found_symbol_facts(
    symbols: Iterable[str],
) -> list[dict[str, Any]]:
    """Create source facts for absent previous HGNC symbols."""

    facts = []
    for symbol in symbols:
        fact_group_id = "group_not_found_" + stable_sha256(
            {"gene_symbol": symbol, "status": "not_found"}
        )[:24]
        fact = make_fact(
            FAMILIES[0],
            fact_group_id,
            {"gene_symbol": symbol},
            {
                "status": "not_found",
                "gene_symbol": symbol,
                "candidate_gene_ids": [],
            },
        )
        fact["source_kind"] = "hgnc_previous_symbol"
        facts.append(fact)
    return facts


def assign_not_found_facts(
    symbols: list[str],
    config: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Assign disjoint absent symbols to train, validation, and test."""

    seed = int(config["assignment_seed"])
    ordered = sorted(
        symbols,
        key=lambda symbol: stable_sha256(
            {"assignment_seed": seed, "gene_symbol": symbol}
        ),
    )
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split_name in ("train", "validation", "test"):
        count = _require_nonnegative_int(
            config.get(f"{split_name}_rows"),
            f"not_found_symbols.{split_name}_rows",
        )
        selected = ordered[offset : offset + count]
        if len(selected) != count:
            raise ValueError("The HGNC source cannot fill the absent rows")
        result[split_name] = make_not_found_symbol_facts(selected)
        offset += count
    return result


def _shared_record_id(
    fact: Mapping[str, Any],
    *,
    prompt_form: str,
    split: str,
) -> str:
    """Return one stable shared trace record ID."""

    return "wm2_s0_shared_" + stable_sha256(
        {
            "fact_id": fact["fact_id"],
            "prompt_form": prompt_form,
            "split": split,
            "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        }
    )[:24]


def make_trace(
    fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    registry_id: str,
    prompt_form: str,
    split: str,
    fact_role: str,
    runtime: S0IdentifierToolRuntime,
) -> SupervisedTraceRecord:
    """Create one shared trace for a source fact."""

    question, rendering_index = render_question(
        fact,
        config=config,
        prompt_form=prompt_form,
    )
    tool_name = _tool_name(str(fact["family"]))
    result = runtime.execute(tool_name, fact["input"])
    return build_s0_shared_trace(
        record_id=_shared_record_id(
            fact,
            prompt_form=prompt_form,
            split=split,
        ),
        question=question,
        question_family=str(fact["family"]),
        tool_name=tool_name,
        arguments=fact["input"],
        tool_result=result.payload,
        split="train" if split == "train" else "val",
        identifier_registry_id=registry_id,
        provenance={
            "fact_id": fact["fact_id"],
            "fact_group_id": fact["fact_group_id"],
            "fact_role": fact_role,
            "prompt_form_id": prompt_form,
            "rendering_index": rendering_index,
            "source_kind": fact.get("source_kind", "ensembl_registry"),
            "tool_result_provenance": dict(result.provenance),
        },
    )


def public_question_row(
    trace: SupervisedTraceRecord,
    fact: Mapping[str, Any],
    *,
    split: str,
) -> dict[str, Any]:
    """Return one answer-free question from a private trace."""

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
            "fact_id": trace.provenance["fact_id"],
            "fact_group_id": trace.provenance["fact_group_id"],
            "fact_role": trace.provenance["fact_role"],
            "prompt_form_id": trace.provenance["prompt_form_id"],
            "rendering_index": trace.provenance["rendering_index"],
        },
    }


def answer_key_row(
    trace: SupervisedTraceRecord,
    fact: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one private answer key for a shared trace."""

    call_message = trace.messages[2]
    tool_message = trace.messages[3]
    final_message = trace.messages[4]
    answer = trace.final_answer
    call = call_message["tool_call"]["function"]
    return {
        "record_id": trace.record_id,
        "fact_id": fact["fact_id"],
        "fact_group_id": fact["fact_group_id"],
        "family": fact["family"],
        "expected_status": answer.status,
        "expected_continuation_decision": answer.continuation_decision,
        "expected_requires_user_input": (
            answer.continuation_decision == "defer"
        ),
        "expected_tool_call": {
            "name": call["name"],
            "arguments": dict(call["arguments"]),
        },
        "expected_tool_payload": dict(tool_message["content"]),
        "expected_assistant_thinking": call_message[
            "assistant_explanation"
        ],
        "expected_assistant_explanation": call_message[
            "assistant_explanation"
        ],
        "expected_call_explanation_receipt": dict(
            call_message["explanation_receipt"]
        ),
        "expected_final_explanation": final_message[
            "assistant_explanation"
        ],
        "expected_final_explanation_receipt": dict(
            final_message["explanation_receipt"]
        ),
        "expected_final_answer": final_message["content"],
        "source_kind": fact.get("source_kind", "ensembl_registry"),
    }


def _family_counts(
    traces: Iterable[SupervisedTraceRecord],
) -> dict[str, int]:
    """Count each question family."""

    counts = {family: 0 for family in FAMILIES}
    for trace in traces:
        counts[str(trace.metadata["question_family"])] += 1
    return counts


def _status_counts(
    traces: Iterable[SupervisedTraceRecord],
) -> dict[str, int]:
    """Count each final answer status."""

    counts = {"resolved": 0, "ambiguous": 0, "not_found": 0}
    for trace in traces:
        counts[trace.final_answer.status] += 1
    return counts


def _audit_sample(
    rows: list[SupervisedTraceRecord],
    *,
    sample_size: int,
    seed: int,
    prompt_form: str,
) -> list[dict[str, Any]]:
    """Select one deterministic explanation review sample."""

    if len(rows) < sample_size:
        raise ValueError("The explanation sample exceeds its row set")
    selected = sorted(
        rows,
        key=lambda row: stable_json_sha256(
            {
                "sample_seed": seed,
                "prompt_form": prompt_form,
                "record_id": row.record_id,
            }
        ),
    )[:sample_size]
    return [
        {
            "record_id": row.record_id,
            "question_family": row.metadata["question_family"],
            "actions": [
                {
                    "action_index": action_index,
                    "assistant_explanation": message[
                        "assistant_explanation"
                    ],
                    "explanation_receipt": dict(
                        message["explanation_receipt"]
                    ),
                }
                for action_index, message in enumerate(
                    (row.messages[2], row.messages[-1])
                )
            ],
        }
        for row in selected
    ]


def build_explanation_audit(
    train_rows: list[SupervisedTraceRecord],
    validation_rows: list[SupervisedTraceRecord],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic explanation audit artifact."""

    sample_size = int(config["sample_size_per_prompt_form"])
    seed = int(config["sample_seed"])
    templates = {
        action_path: [
            {
                "explanation_template_id": template.template_id,
                "explanation_template": template.text,
                "explanation_template_sha256": text_sha256(template.text),
                "explanation_rule_ids": list(template.rule_ids),
                "explanation_source_refs": list(S0_SOURCE_REFS),
            }
            for template in action_templates
        ]
        for action_path, action_templates in sorted(
            S0_EXPLANATION_TEMPLATES.items()
        )
    }
    samples = {
        "train": _audit_sample(
            train_rows,
            sample_size=sample_size,
            seed=seed,
            prompt_form="train",
        ),
        "validation": _audit_sample(
            validation_rows,
            sample_size=sample_size,
            seed=seed,
            prompt_form="validation",
        ),
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
        "template_count": sum(len(items) for items in templates.values()),
        "templates": templates,
        "sample_contract": {
            "sample_seed": seed,
            "sample_size_per_prompt_form": sample_size,
            "selection": "stable_hash_of_prompt_form_and_record_id",
        },
        "samples": samples,
    }


def build_corpus(config_path: Path) -> dict[str, Any]:
    """Build and write the version 7 S0 shared corpus."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("The config must use the S0 shared build schema")
    if config.get("dataset_id") != DATASET_ID:
        raise ValueError("The S0 shared dataset ID changed")

    registry_config = _require_object(
        config.get("identifier_registry"),
        "identifier_registry",
    )
    registry_path = resolve_repo_path(registry_config.get("path"))
    registry_sha256 = str(registry_config.get("expected_sha256", ""))
    if sha256_file(registry_path) != registry_sha256:
        raise ValueError("The identifier registry SHA-256 value changed")
    registry = read_json(registry_path)
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION_V2:
        raise ValueError("The shared corpus requires registry version 2")
    mappings = registry.get("gene_symbols_by_id")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The identifier registry has no mappings")
    runtime = S0IdentifierToolRuntime.from_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )

    components = build_components(mappings)
    assignments = assign_evaluation_components(components, config["split"])
    train_facts = select_shared_train_facts(
        components,
        assignments,
        config["train"],
    )
    validation_facts = sorted(
        (
            fact
            for component in components
            if assignments.get(component["fact_group_id"]) == "validation"
            for fact in component["facts"]
        ),
        key=lambda fact: str(fact["fact_id"]),
    )
    test_facts = sorted(
        (
            fact
            for component in components
            if assignments.get(component["fact_group_id"]) == "test"
            for fact in component["facts"]
        ),
        key=lambda fact: str(fact["fact_id"]),
    )

    current_symbols = {
        str(symbol)
        for symbols in mappings.values()
        for symbol in symbols
    }
    missing_config = _require_object(
        config.get("not_found_symbols"),
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
    train_facts.extend(absent_facts["train"])
    validation_facts.extend(absent_facts["validation"])
    test_facts.extend(absent_facts["test"])
    train_facts = _ordered_facts(
        train_facts,
        seed=int(config["train"]["assignment_seed"]),
        pool="shared_train_with_not_found",
    )
    validation_facts.sort(key=lambda fact: str(fact["fact_id"]))
    test_facts.sort(key=lambda fact: str(fact["fact_id"]))

    registry_id = f"sha256:{registry_sha256}"
    train_traces = [
        make_trace(
            fact,
            config=config,
            registry_id=registry_id,
            prompt_form="train",
            split="train",
            fact_role="train",
            runtime=runtime,
        )
        for fact in train_facts
    ]
    validation_traces = [
        make_trace(
            fact,
            config=config,
            registry_id=registry_id,
            prompt_form="validation",
            split="val",
            fact_role="unseen",
            runtime=runtime,
        )
        for fact in validation_facts
    ]
    test_traces = [
        make_trace(
            fact,
            config=config,
            registry_id=registry_id,
            prompt_form="test",
            split="test",
            fact_role="unseen",
            runtime=runtime,
        )
        for fact in test_facts
    ]

    validation_questions = [
        public_question_row(trace, fact, split="val")
        for trace, fact in zip(
            validation_traces,
            validation_facts,
            strict=True,
        )
    ]
    test_questions = [
        public_question_row(trace, fact, split="test")
        for trace, fact in zip(test_traces, test_facts, strict=True)
    ]
    validation_answers = [
        answer_key_row(trace, fact)
        for trace, fact in zip(
            validation_traces,
            validation_facts,
            strict=True,
        )
    ]
    test_answers = [
        answer_key_row(trace, fact)
        for trace, fact in zip(test_traces, test_facts, strict=True)
    ]

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
    write_jsonl(paths["train"], [row.to_dict() for row in train_traces])
    write_jsonl(
        paths["validation"],
        [row.to_dict() for row in validation_traces],
    )
    write_jsonl(paths["validation_questions"], validation_questions)
    write_jsonl(paths["validation_answers"], validation_answers)
    write_jsonl(paths["test_questions"], test_questions)
    write_jsonl(paths["test_answers"], test_answers)

    audit_config = _require_object(
        config.get("explanation_audit"),
        "explanation_audit",
    )
    explanation_audit = build_explanation_audit(
        train_traces,
        validation_traces,
        config=audit_config,
    )
    write_json(paths["explanation_audit"], explanation_audit)

    row_counts = {
        "train": len(train_traces),
        "validation": len(validation_traces),
        "test": len(test_traces),
    }
    train_groups = {
        str(trace.provenance["fact_group_id"]) for trace in train_traces
    }
    validation_groups = {
        str(trace.provenance["fact_group_id"])
        for trace in validation_traces
    }
    test_groups = {
        str(trace.provenance["fact_group_id"]) for trace in test_traces
    }
    if train_groups & validation_groups or train_groups & test_groups:
        raise ValueError("A train component occurs in an evaluation split")
    if validation_groups & test_groups:
        raise ValueError("An evaluation component occurs in both panels")

    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "assignment_seed": config["split"]["assignment_seed"],
        "fact_role": "unseen",
        "row_counts": row_counts,
        "family_counts": {
            "train": _family_counts(train_traces),
            "validation": _family_counts(validation_traces),
            "test": _family_counts(test_traces),
        },
        "status_counts": {
            "train": _status_counts(train_traces),
            "validation": _status_counts(validation_traces),
            "test": _status_counts(test_traces),
        },
        "train_fact_group_ids_sha256": stable_sha256(sorted(train_groups)),
        "validation_fact_group_ids": sorted(validation_groups),
        "test_fact_group_ids": sorted(test_groups),
    }
    write_json(paths["split_manifest"], split_manifest)

    evaluator_manifest = {
        "schema_version": EVALUATOR_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "evaluation_contract": EVALUATION_CONTRACT,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "answer_schema_version": train_traces[0].metadata[
            "answer_schema_version"
        ],
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
        },
        "validation": {
            "questions_path": str(
                paths["validation_questions"].relative_to(REPO_ROOT)
            ),
            "questions_sha256": sha256_file(paths["validation_questions"]),
            "answer_key_path": str(
                paths["validation_answers"].relative_to(REPO_ROOT)
            ),
            "answer_key_sha256": sha256_file(paths["validation_answers"]),
            "row_count": len(validation_traces),
        },
        "test": {
            "questions_path": str(
                paths["test_questions"].relative_to(REPO_ROOT)
            ),
            "questions_sha256": sha256_file(paths["test_questions"]),
            "answer_key_path": str(
                paths["test_answers"].relative_to(REPO_ROOT)
            ),
            "answer_key_sha256": sha256_file(paths["test_answers"]),
            "row_count": len(test_traces),
            "test_panel_id": sha256_file(paths["test_questions"]),
        },
    }
    write_json(paths["evaluator_manifest"], evaluator_manifest)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "record_schema_version": SUPERVISED_TRACE_SCHEMA_VERSION,
        "training_contract": TRAINING_CONTRACT,
        "loss_contract": SUPERVISED_EXPLANATION_LOSS_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
            "source": registry.get("source"),
            "counts": registry.get("counts"),
            "normalization": registry.get("normalization"),
        },
        "not_found_symbol_source": {
            "path": str(hgnc_path.relative_to(REPO_ROOT)),
            "sha256": hgnc_sha256,
            "eligible_previous_symbol_count": len(previous_symbols),
        },
        "system_prompt_contract": s0_shared_trace_prompt_contract().to_dict(),
        "tool_contract_version": TOOL_CONTRACT_VERSION_V2,
        "tool_definitions": identifier_tool_definitions(
            contract_version=TOOL_CONTRACT_VERSION_V2
        ),
        "config_sha256": sha256_file(config_path),
        "row_counts": row_counts,
        "family_counts": split_manifest["family_counts"],
        "status_counts": split_manifest["status_counts"],
        "train_population": {
            "eligible_train_rows": len(train_traces),
            "registry_symbol_rows": config["train"][
                "registry_symbol_rows"
            ],
            "not_found_symbol_rows": missing_config["train_rows"],
            "named_gene_id_rows": config["train"]["named_gene_id_rows"],
            "unnamed_gene_id_rows": config["train"][
                "unnamed_gene_id_rows"
            ],
            "excluded_fact_ids": list(
                config["train"].get("excluded_fact_ids", [])
            ),
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
            REPO_ROOT / "config/world_model_v2_s0_shared_traces_v7.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured shared corpus."""

    manifest = build_corpus(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
