#!/usr/bin/env python3
"""Build the canonical S0 tool trajectory corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_prompts import (  # noqa: E402
    s0_tool_trajectory_prompt_contract,
)
from runtime.world_model_s0_tools import (  # noqa: E402
    ENSEMBL_TOOL_NAME,
    SYMBOL_TOOL_NAME,
    TOOL_CONTRACT_VERSION,
    S0IdentifierToolRuntime,
    identifier_tool_definitions,
)
from runtime.world_model_schemas import (  # noqa: E402
    IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
    TOOL_TRAJECTORY_FORMAT,
    IdentifierToolSFTMetadata,
    IdentifierToolSFTRecord,
)


CONFIG_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-tool-build-v6"
)
MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-manifest-v6"
)
SPLIT_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-split-manifest-v6"
)
EVALUATOR_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-evaluator-manifest-v6"
)
TRAINING_CONTRACT = "tool_trajectory_v1"
EVALUATION_CONTRACT = (
    "unseen_component_tool_trajectory_v1"
)
FAMILIES = (
    "human_symbol_to_ensembl",
    "human_ensembl_to_symbol",
    "human_ambiguous_symbol",
)
SYMBOL_FAMILIES = frozenset((FAMILIES[0], FAMILIES[2]))
PROMPT_FORM_INDEX = {"train": 0, "validation": 100, "test": 200}
SUPERVISED_TRAJECTORY_FIELDS = frozenset(
    {
        "assistant_tool_call",
        "assistant_thinking",
        "tool_result",
        "assistant_final",
    }
)


def canonical_json(value: Any) -> str:
    """Return compact canonical JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def render_question(
    fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    prompt_form: str,
) -> tuple[str, int]:
    """Render one deterministic question form."""

    family = str(fact["family"])
    configured = config["prompt_forms"][prompt_form][family]

    if isinstance(configured, str):
        templates = (configured,)
    elif (
        isinstance(configured, list)
        and configured
        and all(
            isinstance(template, str)
            and template
            for template in configured
        )
    ):
        templates = tuple(configured)
    else:
        raise TypeError(
            "Each prompt form must be a string or a non-empty string list"
        )

    template_index = int(
        stable_sha256(
            {
                "prompt_assignment_seed": config.get(
                    "prompt_assignment_seed",
                    0,
                ),
                "prompt_form": prompt_form,
                "fact_id": fact["fact_id"],
            }
        )[:16],
        16,
    ) % len(templates)

    question = templates[template_index].format(
        input_json=canonical_json(fact["input"]),
        **dict(fact["input"]),
    )
    rendering_index = (
        PROMPT_FORM_INDEX[prompt_form] + template_index
    )
    return question, rendering_index


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write stable JSON Lines rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(canonical_json(row) + "\n")


def resolve_repo_path(value: Any) -> Path:
    """Resolve one path below the repository."""

    if not isinstance(value, str) or not value:
        raise ValueError("A repository path must be a nonempty string")
    declared = Path(value)
    path = declared if declared.is_absolute() else REPO_ROOT / declared
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("A corpus path escapes the repository") from error
    return path


def make_fact(
    family: str,
    fact_group_id: str,
    inputs: dict[str, str],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Create one stable source fact."""

    fact_id = "fact_" + stable_sha256(
        {
            "family": family,
            "input": inputs,
            "answer": answer,
        }
    )[:24]
    return {
        "fact_id": fact_id,
        "fact_group_id": fact_group_id,
        "family": family,
        "input": inputs,
        "answer": answer,
    }


def build_components(
    mappings: Mapping[str, list[str]],
) -> list[dict[str, Any]]:
    """Build all connected identifier components and facts."""

    symbols_by_id = {
        gene_id: sorted(set(symbols))
        for gene_id, symbols in mappings.items()
    }
    ids_by_symbol: dict[str, set[str]] = {}
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    for gene_id, symbols in symbols_by_id.items():
        gene_node = f"id:{gene_id}"
        find(gene_node)
        for symbol in symbols:
            union(gene_node, f"symbol:{symbol}")
            ids_by_symbol.setdefault(symbol, set()).add(gene_id)

    groups: dict[str, dict[str, set[str]]] = {}
    for gene_id, symbols in symbols_by_id.items():
        group = groups.setdefault(
            find(f"id:{gene_id}"),
            {"gene_ids": set(), "gene_symbols": set()},
        )
        group["gene_ids"].add(gene_id)
        group["gene_symbols"].update(symbols)

    family_order = {
        family: index for index, family in enumerate(FAMILIES)
    }
    components = []
    for group in groups.values():
        gene_ids = sorted(group["gene_ids"])
        gene_symbols = sorted(group["gene_symbols"])
        fact_group_id = "group_" + stable_sha256(
            {"gene_ids": gene_ids, "gene_symbols": gene_symbols}
        )[:24]
        facts = []
        is_ambiguous = False
        for symbol in gene_symbols:
            candidate_gene_ids = sorted(ids_by_symbol[symbol])
            if len(candidate_gene_ids) == 1:
                facts.append(
                    make_fact(
                        FAMILIES[0],
                        fact_group_id,
                        {"gene_symbol": symbol},
                        {
                            "status": "resolved",
                            "gene_id": candidate_gene_ids[0],
                            "gene_symbol": symbol,
                        },
                    )
                )
            else:
                is_ambiguous = True
                facts.append(
                    make_fact(
                        FAMILIES[2],
                        fact_group_id,
                        {"gene_symbol": symbol},
                        {
                            "status": "ambiguous",
                            "gene_symbol": symbol,
                            "candidate_gene_ids": candidate_gene_ids,
                            "action": "defer",
                        },
                    )
                )
        for gene_id in gene_ids:
            facts.append(
                make_fact(
                    FAMILIES[1],
                    fact_group_id,
                    {"gene_id": gene_id},
                    {
                        "status": "resolved",
                        "gene_id": gene_id,
                        "gene_symbols": symbols_by_id[gene_id],
                    },
                )
            )
        facts.sort(
            key=lambda fact: (
                family_order[str(fact["family"])],
                str(fact["fact_id"]),
            )
        )
        components.append(
            {
                "fact_group_id": fact_group_id,
                "gene_ids": gene_ids,
                "gene_symbols": gene_symbols,
                "is_ambiguous": is_ambiguous,
                "facts": facts,
            }
        )
    components.sort(key=lambda item: str(item["fact_group_id"]))
    return components


def assign_evaluation_components(
    components: list[dict[str, Any]],
    split_config: Mapping[str, Any],
) -> dict[str, str]:
    """Assign disjoint components to validation and test."""

    pools = {
        "unambiguous": [
            item for item in components if not item["is_ambiguous"]
        ],
        "ambiguous": [
            item for item in components if item["is_ambiguous"]
        ],
    }
    seed = int(split_config["assignment_seed"])
    assignments = {}
    for split_name in ("validation", "test"):
        for component_type in ("unambiguous", "ambiguous"):
            count_key = f"{component_type}_component_count"
            required_count = int(
                split_config[split_name][count_key]
            )
            ordered = sorted(
                pools[component_type],
                key=lambda item: stable_sha256(
                    {
                        "assignment_seed": seed,
                        "split": split_name,
                        "fact_group_id": item["fact_group_id"],
                    }
                ),
            )
            if len(ordered) < required_count:
                raise ValueError(
                    f"The {component_type} pool cannot fill "
                    f"the {split_name} panel"
                )
            selected = ordered[:required_count]
            pools[component_type] = ordered[required_count:]
            for component in selected:
                assignments[str(component["fact_group_id"])] = split_name
    return assignments


def _tool_name(family: str) -> str:
    """Return the required tool for one question family."""

    if family in SYMBOL_FAMILIES:
        return SYMBOL_TOOL_NAME
    if family == FAMILIES[1]:
        return ENSEMBL_TOOL_NAME
    raise ValueError(f"Unknown S0 family: {family!r}")


def trajectory_fields(
    fact: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the supervised fields for one tool trajectory."""

    family = str(fact["family"])
    inputs = fact["input"]
    answer = fact["answer"]

    if not isinstance(inputs, Mapping):
        raise TypeError("The fact input must be one object")
    if not isinstance(answer, Mapping):
        raise TypeError("The fact answer must be one object")

    if family in SYMBOL_FAMILIES:
        gene_symbol = str(inputs["gene_symbol"])
        assistant_thinking = (
            "The next step requires a human Ensembl gene ID. "
            f"I need to resolve {gene_symbol} against the Ensembl "
            "release 116 registry."
        )

        if family == FAMILIES[0]:
            if answer.get("status") != "resolved":
                raise ValueError(
                    "A symbol mapping must have resolved status"
                )

            gene_id = str(answer["gene_id"])
            assistant_final = (
                f"The Ensembl gene ID for {gene_symbol} is "
                f"{gene_id}."
            )
        else:
            candidate_gene_ids = answer.get("candidate_gene_ids")
            if (
                answer.get("status") != "ambiguous"
                or answer.get("action") != "defer"
                or not isinstance(candidate_gene_ids, list)
                or len(candidate_gene_ids) < 2
            ):
                raise ValueError(
                    "An ambiguous mapping must contain candidates "
                    "and the defer action"
                )
            candidate_text = ", ".join(candidate_gene_ids)
            assistant_final = (
                f"The gene symbol {gene_symbol} maps to multiple "
                f"Ensembl gene IDs, {candidate_text}, so this "
                "analysis must defer the mapping."
            )
    elif family == FAMILIES[1]:
        gene_id = str(inputs["gene_id"])
        gene_symbols = answer.get("gene_symbols")
        if (
            answer.get("status") != "resolved"
            or not isinstance(gene_symbols, list)
            or not gene_symbols
        ):
            raise ValueError(
                "A gene ID mapping must contain gene symbols"
            )

        assistant_thinking = (
            "The next step requires a human gene symbol. "
            f"I need to resolve {gene_id} against the Ensembl "
            "release 116 registry."
        )

        if len(gene_symbols) == 1:
            assistant_final = (
                f"The human gene symbol for Ensembl gene ID "
                f"{gene_id} is {gene_symbols[0]}."
            )
        else:
            symbol_text = ", ".join(gene_symbols)
            assistant_final = (
                f"The human gene symbols for Ensembl gene ID "
                f"{gene_id} are {symbol_text}."
            )

    else:
        raise ValueError(f"Unknown S0 family: {family!r}")

    return {
        "assistant_thinking": assistant_thinking,
        "tool_result": dict(answer),
        "assistant_final": assistant_final,
    }


def _ordered_facts(
    facts: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    pool: str,
) -> list[dict[str, Any]]:
    """Return one deterministic fact order."""

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


def select_train_facts(
    components: list[dict[str, Any]],
    assignments: Mapping[str, str],
    train_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select balanced train facts from non-evaluation components."""

    symbol_rows = int(train_config["symbol_rows"])
    gene_id_rows = int(train_config["gene_id_rows"])
    seed = int(train_config["assignment_seed"])
    excluded_fact_ids = train_config.get("excluded_fact_ids", [])
    if (
        not isinstance(excluded_fact_ids, list)
        or any(
            not isinstance(fact_id, str) or not fact_id
            for fact_id in excluded_fact_ids
        )
        or len(set(excluded_fact_ids)) != len(excluded_fact_ids)
    ):
        raise ValueError(
            "excluded_fact_ids must contain unique fact ID strings"
        )
    excluded = frozenset(excluded_fact_ids)
    available = [
        fact
        for component in components
        if component["fact_group_id"] not in assignments
        for fact in component["facts"]
    ]
    missing_exclusions = excluded - {
        str(fact["fact_id"])
        for fact in available
    }
    if missing_exclusions:
        raise ValueError(
            "excluded_fact_ids contains unavailable train facts: "
            f"{sorted(missing_exclusions)}"
        )
    available = [
        fact
        for fact in available
        if fact["fact_id"] not in excluded
    ]
    ambiguous = [fact for fact in available if fact["family"] == FAMILIES[2]]
    resolved_symbols = [
        fact for fact in available if fact["family"] == FAMILIES[0]
    ]
    gene_ids = [fact for fact in available if fact["family"] == FAMILIES[1]]
    if len(ambiguous) > symbol_rows:
        raise ValueError(
            "The symbol allocation cannot include all ambiguous train symbols"
        )
    symbol_fill = symbol_rows - len(ambiguous)
    ordered_symbols = _ordered_facts(
        resolved_symbols,
        seed=seed,
        pool="resolved_symbols",
    )
    ordered_gene_ids = _ordered_facts(
        gene_ids,
        seed=seed,
        pool="gene_ids",
    )
    if len(ordered_symbols) < symbol_fill or len(ordered_gene_ids) < gene_id_rows:
        raise ValueError("The registry cannot fill the requested train corpus")
    selected = ambiguous + ordered_symbols[:symbol_fill]
    selected.extend(ordered_gene_ids[:gene_id_rows])
    selected.sort(
        key=lambda fact: stable_sha256(
            {
                "assignment_seed": seed,
                "pool": "train_output",
                "fact_id": fact["fact_id"],
            }
        )
    )
    return selected


def make_record(
    fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    registry_id: str,
    prompt_form: str,
) -> dict[str, Any]:
    """Create one supervised S0 tool-call record."""

    split = {"train": "train", "validation": "val"}[prompt_form]
    question, rendering_index = render_question(
        fact,
        config=config,
        prompt_form=prompt_form,
    )
    call = {
        "type": "function",
        "function": {
            "name": _tool_name(str(fact["family"])),
            "arguments": dict(fact["input"]),
        },
    }
    prompt = s0_tool_trajectory_prompt_contract()
    record_identity = {
        "fact_id": fact["fact_id"],
        "prompt_form": prompt_form,
        "split": split,
    }
    record_identity["record_schema_version"] = (
        IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION
    )
    record_id = "wm2_s0_tool_" + stable_sha256(record_identity)[:24]
    trajectory = trajectory_fields(fact)
    record = IdentifierToolSFTRecord(
        record_id=record_id,
        metadata=IdentifierToolSFTMetadata(
            book_mode="tool_call",
            step="S0",
            question_family=str(fact["family"]),
            species_taxon_id="NCBITaxon:9606",
            ensembl_release="Ensembl_116",
            identifier_registry_id=registry_id,
            system_prompt_sha256=prompt.system_prompt_sha256,
            tool_contract_version=TOOL_CONTRACT_VERSION,
            schema_version=IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
            target_format=TOOL_TRAJECTORY_FORMAT,
        ),
        system=prompt.system_prompt,
        question=question,
        input=dict(fact["input"]),
        tools=tuple(identifier_tool_definitions()),
        assistant_tool_call=call,
        split=split,
        provenance={
            "fact_id": fact["fact_id"],
            "fact_group_id": fact["fact_group_id"],
            "fact_role": "train" if split == "train" else "unseen",
            "prompt_form_id": prompt_form,
            "rendering_index": rendering_index,
        },
        **trajectory,
    )
    return record.to_dict()


def make_test_question(
    fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    registry_id: str,
    fact_role: str = "unseen",
) -> dict[str, Any]:
    """Create one answer-free test question."""

    prompt = s0_tool_trajectory_prompt_contract()
    question, rendering_index = render_question(
        fact,
        config=config,
        prompt_form="test",
    )
    record_id = "wm2_s0_tool_" + stable_sha256(
        {
            "fact_id": fact["fact_id"],
            "prompt_form": "test",
            "split": "test",
        }
    )[:24]
    return {
        "record_id": record_id,
        "metadata": {
            "schema_version": IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
            "book_mode": "tool_call",
            "step": "S0",
            "question_family": fact["family"],
            "species_taxon_id": "NCBITaxon:9606",
            "ensembl_release": "Ensembl_116",
            "identifier_registry_id": registry_id,
            "system_prompt_sha256": prompt.system_prompt_sha256,
            "tool_contract_version": TOOL_CONTRACT_VERSION,
            "target_format": TOOL_TRAJECTORY_FORMAT,
        },
        "system": prompt.system_prompt,
        "question": question,
        "input": dict(fact["input"]),
        "tools": identifier_tool_definitions(),
        "split": "test",
        "provenance": {
            "fact_id": fact["fact_id"],
            "fact_group_id": fact["fact_group_id"],
            "fact_role": fact_role,
            "prompt_form_id": "test",
            "rendering_index": rendering_index,
        },
    }


def answer_key_row(
    question: Mapping[str, Any],
    fact: Mapping[str, Any],
    runtime: S0IdentifierToolRuntime,
    *,
    include_final_answer: bool = False,
) -> dict[str, Any]:
    """Create one private tool result row."""

    tool_name = _tool_name(str(fact["family"]))
    result = runtime.execute(tool_name, fact["input"])
    if result.payload != fact["answer"]:
        raise ValueError("The runtime result differs from the source fact")
    answer_key = {
        "record_id": question["record_id"],
        "fact_id": fact["fact_id"],
        "fact_group_id": fact["fact_group_id"],
        "family": fact["family"],
        "expected_tool_call": {
            "name": tool_name,
            "arguments": dict(fact["input"]),
        },
        "expected_tool_payload": dict(result.payload),
    }
    if include_final_answer:
        answer_key["expected_final_answer"] = trajectory_fields(fact)[
            "assistant_final"
        ]
    return answer_key


def public_question_row(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove all supervised assistant fields from one record."""

    return {
        key: value
        for key, value in record.items()
        if key not in SUPERVISED_TRAJECTORY_FIELDS
    }


def family_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count records for each S0 family."""

    counts = {family: 0 for family in FAMILIES}
    for row in rows:
        counts[str(row["metadata"]["question_family"])] += 1
    return counts


def protect_evaluator_dir(path: Path) -> None:
    """Restrict access to private evaluator files."""

    path.chmod(0o700)
    for child in path.rglob("*"):
        child.chmod(0o700 if child.is_dir() else 0o600)


def build_corpus(config_path: Path) -> dict[str, Any]:
    """Build and write one versioned S0 tool corpus."""

    config_path = config_path.resolve()
    config = read_json(config_path)
    config_schema_version = config.get("schema_version")
    if config_schema_version != CONFIG_SCHEMA_VERSION:
        raise ValueError("The config must use the v6 S0 build schema")
    registry_config = config.get("identifier_registry")
    if not isinstance(registry_config, Mapping):
        raise TypeError("identifier_registry must be one object")
    registry_path = resolve_repo_path(registry_config.get("path"))
    registry_sha256 = str(registry_config.get("expected_sha256", ""))
    if sha256_file(registry_path) != registry_sha256:
        raise ValueError("The identifier registry SHA-256 value changed")
    registry = read_json(registry_path)
    mappings = registry.get("gene_symbols_by_id")
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The identifier registry has no mappings")
    runtime = S0IdentifierToolRuntime.from_registry(
        registry_path,
        expected_sha256=registry_sha256,
    )
    components = build_components(mappings)
    assignments = assign_evaluation_components(components, config["split"])
    selected_train_facts = select_train_facts(
        components,
        assignments,
        config["train"],
    )
    registry_id = f"sha256:{registry_sha256}"
    train_rows = [
        make_record(
            fact,
            config=config,
            registry_id=registry_id,
            prompt_form="train",
        )
        for fact in selected_train_facts
    ]
    validation_facts = sorted(
        (
            fact
            for component in components
            if assignments.get(component["fact_group_id"]) == "validation"
            for fact in component["facts"]
        ),
        key=lambda fact: fact["fact_id"],
    )
    test_facts = sorted(
        (
            fact
            for component in components
            if assignments.get(component["fact_group_id"]) == "test"
            for fact in component["facts"]
        ),
        key=lambda fact: fact["fact_id"],
    )
    validation_rows = [
        make_record(
            fact,
            config=config,
            registry_id=registry_id,
            prompt_form="validation",
        )
        for fact in validation_facts
    ]
    validation_questions = [
        public_question_row(row)
        for row in validation_rows
    ]
    test_questions = [
        make_test_question(
            fact,
            config=config,
            registry_id=registry_id,
        )
        for fact in test_facts
    ]
    validation_answers = [
        answer_key_row(
            question,
            fact,
            runtime,
            include_final_answer=True,
        )
        for question, fact in zip(validation_questions, validation_facts, strict=True)
    ]
    test_answers = [
        answer_key_row(
            question,
            fact,
            runtime,
            include_final_answer=True,
        )
        for question, fact in zip(test_questions, test_facts, strict=True)
    ]
    outputs = config["outputs"]
    corpus_dir = resolve_repo_path(outputs["corpus_dir"])
    evaluator_dir = resolve_repo_path(outputs["evaluator_dir"])
    paths = {
        "train": corpus_dir / "train.jsonl",
        "validation": corpus_dir / "val.jsonl",
        "split_manifest": corpus_dir / "split_manifest.json",
        "manifest": corpus_dir / "manifest.json",
        "validation_questions": evaluator_dir / "validation_questions.jsonl",
        "validation_answers": evaluator_dir / "validation_answer_key.jsonl",
        "test_questions": evaluator_dir / "test_questions.jsonl",
        "test_answers": evaluator_dir / "test_answer_key.jsonl",
        "evaluator_manifest": evaluator_dir / "manifest.json",
    }
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["validation"], validation_rows)
    write_jsonl(paths["validation_questions"], validation_questions)
    write_jsonl(paths["validation_answers"], validation_answers)
    write_jsonl(paths["test_questions"], test_questions)
    write_jsonl(paths["test_answers"], test_answers)
    row_counts = {
        "train": len(train_rows),
        "validation": len(validation_rows),
        "test": len(test_questions),
    }
    train_groups = {
        row["provenance"]["fact_group_id"] for row in train_rows
    }
    validation_groups = {
        row["provenance"]["fact_group_id"] for row in validation_rows
    }
    test_groups = {
        row["provenance"]["fact_group_id"] for row in test_questions
    }
    if train_groups & validation_groups or train_groups & test_groups:
        raise ValueError("A train component occurs in an evaluation split")
    if validation_groups & test_groups:
        raise ValueError("An evaluation component occurs in both panels")
    split_manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "assignment_seed": config["split"]["assignment_seed"],
        "fact_role": "unseen",
        "row_counts": row_counts,
        "family_counts": {
            "train": family_counts(train_rows),
            "validation": family_counts(validation_rows),
            "test": family_counts(test_questions),
        },
        "train_fact_group_ids_sha256": stable_sha256(sorted(train_groups)),
        "validation_fact_group_ids": sorted(validation_groups),
        "test_fact_group_ids": sorted(test_groups),
    }
    write_json(paths["split_manifest"], split_manifest)
    evaluator_manifest = {
        "schema_version": EVALUATOR_MANIFEST_SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
        },
        "validation": {
            "questions_path": str(paths["validation_questions"].relative_to(REPO_ROOT)),
            "questions_sha256": sha256_file(paths["validation_questions"]),
            "answer_key_path": str(paths["validation_answers"].relative_to(REPO_ROOT)),
            "answer_key_sha256": sha256_file(paths["validation_answers"]),
            "row_count": len(validation_rows),
        },
        "test": {
            "questions_path": str(paths["test_questions"].relative_to(REPO_ROOT)),
            "questions_sha256": sha256_file(paths["test_questions"]),
            "answer_key_path": str(paths["test_answers"].relative_to(REPO_ROOT)),
            "answer_key_sha256": sha256_file(paths["test_answers"]),
            "row_count": len(test_questions),
            "test_panel_id": sha256_file(paths["test_questions"]),
        },
    }
    write_json(paths["evaluator_manifest"], evaluator_manifest)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": config["dataset_id"],
        "record_schema_version": IDENTIFIER_TOOL_TRAJECTORY_SCHEMA_VERSION,
        "training_contract": TRAINING_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": registry_id,
            "sha256": registry_sha256,
            "source": registry.get("source"),
            "counts": registry.get("counts"),
            "normalization": registry.get("normalization"),
        },
        "system_prompt_contract": (
            s0_tool_trajectory_prompt_contract().to_dict()
        ),
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "tool_definitions": identifier_tool_definitions(),
        "config_sha256": sha256_file(config_path),
        "row_counts": row_counts,
        "family_counts": split_manifest["family_counts"],
        "train_population": {
            "eligible_train_rows": len(train_rows),
            "symbol_rows": int(config["train"]["symbol_rows"]),
            "gene_id_rows": int(config["train"]["gene_id_rows"]),
            "all_available_ambiguous_symbols_included": not bool(
                config["train"].get("excluded_fact_ids", [])
            ),
            "excluded_fact_ids": list(
                config["train"].get("excluded_fact_ids", [])
            ),
        },
        "file_hashes": {
            "train.jsonl": sha256_file(paths["train"]),
            "val.jsonl": sha256_file(paths["validation"]),
            "split_manifest.json": sha256_file(paths["split_manifest"]),
            "evaluator_manifest.json": sha256_file(paths["evaluator_manifest"]),
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
        default=REPO_ROOT / "config/world_model_v2_s0_tool_calls_v6.json",
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured corpus."""

    manifest = build_corpus(parse_args().config)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
