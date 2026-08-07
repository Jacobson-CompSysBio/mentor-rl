#!/usr/bin/env python3
"""Build the s0 human gene identifier corpus."""

# imports
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from collections.abc import Iterable, Mapping

# find repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# import prompt contract
from runtime.world_model_prompts import stage_prompt_contract

# schema, data, training, eval, and family constants
CONFIG_SCHEMA_VERSION = "mentor-rl-world-model-s0-build-v4"
REGISTRY_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-ensembl-registry-v1"
)
RECORD_SCHEMA_VERSION = "identifier_sft_v2"
TRAINING_CONTRACT = "closed_book_only_v1"
EVALUATION_CONTRACT = "seen_fact_closed_book_recall_v1"

FAMILIES = (
    "human_symbol_to_ensembl",
    "human_ensembl_to_symbol",
    "human_ambiguous_symbol",
)

# training split details
S0_PROMPT_CONTRACT = stage_prompt_contract("S0")

ROW_SPLITS = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
RENDERING_INDICES = {
    "train": 0,
    "validation": 100,
    "test": 200,
}
GENE_ID_PATTERN = r"^ENSG[0-9]{11}$"

### JSON, HASH HELPERS
def canonical_json(value: Any) -> str:
    """Return compact canonical json."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True
    )

def stable_hash(value: Any) -> str:
    """Return a stable sha-256 hash for one json"""
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def sha256_file(path: Path) -> str:
    """Return sha-256 hash of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object"""
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise TypeError(f"expected one json object: {path}")

    return value

def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one readable JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write compact JSON Lines rows."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(canonical_json(row) + "\n")

### LOADING FUNCTIONS
def resolve_repo_path(value: str) -> Path:
    """Resolve a path relative to the repo root."""
    path = Path(value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (REPO_ROOT / path).resolve()

def load_config(config_path: Path) -> dict[str, Any]:
    """load v4 corpus config."""
    config = read_json(config_path)

    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"the config must use {CONFIG_SCHEMA_VERSION}"
        )

    return config

def load_identifier_registry(
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, list[str]]]:
    """Load the pinned ensembl identifier registry."""
    source = config["identifier_registry"]

    if not isinstance(source, Mapping):
        raise TypeError("identifier_registry must be an object")

    registry_path = resolve_repo_path(source["path"])
    actual_hash = sha256_file(registry_path)

    if actual_hash != source["expected_sha256"]:
        raise ValueError(
            f"The identifier registry hash changed: {actual_hash}"
        )

    registry = read_json(registry_path)

    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"The registry must use {REGISTRY_SCHEMA_VERSION}"
        )

    mappings = registry.get("gene_symbols_by_id")

    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("The registry contains no gene mappings")

    return registry_path, registry, mappings

### CONSTRUCT A SINGLE ROW OF THE DATASET
def make_fact(
    family: str,
    fact_group_id: str,
    inputs: dict[str, str],
    answer: dict[str, Any],
) -> dict[str, Any]:
    """Using the ensembl registry, construct a single fact (row) for the dataset."""

    # tracking
    fact_id = "fact_" + stable_hash(
        {
            "family": family,
            "input": inputs,
            "answer": answer
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
    """Build identifier components and their facts."""

    # Remove duplicate symbols and give each gene ID a stable symbol order.
    symbols_by_id = {
        gene_id: sorted(set(symbols))
        for gene_id, symbols in mappings.items()
    }

    # Map each symbol back to all gene IDs that use it.
    ids_by_symbol: dict[str, set[str]] = {}

    # Store the parent node for each graph node.
    parent: dict[str, str] = {}

    # Find the root of one node and shorten its stored path.
    def find(value: str) -> str:
        parent.setdefault(value, value)

        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]

        return value

    # Join two graph nodes under one deterministic root.
    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)

        if left_root == right_root:
            return

        low, high = sorted((left_root, right_root))
        parent[high] = low

    # Connect each gene ID to each of its symbols.
    for gene_id, symbols in symbols_by_id.items():
        gene_node = f"id:{gene_id}"
        find(gene_node)

        for symbol in symbols:
            union(gene_node, f"symbol:{symbol}")
            ids_by_symbol.setdefault(symbol, set()).add(gene_id)

    # Collect all IDs and symbols under each graph root.
    groups: dict[str, dict[str, set[str]]] = {}

    for gene_id, symbols in symbols_by_id.items():
        root = find(f"id:{gene_id}")
        group = groups.setdefault(
            root,
            {
                "gene_ids": set(),
                "gene_symbols": set(),
            },
        )
        group["gene_ids"].add(gene_id)
        group["gene_symbols"].update(symbols)

    # Give each family a stable sort position.
    family_order = {
        family: index
        for index, family in enumerate(FAMILIES)
    }
    components: list[dict[str, Any]] = []

    # Convert each graph component into its complete set of facts.
    for group in groups.values():
        gene_ids = sorted(group["gene_ids"])
        gene_symbols = sorted(group["gene_symbols"])

        # Create a stable group ID from all component values.
        fact_group_id = "group_" + stable_hash(
            {
                "gene_ids": gene_ids,
                "gene_symbols": gene_symbols,
            }
        )[:24]

        facts: list[dict[str, Any]] = []
        is_ambiguous = False

        # Create S0.1 facts for unique symbols and S0.3 facts for ambiguous symbols.
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

        # Create one S0.2 fact for each gene ID.
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

        # Sort the facts and store the completed component.
        facts.sort(
            key=lambda fact: (
                family_order[fact["family"]],
                fact["fact_id"],
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

    # Give the complete component list a stable order.
    components.sort(key=lambda component: component["fact_group_id"])
    return components

### DIVIDE TRAIN/TEST/VAL INTO CONNECTED COMPONENTS
def assign_evaluation_components(
    components: list[dict[str, Any]],
    split_config: Mapping[str, Any],
) -> dict[str, str]:
    """Assign disjoint components to validation and test."""

    # Separate components into ambiguous and unambiguous groups.
    pools = {
        "unambiguous": [
            component
            for component in components
            if not component["is_ambiguous"]
        ],
        "ambiguous": [
            component
            for component in components
            if component["is_ambiguous"]
        ]
    }

    seed = int(split_config["assignment_seed"])
    assignments: dict[str, str] = {}

    # fill validation first and test second from remaining components
    for split_name in ("validation", "test"):
        for component_type in ("unambiguous", "ambiguous"):
            count_key = f"{component_type}_component_count"
            required_count = int(split_config[split_name][count_key])

            # give pool a stable split-specific order
            ordered = sorted(
                pools[component_type],
                key=lambda component: stable_hash(
                    {
                        "assignment_seed": seed,
                        "split": split_name,
                        "fact_group_id": component[
                            "fact_group_id"
                        ],
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
                assignments[component["fact_group_id"]] = split_name

    return assignments

def validator_specs(family: str) -> dict[str, Any]:
    """Return the validators for one S0 family."""

    # S0.1 and S0.2 both require a resolved gene ID.
    if family in FAMILIES[:2]:
        validators = {
            "status": {
                "primitive": "ENUM",
                "required": True,
                "options": {"values": ["resolved"]},
            },
            "gene_id": {
                "primitive": "EXACT_ID",
                "required": True,
                "options": {"pattern": GENE_ID_PATTERN},
            },
        }

        if family == FAMILIES[0]:
            validators["gene_symbol"] = {
                "primitive": "SYMBOL",
                "required": True,
                "options": {},
            }
        else:
            validators["gene_symbols"] = {
                "primitive": "SYMBOL_SET",
                "required": True,
                "options": {},
            }

        return validators

    # S0.3 requires the complete candidate set and a defer action.
    if family == FAMILIES[2]:
        return {
            "status": {
                "primitive": "ENUM",
                "required": True,
                "options": {"values": ["ambiguous"]},
            },
            "gene_symbol": {
                "primitive": "SYMBOL",
                "required": True,
                "options": {},
            },
            "candidate_gene_ids": {
                "primitive": "ID_SET",
                "required": True,
                "options": {"item_pattern": GENE_ID_PATTERN},
            },
            "action": {
                "primitive": "ENUM",
                "required": True,
                "options": {"values": ["defer"]},
            },
        }

    raise ValueError(f"Unknown S0 family: {family}")


def make_record(
    fact: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    identifier_registry_id: str,
    prompt_form_id: str,
) -> dict[str, Any]:
    """Create one train, validation, or test record."""

    # Select the row split and stable rendering index.
    split = ROW_SPLITS[prompt_form_id]
    rendering_index = RENDERING_INDICES[prompt_form_id]

    # Insert the compact fact input into the selected question form.
    family = fact["family"]
    question_template = config["prompt_forms"][
        prompt_form_id
    ][family]
    question = question_template.format(
        input_json=canonical_json(fact["input"])
    )

    # Give each split-specific rendering its own record ID.
    record_id = "wm2_s0_" + stable_hash(
        {
            "fact_id": fact["fact_id"],
            "rendering_index": rendering_index,
            "split": split,
        }
    )[:24]

    return {
        "record_id": record_id,
        "split": split,
        "system": S0_PROMPT_CONTRACT.system_prompt,
        "input": fact["input"],
        "question": question,
        "context": None,
        "answer": fact["answer"],
        "metadata": {
            "schema_version": RECORD_SCHEMA_VERSION,
            "book_mode": "closed_book",
            "step": "S0",
            "question_family": family,
            "species_taxon_id": "NCBITaxon:9606",
            "ensembl_release": "Ensembl_116",
            "identifier_registry_id": identifier_registry_id,
            "system_prompt_sha256": (
                S0_PROMPT_CONTRACT.system_prompt_sha256
            ),
            "answer_format": "json",
        },
        "validators": validator_specs(family),
        "provenance": {
            "fact_id": fact["fact_id"],
            "fact_group_id": fact["fact_group_id"],
            "fact_role": "seen",
            "rendering_index": rendering_index,
            "prompt_form_id": prompt_form_id,
        },
    }

def public_question_row(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove the answer from one evaluation record."""

    # Keep the question, metadata, validators, and provenance.
    return {
        key: value
        for key, value in record.items()
        if key != "answer"
    }


def answer_key_row(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Create one private answer-key row."""

    # Store only the fields required to join and score the answer.
    return {
        "record_id": record["record_id"],
        "fact_id": record["provenance"]["fact_id"],
        "fact_group_id": record["provenance"][
            "fact_group_id"
        ],
        "family": record["metadata"]["question_family"],
        "answer": record["answer"],
    }

def family_counts(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count records for each S0 family."""
    counts = {family: 0 for family in FAMILIES}

    for record in records:
        family = record["metadata"]["question_family"]
        counts[family] += 1

    return counts


def protect_evaluator_dir(evaluator_dir: Path) -> None:
    """Restrict access to evaluator files."""
    evaluator_dir.chmod(0o700)

    for path in evaluator_dir.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)


def build_corpus(config_path: Path) -> dict[str, Any]:
    """Build and write the complete v4 corpus."""
    config = load_config(config_path)
    registry_path, registry, mappings = (
        load_identifier_registry(config)
    )

    components = build_components(mappings)
    assignments = assign_evaluation_components(
        components,
        config["split"],
    )

    registry_hash = config["identifier_registry"][
        "expected_sha256"
    ]
    identifier_registry_id = f"sha256:{registry_hash}"

    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    # Create one train record for every fact.
    for component in components:
        evaluation_split = assignments.get(
            component["fact_group_id"]
        )

        for fact in component["facts"]:
            train_records.append(
                make_record(
                    fact,
                    config=config,
                    identifier_registry_id=(
                        identifier_registry_id
                    ),
                    prompt_form_id="train",
                )
            )

            # Create one evaluation record for each selected fact.
            if evaluation_split is not None:
                evaluation_record = make_record(
                    fact,
                    config=config,
                    identifier_registry_id=(
                        identifier_registry_id
                    ),
                    prompt_form_id=evaluation_split,
                )

                if evaluation_split == "validation":
                    validation_records.append(
                        evaluation_record
                    )
                else:
                    test_records.append(evaluation_record)

    # Give each output file a stable row order.
    train_records.sort(key=lambda row: row["record_id"])
    validation_records.sort(key=lambda row: row["record_id"])
    test_records.sort(key=lambda row: row["record_id"])

    validation_questions = [
        public_question_row(record)
        for record in validation_records
    ]
    test_questions = [
        public_question_row(record)
        for record in test_records
    ]
    validation_answers = [
        answer_key_row(record)
        for record in validation_records
    ]
    test_answers = [
        answer_key_row(record)
        for record in test_records
    ]

    outputs = config["outputs"]
    corpus_dir = resolve_repo_path(outputs["corpus_dir"])
    evaluator_dir = resolve_repo_path(
        outputs["evaluator_dir"]
    )

    train_path = corpus_dir / "train.jsonl"
    validation_path = corpus_dir / "val.jsonl"
    split_manifest_path = corpus_dir / "split_manifest.json"
    manifest_path = corpus_dir / "manifest.json"

    validation_answer_path = (
        evaluator_dir / "validation_answer_key.jsonl"
    )
    test_question_path = evaluator_dir / "test_questions.jsonl"
    test_answer_path = evaluator_dir / "test_answer_key.jsonl"
    evaluator_manifest_path = evaluator_dir / "manifest.json"

    # Write public train and validation files.
    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_questions)

    # Write private validation answers and the sealed test panel.
    write_jsonl(
        validation_answer_path,
        validation_answers,
    )
    write_jsonl(test_question_path, test_questions)
    write_jsonl(test_answer_path, test_answers)

    train_hash = sha256_file(train_path)
    validation_question_hash = sha256_file(
        validation_path
    )
    validation_answer_hash = sha256_file(
        validation_answer_path
    )
    test_question_hash = sha256_file(test_question_path)
    test_answer_hash = sha256_file(test_answer_path)
    test_panel_id = test_question_hash

    row_counts = {
        "train": len(train_records),
        "validation": len(validation_records),
        "test": len(test_records),
    }
    counts_by_family = {
        "train": family_counts(train_records),
        "validation": family_counts(validation_records),
        "test": family_counts(test_records),
    }

    validation_group_ids = sorted(
        group_id
        for group_id, split_name in assignments.items()
        if split_name == "validation"
    )
    test_group_ids = sorted(
        group_id
        for group_id, split_name in assignments.items()
        if split_name == "test"
    )

    split_manifest = {
        "schema_version": (
            "mentor-rl-world-model-s0-split-manifest-v4"
        ),
        "dataset_id": config["dataset_id"],
        "training_contract": TRAINING_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "assignment_seed": config["split"][
            "assignment_seed"
        ],
        "fact_role": "seen",
        "validation_fact_group_ids": validation_group_ids,
        "test_fact_group_ids": test_group_ids,
        "row_counts": row_counts,
        "family_counts": counts_by_family,
        "train_sha256": train_hash,
        "validation_questions_sha256": (
            validation_question_hash
        ),
        "test_questions_sha256": test_question_hash,
        "test_panel_id": test_panel_id,
    }
    write_json(split_manifest_path, split_manifest)
    split_manifest_hash = sha256_file(split_manifest_path)

    evaluator_manifest = {
        "schema_version": (
            "mentor-rl-world-model-s0-evaluator-manifest-v4"
        ),
        "dataset_id": config["dataset_id"],
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry_id": identifier_registry_id,
        "validation": {
            "questions_path": str(
                validation_path.relative_to(REPO_ROOT)
            ),
            "questions_sha256": validation_question_hash,
            "answer_key_path": str(
                validation_answer_path.relative_to(REPO_ROOT)
            ),
            "answer_key_sha256": validation_answer_hash,
            "row_count": len(validation_records),
        },
        "test": {
            "test_panel_id": test_panel_id,
            "questions_path": str(
                test_question_path.relative_to(REPO_ROOT)
            ),
            "questions_sha256": test_question_hash,
            "answer_key_path": str(
                test_answer_path.relative_to(REPO_ROOT)
            ),
            "answer_key_sha256": test_answer_hash,
            "row_count": len(test_records),
        },
    }
    write_json(evaluator_manifest_path, evaluator_manifest)
    evaluator_manifest_hash = sha256_file(
        evaluator_manifest_path
    )

    manifest = {
        "schema_version": (
            "mentor-rl-world-model-s0-manifest-v4"
        ),
        "dataset_id": config["dataset_id"],
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "training_contract": TRAINING_CONTRACT,
        "evaluation_contract": EVALUATION_CONTRACT,
        "identifier_registry": {
            "path": str(registry_path.relative_to(REPO_ROOT)),
            "id": identifier_registry_id,
            "source": registry["source"],
            "counts": registry["counts"],
            "normalization": registry["normalization"],
        },
        "system_prompt_contract": (
            S0_PROMPT_CONTRACT.to_dict()
        ),
        "config_sha256": sha256_file(config_path),
        "row_counts": row_counts,
        "family_counts": counts_by_family,
        "file_hashes": {
            "train.jsonl": train_hash,
            "val.jsonl": validation_question_hash,
            "split_manifest.json": split_manifest_hash,
            "evaluator_manifest.json": (
                evaluator_manifest_hash
            ),
        },
    }
    write_json(manifest_path, manifest)
    protect_evaluator_dir(evaluator_dir)

    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the S0 human gene identifier corpus."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "config"
            / "world_model_v2_s0_closed_book_recall_v4.json"
        ),
        help="Path to the v4 corpus config.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the configured S0 corpus."""
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    manifest = build_corpus(config_path)

    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "row_counts": manifest["row_counts"],
                "family_counts": manifest["family_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
