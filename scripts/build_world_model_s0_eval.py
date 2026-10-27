#!/usr/bin/env python3
"""Build answer-separated evaluation panels from the canonical S0 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import (  # noqa: E402
    load_contract,
    tool_definitions,
)
from runtime.world_model.s0_corpus import (  # noqa: E402
    EvidenceSequence,
    IdentityFact,
    _build_row,
    build_identity_components,
)
from runtime.world_model.source_index import check_source_index  # noqa: E402
from runtime.world_model.tools import WorldModelToolRuntime  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "data/world_model_v3/corpora/s0"
OUTPUT_ROOT = REPO_ROOT / "data/world_model_v3/eval/s0"
FULL_CORPUS_OUTPUT_ROOT = REPO_ROOT / "data/world_model_v3/eval/s0_full_corpus"
FULL_REGISTRY_OUTPUT_ROOT = (
    REPO_ROOT / "data/world_model_v3/eval/s0_full_registry"
)
FROZEN_FULL_REGISTRY_HASHES = {
    "manifest.json": (
        "dbca23382f0881aba83a6100ea54004031337bef53415236e8f1042970c53beb"
    ),
    "test_questions.jsonl": (
        "c9714921c9bd348051ad05452dade6bd32166102a3ae3c9fb12e118c04e41320"
    ),
    "test_answer_key.jsonl": (
        "bc8d4c9b83781d69f8e4f0457ec9d84ddf4eefe9b3be76b940c45a5d1598b65d"
    ),
}
SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-v3"
EVALUATION_CONTRACT = "held_out_multiplex_sft_v3"
FULL_CORPUS_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-full-corpus-evaluator-v3"
)
FULL_CORPUS_EVALUATION_CONTRACT = "full_corpus_multiplex_sft_v3"
FULL_REGISTRY_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-full-registry-evaluator-v3"
)
FULL_REGISTRY_EVALUATION_CONTRACT = "full_registry_multiplex_sft_v3"
FULL_REGISTRY_DATASET_ID = (
    "world_model_v3_s0_human_identifiers_full_registry"
)
FULL_REGISTRY_MAXIMUM_AMBIGUOUS_CANDIDATES = 49
FULL_REGISTRY_EXCLUDED_FACT_IDS = frozenset(
    {
        "fact_1544b1236c875b2ad88138d0",
        "fact_39c1052bc4d239a961544728",
        "fact_fd3fc526542e105e08c060b9",
    }
)
CORPUS_SPLITS = ("train", "validation", "test")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected one JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def project_row(
    row: Mapping[str, Any], split: str, *, fact_role: str = "unseen"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate one canonical row into a public prompt and private answer."""

    row_id = row.get("row_id")
    messages = row.get("messages")
    tools = row.get("tools")
    metadata = row.get("metadata")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError("An S0 row has no row_id")
    if not isinstance(messages, list) or len(messages) != 5:
        raise ValueError(f"{row_id} is not a five-message S0 trace")
    if not isinstance(tools, list) or not isinstance(metadata, Mapping):
        raise ValueError(f"{row_id} lacks tools or metadata")
    system, user, assistant_call, tool_result, assistant_final = messages
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user", "assistant", "tool", "assistant"]:
        raise ValueError(f"{row_id} has an invalid S0 role sequence")
    call = assistant_call.get("tool_call")
    if not isinstance(call, Mapping):
        raise ValueError(f"{row_id} lacks one tool call")
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{row_id} has invalid tool arguments")
    payload = tool_result.get("content")
    final = assistant_final.get("content")
    if not isinstance(payload, Mapping) or not isinstance(final, Mapping):
        raise ValueError(f"{row_id} lacks a tool payload or final action")
    split_unit = metadata.get("split_unit")
    family = metadata.get("question_family")
    if not isinstance(split_unit, str) or not isinstance(family, str):
        raise ValueError(f"{row_id} lacks split metadata")
    if split not in CORPUS_SPLITS or not isinstance(fact_role, str) or not fact_role:
        raise ValueError(f"{row_id} has invalid evaluation provenance")
    provenance = {
        "fact_group_id": split_unit,
        "fact_id": split_unit,
        "fact_role": fact_role,
        "prompt_form_id": split,
        "rendering_index": 0,
    }
    question = {
        "input": dict(arguments),
        "metadata": dict(metadata),
        "provenance": provenance,
        "question": user["content"],
        "record_id": row_id,
        "split": split,
        "system": system["content"],
        "tools": tools,
    }
    answer = {
        "expected_assistant_thinking": assistant_call.get("thinking"),
        "expected_final_answer": json.dumps(
            final,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "expected_final_explanation": assistant_final.get("thinking"),
        "expected_status": final.get("status"),
        "expected_continuation_decision": final.get("continuation_decision"),
        "expected_tool_call": {
            "name": call.get("name"),
            "arguments": dict(arguments),
        },
        "expected_tool_payload": dict(payload),
        "fact_group_id": split_unit,
        "fact_id": split_unit,
        "family": family,
        "record_id": row_id,
    }
    if any(
        not isinstance(answer[key], str) or not answer[key]
        for key in (
            "expected_assistant_thinking",
            "expected_final_answer",
            "expected_final_explanation",
            "expected_status",
            "expected_continuation_decision",
        )
    ):
        raise ValueError(f"{row_id} has an incomplete expected action")
    return question, answer


def _registry_contract(corpus: Mapping[str, Any]) -> dict[str, str]:
    """Return the shared content-addressed source-index contract."""

    source_index = corpus["source_index"]
    return {
        "path": str(
            Path("data/world_model_v3/source_indexes")
            / source_index["file_name"]
        ),
        "id": f"sha256:{source_index['content_sha256']}",
        "sha256": source_index["content_sha256"],
    }


def build() -> dict[str, Any]:
    corpus = read_json(CORPUS_ROOT / "manifest.json")
    panels: dict[str, Any] = {}
    for split in ("validation", "test"):
        questions: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []
        for row in read_jsonl(CORPUS_ROOT / f"{split}.jsonl"):
            question, answer = project_row(row, split)
            questions.append(question)
            answers.append(answer)
        questions_path = OUTPUT_ROOT / f"{split}_questions.jsonl"
        answers_path = OUTPUT_ROOT / f"{split}_answer_key.jsonl"
        write_jsonl(questions_path, questions)
        write_jsonl(answers_path, answers)
        questions_hash = sha256_file(questions_path)
        panels[split] = {
            "questions_path": str(questions_path.relative_to(REPO_ROOT)),
            "questions_sha256": questions_hash,
            "answer_key_path": str(answers_path.relative_to(REPO_ROOT)),
            "answer_key_sha256": sha256_file(answers_path),
            "row_count": len(questions),
            f"{split}_panel_id": questions_hash,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": corpus["dataset_id"],
        "evaluation_contract": EVALUATION_CONTRACT,
        "loss_contract": "multiplex_sft_v3",
        "corpus_manifest": str(
            (CORPUS_ROOT / "manifest.json").relative_to(REPO_ROOT)
        ),
        "corpus_manifest_sha256": sha256_file(CORPUS_ROOT / "manifest.json"),
        "identifier_registry": _registry_contract(corpus),
        "validation": panels["validation"],
        "test": panels["test"],
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return manifest


def build_full_corpus(
    *,
    corpus_root: Path = CORPUS_ROOT,
    output_root: Path = FULL_CORPUS_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Build one diagnostic panel containing every canonical S0 row."""

    corpus = read_json(corpus_root / "manifest.json")
    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    split_row_counts: dict[str, int] = {}
    for split in CORPUS_SPLITS:
        split_rows = read_jsonl(corpus_root / f"{split}.jsonl")
        split_row_counts[split] = len(split_rows)
        fact_role = "train" if split == "train" else "unseen"
        for row in split_rows:
            question, answer = project_row(
                row,
                split,
                fact_role=fact_role,
            )
            questions.append(question)
            answers.append(answer)

    record_ids = [str(row["record_id"]) for row in questions]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("The full S0 corpus has duplicate row IDs")
    declared_counts = corpus.get("row_counts")
    if not isinstance(declared_counts, Mapping) or split_row_counts != {
        split: declared_counts.get(split) for split in CORPUS_SPLITS
    }:
        raise ValueError("The full S0 corpus split counts changed")

    questions_path = output_root / "questions.jsonl"
    answers_path = output_root / "answer_key.jsonl"
    write_jsonl(questions_path, questions)
    write_jsonl(answers_path, answers)
    questions_hash = sha256_file(questions_path)
    panel = {
        "questions_path": str(questions_path.relative_to(REPO_ROOT)),
        "questions_sha256": questions_hash,
        "answer_key_path": str(answers_path.relative_to(REPO_ROOT)),
        "answer_key_sha256": sha256_file(answers_path),
        "row_count": len(questions),
        "test_panel_id": questions_hash,
    }
    manifest = {
        "schema_version": FULL_CORPUS_SCHEMA_VERSION,
        "dataset_id": corpus["dataset_id"],
        "evaluation_contract": FULL_CORPUS_EVALUATION_CONTRACT,
        "loss_contract": "multiplex_sft_v3",
        "corpus_manifest": str(
            (corpus_root / "manifest.json").relative_to(REPO_ROOT)
        ),
        "corpus_manifest_sha256": sha256_file(
            corpus_root / "manifest.json"
        ),
        "identifier_registry": _registry_contract(corpus),
        "population": {
            "scope": "canonical_s0_corpus_all_splits",
            "includes_training_rows": True,
            "split_order": list(CORPUS_SPLITS),
            "split_row_counts": split_row_counts,
            "row_count": len(questions),
        },
        # Keep the transport key used by the existing generation/scoring
        # pipeline; the distinct contract and population declare its scope.
        "test": panel,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def full_registry_facts(
    genes_by_id: Mapping[str, list[str | None]],
) -> list[IdentityFact]:
    """Return exactly one fact per registry ID and casefolded symbol."""

    components, _ = build_identity_components(genes_by_id)
    facts = sorted(
        (fact for component in components for fact in component.facts),
        key=lambda fact: fact.fact_id,
    )
    expected = len(genes_by_id) + len(
        {
            str(values[0]).casefold()
            for values in genes_by_id.values()
            if values[0] is not None
        }
    )
    if len(facts) != expected:
        raise ValueError(
            f"The full registry produced {len(facts)} facts, expected {expected}"
        )
    return facts


def build_full_registry(
    *,
    corpus_root: Path = CORPUS_ROOT,
    output_root: Path = FULL_REGISTRY_OUTPUT_ROOT,
    allow_frozen_rebuild: bool = False,
) -> dict[str, Any]:
    """Build an evaluation panel over every current Ensembl ID and symbol."""

    if (
        output_root.resolve() == FULL_REGISTRY_OUTPUT_ROOT.resolve()
        and not allow_frozen_rebuild
    ):
        raise RuntimeError(
            "The v3 full-registry gate is byte-frozen and cannot be rebuilt"
        )

    contract = load_contract()
    corpus = read_json(corpus_root / "manifest.json")
    source_index = corpus["source_index"]
    source_index_path = (
        REPO_ROOT
        / "data/world_model_v3/source_indexes"
        / source_index["file_name"]
    )
    payload = check_source_index(source_index_path, contract)
    all_facts = full_registry_facts(payload["genes_by_id"])
    excluded_facts = [
        fact
        for fact in all_facts
        if fact.candidate_count
        > FULL_REGISTRY_MAXIMUM_AMBIGUOUS_CANDIDATES
    ]
    excluded_fact_ids = frozenset(fact.fact_id for fact in excluded_facts)
    if excluded_fact_ids != FULL_REGISTRY_EXCLUDED_FACT_IDS:
        raise ValueError(
            "The unsafe full-registry ambiguity set changed: "
            f"expected={sorted(FULL_REGISTRY_EXCLUDED_FACT_IDS)} "
            f"actual={sorted(excluded_fact_ids)}"
        )
    facts = [
        fact for fact in all_facts if fact.fact_id not in excluded_fact_ids
    ]
    seed = int(contract["curriculum"]["families"]["S0"]["assignment_seed"])
    runtime = WorldModelToolRuntime.from_source_index(
        source_index_path,
        contract=contract,
        evidence_id_factory=EvidenceSequence(),
    )
    tools = tool_definitions(contract)
    excluded_records = sorted(
        (
            {
                "record_id": "wm3_s0_"
                + hashlib.sha256(
                    canonical_json(
                        {
                            "fact_id": fact.fact_id,
                            "schema_version": contract["runtime"][
                                "row_schema_version"
                            ],
                            "split": "test",
                        }
                    ).encode("utf-8")
                ).hexdigest()[:24],
                "fact_id": fact.fact_id,
                "family": "human_ambiguous_symbol",
                "gene_symbol": fact.value,
                "candidate_gene_id_count": fact.candidate_count,
            }
            for fact in excluded_facts
        ),
        key=lambda row: str(row["record_id"]),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    questions_path = output_root / "test_questions.jsonl"
    answers_path = output_root / "test_answer_key.jsonl"
    questions_temporary = questions_path.with_suffix(".jsonl.tmp")
    answers_temporary = answers_path.with_suffix(".jsonl.tmp")
    family_counts = {
        "human_symbol_to_ensembl": 0,
        "human_ambiguous_symbol": 0,
        "human_ensembl_to_symbol": 0,
    }
    status_counts = {"resolved": 0, "ambiguous": 0}
    record_ids: set[str] = set()
    try:
        with questions_temporary.open("w", encoding="utf-8") as question_file, (
            answers_temporary.open("w", encoding="utf-8")
        ) as answer_file:
            for fact in facts:
                row, _ = _build_row(
                    fact,
                    split="test",
                    seed=seed,
                    runtime=runtime,
                    contract=contract,
                    tools=tools,
                )
                question, answer = project_row(
                    row,
                    "test",
                    fact_role="registry_population",
                )
                record_id = str(question["record_id"])
                if record_id in record_ids:
                    raise ValueError(
                        "The full registry panel has duplicate row IDs"
                    )
                record_ids.add(record_id)
                question_file.write(canonical_json(question) + "\n")
                answer_file.write(canonical_json(answer) + "\n")
                family = str(row["metadata"]["question_family"])
                status = str(row["messages"][-1]["content"]["status"])
                family_counts[family] += 1
                status_counts[status] += 1
        questions_temporary.replace(questions_path)
        answers_temporary.replace(answers_path)
    finally:
        questions_temporary.unlink(missing_ok=True)
        answers_temporary.unlink(missing_ok=True)

    questions_hash = sha256_file(questions_path)
    counts = payload["counts"]
    population = {
        "scope": (
            "all_ensembl_registry_gene_ids_and_symbols_except_declared_"
            "long_ambiguities"
        ),
        "row_count": len(facts),
        "family_counts": family_counts,
        "status_counts": status_counts,
        "registry_gene_id_count": counts["registry_gene_count"],
        "registry_symbol_count": counts["registry_symbol_count"],
        "registry_ambiguous_symbol_count": counts[
            "registry_ambiguous_symbol_count"
        ],
        "registry_unnamed_gene_id_count": counts[
            "registry_unnamed_gene_count"
        ],
        "registry_largest_ambiguous_candidate_count": counts[
            "registry_largest_candidate_count"
        ],
        "largest_included_ambiguous_candidate_count": max(
            fact.candidate_count
            for fact in facts
            if fact.candidate_count > 1
        ),
        "population_filter": {
            "maximum_ambiguous_candidate_gene_ids": (
                FULL_REGISTRY_MAXIMUM_AMBIGUOUS_CANDIDATES
            )
        },
        "excluded_row_count": len(excluded_records),
        "excluded_records": excluded_records,
    }
    panel = {
        "questions_path": str(questions_path.relative_to(REPO_ROOT)),
        "questions_sha256": questions_hash,
        "answer_key_path": str(answers_path.relative_to(REPO_ROOT)),
        "answer_key_sha256": sha256_file(answers_path),
        "row_count": len(facts),
        "test_panel_id": questions_hash,
        "population_filter": {
            "maximum_ambiguous_candidate_gene_ids": (
                FULL_REGISTRY_MAXIMUM_AMBIGUOUS_CANDIDATES
            ),
            "excluded_records": excluded_records,
        },
    }
    manifest = {
        "schema_version": FULL_REGISTRY_SCHEMA_VERSION,
        "dataset_id": FULL_REGISTRY_DATASET_ID,
        "source_dataset_id": corpus["dataset_id"],
        "evaluation_contract": FULL_REGISTRY_EVALUATION_CONTRACT,
        "loss_contract": "multiplex_sft_v3",
        "corpus_manifest": str(
            (corpus_root / "manifest.json").relative_to(REPO_ROOT)
        ),
        "corpus_manifest_sha256": sha256_file(corpus_root / "manifest.json"),
        "identifier_registry": _registry_contract(corpus),
        "population": population,
        "test": panel,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def verify_frozen_full_registry() -> dict[str, str]:
    """Fail closed unless every byte-frozen v3 gate artifact is unchanged."""

    actual = {}
    for name, expected in FROZEN_FULL_REGISTRY_HASHES.items():
        path = FULL_REGISTRY_OUTPUT_ROOT / name
        if not path.is_file():
            raise RuntimeError(f"The frozen v3 gate artifact is absent: {path}")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"The frozen v3 gate artifact changed: {name}"
            )
        actual[name] = digest
    return actual


def main() -> int:
    """Verify the frozen gate; panel construction moved to versioned builders."""

    print(
        json.dumps(
            {
                "status": "verified",
                "full_registry_output": str(FULL_REGISTRY_OUTPUT_ROOT),
                "artifacts": verify_frozen_full_registry(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
