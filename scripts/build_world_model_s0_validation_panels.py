#!/usr/bin/env python3
"""Build the S0 v4 validation panels from the published v4 corpus.

The trainer scores teacher-forced validation loss on the 622-row
teacher_forced_validation panel. The manifest also holds the 4,096-row
behavioral and 512-row snapshot panels of the retired snapshot pipeline. They
stay in the build so that the manifest keeps the SHA-256 value that the v4
training configs pin.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model.contract import load_contract, tool_definitions  # noqa: E402
from runtime.world_model.s0_corpus import (  # noqa: E402
    AMBIGUOUS_SYMBOL,
    HISTORICAL_SYMBOL,
    NAMED_GENE_ID,
    RESOLVED_SYMBOL,
    RETIRED_GENE_ID,
    UNNAMED_GENE_ID,
    EvidenceSequence,
    IdentityFact,
    _build_row,
    _ordered_facts,
    _stable_sha256,
    assign_evaluation_components,
    build_identity_components,
    load_s0_v4_settings,
    select_registry_facts,
    select_supplemental_facts,
)
from runtime.world_model.source_index import check_source_index  # noqa: E402
from runtime.world_model.tools import WorldModelToolRuntime  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "data/world_model_v4/corpora/s0"
OUTPUT_ROOT = REPO_ROOT / "data/world_model_v4/eval/s0_validation"
SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-v4"
EVALUATION_CONTRACT = "held_out_multiplex_sft_v4"
BEHAVIORAL_COUNTS = {
    RESOLVED_SYMBOL: 1024,
    NAMED_GENE_ID: 1024,
    UNNAMED_GENE_ID: 1024,
    AMBIGUOUS_SYMBOL: 512,
    HISTORICAL_SYMBOL: 256,
    RETIRED_GENE_ID: 256,
}
MICRO_COUNTS = {
    RESOLVED_SYMBOL: 128,
    NAMED_GENE_ID: 128,
    UNNAMED_GENE_ID: 128,
    AMBIGUOUS_SYMBOL: 64,
    HISTORICAL_SYMBOL: 32,
    RETIRED_GENE_ID: 32,
}


CORPUS_SPLITS = ("train", "validation", "test")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected one JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(canonical_json(row) + "\n")
    temporary.replace(path)


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


def _validation_facts(
    payload: Mapping[str, Any], settings: Mapping[str, Any]
) -> list[IdentityFact]:
    seed = int(settings["assignment_seed"])
    components, component_by_gene = build_identity_components(payload["genes_by_id"])
    assignments = assign_evaluation_components(
        components, settings["evaluation"], seed=seed
    )
    registry = select_registry_facts(
        components,
        assignments,
        settings,
        seed=seed,
        maximum_ambiguous_candidates=int(
            settings["maximum_train_ambiguous_candidates"]
        ),
    )
    supplemental = select_supplemental_facts(
        payload, component_by_gene, assignments, settings, seed=seed
    )
    return _ordered_facts(
        registry["validation"] + supplemental["validation"],
        seed=seed,
        pool="snapshot_validation_facts",
    )


def _rendering_plan(
    facts: Sequence[IdentityFact], counts: Mapping[str, int]
) -> list[tuple[IdentityFact, str, int, str]]:
    by_kind: dict[str, list[IdentityFact]] = defaultdict(list)
    for fact in facts:
        by_kind[fact.kind].append(fact)
    result: list[tuple[IdentityFact, str, int, str]] = []
    for kind, total in counts.items():
        if total % 8:
            raise ValueError(f"{kind} allocation is not divisible by eight")
        per_cell = total // 8
        pool = sorted(
            by_kind[kind],
            key=lambda fact: _stable_sha256(
                {"pool": "behavioral_validation", "kind": kind, "fact_id": fact.fact_id}
            ),
        )
        for framing in ("direct", "step"):
            for template_index in range(4):
                ordered = sorted(
                    pool,
                    key=lambda fact: _stable_sha256(
                        {
                            "pool": "behavioral_validation_cell",
                            "kind": kind,
                            "framing": framing,
                            "template_index": template_index,
                            "fact_id": fact.fact_id,
                        }
                    ),
                )
                casefold = max(per_cell // 8, per_cell - len(ordered))
                canonical = per_cell - casefold
                casefold_pool = [fact for fact in ordered if fact.value.lower() != fact.value]
                if canonical > len(ordered) or casefold > len(casefold_pool):
                    raise ValueError(f"{kind} cannot fill one validation cell")
                result.extend(
                    (fact, framing, template_index, "canonical")
                    for fact in ordered[:canonical]
                )
                if casefold:
                    result.extend(
                        (fact, framing, template_index, "casefold")
                        for fact in casefold_pool[-casefold:]
                    )
    if len(result) != sum(counts.values()):
        raise AssertionError("the validation rendering plan is incomplete")
    if len(set(result)) != len(result):
        raise AssertionError("the validation rendering plan has duplicate rows")
    return sorted(
        result,
        key=lambda item: _stable_sha256(
            {
                "pool": "behavioral_validation_order",
                "fact_id": item[0].fact_id,
                "framing": item[1],
                "template_index": item[2],
                "input_variant": item[3],
            }
        ),
    )


def _project(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for row in rows:
        question, answer = project_row(row, "validation")
        metadata = row["metadata"]
        question["provenance"]["prompt_form_id"] = (
            f"validation_{metadata['framing']}"
        )
        question["provenance"]["rendering_index"] = metadata[
            "rendering_index"
        ]
        questions.append(question)
        answers.append(answer)
    return questions, answers


def _panel(
    prefix: str,
    questions: Sequence[Mapping[str, Any]],
    answers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    questions_path = OUTPUT_ROOT / f"{prefix}_questions.jsonl"
    answers_path = OUTPUT_ROOT / f"{prefix}_answer_key.jsonl"
    write_jsonl(questions_path, questions)
    write_jsonl(answers_path, answers)
    questions_sha = sha256_file(questions_path)
    return {
        "questions_path": str(questions_path.relative_to(REPO_ROOT)),
        "questions_sha256": questions_sha,
        "answer_key_path": str(answers_path.relative_to(REPO_ROOT)),
        "answer_key_sha256": sha256_file(answers_path),
        "row_count": len(questions),
        "validation_panel_id": questions_sha,
    }


def build() -> dict[str, Any]:
    corpus = read_json(CORPUS_ROOT / "manifest.json")
    contract = load_contract()
    settings, _ = load_s0_v4_settings(contract)
    source_path = (
        REPO_ROOT
        / "data/world_model_v3/source_indexes"
        / corpus["source_index"]["file_name"]
    )
    payload = check_source_index(source_path, contract)
    facts = _validation_facts(payload, settings)
    plan = _rendering_plan(facts, BEHAVIORAL_COUNTS)
    seed = int(settings["assignment_seed"])
    runtime = WorldModelToolRuntime.from_source_index(
        source_path,
        contract=contract,
        evidence_id_factory=EvidenceSequence(),
    )
    tools = tool_definitions(contract)
    rows = [
        _build_row(
            fact,
            split="validation",
            seed=seed,
            runtime=runtime,
            contract=contract,
            tools=tools,
            framing=framing,
            template_index=template_index,
            rendering_index=template_index,
            input_variant=input_variant,
        )[0]
        for fact, framing, template_index, input_variant in plan
    ]
    record_ids = [str(row["row_id"]) for row in rows]
    if len(set(record_ids)) != len(rows):
        raise ValueError("The behavioral validation panel has duplicate records")
    questions, answers = _project(rows)
    behavioral = _panel("behavioral_validation", questions, answers)

    micro_plan = set(_rendering_plan(facts, MICRO_COUNTS))
    micro_indexes = [
        index for index, rendering in enumerate(plan) if rendering in micro_plan
    ]
    if len(micro_indexes) != 512:
        raise ValueError("The 512-row micro-panel allocation is incomplete")
    micro_questions = [questions[index] for index in micro_indexes]
    micro_answers = [answers[index] for index in micro_indexes]
    micro = _panel("snapshot_validation", micro_questions, micro_answers)
    teacher_questions, teacher_answers = _project(
        read_jsonl(CORPUS_ROOT / "validation.jsonl")
    )
    teacher_forced = _panel(
        "teacher_forced_validation", teacher_questions, teacher_answers
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": corpus["dataset_id"],
        "evaluation_contract": EVALUATION_CONTRACT,
        "row_schema_version": "multiplex_sft_v3",
        "compatible_loss_contracts": [
            "multiplex_sft_v3",
            "multiplex_sft_v4_priority",
        ],
        "corpus_manifest": str((CORPUS_ROOT / "manifest.json").relative_to(REPO_ROOT)),
        "corpus_manifest_sha256": sha256_file(CORPUS_ROOT / "manifest.json"),
        "identifier_registry": {
            "path": str(source_path.relative_to(REPO_ROOT)),
            "id": f"sha256:{corpus['source_index']['content_sha256']}",
            "sha256": corpus["source_index"]["content_sha256"],
        },
        "population": {
            "scope": "validation_only_prompt_renderings",
            "test_prompt_access": False,
            "behavioral_row_counts": BEHAVIORAL_COUNTS,
            "micro_row_counts": MICRO_COUNTS,
        },
        "teacher_forced_validation": teacher_forced,
        "behavioral_validation": behavioral,
        "snapshot_validation": micro,
    }
    write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
