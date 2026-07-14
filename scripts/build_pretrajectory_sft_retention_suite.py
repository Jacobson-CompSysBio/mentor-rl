#!/usr/bin/env python3
"""Build an evaluation-only, seen-fact/new-prompt stage-1 retention suite."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "pretrajectory_sft" / "v5_curriculum_patchcheck"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "artifacts" / "pretrajectory_sft_retention" / "seen_fact_heldout_rendering_v1"
)
SUITE_SCHEMA_VERSION = "pretrajectory-sft-retention-suite-v1"
EVALUATION_REGIME = "seen_fact_heldout_rendering"
DEFAULT_SAMPLES_PER_FAMILY = 32
REQUIRED_FAMILIES = (
    "entity_symbol_to_ensembl",
    "entity_ensembl_to_symbol",
    "ambiguous_alias_resolution",
)

QUESTION_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "entity_symbol_to_ensembl": (
        (
            "resolve_alias_v1",
            "For multiplex `{multiplex_id}`, resolve alias `{alias}` to its canonical graph entity. "
            "Use the learned normalization schema and return JSON only.",
        ),
        (
            "normalize_registry_v1",
            "Return the learned alias-registry normalization for `{alias}` in multiplex "
            "`{multiplex_id}` as JSON.",
        ),
        (
            "canonical_identity_v1",
            "Which canonical Ensembl graph identity does `{alias}` denote in `{multiplex_id}`? "
            "Return the complete normalization JSON object.",
        ),
    ),
    "entity_ensembl_to_symbol": (
        (
            "decode_graph_id_v1",
            "Decode canonical graph ID `{gene_id}` in multiplex `{multiplex_id}` into its learned "
            "display-symbol identity record. Return JSON only.",
        ),
        (
            "display_identity_v1",
            "For `{multiplex_id}`, return the display symbol and canonical entity associated with "
            "Ensembl graph ID `{gene_id}` as JSON.",
        ),
        (
            "reverse_normalization_v1",
            "Reproduce the learned reverse-normalization record for `{gene_id}` in "
            "`{multiplex_id}`. Output the complete JSON object.",
        ),
    ),
    "ambiguous_alias_resolution": (
        (
            "ambiguity_record_v1",
            "Alias `{alias}` is ambiguous in the learned registry for `{multiplex_id}`. Return its "
            "complete ambiguity record as JSON, including every candidate Ensembl ID.",
        ),
        (
            "disambiguation_policy_v1",
            "Before any graph lookup for ambiguous alias `{alias}`, reproduce the learned candidate "
            "set and disambiguation policy as JSON. Do not select a single candidate.",
        ),
        (
            "candidate_set_v1",
            "Return the learned unresolved-alias record for `{alias}` in `{multiplex_id}` as JSON, "
            "including status, exact candidates, required action, and allowed claim.",
        ),
    ),
}


class RetentionSuiteError(ValueError):
    """Raised when the source corpus cannot produce a contract-valid retention suite."""


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RetentionSuiteError(f"Expected a JSON object in {path}.")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RetentionSuiteError(f"Expected an object at {path}:{line_number}.")
            rows.append(payload)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, separators=(",", ":"), sort_keys=True))


def normalized_question(value: str) -> str:
    return " ".join(value.casefold().split())


def normalized_prompt(row: dict[str, Any]) -> str:
    return normalized_question(f"{row.get('system', '')}\n{row.get('question', '')}")


def load_canonical_objects(path: Path) -> dict[str, dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        object_id = row.get("object_id")
        if isinstance(object_id, str) and object_id:
            objects[object_id] = row
    return objects


def _source_payload(row: dict[str, Any], canonical: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise RetentionSuiteError("Stage-1 source row is missing metadata.")
    object_id = metadata.get("canonical_object_id") or metadata.get("oracle_fact_id")
    canonical_row = canonical.get(str(object_id))
    if not isinstance(canonical_row, dict) or not isinstance(canonical_row.get("payload"), dict):
        raise RetentionSuiteError(f"Missing canonical payload for source object {object_id!r}.")
    family = metadata.get("question_family")
    if canonical_row.get("object_id") != object_id:
        raise RetentionSuiteError(f"Canonical object ID mismatch for source object {object_id!r}.")
    if canonical_row.get("object_type") != family:
        raise RetentionSuiteError(
            f"Canonical object type mismatch for {object_id!r}: "
            f"{canonical_row.get('object_type')!r} != {family!r}."
        )
    return canonical_row["payload"]


def _hidden_targets(family: str, payload: dict[str, Any]) -> tuple[str, ...]:
    if family == "entity_symbol_to_ensembl":
        values = (payload.get("gene_id"),)
    elif family == "entity_ensembl_to_symbol":
        values = (payload.get("symbol"),)
    elif family == "ambiguous_alias_resolution":
        candidate_ids = payload.get("candidate_gene_ids")
        values = tuple(candidate_ids) if isinstance(candidate_ids, list) else ()
    else:  # Guarded by the public builder contract.
        values = ()
    targets = tuple(str(value) for value in values if isinstance(value, str) and value)
    if not targets:
        raise RetentionSuiteError(f"Canonical payload for {family!r} has no hidden target values.")
    return targets


def _contains_identifier(text: str, identifier: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _render_question(
    row: dict[str, Any],
    canonical: dict[str, dict[str, Any]],
    *,
    template_index: int,
) -> tuple[str, str]:
    metadata = row["metadata"]
    family = str(metadata["question_family"])
    payload = _source_payload(row, canonical)
    multiplex_id = str(metadata.get("multiplex_id") or "full_brain_multiplex_v1")
    template_id, template = QUESTION_TEMPLATES[family][template_index]
    values = {
        "multiplex_id": multiplex_id,
        "alias": payload.get("alias"),
        "gene_id": payload.get("gene_id"),
    }
    required_key = "gene_id" if family == "entity_ensembl_to_symbol" else "alias"
    if not isinstance(values[required_key], str) or not values[required_key]:
        raise RetentionSuiteError(
            f"Canonical payload for {metadata.get('record_id')} lacks {required_key!r}."
        )
    return template_id, template.format(**values)


def build_retention_suite(
    *,
    dataset_root: Path,
    out_dir: Path,
    seed: int,
    samples_per_family: int = DEFAULT_SAMPLES_PER_FAMILY,
    families: tuple[str, ...] = REQUIRED_FAMILIES,
) -> dict[str, Any]:
    if samples_per_family <= 0:
        raise RetentionSuiteError("samples_per_family must be positive.")
    unsupported = sorted(set(families) - set(REQUIRED_FAMILIES))
    if unsupported:
        raise RetentionSuiteError(f"Unsupported retention families: {unsupported}.")

    source_path = dataset_root / "curriculum" / "stage1_entity_schema" / "train.jsonl"
    canonical_path = dataset_root / "canonical_objects.jsonl"
    dataset_manifest_path = dataset_root / "manifest.json"
    required_paths = (source_path, canonical_path, dataset_manifest_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RetentionSuiteError(f"Missing retention-suite inputs: {missing}.")

    dataset_manifest = read_json(dataset_manifest_path)
    source_rows = read_jsonl(source_path)
    all_train_path = dataset_root / "train.jsonl"
    all_train_rows = read_jsonl(all_train_path) if all_train_path.is_file() else source_rows
    canonical = load_canonical_objects(canonical_path)
    source_record_ids: set[str] = set()
    all_training_prompts = {normalized_prompt(row) for row in all_train_rows}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise RetentionSuiteError("Stage-1 source row is missing metadata.")
        record_id = metadata.get("record_id")
        question = row.get("question")
        if not isinstance(record_id, str) or not record_id:
            raise RetentionSuiteError("Stage-1 source row is missing record_id.")
        if record_id in source_record_ids:
            raise RetentionSuiteError(f"Duplicate stage-1 source record_id: {record_id}.")
        if not isinstance(question, str) or not question:
            raise RetentionSuiteError(f"Stage-1 source row {record_id} is missing a question.")
        source_record_ids.add(record_id)
        family = metadata.get("question_family")
        if family in families:
            by_family[str(family)].append(row)

    missing_families = [family for family in families if not by_family.get(family)]
    if missing_families:
        raise RetentionSuiteError(f"Source stage is missing retention families: {missing_families}.")
    underfilled_families = {
        family: len(by_family[family])
        for family in families
        if len(by_family[family]) < samples_per_family
    }
    if underfilled_families:
        raise RetentionSuiteError(
            "Source stage cannot satisfy samples_per_family="
            f"{samples_per_family}: {underfilled_families}."
        )

    suite_rows: list[dict[str, Any]] = []
    template_counts: Counter[str] = Counter()
    target_leak_count = 0
    for family in families:
        ordered = sorted(
            by_family[family],
            key=lambda row: sha256_text(f"{seed}:{row['metadata']['record_id']}"),
        )[:samples_per_family]
        templates = QUESTION_TEMPLATES[family]
        for family_index, source_row in enumerate(ordered):
            source_metadata = source_row["metadata"]
            source_record_id = str(source_metadata["record_id"])
            template_index = family_index % len(templates)
            template_id, question = _render_question(
                source_row,
                canonical,
                template_index=template_index,
            )
            candidate_row = {**source_row, "question": question}
            normalized = normalized_prompt(candidate_row)
            if normalized in all_training_prompts:
                raise RetentionSuiteError(
                    f"Retention prompt duplicates a training prompt: {source_record_id}/{template_id}."
                )
            payload = _source_payload(source_row, canonical)
            leaked_targets = [
                target
                for target in _hidden_targets(family, payload)
                if _contains_identifier(question, target)
            ]
            if leaked_targets:
                target_leak_count += 1
                raise RetentionSuiteError(
                    f"Retention prompt leaks hidden targets for {source_record_id}/{template_id}: "
                    f"{leaked_targets}."
                )

            row = copy.deepcopy(source_row)
            metadata = row["metadata"]
            metadata.pop("answer_budget", None)
            metadata.pop("budget_measurement", None)
            new_record_id = "retention_" + sha256_text(
                f"{source_record_id}:{template_id}:{question}"
            )[:20]
            metadata.update(
                {
                    "record_id": new_record_id,
                    "split": "evaluation",
                    "curriculum_role": "evaluation_only",
                    "evaluation_regime": EVALUATION_REGIME,
                    "source_split": "train",
                    "source_record_id": source_record_id,
                    "source_oracle_fact_id": source_metadata.get("oracle_fact_id"),
                    "retention_template_id": template_id,
                    "official_readiness_eligible": False,
                    "source_question_sha256": sha256_text(str(source_row["question"])),
                    "question_sha256": sha256_text(question),
                    "source_answer_sha256": sha256_text(str(source_row["answer"])),
                }
            )
            row["question"] = question
            suite_rows.append(row)
            template_counts[template_id] += 1

    suite_rows.sort(key=lambda row: str(row["metadata"]["record_id"]))
    suite_record_ids = [str(row["metadata"]["record_id"]) for row in suite_rows]
    suite_questions = [normalized_prompt(row) for row in suite_rows]
    if len(set(suite_record_ids)) != len(suite_record_ids):
        raise RetentionSuiteError("Retention suite produced duplicate record IDs.")
    if len(set(suite_questions)) != len(suite_questions):
        raise RetentionSuiteError("Retention suite produced duplicate normalized questions.")

    suite_path = out_dir / "seen_fact_heldout_rendering.jsonl"
    manifest_path = out_dir / "suite_manifest.json"
    write_jsonl(suite_path, suite_rows)
    row_counts = Counter(str(row["metadata"]["question_family"]) for row in suite_rows)
    prompt_overlap_count = sum(question in all_training_prompts for question in suite_questions)
    manifest = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "evaluation_regime": EVALUATION_REGIME,
        "purpose": "intermediate_memorization_diagnostic",
        "official_readiness_eligible": False,
        "passed": prompt_overlap_count == 0 and target_leak_count == 0,
        "seed": seed,
        "selection": {
            "method": "sha256_rank_within_question_family",
            "samples_per_family": samples_per_family,
            "requested_families": list(families),
            "source_population_count_by_family": {
                family: len(by_family[family]) for family in families
            },
        },
        "source_dataset": {
            "dataset_schema_version": dataset_manifest.get("dataset_schema_version"),
            "plan_hash": dataset_manifest.get("plan_hash"),
            "content_hash": dataset_manifest.get("content_hash"),
            "stage": "stage1_entity_schema",
            "split": "train",
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "row_count": len(source_rows),
            "all_train_path": str(all_train_path.resolve()) if all_train_path.is_file() else None,
            "all_train_sha256": sha256_file(all_train_path) if all_train_path.is_file() else None,
            "all_train_row_count": len(all_train_rows),
        },
        "suite": {
            "path": str(suite_path.resolve()),
            "sha256": sha256_file(suite_path),
            "content_hash": stable_json_hash(suite_rows),
            "row_count": len(suite_rows),
            "row_count_by_family": dict(sorted(row_counts.items())),
            "template_count_by_id": dict(sorted(template_counts.items())),
            "required_families": list(families),
        },
        "audit": {
            "source_record_missing_count": 0,
            "duplicate_record_id_count": len(suite_record_ids) - len(set(suite_record_ids)),
            "duplicate_normalized_question_count": len(suite_questions) - len(set(suite_questions)),
            "source_prompt_overlap_count": prompt_overlap_count,
            "hidden_target_leak_count": target_leak_count,
            "answer_changed_count": 0,
            "oracle_fact_changed_count": 0,
        },
        "template_contract": {
            "template_set_hash": stable_json_hash(QUESTION_TEMPLATES),
            "one_heldout_rendering_per_source_fact": True,
            "facts_must_exist_in_stage1_train": True,
            "prompts_must_not_exist_in_stage1_train": True,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=DEFAULT_SAMPLES_PER_FAMILY,
        help="Deterministic number of seen source facts selected per mapping family.",
    )
    parser.add_argument(
        "--families",
        default=",".join(REQUIRED_FAMILIES),
        help="Comma-separated mapping families to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    manifest = build_retention_suite(
        dataset_root=args.dataset_root.resolve(),
        out_dir=args.out_dir.resolve(),
        seed=args.seed,
        samples_per_family=args.samples_per_family,
        families=families,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
