"""Prepare exact S0 train inputs and content-addressed receipts.

This module contains pure helpers for S0 train and inference programs. It
validates local artifacts, builds model-visible prompts, defines row order,
and records logical and physical exposure. It does not load model weights.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any


# These files can change the token IDs that the model receives.
TOKENIZER_ARTIFACT_NAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json",
    }
)

# Hugging Face can use several tokenizer and chat-template file names.
TOKENIZER_ARTIFACT_PREFIXES = ("chat_template", "tokenizer")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SHA256_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

S0_RECORD_METADATA_KEYS = (
    "schema_version",
    "book_mode",
    "step",
    "question_family",
    "species_taxon_id",
    "ensembl_release",
    "identifier_registry_id",
    "system_prompt_sha256",
    "answer_format",
)
S0_TOKENIZER_CODEC_KEY = "s0_tokenizer_codec"


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for one value.

    All content hashes use this exact byte representation. Sorted object keys
    prevent dictionary insertion order from changing an identity.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 digest for one canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file.

    The function reads fixed-size blocks, so a large model artifact does not
    enter memory as one object.
    """

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject another top-level JSON type."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return payload


def _require_sha256(value: Any, label: str) -> str:
    """Return one valid lowercase SHA-256 digest."""

    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_relative_path(value: Any, label: str) -> Path:
    """Return one safe relative artifact path.

    This check blocks absolute paths and parent traversal. It permits nested
    artifact paths and intentional symlinks inside a generated artifact tree.
    """

    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} has an unsafe file path: {value}")
    return relative


def validate_file_hashes(
    root: Path,
    expected: Any,
    *,
    label: str,
) -> None:
    """Check every declared file under one artifact root.

    The caller selects the correct root for each manifest section. This avoids
    a false assumption that corpus and evaluator files share one directory.
    """

    if not isinstance(expected, Mapping) or not expected:
        raise ValueError(f"{label} has no file hashes")
    for name, expected_hash in expected.items():
        relative = _safe_relative_path(name, label)
        required_hash = _require_sha256(
            expected_hash,
            f"{label} hash for {name}",
        )
        path = root / relative
        if not path.is_file():
            raise ValueError(f"{label} file is absent: {name}")
        if sha256_file(path) != required_hash:
            raise ValueError(f"{label} file identity changed: {name}")


def tokenizer_artifact_hashes(path: Path) -> dict[str, str]:
    """Return hashes for all tokenizer files in one local directory.

    The filter excludes model weights and unrelated configuration files. The
    result detects a tokenizer change before a train or evaluation job starts.
    """

    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"The tokenizer path is not a directory: {root}")
    artifacts = {
        item.name: sha256_file(item)
        for item in sorted(root.iterdir())
        if item.is_file()
        and (
            item.name in TOKENIZER_ARTIFACT_NAMES
            or item.name.startswith(TOKENIZER_ARTIFACT_PREFIXES)
        )
    }
    if not artifacts:
        raise ValueError(f"The tokenizer path has no tokenizer files: {root}")
    return artifacts


def validated_tokenizer_manifest(path: Path) -> dict[str, Any]:
    """Return one tokenizer manifest after its internal identity check.

    The internal digest covers every field except `manifest_sha256`. This check
    detects a changed method, codec reference, token list, or row count.
    """

    payload = read_json_object(path)
    claimed = _require_sha256(
        payload.get("manifest_sha256"),
        "tokenizer manifest_sha256",
    )
    identity = {
        str(key): value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if stable_sha256(identity) != claimed:
        raise ValueError("The tokenizer manifest failed its internal identity")
    return payload


# Block 2: Validate the complete v4 corpus and tokenizer identities.
S0_DATASET_ID = "world_model_v2_s0_human_identifiers_v4"
S0_MANIFEST_SCHEMA = "mentor-rl-world-model-s0-manifest-v4"
S0_SPLIT_MANIFEST_SCHEMA = "mentor-rl-world-model-s0-split-manifest-v4"
S0_EVALUATOR_MANIFEST_SCHEMA = (
    "mentor-rl-world-model-s0-evaluator-manifest-v4"
)
S0_TOKENIZER_ARM_SCHEMA = "mentor-rl-world-model-s0-tokenizer-arm-v4"
S0_TOKENIZER_AUDIT_SCHEMA = "mentor-rl-world-model-s0-tokenizer-audit-v4"
S0_TOKENIZER_MANIFEST_SCHEMA = "mentor-rl-world-model-s0-tokenizer-v3"
S0_RECORD_SCHEMA = "identifier_sft_v2"
S0_TRAINING_CONTRACT = "closed_book_only_v1"
S0_EXPOSURE_SCOPE_BY_RUN_SCOPE = {
    "debug_qualification": "bounded_debug_subset",
    "qualification": "all_eligible_train_rows",
    "matched_matrix": "all_eligible_train_rows",
}
S0_EVALUATION_CONTRACT = "seen_fact_closed_book_recall_v1"
S0_SYSTEM_PROMPT_SHA256 = (
    "06646540ea70f94d8cb8ca5fc9764980e0a8b251953d53ff2259c55962d5005b"
)
S0_TOKENIZER_METHODS = (
    "plain_base_tokenizer",
    "ordinary_domain_bpe",
    "atomic_plus_domain_bpe",
)
S0_QUESTION_FAMILIES = (
    "human_symbol_to_ensembl",
    "human_ensembl_to_symbol",
    "human_ambiguous_symbol",
)
S0_TOKEN_ROWS = {
    "plain_base_tokenizer": 0,
    "ordinary_domain_bpe": 482,
    "atomic_plus_domain_bpe": 481,
}


def _count_jsonl_rows(path: Path) -> int:
    """Count nonempty JSON Lines rows without loading the file."""

    with path.open(encoding="utf-8") as source_file:
        return sum(1 for line in source_file if line.strip())


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    """Reject one changed contract value."""

    if actual != expected:
        raise ValueError(
            f"{label} changed: expected={expected!r}, actual={actual!r}"
        )


def validate_s0_corpus_identity(
    corpus_root: Path,
    *,
    evaluator_manifest_path: Path,
    validation_answer_key_path: Path,
    expected_manifest_sha256: str,
    expected_train_sha256: str,
    expected_train_rows: int,
    expected_validation_sha256: str,
    expected_validation_rows: int,
    expected_validation_answer_key_sha256: str,
) -> dict[str, Any]:
    """Validate the complete v4 corpus identity at job start.

    The run contract pins the corpus manifest, train file, and row count. The
    corpus manifest then pins the split and evaluator manifests.

    The train process uses the private validation key for validation loss.
    This function verifies that key before the train process starts.
    """

    if expected_train_rows < 1:
        raise ValueError("expected_train_rows must be positive")
    required_manifest_hash = _require_sha256(
        expected_manifest_sha256,
        "expected corpus manifest SHA-256",
    )
    required_train_hash = _require_sha256(
        expected_train_sha256,
        "expected train SHA-256",
    )
    required_validation_hash = _require_sha256(
        expected_validation_sha256,
        "expected validation SHA-256",
    )
    required_validation_answer_hash = _require_sha256(
        expected_validation_answer_key_sha256,
        "expected validation answer key SHA-256",
    )
    if expected_validation_rows < 1:
        raise ValueError("expected_validation_rows must be positive")

    root = corpus_root.resolve()
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != required_manifest_hash:
        raise ValueError("The S0 corpus manifest identity changed")
    manifest = read_json_object(manifest_path)

    # These fixed values prevent an old S0 corpus from entering a v4 job.
    fixed_contract = {
        "schema_version": S0_MANIFEST_SCHEMA,
        "dataset_id": S0_DATASET_ID,
        "record_schema_version": S0_RECORD_SCHEMA,
        "training_contract": S0_TRAINING_CONTRACT,
        "evaluation_contract": S0_EVALUATION_CONTRACT,
    }
    for key, expected in fixed_contract.items():
        _require_equal(manifest.get(key), expected, f"S0 manifest {key}")

    prompt_contract = manifest.get("system_prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise ValueError("The S0 manifest has no system prompt contract")
    _require_equal(
        prompt_contract.get("system_prompt_sha256"),
        S0_SYSTEM_PROMPT_SHA256,
        "S0 system prompt SHA-256",
    )
    _require_equal(
        prompt_contract.get("allowed_book_modes"),
        ["closed_book"],
        "S0 allowed book modes",
    )

    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, Mapping):
        raise ValueError("The S0 manifest has no file hashes")
    local_hashes = {
        name: file_hashes.get(name)
        for name in ("train.jsonl", "val.jsonl", "split_manifest.json")
    }
    validate_file_hashes(root, local_hashes, label="The S0 corpus")
    _require_equal(
        file_hashes.get("train.jsonl"),
        required_train_hash,
        "S0 train SHA-256",
    )
    _require_equal(
        file_hashes.get("val.jsonl"),
        required_validation_hash,
        "S0 validation SHA-256",
    )

    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("The S0 manifest has no row counts")
    _require_equal(
        row_counts.get("train"),
        expected_train_rows,
        "S0 train row count",
    )
    _require_equal(
        _count_jsonl_rows(root / "train.jsonl"),
        expected_train_rows,
        "S0 train file row count",
    )
    _require_equal(
        _count_jsonl_rows(root / "val.jsonl"),
        expected_validation_rows,
        "S0 validation file row count",
    )
    _require_equal(
        row_counts.get("validation"),
        expected_validation_rows,
        "S0 validation row count",
    )

    # Distinguish source candidates from rows that full runs can expose.
    train_exclusions = manifest.get("train_exclusions")
    if not isinstance(train_exclusions, Mapping):
        raise ValueError("The S0 manifest has no train exclusions")
    exclusion_records = train_exclusions.get("records")
    if not isinstance(exclusion_records, list):
        raise ValueError("The S0 train exclusion records are invalid")
    excluded_rows = train_exclusions.get("row_count")
    if (
        not isinstance(excluded_rows, int)
        or isinstance(excluded_rows, bool)
        or excluded_rows != len(exclusion_records)
    ):
        raise ValueError("The S0 train exclusion row count is invalid")
    exclusion_ids = [
        record.get("record_id")
        for record in exclusion_records
        if isinstance(record, Mapping)
    ]
    if (
        len(exclusion_ids) != excluded_rows
        or any(
            not isinstance(value, str) or not value
            for value in exclusion_ids
        )
        or len(set(exclusion_ids)) != excluded_rows
    ):
        raise ValueError("The S0 train exclusion identities are invalid")

    train_population = manifest.get("train_population")
    expected_train_population = {
        "source_candidate_rows": expected_train_rows + excluded_rows,
        "excluded_rows": excluded_rows,
        "eligible_train_rows": expected_train_rows,
        "full_run_exposure_requirement": "all_eligible_train_rows",
    }
    _require_equal(
        train_population,
        expected_train_population,
        "S0 train population contract",
    )

    # The split manifest proves that all methods use the same fixed panels.
    split_manifest = read_json_object(root / "split_manifest.json")
    split_contract = {
        "schema_version": S0_SPLIT_MANIFEST_SCHEMA,
        "dataset_id": S0_DATASET_ID,
        "training_contract": S0_TRAINING_CONTRACT,
        "evaluation_contract": S0_EVALUATION_CONTRACT,
        "fact_role": "seen",
        "row_counts": dict(row_counts),
        "family_counts": manifest.get("family_counts"),
        "train_exclusions": dict(train_exclusions),
        "train_population": expected_train_population,
        "train_sha256": required_train_hash,
        "validation_questions_sha256": file_hashes.get("val.jsonl"),
    }
    for key, expected in split_contract.items():
        _require_equal(
            split_manifest.get(key),
            expected,
            f"S0 split manifest {key}",
        )

    # The corpus manifest stores this cross-directory file hash by file name.
    evaluator_path = evaluator_manifest_path.resolve()
    _require_equal(
        sha256_file(evaluator_path),
        file_hashes.get("evaluator_manifest.json"),
        "S0 evaluator manifest SHA-256",
    )
    evaluator_manifest = read_json_object(evaluator_path)
    evaluator_contract = {
        "schema_version": S0_EVALUATOR_MANIFEST_SCHEMA,
        "dataset_id": S0_DATASET_ID,
        "evaluation_contract": S0_EVALUATION_CONTRACT,
    }
    for key, expected in evaluator_contract.items():
        _require_equal(
            evaluator_manifest.get(key),
            expected,
            f"S0 evaluator manifest {key}",
        )
    validation = evaluator_manifest.get("validation")
    test = evaluator_manifest.get("test")
    if not isinstance(validation, Mapping) or not isinstance(test, Mapping):
        raise ValueError("The S0 evaluator manifest has incomplete panels")
    _require_equal(
        validation.get("questions_sha256"),
        file_hashes.get("val.jsonl"),
        "S0 validation panel SHA-256",
    )
    _require_equal(
        validation.get("row_count"),
        expected_validation_rows,
        "S0 validation panel row count",
    )
    _require_equal(
        validation.get("answer_key_sha256"),
        required_validation_answer_hash,
        "S0 validation answer key SHA-256",
    )
    answer_key_path = validation_answer_key_path.resolve()
    if not answer_key_path.is_file():
        raise ValueError("The S0 validation answer key is absent")
    _require_equal(
        sha256_file(answer_key_path),
        required_validation_answer_hash,
        "S0 validation answer key file SHA-256",
    )
    _require_equal(
        _count_jsonl_rows(answer_key_path),
        expected_validation_rows,
        "S0 validation answer key row count",
    )
    _require_equal(
        test.get("questions_sha256"),
        split_manifest.get("test_questions_sha256"),
        "S0 test panel SHA-256",
    )
    _require_equal(
        test.get("row_count"),
        row_counts.get("test"),
        "S0 test panel row count",
    )
    return manifest


def validate_s0_tokenizer_arm_identity(
    root: Path,
    *,
    method: str,
    expected_corpus_manifest_sha256: str,
    expected_train_sha256: str,
    expected_arm_manifest_sha256: str,
    maximum_sequence_tokens: int,
    tokenizer_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one complete v4 tokenizer method at job start.

    The check joins the tokenizer arm to the exact corpus. It also checks the
    audit receipt, token manifest, tokenizer bytes, and used model row count.
    """

    if method not in S0_TOKENIZER_METHODS:
        raise ValueError(f"Unsupported S0 tokenizer method: {method!r}")
    corpus_manifest_hash = _require_sha256(
        expected_corpus_manifest_sha256,
        "expected corpus manifest SHA-256",
    )
    train_hash = _require_sha256(
        expected_train_sha256,
        "expected train SHA-256",
    )
    arm_manifest_hash = _require_sha256(
        expected_arm_manifest_sha256,
        "expected tokenizer arm manifest SHA-256",
    )

    arm_root = root.resolve()
    manifest_path = arm_root / "manifest.json"
    if sha256_file(manifest_path) != arm_manifest_hash:
        raise ValueError("The S0 tokenizer arm manifest identity changed")
    manifest = read_json_object(manifest_path)
    _require_equal(
        manifest.get("schema_version"),
        S0_TOKENIZER_ARM_SCHEMA,
        "S0 tokenizer arm schema",
    )
    _require_equal(manifest.get("method"), method, "S0 tokenizer method")
    _require_equal(
        manifest.get("unused_model_rows_consumed"),
        S0_TOKEN_ROWS[method],
        "S0 used model row count",
    )
    _require_equal(
        manifest.get("audit_passed"),
        True,
        "S0 tokenizer audit status",
    )

    parent = manifest.get("parent_dataset")
    if not isinstance(parent, Mapping):
        raise ValueError("The S0 tokenizer arm has no parent dataset")
    parent_contract = {
        "dataset_id": S0_DATASET_ID,
        "manifest_sha256": corpus_manifest_hash,
        "train_sha256": train_hash,
    }
    for key, expected in parent_contract.items():
        _require_equal(
            parent.get(key),
            expected,
            f"S0 tokenizer parent {key}",
        )

    # Check every generated file before a loader opens any codec reference.
    validate_file_hashes(
        arm_root,
        manifest.get("file_hashes"),
        label="The S0 tokenizer arm",
    )
    audit = read_json_object(arm_root / "audit_report.json")
    audit_contract = {
        "schema_version": S0_TOKENIZER_AUDIT_SCHEMA,
        "passed": True,
        "method": method,
        "parent_manifest_sha256": corpus_manifest_hash,
        "parent_train_sha256": train_hash,
    }
    for key, expected in audit_contract.items():
        _require_equal(
            audit.get(key),
            expected,
            f"S0 tokenizer audit {key}",
        )

    # Reject a sequence limit below the exact audited train maximum.
    audited_sequence_tokens = audit.get("maximum_sequence_tokens")
    if (
        not isinstance(audited_sequence_tokens, int)
        or isinstance(audited_sequence_tokens, bool)
        or audited_sequence_tokens < 1
    ):
        raise ValueError(
            "The S0 tokenizer audit has no valid maximum sequence length"
        )
    if (
        not isinstance(maximum_sequence_tokens, int)
        or isinstance(maximum_sequence_tokens, bool)
        or maximum_sequence_tokens < audited_sequence_tokens
    ):
        raise ValueError(
            "The run maximum sequence length is below the tokenizer audit "
            f"maximum: {maximum_sequence_tokens} < "
            f"{audited_sequence_tokens}"
        )

    token_manifest_path = arm_root / "tokenizer_manifest.json"
    token_manifest = validated_tokenizer_manifest(token_manifest_path)
    token_manifest_contract = {
        "schema_version": S0_TOKENIZER_MANIFEST_SCHEMA,
        "method": method,
        "parent_manifest_sha256": corpus_manifest_hash,
        "parent_train_sha256": train_hash,
        "unused_model_rows_consumed": S0_TOKEN_ROWS[method],
    }
    for key, expected in token_manifest_contract.items():
        _require_equal(
            token_manifest.get(key),
            expected,
            f"S0 token manifest {key}",
        )
    _require_equal(
        token_manifest.get("manifest_sha256"),
        manifest.get("tokenizer_manifest_sha256"),
        "S0 token manifest identity",
    )

    # Compare the exact tokenizer bytes that the job will load.
    selected_tokenizer = (
        arm_root / "tokenizer"
        if tokenizer_path is None
        else tokenizer_path.resolve()
    )
    observed_artifacts = tokenizer_artifact_hashes(selected_tokenizer)
    declared_artifacts = manifest.get("tokenizer_artifact_hashes")
    if not isinstance(declared_artifacts, Mapping):
        raise ValueError("The S0 tokenizer arm has no tokenizer hashes")
    expected_artifacts = {
        Path(str(name)).name: _require_sha256(
            value,
            f"S0 tokenizer hash for {name}",
        )
        for name, value in declared_artifacts.items()
    }
    if len(expected_artifacts) != len(declared_artifacts):
        raise ValueError("The S0 tokenizer artifact names are not unique")
    _require_equal(
        observed_artifacts,
        expected_artifacts,
        "S0 selected tokenizer identity",
    )
    return manifest


def model_text_codec_key(token_manifest_path: Path) -> str | None:
    """Return the optional S0 codec key for one tokenizer manifest."""

    token_manifest = validated_tokenizer_manifest(token_manifest_path)
    _require_equal(
        token_manifest.get("schema_version"),
        S0_TOKENIZER_MANIFEST_SCHEMA,
        "S0 token manifest schema_version",
    )
    method = token_manifest.get("method")
    if method not in S0_TOKENIZER_METHODS:
        raise ValueError(f"Unsupported S0 tokenizer method: {method!r}")

    codec_reference = token_manifest.get(S0_TOKENIZER_CODEC_KEY)
    if method == "plain_base_tokenizer":
        if codec_reference is not None:
            raise ValueError("The plain tokenizer cannot declare an S0 codec")
        return None
    if not isinstance(codec_reference, Mapping):
        raise ValueError("A Domain-BPE tokenizer must declare an S0 codec")
    return S0_TOKENIZER_CODEC_KEY


def load_model_text_codec_for_token_manifest(token_manifest_path: Path):
    """Load the optional S0 text codec for one tokenizer manifest.

    The plain tokenizer returns no codec. Both Domain-BPE methods require the
    codec that their content-addressed token manifest references.
    """

    codec_key = model_text_codec_key(token_manifest_path)

    from runtime.world_model_s0_tokenizer import (
        load_s0_tokenizer_codec_for_token_manifest,
    )

    codec = load_s0_tokenizer_codec_for_token_manifest(token_manifest_path)
    if codec_key is None and codec is not None:
        raise ValueError("The plain tokenizer cannot declare an S0 codec")
    if codec_key is not None and codec is None:
        raise ValueError("A Domain-BPE tokenizer must declare an S0 codec")
    return codec


# Block 3: Build the exact v4 model prompt and flat train record.
def validate_s0_record_metadata(metadata: Any) -> dict[str, Any]:
    """Return validated S0 metadata for internal runtime use."""

    if not isinstance(metadata, Mapping):
        raise ValueError("S0 metadata must be one object")
    missing = [
        key for key in S0_RECORD_METADATA_KEYS if key not in metadata
    ]
    if missing:
        raise ValueError(f"S0 metadata has missing model fields: {missing}")

    fixed_contract = {
        "schema_version": S0_RECORD_SCHEMA,
        "book_mode": "closed_book",
        "step": "S0",
        "species_taxon_id": "NCBITaxon:9606",
        "ensembl_release": "Ensembl_116",
        "system_prompt_sha256": S0_SYSTEM_PROMPT_SHA256,
        "answer_format": "json",
    }
    for key, expected in fixed_contract.items():
        _require_equal(metadata.get(key), expected, f"S0 metadata {key}")

    family = metadata.get("question_family")
    if family not in S0_QUESTION_FAMILIES:
        raise ValueError(f"Unsupported S0 question family: {family!r}")
    registry_id = metadata.get("identifier_registry_id")
    if (
        not isinstance(registry_id, str)
        or SHA256_ID_PATTERN.fullmatch(registry_id) is None
    ):
        raise ValueError("S0 identifier_registry_id must be a SHA-256 ID")
    return {key: metadata[key] for key in S0_RECORD_METADATA_KEYS}


def serialize_sft_answer(answer: Any) -> str:
    """Return one compact S0 answer or one existing answer string."""

    if isinstance(answer, Mapping):
        return canonical_json(dict(answer))
    if isinstance(answer, str) and answer.strip():
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError as exc:
            raise ValueError("The S0 answer string must contain JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("The S0 answer string must contain one object")
        canonical_answer = canonical_json(dict(payload))
        if answer != canonical_answer:
            raise ValueError("The S0 answer string must use compact canonical JSON")
        return canonical_answer
    raise ValueError("The S0 answer must be one JSON object or a nonempty string")


def iter_s0_validation_records(
    question_path: Path,
    answer_key_path: Path,
):
    """Yield each validation question with its private answer."""

    answers: dict[str, dict[str, Any]] = {}
    with answer_key_path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                key_row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid validation answer row at {answer_key_path}:{line_number}"
                ) from error
            if not isinstance(key_row, Mapping):
                raise ValueError("A validation answer row must be one object")
            record_id = key_row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("A validation answer row lacks its record ID")
            if record_id in answers:
                raise ValueError(
                    f"The validation answer key repeats record ID {record_id}"
                )
            answers[record_id] = dict(key_row)

    seen: set[str] = set()
    with question_path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                question_row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid validation question at {question_path}:{line_number}"
                ) from error
            if not isinstance(question_row, Mapping):
                raise ValueError("A validation question must be one object")
            if "answer" in question_row:
                raise ValueError("A public validation question contains an answer")
            record_id = question_row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("A validation question lacks its record ID")
            if record_id in seen:
                raise ValueError(
                    f"The validation questions repeat record ID {record_id}"
                )
            key_row = answers.get(record_id)
            if key_row is None:
                raise ValueError(
                    f"The validation answer key lacks record ID {record_id}"
                )
            metadata = question_row.get("metadata")
            provenance = question_row.get("provenance")
            if not isinstance(metadata, Mapping) or not isinstance(
                provenance, Mapping
            ):
                raise ValueError("A validation question lacks its identity fields")
            identity_pairs = (
                ("family", key_row.get("family"), metadata.get("question_family")),
                ("fact_id", key_row.get("fact_id"), provenance.get("fact_id")),
                (
                    "fact_group_id",
                    key_row.get("fact_group_id"),
                    provenance.get("fact_group_id"),
                ),
            )
            for name, answer_value, question_value in identity_pairs:
                if answer_value != question_value:
                    raise ValueError(
                        f"The validation {name} differs for record ID {record_id}"
                    )
            merged = dict(question_row)
            merged["answer"] = key_row.get("answer")
            seen.add(record_id)
            yield merged

    unused = sorted(set(answers) - seen)
    if unused:
        raise ValueError(
            f"The validation answer key has {len(unused)} unmatched rows"
        )


def build_world_model_prompt_messages(
    *,
    system: str,
    question: str,
    metadata: Any,
    context: Any = None,
    in_context_examples: Any = None,
) -> list[dict[str, str]]:
    """Build the exact S0 system and user messages."""

    if not isinstance(system, str) or not system.strip():
        raise ValueError("The S0 system prompt must be a nonempty string")
    prompt_hash = hashlib.sha256(system.encode("utf-8")).hexdigest()
    _require_equal(
        prompt_hash,
        S0_SYSTEM_PROMPT_SHA256,
        "S0 system prompt SHA-256",
    )
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The S0 question must be a nonempty string")
    if context is not None:
        raise ValueError("S0 requires null context")
    validate_s0_record_metadata(metadata)
    if in_context_examples not in (None, "", []):
        raise ValueError("S0 does not permit in-context examples")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def flatten_sft_record_for_arrow(
    record: Mapping[str, Any],
    *,
    expected_split: str = "train",
) -> dict[str, str]:
    """Convert one v4 SFT record to a stable scalar schema."""

    if not isinstance(record, Mapping):
        raise ValueError("The S0 train record must be one object")
    metadata = record.get("metadata")
    validated_metadata = validate_s0_record_metadata(metadata)
    system = record.get("system")
    question = record.get("question")
    build_world_model_prompt_messages(
        system=system,
        question=question,
        metadata=metadata,
        context=record.get("context"),
    )
    prompt_form_id = {
        "train": "train",
        "val": "validation",
    }.get(expected_split)
    if prompt_form_id is None:
        raise ValueError(f"The S0 split is not supported: {expected_split!r}")
    if record.get("split") != expected_split:
        raise ValueError(
            f"The S0 adapter requires split {expected_split!r}"
        )
    if not isinstance(record.get("input"), Mapping):
        raise ValueError("The S0 train record requires one input object")

    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("The S0 train record requires one record_id")
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("The S0 train record requires provenance")
    fact_id = provenance.get("fact_id")
    if not isinstance(fact_id, str) or not fact_id.strip():
        raise ValueError("The S0 train record requires one fact_id")
    _require_equal(
        provenance.get("fact_role"),
        "seen",
        "S0 train fact_role",
    )
    _require_equal(
        provenance.get("prompt_form_id"),
        prompt_form_id,
        "S0 prompt_form_id",
    )
    rendering_index = provenance.get("rendering_index")
    if not isinstance(rendering_index, int) or rendering_index < 0:
        raise ValueError("S0 rendering_index must be a nonnegative integer")

    return {
        "system": system,
        "question": question,
        "answer": serialize_sft_answer(record.get("answer")),
        "metadata_json": canonical_json(validated_metadata),
        "question_family": str(validated_metadata["question_family"]),
        "record_id": record_id,
        "fact_id": fact_id,
        "prompt_form_id": prompt_form_id,
        "split": expected_split,
    }


def normalize_token_ids(value: Any) -> list[int]:
    """Return one flat token sequence from one tokenizer result."""

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("The tokenizer result has no input_ids field")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Tokenizer input_ids must be one sequence")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected one token sequence, but received a batch")
        value = value[0]
    if any(isinstance(item, (list, tuple, Mapping)) for item in value):
        raise ValueError("Tokenizer input_ids must be one flat sequence")
    token_ids = [int(item) for item in value]
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("Tokenizer input_ids cannot contain negative values")
    return token_ids


# Block 4: Define the exact logical and physical row order.
def epoch_training_indices(
    dataset_size: int,
    seed: int,
    preserve_order: bool,
) -> list[int]:
    """Return one explicit epoch order."""

    if (
        not isinstance(dataset_size, int)
        or isinstance(dataset_size, bool)
        or dataset_size < 1
    ):
        raise ValueError("dataset_size must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_order, bool):
        raise ValueError("preserve_order must be a Boolean")
    indices = list(range(dataset_size))
    if not preserve_order:
        random.Random(seed).shuffle(indices)
    return indices


def epoch_training_indices_with_replica_padding(
    dataset_size: int,
    seed: int,
    preserve_order: bool,
    replica_count: int,
) -> tuple[list[int], list[int]]:
    """Add one local repeat for each short strided replica shard."""

    if (
        not isinstance(replica_count, int)
        or isinstance(replica_count, bool)
        or replica_count < 1
    ):
        raise ValueError("replica_count must be positive")
    order = epoch_training_indices(dataset_size, seed, preserve_order)
    remainder = dataset_size % replica_count
    if remainder == 0:
        return order, []
    if dataset_size < replica_count:
        raise ValueError(
            "dataset_size must be at least replica_count when padding is required"
        )
    padding = order[remainder:replica_count]
    expected_padding = replica_count - remainder
    if len(padding) != expected_padding:
        raise RuntimeError(
            f"Replica padding produced {len(padding)} rows, "
            f"but expected {expected_padding}"
        )
    return order + padding, padding


def consumed_training_index_plan(
    dataset_size: int,
    *,
    total_steps: int,
    num_train_epochs: int,
    replica_count: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    seed: int,
    preserve_order: bool,
) -> dict[str, Any]:
    """Separate logical exposure from distributed padding."""

    for name, value in (
        ("dataset_size", dataset_size),
        ("total_steps", total_steps),
        ("num_train_epochs", num_train_epochs),
        ("replica_count", replica_count),
        ("per_device_train_batch_size", per_device_train_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_order, bool):
        raise ValueError("preserve_order must be a Boolean")

    logical_indices: list[int] = []
    padding_indices: list[int] = []
    completed_steps = 0
    steps_per_epoch = 0
    padded_rows_per_epoch = 0
    for epoch in range(num_train_epochs):
        padded_order, epoch_padding = epoch_training_indices_with_replica_padding(
            dataset_size,
            seed + epoch,
            preserve_order,
            replica_count,
        )
        per_replica_rows = len(padded_order) // replica_count
        rows_per_update = (
            per_device_train_batch_size * gradient_accumulation_steps
        )
        steps_per_epoch = math.ceil(per_replica_rows / rows_per_update)
        padded_rows_per_epoch = len(padded_order)
        steps_in_epoch = min(
            steps_per_epoch,
            total_steps - completed_steps,
        )
        physical_limit = min(
            padded_rows_per_epoch,
            steps_in_epoch * replica_count * rows_per_update,
        )
        logical_limit = min(dataset_size, physical_limit)
        logical_indices.extend(padded_order[:logical_limit])
        padding_limit = max(0, physical_limit - dataset_size)
        padding_indices.extend(epoch_padding[:padding_limit])
        completed_steps += steps_in_epoch
        if completed_steps == total_steps:
            return {
                "logical_indices": logical_indices,
                "distributed_padding_indices": padding_indices,
                "completed_steps": completed_steps,
                "steps_per_epoch": steps_per_epoch,
                "logical_rows_per_epoch": dataset_size,
                "padded_rows_per_epoch": padded_rows_per_epoch,
                "padding_rows_per_epoch": (
                    padded_rows_per_epoch - dataset_size
                ),
                "physical_record_occurrences": (
                    len(logical_indices) + len(padding_indices)
                ),
                "padding_policy": (
                    "repeat_first_local_item_for_short_strided_replica_shards"
                ),
            }
    raise ValueError(
        f"total_steps={total_steps} exceeds the {completed_steps} updates "
        f"available within num_train_epochs={num_train_epochs}"
    )


def _sequence_sha256(values: list[str]) -> str:
    """Return the identity of one ordered string sequence."""

    return stable_sha256(values)


def s0_exposure_scope(run_scope: str) -> str:
    """Return the exposure scope for one validated S0 run scope."""

    try:
        return S0_EXPOSURE_SCOPE_BY_RUN_SCOPE[run_scope]
    except KeyError as error:
        raise ValueError(
            f"The S0 run scope is invalid: {run_scope!r}"
        ) from error


def build_training_exposure_manifest(
    *,
    run_id: str,
    method_id: str,
    run_config_sha256: str,
    corpus_manifest_sha256: str,
    train_sha256: str,
    tokenizer_arm_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    record_ids: list[str],
    fact_ids: list[str],
    question_families: list[str],
    prompt_form_ids: list[str],
    consumed_indices: list[int],
    distributed_padding_indices: list[int],
    seed: int,
    total_steps: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    data_parallel_size: int,
    preserve_order: bool,
    padding_policy: str,
    exposure_scope: str,
    status: str,
    completed_global_step: int | None = None,
) -> dict[str, Any]:
    """Build one compact content-addressed S0 exposure receipt."""

    for label, value in (("run_id", run_id), ("method_id", method_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string")
    identities = {
        "run_config_sha256": run_config_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "train_sha256": train_sha256,
        "tokenizer_arm_manifest_sha256": tokenizer_arm_manifest_sha256,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
    }
    for label, value in identities.items():
        _require_sha256(value, label)

    row_count = len(record_ids)
    if row_count < 1:
        raise ValueError("The S0 exposure receipt requires train rows")
    parallel_values = (
        fact_ids,
        question_families,
        prompt_form_ids,
    )
    if any(len(values) != row_count for values in parallel_values):
        raise ValueError("The S0 train identity lists have different lengths")
    if len(set(record_ids)) != row_count:
        raise ValueError("S0 train record_ids must be unique")
    if any(not isinstance(value, str) or not value for value in record_ids):
        raise ValueError("Each S0 record_id must be a nonempty string")
    if any(not isinstance(value, str) or not value for value in fact_ids):
        raise ValueError("Each S0 fact_id must be a nonempty string")
    if any(family not in S0_QUESTION_FAMILIES for family in question_families):
        raise ValueError("The S0 exposure receipt has an unknown family")
    if set(prompt_form_ids) != {"train"}:
        raise ValueError("The S0 exposure receipt requires train prompt forms")

    for name, value in (
        ("total_steps", total_steps),
        ("num_train_epochs", num_train_epochs),
        ("per_device_train_batch_size", per_device_train_batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("data_parallel_size", data_parallel_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(preserve_order, bool):
        raise ValueError("preserve_order must be a Boolean")
    if status not in {"planned", "complete"}:
        raise ValueError("status must be 'planned' or 'complete'")
    if status == "planned" and completed_global_step is not None:
        raise ValueError("A planned receipt cannot have a completed step")
    if status == "complete" and completed_global_step != total_steps:
        raise ValueError("A complete receipt must match total_steps")
    if not isinstance(padding_policy, str) or not padding_policy:
        raise ValueError("padding_policy must be a nonempty string")
    if exposure_scope not in set(S0_EXPOSURE_SCOPE_BY_RUN_SCOPE.values()):
        raise ValueError("The S0 exposure scope is invalid")

    expected_plan = consumed_training_index_plan(
        row_count,
        total_steps=total_steps,
        num_train_epochs=num_train_epochs,
        replica_count=data_parallel_size,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        preserve_order=preserve_order,
    )
    _require_equal(
        consumed_indices,
        expected_plan["logical_indices"],
        "S0 logical row plan",
    )
    _require_equal(
        distributed_padding_indices,
        expected_plan["distributed_padding_indices"],
        "S0 distributed padding plan",
    )
    _require_equal(
        padding_policy,
        expected_plan["padding_policy"],
        "S0 distributed padding policy",
    )

    for label, indices in (
        ("consumed_indices", consumed_indices),
        ("distributed_padding_indices", distributed_padding_indices),
    ):
        if any(
            not isinstance(index, int) or index < 0 or index >= row_count
            for index in indices
        ):
            raise ValueError(f"{label} must reference S0 train rows")
    if not consumed_indices:
        raise ValueError("consumed_indices must reference S0 train rows")
    all_eligible_rows_exposed = (
        set(consumed_indices) == set(range(row_count))
    )
    if (
        exposure_scope == "all_eligible_train_rows"
        and not all_eligible_rows_exposed
    ):
        raise ValueError(
            "The S0 full row plan must expose every eligible train row"
        )
    if (
        exposure_scope == "bounded_debug_subset"
        and all_eligible_rows_exposed
    ):
        raise ValueError(
            "The S0 debug row plan must stop before full exposure"
        )

    consumed_record_ids = [record_ids[index] for index in consumed_indices]
    consumed_fact_ids = [fact_ids[index] for index in consumed_indices]
    consumed_families = [
        question_families[index] for index in consumed_indices
    ]
    padding_record_ids = [
        record_ids[index] for index in distributed_padding_indices
    ]
    padding_fact_ids = [
        fact_ids[index] for index in distributed_padding_indices
    ]
    padding_families = [
        question_families[index] for index in distributed_padding_indices
    ]
    physical_record_ids = consumed_record_ids + padding_record_ids
    physical_fact_ids = consumed_fact_ids + padding_fact_ids
    physical_families = consumed_families + padding_families
    payload: dict[str, Any] = {
        "schema_version": "mentor-rl-s0-training-exposure-v2",
        "status": status,
        "run_id": run_id,
        "method_id": method_id,
        "identity": identities,
        "corpus": {
            "dataset_id": S0_DATASET_ID,
            "eligible_train_rows": row_count,
            "record_sequence_sha256": _sequence_sha256(record_ids),
            "fact_sequence_sha256": _sequence_sha256(fact_ids),
            "question_family_counts": dict(
                sorted(Counter(question_families).items())
            ),
            "prompt_form_id": "train",
            "system_prompt_sha256": S0_SYSTEM_PROMPT_SHA256,
        },
        "schedule": {
            "seed": seed,
            "total_steps": total_steps,
            "completed_global_step": completed_global_step,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "data_parallel_size": data_parallel_size,
            "global_batch_size": (
                per_device_train_batch_size
                * gradient_accumulation_steps
                * data_parallel_size
            ),
            "preserve_order": preserve_order,
            "order_strategy": (
                "source_order_each_epoch"
                if preserve_order
                else "python_seeded_epoch_shuffle"
            ),
        },
        "exposure_contract": {
            "scope": exposure_scope,
            "all_eligible_train_rows_required": (
                exposure_scope == "all_eligible_train_rows"
            ),
            "satisfied": True,
        },
        "logical_exposure": {
            "record_occurrences": len(consumed_record_ids),
            "unique_record_count": len(set(consumed_record_ids)),
            "unique_fact_count": len(set(consumed_fact_ids)),
            "all_eligible_train_rows_exposed": (
                all_eligible_rows_exposed
            ),
            "record_sequence_sha256": _sequence_sha256(
                consumed_record_ids
            ),
            "fact_sequence_sha256": _sequence_sha256(consumed_fact_ids),
            "question_family_counts": dict(
                sorted(Counter(consumed_families).items())
            ),
        },
        "physical_exposure": {
            "record_occurrences": len(physical_record_ids),
            "record_sequence_sha256": _sequence_sha256(
                physical_record_ids
            ),
            "fact_sequence_sha256": _sequence_sha256(physical_fact_ids),
            "question_family_counts": dict(
                sorted(Counter(physical_families).items())
            ),
            "distributed_padding": {
                "policy": padding_policy,
                "included_in_logical_exposure": False,
                "record_occurrences": len(padding_record_ids),
                "record_ids": padding_record_ids,
                "fact_ids": padding_fact_ids,
                "record_sequence_sha256": _sequence_sha256(
                    padding_record_ids
                ),
                "fact_sequence_sha256": _sequence_sha256(
                    padding_fact_ids
                ),
                "question_family_counts": dict(
                    sorted(Counter(padding_families).items())
                ),
            },
        },
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    return payload
