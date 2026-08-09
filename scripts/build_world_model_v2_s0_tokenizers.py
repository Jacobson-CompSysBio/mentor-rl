#!/usr/bin/env python3
"""Build and audit all three S0 v4 tokenizer methods.

The script validates the fixed corpus and both GPT-OSS tokenizer identities.
It creates each tokenizer arm and tests all train rows.
It uses a temporary directory until all three arms pass their audits.
The script does not change model weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_domain_bpe import (  # noqa: E402
    DOMAIN_BPE_SCHEMA_VERSION,
    DOMAIN_BPE_VOCAB_SIZE,
)
from runtime.world_model_s0 import validate_s0_contract  # noqa: E402
from runtime.world_model_s0_tokenizer import (  # noqa: E402
    ATOMIC_PREFIX_CONTRACT,
    S0_CODEC_MANIFEST,
    S0_CODEC_SCHEMA_VERSION,
    load_s0_tokenizer_codec_for_token_manifest,
)
from runtime.world_model_schemas import WorldModelRecord  # noqa: E402


# These schema names identify each JSON contract that this script writes.
CONFIG_SCHEMA = "mentor-rl-world-model-s0-tokenizer-matrix-v4"
ARM_MANIFEST_SCHEMA = "mentor-rl-world-model-s0-tokenizer-arm-v4"
ARM_AUDIT_SCHEMA = "mentor-rl-world-model-s0-tokenizer-audit-v4"
TOKEN_MANIFEST_SCHEMA = "mentor-rl-world-model-s0-tokenizer-v3"
PAIR_MANIFEST_SCHEMA = "mentor-rl-world-model-s0-tokenizer-pair-v4"
COMPARISON_REPORT_SCHEMA = "mentor-rl-world-model-s0-tokenizer-comparison-v4"
# This order controls the arm order in each result and hashable manifest.
METHODS = (
    "plain_base_tokenizer",
    "ordinary_domain_bpe",
    "atomic_plus_domain_bpe",
)
# Each tuple gives a namespace, a file slug, and base text for row initialization.
NAMESPACES = (
    ("ensembl_human_gene", "s0ens", "ENSG"),
    ("human_gene_symbol", "s0sym", "human gene symbol"),
)
# These field names tell the fit scan which identifier namespace to use.
ENSEMBL_FIELDS = frozenset({"gene_id", "candidate_gene_ids"})
SYMBOL_FIELDS = frozenset({"gene_symbol", "gene_symbols"})
# Each arm copies these files from the selected GPT-OSS model.
TOKENIZER_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
# The command uses the repository contract unless the user supplies another path.
DEFAULT_CONFIG = (
    REPO_ROOT / "config/world_model_v2_s0_tokenizer_matrix_v4.json"
)


# Block 1: Read canonical files and validate the fixed build contract.
def canonical_json(value: Any) -> str:
    """Return compact canonical JSON for hashes and train targets."""

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
    """Return the SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected one JSON object: {path}")
    return payload


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one readable canonical JSON object."""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_repo_path(value: Any) -> Path:
    """Resolve one repository-relative config path."""

    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require(condition: bool, message: str) -> None:
    """Fail the build when one fixed contract condition is false."""

    if not condition:
        raise ValueError(message)


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the complete tokenizer matrix contract."""

    config = read_json(path)
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")

    # The build contract fixes the method order and both BPE namespaces.
    build = config.get("tokenizer_build")
    _require(isinstance(build, Mapping), "tokenizer_build must be an object")
    _require(
        tuple(build.get("methods", {})) == METHODS,
        "tokenizer method order changed",
    )
    _require(
        build.get("vocab_per_namespace") == DOMAIN_BPE_VOCAB_SIZE,
        "Domain-BPE vocabulary size changed",
    )
    _require(
        tuple(build.get("namespaces", []))
        == tuple(namespace for namespace, _, _ in NAMESPACES),
        "Domain-BPE namespace order changed",
    )

    # The 20B qualification must contain each method once with LoRA r32.
    qualification = config.get("qualification_20b")
    _require(isinstance(qualification, list), "qualification_20b must be a list")
    _require(len(qualification) == 3, "qualification_20b must have three methods")
    _require(
        {row.get("tokenizer_method") for row in qualification} == set(METHODS),
        "qualification_20b tokenizer methods changed",
    )
    _require(
        all(row.get("fine_tune_configuration") == "lora_r32" for row in qualification),
        "qualification_20b must use LoRA r32",
    )

    # The 120B matrix must contain every tokenizer and fine-tune combination.
    matrix = config.get("matrix_120b")
    fine_tunes = config.get("fine_tune_configurations")
    _require(isinstance(matrix, list), "matrix_120b must be a list")
    _require(isinstance(fine_tunes, Mapping), "fine_tune configurations are absent")
    observed = {
        (row.get("tokenizer_method"), row.get("fine_tune_configuration"))
        for row in matrix
    }
    expected = set(product(METHODS, tuple(fine_tunes)))
    _require(observed == expected and len(matrix) == 12, "120B matrix is incomplete")
    method_ids = [
        row.get("method_id")
        for row in qualification + matrix
    ]
    # Unique method IDs prevent two result sets from using the same name.
    _require(
        all(isinstance(value, str) and value for value in method_ids),
        "a matrix method ID is empty",
    )
    _require(len(method_ids) == len(set(method_ids)), "matrix method IDs repeat")
    return config


def _import_tokenizer_types():
    """Import the pinned Rust-backed tokenizer types."""

    try:
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
    except ImportError as error:
        raise RuntimeError(
            "The S0 tokenizer builder requires the tokenizers package"
        ) from error
    return AddedToken, Tokenizer, BPE, BpeTrainer


def _compile_chat_template(path: Path):
    """Compile the pinned chat template without a PyTorch import."""

    try:
        from datetime import datetime
        import jinja2
        from jinja2.ext import LoopControlExtension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:
        raise RuntimeError(
            "The S0 tokenizer audit requires the jinja2 package"
        ) from error

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(
        value: Any,
        ensure_ascii: bool = False,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
    ) -> str:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    def strftime_now(format_string: str) -> str:
        return datetime.now().strftime(format_string)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[LoopControlExtension],
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = strftime_now
    return environment.from_string(path.read_text(encoding="utf-8"))


def _tokenizer_artifact_hashes(root: Path) -> dict[str, str]:
    """Return all tokenizer file hashes below one arm root."""

    # These hashes identify only the files that a tokenizer loader uses.
    tokenizer_root = root / "tokenizer"
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(tokenizer_root.iterdir())
        if path.is_file()
    }


def _tree_file_hashes(root: Path) -> dict[str, str]:
    """Return hashes for every completed arm file except its outer manifest."""

    # Exclude the outer manifest because it stores this hash map.
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


# Block 2: Pin both base tokenizer identities and the complete train corpus.
def validate_base_models(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate both GPT-OSS base tokenizers and return their shared identity."""

    _, Tokenizer, _, _ = _import_tokenizer_types()
    base_models = config.get("base_models")
    _require(isinstance(base_models, Mapping), "base_models must be an object")
    _require(set(base_models) == {"oss20b", "oss120b"}, "base model set changed")
    shared_artifacts: dict[str, str] | None = None
    shared_shape: tuple[int, int, int] | None = None
    for model_size in ("oss20b", "oss120b"):
        # Check each model against its pinned config and tokenizer hashes.
        contract = base_models[model_size]
        _require(isinstance(contract, Mapping), f"{model_size} contract is invalid")
        root = resolve_repo_path(contract.get("path"))
        _require(root.is_dir(), f"{model_size} model path is absent: {root}")
        _require(
            sha256_file(root / "config.json") == contract.get("config_sha256"),
            f"{model_size} config identity changed",
        )
        expected_artifacts = contract.get("tokenizer_artifact_hashes")
        _require(
            isinstance(expected_artifacts, Mapping),
            f"{model_size} tokenizer hashes are absent",
        )
        observed_artifacts = {
            name: sha256_file(root / name)
            for name in TOKENIZER_FILES
        }
        _require(
            observed_artifacts == dict(expected_artifacts),
            f"{model_size} tokenizer identity changed",
        )
        tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        base_rows = tokenizer.get_vocab_size(with_added_tokens=True)
        model_vocab = int(read_json(root / "config.json")["vocab_size"])

        # GPT-OSS reserves model rows after the rows that the base tokenizer uses.
        shape = (base_rows, model_vocab, model_vocab - base_rows)
        _require(
            shape
            == (
                contract.get("base_tokenizer_rows"),
                contract.get("model_vocab_size"),
                contract.get("unused_model_rows"),
            ),
            f"{model_size} tokenizer row contract changed",
        )
        if shared_artifacts is None:
            # Use the 20B values as the shared identity for this build.
            shared_artifacts = observed_artifacts
            shared_shape = shape
        else:
            # Both model sizes must expose the same tokenizer bytes and row shape.
            _require(
                observed_artifacts == shared_artifacts,
                "20B and 120B tokenizer bytes differ",
            )
            _require(shape == shared_shape, "20B and 120B tokenizer shapes differ")
    assert shared_artifacts is not None and shared_shape is not None
    return {
        "tokenizer_artifact_hashes": shared_artifacts,
        "base_tokenizer_rows": shared_shape[0],
        "model_vocab_size": shared_shape[1],
        "unused_model_rows": shared_shape[2],
    }


def validate_selected_model(
    model_root: Path,
    shared_identity: Mapping[str, Any],
) -> None:
    """Validate the model path that supplies tokenizer files for the build."""

    # A command-line override must match the shared identity from the config.
    _, Tokenizer, _, _ = _import_tokenizer_types()
    observed = {
        name: sha256_file(model_root / name)
        for name in TOKENIZER_FILES
    }
    _require(
        observed == shared_identity.get("tokenizer_artifact_hashes"),
        "selected base tokenizer identity changed",
    )
    tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
    _require(
        tokenizer.get_vocab_size(with_added_tokens=True)
        == shared_identity.get("base_tokenizer_rows"),
        "selected base tokenizer row count changed",
    )
    _require(
        read_json(model_root / "config.json").get("vocab_size")
        == shared_identity.get("model_vocab_size"),
        "selected model vocabulary size changed",
    )


def validate_corpus(
    config: Mapping[str, Any],
    parent_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Validate the pinned v4 corpus and return its train path."""

    # File hashes bind this tokenizer artifact to one exact corpus artifact.
    contract = config.get("corpus")
    _require(isinstance(contract, Mapping), "corpus contract is absent")
    manifest_path = parent_root / "manifest.json"
    train_path = parent_root / "train.jsonl"
    _require(
        sha256_file(manifest_path) == contract.get("manifest_sha256"),
        "corpus manifest identity changed",
    )
    _require(
        sha256_file(train_path) == contract.get("train_sha256"),
        "train file identity changed",
    )
    manifest = read_json(manifest_path)
    # These fields protect the data split and the evaluation boundary.
    _require(
        manifest.get("dataset_id") == contract.get("dataset_id"),
        "corpus dataset ID changed",
    )
    _require(
        manifest.get("training_contract") == contract.get("training_contract"),
        "corpus train contract changed",
    )
    _require(
        manifest.get("evaluation_contract") == contract.get("evaluation_contract"),
        "corpus evaluation contract changed",
    )
    _require(
        manifest.get("row_counts", {}).get("train") == contract.get("train_rows"),
        "corpus train row count changed",
    )
    _require(
        manifest.get("identifier_registry", {}).get("id")
        == contract.get("identifier_registry_id"),
        "identifier registry identity changed",
    )
    with train_path.open(encoding="utf-8") as source_file:
        observed_rows = sum(1 for line in source_file if line.strip())
    # Check the physical line count after the manifest checks pass.
    _require(observed_rows == contract.get("train_rows"), "train file row count changed")
    return manifest, train_path


def _collect_typed_values(
    value: Any,
    result: dict[str, set[str]],
    field_name: str | None = None,
) -> None:
    """Collect complete typed values from one input or answer object."""

    if isinstance(value, Mapping):
        # Use each object key as the type of its nested value.
        for key, item in value.items():
            _collect_typed_values(item, result, str(key))
        return
    if isinstance(value, list):
        # Keep the parent field type for each list item.
        for item in value:
            _collect_typed_values(item, result, field_name)
        return
    if not isinstance(value, str) or field_name is None:
        return

    # Add only complete identifiers from fields in the fixed schema.
    if field_name in ENSEMBL_FIELDS:
        result["ensembl_human_gene"].add(value)
    if field_name in SYMBOL_FIELDS:
        result["human_gene_symbol"].add(value)


def read_and_validate_train_rows(
    train_path: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Validate each train row and collect all tokenizer fit values."""

    rows: list[dict[str, Any]] = []
    fit_values = {namespace: set() for namespace, _, _ in NAMESPACES}
    with train_path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected one object at {train_path}:{line_number}")

            # Apply both the general record schema and the S0 data contract.
            record = WorldModelRecord.from_dict(payload)
            errors = validate_s0_contract(record)
            if errors:
                raise ValueError(
                    f"S0 train row {line_number} failed its contract: {errors}"
                )
            _require(record.split == "train", f"row {line_number} is not train")
            _collect_typed_values(record.input, fit_values)
            _collect_typed_values(record.answer, fit_values)
            rows.append(payload)

    # The contract pins both the row count and each unique fit-value count.
    corpus_contract = config["corpus"]
    _require(len(rows) == corpus_contract["train_rows"], "loaded train row count changed")
    expected_counts = config["tokenizer_build"].get("fit_value_counts")
    _require(isinstance(expected_counts, Mapping), "fit value counts are absent")
    observed_counts = {
        namespace: len(values)
        for namespace, values in fit_values.items()
    }
    _require(observed_counts == dict(expected_counts), "fit value counts changed")
    return rows, fit_values


# Block 3: Fit both BPE namespaces and create trainable model token rows.
def _base_source_ids(tokenizer: Any, text: str) -> list[int]:
    """Return base tokenizer IDs for one new-row initialization source."""

    # The train code uses the mean of these base rows for each new row.
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    if not ids:
        raise RuntimeError(f"The base tokenizer returned no IDs for {text!r}")
    return [int(value) for value in ids]


def fit_domain_bpe(
    root: Path,
    fit_values: Mapping[str, set[str]],
    *,
    method: str,
    parent_manifest_sha256: str,
    parent_train_sha256: str,
) -> dict[str, Any]:
    """Fit two exact 240-piece BPE models and write their manifest."""

    _, Tokenizer, BPE, BpeTrainer = _import_tokenizer_types()
    domain_root = root / "domain_bpe"
    domain_root.mkdir()
    namespaces: dict[str, Any] = {}
    for namespace, slug, _ in NAMESPACES:
        # Sort the values so repeated builds produce the same BPE model.
        canonical_values = sorted(fit_values[namespace])
        _require(bool(canonical_values), f"no fit values for {namespace}")
        value_prefix = ""
        values = canonical_values
        if method == "atomic_plus_domain_bpe" and namespace == "ensembl_human_gene":
            # The atomic token represents ENSG, so this BPE fits numeric suffixes.
            _require(
                all(
                    value.startswith("ENSG")
                    and len(value) == 15
                    and value[4:].isdigit()
                    for value in canonical_values
                ),
                "atomic Ensembl values are not canonical",
            )
            value_prefix = "ENSG"
            values = [value[4:] for value in canonical_values]
            _require(len(values) == len(set(values)), "Ensembl suffixes repeat")

        # Fit one independent BPE model with exactly 240 pieces.
        tokenizer = Tokenizer(BPE(unk_token=None))
        trainer = BpeTrainer(
            vocab_size=DOMAIN_BPE_VOCAB_SIZE,
            min_frequency=2,
            initial_alphabet=sorted(set("".join(values))),
            show_progress=False,
        )
        tokenizer.train_from_iterator(values, trainer=trainer, length=len(values))
        vocabulary = tokenizer.get_vocab()
        _require(
            len(vocabulary) == DOMAIN_BPE_VOCAB_SIZE,
            f"{namespace} Domain-BPE does not have 240 pieces",
        )

        # Require every fit value to reconstruct exactly from its BPE pieces.
        for value in values:
            pieces = tokenizer.encode(value, add_special_tokens=False).tokens
            _require(
                bool(pieces) and "".join(pieces) == value,
                f"{namespace} Domain-BPE failed an exact fit-value cycle",
            )
        tokenizer_file = domain_root / f"{slug}.json"
        tokenizer.save(str(tokenizer_file))

        # Give each BPE piece a stable marker for the GPT-OSS tokenizer.
        pieces = [
            {
                "piece": piece,
                "piece_id": piece_id,
                "marker": f"<|dbpe_p_{slug}_{piece_id:03d}|>",
            }
            for piece, piece_id in sorted(
                vocabulary.items(),
                key=lambda item: item[1],
            )
        ]
        if method == "ordinary_domain_bpe":
            # The ordinary method adds a namespace marker before each value.
            namespace_marker: str | None = f"<|dbpe_ns_{slug}|>"
        elif namespace == "ensembl_human_gene":
            # The atomic method uses the literal ENSG token as its prefix.
            namespace_marker = "ENSG"
        else:
            # A gene symbol needs no prefix because its field gives the type.
            namespace_marker = None
        spec: dict[str, Any] = {
            "namespace_marker": namespace_marker,
            "tokenizer_file": str(tokenizer_file.relative_to(root)),
            "tokenizer_sha256": sha256_file(tokenizer_file),
            "pieces": pieces,
            "fit_value_count": len(values),
            "fit_values_sha256": stable_sha256(values),
        }
        if value_prefix:
            # Record both suffix values and their complete canonical values.
            spec["value_prefix"] = value_prefix
            spec["canonical_fit_values_sha256"] = stable_sha256(canonical_values)
        namespaces[namespace] = spec

    # Bind both fitted models to the exact source corpus.
    manifest = {
        "schema_version": DOMAIN_BPE_SCHEMA_VERSION,
        "method": method,
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_train_sha256": parent_train_sha256,
        "vocab_per_namespace": DOMAIN_BPE_VOCAB_SIZE,
        "namespaces": namespaces,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(root / "domain_bpe_manifest.json", manifest)
    return manifest


def _domain_token_rows(
    domain_manifest: Mapping[str, Any],
    base_tokenizer: Any,
    *,
    method: str,
) -> list[dict[str, Any]]:
    """Create the ordered model rows for namespace and piece tokens."""

    source_by_namespace = {
        namespace: source
        for namespace, _, source in NAMESPACES
    }
    rows: list[dict[str, Any]] = []
    for namespace, _, _ in NAMESPACES:
        spec = domain_manifest["namespaces"][namespace]
        if method == "ordinary_domain_bpe":
            # Reserve one model row for each ordinary namespace marker.
            rows.append(
                {
                    "content": spec["namespace_marker"],
                    "namespace": namespace,
                    "object_type": "domain_bpe_namespace",
                    "base_token_ids": _base_source_ids(
                        base_tokenizer,
                        source_by_namespace[namespace],
                    ),
                }
            )

        # Reserve one model row for each fitted BPE piece marker.
        for piece in spec["pieces"]:
            rows.append(
                {
                    "content": piece["marker"],
                    "namespace": namespace,
                    "object_type": "domain_bpe_piece",
                    "source_piece": piece["piece"],
                    "base_token_ids": _base_source_ids(
                        base_tokenizer,
                        str(piece["piece"]),
                    ),
                }
            )
    return rows


def _copy_tokenizer_files(
    model_root: Path,
    target: Path,
    *,
    tokenizer: Any | None = None,
) -> None:
    """Copy the shared tokenizer files and optionally save a changed model."""

    target.mkdir()
    for name in TOKENIZER_FILES:
        # Keep all auxiliary files unchanged for every tokenizer arm.
        if tokenizer is not None and name == "tokenizer.json":
            continue
        shutil.copy2(model_root / name, target / name)
    if tokenizer is not None:
        # Save the tokenizer model after it receives the new token rows.
        tokenizer.save(str(target / "tokenizer.json"))


def write_plain_tokenizer(
    root: Path,
    model_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy and describe the unchanged GPT-OSS tokenizer."""

    _, Tokenizer, _, _ = _import_tokenizer_types()
    tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
    base_rows = tokenizer.get_vocab_size(with_added_tokens=True)
    model_vocab = int(read_json(model_root / "config.json")["vocab_size"])

    # Copy the base files without a tokenizer change.
    _copy_tokenizer_files(model_root, root / "tokenizer")

    # A byte hash confirms that the plain arm remains identical to the base.
    _require(
        sha256_file(root / "tokenizer/tokenizer.json")
        == sha256_file(model_root / "tokenizer.json"),
        "plain base tokenizer bytes changed",
    )
    # The plain arm consumes no reserved model rows and defines no codec.
    manifest = {
        "schema_version": TOKEN_MANIFEST_SCHEMA,
        "method": "plain_base_tokenizer",
        "strategy": "unchanged_base_tokenizer",
        "parent_manifest_sha256": config["corpus"]["manifest_sha256"],
        "parent_train_sha256": config["corpus"]["train_sha256"],
        "base_tokenizer_sha256": sha256_file(model_root / "tokenizer.json"),
        "base_tokenizer_length": base_rows,
        "model_vocab_size": model_vocab,
        "final_tokenizer_length": base_rows,
        "unused_model_rows_consumed": 0,
        "unused_model_rows_remaining": model_vocab - base_rows,
        "tokens": [],
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(root / "tokenizer_manifest.json", manifest)
    return manifest


def write_coded_tokenizer(
    root: Path,
    model_root: Path,
    fit_values: Mapping[str, set[str]],
    config: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    """Fit one custom method and write all model token row contracts."""

    AddedToken, Tokenizer, _, _ = _import_tokenizer_types()
    corpus = config["corpus"]

    # Fit both BPE namespaces before the script changes the base tokenizer.
    domain_manifest = fit_domain_bpe(
        root,
        fit_values,
        method=method,
        parent_manifest_sha256=corpus["manifest_sha256"],
        parent_train_sha256=corpus["train_sha256"],
    )
    tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
    base_rows = tokenizer.get_vocab_size(with_added_tokens=True)
    model_vocab = int(read_json(model_root / "config.json")["vocab_size"])
    token_rows = _domain_token_rows(domain_manifest, tokenizer, method=method)
    atomic: dict[str, Any] | None = None
    if method == "atomic_plus_domain_bpe":
        # Put the atomic ENSG row before all BPE piece rows.
        atomic = dict(ATOMIC_PREFIX_CONTRACT)
        token_rows.insert(
            0,
            {
                "content": "ENSG",
                "namespace": "ensembl_human_gene",
                "object_type": "atomic_prefix",
                "base_token_ids": _base_source_ids(tokenizer, "ENSG"),
            },
        )
    method_contract = config["tokenizer_build"]["methods"][method]
    expected_rows = int(method_contract["used_model_rows"])

    # Confirm that all new tokens fit in the unused model rows.
    _require(len(token_rows) == expected_rows, f"{method} token row count changed")
    _require(
        model_vocab - base_rows >= expected_rows,
        f"{method} does not fit in unused model rows",
    )
    # Add markers as normal tokens with no whitespace or text normalization.
    added = tokenizer.add_tokens(
        [
            AddedToken(
                str(row["content"]),
                single_word=False,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=False,
            )
            for row in token_rows
        ]
    )
    _require(added == expected_rows, f"{method} did not add every token row")
    final_rows = tokenizer.get_vocab_size(with_added_tokens=True)
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)

    # Record the IDs that the tokenizer assigned to the new rows.
    for row in token_rows:
        row["token_id"] = int(vocabulary[str(row["content"])])
    token_ids = sorted(int(row["token_id"]) for row in token_rows)

    # Require one contiguous range directly after the base tokenizer rows.
    _require(
        token_ids == list(range(base_rows, base_rows + expected_rows)),
        f"{method} token rows are not contiguous",
    )
    _copy_tokenizer_files(model_root, root / "tokenizer", tokenizer=tokenizer)

    # The codec manifest links runtime text conversion to both BPE models.
    domain_path = root / "domain_bpe_manifest.json"
    codec_manifest = {
        "schema_version": S0_CODEC_SCHEMA_VERSION,
        "method": method,
        "parent_manifest_sha256": corpus["manifest_sha256"],
        "parent_train_sha256": corpus["train_sha256"],
        "domain_bpe": {
            "manifest_file": domain_path.name,
            "manifest_sha256": sha256_file(domain_path),
            "internal_manifest_sha256": domain_manifest["manifest_sha256"],
        },
        "atomic": atomic,
    }
    codec_manifest["manifest_sha256"] = stable_sha256(codec_manifest)
    write_json(root / S0_CODEC_MANIFEST, codec_manifest)
    codec_path = root / S0_CODEC_MANIFEST

    # The token manifest tells the train code which model rows need updates.
    token_manifest = {
        "schema_version": TOKEN_MANIFEST_SCHEMA,
        "method": method,
        "strategy": method_contract["strategy"],
        "parent_manifest_sha256": corpus["manifest_sha256"],
        "parent_train_sha256": corpus["train_sha256"],
        "base_tokenizer_sha256": sha256_file(model_root / "tokenizer.json"),
        "base_tokenizer_length": base_rows,
        "model_vocab_size": model_vocab,
        "final_tokenizer_length": final_rows,
        "unused_model_rows_consumed": expected_rows,
        "unused_model_rows_remaining": model_vocab - final_rows,
        "domain_bpe_namespace_rows": method_contract["namespace_rows"],
        "domain_bpe_piece_rows": method_contract["piece_rows"],
        "atomic_prefix_rows": method_contract["atomic_rows"],
        "tokens": token_rows,
        "trainable_rows": {
            "input_embeddings": True,
            "output_head": True,
            "initialization": "mean_of_base_subtoken_rows",
        },
        "s0_tokenizer_codec": {
            "manifest_file": codec_path.name,
            "manifest_sha256": sha256_file(codec_path),
            "internal_manifest_sha256": codec_manifest["manifest_sha256"],
        },
    }
    token_manifest["manifest_sha256"] = stable_sha256(token_manifest)
    write_json(root / "tokenizer_manifest.json", token_manifest)
    return token_manifest


# Block 4: Tokenize every train row and prove exact custom-code cycles.
def _model_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the exact model-visible content for one train row."""

    # Metadata selects the loss target. It does not enter a model message.
    return [
        {"role": "system", "content": str(row["system"])},
        {"role": "user", "content": str(row["question"])},
        {"role": "assistant", "content": canonical_json(row["answer"])},
    ]


def audit_arm(
    root: Path,
    rows: list[dict[str, Any]],
    fit_values: Mapping[str, set[str]],
    config: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    """Audit token rows, all fit values, and every full train record."""

    _, Tokenizer, _, _ = _import_tokenizer_types()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
    chat_template = _compile_chat_template(
        root / "tokenizer/chat_template.jinja"
    )
    token_manifest = read_json(root / "tokenizer_manifest.json")
    codec = load_s0_tokenizer_codec_for_token_manifest(
        root / "tokenizer_manifest.json"
    )
    failures: list[str] = []

    # Each declared model row must map to exactly one expected token ID.
    for token_row in token_manifest["tokens"]:
        encoded = tokenizer.encode(
            str(token_row["content"]),
            add_special_tokens=False,
        )
        if encoded.ids != [int(token_row["token_id"])]:
            failures.append(f"token_row:{token_row['content']}")

    value_count = 0
    value_round_trip_failures = 0
    if method == "plain_base_tokenizer":
        # The plain method has no codec, but each fit value must produce tokens.
        if codec is not None:
            failures.append("plain_tokenizer_has_codec")
        for namespace, values in fit_values.items():
            del namespace
            for value in values:
                if not tokenizer.encode(value, add_special_tokens=False).ids:
                    value_round_trip_failures += 1
                value_count += 1
    else:
        # Each custom value must produce tokens and return its exact source text.
        if codec is None:
            failures.append("custom_tokenizer_has_no_codec")
        else:
            for namespace, values in fit_values.items():
                for value in sorted(values):
                    try:
                        coded = codec.encode_value(namespace, value)
                        ids = tokenizer.encode(coded, add_special_tokens=False).ids
                        restored = codec.decode_text(coded)
                    except Exception as error:  # Record an artifact failure.
                        if len(failures) < 20:
                            failures.append(
                                f"value:{namespace}:{value}:{type(error).__name__}"
                            )
                        value_round_trip_failures += 1
                    else:
                        if not ids or restored != value:
                            value_round_trip_failures += 1
                    value_count += 1

    tokenized_rows = 0
    row_failures = 0
    representation_failures = 0
    maximum_message_tokens = 0
    maximum_sequence_tokens = 0
    maximum_sequence_record_id = ""
    total_message_tokens = 0
    total_sequence_tokens = 0

    # Test every model-visible message from every train row.
    for row_index, row in enumerate(rows):
        messages = _model_messages(row)
        if codec is not None:
            # Convert typed identifiers before the tokenizer reads each message.
            messages = codec.encode_messages(
                messages,
                question_family=row["metadata"]["question_family"],
            )
        try:
            for message in messages:
                # Every system, user, and assistant message must produce tokens.
                token_ids = tokenizer.encode(
                    message["content"],
                    add_special_tokens=False,
                ).ids
                if not token_ids:
                    raise ValueError("empty token sequence")
                maximum_message_tokens = max(maximum_message_tokens, len(token_ids))
                total_message_tokens += len(token_ids)

            # Apply the same chat template and codec that the trainer uses.
            rendered = chat_template.render(
                messages=messages,
                tools=None,
                documents=None,
                add_generation_prompt=False,
            )
            sequence_ids = tokenizer.encode(
                rendered,
                add_special_tokens=False,
            ).ids
            if not sequence_ids:
                raise ValueError("empty full sequence")
            total_sequence_tokens += len(sequence_ids)
            if len(sequence_ids) > maximum_sequence_tokens:
                maximum_sequence_tokens = len(sequence_ids)
                maximum_sequence_record_id = row["record_id"]
            if codec is not None:
                # The assistant target must decode to its exact canonical JSON.
                decoded, report = codec.decode_generated_answer(
                    messages[-1]["content"]
                )
                if not report["valid"]:
                    representation_failures += 1
                if decoded != canonical_json(row["answer"]):
                    raise ValueError("assistant target round trip changed")
        except Exception as error:  # Keep a bounded failure sample.
            row_failures += 1
            if len(failures) < 20:
                failures.append(
                    f"row:{row_index}:{row['record_id']}:{type(error).__name__}"
                )
        tokenized_rows += 1

    expected_rows = int(config["corpus"]["train_rows"])

    # The arm passes only if every token, value, and row check passes.
    passed = (
        not failures
        and value_round_trip_failures == 0
        and row_failures == 0
        and representation_failures == 0
        and tokenized_rows == expected_rows
    )
    report = {
        "schema_version": ARM_AUDIT_SCHEMA,
        "passed": passed,
        "method": method,
        "parent_manifest_sha256": config["corpus"]["manifest_sha256"],
        "parent_train_sha256": config["corpus"]["train_sha256"],
        "full_corpus_row_count": expected_rows,
        "full_corpus_tokenized_rows": tokenized_rows,
        "fit_value_count": value_count,
        "fit_value_round_trip_failures": value_round_trip_failures,
        "full_corpus_row_failures": row_failures,
        "representation_failures": representation_failures,
        "maximum_message_tokens": maximum_message_tokens,
        "maximum_sequence_tokens": maximum_sequence_tokens,
        "maximum_sequence_record_id": maximum_sequence_record_id,
        "total_message_tokens": total_message_tokens,
        "total_sequence_tokens": total_sequence_tokens,
        "failure_sample": failures,
        "unused_model_rows_consumed": token_manifest[
            "unused_model_rows_consumed"
        ],
        "unused_model_rows_remaining": token_manifest[
            "unused_model_rows_remaining"
        ],
        "promotion_eligible": False,
    }
    write_json(root / "audit_report.json", report)
    return report


# Block 5: Build all arms in one temporary tree and publish them together.
def build_arm(
    root: Path,
    model_root: Path,
    rows: list[dict[str, Any]],
    fit_values: Mapping[str, set[str]],
    config: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    """Build, audit, and describe one tokenizer method."""

    root.mkdir()

    # First write the tokenizer files and their internal manifests.
    if method == "plain_base_tokenizer":
        token_manifest = write_plain_tokenizer(root, model_root, config)
    else:
        token_manifest = write_coded_tokenizer(
            root,
            model_root,
            fit_values,
            config,
            method=method,
        )
    audit = audit_arm(
        root,
        rows,
        fit_values,
        config,
        method=method,
    )

    # Stop before publication if any full-corpus audit check fails.
    if not audit["passed"]:
        raise RuntimeError(f"The {method} audit failed: {audit['failure_sample']}")
    # The outer manifest records all files that belong to this arm.
    manifest = {
        "schema_version": ARM_MANIFEST_SCHEMA,
        "dataset_id": f"{config['corpus']['dataset_id']}__{method}",
        "method": method,
        "parent_dataset": {
            "dataset_id": config["corpus"]["dataset_id"],
            "manifest_sha256": config["corpus"]["manifest_sha256"],
            "train_sha256": config["corpus"]["train_sha256"],
        },
        "tokenizer_manifest_sha256": token_manifest["manifest_sha256"],
        "unused_model_rows_consumed": token_manifest[
            "unused_model_rows_consumed"
        ],
        "unused_model_rows_remaining": token_manifest[
            "unused_model_rows_remaining"
        ],
        "tokenizer_artifact_hashes": _tokenizer_artifact_hashes(root),
        "file_hashes": _tree_file_hashes(root),
        "audit_passed": True,
        "promotion_eligible": False,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def build(
    config_path: Path,
    parent_root: Path,
    model_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Build and publish all three tokenizer methods as one artifact."""

    # Complete all read-only contract checks before the script creates output.
    config = load_config(config_path)
    shared_identity = validate_base_models(config)
    validate_selected_model(model_root, shared_identity)
    _, train_path = validate_corpus(config, parent_root)
    rows, fit_values = read_and_validate_train_rows(train_path, config)

    # A process-specific directory keeps partial results outside the final path.
    temporary = output_root.with_name(
        output_root.name + f".building-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        # Build and audit all three methods in the fixed order.
        manifests = {
            method: build_arm(
                temporary / method,
                model_root,
                rows,
                fit_values,
                config,
                method=method,
            )
            for method in METHODS
        }
        # The comparison report gives one summary for all tokenizer arms.
        comparison = {
            "schema_version": COMPARISON_REPORT_SCHEMA,
            "dataset_id": config["corpus"]["dataset_id"],
            "parent_manifest_sha256": config["corpus"]["manifest_sha256"],
            "parent_train_sha256": config["corpus"]["train_sha256"],
            "full_corpus_row_count": len(rows),
            "fit_value_counts": {
                namespace: len(values)
                for namespace, values in fit_values.items()
            },
            "methods": {
                method: {
                    "audit_passed": manifests[method]["audit_passed"],
                    "unused_model_rows_consumed": manifests[method][
                        "unused_model_rows_consumed"
                    ],
                    "unused_model_rows_remaining": manifests[method][
                        "unused_model_rows_remaining"
                    ],
                }
                for method in METHODS
            },
        }
        comparison["report_sha256"] = stable_sha256(comparison)
        write_json(temporary / "comparison_report.json", comparison)

        # The pair manifest binds every arm and report to the matrix contract.
        pair = {
            "schema_version": PAIR_MANIFEST_SCHEMA,
            "matrix_id": config["matrix_id"],
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "config_sha256": sha256_file(config_path),
            "parent_dataset": {
                "dataset_id": config["corpus"]["dataset_id"],
                "manifest_sha256": config["corpus"]["manifest_sha256"],
                "train_sha256": config["corpus"]["train_sha256"],
                "train_rows": len(rows),
            },
            "shared_base_tokenizer": dict(shared_identity),
            "methods": {
                method: {
                    "root": method,
                    "manifest_sha256": sha256_file(
                        temporary / method / "manifest.json"
                    ),
                    "tokenizer_manifest_sha256": manifests[method][
                        "tokenizer_manifest_sha256"
                    ],
                    "audit_sha256": sha256_file(
                        temporary / method / "audit_report.json"
                    ),
                }
                for method in METHODS
            },
            "comparison_report": {
                "path": "comparison_report.json",
                "file_sha256": sha256_file(
                    temporary / "comparison_report.json"
                ),
                "internal_report_sha256": comparison["report_sha256"],
            },
        }
        pair["manifest_sha256"] = stable_sha256(pair)
        write_json(temporary / "pair_manifest.json", pair)

        if output_root.exists():
            # Replace an old artifact only when the user gives --overwrite.
            if not overwrite:
                raise FileExistsError(
                    f"Output exists; use --overwrite to replace it: {output_root}"
                )
            shutil.rmtree(output_root)

        # One directory rename exposes the complete artifact at the final path.
        temporary.rename(output_root)
        return pair
    except Exception:
        # Remove a partial temporary artifact after any build or audit failure.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Build all tokenizer methods and print the pair manifest."""

    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)

    # Use each config path unless the command supplies an explicit override.
    parent_root = (
        resolve_repo_path(config["corpus"]["root"])
        if args.parent_root is None
        else args.parent_root.expanduser().resolve()
    )
    model_root = (
        resolve_repo_path(config["base_models"]["oss20b"]["path"])
        if args.model_path is None
        else args.model_path.expanduser().resolve()
    )
    output_root = (
        resolve_repo_path(config["tokenizer_build"]["output_root"])
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    result = build(
        config_path,
        parent_root,
        model_root,
        output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
