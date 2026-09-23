#!/usr/bin/env python3
"""Build the frozen plain GPT-OSS tokenizer artifact for S0 v6."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = REPO_ROOT.parent / "models/gpt-oss-20b-bf16"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "data/world_model_v2/sft/s0_human_identifier_tokenizers_v4"
    / "plain_base_tokenizer"
)

TOKENIZER_METHOD = "plain_base_tokenizer"
TOKENIZER_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
TOKENIZER_ARTIFACT_HASHES = {
    "chat_template.jinja": (
        "9ad8adf1324b8eb902d6e70087f3184b1b9b345c453bbd3d38949c2f13205c1d"
    ),
    "special_tokens_map.json": (
        "8464cabd6eda239fe46ebf8ae63b46c417721784a961a022f6b59174a2cda0e2"
    ),
    "tokenizer.json": (
        "0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3"
    ),
    "tokenizer_config.json": (
        "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070"
    ),
}
MODEL_CONFIG_HASHES = frozenset(
    {
        "abbfcc94fd07e21544a068152e6cf80f2aa4df0f1c7d4829c125301c1f116bdb",
        "8760a655ceb653e953eba13d8e0d68ba1dcb98b78a2444e155394643e5775fdb",
    }
)

TOKENIZER_MANIFEST_SHA256 = (
    "245163b5189a64afb8e5b061585f90d4c15bcf94599b68307e9ff008880fe3c0"
)
TOKENIZER_MANIFEST_FILE_SHA256 = (
    "d7109e8b4f6dd4eebb1f97888b3dc39848e6ac1284c50404c81d993a308a9b6f"
)
AUDIT_REPORT_FILE_SHA256 = (
    "c3f85de851d9bba193a2939f434b9575224b00e6d1251dc66c08db29efd31aac"
)
ARM_MANIFEST_FILE_SHA256 = (
    "b1379f9a1a751ce7bdc2a99389717ba3095c64e3fbfa608ef77eee181bb8cb6e"
)

PARENT_DATASET_ID = "world_model_v2_s0_human_identifiers_v4"
PARENT_MANIFEST_SHA256 = (
    "8faeef0a97fa8375f7f62a6baa0736b48655e0dd78509b5e939ce7651bab482d"
)
PARENT_TRAIN_SHA256 = (
    "55a77ede53d6f18d5f7699b23c5293857c3d8002e53d26ad827afcef92ba530b"
)


def canonical_json(value: Any) -> str:
    """Return compact canonical JSON for one hash."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one readable JSON object."""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def frozen_tokenizer_manifest() -> dict[str, Any]:
    """Return the token manifest that the passed gate used."""

    manifest = {
        "base_tokenizer_length": 200019,
        "base_tokenizer_sha256": TOKENIZER_ARTIFACT_HASHES["tokenizer.json"],
        "final_tokenizer_length": 200019,
        "method": TOKENIZER_METHOD,
        "model_vocab_size": 201088,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_train_sha256": PARENT_TRAIN_SHA256,
        "schema_version": "mentor-rl-world-model-s0-tokenizer-v3",
        "strategy": "unchanged_base_tokenizer",
        "tokens": [],
        "unused_model_rows_consumed": 0,
        "unused_model_rows_remaining": 1069,
    }
    observed = stable_sha256(manifest)
    if observed != TOKENIZER_MANIFEST_SHA256:
        raise ValueError("The frozen token manifest identity changed")
    manifest["manifest_sha256"] = observed
    return manifest


def frozen_audit_report() -> dict[str, Any]:
    """Return the audit receipt that the passed gate used."""

    return {
        "failure_sample": [],
        "fit_value_count": 85286,
        "fit_value_round_trip_failures": 0,
        "full_corpus_row_count": 85284,
        "full_corpus_row_failures": 0,
        "full_corpus_tokenized_rows": 85284,
        "maximum_message_tokens": 372,
        "maximum_sequence_record_id": "wm2_s0_39a5407f206e099247e24ffe",
        "maximum_sequence_tokens": 555,
        "method": TOKENIZER_METHOD,
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_train_sha256": PARENT_TRAIN_SHA256,
        "passed": True,
        "promotion_eligible": False,
        "representation_failures": 0,
        "schema_version": "mentor-rl-world-model-s0-tokenizer-audit-v4",
        "total_message_tokens": 11330901,
        "total_sequence_tokens": 17983053,
        "unused_model_rows_consumed": 0,
        "unused_model_rows_remaining": 1069,
    }


def frozen_arm_manifest() -> dict[str, Any]:
    """Return the arm manifest that the passed gate used."""

    tokenizer_hashes = {
        f"tokenizer/{name}": value
        for name, value in TOKENIZER_ARTIFACT_HASHES.items()
    }
    return {
        "appended_model_rows": 0,
        "audit_passed": True,
        "dataset_id": f"{PARENT_DATASET_ID}__{TOKENIZER_METHOD}",
        "file_hashes": {
            "audit_report.json": AUDIT_REPORT_FILE_SHA256,
            **tokenizer_hashes,
            "tokenizer_manifest.json": TOKENIZER_MANIFEST_FILE_SHA256,
        },
        "method": TOKENIZER_METHOD,
        "parent_dataset": {
            "dataset_id": PARENT_DATASET_ID,
            "manifest_sha256": PARENT_MANIFEST_SHA256,
            "train_sha256": PARENT_TRAIN_SHA256,
        },
        "promotion_eligible": False,
        "schema_version": "mentor-rl-world-model-s0-tokenizer-arm-v4",
        "token_rows": 0,
        "tokenizer_artifact_hashes": tokenizer_hashes,
        "tokenizer_manifest_sha256": TOKENIZER_MANIFEST_SHA256,
        "unused_model_rows_consumed": 0,
        "unused_model_rows_remaining": 1069,
    }


def validate_model_root(model_root: Path) -> None:
    """Check the pinned GPT-OSS model and tokenizer files."""

    config_path = model_root / "config.json"
    if sha256_file(config_path) not in MODEL_CONFIG_HASHES:
        raise ValueError("The GPT-OSS model config identity changed")
    if read_json(config_path).get("vocab_size") != 201088:
        raise ValueError("The GPT-OSS model vocabulary size changed")

    for name, expected in TOKENIZER_ARTIFACT_HASHES.items():
        if sha256_file(model_root / name) != expected:
            raise ValueError(f"The GPT-OSS tokenizer file changed: {name}")


def validate_frozen_artifact(root: Path) -> dict[str, Any]:
    """Check all files in one frozen tokenizer artifact."""

    manifest = frozen_arm_manifest()
    expected_paths = set(manifest["file_hashes"]) | {"manifest.json"}
    observed_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed_paths != expected_paths:
        raise ValueError("The frozen tokenizer file set changed")

    for name, expected in manifest["file_hashes"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"The frozen tokenizer file changed: {name}")
    if sha256_file(root / "manifest.json") != ARM_MANIFEST_FILE_SHA256:
        raise ValueError("The frozen tokenizer arm identity changed")
    return manifest


def build(
    model_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build the exact plain tokenizer artifact that the passed gate used."""

    model_root = model_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    validate_model_root(model_root)

    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists; use --overwrite to replace it: {output_root}"
        )

    temporary = output_root.with_name(
        output_root.name + f".building-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        tokenizer_root = temporary / "tokenizer"
        tokenizer_root.mkdir()
        for name in TOKENIZER_FILES:
            shutil.copy2(model_root / name, tokenizer_root / name)

        write_json(temporary / "audit_report.json", frozen_audit_report())
        write_json(
            temporary / "tokenizer_manifest.json",
            frozen_tokenizer_manifest(),
        )
        write_json(temporary / "manifest.json", frozen_arm_manifest())
        manifest = validate_frozen_artifact(temporary)

        if output_root.exists():
            shutil.rmtree(output_root)
        temporary.rename(output_root)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args() -> argparse.Namespace:
    """Parse the command arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Build the tokenizer artifact and print its manifest."""

    args = parse_args()
    result = build(
        args.model_path,
        args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
