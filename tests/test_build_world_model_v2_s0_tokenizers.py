from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_world_model_v2_s0_tokenizers import (
    ARM_MANIFEST_FILE_SHA256,
    AUDIT_REPORT_FILE_SHA256,
    DEFAULT_MODEL_ROOT,
    TOKENIZER_MANIFEST_FILE_SHA256,
    TOKENIZER_MANIFEST_SHA256,
    TOKENIZER_METHOD,
    build,
    frozen_arm_manifest,
    frozen_audit_report,
    frozen_tokenizer_manifest,
    sha256_file,
    stable_sha256,
    validate_frozen_artifact,
    validate_model_root,
    write_json,
)


def test_frozen_tokenizer_manifest_has_gate_identity() -> None:
    manifest = frozen_tokenizer_manifest()
    identity = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }

    assert manifest["method"] == TOKENIZER_METHOD
    assert manifest["tokens"] == []
    assert manifest["manifest_sha256"] == TOKENIZER_MANIFEST_SHA256
    assert stable_sha256(identity) == TOKENIZER_MANIFEST_SHA256


def test_frozen_json_files_have_gate_identities(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit_report.json"
    token_manifest_path = tmp_path / "tokenizer_manifest.json"
    arm_manifest_path = tmp_path / "manifest.json"

    write_json(audit_path, frozen_audit_report())
    write_json(token_manifest_path, frozen_tokenizer_manifest())
    write_json(arm_manifest_path, frozen_arm_manifest())

    assert sha256_file(audit_path) == AUDIT_REPORT_FILE_SHA256
    assert sha256_file(token_manifest_path) == TOKENIZER_MANIFEST_FILE_SHA256
    assert sha256_file(arm_manifest_path) == ARM_MANIFEST_FILE_SHA256


def test_build_copies_only_the_frozen_plain_tokenizer(tmp_path: Path) -> None:
    if not DEFAULT_MODEL_ROOT.is_dir():
        pytest.skip("The pinned GPT-OSS model is not available")

    output_root = tmp_path / "plain_base_tokenizer"
    manifest = build(DEFAULT_MODEL_ROOT, output_root)

    assert manifest == frozen_arm_manifest()
    assert validate_frozen_artifact(output_root) == manifest
    assert {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
    } == {
        "audit_report.json",
        "manifest.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_manifest.json",
    }


def test_model_check_rejects_an_unknown_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model config identity"):
        validate_model_root(tmp_path)
