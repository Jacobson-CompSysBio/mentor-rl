"""Reject S2 launch drift before scheduling expensive distributed training."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import validate_world_model_s2_train_contract as contract


@pytest.fixture
def config():
    """Use the concrete production settings as the launch contract fixture."""
    path = Path(__file__).resolve().parents[1] / "config/world_model_s2_120b_training_v1_184n.json"
    return json.loads(path.read_text())


def test_requested_ten_hour_recipe_is_accepted(config):
    contract.validate_recipe(config)
    assert config["resource_plan"]["requested_walltime"] == "10:00:00"


@pytest.mark.parametrize("section,key,value", [
    ("run_settings", "learning_rate", 0.000575),
    ("run_settings", "global_batch_size", 32),
    ("run_settings", "num_train_epochs", 6),
    ("run_settings", "bf16", 1),
    ("run_settings", "max_length", 4096),
    ("run_settings", "eval_steps", 1000),
    ("resource_plan", "requested_walltime", "12:00:00"),
    ("resource_plan", "gpu_preflight_required", False),
    ("trainer", "segment_loss_telemetry", False),
    ("trainer", "adam_beta2", 0.95),
    ("initialization", "restore_parent_optimizer", True),
    ("initialization", "restore_parent_scheduler", True),
    ("initialization", "start_step", 1728),
    ("checkpoint_selection", "test_panel_access_during_training", True),
])
def test_recipe_drift_is_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(SystemExit):
        contract.validate_recipe(config)


def test_source_edit_after_sealing_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "CORE_SOURCE_PATHS", {"trainer.py"})
    trainer = tmp_path / "trainer.py"
    trainer.write_text("pinned source\n")
    pins = [{"path": "trainer.py", "sha256": contract.sha256_file(trainer)}]
    assert len(contract.validate_source_pins(tmp_path, pins)) == 64
    trainer.write_text("changed after submission\n")
    with pytest.raises(SystemExit, match="training source identity changed"):
        contract.validate_source_pins(tmp_path, pins)


def test_missing_core_source_pin_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "CORE_SOURCE_PATHS", {"trainer.py", "adapter.py"})
    (tmp_path / "trainer.py").write_text("pinned\n")
    pins = [{"path": "trainer.py", "sha256": contract.sha256_file(tmp_path / "trainer.py")}]
    with pytest.raises(SystemExit, match="omit core files"):
        contract.validate_source_pins(tmp_path, pins)


@pytest.mark.parametrize("name", ["../outside.py", "/tmp/outside.py", "a/../trainer.py"])
def test_pinned_paths_cannot_escape_or_hide_traversal(tmp_path, name):
    with pytest.raises(SystemExit):
        contract.repo_path(tmp_path, name)


def test_bank_authorization_cannot_be_replaced_by_a_false_review_claim(tmp_path):
    receipt = {
        "accepted_as_is_for_requested_s2_run": True,
        "human_wording_review_claimed": True,
        "corpus_manifest": {"sha256": contract.CORPUS_SHA256},
        "original_corpus_manifest_modified": False,
        "frozen_banks": [{}] * 6,
    }
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(receipt))
    cfg = {"bank_acceptance": {"path": path.name, "sha256": contract.sha256_file(path)}}
    with pytest.raises(SystemExit, match="as-is bank acceptance changed"):
        contract.validate_bank_acceptance(tmp_path, cfg)


def test_training_preflight_hashes_train_and_validation_without_opening_test(tmp_path, config, monkeypatch):
    verified = []
    monkeypatch.setattr(contract, "verify_file", lambda path, expected, name: verified.append(path.name))
    files = {
        split + ".jsonl": {"sha256": digest, "row_count": rows}
        for split, (rows, digest) in contract.SPLITS.items()
    }
    manifest = {
        "schema_version": "mentor-rl-s2-production-corpus-v1",
        "production_id": "s2_production_v1_8k", "data_ready": True,
        "whole_trace_source_and_supervision_checked": True,
        "truncation_applied": False, "max_length": 8192,
        "panel_id": config["corpus"]["test_panel_id"], "files": files,
    }
    monkeypatch.setattr(contract, "read_json", lambda path, name: deepcopy(manifest))
    audit = {
        "corpus_manifest": {"sha256": contract.CORPUS_SHA256},
        "data_verification_passed": True,
        "upstream_complete_token_receipts_matched": 302592,
        "max_tokens": {"train": 7368, "validation": 7216, "test": 8157},
    }
    monkeypatch.setattr(contract, "checked_reference", lambda *args: (tmp_path, audit))
    contract.validate_corpus(tmp_path, config)
    assert verified == ["manifest.json", "train.jsonl", "validation.jsonl"]
    config["corpus"]["train_file"] = "test.jsonl"
    with pytest.raises(SystemExit, match="train declaration changed"):
        contract.validate_corpus(tmp_path, config)


def test_failed_prior_audit_cannot_authorize_training(tmp_path, config, monkeypatch):
    monkeypatch.setattr(contract, "verify_file", lambda *args: None)
    manifest = {
        "schema_version": "mentor-rl-s2-production-corpus-v1",
        "production_id": "s2_production_v1_8k", "data_ready": True,
        "whole_trace_source_and_supervision_checked": True,
        "truncation_applied": False, "max_length": 8192,
        "panel_id": config["corpus"]["test_panel_id"],
        "files": {split + ".jsonl": {"sha256": digest, "row_count": rows}
                  for split, (rows, digest) in contract.SPLITS.items()},
    }
    monkeypatch.setattr(contract, "read_json", lambda *args: manifest)
    monkeypatch.setattr(contract, "checked_reference", lambda *args: (tmp_path, {"data_verification_passed": False}))
    with pytest.raises(SystemExit, match="independent verification is incomplete"):
        contract.validate_corpus(tmp_path, config)
