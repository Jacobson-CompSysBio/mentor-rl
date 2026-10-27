"""Recognize v10 data without weakening v9's recipe or artifact checks."""

import json
from pathlib import Path

import pytest

from runtime.world_model.contract import load_contract
from runtime.world_model.s1_overlay import S1_OVERLAY_V9_SCHEMA_VERSION
from scripts import build_world_model_s1_generation_bundle as generation
from scripts import validate_world_model_s1_eval_contract as evaluation
from scripts import validate_world_model_v2_s0_train_contract as training


ROOT = Path(__file__).resolve().parents[1]
METHOD = "oss120b-plain-base-tokenizer-lora-r32"


def _training_fixture():
    """Derive only a temporary validator fixture; its historical hashes are not launch pins."""

    config = json.loads((ROOT / "config/world_model_s1_120b_training_v9_32n.json").read_text())
    config["schema_version"] = training.S1_V10_PRODUCTION_SCHEMA
    config["training_id"] = "world_model_s1_120b_training_v10_32n"
    config["corpus"].update(training.S1_FULL_CORPUS_LAYOUTS[training.S1_V10_PRODUCTION_SCHEMA])
    return config


def test_v10_registration_preserves_the_comparison_recipe():
    """Register the new data identity with the existing topology, steps, and telemetry."""

    overlay = Path("runtime/world_model/s1_v10.yaml")
    assert generation.S1_OVERLAY_SCHEMAS[overlay] == S1_OVERLAY_V9_SCHEMA_VERSION
    assert evaluation.S1_OVERLAY_SCHEMAS[str(overlay)] == S1_OVERLAY_V9_SCHEMA_VERSION
    assert training.S1_OVERLAYS[training.S1_V10_PRODUCTION_SCHEMA] == (
        str(overlay), S1_OVERLAY_V9_SCHEMA_VERSION,
    )
    assert training.S1_FULL_RUN_PROFILES["world_model_s1_120b_training_v10_32n"] == (
        training.S1_FULL_RUN_PROFILES["world_model_s1_120b_training_v9_32n"]
    )


@pytest.mark.parametrize(("field", "value", "error"), [
    ("initialization", None, "S1 v10 must start from an S0 v4 checkpoint"),
    ("training_id", "world_model_s1_120b_training_v9_32n", "S1 run identity changed"),
    ("corpus.root", "data/world_model_v4/corpora/s1_v9", "schedule/data contract changed"),
    ("run_settings.max_length", 8192, "schedule/data contract changed"),
    ("run_settings.num_train_epochs", 2, "schedule/data contract changed"),
    ("run_settings.diagnostic_snapshot_steps", [400, 800], "schedule/data contract changed"),
    ("trainer.segment_loss_telemetry", False, "v10 requires segment loss telemetry"),
])
def test_v10_rejects_recipe_and_identity_drift_before_artifact_checks(tmp_path, field, value, error):
    """The new version must retain the same parent, data layout, recipe, and observability gates."""

    config = _training_fixture()
    parent = config
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    path = tmp_path / "validator_fixture.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match=error):
        training.validate_train_contract(tmp_path, path, METHOD)


def test_v10_generation_and_panel_consumers_accept_the_frozen_data(tmp_path):
    """Verify real corpus/panel pins without loading a checkpoint or launching evaluation."""

    evaluator_root = ROOT / "data/world_model_v4/eval/s1_v10"
    evaluator_path = evaluator_root / "manifest.json"
    evaluator = generation.read_json(evaluator_path)
    contract = load_contract()
    source = contract["source_index"]
    source_path = ROOT / source["output_directory"] / (
        f"world_model_source_index_{source['content_sha256']}.json.gz"
    )
    bundle = generation.build_generation_bundle(
        evaluator_manifest_path=evaluator_path,
        evaluator_manifest_sha256=generation.sha256_file(evaluator_path),
        source_index_path=source_path, source_index_sha256=source["content_sha256"],
        output_dir=tmp_path / "generation",
    )
    assert bundle["record_count"] == 6875
    assert bundle["reads_private_answer_keys"] is False
    assert bundle["contract"]["s1_overlay"]["schema_version"] == S1_OVERLAY_V9_SCHEMA_VERSION
    old = generation.read_json(ROOT / "data/world_model_v4/eval/s1_v9/manifest.json")
    assert bundle["test_panel_id"] == old["test"]["test_panel_id"]
    panel = json.loads((ROOT / "config/world_model_s1_120b_32n_test_v9_structured95_512n.json").read_text())["test_panel"]
    panel.update(
        evaluator_root=evaluator_root.relative_to(ROOT).as_posix(),
        evaluator_manifest_sha256=generation.sha256_file(evaluator_path),
    )
    checked = evaluation._validate_panel(
        repo_root=ROOT, panel_config=panel, contract=contract,
        corpus_manifest_sha256=evaluator["corpus_manifest_sha256"], allow_s1_overlay=True,
    )
    assert checked[0] == evaluator and checked[-1] == 6875
