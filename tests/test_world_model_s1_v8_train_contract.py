"""Keep the v8 data experiment on the frozen v6 training recipe."""

import copy
import json
from pathlib import Path

import pytest

from runtime.world_model.s1_overlay import S1_OVERLAY_V7_SCHEMA_VERSION
from scripts import validate_world_model_v2_s0_train_contract as train_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/world_model_s1_120b_training_v8_32n.json"
METHOD = "oss120b-plain-base-tokenizer-lora-r32"


def test_v8_changes_only_data_identity_and_output_paths():
    """Exclude a recipe, parent, tokenizer, or loss change from this experiment."""
    baseline = json.loads(
        (ROOT / "config/world_model_s1_120b_training_v6_32n.json").read_text()
    )
    config = json.loads(CONFIG.read_text())
    normalized = copy.deepcopy(config)
    for key in ("schema_version", "training_id"):
        normalized[key] = baseline[key]
    for key in (
        "root", "manifest_sha256", "train_sha256", "evaluator_manifest",
        "evaluator_manifest_sha256",
    ):
        normalized["corpus"][key] = baseline["corpus"][key]
    normalized["methods"][0]["output_root"] = baseline["methods"][0]["output_root"]
    assert normalized == baseline
    assert config["methods"][0]["output_root"] == (
        "data/world_model_v4/checkpoints/s1_v8_120b_32n/" + METHOD
    )


def test_real_v8_contract_validates_published_artifacts_and_parent():
    """Resolve actual data, tokenizer, model metadata, and complete S0 parent."""
    assert train_contract.S1_OVERLAYS[train_contract.S1_V8_PRODUCTION_SCHEMA] == (
        "runtime/world_model/s1_v8.yaml", S1_OVERLAY_V7_SCHEMA_VERSION,
    )
    values = train_contract.validate_train_contract(ROOT, CONFIG, METHOD)
    expected = {
        "S0_CONFIG_ID": "world_model_s1_120b_training_v8_32n",
        "S0_DATASET_ID": "world_model_v3_s1_coordinate_system",
        "NUM_NODES": "32", "WORLD_SIZE_EXPECTED": "256",
        "GLOBAL_BATCH_SIZE": "32", "TRAIN_NUM_EPOCHS": "1",
        "TRAIN_TOTAL_STEPS": "1728", "LEARNING_RATE": "0.0001",
        "MAX_LENGTH": "4096", "LOSS_CONTRACT": "multiplex_sft_v3",
        "INIT_ADAPTER_RUN_ID": "oss120b-plain-base-tokenizer-lora-r32-slurm-5631883",
    }
    assert {key: values[key] for key in expected} == expected
    assert Path(values["S0_TRAIN_PATH"]) == ROOT / "data/world_model_v4/corpora/s1_v8/train.jsonl"


@pytest.mark.parametrize(("field", "value", "error"), [
    ("initialization", None, "S1 v8 must start from an S0 v4 checkpoint"),
    ("training_id", "world_model_s1_120b_training_v6_32n", "S1 run identity changed"),
    ("corpus.root", "data/world_model_v4/corpora/s1_v6", "S1 full training schedule/data contract changed"),
    ("corpus.evaluator_manifest", "data/world_model_v4/eval/s1_v6/manifest.json", "S1 full training schedule/data contract changed"),
    ("run_settings.num_nodes", 16, "S1 full training schedule/data contract changed"),
    ("run_settings.num_train_epochs", 2, "S1 full training schedule/data contract changed"),
    ("run_settings.learning_rate", 0.0002, "S1 full training schedule/data contract changed"),
    ("run_settings.max_length", 8192, "S1 full training schedule/data contract changed"),
])
def test_v8_rejects_mixed_versions_or_recipe_changes(tmp_path, field, value, error):
    """Fail identity and recipe mismatches before loading any model artifacts."""
    config = json.loads(CONFIG.read_text())
    parent = config
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match=error):
        train_contract.validate_train_contract(tmp_path, path, METHOD)
