"""Pin v9 data, observational telemetry, and unchanged production gate behavior."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from runtime.world_model.s1_overlay import S1_OVERLAY_V8_SCHEMA_VERSION
from scripts import validate_world_model_s1_eval_contract as s1_eval
from scripts import validate_world_model_v2_s0_eval_contract as s0_eval
from scripts import validate_world_model_v2_s0_train_contract as train_contract


ROOT = Path(__file__).resolve().parents[1]
METHOD = "oss120b-plain-base-tokenizer-lora-r32"
TRAIN = ROOT / "config/world_model_s1_120b_training_v9_32n.json"
S1 = ROOT / "config/world_model_s1_120b_32n_test_v9_retained_512n.json"
S1_95 = ROOT / "config/world_model_s1_120b_32n_test_v9_structured95_512n.json"
S0 = ROOT / "config/world_model_s1_120b_s0_forgetting_test_v9_512n.json"


def read(path):
    """Read one config fixture."""
    return json.loads(path.read_text())


def test_v9_preserves_optimizer_parent_tokenizer_and_objective():
    """The new observation settings do not change the comparison recipe."""
    baseline = read(ROOT / "config/world_model_s1_120b_training_v8_32n.json")
    config = read(TRAIN)
    normalized = copy.deepcopy(config)
    for key in ("schema_version", "training_id", "corpus"):
        normalized[key] = baseline[key]
    normalized["methods"][0]["output_root"] = baseline["methods"][0]["output_root"]
    assert normalized["trainer"].pop("segment_loss_telemetry") is True
    assert normalized["run_settings"].pop("diagnostic_snapshot_steps") == [400, 800, 1200, 1600]
    assert normalized == baseline


def test_real_v9_training_contract_and_legacy_telemetry_defaults():
    """Validate the published corpus and completed parent before submission."""
    assert train_contract.S1_OVERLAYS[train_contract.S1_V9_PRODUCTION_SCHEMA] == (
        "runtime/world_model/s1_v9.yaml", S1_OVERLAY_V8_SCHEMA_VERSION,
    )
    values = train_contract.validate_train_contract(ROOT, TRAIN, METHOD)
    assert values["SEGMENT_LOSS_TELEMETRY"] == "1"
    assert values["DIAGNOSTIC_SNAPSHOT_STEPS"] == "400,800,1200,1600"
    assert values["TRAIN_TOTAL_STEPS"] == "1728"
    assert values["GLOBAL_BATCH_SIZE"] == "32"
    assert values["MAX_LENGTH"] == "4096"
    assert Path(values["S0_TRAIN_PATH"]) == ROOT / "data/world_model_v4/corpora/s1_v9/train.jsonl"
    legacy = train_contract.validate_train_contract(
        ROOT, ROOT / "config/world_model_s1_120b_training_v8_32n.json", METHOD,
    )
    assert legacy["SEGMENT_LOSS_TELEMETRY"] == "0"
    assert legacy.get("DIAGNOSTIC_SNAPSHOT_STEPS", "") == ""


@pytest.mark.parametrize(("field", "value", "error"), [
    ("initialization", None, "S1 v9 must start from an S0 v4 checkpoint"),
    ("training_id", "world_model_s1_120b_training_v8_32n", "S1 run identity changed"),
    ("corpus.root", "data/world_model_v4/corpora/s1_v8", "schedule/data contract changed"),
    ("run_settings.max_length", 8192, "schedule/data contract changed"),
    ("run_settings.num_train_epochs", 2, "schedule/data contract changed"),
    ("run_settings.diagnostic_snapshot_steps", [400, 800], "schedule/data contract changed"),
    ("trainer.segment_loss_telemetry", False, "v9 requires segment loss telemetry"),
])
def test_v9_rejects_changed_recipe_and_disabled_observability(tmp_path, field, value, error):
    """Reject drift before opening any expensive artifacts."""
    config = read(TRAIN)
    parent = config
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match=error):
        train_contract.validate_train_contract(tmp_path, path, METHOD)


@pytest.mark.parametrize(("config_path", "diagnostic", "rows", "nodes", "gate"), [
    (S1, "0", "6875", "512", "tiered_v3"),
    (S1_95, "0", "6875", "512", "structured_95_v1"),
    (ROOT / "config/world_model_s1_120b_32n_robustness_v9_32n.json", "1", "344", "32", "diagnostic_only"),
])
def test_v9_s1_preflight_checks_real_inputs_with_explicit_future_checkpoint(
    monkeypatch, config_path, diagnostic, rows, nodes, gate,
):
    """Stub only the unavailable future adapter, preserving panel/hash checks."""
    config = read(config_path)
    checked = []

    def future_checkpoint(**kwargs):
        checked.append(kwargs)
        return "explicit-future-v9-checkpoint-stub", "1" * 64

    resolve = s1_eval.resolve_repo_path

    def future_path(repo_root, value, label, **kwargs):
        if label in {"checkpoint root", "training output root"}:
            kwargs["must_exist"] = False
        return resolve(repo_root, value, label, **kwargs)

    monkeypatch.setattr(s1_eval, "_validate_checkpoint", future_checkpoint)
    monkeypatch.setattr(s1_eval, "resolve_repo_path", future_path)
    values = s1_eval.validate_and_export(
        repo_root=ROOT, config_path=config_path, method_id=METHOD,
        checkpoint_path=ROOT / config["training"]["checkpoint_root"] / "explicit-future-v9-checkpoint-stub",
    )
    assert values["S1_DIAGNOSTIC_ONLY"] == diagnostic
    assert values["S1_HISTORY_POLICY"] == "retain_current_turn"
    assert values.get("S1_GATE_PROFILE", "diagnostic_only") == gate
    assert values["S1_TEST_ROWS"] == rows
    assert values["S1_EVAL_NUM_NODES"] == nodes
    assert checked[0]["training_config_sha256"] == hashlib.sha256(TRAIN.read_bytes()).hexdigest()


def test_v9_gate_inputs_and_thresholds_match_v8():
    """New identities must preserve the frozen questions and promotion criteria."""
    for path, baseline_path in (
        (S1, ROOT / "config/world_model_s1_120b_32n_test_v8_retained_512n.json"),
        (S0, ROOT / "config/world_model_s1_120b_s0_forgetting_test_v8_512n.json"),
    ):
        config, baseline = read(path), read(baseline_path)
        assert config["generation"] == baseline["generation"]
        assert config["gate"] == baseline["gate"]
        assert config["tracking"] == baseline["tracking"]
        for key in ("questions_sha256", "answer_key_sha256", "test_panel_id", "row_count"):
            assert config["test_panel"][key] == baseline["test_panel"][key]
        assert config["training"]["config_sha256"] == hashlib.sha256(TRAIN.read_bytes()).hexdigest()
    assert read(S0)["schema_version"] == s0_eval.S1_PINNED_FORGETTING_TEST_SCHEMA_VERSION


def test_structured95_config_changes_only_gate_and_output_identity():
    """The relaxed gate preserves the checkpoint, frozen panel, and decoding."""
    original, revised = read(S1), read(S1_95)
    assert revised["evaluation_id"] != original["evaluation_id"]
    assert revised["methods"][0]["output_root"] != original["methods"][0]["output_root"]
    assert revised["gate"] == {"profile": "structured_95_v1", "require_network_used_false": True}
    revised["gate"] = original["gate"]
    revised["evaluation_id"] = original["evaluation_id"]
    revised["methods"][0]["output_root"] = original["methods"][0]["output_root"]
    assert revised == original


@pytest.mark.parametrize("change", [
    {"require_network_used_false": False},
    {"profile": "tiered_v3"},
    {"minimum_structured_answer_accuracy": 0.90},
])
def test_structured95_gate_cannot_silently_change_threshold_or_network_policy(change):
    """A named gate must retain its documented threshold and offline policy."""
    with pytest.raises(s1_eval.S1EvaluationError):
        s1_eval._validate_structured_95_gate({**read(S1_95)["gate"], **change})


def _pinned_forgetting_fixture(tmp_path):
    """Create only the files consumed by the new training-link check."""
    config = read(S0)
    training = read(TRAIN)
    training["methods"][0]["output_root"] = "checkpoints"
    raw = (json.dumps(training, indent=2) + "\n").encode()
    (tmp_path / "training.json").write_bytes(raw)
    checkpoint = tmp_path / "checkpoints/slurm-fixture"
    (checkpoint / "run_contract").mkdir(parents=True)
    (checkpoint / "run_contract/run_config.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    config["training"] = {
        "config": "training.json", "config_sha256": digest,
        "training_id": training["training_id"], "checkpoint_root": "checkpoints",
    }
    return config, config["methods"][0], checkpoint, {"run_config_sha256": digest}


def test_v9_forgetting_training_link_accepts_exact_final_identity(tmp_path):
    """Accept one adapter pinned to its production training configuration."""
    config, method, checkpoint, identity = _pinned_forgetting_fixture(tmp_path)
    s0_eval.validate_training_link(tmp_path, config, method, checkpoint, identity)


@pytest.mark.parametrize("change", ["missing_link", "wrong_config_hash", "wrong_adapter_hash", "wrong_root", "copied_config"])
def test_v9_forgetting_training_link_rejects_identity_drift(tmp_path, change):
    """Never evaluate an older corpus/config or a mismatched copied receipt."""
    config, method, checkpoint, identity = _pinned_forgetting_fixture(tmp_path)
    if change == "missing_link":
        config.pop("training")
    elif change == "wrong_config_hash":
        config["training"]["config_sha256"] = "0" * 64
    elif change == "wrong_adapter_hash":
        identity["run_config_sha256"] = "0" * 64
    elif change == "wrong_root":
        config["training"]["checkpoint_root"] = "another-output"
    else:
        (checkpoint / "run_contract/run_config.json").write_text("{}\n")
    with pytest.raises(SystemExit):
        s0_eval.validate_training_link(tmp_path, config, method, checkpoint, identity)


@pytest.mark.parametrize("complete", [True, False])
def test_v9_s0_preflight_checks_real_inputs_with_explicit_checkpoint_fixture(tmp_path, monkeypatch, complete):
    """Check real frozen inputs while keeping future-checkpoint fixtures explicit."""
    config = read(S0)
    checkpoint = tmp_path / "fixture-checkpoints/slurm-explicit-fixture"
    (checkpoint / "run_contract").mkdir(parents=True)
    (checkpoint / "run_contract/run_config.json").write_bytes(TRAIN.read_bytes())
    (checkpoint / "adapter_model.safetensors").write_bytes(b"explicit temporary adapter fixture")
    (checkpoint / "adapter_config.json").write_text('{"r":32}\n')
    run_id = METHOD + "-slurm-explicit-fixture"
    identity = {
        "method_id": METHOD, "run_id": run_id,
        "model_identity_sha256": config["model"]["model_identity_sha256"],
        "corpus_manifest_sha256": config["corpus"]["manifest_sha256"],
        "train_sha256": config["corpus"]["train_sha256"],
        "tokenizer_manifest_sha256": config["methods"][0]["tokenizer_manifest_sha256"],
        "run_config_sha256": config["training"]["config_sha256"],
    }
    adapter = {
        "format": "mentor-rl-s0-tp-lora-v1", "identity": identity,
        "lora": {"r": 32}, "objective": {"loss_contract": "multiplex_sft_v3"},
        "initialization": {"parent": "s0_checkpoint", "run_id": read(TRAIN)["initialization"]["run_id"]},
    }
    (checkpoint / "tp_adapter_manifest.json").write_text(json.dumps(adapter))
    exposure = {
        "schema_version": s0_eval.EXPOSURE_SCHEMA_VERSION,
        "status": "complete" if complete else "running", "run_id": run_id,
        "method_id": METHOD, "logical_exposure": {"all_eligible_train_rows_exposed": complete},
        "exposure_contract": {"satisfied": complete},
    }
    exposure["manifest_sha256"] = s0_eval.stable_sha256(exposure)
    (checkpoint / "run_contract/training_exposure.json").write_text(json.dumps(exposure))
    resolve = s0_eval.resolve_repo_path

    def fixture_path(root, value, label):
        if label in {"checkpoint root", "training output root"}:
            return checkpoint.parent
        return resolve(root, value, label)

    monkeypatch.setattr(s0_eval, "resolve_repo_path", fixture_path)
    monkeypatch.setattr("sys.argv", ["validator", str(ROOT), str(S0), METHOD, str(checkpoint)])
    if complete:
        assert s0_eval.main() == 0
    else:
        with pytest.raises(SystemExit, match="requires one complete checkpoint"):
            s0_eval.main()
