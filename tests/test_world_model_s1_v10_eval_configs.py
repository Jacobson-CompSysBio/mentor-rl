"""Check current v10 admission, exact pins, and rejection of incomplete checkpoints."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import validate_world_model_s1_eval_contract as s1_eval
from scripts import validate_world_model_v2_s0_eval_contract as s0_eval
from scripts import validate_world_model_v2_s0_train_contract as train_contract


ROOT = Path(__file__).resolve().parents[1]
METHOD = "oss120b-plain-base-tokenizer-lora-r32"
TRAIN = ROOT / "config/world_model_s1_120b_training_v10_32n.json"
S1 = ROOT / "config/world_model_s1_120b_32n_test_v10_structured95_512n.json"
S0 = ROOT / "config/world_model_s1_120b_s0_forgetting_test_v10_512n.json"


def read(path):
    """Read one pinned run configuration."""

    return json.loads(path.read_text())


@pytest.mark.parametrize("path", [S1, S0])
def test_v10_eval_binds_training_identity(path):
    """Both evaluations bind the exact v10 recipe and use sequential run tracking."""

    config = read(path)
    training = read(TRAIN)
    assert config["training"] == {
        "config": TRAIN.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
        "training_id": training["training_id"],
        "checkpoint_root": training["methods"][0]["output_root"],
    }
    for key in ("root", "manifest_sha256", "train_sha256"):
        assert config["corpus"][key] == training["corpus"][key]
    assert config["generation"]["num_nodes"] == 512
    assert config["tracking"]["resume_training_run"] is True


def test_v10_s1_readiness_with_passed_checkpoint():
    """Validate actual corpus, tokenizer, exposure, and completed adapter pins."""
    config = read(S1)
    values = s1_eval.validate_and_export(
        repo_root=ROOT, config_path=S1, method_id=METHOD,
        checkpoint_path=ROOT / config["training"]["checkpoint_root"] / "slurm-5768596",
    )
    assert values["S1_HISTORY_POLICY"] == "retain_current_turn"
    assert values["S1_GATE_PROFILE"] == "structured_95_v1"
    assert values["S1_TEST_ROWS"] == "6875"
    assert values["S1_EVAL_NUM_NODES"] == "512"


@pytest.mark.parametrize("complete", [True, False])
def test_v10_s0_readiness_with_explicit_checkpoint_fixture(tmp_path, monkeypatch, capsys, complete):
    """Validate real retention inputs and reject incomplete training using a temporary adapter fixture."""

    config = read(S0)
    checkpoint = tmp_path / "fixture-checkpoints/slurm-explicit-v10-fixture"
    (checkpoint / "run_contract").mkdir(parents=True)
    (checkpoint / "run_contract/run_config.json").write_bytes(TRAIN.read_bytes())
    (checkpoint / "adapter_model.safetensors").write_bytes(b"explicit temporary adapter fixture")
    (checkpoint / "adapter_config.json").write_text('{"r":32}\n')
    run_id = METHOD + "-slurm-explicit-v10-fixture"
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
        output = capsys.readouterr().out
        assert "120766" in output and "512" in output
    else:
        with pytest.raises(SystemExit, match="requires one complete checkpoint"):
            s0_eval.main()


def test_v10_training_admission_pins_parent_and_schedule():
    values = train_contract.validate_train_contract(ROOT, TRAIN, METHOD)
    assert values["S0_DATASET_ID"] == "world_model_v3_s1_coordinate_system"
    assert values["TRAIN_TOTAL_STEPS"] == "1728"
    assert values["GLOBAL_BATCH_SIZE"] == "32"
    assert values["INIT_ADAPTER_RUN_ID"].endswith("slurm-5631883")
    assert values["SEGMENT_LOSS_TELEMETRY"] == "1"
    assert values["DIAGNOSTIC_SNAPSHOT_STEPS"] == "400,800,1200,1600"


@pytest.mark.parametrize("profile", ["promotion_v1", "tiered_v2", "tiered_v3"])
def test_retired_gate_profiles_are_rejected(profile):
    with pytest.raises(s1_eval.S1EvaluationError, match="profile changed"):
        s1_eval._validate_structured_95_gate({"profile": profile, "require_network_used_false": True})
