"""Verify v10 gate readiness while keeping future checkpoint fixtures explicit."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import validate_world_model_s1_eval_contract as s1_eval
from scripts import validate_world_model_v2_s0_eval_contract as s0_eval


ROOT = Path(__file__).resolve().parents[1]
METHOD = "oss120b-plain-base-tokenizer-lora-r32"
TRAIN = ROOT / "config/world_model_s1_120b_training_v10_32n.json"
S1 = ROOT / "config/world_model_s1_120b_32n_test_v10_structured95_512n.json"
S0 = ROOT / "config/world_model_s1_120b_s0_forgetting_test_v10_512n.json"


def read(path):
    """Read one pinned run configuration."""

    return json.loads(path.read_text())


@pytest.mark.parametrize("path", [S1, S0])
def test_v10_eval_changes_only_artifact_and_output_identity(path):
    """Keep frozen panels, decoding, gates, and sequential training-run tracking unchanged."""

    config = read(path)
    old = read(Path(str(path).replace("v10", "v9")))
    normalized = deepcopy(config)
    for key in ("evaluation_id", "training", "corpus"):
        normalized[key] = old[key]
    normalized["methods"][0]["output_root"] = old["methods"][0]["output_root"]
    if path == S1:
        for key in ("evaluator_root", "evaluator_manifest_sha256"):
            normalized["test_panel"][key] = old["test_panel"][key]
    assert normalized == old
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


def test_v10_s1_readiness_with_explicit_future_checkpoint(monkeypatch):
    """Check real data/config/tokenizer pins while stubbing only the unavailable future adapter."""

    config = read(S1)
    checked = []

    def future_checkpoint(**kwargs):
        checked.append(kwargs)
        return "explicit-future-v10-checkpoint-stub", "1" * 64

    resolve = s1_eval.resolve_repo_path

    def future_path(repo_root, value, label, **kwargs):
        if label in {"checkpoint root", "training output root"}:
            kwargs["must_exist"] = False
        return resolve(repo_root, value, label, **kwargs)

    monkeypatch.setattr(s1_eval, "_validate_checkpoint", future_checkpoint)
    monkeypatch.setattr(s1_eval, "resolve_repo_path", future_path)
    values = s1_eval.validate_and_export(
        repo_root=ROOT, config_path=S1, method_id=METHOD,
        checkpoint_path=ROOT / config["training"]["checkpoint_root"] / "explicit-future-v10-checkpoint-stub",
    )
    assert values["S1_DIAGNOSTIC_ONLY"] == "0"
    assert values["S1_HISTORY_POLICY"] == "retain_current_turn"
    assert values["S1_GATE_PROFILE"] == "structured_95_v1"
    assert values["S1_TEST_ROWS"] == "6875" and values["S1_EVAL_NUM_NODES"] == "512"
    assert checked[0]["training_config_sha256"] == hashlib.sha256(TRAIN.read_bytes()).hexdigest()


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
