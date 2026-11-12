"""Check the frozen S2 eval contract and its two-mode Slurm entrypoint."""

from __future__ import annotations

import hashlib
import json
from itertools import islice
from pathlib import Path
import subprocess

import pytest

import scripts.validate_world_model_s2_eval_contract as s2_validator
from scripts.validate_world_model_s2_eval_contract import (
    S2EvaluationContractError,
    validate_and_export,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config/world_model_s2_120b_test_v1_512n.json"
CHECKPOINT = (
    REPO_ROOT
    / "data/world_model_v4/checkpoints/s2_v1_184n/"
    "oss120b-plain-base-tokenizer-lora-r32/slurm-5856263"
)
METHOD_ID = "oss120b-plain-base-tokenizer-lora-r32"


def test_s2_eval_config_pins_report_only_completed_run() -> None:
    """The run config names only the completed run and frozen ordered panel."""

    config = json.loads(CONFIG.read_text())
    assert config["schema_version"] == "mentor-rl-world-model-s2-test-v1"
    assert config["evaluation_contract"] == "held_out_s2_structured_rollout_v1"
    assert config["training"] == {
        "config": "config/world_model_s2_120b_training_v1_184n.json",
        "config_sha256": "bd7687f1cd6cbc037bfde3ecc617196a509fde855d953311b70b234e10790f25",
        "training_id": "world_model_s2_120b_training_v1_184n",
        "checkpoint_root": "data/world_model_v4/checkpoints/s2_v1_184n/oss120b-plain-base-tokenizer-lora-r32",
        "checkpoint_path": "data/world_model_v4/checkpoints/s2_v1_184n/oss120b-plain-base-tokenizer-lora-r32/slurm-5856263",
        "train_run_id": "oss120b-plain-base-tokenizer-lora-r32-slurm-5856263",
    }
    assert config["test_panel"]["evaluator_manifest_sha256"] == (
        "79b0fcfd04c4ed9eedb9d398119bf6be78a7e908e165873ea3d1637d23246ee5"
    )
    assert config["test_panel"]["questions_sha256"] == (
        "0b60dc7ea96031baa5b87a000a70e9de5eb20a10b5febb39ac078ac00c9471c9"
    )
    assert config["test_panel"]["answer_key_sha256"] == (
        "e27144117ec11e2e65710ccc37dda3fefb5dc92f2c11a2cc13f9b6b05f7ed811"
    )
    assert config["generation"] == {
        "num_nodes": 512,
        "tasks_per_node": 1,
        "gpus_per_node": 8,
        "batch_size": 4,
        "max_tool_call_tokens": 1024,
        "max_final_prompt_tokens": 8191,
        "max_new_tokens": 4096,
        "max_total_tokens": 8192,
        "maximum_tool_calls": 4,
        "do_sample": False,
        "enable_thinking": True,
        "reasoning_effort": "low",
        "history_policy": "retain_current_turn",
        "seed": 900913,
        "local_files_only": True,
    }
    assert config["gate"] == {
        "profile": "report_only_v1",
        "minimum_structured_accuracy": None,
        "require_complete_coverage": True,
        "require_network_used_false": True,
    }
    assert config["tracking"]["resume_policy"] == "must"
    assert config["tracking"]["publish_after_complete_scoring"] is True


@pytest.mark.skipif(not CHECKPOINT.is_dir(), reason="completed S2 checkpoint is absent")
def test_s2_eval_validator_accepts_real_completed_checkpoint() -> None:
    """The checked-in contract accepts the actual final S2 artifact set."""

    values = validate_and_export(
        repo_root=REPO_ROOT,
        config_path=CONFIG,
        method_id=METHOD_ID,
        checkpoint_path=CHECKPOINT,
    )
    assert values["S2_TRAIN_RUN_ID"] == (
        "oss120b-plain-base-tokenizer-lora-r32-slurm-5856263"
    )
    assert values["S2_CHECKPOINT_ARTIFACT_SHA256"] == (
        "c42aee403e0914cb84e8bf5b5ce3819f9095002cf8524b7868cf22452d780596"
    )
    assert values["S2_EVALUATOR_MANIFEST_SHA256"] == (
        "79b0fcfd04c4ed9eedb9d398119bf6be78a7e908e165873ea3d1637d23246ee5"
    )
    assert values["S2_RUNTIME_CONTRACT_SHA256"] == (
        "f629c28de87a0440b40130d78614d4c5f65d55cd32e68b7bd144d09df81d9f0f"
    )
    assert values["S2_TEST_ROWS"] == "25216"
    assert values["S2_GATE_PROFILE"] == "report_only_v1"


def test_s2_eval_validator_fails_closed_before_artifact_access() -> None:
    """An unknown method is rejected before any launch can proceed."""

    with pytest.raises(S2EvaluationContractError, match="METHOD_ID"):
        validate_and_export(
            repo_root=REPO_ROOT,
            config_path=CONFIG,
            method_id="unknown-method",
            checkpoint_path=CHECKPOINT,
        )


def test_training_config_pins_the_runtime_contract_used_by_generation() -> None:
    """Evaluation reuses the exact runtime contract that produced training."""

    training = json.loads(
        (REPO_ROOT / "config/world_model_s2_120b_training_v1_184n.json").read_text()
    )
    wire_paths = {
        "runtime/world_model/contract.py",
        "runtime/world_model/contract.yaml",
        "runtime/world_model/sources.py",
        "runtime/world_model/s2_3_tokens.py",
        "runtime/world_model/s2_production_adapter.py",
        "runtime/world_model_training.py",
    }
    pins = {
        item["path"]: item["sha256"]
        for item in training["implementation_files"]
        if item["path"] in wire_paths
    }
    assert set(pins) == wire_paths
    assert pins["runtime/world_model/contract.yaml"] == (
        "f629c28de87a0440b40130d78614d4c5f65d55cd32e68b7bd144d09df81d9f0f"
    )
    for relative_path, expected in pins.items():
        actual = hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected
    launcher = (REPO_ROOT / "world_model_s2_eval.slurm").read_text()
    assert all(relative_path in launcher for relative_path in wire_paths)


def test_validator_rejects_training_wire_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued job cannot silently use a changed runtime contract."""

    real_sha256_file = s2_validator.sha256_file

    def drift_contract(path: Path) -> str:
        if path.resolve() == (REPO_ROOT / "runtime/world_model/contract.yaml").resolve():
            return "0" * 64
        return real_sha256_file(path)

    monkeypatch.setattr(s2_validator, "sha256_file", drift_contract)
    with pytest.raises(S2EvaluationContractError, match="training wire changed"):
        validate_and_export(
            repo_root=REPO_ROOT,
            config_path=CONFIG,
            method_id=METHOD_ID,
            checkpoint_path=CHECKPOINT,
        )


def test_preflight_row_is_the_frozen_four_call_path() -> None:
    """Index 3100 is a deterministic one-row stride exercising four tool turns."""

    evaluator = REPO_ROOT / "data/world_model_v4/eval/s2_v1"
    with (evaluator / "test_questions.jsonl").open() as source:
        question = json.loads(next(islice(source, 3100, None)))
    with (evaluator / "test_answer_key.jsonl").open() as source:
        answer = json.loads(next(islice(source, 3100, None)))
    assert question["row_id"] == answer["row_id"]
    assert len(answer["target"]["expected_tool_calls"]) == 4
    assert len(answer["target"]["expected_tool_responses"]) == 4


def test_s2_eval_slurm_is_one_direct_two_mode_entrypoint() -> None:
    """The Slurm file stages concurrently and isolates preflight from scoring."""

    launcher = REPO_ROOT / "world_model_s2_eval.slurm"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    source = launcher.read_text()
    assert "#SBATCH -N 520" in source
    assert "#SBATCH -t 08:00:00" in source
    assert "#SBATCH -p batch" in source
    assert "PREFLIGHT_ONLY=1" in source
    assert "EVAL_ALLOCATION_NODES=1" in source
    assert "preflight-slurm-${execution_id}" in source
    assert "for S2_PREFLIGHT_SHARD_INDEX in 3100 11421; do" in source
    assert 'S2_PREFLIGHT_LONG_ROW_ID="s2prod_e16d4ce1f1ddb81075d6ccbc288a8bc9"' in source
    assert "first_prompt_tokens < 6000" in source
    assert '--shard-count "${S2_TEST_ROWS}"' in source
    assert '--batch-size 1' in source
    assert 'len(fixture.get("steps", [])) != 4' in source
    assert "No scoring or W&B publication ran." in source
    assert 'predictions[0].get("network_used_false") is not True' in source
    assert '"reads_private_answer_keys": False' in source
    assert "python_audit_deny_outbound_socket_v1" in source
    assert "WANDB_API_KEY is required before the full 512-node evaluation" in source
    assert 'srun --kill-on-bad-exit=0 --nodes="${SLURM_NNODES}"' in source
    assert "stage_model_wave" not in source
    assert "--gate-profile \"${S2_GATE_PROFILE}\"" in source
    assert "--publish-wandb" in source
    assert source.rindex("run_full_evaluation") > source.index(
        "No scoring or W&B publication ran."
    )
    assert "scripts/submit_world_model_s2" not in source


def test_s2_eval_slurm_survives_one_bad_node() -> None:
    """One node's fault leaves it out instead of killing all 512 shards."""

    source = (REPO_ROOT / "world_model_s2_eval.slurm").read_text()
    # Every srun step opts out of Frontier's cluster-wide KillOnBadExit=1,
    # except the one-row preflight, where any failure must stop the job.
    steps = source.count("srun --")
    assert steps == 4
    assert source.count("--kill-on-bad-exit=0") == steps - 1
    assert source.count("--kill-on-bad-exit=1") == 1
    assert "(( SLURM_NNODES >= EVAL_ALLOCATION_NODES ))" in source
    assert "for attempt in 1 2; do" in source
    assert 'touch "${STAGED_NODES_DIR}/${SLURMD_NODENAME}"' in source
    assert '"${STAGED_NODES[@]:0:EVAL_ALLOCATION_NODES}"' in source
    assert source.count('--nodelist="${GENERATION_NODES_FILE}"') == 2
    assert 'touch "${FAILED_NODES_DIR}/${SLURMD_NODENAME}"' in source
    assert 'shard_index="${S2_SHARD_INDEX:-${SLURM_PROCID}}"' in source
    assert "unset S2_SHARD_INDEX" in source
    assert 'S2_GENERATION_ATTEMPTS=1' in source
    assert source.index("repair_failed_shards\n") < source.index(
        "scripts/merge_world_model_s2_generation_shards.py \\"
    )


def test_preflight_long_row_is_the_panel_longest_prompt() -> None:
    """Index 11421 is frozen as the 6,350-token prompt that job 5892323 failed."""

    evaluator = REPO_ROOT / "data/world_model_v4/eval/s2_v1"
    with (evaluator / "test_questions.jsonl").open() as source:
        question = json.loads(next(islice(source, 11421, None)))
    assert question["row_id"] == "s2prod_e16d4ce1f1ddb81075d6ccbc288a8bc9"
    assert question["metadata"]["question_family"] == "S2.8"
