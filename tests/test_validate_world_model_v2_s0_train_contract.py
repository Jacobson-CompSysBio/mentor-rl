"""Test the canonical S0 train contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.validate_world_model_v2_s0_train_contract as train_contract
from runtime.world_model.s1_overlay import (
    S1_OVERLAY_V3_SCHEMA_VERSION,
    S1_OVERLAY_V4_SCHEMA_VERSION,
    S1_OVERLAY_V5_SCHEMA_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = REPO_ROOT / "scripts/validate_world_model_v2_s0_train_contract.py"
QUALIFICATION_CONFIG = (
    REPO_ROOT
    / "config/world_model_v2_s0_20b_tool_trajectory_qualification_v6.json"
)
TRAINING_CONFIG = (
    REPO_ROOT
    / "config/world_model_v2_s0_120b_tool_trajectory_training_v6.json"
)
S1_QUALIFICATION_CONFIG = (
    REPO_ROOT / "config/world_model_s1_20b_qualification_v1.json"
)
S1_SMOKE_CONFIG = REPO_ROOT / "config/world_model_s1_20b_smoke_v1.json"
S1_TRAINING_CONFIG = (
    REPO_ROOT / "config/world_model_s1_120b_training_v1.json"
)
V4_TRAINING_CONFIGS = [
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r32_uniform.json",
        "oss120b-plain-base-tokenizer-lora-r32",
        32,
        "multiplex_sft_v3",
    ),
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r32_priority.json",
        "oss120b-plain-base-tokenizer-lora-r32",
        32,
        "multiplex_sft_v4_priority",
    ),
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r128_uniform.json",
        "oss120b-plain-base-tokenizer-lora-r128",
        128,
        "multiplex_sft_v3",
    ),
    (
        REPO_ROOT / "config/world_model_s0_120b_training_v4_r128_priority.json",
        "oss120b-plain-base-tokenizer-lora-r128",
        128,
        "multiplex_sft_v4_priority",
    ),
]
QUALIFICATION_METHOD = "oss20b-plain-base-tokenizer-lora-r32"
TRAINING_METHOD = "oss120b-plain-base-tokenizer-lora-r32"
TRAIN_LAUNCHER = REPO_ROOT / "world_model_v2_s0.slurm"


def run_validator(
    config: Path,
    method_id: str,
    *,
    root: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run the train validator."""

    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(root),
            str(config),
            method_id,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("config", "method_id", "model_id", "run_scope"),
    [
        (
            QUALIFICATION_CONFIG,
            QUALIFICATION_METHOD,
            "gpt-oss-20b-bf16",
            "qualification",
        ),
        (
            TRAINING_CONFIG,
            TRAINING_METHOD,
            "gpt-oss-120b-bf16",
            "production",
        ),
    ],
)
def test_validator_accepts_the_v6_trajectory_contracts(
    config: Path,
    method_id: str,
    model_id: str,
    run_scope: str,
) -> None:
    """Accept each canonical trajectory train contract."""

    result = run_validator(config, method_id)

    assert result.returncode == 0, result.stderr
    assert f"MODEL_ID={model_id}" in result.stdout
    assert f"S0_RUN_SCOPE={run_scope}" in result.stdout
    assert "S0_TRAIN_ROWS=4096" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=128" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1280" in result.stdout
    assert "LOSS_CONTRACT=s0_tool_trajectory_v1" in result.stdout


def test_validator_accepts_the_s1_v3_qualification_contract() -> None:
    """Accept the aggregate S1 corpus with its dataset-aware contract."""

    result = run_validator(S1_QUALIFICATION_CONFIG, QUALIFICATION_METHOD)

    assert result.returncode == 0, result.stderr
    assert "S0_CONFIG_ID=world_model_s1_20b_qualification_v1" in result.stdout
    assert "S0_DATASET_ID=world_model_v3_s1_coordinate_system" in result.stdout
    assert "S0_TRAIN_ROWS=55276" in result.stdout
    assert "S0_VALIDATION_ROWS=6875" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=1728" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1728" in result.stdout
    assert "MAX_LENGTH=4096" in result.stdout
    assert "LOSS_CONTRACT=multiplex_sft_v3" in result.stdout


def test_validator_accepts_the_s1_v3_production_contract() -> None:
    """Accept the 120B production cell over the qualified S1 schedule."""

    result = run_validator(S1_TRAINING_CONFIG, TRAINING_METHOD)

    assert result.returncode == 0, result.stderr
    assert "S0_CONFIG_ID=world_model_s1_120b_training_v1" in result.stdout
    assert "S0_RUN_SCOPE=production" in result.stdout
    assert "S0_DATASET_ID=world_model_v3_s1_coordinate_system" in result.stdout
    assert "MODEL_ID=gpt-oss-120b-bf16" in result.stdout
    assert "S0_TRAIN_ROWS=55276" in result.stdout
    assert "S0_VALIDATION_ROWS=6875" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=1728" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1728" in result.stdout
    assert "MAX_LENGTH=4096" in result.stdout
    assert "LOSS_CONTRACT=multiplex_sft_v3" in result.stdout


def test_validator_accepts_the_s1_one_node_smoke_contract() -> None:
    """Accept the pinned behavior-stratified one-node smoke corpus."""

    result = run_validator(S1_SMOKE_CONFIG, QUALIFICATION_METHOD)

    assert result.returncode == 0, result.stderr
    assert "S0_CONFIG_ID=world_model_s1_20b_smoke_v1" in result.stdout
    assert "S0_DATASET_ID=world_model_v3_s1_coordinate_system" in result.stdout
    assert "S0_TRAIN_ROWS=248" in result.stdout
    assert "S0_VALIDATION_ROWS=111" in result.stdout
    assert "NUM_NODES=1" in result.stdout
    assert "DATA_PARALLEL_SIZE=1" in result.stdout
    assert "GLOBAL_BATCH_SIZE=2" in result.stdout
    assert "TRAIN_TOTAL_STEPS=124" in result.stdout


@pytest.mark.parametrize(
    ("config", "method_id", "rank", "loss_contract"),
    V4_TRAINING_CONFIGS,
    ids=("r32-uniform", "r32-priority", "r128-uniform", "r128-priority"),
)
def test_validator_accepts_each_v4_lora_and_loss_arm(
    config: Path,
    method_id: str,
    rank: int,
    loss_contract: str,
) -> None:
    result = run_validator(config, method_id)

    assert result.returncode == 0, result.stderr
    assert "S0_DATASET_ID=world_model_v4_s0_human_identifiers" in result.stdout
    assert "S0_TRAIN_ROWS=16384" in result.stdout
    assert "S0_VALIDATION_ROWS=622" in result.stdout
    assert "TRAIN_UPDATES_PER_EPOCH=512" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1536" in result.stdout
    assert "MAX_LENGTH=4096" in result.stdout
    assert "SNAPSHOT" not in result.stdout
    assert f"LORA_R={rank}" in result.stdout
    assert f"LORA_ALPHA={2 * rank}" in result.stdout
    assert f"LOSS_CONTRACT={loss_contract}" in result.stdout


@pytest.mark.parametrize(
    ("path", "field", "value", "message"),
    [
        (
            ("trainer", "lora_alpha"),
            "lora_alpha",
            63,
            "The v4 LoRA rank/alpha contract changed",
        ),
        (
            ("run_settings", "snapshot_steps"),
            "snapshot_steps",
            256,
            "The S0 v4 training/snapshot schedule changed",
        ),
        (
            ("run_settings", "snapshot_steps"),
            "snapshot_steps_enabled",
            128,
            "The S0 v4 training/snapshot schedule changed",
        ),
        (
            ("trainer", "token_loss_weights"),
            "token_loss_weights",
            None,
            "The S0 v4 priority loss weights changed",
        ),
    ],
)
def test_validator_rejects_v4_rank_snapshot_or_priority_drift(
    tmp_path: Path,
    path: tuple[str, str],
    field: str,
    value: object,
    message: str,
) -> None:
    config = json.loads(V4_TRAINING_CONFIGS[1][0].read_text(encoding="utf-8"))
    config[path[0]][path[1]] = value
    root = tmp_path / "repo"
    config_path = root / "config" / f"changed-{field}.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    result = run_validator(
        config_path,
        V4_TRAINING_CONFIGS[1][1],
        root=root,
    )

    assert result.returncode != 0
    assert message in result.stderr


S1_V2_TRAINING_CONFIG = (
    REPO_ROOT / "config/world_model_s1_120b_training_v2.json"
)


@pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s1/manifest.json").is_file(),
    reason="the S1 v2 corpus is not built",
)
def test_s1_v2_starts_from_the_s0_checkpoint_on_the_v2_corpus() -> None:
    result = run_validator(S1_V2_TRAINING_CONFIG, TRAINING_METHOD)

    assert result.returncode == 0, result.stderr
    assert "S0_TRAIN_ROWS=55276" in result.stdout
    assert "TRAIN_TOTAL_STEPS=1728" in result.stdout
    assert (
        "INIT_ADAPTER_RUN_ID=oss120b-plain-base-tokenizer-lora-r32-slurm-5631883"
        in result.stdout
    )


@pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s1/manifest.json").is_file(),
    reason="the S1 v2 corpus is not built",
)
def test_s1_v2_wide_profile_uses_one_row_per_replica() -> None:
    result = run_validator(
        REPO_ROOT / "config/world_model_s1_120b_training_v2_184n.json",
        TRAINING_METHOD,
    )

    assert result.returncode == 0, result.stderr
    for line in (
        "NUM_NODES=184",
        "GLOBAL_BATCH_SIZE=184",
        "TRAIN_UPDATES_PER_EPOCH=301",
        "TRAIN_TOTAL_STEPS=602",
        "LEARNING_RATE=0.0002",
        "WARMUP_RATIO=0.05",
    ):
        assert line in result.stdout


@pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s1_v3/manifest.json").is_file(),
    reason="the S1 v3 corpus is not built",
)
def test_s1_v3_wide_profile_uses_the_narrow_learning_rate_schedule() -> None:
    result = run_validator(
        REPO_ROOT / "config/world_model_s1_120b_training_v3_184n.json",
        TRAINING_METHOD,
    )

    assert result.returncode == 0, result.stderr
    for line in (
        "NUM_NODES=184",
        "GLOBAL_BATCH_SIZE=184",
        "TRAIN_NUM_EPOCHS=6",
        "TRAIN_UPDATES_PER_EPOCH=301",
        "TRAIN_TOTAL_STEPS=1806",
        "LEARNING_RATE=0.0001",
        "WARMUP_RATIO=0.03",
        "LR_SCHEDULER_TYPE=cosine",
    ):
        assert line in result.stdout


@pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s1_v3/manifest.json").is_file(),
    reason="the S1 v3 corpus is not built",
)
def test_s1_v3_three_epoch_profile_stops_at_903_steps() -> None:
    result = run_validator(
        REPO_ROOT / "config/world_model_s1_120b_training_v3_184n_3ep.json",
        TRAINING_METHOD,
    )

    assert result.returncode == 0, result.stderr
    for line in (
        "NUM_NODES=184",
        "TRAIN_NUM_EPOCHS=3",
        "TRAIN_TOTAL_STEPS=903",
        "LEARNING_RATE=0.0001",
        "WARMUP_RATIO=0.03",
    ):
        assert line in result.stdout


@pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s1_v3/manifest.json").is_file(),
    reason="the S1 v3 corpus is not built",
)
def test_s1_v3_32_node_profile_keeps_the_16_node_optimization() -> None:
    result = run_validator(
        REPO_ROOT / "config/world_model_s1_120b_training_v3_32n.json",
        TRAINING_METHOD,
    )

    assert result.returncode == 0, result.stderr
    for line in (
        "NUM_NODES=32",
        "GLOBAL_BATCH_SIZE=32",
        "GRAD_ACCUM_STEPS=1",
        "TRAIN_NUM_EPOCHS=1",
        "TRAIN_TOTAL_STEPS=1728",
        "LEARNING_RATE=0.0001",
        "WARMUP_RATIO=0.03",
    ):
        assert line in result.stdout


@pytest.mark.skipif(
    not (REPO_ROOT / "data/world_model_v4/corpora/s1_v4/manifest.json").is_file(),
    reason="the S1 v4 corpus is not built",
)
def test_s1_v4_32_node_profile_repeats_the_v3_32_node_run() -> None:
    result = run_validator(
        REPO_ROOT / "config/world_model_s1_120b_training_v4_32n.json",
        TRAINING_METHOD,
    )

    assert result.returncode == 0, result.stderr
    for line in (
        "NUM_NODES=32",
        "GLOBAL_BATCH_SIZE=32",
        "TRAIN_TOTAL_STEPS=1728",
        "LEARNING_RATE=0.0001",
    ):
        assert line in result.stdout
    assert "data/world_model_v4/corpora/s1_v4/train.jsonl" in result.stdout


def test_s1_v4_train_contract_uses_the_v3_overlay_schema() -> None:
    """Keep the v4 training preflight aligned with its normalization overlay."""

    assert train_contract.S1_OVERLAYS[train_contract.S1_V4_PRODUCTION_SCHEMA] == (
        "runtime/world_model/s1_v4.yaml",
        S1_OVERLAY_V3_SCHEMA_VERSION,
    )


def test_s1_v5_profile_preserves_the_v4_recipe_and_parent() -> None:
    """Validate the new data pins while keeping the comparison recipe fixed."""

    path = REPO_ROOT / "config/world_model_s1_120b_training_v5_32n.json"
    config = json.loads(path.read_text())
    baseline = json.loads(
        (REPO_ROOT / "config/world_model_s1_120b_training_v4_32n.json").read_text()
    )
    for key in ("run_settings", "trainer", "initialization", "validation_gate"):
        assert config[key] == baseline[key]
    assert train_contract.S1_OVERLAYS[train_contract.S1_V5_PRODUCTION_SCHEMA] == (
        "runtime/world_model/s1_v5.yaml", S1_OVERLAY_V4_SCHEMA_VERSION,
    )
    result = run_validator(path, TRAINING_METHOD)
    assert result.returncode == 0, result.stderr
    for line in ("NUM_NODES=32", "GLOBAL_BATCH_SIZE=32", "TRAIN_TOTAL_STEPS=1728"):
        assert line in result.stdout
    assert "data/world_model_v4/corpora/s1_v5/train.jsonl" in result.stdout


def test_s1_v6_profile_preserves_the_v5_recipe_and_parent() -> None:
    """Validate the finalized v6 pins with the same S0 parent and optimizer recipe."""

    path = REPO_ROOT / "config/world_model_s1_120b_training_v6_32n.json"
    config = json.loads(path.read_text())
    baseline = json.loads(
        (REPO_ROOT / "config/world_model_s1_120b_training_v5_32n.json").read_text()
    )
    for key in ("model", "run_settings", "trainer", "initialization", "validation_gate"):
        assert config[key] == baseline[key]
    assert config["initialization"]["checkpoint_path"] == (
        "data/world_model_v4/checkpoints/s0_120b/r32_uniform/slurm-5631883"
    )
    assert config["corpus"]["validation_sha256"] == baseline["corpus"]["validation_sha256"]
    old_method, new_method = baseline["methods"][0], config["methods"][0]
    assert {key: value for key, value in new_method.items() if key != "output_root"} == {
        key: value for key, value in old_method.items() if key != "output_root"
    }
    assert new_method["output_root"] == (
        "data/world_model_v4/checkpoints/s1_v6_120b_32n/"
        "oss120b-plain-base-tokenizer-lora-r32"
    )
    result = run_validator(path, TRAINING_METHOD)
    assert result.returncode == 0, result.stderr
    for line in (
        "S0_CONFIG_ID=world_model_s1_120b_training_v6_32n",
        "S0_DATASET_ID=world_model_v3_s1_coordinate_system",
        "NUM_NODES=32", "WORLD_SIZE_EXPECTED=256", "GLOBAL_BATCH_SIZE=32",
        "TRAIN_NUM_EPOCHS=1", "TRAIN_TOTAL_STEPS=1728", "LEARNING_RATE=0.0001",
        "MAX_LENGTH=4096", "LOSS_CONTRACT=multiplex_sft_v3",
        "INIT_ADAPTER_RUN_ID=oss120b-plain-base-tokenizer-lora-r32-slurm-5631883",
    ):
        assert line in result.stdout
    assert "data/world_model_v4/corpora/s1_v6/train.jsonl" in result.stdout


def test_s1_v6_uses_its_overlay_and_s1_row_contract() -> None:
    """V6 must resolve to the S1 trace contract, not the older S0 v6 fallback."""

    schema = train_contract.S1_V6_PRODUCTION_SCHEMA
    assert schema in train_contract.SCHEMAS
    assert schema in train_contract.S1_FULL_SCHEMAS
    assert train_contract.S1_OVERLAYS[schema] == (
        "runtime/world_model/s1_v6.yaml", S1_OVERLAY_V5_SCHEMA_VERSION,
    )
    assert train_contract._contract_for_schema(schema) == train_contract._contract_for_schema(
        train_contract.S1_V5_PRODUCTION_SCHEMA,
    )


@pytest.mark.parametrize(("mutation", "error"), [
    ("no_parent", "S1 v6 must start from an S0 v4 checkpoint"),
    ("old_run_id", "S1 run identity changed"),
    ("old_corpus", "S1 full training schedule/data contract changed"),
    ("old_evaluator", "S1 full training schedule/data contract changed"),
    ("learning_rate", "S1 full training schedule/data contract changed"),
    ("epochs", "S1 full training schedule/data contract changed"),
    ("context_length", "S1 full training schedule/data contract changed"),
])
def test_s1_v6_rejects_mixed_versions_and_changed_recipe(
    tmp_path: Path, mutation: str, error: str,
) -> None:
    """Reject mismatched identities and schedules before touching model artifacts."""

    config = json.loads(
        (REPO_ROOT / "config/world_model_s1_120b_training_v5_32n.json").read_text()
    )
    config["schema_version"] = train_contract.S1_V6_PRODUCTION_SCHEMA
    config["training_id"] = "world_model_s1_120b_training_v6_32n"
    config["corpus"]["root"] = "data/world_model_v4/corpora/s1_v6"
    config["corpus"]["evaluator_manifest"] = "data/world_model_v4/eval/s1_v6/manifest.json"
    if mutation == "no_parent":
        config.pop("initialization")
    elif mutation == "old_run_id":
        config["training_id"] = "world_model_s1_120b_training_v5_32n"
    elif mutation == "old_corpus":
        config["corpus"]["root"] = "data/world_model_v4/corpora/s1_v5"
    elif mutation == "old_evaluator":
        config["corpus"]["evaluator_manifest"] = "data/world_model_v4/eval/s1_v5/manifest.json"
    else:
        key, value = {
            "learning_rate": ("learning_rate", 0.0002),
            "epochs": ("num_train_epochs", 2),
            "context_length": ("max_length", 8192),
        }[mutation]
        config["run_settings"][key] = value
    root = tmp_path / "repo"
    config_path = root / "config" / "s1-v6-invalid.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    result = run_validator(config_path, TRAINING_METHOD, root=root)
    assert result.returncode != 0
    assert error in result.stderr


def test_s1_v3_requires_an_s0_parent(tmp_path: Path) -> None:
    config = json.loads(
        (REPO_ROOT / "config/world_model_s1_120b_training_v3_184n.json").read_text(
            encoding="utf-8"
        )
    )
    config.pop("initialization")
    root = tmp_path / "repo"
    config_path = root / "config" / "s1-v3-without-parent.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    result = run_validator(config_path, TRAINING_METHOD, root=root)

    assert result.returncode != 0
    assert "S1 v3 must start from an S0 v4 checkpoint" in result.stderr


def test_s1_v2_rejects_settings_outside_its_profiles(tmp_path: Path) -> None:
    config = json.loads(S1_V2_TRAINING_CONFIG.read_text(encoding="utf-8"))
    config["run_settings"]["learning_rate"] = 0.0002
    root = tmp_path / "repo"
    config_path = root / "config" / "s1-v2-mixed-profile.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    result = run_validator(config_path, TRAINING_METHOD, root=root)

    assert result.returncode != 0
    assert "S1 full training schedule/data contract changed" in result.stderr


def test_s1_v2_requires_an_s0_parent(tmp_path: Path) -> None:
    config = json.loads(S1_V2_TRAINING_CONFIG.read_text(encoding="utf-8"))
    config.pop("initialization")
    root = tmp_path / "repo"
    config_path = root / "config" / "s1-v2-without-parent.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    result = run_validator(config_path, TRAINING_METHOD, root=root)

    assert result.returncode != 0
    assert "S1 v2 must start from an S0 v4 checkpoint" in result.stderr


@pytest.mark.parametrize(
    ("config", "method_id"),
    [
        (
            S1_QUALIFICATION_CONFIG,
            QUALIFICATION_METHOD,
        ),
        (
            S1_TRAINING_CONFIG,
            TRAINING_METHOD,
        ),
    ],
)
def test_shared_slurm_launcher_resolves_s1_train_contracts(
    config: Path,
    method_id: str,
) -> None:
    """Resolve S1 train configs through the executable launcher entrypoint."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_")
    }
    environment.update(
        {
            "CONTRACT_ONLY": "1",
            "JOB_MODE": "train",
            "RUN_CONFIG": str(config),
            "S0_METHOD_ID": method_id,
        }
    )
    result = subprocess.run(
        ["bash", str(TRAIN_LAUNCHER)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Run: {method_id}-slurm-JOB-ID" in result.stdout
    assert "The selected inputs and config are valid" in result.stdout


def test_shared_slurm_launcher_retries_rank_preflight() -> None:
    """Keep transient Slurm step failures outside the expensive train launch."""

    launcher = TRAIN_LAUNCHER.read_text(encoding="utf-8")

    assert 'S0_PREFLIGHT_ATTEMPTS="${S0_PREFLIGHT_ATTEMPTS:-3}"' in launcher
    assert "if run_rank_preflight; then" in launcher
    assert 'rm -f -- "${rank_receipt_dir}"/rank-*.tsv' in launcher
    assert '"rank_preflight_attempts=${S0_PREFLIGHT_ATTEMPTS}"' in launcher


def test_validator_rejects_an_s1_gate_the_trainer_cannot_execute(
    tmp_path: Path,
) -> None:
    """Reject a gate that promises evaluation at interim checkpoints."""

    config = json.loads(
        S1_QUALIFICATION_CONFIG.read_text(encoding="utf-8")
    )
    config["validation_gate"]["checkpoint_set"] = (
        "each_saved_checkpoint_and_final"
    )
    root = tmp_path / "repo"
    config_path = root / "config/world_model_s1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_validator(
        config_path,
        QUALIFICATION_METHOD,
        root=root,
    )

    assert result.returncode != 0
    assert "must select the final checkpoint" in result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_id", "world_model_s1_20b_qualification_v1"),
        ("run_scope", "qualification"),
        ("promotion_eligible", False),
        ("production_matrix_cell", False),
    ],
)
def test_validator_rejects_a_changed_s1_production_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Keep the 120B config pinned to its production matrix cell."""

    config = json.loads(S1_TRAINING_CONFIG.read_text(encoding="utf-8"))
    config[field] = value
    root = tmp_path / "repo"
    config_path = root / "config/world_model_s1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_validator(config_path, TRAINING_METHOD, root=root)

    assert result.returncode != 0
    assert "The S1 run identity changed" in result.stderr


@pytest.mark.parametrize(
    ("config_path", "method_id"),
    [
        (S1_QUALIFICATION_CONFIG, QUALIFICATION_METHOD),
        (S1_TRAINING_CONFIG, TRAINING_METHOD),
    ],
    ids=("qualification", "production"),
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_nodes", 8),
        ("gpus_per_node", 4),
        ("tp_size", 4),
        ("ep_size", 4),
        ("data_parallel_size", 8),
        ("num_train_epochs", 2),
        ("per_device_train_batch_size", 2),
        ("gradient_accumulation_steps", 1),
        ("global_batch_size", 64),
        ("learning_rate", 0.0002),
        ("lr_scheduler_type", "linear"),
        ("warmup_ratio", 0.04),
        ("weight_decay", 0.01),
        ("max_grad_norm", 0.5),
        ("max_length", 2048),
        ("logging_steps", 5),
        ("eval_strategy", "steps"),
        ("eval_on_start", False),
        ("per_device_eval_batch_size", 2),
        ("save_strategy", "no"),
        ("save_steps", 100),
        ("save_total_limit", 2),
        ("seed", 900914),
        ("preserve_dataset_order", True),
        ("autograd_multithreading", False),
        ("local_files_only", False),
        ("strict_tested_stack", False),
        ("bf16", False),
    ],
)
def test_full_s1_schemas_pin_every_run_setting_before_artifact_checks(
    tmp_path: Path,
    config_path: Path,
    method_id: str,
    field: str,
    value: object,
) -> None:
    """Reject full-run schedule drift before reading pinned artifacts."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["run_settings"][field] = value
    root = tmp_path / "repo"
    mutated_path = root / "config/world_model_s1.json"
    mutated_path.parent.mkdir(parents=True)
    mutated_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
        match="The S1 full training schedule/data contract changed",
    ):
        train_contract.validate_train_contract(
            root,
            mutated_path,
            method_id,
        )


@pytest.mark.parametrize(
    ("config_path", "method_id"),
    [
        (S1_QUALIFICATION_CONFIG, QUALIFICATION_METHOD),
        (S1_TRAINING_CONFIG, TRAINING_METHOD),
    ],
    ids=("qualification", "production"),
)
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root", "data/world_model_v3/corpora/not_s1"),
        ("train_file", "other_train.jsonl"),
        ("train_rows", 55275),
        ("validation_file", "other_validation.jsonl"),
        ("validation_rows", 6874),
        ("evaluator_manifest", "data/world_model_v3/eval/not_s1.json"),
    ],
)
def test_full_s1_schemas_pin_the_training_data_layout_before_hash_checks(
    tmp_path: Path,
    config_path: Path,
    method_id: str,
    field: str,
    value: object,
) -> None:
    """Reject S1 data-layout drift before the known contract-hash drift."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["corpus"][field] = value
    root = tmp_path / "repo"
    mutated_path = root / "config/world_model_s1.json"
    mutated_path.parent.mkdir(parents=True)
    mutated_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SystemExit,
        match="The S1 full training schedule/data contract changed",
    ):
        train_contract.validate_train_contract(
            root,
            mutated_path,
            method_id,
        )


@pytest.mark.parametrize(
    ("config_path", "method_id"),
    [
        (S1_QUALIFICATION_CONFIG, QUALIFICATION_METHOD),
        (S1_TRAINING_CONFIG, TRAINING_METHOD),
    ],
    ids=("qualification", "production"),
)
def test_full_s1_schemas_pin_the_derived_optimizer_steps_before_hash_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    method_id: str,
) -> None:
    """Reject a changed derived schedule before reading pinned artifacts."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = tmp_path / "repo"
    copied_path = root / "config/world_model_s1.json"
    copied_path.parent.mkdir(parents=True)
    copied_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        train_contract,
        "derive_optimizer_step_schedule",
        lambda *args, **kwargs: {
            "batches_per_epoch": 3455,
            "updates_per_epoch": 1727,
            "total_steps": 1727,
        },
    )

    with pytest.raises(
        SystemExit,
        match="The S1 full optimizer-step schedule changed",
    ):
        train_contract.validate_train_contract(
            root,
            copied_path,
            method_id,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "max_steps",
            96,
            "max_steps is not supported",
        ),
        (
            "run_scope",
            "unknown",
            "run_scope must be one of",
        ),
    ],
)
def test_validator_rejects_removed_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject one value outside the canonical contract."""

    root = tmp_path / "repo"
    config_path = root / "config/run.json"
    config_path.parent.mkdir(parents=True)
    config = json.loads(
        QUALIFICATION_CONFIG.read_text(encoding="utf-8")
    )
    if field == "max_steps":
        config["run_settings"][field] = value
    else:
        config[field] = value
    config_path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    result = run_validator(
        config_path,
        QUALIFICATION_METHOD,
        root=root,
    )

    assert result.returncode != 0
    assert message in result.stderr
