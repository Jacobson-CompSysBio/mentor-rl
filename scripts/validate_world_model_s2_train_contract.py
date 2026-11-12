#!/usr/bin/env python3
"""Validate the pinned 184-node S2 run without loading GPU libraries.

The launcher sources the default shell output. Full base-model hashing remains
an explicit staging check, using the unchanged S0/S1 artifact validator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import derive_optimizer_step_schedule  # noqa: E402
from scripts.validate_world_model_v2_s0_train_contract import (  # noqa: E402
    read_json, require_object, require_sha256, sha256_file, stable_sha256,
    validate_base_model, verify_file,
)
from scripts.validate_world_model_s1_eval_contract import (  # noqa: E402
    _model_metadata_identity, _validate_checkpoint, _validate_tokenizer,
)

SCHEMA = "mentor-rl-world-model-s2-120b-training-v1"
TRAINING_ID = "world_model_s2_120b_training_v1_184n"
METHOD_ID = "oss120b-plain-base-tokenizer-lora-r32"
CORPUS_ROOT = "data/world_model_v4/corpora/s2_production_v1_8k"
CORPUS_SHA256 = "2d4e0613f3327e0e98fef908b453327cefea69350bd8d16c3f96f22acdfb1dfc"
SPLITS = {
    "train": (252160, "aa6f22297c0d7f6b1b6aeba6e08043bf31dff0d2fc98a6a46c1dc4c5aa57e8e1"),
    "validation": (25216, "355fb0245a6d8d5d8447ebfe1237086780d57542754fff0c15e6943c36272f29"),
}
RUN_SETTINGS = {
    "num_nodes": 184, "gpus_per_node": 8, "tp_size": 8, "ep_size": 8,
    "data_parallel_size": 184, "num_train_epochs": 1,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
    "global_batch_size": 184, "learning_rate": 0.0001,
    "lr_scheduler_type": "cosine", "warmup_ratio": 0.03,
    "weight_decay": 0, "max_grad_norm": 1, "max_length": 8192,
    "logging_steps": 10, "eval_strategy": "steps", "eval_on_start": True,
    "per_device_eval_batch_size": 1, "save_strategy": "steps",
    "save_steps": 200, "save_total_limit": 3, "seed": 900913,
    "preserve_dataset_order": False, "autograd_multithreading": True,
    "local_files_only": True, "strict_tested_stack": True, "bf16": True,
    "diagnostic_snapshot_steps": [200, 400, 600, 800, 1000, 1200, 1371],
    "max_optimizer_updates": 1371, "eval_steps": 200, "eval_on_epoch_end": True,
}
CORE_SOURCE_PATHS = {
    "scripts/validate_world_model_s2_train_contract.py",
    "scripts/validate_world_model_v2_s0_train_contract.py",
    "scripts/validate_world_model_s1_eval_contract.py",
    "scripts/world_model_s1_eval_common.py",
    "scripts/train_sft_s2_dp_tp_ep.py", "scripts/train_sft_dp_tp_ep.py",
    "scripts/world_model_dp_tp_ep.py", "scripts/tp_ep_autograd.py",
    "scripts/submit_world_model_s2.sh", "world_model_s2.slurm",
    "world_model_s2_check.slurm", "world_model_s2_prepare.slurm",
    "runtime/world_model_training.py", "runtime/world_model_loss_metrics.py",
    "runtime/world_model/s2_training.py", "runtime/world_model/s2_loss_metrics.py",
    "runtime/world_model/contract.yaml",
}


def repo_path(root: Path, value: str) -> Path:
    """Resolve a required repository path and reject traversal or symlinks."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise SystemExit("A pinned path must be repository-relative")
    path = root / value
    try:
        canonical = path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise SystemExit(f"A pinned path escapes the repository: {value}") from error
    if path.is_symlink() or canonical != value:
        raise SystemExit(f"A pinned path is not canonical: {value}")
    return path


def checked_reference(root: Path, reference: object, name: str) -> tuple[Path, dict]:
    """Verify one exact file pin and read its JSON object."""
    pin = require_object(reference, name)
    if set(pin) != {"path", "sha256"}:
        raise SystemExit(f"The {name} pin fields changed")
    path = repo_path(root, pin["path"])
    verify_file(path, require_sha256(pin["sha256"], name), name)
    return path, read_json(path, name)


def validate_recipe(config: dict) -> None:
    """Reject accidental recipe, allocation, objective, or resume changes."""
    if (config.get("schema_version") != SCHEMA or config.get("training_id") != TRAINING_ID
            or config.get("run_scope") != "production"):
        raise SystemExit("The S2 training identity changed")
    if stable_sha256(config.get("run_settings")) != stable_sha256(RUN_SETTINGS):
        raise SystemExit("The S2 training schedule changed")
    schedule = derive_optimizer_step_schedule(
        SPLITS["train"][0], num_train_epochs=1, replica_count=184,
        per_device_train_batch_size=1, gradient_accumulation_steps=1,
    )
    expected = {"optimizer_updates": 1371, "warmup_steps": 41,
                "replica_padding_rows": 104, "sampler_seed": 900913,
                "physical_row_exposures": 252264}
    if config.get("derived_schedule") != expected or schedule["total_steps"] != 1371:
        raise SystemExit("The S2 derived exposure schedule changed")
    resource = require_object(config.get("resource_plan"), "resource plan")
    required = {"nodes": 184, "gpus_per_node": 8, "tensor_parallel_size": 8,
                "expert_parallel_size": 8, "data_parallel_replicas": 184,
                "per_replica_microbatch": 1, "gradient_accumulation_steps": 1,
                "precision": "bf16", "requested_partition": "batch",
                "requested_walltime": "10:00:00", "gpu_preflight_required": True,
                "throughput_calibration_updates": 100}
    if any(stable_sha256(resource.get(k)) != stable_sha256(v) for k, v in required.items()):
        raise SystemExit("The S2 allocation or GPU preflight changed")
    trainer = require_object(config.get("trainer"), "trainer")
    expected_trainer = {
        "path": "scripts/train_sft_s2_dp_tp_ep.py", "fine_tune_configuration": "lora_r32",
        "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "optimizer": "AdamW", "adam_beta1": 0.9, "adam_beta2": 0.999,
        "adam_epsilon": 1e-08, "weight_decay": 0, "max_grad_norm": 1,
        "uniform_assistant_token_loss_weight": 1, "segment_loss_telemetry": True,
        "activation_checkpointing": True, "checkpointing_use_reentrant": False,
        "loss_contract": "multiplex_sft_v3", "completion_loss_weight": 1,
    }
    if any(stable_sha256(trainer.get(k)) != stable_sha256(v) for k, v in expected_trainer.items()):
        raise SystemExit("The S2 trainer objective changed")
    initialization = require_object(config.get("initialization"), "initialization")
    if (initialization.get("parent") != "s1_checkpoint"
            or initialization.get("continue_existing_adapter") is not True
            or initialization.get("restore_parent_optimizer") is not False
            or initialization.get("restore_parent_scheduler") is not False
            or type(initialization.get("start_step")) is not int
            or initialization["start_step"] != 0):
        raise SystemExit("S2 must initialize S1 adapter weights with a fresh optimizer")
    selection = require_object(config.get("checkpoint_selection"), "checkpoint selection")
    if selection.get("test_panel_access_during_training") is not False:
        raise SystemExit("The S2 test panel cannot be used during training")
    if config.get("tracking") != {"mode": "online", "entity": "jail-ai", "project": "mentor-sft"}:
        raise SystemExit("The S2 tracking destination changed")


def validate_source_pins(root: Path, entries: object) -> str:
    """Verify the sealed training implementation and return its aggregate hash."""
    if not isinstance(entries, list) or not entries:
        raise SystemExit("The S2 implementation file pins are absent")
    names = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise SystemExit("An S2 implementation pin is invalid")
        names.append(entry["path"])
        verify_file(repo_path(root, entry["path"]), require_sha256(entry["sha256"], "source"), "training source")
    if names != sorted(set(names)) or not CORE_SOURCE_PATHS.issubset(names):
        raise SystemExit("The S2 implementation pins omit core files or repeat paths")
    return stable_sha256(entries)


def refresh_code_pins(root: Path, config_path: Path) -> None:
    """Seal the final code bytes after all parallel implementation edits finish."""
    config = read_json(config_path, "run config")
    validate_recipe(config)
    paths = set(CORE_SOURCE_PATHS)
    paths.update(p.relative_to(root).as_posix() for p in (root / "runtime").rglob("*.py"))
    config["implementation_files"] = [
        {"path": name, "sha256": sha256_file(repo_path(root, name))}
        for name in sorted(paths)
    ]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def validate_corpus(root: Path, config: dict) -> tuple[Path, dict]:
    """Verify the frozen train/validation bytes and their prior independent audit."""
    corpus = require_object(config.get("corpus"), "corpus")
    if (corpus.get("root") != CORPUS_ROOT or corpus.get("manifest_sha256") != CORPUS_SHA256
            or corpus.get("truncation_allowed") is not False
            or corpus.get("training_mix") != {"S2": 0.8, "S1": 0.15, "S0": 0.05}):
        raise SystemExit("The frozen S2 corpus identity or mixture changed")
    corpus_root = repo_path(root, CORPUS_ROOT)
    manifest_path = corpus_root / "manifest.json"
    verify_file(manifest_path, CORPUS_SHA256, "S2 corpus manifest")
    manifest = read_json(manifest_path, "S2 corpus manifest")
    if (manifest.get("schema_version") != "mentor-rl-s2-production-corpus-v1"
            or manifest.get("production_id") != "s2_production_v1_8k"
            or manifest.get("data_ready") is not True
            or manifest.get("whole_trace_source_and_supervision_checked") is not True
            or manifest.get("truncation_applied") is not False
            or manifest.get("max_length") != 8192
            or manifest.get("panel_id") != corpus.get("test_panel_id")):
        raise SystemExit("The S2 corpus contract changed")
    for split, (rows, expected_hash) in SPLITS.items():
        filename = split + ".jsonl"
        if (corpus.get(split + "_file") != filename or corpus.get(split + "_rows") != rows
                or corpus.get(split + "_sha256") != expected_hash
                or manifest.get("files", {}).get(filename, {}).get("sha256") != expected_hash
                or manifest["files"][filename].get("row_count") != rows):
            raise SystemExit(f"The S2 {split} declaration changed")
        verify_file(corpus_root / filename, expected_hash, f"S2 {split}")
    _, audit = checked_reference(root, config.get("corpus_verification"), "S2 independent verification")
    if (audit.get("corpus_manifest", {}).get("sha256") != CORPUS_SHA256
            or audit.get("data_verification_passed") is not True
            or audit.get("upstream_complete_token_receipts_matched") != 302592
            or audit.get("max_tokens") != {"train": 7368, "validation": 7216, "test": 8157}):
        raise SystemExit("The S2 independent verification is incomplete")
    return corpus_root, manifest


def validate_bank_acceptance(root: Path, config: dict) -> None:
    """Honor the user's as-is instruction while preserving the original manifest."""
    _, receipt = checked_reference(root, config.get("bank_acceptance"), "bank acceptance")
    if (receipt.get("accepted_as_is_for_requested_s2_run") is not True
            or receipt.get("human_wording_review_claimed") is not False
            or receipt.get("corpus_manifest", {}).get("sha256") != CORPUS_SHA256
            or receipt.get("original_corpus_manifest_modified") is not False
            or len(receipt.get("frozen_banks", [])) != 6):
        raise SystemExit("The S2 as-is bank acceptance changed")
    for pin in receipt["frozen_banks"]:
        verify_file(repo_path(root, pin["path"]), require_sha256(pin["sha256"], "bank"), "accepted bank")


def validate_parent(root: Path, config: dict, tokenizer: tuple) -> Path:
    """Bind the actual complete S1 adapter to the passing structured gate."""
    init = config["initialization"]
    checkpoint = repo_path(root, init["checkpoint_path"])
    _, verification = checked_reference(root, init.get("parent_gate"), "parent verification")
    if (verification.get("checkpoint_path") != init["checkpoint_path"]
            or verification.get("gate_passed") is not True):
        raise SystemExit("The selected S1 parent has no passing gate")
    _, gate = checked_reference(root, verification.get("gate_receipt"), "S1 gate")
    parent_config_path = checkpoint / "run_contract/run_config.json"
    parent_config = read_json(parent_config_path, "S1 parent run config")
    if parent_config.get("schema_version") != "mentor-rl-world-model-s1-120b-training-v10":
        raise SystemExit("The S2 parent must be completed S1 v10")
    if parent_config.get("model") != config.get("model"):
        raise SystemExit("The S2 base model differs from the completed S1 parent")
    parent_corpus = parent_config["corpus"]
    parent_adapter = read_json(checkpoint / "tp_adapter_manifest.json", "S1 parent adapter")
    parent_identity = parent_adapter["identity"]
    run_id, checkpoint_identity = _validate_checkpoint(
        checkpoint_path=checkpoint, checkpoint_root=checkpoint.parent,
        method_id=METHOD_ID, model_identity_sha256=config["model"]["model_identity_sha256"],
        training_config_sha256=sha256_file(parent_config_path),
        corpus_manifest_sha256=parent_corpus["manifest_sha256"],
        train_sha256=parent_corpus["train_sha256"],
        tokenizer_arm_manifest_sha256=tokenizer[4], tokenizer_manifest_sha256=tokenizer[2],
        tokenizer_artifacts_sha256=tokenizer[3], validation_sha256=parent_corpus["validation_sha256"],
        validation_answer_key_sha256=parent_identity["validation_answer_key_sha256"],
    )
    if (run_id != init.get("run_id") or checkpoint_identity != init.get("checkpoint_identity_sha256")
            or checkpoint_identity != gate.get("checkpoint_identity_sha256")
            or gate.get("train_run_id") != run_id or gate.get("gate", {}).get("passed") is not True
            or gate["gate"].get("profile") != "structured_95_v1"
            or parent_adapter.get("lora") != {"r": 32, "alpha": 64, "dropout": 0.0}):
        raise SystemExit("The S1 gate and selected adapter do not match")
    verify_file(checkpoint / "adapter_model.safetensors", init["adapter_sha256"], "S1 adapter")
    for path, pin in gate.get("artifacts", {}).items():
        verify_file(repo_path(root, path), pin["sha256"], "S1 gate report")
    return checkpoint


def validate_contract(root: Path, config_path: Path) -> dict[str, str]:
    """Validate every CPU-side input and export the trainer/launcher interface."""
    root = root.resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    config_path = repo_path(root, config_path.resolve().relative_to(root).as_posix())
    config = read_json(config_path, "S2 run config")
    validate_recipe(config)
    code_hash = validate_source_pins(root, config.get("implementation_files"))
    corpus_root, _ = validate_corpus(root, config)
    validate_bank_acceptance(root, config)
    method = require_object(config.get("method"), "method")
    if (method.get("method_id") != METHOD_ID or method.get("tokenizer_method") != "plain_base_tokenizer"
            or method.get("token_adapter_manifest") is not None
            or method.get("output_root") != "data/world_model_v4/checkpoints/s2_v1_184n/" + METHOD_ID):
        raise SystemExit("The S2 method identity changed")
    tokenizer = _validate_tokenizer(repo_root=root, method=method, training_method=method)
    model = config["model"]
    model_path = (root / model["path"]).resolve()
    if (model_path != (root.parent / "models/gpt-oss-120b-bf16").resolve()
            or model.get("model_id") != "gpt-oss-120b-bf16"
            or _model_metadata_identity(model_path) != model.get("model_identity_sha256")
            or model.get("base_model_artifact_schema") != "mentor-rl-base-model-artifact-v2"
            or model.get("weight_shard_count") != 36):
        raise SystemExit("The S2 base model identity changed")
    parent = validate_parent(root, config, tokenizer)
    settings = config["run_settings"]
    values = {
        "S0_RUN_CONFIG": str(config_path), "S0_RUN_CONFIG_SHA256": sha256_file(config_path),
        "S0_CONFIG_ID": TRAINING_ID, "S0_RUN_SCOPE": "production", "S0_DATASET_ID": "s2_production_v1_8k",
        "S0_METHOD_ID": METHOD_ID, "S0_TRAIN_MODE": "lora", "S0_TOKENIZER_METHOD": "plain_base_tokenizer",
        "S0_TOKENIZER_ARM_ROOT": str(repo_path(root, method["arm_root"])),
        "S0_TOKENIZER_ARM_MANIFEST_SHA256": tokenizer[4], "TOKENIZER_PATH": str(tokenizer[0]),
        "TOKENIZER_MANIFEST": str(tokenizer[1]), "S0_TOKENIZER_MANIFEST_SHA256": tokenizer[2],
        "S0_TOKENIZER_ARTIFACTS_SHA256": tokenizer[3], "MODEL_ID": model["model_id"],
        "MODEL_PATH": str(model_path), "MODEL_IDENTITY_SHA256": model["model_identity_sha256"],
        "BASE_MODEL_ARTIFACT_SCHEMA": model["base_model_artifact_schema"],
        "BASE_MODEL_ARTIFACT_SHA256": require_sha256(model.get("base_model_artifact_sha256"), "full base model"),
        "BASE_MODEL_WEIGHT_SHARD_COUNT": "36", "S0_CORPUS_ROOT": str(corpus_root),
        "S0_CORPUS_MANIFEST_SHA256": CORPUS_SHA256,
        "S2_CORPUS_MANIFEST_PATH": str(corpus_root / "manifest.json"),
        "S0_TRAIN_PATH": str(corpus_root / "train.jsonl"), "S0_TRAIN_SHA256": SPLITS["train"][1],
        "S0_TRAIN_ROWS": "252160", "S0_VALIDATION_PATH": str(corpus_root / "validation.jsonl"),
        "S0_VALIDATION_SHA256": SPLITS["validation"][1], "S0_VALIDATION_ROWS": "25216",
        "S0_VALIDATION_ANSWER_KEY_PATH": str(corpus_root / "validation.jsonl"),
        "S0_VALIDATION_ANSWER_KEY_SHA256": SPLITS["validation"][1],
        "S0_EVALUATOR_MANIFEST": str(corpus_root / "manifest.json"),
        "S0_EVALUATOR_MANIFEST_SHA256": CORPUS_SHA256,
        "TRAINER_PATH": str(root / config["trainer"]["path"]),
        "OUTPUT_ROOT": str(repo_path(root, method["output_root"])),
        "WORLD_SIZE_EXPECTED": "1472", "RANKS_PER_NODE": "8", "TRAIN_UPDATES_PER_EPOCH": "1371",
        "TRAIN_TOTAL_STEPS": "1371", "LOSS_CONTRACT": "multiplex_sft_v3",
        "LORA_R": "32", "LORA_ALPHA": "64", "LORA_DROPOUT": "0",
        "INIT_ADAPTER_DIR": str(parent), "INIT_ADAPTER_SHA256": config["initialization"]["adapter_sha256"],
        "INIT_ADAPTER_RUN_ID": config["initialization"]["run_id"],
        "S0_TRAINING_CODE_SHA256": code_hash, "SEGMENT_LOSS_TELEMETRY": "1",
        "DIAGNOSTIC_SNAPSHOT_STEPS": ",".join(map(str, settings["diagnostic_snapshot_steps"])),
        "GPU_PREFLIGHT": "1", "THROUGHPUT_CALIBRATION_UPDATES": "100",
        "S0_TRACKING_MODE": "online", "S0_WANDB_ENTITY": "jail-ai", "WANDB_PROJECT": "mentor-sft",
        "REQUESTED_PARTITION": "batch", "REQUESTED_WALLTIME": "10:00:00",
    }
    setting_env = {
        "NUM_NODES": "num_nodes", "GPUS_PER_NODE": "gpus_per_node", "TP_SIZE": "tp_size", "EP_SIZE": "ep_size",
        "DATA_PARALLEL_SIZE": "data_parallel_size", "TRAIN_NUM_EPOCHS": "num_train_epochs",
        "TRAIN_BATCH_SIZE": "per_device_train_batch_size", "GRAD_ACCUM_STEPS": "gradient_accumulation_steps",
        "GLOBAL_BATCH_SIZE": "global_batch_size", "LEARNING_RATE": "learning_rate", "LR_SCHEDULER_TYPE": "lr_scheduler_type",
        "WARMUP_RATIO": "warmup_ratio", "WEIGHT_DECAY": "weight_decay", "MAX_GRAD_NORM": "max_grad_norm",
        "MAX_LENGTH": "max_length", "LOGGING_STEPS": "logging_steps", "EVAL_STRATEGY": "eval_strategy",
        "EVAL_ON_START": "eval_on_start", "EVAL_STEPS": "eval_steps", "EVAL_BATCH_SIZE": "per_device_eval_batch_size",
        "SAVE_STRATEGY": "save_strategy", "SAVE_STEPS": "save_steps", "SAVE_TOTAL_LIMIT": "save_total_limit", "SEED": "seed",
        "PRESERVE_DATASET_ORDER": "preserve_dataset_order", "LOCAL_FILES_ONLY": "local_files_only", "BF16": "bf16",
        "AUTOGRAD_MULTITHREADING": "autograd_multithreading", "STRICT_TESTED_STACK": "strict_tested_stack",
    }
    for name, field in setting_env.items():
        value = settings[field]
        values[name] = str(int(value) if type(value) is bool else value)
    return values


def main() -> int:
    """Print shell assignments, JSON, refresh pins, or hash one staged model."""
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-base-model":
        if len(sys.argv) != 6:
            raise SystemExit("--validate-base-model needs MODEL SHA256 SHARD_COUNT RECEIPT")
        validate_base_model(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("run_config", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--refresh-code-pins", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config_path = args.run_config if args.run_config.is_absolute() else root / args.run_config
    if args.refresh_code_pins:
        refresh_code_pins(root, config_path)
        return 0
    try:
        values = validate_contract(root, config_path)
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise SystemExit(f"Invalid S2 training contract: {error}") from error
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
    else:
        for name, value in values.items():
            print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
