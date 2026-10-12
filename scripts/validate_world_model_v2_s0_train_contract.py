#!/usr/bin/env python3
"""Validate one S0 v4 training contract and export its values.

The launcher runs this check before it allocates any training work. It pins
the run settings of the v4 sweep, the corpus and its runtime contract, the
teacher-forced validation panel, the tokenizer, and the base model. It then
prints shell assignments for the launcher to source.

With --validate-base-model, it instead hashes one staged base-model copy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import (  # noqa: E402
    S0_LOSS_CONTRACTS,
    S0_V3_TRACE_LOSS_CONTRACT,
    S0_V4_DATASET_ID,
    S0_V4_PRIORITY_LOSS_CONTRACT,
    derive_optimizer_step_schedule,
)


S0_V4_TRAINING_SCHEMA = "mentor-rl-world-model-s0-120b-training-v4"
# All four arms of the v4 sweep share these settings. They differ only in the
# LoRA rank and the loss contract.
S0_V4_RUN_SETTINGS = {
    "num_nodes": 16,
    "gpus_per_node": 8,
    "tp_size": 8,
    "ep_size": 8,
    "data_parallel_size": 16,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "global_batch_size": 32,
    "learning_rate": 0.0001,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0,
    "max_grad_norm": 1,
    "max_length": 4096,
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "eval_on_start": True,
    "per_device_eval_batch_size": 1,
    "save_strategy": "steps",
    "save_steps": 512,
    "save_total_limit": 3,
    "seed": 900913,
    "preserve_dataset_order": False,
    "autograd_multithreading": True,
    "local_files_only": True,
    "strict_tested_stack": True,
    "bf16": True,
}
S0_V4_TRAIN_ROWS = 16384
S0_V4_VALIDATION_ROWS = 622
S0_V4_UPDATES_PER_EPOCH = 512
S0_V4_TOTAL_STEPS = 1536
V4_MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-manifest-v2"
V4_CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s0-corpus-v2"
V4_EVALUATION_CONTRACT = "held_out_multiplex_sft_v4"
V4_EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-v4"
V4_PRIORITY_LOSS_WEIGHTS = {
    "nonassistant": 0,
    "assistant_prose": 1,
    "tool_call": 4,
    "final_json": 4,
    "critical_final_fields": 8,
    "critical_field_names": [
        "status",
        "result",
        "continuation_decision",
        "blocked_reason",
    ],
    "normalization": "positive_token_mean_per_row",
}
V4_OVERLAY_METADATA_ADDITIONS = [
    "fact_kind",
    "framing",
    "prompt_template_id",
    "rendering_index",
    "input_variant",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha256(value: object) -> str:
    """Return the SHA-256 value for canonical JSON."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_base_model(
    model_path: Path,
    expected_hash: str,
    expected_shards: int,
    receipt_path: Path,
) -> None:
    """Validate all base-model files and write one receipt."""

    root = model_path.resolve()
    required = ("config.json", "model.safetensors.index.json")
    if not root.is_dir() or any(
        not (root / name).is_file() for name in required
    ):
        raise SystemExit("The base model metadata is absent")
    index = json.loads((root / required[1]).read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("The base model weight index is invalid")
    shards = set()
    for value in weight_map.values():
        if not isinstance(value, str) or not value:
            raise SystemExit("A base model shard name is invalid")
        relative = Path(value)
        if relative.is_absolute() or relative.as_posix() != value:
            raise SystemExit("A base model shard path is invalid")
        shards.add(value)
    if len(shards) != expected_shards:
        raise SystemExit("The base model shard count changed")
    names = list(required)
    if (root / "generation_config.json").is_file():
        names.append("generation_config.json")
    names.extend(sorted(shards))
    if len(names) != len(set(names)):
        raise SystemExit("The base model artifact paths are not unique")
    files = {}
    for name in sorted(names):
        path = root / name
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, ValueError) as error:
            raise SystemExit(
                "A base model artifact escapes its root"
            ) from error
        if (
            relative.as_posix() != name
            or not path.is_file()
            or path.is_symlink()
        ):
            raise SystemExit(f"A base model artifact is absent: {name}")
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": "mentor-rl-base-model-artifact-v2",
        "weight_shard_count": len(shards),
        "files": files,
    }
    payload["base_model_artifact_sha256"] = stable_sha256(payload)
    if payload["base_model_artifact_sha256"] != expected_hash:
        raise SystemExit("The full base model artifact identity changed")
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_object(value: object, name: str) -> dict[str, Any]:
    """Return one required object."""

    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be an object")
    return dict(value)


def require_string(value: object, name: str) -> str:
    """Return one required string."""

    if not isinstance(value, str) or not value:
        raise SystemExit(f"{name} must be a nonempty string")
    return value


def require_sha256(value: object, name: str) -> str:
    """Return one required SHA-256 value."""

    text = require_string(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise SystemExit(f"{name} must be one SHA-256 value")
    return text


def require_int(value: object, name: str) -> int:
    """Return one positive integer."""

    if type(value) is not int or value < 1:
        raise SystemExit(f"{name} must be a positive integer")
    return value


def require_number(value: object, name: str) -> float:
    """Return one numeric value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{name} must be a number")
    return float(value)


def require_bool(value: object, name: str) -> bool:
    """Return one Boolean value."""

    if type(value) is not bool:
        raise SystemExit(f"{name} must be true or false")
    return value


def absolute_path(root: Path, value: object, name: str) -> str:
    """Resolve one path from the repository."""

    text = require_string(value, name)
    return str((root / text).resolve())


def read_json(path: Path, name: str) -> dict[str, Any]:
    """Read one required JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{name} is invalid: {error}") from error
    return require_object(value, name)


def verify_file(path: Path, expected: str, name: str) -> None:
    """Verify one required file."""

    if not path.is_file() or sha256_file(path) != expected:
        raise SystemExit(f"The {name} identity changed")


def validate_train_contract(
    root: Path,
    config_path: Path,
    method_id: str,
) -> dict[str, str]:
    """Validate one S0 v4 training contract."""

    root = root.resolve()
    config_path = config_path.resolve()
    try:
        config_path.relative_to(root)
    except ValueError as error:
        raise SystemExit("RUN_CONFIG must stay in the repository") from error
    config = read_json(config_path, "run config")
    schema = config.get("schema_version")
    if schema != S0_V4_TRAINING_SCHEMA:
        raise SystemExit(
            f"The train config schema is not supported: {schema!r}"
        )

    methods = [
        value
        for value in config.get("methods", [])
        if isinstance(value, dict) and value.get("method_id") == method_id
    ]
    if len(methods) != 1:
        raise SystemExit("S0_METHOD_ID must select one method")
    method = methods[0]
    model = require_object(config.get("model"), "model")
    corpus = require_object(config.get("corpus"), "corpus")
    trainer = require_object(config.get("trainer"), "trainer")
    settings = require_object(
        config.get("run_settings"), "run_settings"
    )
    overrides = require_object(
        method.get("run_settings", {}), "method run_settings"
    )
    unknown_overrides = sorted(set(overrides) - set(settings))
    if unknown_overrides:
        raise SystemExit(
            f"The method run_settings has unknown keys: {unknown_overrides}"
        )
    settings.update(overrides)
    if settings != S0_V4_RUN_SETTINGS:
        raise SystemExit("The S0 v4 training schedule changed")

    run_scope = require_string(config.get("run_scope"), "run_scope")
    config_id = require_string(config.get("training_id"), "training_id")

    tokenizer_method = require_string(
        method.get("tokenizer_method"), "tokenizer_method"
    )
    if tokenizer_method != "plain_base_tokenizer":
        raise SystemExit(
            "S0 trajectory training requires the base tokenizer"
        )
    if method.get("token_adapter_manifest") is not None:
        raise SystemExit(
            "The base tokenizer must not have a token adapter"
        )
    lora_rank = require_int(trainer.get("lora_rank"), "lora_rank")
    if trainer.get("loader_identity") != "peft_adapter_loader":
        raise SystemExit("The LoRA trainer contract changed")
    if (
        lora_rank not in {32, 128}
        or trainer.get("fine_tune_configuration") != f"lora_r{lora_rank}"
        or require_int(trainer.get("lora_alpha"), "lora_alpha")
        != 2 * lora_rank
        or require_number(trainer.get("lora_dropout"), "lora_dropout")
        != 0.0
    ):
        raise SystemExit("The v4 LoRA rank/alpha contract changed")
    if method_id != f"oss120b-plain-base-tokenizer-lora-r{lora_rank}":
        raise SystemExit(
            "The method ID differs from its selected configuration"
        )
    trainer_path = Path(
        absolute_path(root, trainer.get("path"), "trainer path")
    )
    if trainer_path != (root / "scripts/train_sft_dp_tp_ep.py").resolve():
        raise SystemExit("The selected trainer path changed")

    num_nodes = require_int(settings.get("num_nodes"), "num_nodes")
    gpus_per_node = require_int(
        settings.get("gpus_per_node"), "gpus_per_node"
    )
    tp_size = require_int(settings.get("tp_size"), "tp_size")
    ep_size = require_int(settings.get("ep_size"), "ep_size")
    data_parallel_size = require_int(
        settings.get("data_parallel_size"), "data_parallel_size"
    )
    epochs = require_int(
        settings.get("num_train_epochs"), "num_train_epochs"
    )
    batch_size = require_int(
        settings.get("per_device_train_batch_size"),
        "per_device_train_batch_size",
    )
    grad_accum = require_int(
        settings.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps",
    )
    global_batch = require_int(
        settings.get("global_batch_size"), "global_batch_size"
    )
    train_rows = require_int(
        corpus.get("train_rows"), "corpus train_rows"
    )
    validation_rows = require_int(
        corpus.get("validation_rows"), "corpus validation_rows"
    )
    schedule = derive_optimizer_step_schedule(
        train_rows,
        num_train_epochs=epochs,
        replica_count=data_parallel_size,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
    )
    if (
        train_rows != S0_V4_TRAIN_ROWS
        or validation_rows != S0_V4_VALIDATION_ROWS
        or schedule["global_batch_size"] != global_batch
        or schedule["updates_per_epoch"] != S0_V4_UPDATES_PER_EPOCH
        or schedule["total_steps"] != S0_V4_TOTAL_STEPS
    ):
        raise SystemExit("The S0 v4 optimizer-step schedule changed")
    learning_rate = require_number(
        settings.get("learning_rate"), "learning_rate"
    )
    warmup_ratio = require_number(
        settings.get("warmup_ratio"), "warmup_ratio"
    )
    weight_decay = require_number(
        settings.get("weight_decay"), "weight_decay"
    )
    max_grad_norm = require_number(
        settings.get("max_grad_norm"), "max_grad_norm"
    )

    # The uniform arms use multiplex_sft_v3; the priority arms use the fixed
    # token weights of multiplex_sft_v4_priority.
    loss_contract = require_string(
        trainer.get("loss_contract"), "loss_contract"
    )
    if loss_contract not in S0_LOSS_CONTRACTS:
        raise SystemExit("The S0 loss contract changed")
    priority_weights = trainer.get("token_loss_weights")
    if loss_contract == S0_V4_PRIORITY_LOSS_CONTRACT:
        if priority_weights != V4_PRIORITY_LOSS_WEIGHTS:
            raise SystemExit("The S0 v4 priority loss weights changed")
    elif priority_weights is not None:
        raise SystemExit("The uniform S0 v4 arm has priority weights")
    loss_arm = (
        "priority"
        if loss_contract == S0_V4_PRIORITY_LOSS_CONTRACT
        else "uniform"
    )
    if (
        config_id
        != f"world_model_s0_120b_training_v4_r{lora_rank}_{loss_arm}"
        or run_scope != "production"
        or require_bool(
            config.get("promotion_eligible"), "promotion_eligible"
        )
        is not True
        or require_bool(
            config.get("production_matrix_cell"),
            "production_matrix_cell",
        )
        is not True
    ):
        raise SystemExit("The S0 v4 run identity changed")
    if require_number(
        trainer.get("completion_loss_weight"),
        "completion_loss_weight",
    ) != 1.0:
        raise SystemExit("The S0 trajectory loss weight changed")

    eval_strategy = require_string(
        settings.get("eval_strategy"), "eval_strategy"
    )
    eval_on_start = require_bool(
        settings.get("eval_on_start"), "eval_on_start"
    )
    eval_batch_size = require_int(
        settings.get("per_device_eval_batch_size"),
        "per_device_eval_batch_size",
    )
    save_strategy = require_string(
        settings.get("save_strategy"), "save_strategy"
    )
    max_length = require_int(settings.get("max_length"), "max_length")
    save_steps = require_int(settings.get("save_steps"), "save_steps")
    save_total_limit = require_int(
        settings.get("save_total_limit"), "save_total_limit"
    )
    validation_gate = require_object(
        config.get("validation_gate"), "validation_gate"
    )
    if validation_gate != {
        "checkpoint_set": "each_saved_checkpoint_and_final",
        "selection_metric": "validation/hard_failure_count_then_accuracy",
        "required_method_count": 1,
        "full_registry_access": False,
    }:
        raise SystemExit("The S0 v4 validation gate changed")

    # The corpus must be the published v4 build of the shared contract and
    # the v4 overlay that are in this repository.
    corpus_root = Path(
        absolute_path(root, corpus.get("root"), "corpus root")
    )
    corpus_manifest_sha256 = require_sha256(
        corpus.get("manifest_sha256"), "corpus manifest_sha256"
    )
    corpus_manifest_path = corpus_root / "manifest.json"
    verify_file(
        corpus_manifest_path,
        corpus_manifest_sha256,
        "corpus manifest",
    )
    corpus_manifest = read_json(
        corpus_manifest_path, "corpus manifest"
    )
    if (
        corpus_manifest.get("dataset_id") != S0_V4_DATASET_ID
        or corpus_manifest.get("schema_version")
        != V4_MANIFEST_SCHEMA_VERSION
        or corpus_manifest.get("corpus_schema_version")
        != V4_CORPUS_SCHEMA_VERSION
        or corpus_manifest.get("contract", {}).get("row_schema_version")
        != "multiplex_sft_v3"
        or corpus_manifest.get("row_counts", {}).get("train")
        != train_rows
        or corpus_manifest.get("row_counts", {}).get("validation")
        != validation_rows
    ):
        raise SystemExit("The v4 corpus manifest contract changed")
    runtime_contract = require_object(
        corpus_manifest.get("contract"), "corpus runtime contract"
    )
    runtime_contract_declared = require_string(
        runtime_contract.get("path"),
        "corpus runtime contract path",
    )
    if Path(runtime_contract_declared).as_posix() != (
        "runtime/world_model/contract.yaml"
    ):
        raise SystemExit("The corpus runtime contract path changed")
    verify_file(
        Path(
            absolute_path(
                root,
                runtime_contract_declared,
                "corpus runtime contract path",
            )
        ),
        require_sha256(
            runtime_contract.get("sha256"),
            "corpus runtime contract SHA-256",
        ),
        "corpus runtime contract",
    )
    overlay = require_object(
        runtime_contract.get("s0_overlay"),
        "S0 v4 runtime overlay",
    )
    overlay_declared = require_string(
        overlay.get("path"), "S0 v4 runtime overlay path"
    )
    if (
        Path(overlay_declared).as_posix() != "runtime/world_model/s0_v4.yaml"
        or overlay.get("schema_version")
        != "mentor-rl-world-model-s0-overlay-v1"
        or overlay.get("metadata_optional_additions")
        != V4_OVERLAY_METADATA_ADDITIONS
    ):
        raise SystemExit("The S0 v4 runtime overlay contract changed")
    verify_file(
        Path(
            absolute_path(
                root,
                overlay_declared,
                "S0 v4 runtime overlay path",
            )
        ),
        require_sha256(
            overlay.get("sha256"),
            "S0 v4 runtime overlay SHA-256",
        ),
        "S0 v4 runtime overlay",
    )

    train_path = corpus_root / require_string(
        corpus.get("train_file"), "train_file"
    )
    train_sha256 = require_sha256(
        corpus.get("train_sha256"), "train_sha256"
    )
    verify_file(train_path, train_sha256, "train file")
    validation_path = corpus_root / require_string(
        corpus.get("validation_file"), "validation_file"
    )
    validation_sha256 = require_sha256(
        corpus.get("validation_sha256"), "validation_sha256"
    )
    verify_file(
        validation_path, validation_sha256, "validation file"
    )
    if (
        corpus_manifest.get("files", {}).get(train_path.name, {}).get("sha256")
        != train_sha256
        or corpus_manifest.get("files", {})
        .get(validation_path.name, {})
        .get("sha256")
        != validation_sha256
    ):
        raise SystemExit("The v4 corpus file manifest changed")

    # The trainer scores teacher-forced validation loss on this panel.
    evaluator_path = Path(
        absolute_path(
            root,
            corpus.get("evaluator_manifest"),
            "evaluator manifest",
        )
    )
    evaluator_sha256 = require_sha256(
        corpus.get("evaluator_manifest_sha256"),
        "evaluator manifest SHA-256",
    )
    verify_file(
        evaluator_path, evaluator_sha256, "evaluator manifest"
    )
    evaluator = read_json(evaluator_path, "evaluator manifest")
    if (
        evaluator.get("schema_version") != V4_EVALUATOR_SCHEMA_VERSION
        or evaluator.get("dataset_id") != S0_V4_DATASET_ID
        or evaluator.get("evaluation_contract") != V4_EVALUATION_CONTRACT
        or evaluator.get("corpus_manifest_sha256") != corpus_manifest_sha256
    ):
        raise SystemExit("The evaluator manifest contract changed")
    if (
        evaluator.get("row_schema_version") != "multiplex_sft_v3"
        or evaluator.get("compatible_loss_contracts")
        != [S0_V3_TRACE_LOSS_CONTRACT, S0_V4_PRIORITY_LOSS_CONTRACT]
        or "loss_contract" in evaluator
    ):
        raise SystemExit("The v4 evaluator loss compatibility changed")
    validation_panel = require_object(
        evaluator.get("teacher_forced_validation"),
        "evaluator validation panel",
    )
    if validation_panel.get("row_count") != validation_rows:
        raise SystemExit("The v4 evaluator validation count changed")
    validation_answer_key_path = Path(
        absolute_path(
            root,
            validation_panel.get("answer_key_path"),
            "validation answer_key_path",
        )
    )
    validation_answer_key_sha256 = require_sha256(
        validation_panel.get("answer_key_sha256"),
        "validation answer_key_sha256",
    )
    verify_file(
        validation_answer_key_path,
        validation_answer_key_sha256,
        "validation answer key",
    )

    model_id = require_string(model.get("model_id"), "model_id")
    if model_id != "gpt-oss-120b-bf16":
        raise SystemExit("The selected model differs from the run schema")
    model_path = absolute_path(root, model.get("path"), "model path")
    base_schema = require_string(
        model.get("base_model_artifact_schema"),
        "base_model_artifact_schema",
    )
    if base_schema != "mentor-rl-base-model-artifact-v2":
        raise SystemExit("The base model artifact schema changed")

    values = {
        "S0_RUN_CONFIG": str(config_path),
        "S0_RUN_CONFIG_SHA256": sha256_file(config_path),
        "S0_CONFIG_ID": config_id,
        "S0_RUN_SCOPE": run_scope,
        "S0_DATASET_ID": S0_V4_DATASET_ID,
        "S0_METHOD_ID": method_id,
        "S0_TRAIN_MODE": "lora",
        "S0_TOKENIZER_METHOD": tokenizer_method,
        "S0_TOKENIZER_ARM_ROOT": absolute_path(
            root, method.get("arm_root"), "tokenizer arm_root"
        ),
        "S0_TOKENIZER_ARM_MANIFEST_SHA256": require_sha256(
            method.get("arm_manifest_sha256"),
            "arm_manifest_sha256",
        ),
        "TOKENIZER_PATH": absolute_path(
            root, method.get("tokenizer_path"), "tokenizer_path"
        ),
        "S0_TOKENIZER_MANIFEST_SHA256": require_sha256(
            method.get("tokenizer_manifest_sha256"),
            "tokenizer_manifest_sha256",
        ),
        "MODEL_ID": model_id,
        "MODEL_PATH": model_path,
        "MODEL_IDENTITY_SHA256": require_sha256(
            model.get("model_identity_sha256"),
            "model_identity_sha256",
        ),
        "BASE_MODEL_ARTIFACT_SCHEMA": base_schema,
        "BASE_MODEL_ARTIFACT_SHA256": require_sha256(
            model.get("base_model_artifact_sha256"),
            "base_model_artifact_sha256",
        ),
        "BASE_MODEL_WEIGHT_SHARD_COUNT": str(
            require_int(
                model.get("weight_shard_count"),
                "weight_shard_count",
            )
        ),
        "S0_CORPUS_ROOT": str(corpus_root),
        "S0_CORPUS_MANIFEST_SHA256": corpus_manifest_sha256,
        "S0_TRAIN_PATH": str(train_path),
        "S0_TRAIN_SHA256": train_sha256,
        "S0_TRAIN_ROWS": str(train_rows),
        "S0_VALIDATION_PATH": str(validation_path),
        "S0_VALIDATION_SHA256": validation_sha256,
        "S0_VALIDATION_ROWS": str(validation_rows),
        "S0_VALIDATION_ANSWER_KEY_PATH": str(
            validation_answer_key_path
        ),
        "S0_VALIDATION_ANSWER_KEY_SHA256": (
            validation_answer_key_sha256
        ),
        "S0_EVALUATOR_MANIFEST": str(evaluator_path),
        "S0_EVALUATOR_MANIFEST_SHA256": evaluator_sha256,
        "TRAINER_PATH": str(trainer_path),
        "OUTPUT_ROOT": absolute_path(
            root, method.get("output_root"), "output_root"
        ),
        "NUM_NODES": str(num_nodes),
        "GPUS_PER_NODE": str(gpus_per_node),
        "RANKS_PER_NODE": str(gpus_per_node),
        "WORLD_SIZE_EXPECTED": str(num_nodes * gpus_per_node),
        "TP_SIZE": str(tp_size),
        "EP_SIZE": str(ep_size),
        "DATA_PARALLEL_SIZE": str(data_parallel_size),
        "TRAIN_NUM_EPOCHS": str(epochs),
        "TRAIN_UPDATES_PER_EPOCH": str(
            schedule["updates_per_epoch"]
        ),
        "TRAIN_TOTAL_STEPS": str(schedule["total_steps"]),
        "TRAIN_BATCH_SIZE": str(batch_size),
        "GRAD_ACCUM_STEPS": str(grad_accum),
        "GLOBAL_BATCH_SIZE": str(global_batch),
        "LEARNING_RATE": str(learning_rate),
        "LR_SCHEDULER_TYPE": require_string(
            settings.get("lr_scheduler_type"),
            "lr_scheduler_type",
        ),
        "WARMUP_RATIO": str(warmup_ratio),
        "WEIGHT_DECAY": str(weight_decay),
        "MAX_GRAD_NORM": str(max_grad_norm),
        "MAX_LENGTH": str(max_length),
        "LOGGING_STEPS": str(
            require_int(settings.get("logging_steps"), "logging_steps")
        ),
        "EVAL_STRATEGY": eval_strategy,
        "EVAL_ON_START": str(int(eval_on_start)),
        "EVAL_BATCH_SIZE": str(eval_batch_size),
        "SAVE_STRATEGY": save_strategy,
        "SAVE_STEPS": str(save_steps),
        "SAVE_TOTAL_LIMIT": str(save_total_limit),
        "SEED": str(require_int(settings.get("seed"), "seed")),
        "PRESERVE_DATASET_ORDER": str(
            int(
                require_bool(
                    settings.get("preserve_dataset_order"),
                    "preserve_dataset_order",
                )
            )
        ),
        "LOCAL_FILES_ONLY": str(
            int(
                require_bool(
                    settings.get("local_files_only"),
                    "local_files_only",
                )
            )
        ),
        "BF16": str(
            int(require_bool(settings.get("bf16"), "bf16"))
        ),
        "LOSS_CONTRACT": loss_contract,
        "LORA_R": str(lora_rank),
        "LORA_ALPHA": str(
            require_int(trainer.get("lora_alpha"), "lora_alpha")
        ),
        "LORA_DROPOUT": str(
            require_number(
                trainer.get("lora_dropout"), "lora_dropout"
            )
        ),
        "AUTOGRAD_MULTITHREADING": str(
            int(
                require_bool(
                    settings.get("autograd_multithreading"),
                    "autograd_multithreading",
                )
            )
        ),
        "STRICT_TESTED_STACK": str(
            int(
                require_bool(
                    settings.get("strict_tested_stack"),
                    "strict_tested_stack",
                )
            )
        ),
    }
    return values


def main() -> int:
    """Validate one command-line contract."""

    if len(sys.argv) >= 2 and sys.argv[1] == "--validate-base-model":
        if len(sys.argv) != 6:
            raise SystemExit(
                "--validate-base-model needs a model, hash, "
                "shard count, and receipt"
            )
        validate_base_model(
            Path(sys.argv[2]),
            sys.argv[3],
            int(sys.argv[4]),
            Path(sys.argv[5]),
        )
        return 0
    if len(sys.argv) != 4:
        raise SystemExit(
            "Usage: validate_world_model_v2_s0_train_contract.py "
            "REPO_ROOT RUN_CONFIG METHOD_ID"
        )
    root = Path(sys.argv[1]).resolve()
    config_path = Path(sys.argv[2])
    if not config_path.is_absolute():
        config_path = root / config_path
    values = validate_train_contract(
        root, config_path, sys.argv[3]
    )
    for name, value in values.items():
        print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
