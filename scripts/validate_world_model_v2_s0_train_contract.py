#!/usr/bin/env python3
"""Validate one S0 train run contract and print its shell values."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def validate_train_inputs() -> None:
    """Validate the selected corpus and tokenizer files."""

    from runtime.world_model_training import (
        validate_s0_corpus_identity,
        validate_s0_tokenizer_arm_identity,
    )

    validate_s0_corpus_identity(
        Path(os.environ["S0_CORPUS_ROOT"]),
        evaluator_manifest_path=Path(
            os.environ["S0_EVALUATOR_MANIFEST"]
        ),
        validation_answer_key_path=Path(
            os.environ["S0_VALIDATION_ANSWER_KEY_PATH"]
        ),
        expected_manifest_sha256=os.environ[
            "S0_CORPUS_MANIFEST_SHA256"
        ],
        expected_train_sha256=os.environ["S0_TRAIN_SHA256"],
        expected_train_rows=int(os.environ["S0_TRAIN_ROWS"]),
        expected_validation_sha256=os.environ[
            "S0_VALIDATION_SHA256"
        ],
        expected_validation_rows=int(
            os.environ["S0_VALIDATION_ROWS"]
        ),
        expected_validation_answer_key_sha256=os.environ[
            "S0_VALIDATION_ANSWER_KEY_SHA256"
        ],
    )
    validate_s0_tokenizer_arm_identity(
        Path(os.environ["S0_TOKENIZER_ARM_ROOT"]),
        method=os.environ["S0_TOKENIZER_METHOD"],
        expected_corpus_manifest_sha256=os.environ[
            "S0_CORPUS_MANIFEST_SHA256"
        ],
        expected_train_sha256=os.environ["S0_TRAIN_SHA256"],
        expected_arm_manifest_sha256=os.environ[
            "S0_TOKENIZER_ARM_MANIFEST_SHA256"
        ],
        maximum_sequence_tokens=int(os.environ["MAX_LENGTH"]),
        tokenizer_path=Path(os.environ["TOKENIZER_PATH"]),
    )


if len(sys.argv) >= 2 and sys.argv[1] == "--validate-inputs":
    if len(sys.argv) != 2:
        raise SystemExit("--validate-inputs takes no values")
    validate_train_inputs()
    raise SystemExit(0)

if len(sys.argv) >= 2 and sys.argv[1] == "--validate-base-model":
    if len(sys.argv) != 6:
        raise SystemExit(
            "--validate-base-model needs a model, hash, shard count, and receipt"
        )
    validate_base_model(
        Path(sys.argv[2]),
        sys.argv[3],
        int(sys.argv[4]),
        Path(sys.argv[5]),
    )
    raise SystemExit(0)


if len(sys.argv) != 4:
    raise SystemExit(
        "Usage: validate_world_model_v2_s0_train_contract.py "
        "REPO_ROOT RUN_CONFIG METHOD_ID"
    )

root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2])
if not config_path.is_absolute():
    config_path = root / config_path
config_path = config_path.resolve()
try:
    config_path.relative_to(root)
except ValueError as error:
    raise SystemExit("RUN_CONFIG must stay in the repository") from error
method_id = sys.argv[3]

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"The run config is invalid: {error}") from error

schema = config.get("schema_version")
schemas = {
    "mentor-rl-world-model-s0-20b-qualification-v4",
    "mentor-rl-world-model-s0-120b-matrix-v4",
}
if schema not in schemas:
    raise SystemExit(f"The train config schema is not supported: {schema!r}")

methods = [
    value
    for value in config.get("methods", [])
    if isinstance(value, dict) and value.get("method_id") == method_id
]
if len(methods) != 1:
    raise SystemExit("S0_METHOD_ID must select one method")
method = methods[0]


def require_object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be an object")
    return dict(value)


def require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{name} must be a nonempty string")
    return value


def require_sha256(value: object, name: str) -> str:
    text = require_string(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise SystemExit(f"{name} must be one SHA-256 value")
    return text


def require_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise SystemExit(f"{name} must be a positive integer")
    return value


def require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{name} must be a number")
    return float(value)


def require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise SystemExit(f"{name} must be true or false")
    return value


def absolute_path(value: object, name: str) -> str:
    text = require_string(value, name)
    path = Path(text)
    if path.is_absolute():
        raise SystemExit(f"{name} must be relative to the repository")
    return str((root / path).resolve())


model = require_object(config.get("model"), "model")
corpus = require_object(config.get("corpus"), "corpus")
settings = require_object(config.get("run_settings"), "run_settings")
run_scope = require_string(config.get("run_scope"), "run_scope")
overrides = method.get("run_settings", {})
overrides = require_object(overrides, "method run_settings")
unknown_overrides = sorted(set(overrides) - set(settings))
if unknown_overrides:
    raise SystemExit(
        f"The method run_settings has unknown keys: {unknown_overrides}"
    )
settings.update(overrides)

if schema == "mentor-rl-world-model-s0-20b-qualification-v4":
    config_id = require_string(config.get("qualification_id"), "qualification_id")
    tokenizer = method
    trainer = require_object(config.get("trainer"), "trainer")
    fine_tune = trainer
    train_mode = "lora"
    topology = {
        "num_nodes": settings.get("num_nodes"),
        "gpus_per_node": settings.get("gpus_per_node"),
        "ranks_per_node": settings.get("gpus_per_node"),
        "tp_size": settings.get("tp_size"),
        "ep_size": settings.get("ep_size"),
        "data_parallel_size": settings.get("data_parallel_size"),
    }
    checkpoint = settings
else:
    config_id = require_string(config.get("matrix_id"), "matrix_id")
    tokenizer_method = require_string(
        method.get("tokenizer_method"), "method tokenizer_method"
    )
    tokenizers = require_object(config.get("tokenizers"), "tokenizers")
    tokenizer = require_object(
        tokenizers.get(tokenizer_method), "selected tokenizer"
    )
    configuration_name = require_string(
        method.get("fine_tune_configuration"),
        "method fine_tune_configuration",
    )
    configurations = require_object(
        config.get("fine_tune_configurations"),
        "fine_tune_configurations",
    )
    fine_tune = require_object(
        configurations.get(configuration_name),
        "selected fine_tune_configuration",
    )
    train_mode = require_string(
        fine_tune.get("training_mode"), "training_mode"
    )
    trainers = require_object(config.get("trainers"), "trainers")
    trainer = require_object(trainers.get(train_mode), "selected trainer")
    topologies = require_object(config.get("topologies"), "topologies")
    topology = require_object(topologies.get(train_mode), "selected topology")
    checkpoints = require_object(
        config.get("checkpoint_settings"), "checkpoint_settings"
    )
    checkpoint = require_object(
        checkpoints.get(train_mode), "selected checkpoint settings"
    )

if train_mode not in {"lora", "full_finetune"}:
    raise SystemExit(f"The train mode is not supported: {train_mode!r}")
if train_mode == "full_finetune" and not method_id.startswith("oss120b-"):
    raise SystemExit("Only an OSS-120B method can use full fine-tune")

tokenizer_method = require_string(
    method.get("tokenizer_method"),
    "tokenizer_method",
)
tokenizer_ids = {
    "plain_base_tokenizer": "plain-base-tokenizer",
    "ordinary_domain_bpe": "ordinary-domain-bpe",
    "atomic_plus_domain_bpe": "atomic-plus-domain-bpe",
    "fully_atomic_identifiers": "fully-atomic-identifiers",
}

if tokenizer_method not in tokenizer_ids:
    raise SystemExit(
        "The tokenizer method is not supported: "
        f"{tokenizer_method!r}"
    )
tokenizer_id = tokenizer_ids[tokenizer_method]

expected_suffix = (
    f"{tokenizer_id}-lora-r{fine_tune.get('lora_rank')}"
    if train_mode == "lora"
    else f"{tokenizer_id}-full-finetune"
)
expected_prefix = (
    "oss20b"
    if schema == "mentor-rl-world-model-s0-20b-qualification-v4"
    else "oss120b"
)
if method_id != f"{expected_prefix}-{expected_suffix}":
    raise SystemExit("The method ID differs from its selected configuration")

num_nodes = require_int(topology.get("num_nodes"), "num_nodes")
gpus_per_node = require_int(
    topology.get("gpus_per_node"), "gpus_per_node"
)
ranks_per_node = require_int(
    topology.get("ranks_per_node"), "ranks_per_node"
)
data_parallel_size = require_int(
    topology.get("data_parallel_size"), "data_parallel_size"
)
if gpus_per_node != ranks_per_node:
    raise SystemExit("Each train rank must own one GPU")
if train_mode == "lora":
    tp_size = require_int(topology.get("tp_size"), "tp_size")
    ep_size = require_int(topology.get("ep_size"), "ep_size")
    if tp_size != ranks_per_node or ep_size != ranks_per_node:
        raise SystemExit("One node must hold one complete TP and EP replica")
    if data_parallel_size != num_nodes:
        raise SystemExit("The LoRA data parallel size must equal the node count")
else:
    tp_size = 1
    ep_size = 1
    if data_parallel_size != num_nodes * ranks_per_node:
        raise SystemExit("The full data parallel size differs from the rank count")

epochs = require_int(settings.get("num_train_epochs"), "num_train_epochs")
max_steps = require_int(settings.get("max_steps"), "max_steps")
batch_size = require_int(
    settings.get("per_device_train_batch_size"),
    "per_device_train_batch_size",
)
grad_accum = require_int(
    settings.get("gradient_accumulation_steps"),
    "gradient_accumulation_steps",
)
global_batch = require_int(settings.get("global_batch_size"), "global_batch_size")
if global_batch != data_parallel_size * batch_size * grad_accum:
    raise SystemExit("The global batch size differs from the selected topology")
train_rows = require_int(corpus.get("train_rows"), "corpus train_rows")
validation_rows = require_int(
    corpus.get("validation_rows"),
    "corpus validation_rows",
)
full_exposure_steps = epochs * math.ceil(train_rows / global_batch)
if run_scope == "debug_qualification":
    if schema != "mentor-rl-world-model-s0-20b-qualification-v4":
        raise SystemExit("Only the 20B config can use debug_qualification")
    if max_steps >= full_exposure_steps:
        raise SystemExit(
            "A debug qualification must stop before full exposure"
        )
elif max_steps != full_exposure_steps:
    raise SystemExit(
        f"max_steps must equal {full_exposure_steps} for this exact exposure"
    )

learning_rate = require_number(settings.get("learning_rate"), "learning_rate")
warmup_ratio = require_number(settings.get("warmup_ratio"), "warmup_ratio")
weight_decay = require_number(settings.get("weight_decay"), "weight_decay")
max_grad_norm = require_number(settings.get("max_grad_norm"), "max_grad_norm")
if learning_rate <= 0 or not 0 <= warmup_ratio < 1:
    raise SystemExit("The learning rate or warmup ratio is invalid")
if weight_decay < 0 or max_grad_norm <= 0:
    raise SystemExit("The weight decay or gradient limit is invalid")

eval_strategy = require_string(
    settings.get("eval_strategy"),
    "eval_strategy",
)
if eval_strategy != "epoch":
    raise SystemExit("The S0 validation strategy must be epoch")
eval_on_start = require_bool(
    settings.get("eval_on_start"),
    "eval_on_start",
)
if not eval_on_start:
    raise SystemExit("S0 must validate before the first train step")
eval_batch_size = require_int(
    settings.get("per_device_eval_batch_size"),
    "per_device_eval_batch_size",
)
if eval_batch_size != 1:
    raise SystemExit("S0 requires one validation row per rank")

loss_contract = require_string(
    settings.get("loss_contract", trainer.get("loss_contract")),
    "loss_contract",
)
if loss_contract != "s0_target_aware_v2":
    raise SystemExit("The S0 loss contract changed")
completion_weight = require_number(
    settings.get(
        "completion_loss_weight", trainer.get("completion_loss_weight")
    ),
    "completion_loss_weight",
)
mapping_weight = require_number(
    settings.get(
        "mapping_target_loss_weight",
        trainer.get("mapping_target_loss_weight"),
    ),
    "mapping_target_loss_weight",
)
if completion_weight != 0.5 or mapping_weight != 0.5:
    raise SystemExit("The S0 target-aware loss weights changed")

adapter_manifest = tokenizer.get("token_adapter_manifest")
if adapter_manifest is None:
    adapter_path = ""
elif isinstance(adapter_manifest, str) and adapter_manifest:
    adapter_path = absolute_path(adapter_manifest, "token_adapter_manifest")
else:
    raise SystemExit("token_adapter_manifest must be null or a path")
if bool(adapter_path) != (tokenizer_method != "plain_base_tokenizer"):
    raise SystemExit("Only a custom tokenizer needs a token adapter manifest")

trainer_path = absolute_path(trainer.get("path"), "trainer path")
required_trainer = (
    root / (
        "scripts/train_sft_dp_tp_ep.py"
        if train_mode == "lora"
        else "scripts/train_sft_full_zero3.py"
    )
).resolve()
if Path(trainer_path) != required_trainer:
    raise SystemExit("The selected trainer path changed")

deepspeed_path = ""
deepspeed_sha256 = ""
expected_model_parameters = ""
if train_mode == "full_finetune":
    deepspeed_path = absolute_path(
        trainer.get("deepspeed_config"), "deepspeed_config"
    )
    deepspeed_sha256 = require_sha256(
        trainer.get("deepspeed_config_sha256"),
        "deepspeed_config_sha256",
    )
    expected_model_parameters = str(
        require_int(
            trainer.get("expected_model_parameters"),
            "expected_model_parameters",
        )
    )

save_strategy = require_string(
    checkpoint.get("save_strategy"), "save_strategy"
)
if train_mode == "full_finetune" and save_strategy != "no":
    raise SystemExit("Full fine-tune must save only the final model")
if train_mode == "lora" and save_strategy not in {"no", "steps"}:
    raise SystemExit("The LoRA save strategy is invalid")
save_steps = checkpoint.get("save_steps", 1)
save_total_limit = checkpoint.get("save_total_limit", 1)
save_steps = require_int(save_steps, "save_steps")
save_total_limit = require_int(save_total_limit, "save_total_limit")

config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
output_root = absolute_path(method.get("output_root"), "output_root")
model_id = require_string(model.get("model_id"), "model_id")
model_path = absolute_path(model.get("path"), "model path")
corpus_root = absolute_path(corpus.get("root"), "corpus root")
validation_path = str(
    (
        Path(corpus_root)
        / require_string(corpus.get("validation_file"), "validation_file")
    ).resolve()
)
validation_sha256 = require_sha256(
    corpus.get("validation_sha256"),
    "corpus validation_sha256",
)
evaluator_path = (
    root / "data/world_model_v2/eval/s0_human_identifiers_v4/manifest.json"
).resolve()
try:
    evaluator_manifest = json.loads(evaluator_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"The evaluator manifest is invalid: {error}") from error
validation_panel = require_object(
    evaluator_manifest.get("validation"),
    "evaluator validation panel",
)
if absolute_path(
    validation_panel.get("questions_path"),
    "validation questions_path",
) != validation_path:
    raise SystemExit("The validation question path differs from the corpus")
if require_sha256(
    validation_panel.get("questions_sha256"),
    "validation questions_sha256",
) != validation_sha256:
    raise SystemExit("The validation question identity differs from the corpus")
if require_int(
    validation_panel.get("row_count"),
    "validation row_count",
) != validation_rows:
    raise SystemExit("The validation row count differs from the corpus")
validation_answer_key_path = absolute_path(
    validation_panel.get("answer_key_path"),
    "validation answer_key_path",
)
validation_answer_key_sha256 = require_sha256(
    validation_panel.get("answer_key_sha256"),
    "validation answer_key_sha256",
)
base_schema = require_string(
    model.get("base_model_artifact_schema"),
    "base_model_artifact_schema",
)
if base_schema != "mentor-rl-base-model-artifact-v2":
    raise SystemExit("The base model artifact schema changed")

values = {
    "S0_RUN_CONFIG": str(config_path),
    "S0_RUN_CONFIG_SHA256": config_sha256,
    "S0_CONFIG_ID": config_id,
    "S0_RUN_SCOPE": run_scope,
    "S0_METHOD_ID": method_id,
    "S0_TRAIN_MODE": train_mode,
    "S0_TOKENIZER_METHOD": tokenizer_method,
    "S0_TOKENIZER_ARM_ROOT": absolute_path(
        tokenizer.get("arm_root"), "tokenizer arm_root"
    ),
    "S0_TOKENIZER_ARM_MANIFEST_SHA256": require_sha256(
        tokenizer.get("arm_manifest_sha256"), "arm_manifest_sha256"
    ),
    "TOKENIZER_PATH": absolute_path(
        tokenizer.get("tokenizer_path"), "tokenizer_path"
    ),
    "TOKEN_ADAPTER_MANIFEST": adapter_path,
    "S0_TOKENIZER_MANIFEST_SHA256": require_sha256(
        tokenizer.get("tokenizer_manifest_sha256"),
        "tokenizer_manifest_sha256",
    ),
    "MODEL_ID": model_id,
    "MODEL_PATH": model_path,
    "MODEL_IDENTITY_SHA256": require_sha256(
        model.get("model_identity_sha256"), "model_identity_sha256"
    ),
    "BASE_MODEL_ARTIFACT_SCHEMA": base_schema,
    "BASE_MODEL_ARTIFACT_SHA256": require_sha256(
        model.get("base_model_artifact_sha256"),
        "base_model_artifact_sha256",
    ),
    "BASE_MODEL_WEIGHT_SHARD_COUNT": str(
        require_int(model.get("weight_shard_count"), "weight_shard_count")
    ),
    "S0_CORPUS_ROOT": corpus_root,
    "S0_CORPUS_MANIFEST_SHA256": require_sha256(
        corpus.get("manifest_sha256"), "corpus manifest_sha256"
    ),
    "S0_TRAIN_PATH": str(
        (Path(absolute_path(corpus.get("root"), "corpus root"))
         / require_string(corpus.get("train_file"), "train_file")).resolve()
    ),
    "S0_TRAIN_SHA256": require_sha256(
        corpus.get("train_sha256"), "corpus train_sha256"
    ),
    "S0_TRAIN_ROWS": str(train_rows),
    "S0_VALIDATION_PATH": validation_path,
    "S0_VALIDATION_SHA256": validation_sha256,
    "S0_VALIDATION_ROWS": str(validation_rows),
    "S0_VALIDATION_ANSWER_KEY_PATH": validation_answer_key_path,
    "S0_VALIDATION_ANSWER_KEY_SHA256": validation_answer_key_sha256,
    "S0_EVALUATOR_MANIFEST": str(evaluator_path),
    "TRAINER_PATH": trainer_path,
    "OUTPUT_ROOT": output_root,
    "NUM_NODES": str(num_nodes),
    "GPUS_PER_NODE": str(gpus_per_node),
    "RANKS_PER_NODE": str(ranks_per_node),
    "WORLD_SIZE_EXPECTED": str(num_nodes * ranks_per_node),
    "TP_SIZE": str(tp_size),
    "EP_SIZE": str(ep_size),
    "DATA_PARALLEL_SIZE": str(data_parallel_size),
    "TRAIN_NUM_EPOCHS": str(epochs),
    "TRAIN_MAX_STEPS": str(max_steps),
    "TRAIN_BATCH_SIZE": str(batch_size),
    "GRAD_ACCUM_STEPS": str(grad_accum),
    "GLOBAL_BATCH_SIZE": str(global_batch),
    "LEARNING_RATE": str(learning_rate),
    "LR_SCHEDULER_TYPE": require_string(
        settings.get("lr_scheduler_type"), "lr_scheduler_type"
    ),
    "WARMUP_RATIO": str(warmup_ratio),
    "WEIGHT_DECAY": str(weight_decay),
    "MAX_GRAD_NORM": str(max_grad_norm),
    "MAX_LENGTH": str(require_int(settings.get("max_length"), "max_length")),
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
        int(require_bool(
            settings.get("preserve_dataset_order"),
            "preserve_dataset_order",
        ))
    ),
    "LOCAL_FILES_ONLY": str(
        int(require_bool(settings.get("local_files_only"), "local_files_only"))
    ),
    "BF16": str(int(require_bool(settings.get("bf16"), "bf16"))),
    "LOSS_CONTRACT": loss_contract,
    "LORA_R": str(fine_tune.get("lora_rank", "")),
    "LORA_ALPHA": str(fine_tune.get("lora_alpha", "")),
    "LORA_DROPOUT": str(fine_tune.get("lora_dropout", "")),
    "AUTOGRAD_MULTITHREADING": str(
        int(require_bool(
            settings.get(
                "autograd_multithreading",
                trainer.get("autograd_multithreading", True),
            ),
            "autograd_multithreading",
        ))
    ),
    "STRICT_TESTED_STACK": str(
        int(require_bool(
            settings.get(
                "strict_tested_stack",
                trainer.get("strict_tested_stack", True),
            ),
            "strict_tested_stack",
        ))
    ),
    "DS_CONFIG": deepspeed_path,
    "DS_CONFIG_SHA256_EXPECTED": deepspeed_sha256,
    "EXPECTED_MODEL_PARAMETERS": expected_model_parameters,
}
for name, value in values.items():
    print(f"{name}={shlex.quote(value)}")
