#!/usr/bin/env python3
"""Validate one canonical world-model train contract."""

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

from runtime.world_model.s1_overlay import (  # noqa: E402
    S1_OVERLAY_V3_SCHEMA_VERSION,
    S1_OVERLAY_V4_SCHEMA_VERSION,
    S1_OVERLAY_V5_SCHEMA_VERSION,
    S1_OVERLAY_V7_SCHEMA_VERSION,
    S1_OVERLAY_V8_SCHEMA_VERSION,
    S1_OVERLAY_V9_SCHEMA_VERSION,
)
from runtime.world_model_training import (  # noqa: E402
    S0_EXPOSURE_SCOPE_BY_RUN_SCOPE,
    S1_V3_TRACE_FAMILIES,
    derive_optimizer_step_schedule,
)


S1_QUALIFICATION_SCHEMA = (
    "mentor-rl-world-model-s1-20b-qualification-v1"
)
S1_SMOKE_SCHEMA = "mentor-rl-world-model-s1-20b-smoke-v1"
S1_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v1"
# S1 v2 starts from an S0 v4 checkpoint and trains on the S1 v2 overlay corpus.
S1_V2_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v2"
# S1 v3 keeps the v2 parent and allocation and trains on the prompt-bank corpus.
S1_V3_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v3"
# S1 v4 is the v3 corpus with the S1.1.10 status rule in its training reasoning.
S1_V4_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v4"
# S1 v5 keeps the v4 recipe and uses the reviewed corpus revisions.
S1_V5_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v5"
# S1 v6 keeps the comparison recipe and uses the reviewed reasoning/retry corpus.
S1_V6_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v6"
# S1 v8 changes training data only and keeps the v6 comparison recipe.
S1_V8_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v8"
# V9 keeps the v6 optimizer recipe and adds observational loss/checkpoint receipts.
S1_V9_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v9"
# V10 changes the corpus while retaining v9's recipe and required telemetry.
S1_V10_PRODUCTION_SCHEMA = "mentor-rl-world-model-s1-120b-training-v10"
S1_SCHEMAS = frozenset(
    {
        S1_QUALIFICATION_SCHEMA,
        S1_SMOKE_SCHEMA,
        S1_PRODUCTION_SCHEMA,
        S1_V2_PRODUCTION_SCHEMA,
        S1_V3_PRODUCTION_SCHEMA,
        S1_V4_PRODUCTION_SCHEMA,
        S1_V5_PRODUCTION_SCHEMA,
        S1_V6_PRODUCTION_SCHEMA,
        S1_V8_PRODUCTION_SCHEMA,
        S1_V9_PRODUCTION_SCHEMA,
        S1_V10_PRODUCTION_SCHEMA,
    }
)
S1_FULL_SCHEMAS = frozenset(
    {
        S1_QUALIFICATION_SCHEMA,
        S1_PRODUCTION_SCHEMA,
        S1_V2_PRODUCTION_SCHEMA,
        S1_V3_PRODUCTION_SCHEMA,
        S1_V4_PRODUCTION_SCHEMA,
        S1_V5_PRODUCTION_SCHEMA,
        S1_V6_PRODUCTION_SCHEMA,
        S1_V8_PRODUCTION_SCHEMA,
        S1_V9_PRODUCTION_SCHEMA,
        S1_V10_PRODUCTION_SCHEMA,
    }
)
S1_V2_OVERLAY_PATH = "runtime/world_model/s1_v2.yaml"
S1_OVERLAY_SCHEMA_VERSION = "mentor-rl-world-model-s1-overlay-v1"
# Each overlay-built S1 schema names its overlay and overlay schema.
S1_OVERLAYS = {
    S1_V2_PRODUCTION_SCHEMA: (S1_V2_OVERLAY_PATH, S1_OVERLAY_SCHEMA_VERSION),
    S1_V3_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v3.yaml",
        "mentor-rl-world-model-s1-overlay-v2",
    ),
    S1_V4_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v4.yaml",
        S1_OVERLAY_V3_SCHEMA_VERSION,
    ),
    S1_V5_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v5.yaml",
        S1_OVERLAY_V4_SCHEMA_VERSION,
    ),
    S1_V6_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v6.yaml",
        S1_OVERLAY_V5_SCHEMA_VERSION,
    ),
    S1_V8_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v8.yaml",
        S1_OVERLAY_V7_SCHEMA_VERSION,
    ),
    S1_V9_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v9.yaml",
        S1_OVERLAY_V8_SCHEMA_VERSION,
    ),
    S1_V10_PRODUCTION_SCHEMA: (
        "runtime/world_model/s1_v10.yaml",
        S1_OVERLAY_V9_SCHEMA_VERSION,
    ),
}
S1_FULL_RUN_SETTINGS = {
    "num_nodes": 16,
    "gpus_per_node": 8,
    "tp_size": 8,
    "ep_size": 8,
    "data_parallel_size": 16,
    "num_train_epochs": 1,
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
    "save_steps": 200,
    "save_total_limit": 3,
    "seed": 900913,
    "preserve_dataset_order": False,
    "autograd_multithreading": True,
    "local_files_only": True,
    "strict_tested_stack": True,
    "bf16": True,
}
S1_FULL_CORPUS_LAYOUT = {
    "root": "data/world_model_v3/corpora/s1",
    "train_file": "train.jsonl",
    "train_rows": 55276,
    "validation_file": "validation.jsonl",
    "validation_rows": 6875,
    "evaluator_manifest": "data/world_model_v3/eval/s1/manifest.json",
}
# S1 v2 keeps the v1 row counts and schedule in a separate published location.
S1_FULL_CORPUS_LAYOUTS = {
    S1_QUALIFICATION_SCHEMA: S1_FULL_CORPUS_LAYOUT,
    S1_PRODUCTION_SCHEMA: S1_FULL_CORPUS_LAYOUT,
    S1_V2_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1",
        "evaluator_manifest": "data/world_model_v4/eval/s1/manifest.json",
    },
    S1_V3_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v3",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v3/manifest.json",
    },
    S1_V4_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v4",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v4/manifest.json",
    },
    S1_V5_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v5",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v5/manifest.json",
    },
    S1_V6_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v6",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v6/manifest.json",
    },
    S1_V8_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v8",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v8/manifest.json",
    },
    S1_V9_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v9",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v9/manifest.json",
    },
    S1_V10_PRODUCTION_SCHEMA: {
        **S1_FULL_CORPUS_LAYOUT,
        "root": "data/world_model_v4/corpora/s1_v10",
        "evaluator_manifest": "data/world_model_v4/eval/s1_v10/manifest.json",
    },
}
S1_FULL_UPDATES_PER_EPOCH = 1728
S1_FULL_TOTAL_STEPS = 1728
# The wide S1 v2 run uses one row per replica on 184 nodes. Its larger batch
# takes fewer steps, so it trains 2 epochs at a higher learning rate.
S1_V2_WIDE_RUN_SETTINGS = {
    **S1_FULL_RUN_SETTINGS,
    "num_nodes": 184,
    "data_parallel_size": 184,
    "num_train_epochs": 2,
    "gradient_accumulation_steps": 1,
    "global_batch_size": 184,
    "learning_rate": 0.0002,
    "warmup_ratio": 0.05,
}
# The wide S1 v3 run keeps the 184-node topology and the 16-node learning
# rate (1e-4, 3% warmup). Six epochs give 1,806 optimizer steps, about the
# 1,728 steps of the 16-node run.
S1_V3_WIDE_RUN_SETTINGS = {
    **S1_V2_WIDE_RUN_SETTINGS,
    "num_train_epochs": 6,
    "learning_rate": 0.0001,
    "warmup_ratio": 0.03,
}
# The 3-epoch S1 v3 run ends where the 6-epoch run's validation loss was
# lowest (903 steps), with the cosine schedule completed by then.
S1_V3_WIDE_3_EPOCH_RUN_SETTINGS = {**S1_V3_WIDE_RUN_SETTINGS, "num_train_epochs": 3}
# The 32-node S1 v3 run keeps the 16-node optimization (batch 32, 1 epoch,
# 1,728 steps at 1e-4) with one row per replica instead of 2 accumulated.
S1_V3_32_NODE_RUN_SETTINGS = {
    **S1_FULL_RUN_SETTINGS,
    "num_nodes": 32,
    "data_parallel_size": 32,
    "gradient_accumulation_steps": 1,
}
S1_V9_RUN_SETTINGS = {
    **S1_V3_32_NODE_RUN_SETTINGS,
    "diagnostic_snapshot_steps": [400, 800, 1200, 1600],
}
# Each full S1 config ID selects its run settings and optimizer-step schedule.
S1_FULL_RUN_PROFILES = {
    "world_model_s1_20b_qualification_v1": (
        S1_FULL_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v1": (
        S1_FULL_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v2": (
        S1_FULL_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v2_184n": (S1_V2_WIDE_RUN_SETTINGS, 301, 602),
    "world_model_s1_120b_training_v3_184n": (S1_V3_WIDE_RUN_SETTINGS, 301, 1806),
    "world_model_s1_120b_training_v3_32n": (
        S1_V3_32_NODE_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v3_184n_3ep": (
        S1_V3_WIDE_3_EPOCH_RUN_SETTINGS,
        301,
        903,
    ),
    # S1 v4 repeats the 32-node v3 run, so only the corpus changes.
    "world_model_s1_120b_training_v4_32n": (
        S1_V3_32_NODE_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v5_32n": (
        S1_V3_32_NODE_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v6_32n": (
        S1_V3_32_NODE_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v8_32n": (
        S1_V3_32_NODE_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v9_32n": (
        S1_V9_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
    "world_model_s1_120b_training_v10_32n": (
        S1_V9_RUN_SETTINGS,
        S1_FULL_UPDATES_PER_EPOCH,
        S1_FULL_TOTAL_STEPS,
    ),
}
S0_V4_TRAINING_SCHEMA = "mentor-rl-world-model-s0-120b-training-v4"
# The snapshot pipeline is retired; S0 v4 configs keep snapshot_steps at 0.
S0_V4_SNAPSHOT_STEPS = frozenset({0})
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
    "snapshot_steps": 0,
    "seed": 900913,
    "preserve_dataset_order": False,
    "autograd_multithreading": True,
    "local_files_only": True,
    "strict_tested_stack": True,
    "bf16": True,
}
SCHEMAS = frozenset(
    {
        "mentor-rl-world-model-s0-20b-tool-trajectory-qualification-v6",
        "mentor-rl-world-model-s0-120b-tool-trajectory-training-v6",
        "mentor-rl-world-model-s0-20b-shared-trace-qualification-v7",
        "mentor-rl-world-model-s0-120b-shared-trace-training-v7",
        "mentor-rl-world-model-s0-120b-training-v3",
        S0_V4_TRAINING_SCHEMA,
        *S1_SCHEMAS,
    }
)
DATASET_ID = "world_model_v2_s0_human_identifier_trajectories_v6"
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-manifest-v6"
TRAINING_CONTRACT = "tool_trajectory_v1"
EVALUATION_CONTRACT = "unseen_component_tool_trajectory_v1"
LOSS_CONTRACT = "s0_tool_trajectory_v1"
SHARED_DATASET_ID = (
    "world_model_v2_s0_human_identifier_supervised_traces_v7"
)
SHARED_MANIFEST_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-shared-manifest-v7"
)
SHARED_TRAINING_CONTRACT = "supervised_trace_v1"
SHARED_EVALUATION_CONTRACT = "unseen_component_supervised_trace_v1"
SHARED_LOSS_CONTRACT = "supervised_explanation_trace_v1"
SHARED_EVALUATOR_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-shared-evaluator-manifest-v7"
)
V3_DATASET_ID = "world_model_v3_s0_human_identifiers"
V3_MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-manifest-v1"
V3_CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s0-corpus-v1"
V3_EVALUATION_CONTRACT = "held_out_multiplex_sft_v3"
V3_LOSS_CONTRACT = "multiplex_sft_v3"
V3_EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-v3"
V4_DATASET_ID = "world_model_v4_s0_human_identifiers"
V4_MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-manifest-v2"
V4_CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s0-corpus-v2"
V4_EVALUATION_CONTRACT = "held_out_multiplex_sft_v4"
V4_PRIORITY_LOSS_CONTRACT = "multiplex_sft_v4_priority"
V4_LOSS_CONTRACTS = frozenset(
    {V3_LOSS_CONTRACT, V4_PRIORITY_LOSS_CONTRACT}
)
V4_EVALUATOR_SCHEMA_VERSION = "mentor-rl-world-model-s0-evaluator-v4"
S1_V3_DATASET_ID = "world_model_v3_s1_coordinate_system"
S1_V3_MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s1-manifest-v1"
S1_V3_CORPUS_SCHEMA_VERSION = "mentor-rl-world-model-s1-corpus-v1"
S1_V3_EVALUATION_CONTRACT = "held_out_s1_multiplex_sft_v3"
S1_V3_EVALUATOR_SCHEMA_VERSION = (
    "mentor-rl-world-model-s1-evaluator-v1"
)
S1_SMOKE_SELECTION_ALGORITHM = (
    "longest_plus_sha256_rank_per_behavior_signature_v1"
)
S1_SMOKE_SIGNATURE_FIELDS = [
    "step",
    "question_family",
    "source_trace",
    "book_mode",
    "blocked_reason_code",
    "final_status",
    "continuation_decision",
    "result_keys",
    "tool_call_path",
]


def _contract_for_schema(schema: str) -> dict[str, str]:
    """Return the fixed corpus contract for one train schema."""

    if schema in S1_SCHEMAS:
        return {
            "dataset_id": S1_V3_DATASET_ID,
            "manifest_schema_version": S1_V3_MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": S1_V3_CORPUS_SCHEMA_VERSION,
            "training_contract": V3_LOSS_CONTRACT,
            "evaluation_contract": S1_V3_EVALUATION_CONTRACT,
            "loss_contract": V3_LOSS_CONTRACT,
            "evaluator_schema_version": S1_V3_EVALUATOR_SCHEMA_VERSION,
        }
    if schema == S0_V4_TRAINING_SCHEMA:
        return {
            "dataset_id": V4_DATASET_ID,
            "manifest_schema_version": V4_MANIFEST_SCHEMA_VERSION,
            "corpus_schema_version": V4_CORPUS_SCHEMA_VERSION,
            "training_contract": V3_LOSS_CONTRACT,
            "evaluation_contract": V4_EVALUATION_CONTRACT,
            "loss_contract": V4_PRIORITY_LOSS_CONTRACT,
            "evaluator_schema_version": V4_EVALUATOR_SCHEMA_VERSION,
        }
    if schema.endswith("-v6"):
        return {
            "dataset_id": DATASET_ID,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "training_contract": TRAINING_CONTRACT,
            "evaluation_contract": EVALUATION_CONTRACT,
            "loss_contract": LOSS_CONTRACT,
            "evaluator_schema_version": (
                "mentor-rl-world-model-s0-evaluator-manifest-v6"
            ),
        }
    if schema.endswith("-v7"):
        return {
            "dataset_id": SHARED_DATASET_ID,
            "manifest_schema_version": SHARED_MANIFEST_SCHEMA_VERSION,
            "training_contract": SHARED_TRAINING_CONTRACT,
            "evaluation_contract": SHARED_EVALUATION_CONTRACT,
            "loss_contract": SHARED_LOSS_CONTRACT,
            "evaluator_schema_version": SHARED_EVALUATOR_SCHEMA_VERSION,
        }
    return {
        "dataset_id": V3_DATASET_ID,
        "manifest_schema_version": V3_MANIFEST_SCHEMA_VERSION,
        "corpus_schema_version": V3_CORPUS_SCHEMA_VERSION,
        "training_contract": "multiplex_sft_v3",
        "evaluation_contract": V3_EVALUATION_CONTRACT,
        "loss_contract": V3_LOSS_CONTRACT,
        "evaluator_schema_version": V3_EVALUATOR_SCHEMA_VERSION,
    }


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


def require_v4_snapshot_steps(value: object, name: str) -> int:
    """Return the S0 v4 snapshot interval, which must be zero."""

    if type(value) is not int or value not in S0_V4_SNAPSHOT_STEPS:
        raise SystemExit(f"{name} must be 0")
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


def validate_parent_checkpoint(
    root: Path,
    initialization: object,
    *,
    model_identity_sha256: str,
    tokenizer_manifest_sha256: str,
    lora_rank: int,
    lora_alpha: int,
) -> dict[str, str]:
    """Validate the completed S0 v4 checkpoint that starts one S1 run."""

    initialization = require_object(initialization, "initialization")
    if initialization.get("parent") != "s0_checkpoint":
        raise SystemExit("initialization parent must be s0_checkpoint")
    checkpoint = Path(
        absolute_path(
            root,
            initialization.get("checkpoint_path"),
            "initialization checkpoint_path",
        )
    )
    run_id = require_string(
        initialization.get("run_id"), "initialization run_id"
    )
    adapter_sha256 = require_sha256(
        initialization.get("adapter_sha256"), "initialization adapter_sha256"
    )
    parent_loss_contract = require_string(
        initialization.get("loss_contract"), "initialization loss_contract"
    )
    if require_int(
        initialization.get("lora_rank"), "initialization lora_rank"
    ) != lora_rank:
        raise SystemExit("The parent LoRA rank differs from this run")
    if not checkpoint.is_dir():
        raise SystemExit("The parent checkpoint directory is absent")

    # The consolidated adapter is the exact tensor set that the trainer loads.
    verify_file(
        checkpoint / "adapter_model.safetensors",
        adapter_sha256,
        "parent adapter",
    )
    manifest = read_json(
        checkpoint / "tp_adapter_manifest.json", "the parent adapter manifest"
    )
    identity = require_object(
        manifest.get("identity"), "the parent adapter identity"
    )
    lora = require_object(manifest.get("lora"), "the parent LoRA spec")
    objective = require_object(
        manifest.get("objective"), "the parent objective"
    )
    if (
        manifest.get("format") != "mentor-rl-s0-tp-lora-v1"
        or identity.get("run_id") != run_id
        or identity.get("model_identity_sha256") != model_identity_sha256
        or identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or lora.get("r") != lora_rank
        or lora.get("alpha") != lora_alpha
        or objective.get("loss_contract") != parent_loss_contract
    ):
        raise SystemExit("The parent checkpoint contract changed")

    # Require one complete S0 v4 run, not an interim or diagnostic state.
    run_config_path = checkpoint / "run_contract/run_config.json"
    if (
        not run_config_path.is_file()
        or sha256_file(run_config_path) != identity.get("run_config_sha256")
        or read_json(run_config_path, "the parent run config").get(
            "schema_version"
        )
        != S0_V4_TRAINING_SCHEMA
    ):
        raise SystemExit("The parent is not an S0 v4 training run")
    exposure = read_json(
        checkpoint / "run_contract/training_exposure.json",
        "the parent training exposure",
    )
    claimed = exposure.get("manifest_sha256")
    schedule = exposure.get("schedule")
    exposure_contract = exposure.get("exposure_contract")
    if (
        claimed
        != stable_sha256(
            {key: value for key, value in exposure.items() if key != "manifest_sha256"}
        )
        or exposure.get("status") != "complete"
        or exposure.get("run_id") != run_id
        or not isinstance(schedule, dict)
        or not isinstance(schedule.get("total_steps"), int)
        or schedule.get("completed_global_step") != schedule["total_steps"]
        or not isinstance(exposure_contract, dict)
        or exposure_contract.get("satisfied") is not True
    ):
        raise SystemExit("The parent S0 run is not complete")
    return {
        "INIT_ADAPTER_DIR": str(checkpoint),
        "INIT_ADAPTER_SHA256": adapter_sha256,
        "INIT_ADAPTER_RUN_ID": run_id,
    }


def validate_train_contract(
    root: Path,
    config_path: Path,
    method_id: str,
) -> dict[str, str]:
    """Validate one canonical world-model train contract."""

    root = root.resolve()
    config_path = config_path.resolve()
    try:
        config_path.relative_to(root)
    except ValueError as error:
        raise SystemExit("RUN_CONFIG must stay in the repository") from error
    config = read_json(config_path, "run config")
    schema = config.get("schema_version")
    if schema not in SCHEMAS:
        raise SystemExit(
            f"The train config schema is not supported: {schema!r}"
        )
    contract = _contract_for_schema(str(schema))
    if schema in S1_OVERLAYS and config.get("initialization") is None:
        version = str(schema).rsplit("-", 1)[-1]
        raise SystemExit(f"S1 {version} must start from an S0 v4 checkpoint")

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
    if "max_steps" in settings:
        raise SystemExit(
            "run_settings max_steps is not supported; set num_train_epochs"
        )
    overrides = require_object(
        method.get("run_settings", {}), "method run_settings"
    )
    if "max_steps" in overrides:
        raise SystemExit(
            "method run_settings max_steps is not supported; "
            "set num_train_epochs"
        )
    unknown_overrides = sorted(set(overrides) - set(settings))
    if unknown_overrides:
        raise SystemExit(
            f"The method run_settings has unknown keys: {unknown_overrides}"
        )
    settings.update(overrides)
    if schema == S0_V4_TRAINING_SCHEMA and settings != S0_V4_RUN_SETTINGS:
        raise SystemExit("The S0 v4 training/snapshot schedule changed")

    run_scope = require_string(config.get("run_scope"), "run_scope")
    if run_scope not in S0_EXPOSURE_SCOPE_BY_RUN_SCOPE:
        allowed = ", ".join(sorted(S0_EXPOSURE_SCOPE_BY_RUN_SCOPE))
        raise SystemExit(f"run_scope must be one of: {allowed}")
    is_20b = "-20b-" in str(schema)
    if is_20b:
        config_id = require_string(
            config.get("qualification_id"), "qualification_id"
        )
        expected_prefix = "oss20b"
    else:
        config_id = require_string(
            config.get("training_id"), "training_id"
        )
        expected_prefix = "oss120b"

    if schema in S1_SCHEMAS:
        expected_s1_configs = {
            S1_SMOKE_SCHEMA: (
                {"world_model_s1_20b_smoke_v1"},
                "qualification",
                False,
            ),
            S1_QUALIFICATION_SCHEMA: (
                {"world_model_s1_20b_qualification_v1"},
                "qualification",
                False,
            ),
            S1_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v1"},
                "production",
                True,
            ),
            S1_V2_PRODUCTION_SCHEMA: (
                {
                    "world_model_s1_120b_training_v2",
                    "world_model_s1_120b_training_v2_184n",
                },
                "production",
                True,
            ),
            S1_V3_PRODUCTION_SCHEMA: (
                {
                    "world_model_s1_120b_training_v3_32n",
                    "world_model_s1_120b_training_v3_184n",
                    "world_model_s1_120b_training_v3_184n_3ep",
                },
                "production",
                True,
            ),
            S1_V4_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v4_32n"},
                "production",
                True,
            ),
            S1_V5_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v5_32n"},
                "production",
                True,
            ),
            S1_V6_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v6_32n"},
                "production",
                True,
            ),
            S1_V8_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v8_32n"},
                "production",
                True,
            ),
            S1_V9_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v9_32n"},
                "production",
                True,
            ),
            S1_V10_PRODUCTION_SCHEMA: (
                {"world_model_s1_120b_training_v10_32n"},
                "production",
                True,
            ),
        }
        expected_config_ids, expected_scope, expected_production = (
            expected_s1_configs[str(schema)]
        )
        if (
            config_id not in expected_config_ids
            or run_scope != expected_scope
            or require_bool(
                config.get("promotion_eligible"), "promotion_eligible"
            )
            is not expected_production
            or require_bool(
                config.get("production_matrix_cell"),
                "production_matrix_cell",
            )
            is not expected_production
        ):
            raise SystemExit("The S1 run identity changed")
    if schema in S1_FULL_SCHEMAS:
        expected_layout = S1_FULL_CORPUS_LAYOUTS[schema]
        expected_settings = S1_FULL_RUN_PROFILES[config_id][0]
        corpus_layout = {
            key: corpus.get(key) for key in expected_layout
        }
        if (
            stable_sha256(settings) != stable_sha256(expected_settings)
            or corpus_layout != expected_layout
        ):
            raise SystemExit(
                "The S1 full training schedule/data contract changed"
            )

    segment_loss_telemetry = require_bool(
        trainer.get("segment_loss_telemetry", False), "segment_loss_telemetry"
    )
    if schema in {S1_V9_PRODUCTION_SCHEMA, S1_V10_PRODUCTION_SCHEMA} and not segment_loss_telemetry:
        version = str(schema).rsplit("-", 1)[-1]
        raise SystemExit(f"S1 {version} requires segment loss telemetry")
    diagnostic_snapshot_steps = settings.get("diagnostic_snapshot_steps", [])
    if (
        not isinstance(diagnostic_snapshot_steps, list)
        or any(type(step) is not int or step < 1 for step in diagnostic_snapshot_steps)
        or diagnostic_snapshot_steps != sorted(set(diagnostic_snapshot_steps))
    ):
        raise SystemExit("diagnostic_snapshot_steps must be increasing positive steps")

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
    if schema == S0_V4_TRAINING_SCHEMA:
        if (
            lora_rank not in {32, 128}
            or trainer.get("fine_tune_configuration")
            != f"lora_r{lora_rank}"
            or require_int(trainer.get("lora_alpha"), "lora_alpha")
            != 2 * lora_rank
            or require_number(trainer.get("lora_dropout"), "lora_dropout")
            != 0.0
        ):
            raise SystemExit("The v4 LoRA rank/alpha contract changed")
    elif trainer.get("fine_tune_configuration") != "lora_r32":
        raise SystemExit("The LoRA trainer contract changed")
    if method_id != (
        f"{expected_prefix}-plain-base-tokenizer-lora-r{lora_rank}"
    ):
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
    if tp_size != gpus_per_node or ep_size != gpus_per_node:
        raise SystemExit(
            "One node must hold one complete TP and EP replica"
        )
    if data_parallel_size != num_nodes:
        raise SystemExit(
            "The LoRA data parallel size must equal the node count"
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
    if global_batch != data_parallel_size * batch_size * grad_accum:
        raise SystemExit(
            "The global batch size differs from the selected topology"
        )
    train_rows = require_int(
        corpus.get("train_rows"), "corpus train_rows"
    )
    validation_rows = require_int(
        corpus.get("validation_rows"), "corpus validation_rows"
    )
    try:
        schedule = derive_optimizer_step_schedule(
            train_rows,
            num_train_epochs=epochs,
            replica_count=data_parallel_size,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
        )
    except ValueError as error:
        raise SystemExit(
            f"The optimizer step schedule is invalid: {error}"
        ) from error
    if schema in S1_FULL_SCHEMAS and (
        schedule["updates_per_epoch"] != S1_FULL_RUN_PROFILES[config_id][1]
        or schedule["total_steps"] != S1_FULL_RUN_PROFILES[config_id][2]
    ):
        raise SystemExit(
            "The S1 full optimizer-step schedule changed"
        )

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
    if learning_rate <= 0 or not 0 <= warmup_ratio < 1:
        raise SystemExit(
            "The learning rate or warmup ratio is invalid"
        )
    if weight_decay < 0 or max_grad_norm <= 0:
        raise SystemExit(
            "The weight decay or gradient limit is invalid"
        )
    loss_contract = require_string(
        trainer.get("loss_contract"), "loss_contract"
    )
    expected_loss_contracts = (
        V4_LOSS_CONTRACTS
        if schema == S0_V4_TRAINING_SCHEMA
        else frozenset({contract["loss_contract"]})
    )
    if loss_contract not in expected_loss_contracts:
        raise SystemExit("The S0 loss contract changed")
    expected_priority_weights = {
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
    if schema == S0_V4_TRAINING_SCHEMA:
        priority_weights = trainer.get("token_loss_weights")
        if loss_contract == V4_PRIORITY_LOSS_CONTRACT:
            if priority_weights != expected_priority_weights:
                raise SystemExit("The S0 v4 priority loss weights changed")
        elif priority_weights is not None:
            raise SystemExit("The uniform S0 v4 arm has priority weights")
        loss_arm = (
            "priority"
            if loss_contract == V4_PRIORITY_LOSS_CONTRACT
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
    if eval_strategy != "epoch" or not eval_on_start:
        raise SystemExit("The S0 validation contract changed")
    if eval_batch_size != 1:
        raise SystemExit("S0 requires one validation row per rank")
    save_strategy = require_string(
        settings.get("save_strategy"), "save_strategy"
    )
    if save_strategy not in {"no", "steps"}:
        raise SystemExit("The LoRA save strategy is invalid")
    max_length = require_int(settings.get("max_length"), "max_length")
    save_steps = require_int(settings.get("save_steps", 1), "save_steps")
    save_total_limit = require_int(
        settings.get("save_total_limit", 1), "save_total_limit"
    )
    if schema == S0_V4_TRAINING_SCHEMA:
        snapshot_steps = require_v4_snapshot_steps(
            settings.get("snapshot_steps"), "snapshot_steps"
        )
        if (
            num_nodes != 16
            or gpus_per_node != 8
            or data_parallel_size != 16
            or epochs != 3
            or train_rows != 16384
            or validation_rows != 622
            or global_batch != 32
            or schedule["updates_per_epoch"] != 512
            or schedule["total_steps"] != 1536
            or learning_rate != 1e-4
            or warmup_ratio != 0.03
            or weight_decay != 0.0
            or max_grad_norm != 1.0
            or max_length != 4096
            or save_strategy != "steps"
            or save_steps != 512
            or save_total_limit != 3
            or snapshot_steps not in S0_V4_SNAPSHOT_STEPS
        ):
            raise SystemExit("The S0 v4 training/snapshot schedule changed")
        validation_gate = require_object(
            config.get("validation_gate"), "validation_gate"
        )
        if (
            validation_gate.get("checkpoint_set")
            != "each_saved_checkpoint_and_final"
            or validation_gate.get("selection_metric")
            != "validation/hard_failure_count_then_accuracy"
            or require_int(
                validation_gate.get("required_method_count"),
                "validation_gate required_method_count",
            )
            != 1
            or require_bool(
                validation_gate.get("full_registry_access"),
                "validation_gate full_registry_access",
            )
            or require_v4_snapshot_steps(
                validation_gate.get("snapshot_steps"),
                "validation_gate snapshot_steps",
            )
            != snapshot_steps
        ):
            raise SystemExit("The S0 v4 snapshot validation gate changed")
    if schema == S1_SMOKE_SCHEMA and (
        config_id != "world_model_s1_20b_smoke_v1"
        or num_nodes != 1
        or data_parallel_size != 1
        or global_batch != 2
        or epochs != 1
        or train_rows != 248
        or validation_rows != 111
        or schedule["total_steps"] != 124
        or save_strategy != "no"
    ):
        raise SystemExit("The S1 one-node smoke schedule changed")
    if schema in S1_SCHEMAS:
        validation_gate = require_object(
            config.get("validation_gate"), "validation_gate"
        )
        if validation_gate.get("checkpoint_set") != "final":
            raise SystemExit(
                "The S1 qualification gate must select the final checkpoint"
            )
        if validation_gate.get("selection_metric") != "eval/completion_loss":
            raise SystemExit(
                "The S1 qualification selection metric changed"
            )
        if require_int(
            validation_gate.get("required_method_count"),
            "validation_gate required_method_count",
        ) != 1:
            raise SystemExit("The S1 qualification method count changed")
        if require_bool(
            validation_gate.get("test_panel_access"),
            "validation_gate test_panel_access",
        ):
            raise SystemExit("The S1 qualification cannot access the test panel")

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
    is_v3 = contract["training_contract"] == V3_LOSS_CONTRACT
    if is_v3:
        if (
            corpus_manifest.get("dataset_id") != contract["dataset_id"]
            or corpus_manifest.get("schema_version")
            != contract["manifest_schema_version"]
            or corpus_manifest.get("corpus_schema_version")
            != contract["corpus_schema_version"]
            or corpus_manifest.get("contract", {}).get("row_schema_version")
            != "multiplex_sft_v3"
            or corpus_manifest.get("row_counts", {}).get("train")
            != train_rows
            or corpus_manifest.get("row_counts", {}).get("validation")
            != validation_rows
        ):
            raise SystemExit("The v3 corpus manifest contract changed")
        if schema in S1_SCHEMAS or schema == S0_V4_TRAINING_SCHEMA:
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
            runtime_contract_path = Path(
                absolute_path(
                    root,
                    runtime_contract_declared,
                    "corpus runtime contract path",
                )
            )
            runtime_contract_sha256 = require_sha256(
                runtime_contract.get("sha256"),
                "corpus runtime contract SHA-256",
            )
            verify_file(
                runtime_contract_path,
                runtime_contract_sha256,
                "corpus runtime contract",
            )
            if schema in S1_OVERLAYS:
                expected_overlay, expected_overlay_schema = S1_OVERLAYS[schema]
                s1_overlay = require_object(
                    runtime_contract.get("s1_overlay"), "S1 runtime overlay"
                )
                s1_overlay_declared = require_string(
                    s1_overlay.get("path"), "S1 runtime overlay path"
                )
                if (
                    Path(s1_overlay_declared).as_posix() != expected_overlay
                    or s1_overlay.get("schema_version")
                    != expected_overlay_schema
                ):
                    raise SystemExit("The S1 runtime overlay contract changed")
                verify_file(
                    Path(
                        absolute_path(
                            root, s1_overlay_declared, "S1 overlay path"
                        )
                    ),
                    require_sha256(
                        s1_overlay.get("sha256"), "S1 overlay SHA-256"
                    ),
                    "S1 runtime overlay",
                )
            if schema == S0_V4_TRAINING_SCHEMA:
                overlay = require_object(
                    runtime_contract.get("s0_overlay"),
                    "S0 v4 runtime overlay",
                )
                overlay_declared = require_string(
                    overlay.get("path"), "S0 v4 runtime overlay path"
                )
                if (
                    Path(overlay_declared).as_posix()
                    != "runtime/world_model/s0_v4.yaml"
                    or overlay.get("schema_version")
                    != "mentor-rl-world-model-s0-overlay-v1"
                    or overlay.get("metadata_optional_additions")
                    != [
                        "fact_kind",
                        "framing",
                        "prompt_template_id",
                        "rendering_index",
                        "input_variant",
                    ]
                ):
                    raise SystemExit("The S0 v4 runtime overlay contract changed")
                overlay_path = Path(
                    absolute_path(
                        root,
                        overlay_declared,
                        "S0 v4 runtime overlay path",
                    )
                )
                overlay_sha256 = require_sha256(
                    overlay.get("sha256"),
                    "S0 v4 runtime overlay SHA-256",
                )
                verify_file(
                    overlay_path,
                    overlay_sha256,
                    "S0 v4 runtime overlay",
                )
        if schema == S1_SMOKE_SCHEMA:
            subset = require_object(
                corpus_manifest.get("subset"), "S1 smoke subset"
            )
            if (
                subset.get("kind") != "behavior_stratified_smoke_v1"
                or subset.get("algorithm")
                != S1_SMOKE_SELECTION_ALGORITHM
                or subset.get("behavior_signature_fields")
                != S1_SMOKE_SIGNATURE_FIELDS
            ):
                raise SystemExit("The S1 smoke selection contract changed")
            source_manifest = require_object(
                subset.get("source_manifest"),
                "S1 smoke source manifest",
            )
            source_manifest_path = Path(
                absolute_path(
                    root,
                    source_manifest.get("path"),
                    "S1 smoke source manifest path",
                )
            )
            source_manifest_sha256 = require_sha256(
                source_manifest.get("sha256"),
                "S1 smoke source manifest SHA-256",
            )
            verify_file(
                source_manifest_path,
                source_manifest_sha256,
                "S1 smoke source manifest",
            )
            subset_splits = require_object(
                subset.get("splits"), "S1 smoke subset splits"
            )
            expected_smoke_splits = {
                "train": (2, 124, 248, {"S0": 22, "S1": 226}),
                "validation": (1, 111, 111, {"S1": 111}),
                "test": (1, 111, 111, {"S1": 111}),
            }
            for split, (
                rows_per_signature,
                signature_count,
                selected_count,
                step_counts,
            ) in expected_smoke_splits.items():
                split_contract = require_object(
                    subset_splits.get(split),
                    f"S1 smoke {split} selection",
                )
                if (
                    split_contract.get("rows_per_behavior_signature")
                    != rows_per_signature
                    or split_contract.get("source_signature_count")
                    != signature_count
                    or split_contract.get("selected_row_count")
                    != selected_count
                    or split_contract.get("all_source_signatures_covered")
                    is not True
                    or split_contract.get("step_counts") != step_counts
                ):
                    raise SystemExit(
                        f"The S1 smoke {split} coverage changed"
                    )
            family_counts = require_object(
                corpus_manifest.get("family_counts"),
                "S1 smoke family counts",
            )
            for split in ("train", "validation", "test"):
                split_families = require_object(
                    family_counts.get(split),
                    f"S1 smoke {split} family counts",
                )
                if (
                    set(split_families) != set(S1_V3_TRACE_FAMILIES)
                    or any(
                        type(count) is not int or count < 1
                        for count in split_families.values()
                    )
                ):
                    raise SystemExit(
                        f"The S1 smoke {split} family coverage changed"
                    )
    elif (
        corpus_manifest.get("dataset_id") != contract["dataset_id"]
        or corpus_manifest.get("schema_version")
        != contract["manifest_schema_version"]
        or corpus_manifest.get("training_contract")
        != contract["training_contract"]
        or corpus_manifest.get("evaluation_contract")
        != contract["evaluation_contract"]
        or corpus_manifest.get("row_counts", {}).get("train")
        != train_rows
        or corpus_manifest.get("row_counts", {}).get("validation")
        != validation_rows
    ):
        raise SystemExit("The corpus manifest contract changed")
    if schema.endswith("-v7"):
        if (
            corpus_manifest.get("loss_contract")
            != contract["loss_contract"]
            or corpus_manifest.get("record_schema_version")
            != "mentor-rl-world-model-supervised-trace-v1"
            or corpus_manifest.get("tool_contract_version")
            != "mentor-rl-world-model-s0-identifier-tools-v2"
        ):
            raise SystemExit("The shared trace corpus contract changed")
        audit = require_object(
            corpus_manifest.get("explanation_audit"),
            "explanation_audit",
        )
        audit_path = Path(
            absolute_path(
                root,
                audit.get("path"),
                "explanation audit path",
            )
        )
        audit_sha256 = require_sha256(
            audit.get("sha256"),
            "explanation audit SHA-256",
        )
        verify_file(audit_path, audit_sha256, "explanation audit")
        audit_payload = read_json(audit_path, "explanation audit")
        if (
            audit_payload.get("model_text_created") is not False
            or audit_payload.get("model_sample_used") is not False
        ):
            raise SystemExit("The explanation audit source flags changed")

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
    if is_v3 and (
        corpus_manifest.get("files", {}).get(train_path.name, {}).get("sha256")
        != train_sha256
        or corpus_manifest.get("files", {})
        .get(validation_path.name, {})
        .get("sha256")
        != validation_sha256
    ):
        raise SystemExit("The v3 corpus file manifest changed")
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
        evaluator.get("schema_version")
        != contract["evaluator_schema_version"]
        or evaluator.get("dataset_id") != contract["dataset_id"]
        or evaluator.get("evaluation_contract")
        != contract["evaluation_contract"]
    ):
        raise SystemExit("The evaluator manifest contract changed")
    if is_v3 and evaluator.get("corpus_manifest_sha256") != corpus_manifest_sha256:
        raise SystemExit("The v3 evaluator corpus identity changed")
    if schema == S0_V4_TRAINING_SCHEMA:
        if (
            evaluator.get("row_schema_version") != "multiplex_sft_v3"
            or evaluator.get("compatible_loss_contracts")
            != [V3_LOSS_CONTRACT, V4_PRIORITY_LOSS_CONTRACT]
            or "loss_contract" in evaluator
        ):
            raise SystemExit("The v4 evaluator loss compatibility changed")
    elif is_v3 and evaluator.get("loss_contract") != V3_LOSS_CONTRACT:
        raise SystemExit("The v3 evaluator loss contract changed")
    validation_panel_name = (
        "teacher_forced_validation"
        if schema == S0_V4_TRAINING_SCHEMA
        else "validation"
    )
    validation_panel = require_object(
        evaluator.get(validation_panel_name), "evaluator validation panel"
    )
    if is_v3 and validation_panel.get("row_count") != validation_rows:
        raise SystemExit("The v3 evaluator validation count changed")
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
    expected_model_id = (
        "gpt-oss-20b-bf16" if is_20b else "gpt-oss-120b-bf16"
    )
    if model_id != expected_model_id:
        raise SystemExit("The selected model differs from the run schema")
    model_path = absolute_path(root, model.get("path"), "model path")
    base_schema = require_string(
        model.get("base_model_artifact_schema"),
        "base_model_artifact_schema",
    )
    if base_schema != "mentor-rl-base-model-artifact-v2":
        raise SystemExit("The base model artifact schema changed")

    # An S1 run may start from the adapter of one completed S0 v4 run.
    initialization = config.get("initialization")
    if initialization is not None and schema not in S1_SCHEMAS:
        raise SystemExit("Only an S1 run can start from a parent checkpoint")
    init_values = (
        validate_parent_checkpoint(
            root,
            initialization,
            model_identity_sha256=require_sha256(
                model.get("model_identity_sha256"), "model_identity_sha256"
            ),
            tokenizer_manifest_sha256=require_sha256(
                method.get("tokenizer_manifest_sha256"),
                "tokenizer_manifest_sha256",
            ),
            lora_rank=lora_rank,
            lora_alpha=require_int(trainer.get("lora_alpha"), "lora_alpha"),
        )
        if initialization is not None
        else {
            "INIT_ADAPTER_DIR": "",
            "INIT_ADAPTER_SHA256": "",
            "INIT_ADAPTER_RUN_ID": "",
        }
    )

    values = {
        "S0_RUN_CONFIG": str(config_path),
        "S0_RUN_CONFIG_SHA256": sha256_file(config_path),
        "S0_CONFIG_ID": config_id,
        "S0_RUN_SCOPE": run_scope,
        "S0_DATASET_ID": contract["dataset_id"],
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
        "SEGMENT_LOSS_TELEMETRY": str(int(segment_loss_telemetry)),
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
        **init_values,
    }
    if "diagnostic_snapshot_steps" in settings:
        values["DIAGNOSTIC_SNAPSHOT_STEPS"] = ",".join(map(str, diagnostic_snapshot_steps))
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
