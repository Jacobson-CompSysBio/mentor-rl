"""Keep immutable inference adapters for paired checkpoint diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from runtime.world_model_training import build_training_exposure_manifest
from scripts.world_model_s1_eval_common import read_json, sha256_file, stable_sha256


def capture_validation_snapshot(
    checkpoint_dir: Path,
    output_dir: Path,
    global_step: int,
    exposure_manifest_args: dict[str, Any],
) -> Path:
    """Atomically preserve one completed adapter before rolling checkpoint pruning.

    This intentionally excludes optimizer/RNG shards. Its exposure receipt names
    only the rows consumed by this point; the snapshot cannot pass a final gate.
    The caller invokes this on rank zero after all checkpoint writers finish.
    """
    if not (checkpoint_dir / "CHECKPOINT_COMPLETE").is_file():
        raise ValueError("A diagnostic snapshot requires a complete checkpoint")
    state = read_json(checkpoint_dir / "training_state.json")
    if state.get("global_step") != global_step:
        raise ValueError("Checkpoint and requested snapshot steps disagree")
    args = dict(exposure_manifest_args)
    if args["num_train_epochs"] != 1 or not 0 < global_step < args["total_steps"]:
        raise ValueError("Selected snapshots require an interim single-epoch checkpoint")
    consumed_count = global_step * (
        args["per_device_train_batch_size"]
        * args["gradient_accumulation_steps"] * args["data_parallel_size"]
    )
    if consumed_count >= len(args["consumed_indices"]):
        raise ValueError("An interim snapshot must precede epoch-tail padding")
    args["consumed_indices"] = args["consumed_indices"][:consumed_count]
    args["distributed_padding_indices"] = []
    exposure = build_training_exposure_manifest(
        **args, status="snapshot", completed_global_step=global_step,
    )
    manifest = read_json(checkpoint_dir / "tp_adapter_manifest.json")
    keys = ("run_id", "method_id", "run_config_sha256", "corpus_manifest_sha256", "train_sha256", "tokenizer_arm_manifest_sha256", "tokenizer_manifest_sha256")
    if any(manifest["identity"].get(key) != args[key] for key in keys):
        raise ValueError("Snapshot adapter and exposure identities disagree")
    manifest.update({
        "format": "mentor-rl-s0-tp-lora-inference-v1",
        "inference_only": True, "rank_shards": 0,
    })
    destination = output_dir / "diagnostic_snapshots" / f"step-{global_step:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a diagnostic snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temporary.mkdir(mode=0o700)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(checkpoint_dir / name, temporary / name)
    for name, value in (
        ("tp_adapter_manifest.json", manifest), ("training_exposure.json", exposure),
    ):
        (temporary / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    names = ("adapter_config.json", "adapter_model.safetensors", "tp_adapter_manifest.json", "training_exposure.json")
    files = {name: sha256_file(temporary / name) for name in names}
    metadata = {
        "schema_version": "mentor-rl-s0-inference-snapshot-v1",
        "diagnostic_only": True, "promotion_eligible": False,
        "run_id": args["run_id"], "method_id": args["method_id"],
        "global_step": global_step, "files": files,
        "snapshot_sha256": files["adapter_model.safetensors"],
        "adapter_sha256": files["adapter_model.safetensors"],
        "adapter_config_sha256": files["adapter_config.json"],
        "manifest_sha256": files["tp_adapter_manifest.json"],
        "exposure_sha256": files["training_exposure.json"],
    }
    (temporary / "snapshot_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (temporary / "SNAPSHOT_COMPLETE").write_text(stable_sha256(metadata) + "\n")
    for path in temporary.iterdir():
        path.chmod(0o444)
    temporary.rename(destination)
    destination.chmod(0o555)
    return destination
