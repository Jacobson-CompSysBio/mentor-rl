#!/usr/bin/env python3
"""Generate deterministic S0 trajectories without answer-key access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import (  # noqa: E402
    tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.build_world_model_v2_s0_generation_bundle import (  # noqa: E402
    BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
    BUNDLE_TOOL_TRAJECTORY_SCHEMA_VERSION,
    FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
    QUESTION_KEYS,
    TOOL_TRAJECTORY_EVALUATION_CONTRACT,
    TOOL_TRAJECTORY_EVALUATION_CONTRACTS,
)
from scripts.evaluate_world_model_v2_s0 import (  # noqa: E402
    TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION,
    parse_final_answer,
    parse_tool_trajectory_call,
)
from runtime.tools import ToolExecutionError  # noqa: E402
from runtime.world_model_s0_tools import S0IdentifierToolRuntime  # noqa: E402


SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+?\|>")
ALLOWED_TERMINAL_SPECIAL_TOKENS = frozenset({"<|return|>", "<|end|>"})
ALLOWED_TOOL_SPECIAL_TOKENS = frozenset(
    {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|call|>",
        "<|end|>",
        "<|return|>",
    }
)
GPT_OSS_FINAL_PROMPT_SUFFIX = (
    "<|start|>assistant<|channel|>final<|message|>"
)
GPT_OSS_TOOL_PROMPT_SUFFIX = "<|start|>assistant"
EXPOSURE_SCHEMA_VERSION = "mentor-rl-s0-training-exposure-v2"
CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-v1"
INFERENCE_DEVICE_MAP = "balanced_low_0"
SHARD_PREDICTION_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-prediction-shard-v1"
)
SHARD_STRATEGY = "strided_record_order_v1"


class S0GenerationError(RuntimeError):
    """Report one invalid S0 generation input."""


def canonical_json(value: Any) -> str:
    """Return stable compact JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def stable_sha256(value: Any) -> str:
    """Return the SHA-256 value for canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 value for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise S0GenerationError(
            f"Could not read one JSON object from {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise S0GenerationError(f"Expected one JSON object in {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise S0GenerationError(
                        f"Expected one JSON object at {path}:{line_number}"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise S0GenerationError(
            f"Could not read JSONL from {path}: {error}"
        ) from error
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o640)


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Write stable JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o640)


def load_bundle(bundle_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one valid answer-free test bundle."""

    manifest = read_json(bundle_root / "manifest.json")
    claimed = manifest.get("bundle_sha256")
    identity = {
        str(key): value
        for key, value in manifest.items()
        if key != "bundle_sha256"
    }
    evaluation_contract = manifest.get("evaluation_contract")
    if evaluation_contract == FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT:
        expected_schema = BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION
    elif evaluation_contract == TOOL_TRAJECTORY_EVALUATION_CONTRACT:
        expected_schema = BUNDLE_TOOL_TRAJECTORY_SCHEMA_VERSION
    else:
        raise S0GenerationError(
            "The bundle is not a v6 trajectory contract"
        )
    if (
        manifest.get("schema_version") != expected_schema
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or manifest.get("reads_private_answer_keys") is not False
    ):
        raise S0GenerationError("The generation bundle identity changed")
    questions_path = bundle_root / "questions.jsonl"
    if sha256_file(questions_path) != manifest.get("questions_sha256"):
        raise S0GenerationError("The bundled questions changed")
    rows = read_jsonl(questions_path)
    record_ids = []
    for row in rows:
        if set(row) != QUESTION_KEYS or "answer" in row:
            raise S0GenerationError("A bundled question has unsafe fields")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise S0GenerationError("A bundled record ID is invalid")
        record_ids.append(record_id)
    if (
        len(rows) != manifest.get("record_count")
        or len(set(record_ids)) != len(record_ids)
        or stable_sha256(record_ids) != manifest.get("record_ids_sha256")
    ):
        raise S0GenerationError("The generation record set changed")
    registry = manifest.get("identifier_registry")
    if not isinstance(registry, Mapping) or set(registry) != {
        "path",
        "id",
        "sha256",
    }:
        raise S0GenerationError(
            "The trajectory bundle lacks its public registry"
        )
    registry_sha256 = registry.get("sha256")
    if (
        not isinstance(registry_sha256, str)
        or registry.get("id") != f"sha256:{registry_sha256}"
    ):
        raise S0GenerationError(
            "The trajectory registry identity changed"
        )
    registry_path = (REPO_ROOT / str(registry.get("path", ""))).resolve()
    try:
        registry_path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise S0GenerationError(
            "The trajectory registry path escapes the repository"
        ) from error
    if (
        not registry_path.is_file()
        or sha256_file(registry_path) != registry_sha256
    ):
        raise S0GenerationError("The trajectory registry file changed")
    return manifest, rows


def checkpoint_artifact_identity(checkpoint_path: Path) -> dict[str, Any]:
    """Return the exact inference checkpoint identity."""

    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
        "run_contract/training_exposure.json",
    ]
    files = {}
    for name in required:
        path = checkpoint_path / name
        if not path.is_file():
            raise S0GenerationError(f"A checkpoint file is absent: {name}")
        files[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    payload = {"files": files}
    payload["checkpoint_identity_sha256"] = stable_sha256(payload)
    return payload


def validate_checkpoint(
    checkpoint_path: Path,
    *,
    method_id: str,
    corpus_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    model_identity_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Verify one complete LoRA checkpoint."""

    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if (
        adapter.get("format") != CHECKPOINT_FORMAT
        or not isinstance(identity, Mapping)
        or identity.get("method_id") != method_id
        or identity.get("corpus_manifest_sha256")
        != corpus_manifest_sha256
        or identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or identity.get("model_identity_sha256") != model_identity_sha256
    ):
        raise S0GenerationError("The checkpoint contract changed")
    exposure = read_json(
        checkpoint_path / "run_contract" / "training_exposure.json"
    )
    claimed = exposure.get("manifest_sha256")
    exposure_identity = {
        str(key): value
        for key, value in exposure.items()
        if key != "manifest_sha256"
    }
    logical = exposure.get("logical_exposure")
    contract = exposure.get("exposure_contract")
    if (
        exposure.get("schema_version") != EXPOSURE_SCHEMA_VERSION
        or exposure.get("status") != "complete"
        or exposure.get("method_id") != method_id
        or not isinstance(claimed, str)
        or stable_sha256(exposure_identity) != claimed
        or not isinstance(logical, Mapping)
        or not isinstance(contract, Mapping)
        or contract.get("satisfied") is not True
    ):
        raise S0GenerationError("The test requires one complete checkpoint")
    artifact_identity = checkpoint_artifact_identity(checkpoint_path)
    return artifact_identity, str(identity.get("run_id"))


def clean_decoded_generation(
    raw: str,
    *,
    eos_token: str | None,
    pad_token: str | None,
    allowed_special_tokens: frozenset[str] = frozenset(),
) -> tuple[str, list[str]]:
    """Remove only valid terminal special tokens."""

    terminal_tokens = set(ALLOWED_TERMINAL_SPECIAL_TOKENS)
    if eos_token:
        terminal_tokens.add(str(eos_token))
    if pad_token:
        terminal_tokens.add(str(pad_token))
    special_tokens = SPECIAL_TOKEN_RE.findall(raw)
    allowed_tokens = terminal_tokens | set(allowed_special_tokens)
    disallowed = sorted(
        {token for token in special_tokens if token not in allowed_tokens}
    )
    prediction = raw.strip()
    ordered = sorted(
        {token for token in terminal_tokens if token},
        key=len,
        reverse=True,
    )
    while True:
        for token in ordered:
            if prediction.endswith(token):
                prediction = prediction[: -len(token)].rstrip()
                break
        else:
            break
    return prediction, disallowed


def load_backend(
    *,
    base_model_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    local_files_only: bool,
):
    """Load one consolidated LoRA checkpoint for inference."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=local_files_only,
        trust_remote_code=True,
        device_map=INFERENCE_DEVICE_MAP,
    )
    model = PeftModel.from_pretrained(
        model,
        str(checkpoint_path),
        local_files_only=local_files_only,
        is_trainable=False,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def select_shard_rows(
    rows: list[dict[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Select one deterministic strided row shard."""

    if shard_count < 1:
        raise S0GenerationError("The shard count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise S0GenerationError("The shard index is outside the shard count")
    return rows[shard_index::shard_count]


def _validated_shard_manifest(
    shard_dir: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one complete prediction shard."""

    manifest_path = shard_dir / "generation_manifest.json"
    predictions_path = shard_dir / "predictions.jsonl"
    manifest = read_json(manifest_path)
    claimed = manifest.get("manifest_sha256")
    identity = {
        str(key): value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    if (
        manifest.get("schema_version")
        != SHARD_PREDICTION_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or manifest.get("shard_index") != shard_index
        or manifest.get("shard_count") != shard_count
        or manifest.get("shard_strategy") != SHARD_STRATEGY
        or manifest.get("reads_private_answer_keys") is not False
    ):
        raise S0GenerationError(
            f"Prediction shard {shard_index} has an invalid identity"
        )
    generation_config = manifest.get("generation_config")
    elapsed_seconds = manifest.get("elapsed_seconds")
    if (
        not isinstance(generation_config, dict)
        or stable_sha256(generation_config)
        != manifest.get("generation_config_sha256")
        or isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0.0
    ):
        raise S0GenerationError(
            f"Prediction shard {shard_index} has invalid metadata"
        )
    if sha256_file(predictions_path) != manifest.get("predictions_sha256"):
        raise S0GenerationError(
            f"Prediction shard {shard_index} changed after generation"
        )
    predictions = read_jsonl(predictions_path)
    if len(predictions) != manifest.get("record_count"):
        raise S0GenerationError(
            f"Prediction shard {shard_index} has an invalid row count"
        )
    return manifest, predictions


def merge_prediction_shards(
    *,
    bundle_root: Path,
    shards_root: Path,
    output_dir: Path,
    expected_shard_count: int,
) -> dict[str, Any]:
    """Merge complete shards in the original bundle order."""

    if expected_shard_count < 1:
        raise S0GenerationError("The expected shard count must be positive")
    bundle, rows = load_bundle(bundle_root.resolve())
    shards_root = shards_root.resolve()
    expected_dirs = {
        f"shard-{index:05d}" for index in range(expected_shard_count)
    }
    try:
        actual_dirs = {
            path.name for path in shards_root.iterdir() if path.is_dir()
        }
    except OSError as error:
        raise S0GenerationError(
            f"Could not read the shard directory: {error}"
        ) from error
    if actual_dirs != expected_dirs:
        missing = sorted(expected_dirs - actual_dirs)
        unexpected = sorted(actual_dirs - expected_dirs)
        raise S0GenerationError(
            "The prediction shard set changed: "
            f"missing={missing} unexpected={unexpected}"
        )

    common_keys = (
        "evaluation_contract",
        "test_panel_id",
        "method_id",
        "train_run_id",
        "base_model_identity_sha256",
        "checkpoint_identity_sha256",
        "tokenizer_manifest_sha256",
        "generation_bundle_sha256",
        "generation_config",
        "generation_config_sha256",
        "reads_private_answer_keys",
    )
    bundle_common = {
        "evaluation_contract": bundle["evaluation_contract"],
        "test_panel_id": bundle["test_panel_id"],
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "reads_private_answer_keys": False,
    }
    expected_common: dict[str, Any] | None = None
    merged_by_id: dict[str, dict[str, Any]] = {}
    shard_receipts = []
    first_manifest: dict[str, Any] | None = None
    for shard_index in range(expected_shard_count):
        shard_dir = shards_root / f"shard-{shard_index:05d}"
        manifest, predictions = _validated_shard_manifest(
            shard_dir,
            shard_index=shard_index,
            shard_count=expected_shard_count,
        )
        if any(
            manifest.get(key) != value
            for key, value in bundle_common.items()
        ) or manifest.get("full_record_count") != len(rows):
            raise S0GenerationError(
                f"Prediction shard {shard_index} does not match the bundle"
            )
        if any(key not in manifest for key in common_keys):
            raise S0GenerationError(
                f"Prediction shard {shard_index} lacks common metadata"
            )
        if first_manifest is None:
            first_manifest = manifest
            expected_common = {
                key: manifest.get(key) for key in common_keys
            }
        if expected_common is None or any(
            manifest.get(key) != expected_common.get(key)
            for key in common_keys
        ):
            raise S0GenerationError(
                f"Prediction shard {shard_index} uses different inputs"
            )
        expected_ids = [
            str(row["record_id"])
            for row in rows[shard_index::expected_shard_count]
        ]
        actual_ids = [prediction.get("record_id") for prediction in predictions]
        if actual_ids != expected_ids:
            raise S0GenerationError(
                f"Prediction shard {shard_index} has the wrong records"
            )
        for prediction in predictions:
            record_id = str(prediction["record_id"])
            if record_id in merged_by_id:
                raise S0GenerationError(
                    f"A prediction record occurs more than once: {record_id}"
                )
            merged_by_id[record_id] = prediction
        shard_receipts.append(
            {
                "shard_index": shard_index,
                "record_count": len(predictions),
                "predictions_sha256": manifest["predictions_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "elapsed_seconds": manifest["elapsed_seconds"],
            }
        )

    if first_manifest is None:
        raise S0GenerationError("No prediction shards exist")
    ordered_ids = [str(row["record_id"]) for row in rows]
    if set(merged_by_id) != set(ordered_ids):
        raise S0GenerationError("The merged prediction record set changed")
    predictions = [merged_by_id[record_id] for record_id in ordered_ids]
    output_dir = output_dir.resolve()
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    elapsed_values = [float(item["elapsed_seconds"]) for item in shard_receipts]
    manifest = {
        "schema_version": TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION,
        **{key: first_manifest[key] for key in common_keys},
        "record_count": len(predictions),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": max(elapsed_values),
        "parallelism": {
            "shard_strategy": SHARD_STRATEGY,
            "shard_count": expected_shard_count,
            "completed_shards": len(shard_receipts),
            "maximum_shard_seconds": max(elapsed_values),
            "total_shard_seconds": sum(elapsed_values),
            "shards": shard_receipts,
        },
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest)
    plan = {
        "schema_version": "mentor-rl-world-model-s0-parallel-generation-plan-v1",
        **{key: first_manifest[key] for key in common_keys},
        "record_count": len(rows),
        "parallelism": {
            "shard_strategy": SHARD_STRATEGY,
            "shard_count": expected_shard_count,
        },
    }
    write_json(output_dir / "generation_plan.json", plan)
    return {"status": "complete", "manifest": manifest}


def generate_predictions(
    *,
    bundle_root: Path,
    base_model_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    tokenizer_manifest_path: Path,
    output_dir: Path,
    method_id: str,
    corpus_manifest_sha256: str,
    tokenizer_manifest_sha256: str,
    model_identity_sha256: str,
    max_new_tokens: int,
    max_total_tokens: int,
    enable_thinking: bool,
    reasoning_effort: str,
    seed: int,
    batch_size: int,
    local_files_only: bool,
    shard_index: int = 0,
    shard_count: int = 1,
    shard_output: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate and record one S0 test panel shard."""

    if max_new_tokens < 1 or max_total_tokens <= max_new_tokens:
        raise S0GenerationError("The generation token limits are invalid")
    if batch_size < 1:
        raise S0GenerationError("The generation batch size is invalid")
    bundle, all_rows = load_bundle(bundle_root.resolve())
    rows = select_shard_rows(
        all_rows,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    tokenizer_manifest = validated_tokenizer_manifest(
        tokenizer_manifest_path.resolve()
    )
    if tokenizer_manifest.get("manifest_sha256") != tokenizer_manifest_sha256:
        raise S0GenerationError("The tokenizer manifest identity changed")
    tokenizer_files = tokenizer_artifact_hashes(tokenizer_path.resolve())
    checkpoint, train_run_id = validate_checkpoint(
        checkpoint_path.resolve(),
        method_id=method_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        tokenizer_manifest_sha256=tokenizer_manifest_sha256,
        model_identity_sha256=model_identity_sha256,
    )
    evaluation_contract = str(bundle.get("evaluation_contract"))
    if evaluation_contract not in TOOL_TRAJECTORY_EVALUATION_CONTRACTS:
        raise S0GenerationError("The trajectory contract changed")
    registry = bundle["identifier_registry"]
    trajectory_runtime = S0IdentifierToolRuntime.from_registry(
        (REPO_ROOT / str(registry["path"])).resolve(),
        expected_sha256=str(registry["sha256"]),
    )
    generation_config = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "max_total_tokens": max_total_tokens,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "seed": seed,
        "batch_size": batch_size,
        "device_map": INFERENCE_DEVICE_MAP,
    }
    generation_config.update(
        {
            "generation_passes": 2,
            "final_enable_thinking": False,
        }
    )
    plan = {
        "schema_version": "mentor-rl-world-model-s0-generation-plan-v6",
        "evaluation_contract": bundle["evaluation_contract"],
        "test_panel_id": bundle["test_panel_id"],
        "record_count": len(rows),
        "full_record_count": len(all_rows),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "shard_strategy": SHARD_STRATEGY,
        "method_id": method_id,
        "train_run_id": train_run_id,
        "base_model_identity_sha256": model_identity_sha256,
        "checkpoint_identity_sha256": checkpoint[
            "checkpoint_identity_sha256"
        ],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "tokenizer_artifact_hashes": tokenizer_files,
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "reads_private_answer_keys": False,
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output_dir / "generation_plan.json", plan)
    if dry_run:
        return {"status": "dry_run", "plan": plan}

    import torch

    torch.manual_seed(seed)
    model, tokenizer = load_backend(
        base_model_path=base_model_path.resolve(),
        checkpoint_path=checkpoint_path.resolve(),
        tokenizer_path=tokenizer_path.resolve(),
        local_files_only=local_files_only,
    )
    device = next(model.parameters()).device
    if batch_size > 1:
        tokenizer.padding_side = "left"
    predictions: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompts = []
        for row in batch_rows:
            messages = [
                {"role": "system", "content": str(row["system"])},
                {"role": "user", "content": str(row["question"])},
            ]
            template_args = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
            }
            template_args["tools"] = row["tools"]
            prompt = tokenizer.apply_chat_template(
                messages,
                **template_args,
            )
            if (
                not enable_thinking
                or reasoning_effort != "low"
                or not prompt.endswith(GPT_OSS_TOOL_PROMPT_SUFFIX)
            ):
                raise S0GenerationError(
                    "The tool prompt lacks its assistant call boundary"
                )
            prompts.append(prompt)

        if len(prompts) == 1:
            inputs = tokenizer(prompts[0], return_tensors="pt")
        else:
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_width = int(inputs["input_ids"].shape[1])
        prompt_token_counts = [
            int(value)
            for value in inputs["attention_mask"].sum(dim=1).tolist()
        ]
        remaining = [
            max_total_tokens - prompt_tokens
            for prompt_tokens in prompt_token_counts
        ]
        if min(remaining) < 1:
            raise S0GenerationError("A test prompt exceeds max_total_tokens")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, min(remaining)),
                do_sample=False,
                eos_token_id=(
                    model.generation_config.eos_token_id
                    if model.generation_config.eos_token_id is not None
                    else tokenizer.eos_token_id
                ),
                pad_token_id=tokenizer.pad_token_id,
            )
        batch_predictions: list[dict[str, Any]] = []
        continuation_inputs: list[
            tuple[dict[str, Any], int, dict[str, Any], str]
        ] = []
        for offset, row in enumerate(batch_rows):
            generated = output[offset][input_width:]
            raw = tokenizer.decode(generated, skip_special_tokens=False)
            prediction, disallowed = clean_decoded_generation(
                raw,
                eos_token=tokenizer.eos_token,
                pad_token=tokenizer.pad_token,
                allowed_special_tokens=ALLOWED_TOOL_SPECIAL_TOKENS,
            )
            prediction_row = {
                "record_id": row["record_id"],
                "encoded_prediction": prediction,
                "raw_generation": raw,
                "prompt_tokens": prompt_token_counts[offset],
                "generated_tokens": int(generated.shape[0]),
                "disallowed_special_tokens": disallowed,
            }
            parsed, thinking, parse_error, direct_answer = (
                parse_tool_trajectory_call(prediction)
            )
            prediction_row.update(
                {
                    "assistant_thinking": thinking,
                    "trajectory_parse_error": parse_error,
                    "tool_payload": None,
                    "tool_error": None,
                    "network_used_false": True,
                    "encoded_final_answer": None,
                    "raw_final_generation": None,
                    "assistant_final": None,
                    "final_prompt_tokens": None,
                    "final_generated_tokens": None,
                    "final_disallowed_special_tokens": [],
                }
            )
            if (
                parsed is not None
                and parse_error is None
                and not direct_answer
                and not disallowed
            ):
                try:
                    result = trajectory_runtime.execute(
                        parsed["name"], parsed["arguments"]
                    )
                except ToolExecutionError as error:
                    prediction_row["tool_error"] = str(error)
                else:
                    payload = dict(result.payload)
                    prediction_row["tool_payload"] = payload
                    prediction_row["network_used_false"] = (
                        result.provenance.get("network_used") is False
                    )
                    continuation_inputs.append(
                        (
                            row,
                            len(batch_predictions),
                            parsed,
                            str(thinking),
                        )
                    )
            batch_predictions.append(prediction_row)

        if continuation_inputs:
            final_prompts = []
            for row, prediction_index, parsed, thinking in continuation_inputs:
                prediction_row = batch_predictions[prediction_index]
                messages = [
                    {"role": "system", "content": str(row["system"])},
                    {"role": "user", "content": str(row["question"])},
                    {
                        "role": "assistant",
                        "thinking": thinking,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": parsed["name"],
                                    "arguments": parsed["arguments"],
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": prediction_row["tool_payload"],
                    },
                ]
                final_prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=row["tools"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    reasoning_effort=reasoning_effort,
                )
                if not final_prompt.endswith(GPT_OSS_FINAL_PROMPT_SUFFIX):
                    raise S0GenerationError(
                        "The final prompt lacks its assistant boundary"
                    )
                final_prompts.append(final_prompt)

            if len(final_prompts) == 1:
                final_inputs = tokenizer(
                    final_prompts[0], return_tensors="pt"
                )
            else:
                final_inputs = tokenizer(
                    final_prompts,
                    return_tensors="pt",
                    padding=True,
                )
            final_inputs = {
                key: value.to(device) for key, value in final_inputs.items()
            }
            final_input_width = int(final_inputs["input_ids"].shape[1])
            final_prompt_token_counts = [
                int(value)
                for value in final_inputs["attention_mask"].sum(dim=1).tolist()
            ]
            final_remaining = [
                max_total_tokens - prompt_tokens
                for prompt_tokens in final_prompt_token_counts
            ]
            if min(final_remaining) < 1:
                raise S0GenerationError(
                    "A final trajectory prompt exceeds max_total_tokens"
                )
            with torch.no_grad():
                final_output = model.generate(
                    **final_inputs,
                    max_new_tokens=min(
                        max_new_tokens, min(final_remaining)
                    ),
                    do_sample=False,
                    eos_token_id=(
                        model.generation_config.eos_token_id
                        if model.generation_config.eos_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    pad_token_id=tokenizer.pad_token_id,
                )
            for final_offset, (_, prediction_index, _, _) in enumerate(
                continuation_inputs
            ):
                final_generated = final_output[final_offset][
                    final_input_width:
                ]
                raw_final = tokenizer.decode(
                    final_generated, skip_special_tokens=False
                )
                encoded_final, final_disallowed = clean_decoded_generation(
                    raw_final,
                    eos_token=tokenizer.eos_token,
                    pad_token=tokenizer.pad_token,
                )
                assistant_final, _ = parse_final_answer(encoded_final)
                batch_predictions[prediction_index].update(
                    {
                        "encoded_final_answer": encoded_final,
                        "raw_final_generation": raw_final,
                        "assistant_final": assistant_final,
                        "final_prompt_tokens": (
                            final_prompt_token_counts[final_offset]
                        ),
                        "final_generated_tokens": int(
                            final_generated.shape[0]
                        ),
                        "final_disallowed_special_tokens": final_disallowed,
                    }
                )

        predictions.extend(batch_predictions)
        completed = len(predictions)
        if (
            completed == len(batch_rows)
            or completed % max(25, batch_size * 10) == 0
            or completed == len(rows)
        ):
            print(
                json.dumps(
                    {
                        "event": "test_generation_progress",
                        "completed": completed,
                        "total": len(rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    manifest = {
        "schema_version": (
            SHARD_PREDICTION_SCHEMA_VERSION
            if shard_count > 1 or shard_output
            else TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION
        ),
        "evaluation_contract": bundle["evaluation_contract"],
        "test_panel_id": bundle["test_panel_id"],
        "record_count": len(predictions),
        "method_id": method_id,
        "train_run_id": train_run_id,
        "base_model_identity_sha256": model_identity_sha256,
        "checkpoint_identity_sha256": checkpoint[
            "checkpoint_identity_sha256"
        ],
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256,
        "generation_bundle_sha256": bundle["bundle_sha256"],
        "generation_config": generation_config,
        "generation_config_sha256": stable_sha256(generation_config),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": time.monotonic() - started,
        "reads_private_answer_keys": False,
    }
    if shard_count > 1 or shard_output:
        manifest.update(
            {
                "full_record_count": len(all_rows),
                "shard_index": shard_index,
                "shard_count": shard_count,
                "shard_strategy": SHARD_STRATEGY,
            }
        )
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "generation_manifest.json", manifest)
    return {"status": "complete", "manifest": manifest}


def parse_args() -> argparse.Namespace:
    """Parse command-line values."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-bundle", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--tokenizer-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--corpus-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-manifest-sha256", required=True)
    parser.add_argument("--model-identity-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-total-tokens", type=int, required=True)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-output", action="store_true")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Generate one exact S0 test result."""

    args = parse_args()
    result = generate_predictions(
        bundle_root=args.generation_bundle,
        base_model_path=args.base_model_path,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        tokenizer_manifest_path=args.tokenizer_manifest,
        output_dir=args.output_dir,
        method_id=args.method_id,
        corpus_manifest_sha256=args.corpus_manifest_sha256,
        tokenizer_manifest_sha256=args.tokenizer_manifest_sha256,
        model_identity_sha256=args.model_identity_sha256,
        max_new_tokens=args.max_new_tokens,
        max_total_tokens=args.max_total_tokens,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        batch_size=args.batch_size,
        local_files_only=not args.allow_remote_files,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        shard_output=args.shard_output,
        dry_run=args.dry_run,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
