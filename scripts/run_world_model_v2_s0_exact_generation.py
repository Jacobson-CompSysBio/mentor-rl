#!/usr/bin/env python3
"""Generate deterministic S0 trajectories without answer-key access."""

from __future__ import annotations

import argparse
import gc
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
    renderable_v3_tools,
    tokenizer_artifact_hashes,
    validated_tokenizer_manifest,
)
from scripts.build_world_model_v2_s0_generation_bundle import (  # noqa: E402
    BUNDLE_FULL_SHARED_TRACE_SCHEMA_VERSION,
    BUNDLE_FULL_TOOL_TRAJECTORY_SCHEMA_VERSION,
    BUNDLE_SHARED_TRACE_SCHEMA_VERSION,
    BUNDLE_TOOL_TRAJECTORY_SCHEMA_VERSION,
    BUNDLE_V3_SCHEMA_VERSION,
    FULL_REGISTRY_BUNDLE_V3_SCHEMA_VERSION,
    FULL_REGISTRY_V3_EVALUATION_CONTRACT,
    FULL_SHARED_TRACE_EVALUATION_CONTRACT,
    FULL_TOOL_TRAJECTORY_EVALUATION_CONTRACT,
    QUESTION_KEYS,
    SHARED_TRACE_EVALUATION_CONTRACT,
    SNAPSHOT_VALIDATION_BUNDLE_SCHEMA_VERSION,
    SNAPSHOT_VALIDATION_EVALUATION_CONTRACT,
    TOOL_TRAJECTORY_EVALUATION_CONTRACT,
    TOOL_TRAJECTORY_EVALUATION_CONTRACTS,
    V3_EVALUATION_CONTRACT,
    V3_EVALUATION_CONTRACTS,
)
from scripts.evaluate_world_model_v2_s0 import (  # noqa: E402
    SHARED_TRACE_EVALUATION_CONTRACTS,
    SHARED_TRACE_PREDICTION_SCHEMA_VERSION,
    TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION,
    V3_PREDICTION_SCHEMA_VERSION,
    parse_final_answer,
    parse_shared_final_action,
    parse_tool_trajectory_call,
)
from runtime.tools import ToolExecutionError  # noqa: E402
from runtime.world_model_s0_tools import S0IdentifierToolRuntime  # noqa: E402
from runtime.world_model.contract import load_contract  # noqa: E402
from runtime.world_model.source_index import (  # noqa: E402
    SourceIndexError,
    check_source_index,
    content_sha256,
)
from runtime.world_model.tools import (  # noqa: E402
    ToolCallError,
    WorldModelToolRuntime,
)


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
ALLOWED_SHARED_FINAL_SPECIAL_TOKENS = frozenset(
    {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
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
SNAPSHOT_CHECKPOINT_FORMAT = "mentor-rl-s0-tp-lora-inference-v1"
INFERENCE_DEVICE_MAP = "balanced_low_0"
SHARD_PREDICTION_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-prediction-shard-v1"
)
PARTIAL_PREDICTION_SCHEMA_VERSION = (
    "mentor-rl-world-model-s0-partial-prediction-shard-v1"
)
SHARD_STRATEGY = "strided_record_order_v1"


def _prediction_schema(evaluation_contract: str) -> str:
    """Return the prediction schema for one evaluation contract."""

    if evaluation_contract in V3_EVALUATION_CONTRACTS:
        return V3_PREDICTION_SCHEMA_VERSION
    if evaluation_contract in SHARED_TRACE_EVALUATION_CONTRACTS:
        return SHARED_TRACE_PREDICTION_SCHEMA_VERSION
    return TOOL_TRAJECTORY_PREDICTION_SCHEMA_VERSION


def final_generation_allowed_special_tokens(
    evaluation_contract: str,
) -> frozenset[str]:
    """Allow the channel framing required by shared final generations."""

    if evaluation_contract in SHARED_TRACE_EVALUATION_CONTRACTS:
        return ALLOWED_SHARED_FINAL_SPECIAL_TOKENS
    return frozenset()


class S0GenerationError(RuntimeError):
    """Report one invalid S0 generation input."""


def is_accelerator_oom(error: BaseException, torch_module: Any) -> bool:
    """Return whether an exception is a CUDA/HIP allocator failure."""

    oom_type = getattr(torch_module, "OutOfMemoryError", None)
    if isinstance(oom_type, type) and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return "out of memory" in message and any(
        marker in message for marker in ("cuda", "hip", "accelerator")
    )


def release_generation_memory(torch_module: Any) -> None:
    """Release unreachable tensors and cached accelerator allocations."""

    gc.collect()
    cuda = getattr(torch_module, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def generate_prompt_batch_with_oom_backoff(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    torch_module: Any,
    prompts: list[str],
    record_ids: list[str],
    max_new_tokens: int,
    max_total_tokens: int,
    stage: str,
    shard_index: int,
    retry_singleton: bool = True,
) -> list[dict[str, Any]]:
    """Generate one prompt batch, bisecting it after accelerator OOMs."""

    if not prompts or len(prompts) != len(record_ids):
        raise S0GenerationError("The generation prompt batch is invalid")
    if max_new_tokens < 1 or max_total_tokens <= max_new_tokens:
        raise S0GenerationError("The generation token limits are invalid")

    inputs: dict[str, Any] | None = None
    output: Any = None
    generated: Any = None
    results: list[dict[str, Any]] | None = None
    oom_message: str | None = None
    input_width: int | None = None
    prompt_token_counts: list[int] = []
    generation_limit: int | None = None
    try:
        prompt_batch: str | list[str] = (
            prompts[0] if len(prompts) == 1 else prompts
        )
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=len(prompts) > 1,
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
            raise S0GenerationError(
                f"A {stage} generation prompt exceeds max_total_tokens"
            )
        generation_limit = min(max_new_tokens, min(remaining))
        with torch_module.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=generation_limit,
                do_sample=False,
                eos_token_id=(
                    model.generation_config.eos_token_id
                    if model.generation_config.eos_token_id is not None
                    else tokenizer.eos_token_id
                ),
                pad_token_id=tokenizer.pad_token_id,
            )
        results = []
        for offset in range(len(prompts)):
            generated = output[offset][input_width:]
            results.append(
                {
                    "raw_generation": tokenizer.decode(
                        generated,
                        skip_special_tokens=False,
                    ),
                    "prompt_tokens": prompt_token_counts[offset],
                    "generated_tokens": int(generated.shape[0]),
                }
            )
    except RuntimeError as error:
        if not is_accelerator_oom(error, torch_module):
            raise
        oom_message = str(error)
        # Traceback frames can otherwise retain failed-forward tensors while
        # the smaller retry is starting.
        error.__traceback__ = None
    finally:
        generated = None
        output = None
        inputs = None
        release_generation_memory(torch_module)

    if oom_message is None:
        if results is None:
            raise S0GenerationError("The generation batch produced no results")
        return results

    event = {
        "stage": stage,
        "shard_index": shard_index,
        "record_ids": record_ids,
        "attempted_batch_size": len(prompts),
        "input_width": input_width,
        "prompt_tokens": prompt_token_counts,
        "max_new_tokens": generation_limit,
        "oom": oom_message,
    }
    if len(prompts) == 1:
        if retry_singleton:
            print(
                json.dumps(
                    {"event": "test_generation_oom_retry", **event},
                    sort_keys=True,
                ),
                flush=True,
            )
            return generate_prompt_batch_with_oom_backoff(
                model=model,
                tokenizer=tokenizer,
                device=device,
                torch_module=torch_module,
                prompts=prompts,
                record_ids=record_ids,
                max_new_tokens=(
                    generation_limit
                    if generation_limit is not None
                    else max_new_tokens
                ),
                max_total_tokens=max_total_tokens,
                stage=stage,
                shard_index=shard_index,
                retry_singleton=False,
            )
        print(
            json.dumps(
                {"event": "test_generation_oom_unrecoverable", **event},
                sort_keys=True,
            ),
            flush=True,
        )
        raise S0GenerationError(
            "Accelerator OOM persisted for singleton "
            f"{stage} generation of {record_ids[0]}"
        ) from None

    midpoint = len(prompts) // 2
    print(
        json.dumps(
            {
                "event": "test_generation_oom_backoff",
                **event,
                "retry_batch_sizes": [midpoint, len(prompts) - midpoint],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "torch_module": torch_module,
        # Preserve the parent attempt's scalar token cap. Retrying a smaller
        # microbatch must not change the requested generation contract.
        "max_new_tokens": (
            generation_limit
            if generation_limit is not None
            else max_new_tokens
        ),
        "max_total_tokens": max_total_tokens,
        "stage": stage,
        "shard_index": shard_index,
    }
    return generate_prompt_batch_with_oom_backoff(
        prompts=prompts[:midpoint],
        record_ids=record_ids[:midpoint],
        **common,
    ) + generate_prompt_batch_with_oom_backoff(
        prompts=prompts[midpoint:],
        record_ids=record_ids[midpoint:],
        **common,
    )


def final_prompt_groups(
    prompts: list[str],
    *,
    batch_size: int,
    singleton_character_threshold: int = 8192,
) -> list[list[int]]:
    """Batch ordinary final prompts while isolating unusually large payloads."""

    groups: list[list[int]] = []
    pending: list[int] = []
    for index, prompt in enumerate(prompts):
        if len(prompt) > singleton_character_threshold:
            if pending:
                groups.append(pending)
                pending = []
            groups.append([index])
            continue
        pending.append(index)
        if len(pending) == batch_size:
            groups.append(pending)
            pending = []
    if pending:
        groups.append(pending)
    return groups


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
    elif evaluation_contract == SHARED_TRACE_EVALUATION_CONTRACT:
        expected_schema = BUNDLE_SHARED_TRACE_SCHEMA_VERSION
    elif evaluation_contract == FULL_SHARED_TRACE_EVALUATION_CONTRACT:
        expected_schema = BUNDLE_FULL_SHARED_TRACE_SCHEMA_VERSION
    elif evaluation_contract == V3_EVALUATION_CONTRACT:
        expected_schema = BUNDLE_V3_SCHEMA_VERSION
    elif evaluation_contract == FULL_REGISTRY_V3_EVALUATION_CONTRACT:
        expected_schema = FULL_REGISTRY_BUNDLE_V3_SCHEMA_VERSION
    elif evaluation_contract == SNAPSHOT_VALIDATION_EVALUATION_CONTRACT:
        expected_schema = SNAPSHOT_VALIDATION_BUNDLE_SCHEMA_VERSION
    else:
        raise S0GenerationError(
            "The bundle is not a supported trajectory contract"
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
    if evaluation_contract in V3_EVALUATION_CONTRACTS:
        try:
            source_index = check_source_index(registry_path, load_contract())
        except SourceIndexError as error:
            raise S0GenerationError(
                f"The trajectory source index changed: {error}"
            ) from error
        if content_sha256(source_index) != registry_sha256:
            raise S0GenerationError(
                "The trajectory source index identity changed"
            )
    elif (
        not registry_path.is_file()
        or sha256_file(registry_path) != registry_sha256
    ):
        raise S0GenerationError("The trajectory registry file changed")
    return manifest, rows


def checkpoint_artifact_identity(
    checkpoint_path: Path, *, inference_snapshot: bool = False
) -> dict[str, Any]:
    """Return the exact inference checkpoint identity."""

    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tp_adapter_manifest.json",
    ]
    required.extend(
        ["snapshot_metadata.json", "training_exposure.json"]
        if inference_snapshot
        else ["run_contract/training_exposure.json"]
    )
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
    snapshot_validation: bool = False,
    promotion_snapshot: bool = False,
) -> tuple[dict[str, Any], str]:
    """Verify one promotion checkpoint or atomic inference snapshot."""

    adapter = read_json(checkpoint_path / "tp_adapter_manifest.json")
    identity = adapter.get("identity")
    if snapshot_validation and promotion_snapshot:
        raise S0GenerationError(
            "Snapshot validation and promotion modes are mutually exclusive"
        )
    inference_snapshot = snapshot_validation or promotion_snapshot
    expected_format = (
        SNAPSHOT_CHECKPOINT_FORMAT if inference_snapshot else CHECKPOINT_FORMAT
    )
    if (
        adapter.get("format") != expected_format
        or not isinstance(identity, Mapping)
        or identity.get("method_id") != method_id
        or identity.get("corpus_manifest_sha256")
        != corpus_manifest_sha256
        or identity.get("tokenizer_manifest_sha256")
        != tokenizer_manifest_sha256
        or identity.get("model_identity_sha256") != model_identity_sha256
    ):
        raise S0GenerationError("The checkpoint contract changed")
    if inference_snapshot:
        if (
            adapter.get("inference_only") is not True
            or adapter.get("rank_shards") != 0
        ):
            raise S0GenerationError("The inference snapshot format changed")
        complete_marker = checkpoint_path / "SNAPSHOT_COMPLETE"
        if not complete_marker.is_file():
            raise S0GenerationError(
                "Snapshot validation requires SNAPSHOT_COMPLETE"
            )
        snapshot = read_json(checkpoint_path / "snapshot_metadata.json")
        files = snapshot.get("files")
        required_snapshot_files = {
            "adapter_config.json",
            "adapter_model.safetensors",
            "tp_adapter_manifest.json",
            "training_exposure.json",
        }
        if (
            snapshot.get("schema_version")
            != "mentor-rl-s0-inference-snapshot-v1"
            or snapshot.get("run_id") != identity.get("run_id")
            or snapshot.get("method_id") != method_id
            or not isinstance(snapshot.get("global_step"), int)
            or snapshot.get("global_step", 0) < 1
            or not isinstance(files, Mapping)
            or set(files) != required_snapshot_files
        ):
            raise S0GenerationError("The snapshot metadata changed")
        for name in sorted(required_snapshot_files):
            path = checkpoint_path / name
            expected_hash = files.get(name)
            if (
                not path.is_file()
                or not isinstance(expected_hash, str)
                or sha256_file(path) != expected_hash
            ):
                raise S0GenerationError(
                    f"The completed snapshot file changed: {name}"
                )
        if (
            snapshot.get("snapshot_sha256")
            != files["adapter_model.safetensors"]
            or snapshot.get("adapter_sha256")
            != files["adapter_model.safetensors"]
            or snapshot.get("adapter_config_sha256")
            != files["adapter_config.json"]
            or snapshot.get("manifest_sha256")
            != files["tp_adapter_manifest.json"]
            or snapshot.get("exposure_sha256")
            != files["training_exposure.json"]
        ):
            raise S0GenerationError("The snapshot file identity changed")
        exposure_path = checkpoint_path / "training_exposure.json"
    else:
        exposure_path = (
            checkpoint_path / "run_contract" / "training_exposure.json"
        )
    exposure = read_json(exposure_path)
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
        or (
            not inference_snapshot
            and exposure.get("status") != "complete"
        )
        or (
            inference_snapshot
            and exposure.get("status") != "snapshot"
        )
        or exposure.get("method_id") != method_id
        or not isinstance(claimed, str)
        or stable_sha256(exposure_identity) != claimed
        or not isinstance(logical, Mapping)
        or not isinstance(contract, Mapping)
        or (
            not inference_snapshot
            and contract.get("satisfied") is not True
        )
        or (
            promotion_snapshot
            and (
                contract.get("satisfied") is not True
                or logical.get("all_eligible_train_rows_exposed") is not True
            )
        )
    ):
        raise S0GenerationError(
            "The checkpoint exposure receipt is invalid"
            if inference_snapshot
            else "The test requires one complete checkpoint"
        )
    artifact_identity = checkpoint_artifact_identity(
        checkpoint_path, inference_snapshot=inference_snapshot
    )
    if inference_snapshot:
        artifact_identity["snapshot_step"] = snapshot["global_step"]
        artifact_identity["snapshot_sha256"] = snapshot["snapshot_sha256"]
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


def _plan_manifest_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return generation-manifest fields fixed by one generation plan."""

    keys = (
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
    fields = {key: plan[key] for key in keys}
    for key in ("snapshot_step", "snapshot_sha256"):
        if key in plan:
            fields[key] = plan[key]
    return fields


def _validate_complete_resume(
    resume_dir: Path,
    *,
    plan: Mapping[str, Any],
    rows: list[dict[str, Any]],
    shard_output: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load a complete compatible shard, or report that none exists."""

    manifest_path = resume_dir / "generation_manifest.json"
    predictions_path = resume_dir / "predictions.jsonl"
    if not manifest_path.exists() or not predictions_path.exists():
        return None
    shard_count = int(plan["shard_count"])
    shard_index = int(plan["shard_index"])
    if shard_count > 1 or shard_output:
        manifest, predictions = _validated_shard_manifest(
            resume_dir,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        if manifest.get("full_record_count") != plan["full_record_count"]:
            raise S0GenerationError(
                "The completed resume shard has the wrong full record count"
            )
    else:
        manifest = read_json(manifest_path)
        claimed = manifest.get("manifest_sha256")
        elapsed_seconds = manifest.get("elapsed_seconds")
        identity = {
            str(key): value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        if (
            manifest.get("schema_version")
            != _prediction_schema(str(plan["evaluation_contract"]))
            or not isinstance(claimed, str)
            or stable_sha256(identity) != claimed
            or sha256_file(predictions_path)
            != manifest.get("predictions_sha256")
            or isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0.0
        ):
            raise S0GenerationError(
                "The completed resume output has an invalid identity"
            )
        predictions = read_jsonl(predictions_path)
    if any(
        manifest.get(key) != value
        for key, value in _plan_manifest_fields(plan).items()
    ):
        raise S0GenerationError(
            "The completed resume shard uses different generation inputs"
        )
    expected_ids = [str(row["record_id"]) for row in rows]
    actual_ids = [prediction.get("record_id") for prediction in predictions]
    if (
        manifest.get("record_count") != len(rows)
        or len(predictions) != len(rows)
        or actual_ids != expected_ids
    ):
        raise S0GenerationError(
            "The completed resume shard has the wrong records"
        )
    return manifest, predictions


def _validate_partial_resume(
    resume_dir: Path,
    *,
    plan: Mapping[str, Any],
    rows: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], float] | None:
    """Load one compatible, batch-aligned prediction prefix."""

    manifest_path = resume_dir / "partial_generation_manifest.json"
    predictions_path = resume_dir / "partial_predictions.jsonl"
    if not manifest_path.exists() and not predictions_path.exists():
        return None
    if not manifest_path.is_file() or not predictions_path.is_file():
        return None
    manifest = read_json(manifest_path)
    claimed = manifest.get("manifest_sha256")
    identity = {
        str(key): value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    elapsed_seconds = manifest.get("elapsed_seconds")
    if (
        manifest.get("schema_version")
        != PARTIAL_PREDICTION_SCHEMA_VERSION
        or not isinstance(claimed, str)
        or stable_sha256(identity) != claimed
        or manifest.get("generation_plan_sha256") != stable_sha256(plan)
        or manifest.get("shard_index") != plan["shard_index"]
        or manifest.get("shard_count") != plan["shard_count"]
        or manifest.get("shard_strategy") != plan["shard_strategy"]
        or manifest.get("full_record_count") != plan["full_record_count"]
        or manifest.get("shard_record_count") != len(rows)
        or manifest.get("reads_private_answer_keys") is not False
        or isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or float(elapsed_seconds) < 0.0
    ):
        raise S0GenerationError(
            "The partial resume checkpoint has an invalid identity"
        )
    if sha256_file(predictions_path) != manifest.get("predictions_sha256"):
        raise S0GenerationError(
            "The partial resume predictions changed after generation"
        )
    predictions = read_jsonl(predictions_path)
    record_count = len(predictions)
    actual_ids = [prediction.get("record_id") for prediction in predictions]
    expected_ids = [str(row["record_id"]) for row in rows[:record_count]]
    if (
        record_count < 1
        or record_count > len(rows)
        or manifest.get("record_count") != record_count
        or actual_ids != expected_ids
        or manifest.get("record_ids_sha256")
        != stable_sha256(expected_ids)
        or (record_count != len(rows) and record_count % batch_size != 0)
    ):
        raise S0GenerationError(
            "The partial resume checkpoint is not an ordered batch prefix"
        )
    return predictions, float(elapsed_seconds)


def _load_resume_shard(
    resume_dir: Path,
    *,
    plan: Mapping[str, Any],
    rows: list[dict[str, Any]],
    batch_size: int,
    shard_output: bool,
) -> tuple[
    str,
    dict[str, Any] | None,
    list[dict[str, Any]],
    float,
]:
    """Load a complete shard or a validated partial shard prefix."""

    if not resume_dir.is_dir():
        raise S0GenerationError(
            f"The resume shard directory is absent: {resume_dir}"
        )
    plan_path = resume_dir / "generation_plan.json"
    artifact_names = (
        "generation_manifest.json",
        "predictions.jsonl",
        "partial_generation_manifest.json",
        "partial_predictions.jsonl",
    )
    has_artifacts = any((resume_dir / name).exists() for name in artifact_names)
    if not has_artifacts:
        return "empty", None, [], 0.0
    if not plan_path.is_file():
        raise S0GenerationError(
            "The resume shard lacks its generation plan"
        )
    if read_json(plan_path) != dict(plan):
        raise S0GenerationError(
            "The resume shard generation plan changed"
        )
    complete = _validate_complete_resume(
        resume_dir,
        plan=plan,
        rows=rows,
        shard_output=shard_output,
    )
    if complete is not None:
        manifest, predictions = complete
        return (
            "complete",
            manifest,
            predictions,
            float(manifest["elapsed_seconds"]),
        )
    partial = _validate_partial_resume(
        resume_dir,
        plan=plan,
        rows=rows,
        batch_size=batch_size,
    )
    if partial is None:
        return "empty", None, [], 0.0
    predictions, elapsed_seconds = partial
    return "partial", None, predictions, elapsed_seconds


def _write_partial_checkpoint(
    output_dir: Path,
    *,
    plan: Mapping[str, Any],
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Atomically replace one batch-complete prediction checkpoint."""

    predictions_path = output_dir / "partial_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    record_ids = [str(prediction["record_id"]) for prediction in predictions]
    manifest = {
        "schema_version": PARTIAL_PREDICTION_SCHEMA_VERSION,
        "generation_plan_sha256": stable_sha256(plan),
        "shard_index": plan["shard_index"],
        "shard_count": plan["shard_count"],
        "shard_strategy": plan["shard_strategy"],
        "full_record_count": plan["full_record_count"],
        "shard_record_count": len(rows),
        "record_count": len(predictions),
        "record_ids_sha256": stable_sha256(record_ids),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": elapsed_seconds,
        "reads_private_answer_keys": False,
    }
    manifest["manifest_sha256"] = stable_sha256(manifest)
    write_json(output_dir / "partial_generation_manifest.json", manifest)
    return manifest


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
    if bundle["evaluation_contract"] == SNAPSHOT_VALIDATION_EVALUATION_CONTRACT:
        common_keys += ("snapshot_step", "snapshot_sha256")
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
        "schema_version": _prediction_schema(
            str(first_manifest["evaluation_contract"])
        ),
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
    max_tool_call_tokens: int = 1024,
    max_final_prompt_tokens: int = 4096,
    shard_index: int = 0,
    shard_count: int = 1,
    shard_output: bool = False,
    resume_shard_dir: Path | None = None,
    snapshot_validation: bool = False,
    promotion_snapshot: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate and record one S0 test panel shard."""

    if max_new_tokens < 1 or max_total_tokens <= max_new_tokens:
        raise S0GenerationError("The generation token limits are invalid")
    if max_tool_call_tokens < 1 or max_final_prompt_tokens < 1:
        raise S0GenerationError("The generation safety limits are invalid")
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
        snapshot_validation=snapshot_validation,
        promotion_snapshot=promotion_snapshot,
    )
    evaluation_contract = str(bundle.get("evaluation_contract"))
    if evaluation_contract not in TOOL_TRAJECTORY_EVALUATION_CONTRACTS:
        raise S0GenerationError("The trajectory contract changed")
    registry = bundle["identifier_registry"]
    if evaluation_contract in V3_EVALUATION_CONTRACTS:
        trajectory_runtime = WorldModelToolRuntime.from_source_index(
            (REPO_ROOT / str(registry["path"])).resolve(),
            contract=load_contract(),
        )
    else:
        trajectory_runtime = S0IdentifierToolRuntime.from_registry(
            (REPO_ROOT / str(registry["path"])).resolve(),
            expected_sha256=str(registry["sha256"]),
        )
    generation_config = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "max_total_tokens": max_total_tokens,
        "max_tool_call_tokens": max_tool_call_tokens,
        "max_final_prompt_tokens": max_final_prompt_tokens,
        "enable_thinking": enable_thinking,
        "reasoning_effort": reasoning_effort,
        "seed": seed,
        "batch_size": batch_size,
        "device_map": INFERENCE_DEVICE_MAP,
    }
    generation_config.update(
        {
            "generation_passes": 2,
            "final_enable_thinking": (
                evaluation_contract in SHARED_TRACE_EVALUATION_CONTRACTS
            ),
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
    if snapshot_validation or promotion_snapshot:
        plan["snapshot_step"] = checkpoint["snapshot_step"]
        plan["snapshot_sha256"] = checkpoint["snapshot_sha256"]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if dry_run:
        write_json(output_dir / "generation_plan.json", plan)
        return {"status": "dry_run", "plan": plan}

    resume_status = "empty"
    resume_manifest: dict[str, Any] | None = None
    predictions: list[dict[str, Any]] = []
    resumed_elapsed_seconds = 0.0
    if resume_shard_dir is not None:
        resume_root = resume_shard_dir.resolve()
        (
            resume_status,
            resume_manifest,
            predictions,
            resumed_elapsed_seconds,
        ) = _load_resume_shard(
            resume_root,
            plan=plan,
            rows=rows,
            batch_size=batch_size,
            shard_output=shard_output,
        )
        print(
            json.dumps(
                {
                    "event": "test_generation_resume_loaded",
                    "shard_index": shard_index,
                    "resume_status": resume_status,
                    "completed": len(predictions),
                    "total": len(rows),
                    "resume_shard_dir": str(resume_root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_json(output_dir / "generation_plan.json", plan)

    if resume_status == "complete":
        if resume_manifest is None:
            raise S0GenerationError(
                "The completed resume shard lacks its manifest"
        )
        predictions_path = output_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        if (
            sha256_file(predictions_path)
            != resume_manifest["predictions_sha256"]
        ):
            raise S0GenerationError(
                "The copied resume shard changed during serialization"
            )
        write_json(
            output_dir / "generation_manifest.json",
            resume_manifest,
        )
        print(
            json.dumps(
                {
                    "event": "test_generation_shard_reused",
                    "shard_index": shard_index,
                    "completed": len(predictions),
                    "total": len(rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return {
            "status": "complete",
            "resume_status": "reused_complete",
            "manifest": resume_manifest,
        }
    if resume_status == "partial":
        _write_partial_checkpoint(
            output_dir,
            plan=plan,
            rows=rows,
            predictions=predictions,
            elapsed_seconds=resumed_elapsed_seconds,
        )

    import torch

    torch.manual_seed(seed)
    model = None
    tokenizer = None
    device = None
    if len(predictions) < len(rows):
        model, tokenizer = load_backend(
            base_model_path=base_model_path.resolve(),
            checkpoint_path=checkpoint_path.resolve(),
            tokenizer_path=tokenizer_path.resolve(),
            local_files_only=local_files_only,
        )
        device = next(model.parameters()).device
        if batch_size > 1:
            tokenizer.padding_side = "left"
    initial_completed = len(predictions)
    started = time.monotonic()
    for batch_start in range(initial_completed, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_record_ids = [str(row["record_id"]) for row in batch_rows]
        print(
            json.dumps(
                {
                    "event": "test_generation_batch_started",
                    "shard_index": shard_index,
                    "batch_start": batch_start,
                    "batch_end": batch_start + len(batch_rows),
                    "record_ids": batch_record_ids,
                    "completed": len(predictions),
                    "total": len(rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if model is None or tokenizer is None or device is None:
            raise S0GenerationError(
                "The generation backend was not loaded for pending rows"
            )
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
            template_args["tools"] = (
                renderable_v3_tools(row["tools"])
                if evaluation_contract in V3_EVALUATION_CONTRACTS
                else row["tools"]
            )
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

        tool_generations = generate_prompt_batch_with_oom_backoff(
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_module=torch,
            prompts=prompts,
            record_ids=batch_record_ids,
            max_new_tokens=min(max_new_tokens, max_tool_call_tokens),
            max_total_tokens=max_total_tokens,
            stage="tool_call",
            shard_index=shard_index,
        )
        batch_predictions: list[dict[str, Any]] = []
        continuation_inputs: list[
            tuple[dict[str, Any], int, dict[str, Any], str]
        ] = []
        for offset, row in enumerate(batch_rows):
            generation = tool_generations[offset]
            raw = str(generation["raw_generation"])
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
                "prompt_tokens": generation["prompt_tokens"],
                "generated_tokens": generation["generated_tokens"],
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
                    "assistant_final_explanation": None,
                    "final_generation_error": None,
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
                except (ToolExecutionError, ToolCallError) as error:
                    prediction_row["tool_error"] = str(error)
                else:
                    payload = dict(
                        result.output
                        if evaluation_contract in V3_EVALUATION_CONTRACTS
                        else result.payload
                    )
                    prediction_row["tool_payload"] = payload
                    prediction_row["network_used_false"] = (
                        True
                        if evaluation_contract in V3_EVALUATION_CONTRACTS
                        else result.provenance.get("network_used") is False
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

        del tool_generations

        if continuation_inputs:
            final_prompts: list[str] = []
            bounded_continuation_inputs: list[
                tuple[dict[str, Any], int, dict[str, Any], str]
            ] = []
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
                    (
                        {
                            "role": "tool",
                            "name": parsed["name"],
                            "content": prediction_row["tool_payload"],
                        }
                        if evaluation_contract in V3_EVALUATION_CONTRACTS
                        else {
                            "role": "tool",
                            "content": prediction_row["tool_payload"],
                        }
                    ),
                ]
                final_enable_thinking = (
                    evaluation_contract
                    in SHARED_TRACE_EVALUATION_CONTRACTS
                )
                final_prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=(
                        renderable_v3_tools(row["tools"])
                        if evaluation_contract in V3_EVALUATION_CONTRACTS
                        else row["tools"]
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=final_enable_thinking,
                    reasoning_effort=reasoning_effort,
                )
                expected_suffix = (
                    GPT_OSS_TOOL_PROMPT_SUFFIX
                    if final_enable_thinking
                    else GPT_OSS_FINAL_PROMPT_SUFFIX
                )
                if not final_prompt.endswith(expected_suffix):
                    raise S0GenerationError(
                        "The final prompt lacks its assistant boundary"
                    )
                final_prompt_inputs = tokenizer(
                    final_prompt,
                    return_tensors="pt",
                )
                final_prompt_tokens = int(
                    final_prompt_inputs["attention_mask"]
                    .sum(dim=1)
                    .tolist()[0]
                )
                del final_prompt_inputs
                prediction_row["final_prompt_tokens"] = final_prompt_tokens
                if final_prompt_tokens > max_final_prompt_tokens:
                    diagnostic = "max_final_prompt_tokens_exceeded"
                    prediction_row["final_generation_error"] = diagnostic
                    print(
                        json.dumps(
                            {
                                "event": "test_generation_final_prompt_skipped",
                                "record_id": row["record_id"],
                                "reason": diagnostic,
                                "final_prompt_tokens": final_prompt_tokens,
                                "max_final_prompt_tokens": (
                                    max_final_prompt_tokens
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                final_prompts.append(final_prompt)
                bounded_continuation_inputs.append(
                    (row, prediction_index, parsed, thinking)
                )

            continuation_inputs = bounded_continuation_inputs

            for final_group in final_prompt_groups(
                final_prompts,
                batch_size=batch_size,
            ):
                grouped_prompts = [final_prompts[index] for index in final_group]
                grouped_record_ids = [
                    str(continuation_inputs[index][0]["record_id"])
                    for index in final_group
                ]
                grouped_generations = generate_prompt_batch_with_oom_backoff(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    torch_module=torch,
                    prompts=grouped_prompts,
                    record_ids=grouped_record_ids,
                    max_new_tokens=max_new_tokens,
                    max_total_tokens=max_total_tokens,
                    stage="final_answer",
                    shard_index=shard_index,
                )
                for grouped_offset, final_index in enumerate(final_group):
                    _, prediction_index, _, _ = continuation_inputs[final_index]
                    generation = grouped_generations[grouped_offset]
                    raw_final = str(generation["raw_generation"])
                    encoded_final, final_disallowed = clean_decoded_generation(
                        raw_final,
                        eos_token=tokenizer.eos_token,
                        pad_token=tokenizer.pad_token,
                        allowed_special_tokens=(
                            final_generation_allowed_special_tokens(
                                evaluation_contract
                            )
                        ),
                    )
                    if evaluation_contract in SHARED_TRACE_EVALUATION_CONTRACTS:
                        (
                            final_explanation,
                            assistant_final,
                            _,
                        ) = parse_shared_final_action(encoded_final)
                    else:
                        final_explanation = None
                        assistant_final, _ = parse_final_answer(encoded_final)
                    batch_predictions[prediction_index].update(
                        {
                            "encoded_final_answer": encoded_final,
                            "raw_final_generation": raw_final,
                            "assistant_final": assistant_final,
                            "assistant_final_explanation": final_explanation,
                            "final_prompt_tokens": generation["prompt_tokens"],
                            "final_generated_tokens": generation[
                                "generated_tokens"
                            ],
                            "final_disallowed_special_tokens": final_disallowed,
                        }
                    )
                del grouped_generations

        predictions.extend(batch_predictions)
        completed = len(predictions)
        elapsed_seconds = (
            resumed_elapsed_seconds + time.monotonic() - started
        )
        partial_manifest = _write_partial_checkpoint(
            output_dir,
            plan=plan,
            rows=rows,
            predictions=predictions,
            elapsed_seconds=elapsed_seconds,
        )
        print(
            json.dumps(
                {
                    "event": "test_generation_batch_checkpointed",
                    "shard_index": shard_index,
                    "batch_start": batch_start,
                    "batch_end": batch_start + len(batch_rows),
                    "record_ids": batch_record_ids,
                    "completed": completed,
                    "total": len(rows),
                    "partial_manifest_sha256": partial_manifest[
                        "manifest_sha256"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if (
            completed == initial_completed + len(batch_rows)
            or completed % max(25, batch_size * 10) == 0
            or completed == len(rows)
        ):
            print(
                json.dumps(
                    {
                        "event": "test_generation_progress",
                        "shard_index": shard_index,
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
            else _prediction_schema(evaluation_contract)
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
        "elapsed_seconds": (
            resumed_elapsed_seconds + time.monotonic() - started
        ),
        "reads_private_answer_keys": False,
    }
    if snapshot_validation or promotion_snapshot:
        manifest["snapshot_step"] = checkpoint["snapshot_step"]
        manifest["snapshot_sha256"] = checkpoint["snapshot_sha256"]
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
    parser.add_argument("--max-tool-call-tokens", type=int, default=1024)
    parser.add_argument("--max-final-prompt-tokens", type=int, default=4096)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-output", action="store_true")
    parser.add_argument("--snapshot-validation", action="store_true")
    parser.add_argument("--promotion-snapshot", action="store_true")
    parser.add_argument("--resume-shard-dir", type=Path)
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
        max_tool_call_tokens=args.max_tool_call_tokens,
        max_final_prompt_tokens=args.max_final_prompt_tokens,
        enable_thinking=args.enable_thinking,
        reasoning_effort=args.reasoning_effort,
        seed=args.seed,
        batch_size=args.batch_size,
        local_files_only=not args.allow_remote_files,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        shard_output=args.shard_output,
        resume_shard_dir=args.resume_shard_dir,
        snapshot_validation=args.snapshot_validation,
        promotion_snapshot=args.promotion_snapshot,
        dry_run=args.dry_run,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
