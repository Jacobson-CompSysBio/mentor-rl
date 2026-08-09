#!/usr/bin/env python3
"""Run full-parameter GPT-OSS S0 SFT with DeepSpeed ZeRO-3 offload."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from collections import OrderedDict
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world_model_training import (  # noqa: E402
    build_training_exposure_manifest,
    consumed_training_index_plan,
    epoch_training_indices_with_replica_padding,
    flatten_sft_record_for_arrow,
    iter_s0_validation_records,
    load_model_text_codec_for_token_manifest,
    normalize_token_ids,
    s0_exposure_scope,
    tokenizer_artifact_hashes,
)
from runtime.world_model_s0 import S0_FAMILIES  # noqa: E402
from runtime.world_model_token_adapter import (  # noqa: E402
    copy_token_adapter_codec_artifacts,
    load_token_manifest,
)
from utils.utils import build_prompt_completion_example  # noqa: E402


# Block 1 defines the S0 identity, loss, and Trainer contract.
MANIFEST_SCHEMA_VERSION = "mentor-rl-world-model-s0-full-finetune-run-v1"
TOKEN_ROW_INITIALIZATION_SCHEMA_VERSION = (
    "mentor-rl-full-finetune-token-row-initialization-v1"
)
DEFAULT_SAFE_SHARD_BYTES = 5_000_000_000

S0_IDENTITY_ENV = {
    "run_id": "S0_RUN_ID",
    "method_id": "S0_METHOD_ID",
    "run_config_sha256": "S0_RUN_CONFIG_SHA256",
    "corpus_manifest_sha256": "S0_CORPUS_MANIFEST_SHA256",
    "train_sha256": "S0_TRAIN_SHA256",
    "validation_sha256": "S0_VALIDATION_SHA256",
    "validation_answer_key_sha256": "S0_VALIDATION_ANSWER_KEY_SHA256",
    "tokenizer_arm_manifest_sha256": "S0_TOKENIZER_ARM_MANIFEST_SHA256",
    "tokenizer_manifest_sha256": "S0_TOKENIZER_MANIFEST_SHA256",
    "model_identity_sha256": "MODEL_IDENTITY_SHA256",
    "training_code_sha256": "S0_TRAINING_CODE_SHA256",
}
S0_TARGET_AWARE_LOSS_CONTRACT = "s0_target_aware_v2"
S0_TARGET_FIELDS = {
    "human_symbol_to_ensembl": "gene_id",
    "human_ensembl_to_symbol": "gene_symbols",
    "human_ambiguous_symbol": "candidate_gene_ids",
}
S0_COMPLETION_LOSS_WEIGHT = 0.5
S0_MAPPING_LOSS_WEIGHT = 0.5


def required_run_identity() -> dict[str, str]:
    """Return the exact S0 run identity from the launch contract."""

    identity: dict[str, str] = {}
    for key, environment_name in S0_IDENTITY_ENV.items():
        value = os.environ.get(environment_name, "").strip()
        if not value:
            raise RuntimeError(
                f"The launch contract lacks {environment_name}"
            )
        if key.endswith("_sha256") and (
            len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise RuntimeError(
                f"The launch contract has an invalid {environment_name}"
            )
        identity[key] = value
    return identity


def loss_contract_config(loss_contract: str) -> dict[str, Any]:
    """Return the fixed weights for the S0 loss contract."""

    if loss_contract != S0_TARGET_AWARE_LOSS_CONTRACT:
        raise ValueError(f"Unknown loss contract: {loss_contract!r}")
    return {
        "loss_contract": loss_contract,
        "completion_loss_weight": S0_COMPLETION_LOSS_WEIGHT,
        "mapping_target_loss_weight": S0_MAPPING_LOSS_WEIGHT,
        "mapping_target_fields": dict(S0_TARGET_FIELDS),
    }


@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Local GPT-OSS BF16 model path"})
    dataset_path: str = field(metadata={"help": "Audited S0 v4 train JSONL"})
    validation_dataset_path: str = field(
        metadata={"help": "Audited S0 v4 validation JSONL"}
    )
    validation_answer_key_path: str = field(
        metadata={"help": "Private S0 v4 validation answer key"}
    )
    tokenizer_path: str = field(metadata={"help": "Local S0 tokenizer path"})
    token_adapter_manifest: Optional[str] = field(
        default=None,
        metadata={"help": "Manifest for a custom S0 tokenizer"},
    )
    expected_model_parameters: int = field(default=116_829_156_672)
    loss_contract: str = field(default=S0_TARGET_AWARE_LOSS_CONTRACT)
    local_files_only: bool = field(default=True)
    model_init_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Load and validate the distributed ZeRO-3 model, publish the "
                "test receipts, and exit before the train or save pass."
            )
        },
    )


class EpochSeededSampler(Sampler[int]):
    """Use the same seed, shuffle, and replica padding as the LoRA jobs."""

    def __init__(self, data_source, seed: int, replica_count: int):
        self.data_source = data_source
        self.seed = seed
        self.replica_count = replica_count
        self.epoch = 0

    def __iter__(self):
        padded, _ = epoch_training_indices_with_replica_padding(
            len(self.data_source),
            self.seed + self.epoch,
            preserve_order=False,
            replica_count=self.replica_count,
        )
        return iter(padded)

    def __len__(self):
        return (
            math.ceil(len(self.data_source) / self.replica_count)
            * self.replica_count
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class S0FullSFTTrainer(SFTTrainer):
    """Use the S0 row order and target-aware loss with ZeRO-3."""

    def __init__(self, *args, loss_contract: str, **kwargs):
        self.loss_contract = loss_contract
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None
        return EpochSeededSampler(
            train_dataset,
            seed=self.args.seed,
            replica_count=int(self.args.world_size),
        )

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        if "mapping_target_mask" not in self._signature_columns:
            self._signature_columns.append("mapping_target_mask")

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """Compute and report the fixed S0 target-aware loss."""

        del num_items_in_batch
        objective = compute_loss_objective(
            model,
            inputs,
            self.loss_contract,
            return_outputs=True,
        )
        mode = "train" if model.training else "eval"
        for key in ("completion_loss", "mapping_target_loss"):
            value = self.accelerator.gather_for_metrics(
                objective[key].detach()
            ).mean().item()
            self._metrics[mode][key].append(value)
        if mode == "train":
            tokens = torch.tensor(
                objective["input_tokens"],
                device=objective["loss"].device,
            )
            self._total_train_tokens += self.accelerator.gather_for_metrics(
                tokens
            ).sum().item()
            self._metrics[mode]["num_tokens"] = [
                self._total_train_tokens
            ]
        if return_outputs:
            return objective["loss"], objective["outputs"]
        return objective["loss"]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSON Lines file as one atomic change."""

    _write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Block 2 sets the custom token rows in the full model.
def _mean_source_rows(
    weight: torch.Tensor,
    source_ids: list[int],
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Use the token-adapter mean for one full-finetune row."""

    if not source_ids:
        raise ValueError("A custom token row has no base token IDs")
    total = torch.zeros(
        weight.shape[1],
        device=weight.device,
        dtype=torch.float32,
    )
    for start in range(0, len(source_ids), chunk_size):
        indices = torch.tensor(
            source_ids[start : start + chunk_size],
            device=weight.device,
            dtype=torch.long,
        )
        total.add_(weight.index_select(0, indices).float().sum(dim=0))
    return (total / len(source_ids)).to(dtype=weight.dtype)


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def initialize_full_finetune_token_rows(
    model,
    manifest_path: Path,
) -> dict[str, Any]:
    """Set custom full-finetune rows in one ZeRO-3 gathered scope."""

    manifest = load_token_manifest(manifest_path)
    tokens = sorted(manifest["tokens"], key=lambda row: int(row["token_id"]))
    input_parameter = model.get_input_embeddings().weight
    output_parameter = model.get_output_embeddings().weight
    unique_parameters = []
    seen_parameter_ids: set[int] = set()
    for parameter in (input_parameter, output_parameter):
        identity = int(getattr(parameter, "ds_id", id(parameter)))
        if identity not in seen_parameter_ids:
            seen_parameter_ids.add(identity)
            unique_parameters.append(parameter)

    try:
        import deepspeed
    except ImportError as error:
        raise RuntimeError(
            "Custom full-finetune token rows require DeepSpeed"
        ) from error
    distributed_rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )
    input_digest = None
    output_digest = None
    with deepspeed.zero.GatheredParameters(
        unique_parameters,
        modifier_rank=0,
    ):
        if distributed_rank == 0:
            token_ids = [int(row["token_id"]) for row in tokens]
            if (
                min(token_ids) < 0
                or max(token_ids) >= input_parameter.shape[0]
                or max(token_ids) >= output_parameter.shape[0]
            ):
                raise RuntimeError(
                    "A custom token row is outside the model vocabulary"
                )
            with torch.no_grad():
                input_means = [
                    _mean_source_rows(
                        input_parameter,
                        [int(value) for value in row["base_token_ids"]],
                    )
                    for row in tokens
                ]
                output_means = [
                    _mean_source_rows(
                        output_parameter,
                        [int(value) for value in row["base_token_ids"]],
                    )
                    for row in tokens
                ]
                for row, input_mean, output_mean in zip(
                    tokens,
                    input_means,
                    output_means,
                ):
                    token_id = int(row["token_id"])
                    input_parameter[token_id].copy_(input_mean)
                    output_parameter[token_id].copy_(output_mean)
                selected = torch.tensor(
                    token_ids,
                    device=input_parameter.device,
                    dtype=torch.long,
                )
                input_digest = _tensor_sha256(
                    input_parameter.index_select(0, selected)
                )
                output_digest = _tensor_sha256(
                    output_parameter.index_select(0, selected)
                )
    return {
        "schema_version": TOKEN_ROW_INITIALIZATION_SCHEMA_VERSION,
        "applied": True,
        "token_manifest_sha256": manifest["manifest_sha256"],
        "token_start": int(tokens[0]["token_id"]),
        "token_count": len(tokens),
        "source_rule": "float32_chunked_mean_of_base_token_ids",
        "input_rows_sha256": input_digest,
        "output_rows_sha256": output_digest,
        "modifier_rank": 0,
    }


# Block 3 writes the full ZeRO-3 checkpoint with bounded host memory.
def _persistent_state_entries(model) -> list[tuple[str, torch.Tensor, bool]]:
    """Return each persistent parameter and buffer in state-dictionary order."""

    entries: list[tuple[str, torch.Tensor, bool]] = []
    names: set[str] = set()
    parameter_objects: dict[int, str] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for name, parameter in module.named_parameters(recurse=False):
            if parameter is None:
                continue
            full_name = prefix + name
            if full_name in names:
                raise RuntimeError(f"Duplicate model state name: {full_name}")
            object_key = int(getattr(parameter, "ds_id", id(parameter)))
            prior_name = parameter_objects.setdefault(object_key, full_name)
            if prior_name != full_name:
                raise RuntimeError(
                    "The streamed save does not support tied parameters: "
                    f"{prior_name}, {full_name}"
                )
            names.add(full_name)
            entries.append((full_name, parameter, True))
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is None or name in module._non_persistent_buffers_set:
                continue
            full_name = prefix + name
            if full_name in names:
                raise RuntimeError(f"Duplicate model state name: {full_name}")
            names.add(full_name)
            entries.append((full_name, buffer, False))
    if not entries:
        raise RuntimeError("The model has no persistent state to save")
    return entries


def _release_host_allocator() -> None:
    """Return free host pages to the operating system when libc supports it."""

    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def stream_zero3_safetensors(
    *,
    model,
    output_dir: Path,
    expected_total_size: int,
    max_shard_size: int = DEFAULT_SAFE_SHARD_BYTES,
) -> dict:
    """Gather one ZeRO-3 tensor at a time and write bounded HF shards."""

    import deepspeed
    import torch.distributed as dist
    from safetensors.torch import save_file

    if not dist.is_initialized():
        raise RuntimeError("The streamed ZeRO-3 save requires a process group")
    if max_shard_size < 1:
        raise RuntimeError("The streamed ZeRO-3 shard size must be positive")

    rank = dist.get_rank()
    entries = _persistent_state_entries(model)
    job_id = os.environ.get("SLURM_JOB_ID", "no-slurm")
    staging_dir = output_dir / f".streamed-zero3-save-{job_id}"
    if rank == 0:
        if staging_dir.exists():
            raise RuntimeError(
                f"The streamed-save directory exists: {staging_dir}"
            )
        staging_dir.mkdir(parents=True)
    dist.barrier()

    pending: OrderedDict[str, torch.Tensor] = OrderedDict()
    pending_bytes = 0
    total_size = 0
    temporary_shards: list[Path] = []
    temporary_weight_map: dict[str, str] = {}

    def flush_shard() -> None:
        nonlocal pending_bytes
        if rank != 0 or not pending:
            return
        shard_number = len(temporary_shards) + 1
        shard_name = f"model-{shard_number:05d}.safetensors.incomplete"
        shard_path = staging_dir / shard_name
        save_file(pending, shard_path, metadata={"format": "pt"})
        for tensor_name in pending:
            temporary_weight_map[tensor_name] = shard_name
        temporary_shards.append(shard_path)
        pending.clear()
        pending_bytes = 0
        _release_host_allocator()

    for name, tensor, is_parameter in entries:
        logical_numel = int(getattr(tensor, "ds_numel", tensor.numel()))
        logical_bytes = logical_numel * tensor.element_size()
        if rank == 0 and pending and pending_bytes + logical_bytes > max_shard_size:
            flush_shard()
        context = (
            deepspeed.zero.GatheredParameters([tensor], modifier_rank=0)
            if is_parameter
            else None
        )
        if context is None:
            if rank == 0:
                cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
                actual_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
                pending[name] = cpu_tensor
                pending_bytes += actual_bytes
                total_size += actual_bytes
                del cpu_tensor
        else:
            with context:
                if rank == 0:
                    cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
                    actual_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
                    if actual_bytes != logical_bytes:
                        raise RuntimeError(
                            f"Gathered tensor size changed for {name}: "
                            f"{actual_bytes} != {logical_bytes}"
                        )
                    pending[name] = cpu_tensor
                    pending_bytes += actual_bytes
                    total_size += actual_bytes
                    del cpu_tensor
    if rank == 0:
        flush_shard()
        if total_size != expected_total_size:
            raise RuntimeError(
                f"The streamed BF16 state has {total_size} bytes, expected {expected_total_size}"
            )
        shard_count = len(temporary_shards)
        if shard_count < 1:
            raise RuntimeError("The streamed save produced no shards")
        final_weight_map: dict[str, str] = {}
        final_shards: list[str] = []
        renamed_shards: dict[str, str] = {}
        for shard_number, temporary_path in enumerate(temporary_shards, start=1):
            final_name = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            final_path = output_dir / final_name
            temporary_path.replace(final_path)
            final_shards.append(final_name)
            renamed_shards[temporary_path.name] = final_name
        for tensor_name, temporary_name in temporary_weight_map.items():
            final_weight_map[tensor_name] = renamed_shards[temporary_name]
        _write_json(
            output_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_size},
                "weight_map": final_weight_map,
            },
        )
        staging_dir.rmdir()
        result = {
            "format": "huggingface_safetensors_sharded",
            "method": "streamed_zero3_parameter_gather",
            "shard_count": shard_count,
            "max_shard_size_bytes": max_shard_size,
            "total_size": total_size,
            "tensor_count": len(final_weight_map),
        }
    else:
        result = {}
    dist.barrier()
    return result


# Block 4 checks the all-rank model load test.
def validate_model_init_receipts(
    receipts: list[dict],
    *,
    expected_world_size: int,
    expected_nodes: int,
    expected_local_world_size: int,
    expected_model_parameters: int,
) -> dict:
    """Validate the exact all-rank topology for the model-init-only qualification."""

    if len(receipts) != expected_world_size:
        raise RuntimeError(
            f"Model-init qualification produced {len(receipts)}/{expected_world_size} rank receipts"
        )
    ranks = [int(receipt.get("rank", -1)) for receipt in receipts]
    expected_ranks = list(range(expected_world_size))
    if sorted(ranks) != expected_ranks:
        raise RuntimeError(
            "Model-init qualification rank set is incomplete or duplicated: "
            f"observed={sorted(ranks)}, expected={expected_ranks}"
        )

    identity_fields = (*S0_IDENTITY_ENV, "deepspeed_config_sha256")
    identities: dict[str, str] = {}
    job_ids: set[str] = set()
    host_local_ranks: dict[str, set[int]] = {}
    for receipt in receipts:
        rank = int(receipt["rank"])
        if receipt.get("schema_version") != (
            "mentor-rl-s0-full-finetune-model-init-rank-v1"
        ):
            raise RuntimeError(f"Model-init rank {rank} reported the wrong receipt schema")
        if receipt.get("status") != "passed":
            raise RuntimeError(f"Model-init rank {rank} did not report status=passed")
        if receipt.get("scope") != "rank_shell_python_zero3_model_init_only":
            raise RuntimeError(f"Model-init rank {rank} reported the wrong qualification scope")
        if int(receipt.get("world_size", -1)) != expected_world_size:
            raise RuntimeError(f"Model-init rank {rank} reported the wrong world size")
        if int(receipt.get("local_world_size", -1)) != expected_local_world_size:
            raise RuntimeError(f"Model-init rank {rank} reported the wrong local world size")
        if int(receipt.get("logical_total_parameters", -1)) != expected_model_parameters:
            raise RuntimeError(f"Model-init rank {rank} reported the wrong total parameter count")
        if int(receipt.get("logical_trainable_parameters", -1)) != expected_model_parameters:
            raise RuntimeError(f"Model-init rank {rank} did not expose every parameter as trainable")
        if int(receipt.get("optimizer_steps", -1)) != 0:
            raise RuntimeError(f"Model-init rank {rank} reported a nonzero optimizer-step count")
        for field_name in (
            "training_performed",
            "evaluation_performed",
            "model_saved",
            "promotion_eligible",
        ):
            if receipt.get(field_name) is not False:
                raise RuntimeError(f"Model-init rank {rank} reported {field_name}=true")
        if receipt.get("distributed_initialized") is not True:
            raise RuntimeError(f"Model-init rank {rank} was not in an initialized process group")
        if str(receipt.get("distributed_backend", "")) != "nccl":
            raise RuntimeError(f"Model-init rank {rank} did not use the NCCL/RCCL backend")
        model_load = receipt.get("model_load_contract", {})
        required_model_load = {
            "active_zero_stage": 3,
            "all_parameter_objects_partitioned": True,
            "deepspeed_comm_initialized_before_from_pretrained": True,
            "hf_zero3_enabled_before_from_pretrained": True,
            "process_group_backend": "nccl",
            "process_group_initialized_before_from_pretrained": True,
            "zero3_init_flag": True,
        }
        wrong_model_load = {
            key: (model_load.get(key), value)
            for key, value in required_model_load.items()
            if model_load.get(key) != value
        }
        if wrong_model_load:
            raise RuntimeError(
                f"Model-init rank {rank} has an invalid ZeRO-3 load: "
                f"{wrong_model_load}"
            )
        if int(
            model_load.get("zero3_partitioned_parameter_objects", -1)
        ) != int(receipt.get("parameter_objects", -2)):
            raise RuntimeError(
                f"Model-init rank {rank} did not partition each parameter"
            )
        job_id = str(receipt.get("job_id", ""))
        if not job_id:
            raise RuntimeError(f"Model-init rank {rank} is missing its Slurm job ID")
        job_ids.add(job_id)
        host = str(receipt.get("host", ""))
        local_rank = int(receipt.get("local_rank", -1))
        if not host or not 0 <= local_rank < expected_local_world_size:
            raise RuntimeError(f"Model-init rank {rank} has invalid host/local-rank metadata")
        host_local_ranks.setdefault(host, set()).add(local_rank)
        for field_name in identity_fields:
            field_value = str(receipt.get(field_name, ""))
            if not field_value:
                raise RuntimeError(f"Model-init rank {rank} is missing {field_name}")
            prior = identities.setdefault(field_name, field_value)
            if prior != field_value:
                raise RuntimeError(
                    f"Model-init ranks disagree on {field_name}: {prior} != {field_value}"
                )

    if len(job_ids) != 1:
        raise RuntimeError(f"Model-init ranks disagree on Slurm job ID: {sorted(job_ids)}")

    if len(host_local_ranks) != expected_nodes:
        raise RuntimeError(
            f"Model-init qualification used {len(host_local_ranks)}/{expected_nodes} hosts"
        )
    expected_local_ranks = set(range(expected_local_world_size))
    malformed_hosts = {
        host: sorted(local_ranks)
        for host, local_ranks in host_local_ranks.items()
        if local_ranks != expected_local_ranks
    }
    if malformed_hosts:
        raise RuntimeError(f"Model-init host/local-rank topology is incomplete: {malformed_hosts}")
    return {
        "host_count": len(host_local_ranks),
        "hosts": {
            host: sorted(local_ranks) for host, local_ranks in sorted(host_local_ranks.items())
        },
        "identity": identities,
        "job_id": next(iter(job_ids)),
        "rank_count": len(receipts),
        "ranks": expected_ranks,
    }


# Block 5 creates the exact S0 token sequence and loss masks.
def _canonical_json_field_value_spans(
    answer_text: str,
    question_family: str,
) -> list[tuple[int, int]]:
    """Return each string span for the required S0 mapping value."""

    target_field = S0_TARGET_FIELDS.get(question_family)
    if target_field is None:
        raise ValueError(
            "The target-aware loss supports only the three S0 question families"
        )
    try:
        answer = json.loads(answer_text)
    except json.JSONDecodeError as error:
        raise ValueError("The target-aware S0 answer must be JSON") from error
    if not isinstance(answer, dict):
        raise ValueError("The target-aware S0 answer must be one JSON object")
    if target_field not in answer:
        raise ValueError(
            f"The S0 answer lacks its mapping target field: {target_field}"
        )

    # Recreate the exact JSON bytes so each character offset stays stable.
    ensure_ascii = None
    for candidate in (False, True):
        rendered = json.dumps(
            answer,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=candidate,
        )
        if rendered == answer_text:
            ensure_ascii = candidate
            break
    if ensure_ascii is None:
        raise ValueError("The target-aware S0 answer is not canonical compact JSON")

    # Walk the sorted object and exclude keys, quotes, and list syntax.
    cursor = 1
    for index, key in enumerate(sorted(answer)):
        if index:
            cursor += 1
        key_text = json.dumps(key, ensure_ascii=ensure_ascii)
        value_text = json.dumps(
            answer[key],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
        )
        cursor += len(key_text) + 1
        value_start = cursor
        value_end = value_start + len(value_text)
        if key == target_field:
            if isinstance(answer[key], str) and answer[key]:
                return [(value_start + 1, value_end - 1)]
            if isinstance(answer[key], list) and answer[key]:
                spans = []
                item_cursor = value_start + 1
                for item_index, item in enumerate(answer[key]):
                    if not isinstance(item, str) or not item:
                        raise ValueError(
                            "Each S0 mapping target list item must be text"
                        )
                    if item_index:
                        item_cursor += 1
                    item_text = json.dumps(item, ensure_ascii=ensure_ascii)
                    spans.append(
                        (item_cursor + 1, item_cursor + len(item_text) - 1)
                    )
                    item_cursor += len(item_text)
                return spans
            raise ValueError("The S0 mapping target must be text or a text list")
        cursor = value_end
    raise RuntimeError("The S0 mapping target span was not found")


def _character_spans_token_mask(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[int]:
    """Mark each token whose character span overlaps a target string."""

    if not spans or any(start < 0 or end <= start for start, end in spans):
        raise ValueError("One target character span is invalid")
    mask = [
        int(
            any(
                token_end > span_start and token_start < span_end
                for span_start, span_end in spans
            )
        )
        for token_start, token_end in offsets
    ]
    if not any(mask):
        raise ValueError("The tokenizer produced no token for the S0 mapping target")
    return mask


def _tokenize_prompt_completion(
    record: dict[str, Any],
    tokenizer,
    max_length: int,
    s0_codec: Any | None = None,
) -> dict[str, list[int]]:
    """Render once and derive all loss masks from exact character offsets."""

    # Metadata selects the target field. It never enters a model message.
    metadata = record.get("metadata")
    if metadata is None and isinstance(record.get("metadata_json"), str):
        metadata = json.loads(record["metadata_json"])
    converted = build_prompt_completion_example(
        {
            "system": record.get("system"),
            "question": record.get("question"),
            "answer": record.get("answer"),
            "metadata": metadata,
            "context": None,
            "in_context_examples": None,
        }
    )
    answer = converted["answer"]
    messages = converted["prompt"] + converted["completion"]
    # Apply the S0 codec before token offsets. The mask must match model text.
    if s0_codec is not None:
        answer = s0_codec.encode_answer_text(answer)
        messages = s0_codec.encode_messages(
            messages,
            question_family=(
                None if not isinstance(metadata, dict) else metadata.get("question_family")
            ),
        )
    # Locate the exact answer after the chat template adds control tokens.
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False
    )
    answer_start = rendered.rfind(answer)
    if answer_start < 0:
        raise RuntimeError("Rendered chat does not contain the exact assistant answer")
    answer_end = answer_start + len(answer)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = normalize_token_ids(encoded)
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    offsets = [tuple(pair) for pair in offsets]
    if len(full_ids) != len(offsets):
        raise RuntimeError("Tokenizer returned mismatched input IDs and character offsets")
    if len(full_ids) > max_length:
        raise RuntimeError(
            f"Tokenized SFT row has {len(full_ids)} tokens, exceeding "
            f"--max_length={max_length}; rebuild/audit the corpus instead of truncating the answer."
        )
    # The completion mask includes the answer and assistant suffix tokens.
    completion_mask = [
        int((end > answer_start and start < answer_end) or start >= answer_end)
        for start, end in offsets
    ]
    if not full_ids or 0 not in completion_mask or 1 not in completion_mask:
        raise RuntimeError("Tokenized row lacks distinct prompt and assistant completion tokens")
    first_completion = completion_mask.index(1)
    if any(not flag for flag in completion_mask[first_completion:]):
        raise RuntimeError("Assistant completion mask is not one contiguous suffix")
    if not isinstance(metadata, dict):
        raise ValueError("The target-aware S0 row lacks metadata")
    question_family = metadata.get("question_family")
    if not isinstance(question_family, str):
        raise ValueError("The target-aware S0 row lacks a question family")
    relative_spans = _canonical_json_field_value_spans(
        answer,
        question_family,
    )
    # Convert answer-relative spans to offsets in the complete chat text.
    mapping_target_mask = _character_spans_token_mask(
        offsets,
        [
            (answer_start + relative_start, answer_start + relative_end)
            for relative_start, relative_end in relative_spans
        ],
    )
    if any(
        target and not completion
        for target, completion in zip(
            mapping_target_mask,
            completion_mask,
            strict=True,
        )
    ):
        raise RuntimeError("The S0 mapping target mask overlaps the prompt")
    return {
        "input_ids": full_ids,
        "completion_mask": completion_mask,
        "mapping_target_mask": mapping_target_mask,
    }


def _iter_flattened_sft_records(
    path: str,
    source_size: int,
    source_mtime_ns: int,
    answer_key_path: str,
    answer_key_size: int,
    answer_key_mtime_ns: int,
):
    """Yield a stable scalar schema and reset stale Arrow caches."""

    del source_size, source_mtime_ns, answer_key_size, answer_key_mtime_ns
    if answer_key_path:
        try:
            for record in iter_s0_validation_records(
                Path(path),
                Path(answer_key_path),
            ):
                yield flatten_sft_record_for_arrow(
                    record,
                    expected_split="val",
                )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid S0 validation panel: {exc}"
            ) from exc
        return
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                yield flatten_sft_record_for_arrow(record)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid SFT JSONL record at {path}:{line_number}: {exc}") from exc


def _load_flattened_sft_dataset(
    path: str,
    rank: int,
    *,
    answer_key_path: str | None = None,
):
    """Build one homogeneous Arrow cache and share it across ranks."""

    source = Path(path).resolve()
    stat = source.stat()
    answer_key = (
        None if answer_key_path is None else Path(answer_key_path).resolve()
    )
    answer_key_stat = None if answer_key is None else answer_key.stat()
    # Arrow uses one scalar schema for all three S0 answer shapes.
    features = Features(
        {
            "system": Value("string"),
            "question": Value("string"),
            "answer": Value("string"),
            "metadata_json": Value("string"),
            "question_family": Value("string"),
            "record_id": Value("string"),
            "fact_id": Value("string"),
            "prompt_form_id": Value("string"),
            "split": Value("string"),
        }
    )
    # These source attributes force a new cache after a local file change.
    generator_kwargs = {
        "path": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "answer_key_path": "" if answer_key is None else str(answer_key),
        "answer_key_size": (
            0 if answer_key_stat is None else answer_key_stat.st_size
        ),
        "answer_key_mtime_ns": (
            0 if answer_key_stat is None else answer_key_stat.st_mtime_ns
        ),
    }

    # All ranks must use one shared Hugging Face cache directory.
    # Rank 0 parses the source once. The other ranks use the same cache.
    if rank != 0:
        dist.barrier()
    dataset = Dataset.from_generator(
        _iter_flattened_sft_records,
        features=features,
        gen_kwargs=generator_kwargs,
    )
    if rank == 0:
        dist.barrier()
    return dataset


def _tokenization_cache_fingerprint(
    *,
    source_fingerprint: str,
    max_length: int,
    codec_manifest_sha256: str,
    tokenizer_identity_sha256: str,
    loss_contract: str,
) -> str:
    """Return the cache identity for one tokenization contract."""

    return hashlib.sha256(
        json.dumps(
            {
                "source_fingerprint": source_fingerprint,
                "max_length": max_length,
                "codec_manifest_sha256": codec_manifest_sha256,
                "tokenizer_identity_sha256": tokenizer_identity_sha256,
                "loss_contract": loss_contract,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_tokenized_dataset(
    path: str,
    tokenizer,
    rank: int,
    *,
    max_length: int,
    label: str,
    tokenizer_identity_sha256: str,
    loss_contract: str = S0_TARGET_AWARE_LOSS_CONTRACT,
    s0_codec: Any | None = None,
    answer_key_path: str | None = None,
):
    dataset = _load_flattened_sft_dataset(
        path,
        rank,
        answer_key_path=answer_key_path,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"{label} dataset is empty")
    question_families = list(dataset["question_family"])
    identities = {
        "record_ids": list(dataset["record_id"]),
        "fact_ids": list(dataset["fact_id"]),
        "question_families": question_families,
        "prompt_form_ids": list(dataset["prompt_form_id"]),
    }

    def tokenize_record(
        record: dict[str, Any],
        codec_manifest_sha256: str,
        tokenizer_identity_sha256: str,
    ) -> dict[str, list[int]]:
        del codec_manifest_sha256, tokenizer_identity_sha256
        return _tokenize_prompt_completion(
            record,
            tokenizer,
            max_length,
            s0_codec=s0_codec,
        )

    # Rank 0 builds the token cache once. The other ranks use that cache.
    if rank != 0:
        dist.barrier()
    codec_manifest_sha256 = (
        "none" if s0_codec is None else s0_codec.manifest_sha256
    )
    # The fingerprint covers each value that can change token IDs or masks.
    tokenization_fingerprint = _tokenization_cache_fingerprint(
        source_fingerprint=dataset._fingerprint,
        max_length=max_length,
        codec_manifest_sha256=codec_manifest_sha256,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        loss_contract=loss_contract,
    )
    # Remove source columns after token conversion. Keep only IDs and masks.
    tokenized = dataset.map(
        tokenize_record,
        fn_kwargs={
            "codec_manifest_sha256": codec_manifest_sha256,
            "tokenizer_identity_sha256": tokenizer_identity_sha256,
        },
        new_fingerprint=tokenization_fingerprint,
        remove_columns=dataset.column_names,
        desc=f"Tokenize {label} prompt/completion rows" if rank == 0 else None,
    )
    if rank == 0:
        dist.barrier()
    return tokenized, question_families, identities


class S0DataCollator(DataCollatorForLanguageModeling):
    """Add the padded mapping mask to the standard language-model batch."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # The base collator pads IDs, attention masks, and completion labels.
        batch = super().torch_call(examples)
        if any("mapping_target_mask" not in example for example in examples):
            raise ValueError("Each S0 row must have a mapping target mask")
        padded_length = int(batch["input_ids"].shape[1])
        # Pad the mapping mask to the same sequence length.
        mapping_target_mask = torch.zeros(
            (len(examples), padded_length),
            dtype=torch.bool,
        )
        for index, example in enumerate(examples):
            row_mask = example["mapping_target_mask"]
            if len(row_mask) > padded_length:
                raise ValueError("A mapping target mask is longer than its padded batch")
            mapping_target_mask[index, : len(row_mask)] = torch.tensor(
                row_mask,
                dtype=torch.bool,
            )
        batch["mapping_target_mask"] = mapping_target_mask
        return batch


def assert_s0_supervision(
    dataset,
    collator,
    max_examples: int = 8,
) -> dict[str, int]:
    # Inspect a small sample before the model allocates its weights.
    checked_prompt = 0
    checked_completion = 0
    checked_mapping_target = 0
    for index in range(min(max_examples, len(dataset))):
        example = dataset[index]
        mask = example.get("completion_mask")
        if not mask or 0 not in mask or 1 not in mask:
            raise RuntimeError(f"S0 preflight found an invalid completion mask in row {index}")
        batch = collator([example])
        labels = batch["labels"][0].tolist()
        input_ids = batch["input_ids"][0].tolist()
        for position, is_completion in enumerate(mask):
            if is_completion:
                if labels[position] != input_ids[position]:
                    raise RuntimeError(
                        f"Assistant token is masked or altered at row {index}, position {position}"
                    )
                checked_completion += 1
            else:
                if labels[position] != -100:
                    raise RuntimeError(
                        f"Prompt token receives loss at row {index}, position {position}"
                    )
                checked_prompt += 1
        mapping_mask = example.get("mapping_target_mask")
        if not mapping_mask or len(mapping_mask) != len(mask):
            raise RuntimeError(
                f"S0 preflight found an invalid mapping mask in row {index}"
            )
        if any(
            target and not completion
            for target, completion in zip(mapping_mask, mask, strict=True)
        ):
            raise RuntimeError(
                f"S0 preflight found a prompt target in row {index}"
            )
        collated_mapping_mask = batch.get("mapping_target_mask")
        if collated_mapping_mask is None:
            raise RuntimeError("The collator removed the mapping target mask")
        observed = collated_mapping_mask[0, : len(mapping_mask)].tolist()
        if observed != [bool(value) for value in mapping_mask]:
            raise RuntimeError("The collator changed the mapping target mask")
        checked_mapping_target += sum(bool(value) for value in mapping_mask)
    if checked_prompt == 0 or checked_completion == 0:
        raise RuntimeError("S0 preflight did not inspect prompt and completion tokens")
    if checked_mapping_target == 0:
        raise RuntimeError("S0 preflight found no mapping target token")
    return {
        "examples_checked": min(max_examples, len(dataset)),
        "prompt_tokens_masked": checked_prompt,
        "completion_tokens_trainable": checked_completion,
        "mapping_target_tokens": checked_mapping_target,
    }


def compute_loss_objective(
    model,
    batch: dict[str, torch.Tensor],
    loss_contract: str,
    *,
    return_outputs: bool = False,
) -> dict[str, Any]:
    """Compute one forward pass and the fixed S0 loss contract."""

    if loss_contract != S0_TARGET_AWARE_LOSS_CONTRACT:
        raise ValueError(f"Unknown loss contract: {loss_contract!r}")
    mapping_target_mask = batch.get("mapping_target_mask")
    labels = batch["labels"]
    model_batch = {
        key: value
        for key, value in batch.items()
        if key not in {"labels", "mapping_target_mask"}
    }
    outputs = model(**model_batch, use_cache=False)
    logits = outputs.logits.float()

    # Each logit predicts the token at the next position.
    shift_labels = nn.functional.pad(
        labels,
        (0, 1),
        value=-100,
    )[..., 1:].contiguous()
    flat_token_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        shift_labels.view(-1).to(logits.device),
        ignore_index=-100,
        reduction="none",
    )
    token_loss = flat_token_loss.view_as(shift_labels)

    # The completion loss covers the complete assistant response.
    completion_positions = shift_labels != -100
    completion_tokens = int(completion_positions.sum().item())
    if completion_tokens == 0:
        raise RuntimeError("The batch has no completion target token")
    completion_loss = token_loss[completion_positions].mean()

    if mapping_target_mask is None or mapping_target_mask.shape != labels.shape:
        raise RuntimeError(
            "The target-aware batch has no valid mapping target mask"
        )
    # The mapping loss covers only the answer values for the S0 family.
    target_labels = labels.masked_fill(
        ~mapping_target_mask.to(torch.bool),
        -100,
    )
    shift_target_labels = nn.functional.pad(
        target_labels,
        (0, 1),
        value=-100,
    )[..., 1:].contiguous()
    mapping_positions = shift_target_labels != -100
    mapping_target_tokens = int(mapping_positions.sum().item())
    if mapping_target_tokens == 0:
        raise RuntimeError("The batch has no mapping target token")
    mapping_target_loss = token_loss[mapping_positions].mean()
    loss = (
        S0_COMPLETION_LOSS_WEIGHT * completion_loss
        + S0_MAPPING_LOSS_WEIGHT * mapping_target_loss
    )
    result = {
        "loss": loss,
        "completion_loss": completion_loss,
        "mapping_target_loss": mapping_target_loss,
        "completion_tokens": completion_tokens,
        "mapping_target_tokens": mapping_target_tokens,
        "input_tokens": int(
            model_batch.get(
                "attention_mask",
                torch.ones_like(labels),
            ).sum().item()
        ),
    }
    if return_outputs:
        result["outputs"] = outputs
    return result


# Block 6 sets up the ZeRO-3 model and process group.
def _logical_parameter_count(model) -> tuple[int, int, int]:
    total = 0
    trainable = 0
    parameter_objects = 0
    for parameter in model.parameters():
        logical_numel = int(getattr(parameter, "ds_numel", parameter.numel()))
        total += logical_numel
        parameter_objects += 1
        if parameter.requires_grad:
            trainable += logical_numel
    return total, trainable, parameter_objects


def _validate_deepspeed_contract(training_args: SFTConfig) -> dict:
    if not training_args.deepspeed:
        raise RuntimeError(
            "Full fine-tuning requires an explicit DeepSpeed config"
        )
    config_path = Path(str(training_args.deepspeed)).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    zero = config.get("zero_optimization", {})
    required = {
        "stage": 3,
        "offload_param.device": zero.get("offload_param", {}).get("device"),
        "offload_optimizer.device": zero.get("offload_optimizer", {}).get("device"),
        "stage3_gather_16bit_weights_on_model_save": zero.get(
            "stage3_gather_16bit_weights_on_model_save"
        ),
    }
    if required != {
        "stage": 3,
        "offload_param.device": "cpu",
        "offload_optimizer.device": "cpu",
        "stage3_gather_16bit_weights_on_model_save": True,
    }:
        raise RuntimeError(f"Unsafe full-finetune DeepSpeed contract: {required}")
    if config.get("zero3_init_flag") is not True:
        raise RuntimeError(
            "Set DeepSpeed zero3_init_flag before the GPT-OSS-120B load"
        )
    return config


def _prepare_zero3_model_load(
    training_args: SFTConfig,
    *,
    rank: int,
    world_size: int,
) -> dict:
    """Set up the process group and the ZeRO-3 model-load context."""

    import deepspeed
    from transformers.integrations.deepspeed import (
        deepspeed_config,
        is_deepspeed_zero3_enabled,
    )

    if not hasattr(training_args, "hf_deepspeed_config"):
        raise RuntimeError(
            "The Hugging Face DeepSpeed config is absent before model load"
        )
    if not is_deepspeed_zero3_enabled():
        raise RuntimeError("ZeRO-3 is not active before model load")
    active_config = deepspeed_config()
    if (
        not isinstance(active_config, dict)
        or active_config.get("zero_optimization", {}).get("stage") != 3
        or active_config.get("zero3_init_flag") is not True
    ):
        raise RuntimeError(
            "The active model-load config is not the required ZeRO-3 config"
        )

    initialized_here = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        initialized_here = True
    deepspeed.init_distributed(
        dist_backend="nccl",
        auto_mpi_discovery=False,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    if (
        not torch.distributed.is_initialized()
        or not deepspeed.comm.is_initialized()
    ):
        raise RuntimeError(
            "The distributed process group is absent before model load"
        )
    observed_rank = torch.distributed.get_rank()
    observed_world_size = torch.distributed.get_world_size()
    if observed_rank != rank or observed_world_size != world_size:
        raise RuntimeError(
            "The process-group identity differs from the launch identity: "
            f"{observed_rank}/{observed_world_size} != {rank}/{world_size}"
        )
    backend = str(torch.distributed.get_backend())
    if backend != "nccl":
        raise RuntimeError(
            f"The full-finetune process group uses {backend}, not nccl"
        )
    return {
        "active_zero_stage": 3,
        "deepspeed_comm_initialized_before_from_pretrained": True,
        "hf_zero3_enabled_before_from_pretrained": True,
        "process_group_backend": backend,
        "process_group_initialized_before_from_pretrained": True,
        "process_group_initialized_by_this_call": initialized_here,
        "zero3_init_flag": True,
    }


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))


# Block 7 runs the model load test or the full train pass.
def _run_model_init_qualification(
    *,
    script_args: ScriptArguments,
    rank: int,
    local_rank: int,
    world_size: int,
    contract_dir: Path,
    output_dir: Path,
    model_load_contract: dict[str, Any],
    run_identity: dict[str, str],
    run_manifest: dict[str, Any],
    total_parameters: int,
    trainable_parameters: int,
    parameter_objects: int,
) -> int:
    """Write and validate receipts for the bounded ZeRO-3 load test."""

    if not dist.is_initialized():
        raise RuntimeError(
            "The model load test has no distributed process group"
        )
    if dist.get_world_size() != world_size or dist.get_rank() != rank:
        raise RuntimeError(
            "The process group identity differs from the launcher identity"
        )
    deepspeed_config_sha256 = os.environ.get(
        "DS_CONFIG_SHA256_EXPECTED",
        "",
    ).strip()
    if (
        len(deepspeed_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in deepspeed_config_sha256
        )
    ):
        raise RuntimeError(
            "The model load test lacks the DeepSpeed config identity"
        )
    qualification_identity = {
        **run_identity,
        "deepspeed_config_sha256": deepspeed_config_sha256,
    }
    job_id = os.environ.get("SLURM_JOB_ID", "")
    node_count = int(os.environ.get("SLURM_NNODES", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if (
        not job_id
        or node_count != 16
        or local_world_size != 1
        or world_size != 16
    ):
        raise RuntimeError(
            "The model load test requires 16 nodes and one rank per node"
        )

    receipt_dir = contract_dir / "model_init_ranks"
    receipt = {
        "schema_version": "mentor-rl-s0-full-finetune-model-init-rank-v1",
        "status": "passed",
        "scope": "rank_shell_python_zero3_model_init_only",
        "job_id": job_id,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "host": socket.gethostname(),
        "distributed_initialized": True,
        "distributed_backend": str(dist.get_backend()),
        "cuda_device_index": torch.cuda.current_device(),
        "cuda_device_name": torch.cuda.get_device_name(
            torch.cuda.current_device()
        ),
        "logical_total_parameters": total_parameters,
        "logical_trainable_parameters": trainable_parameters,
        "parameter_objects": parameter_objects,
        "model_load_contract": model_load_contract,
        "model_load_seconds": run_manifest["model_load_seconds"],
        "optimizer_steps": 0,
        "training_performed": False,
        "evaluation_performed": False,
        "model_saved": False,
        "promotion_eligible": False,
        **qualification_identity,
    }
    _write_json(receipt_dir / f"rank-{rank:05d}.json", receipt)
    dist.barrier()

    if rank == 0:
        receipt_paths = sorted(receipt_dir.glob("rank-*.json"))
        receipts = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in receipt_paths
        ]
        topology = validate_model_init_receipts(
            receipts,
            expected_world_size=world_size,
            expected_nodes=node_count,
            expected_local_world_size=local_world_size,
            expected_model_parameters=script_args.expected_model_parameters,
        )
        receipt_hashes = {
            path.name: _sha256_file(path) for path in receipt_paths
        }
        receipt_manifest_sha256 = _stable_sha256(receipt_hashes)
        aggregate = {
            "schema_version": (
                "mentor-rl-s0-full-finetune-model-init-qualification-v1"
            ),
            "status": "passed",
            "scope": "rank_shell_python_zero3_model_init_only",
            "job_id": job_id,
            "world_size": world_size,
            "node_count": node_count,
            "local_world_size": local_world_size,
            "logical_total_parameters": total_parameters,
            "logical_trainable_parameters": trainable_parameters,
            "optimizer_steps": 0,
            "training_performed": False,
            "evaluation_performed": False,
            "model_saved": False,
            "promotion_eligible": False,
            "model_load_seconds_max": max(
                float(item["model_load_seconds"]) for item in receipts
            ),
            "zero3_model_load": "passed",
            "rank_receipt_sha256": receipt_hashes,
            "rank_receipt_manifest_sha256": receipt_manifest_sha256,
            "topology": topology,
            "distributed_padding": run_manifest["distributed_padding"],
            **qualification_identity,
        }
        _write_json(
            contract_dir / "model_init_qualification.json",
            aggregate,
        )
        run_manifest.update(
            {
                "status": "model_init_debug_complete",
                "optimizer_steps": 0,
                "training_performed": False,
                "evaluation_performed": False,
                "model_saved": False,
                "promotion_eligible": False,
                "rank_receipt_manifest_sha256": (
                    receipt_manifest_sha256
                ),
            }
        )
        _write_json(
            contract_dir / "full_finetune_manifest.json",
            run_manifest,
        )
        padding = run_manifest["distributed_padding"]
        marker_lines = [
            "status=passed",
            "scope=rank_shell_python_zero3_model_init_only",
            f"job_id={job_id}",
            f"world_size={world_size}",
            f"node_count={node_count}",
            f"logical_total_parameters={total_parameters}",
            f"logical_trainable_parameters={trainable_parameters}",
            "optimizer_steps=0",
            "training_performed=false",
            "evaluation_performed=false",
            "model_saved=false",
            "promotion_eligible=false",
            "zero3_model_load=passed",
            *[
                f"{key}={value}"
                for key, value in qualification_identity.items()
            ],
            f"rank_receipt_manifest_sha256={receipt_manifest_sha256}",
            f"padding_policy={padding['policy']}",
            f"padding_rows_per_epoch={padding['padding_rows_per_epoch']}",
            f"num_train_epochs={run_manifest['num_train_epochs']}",
            f"padding_record_occurrences={padding['padding_record_occurrences']}",
            f"seed={run_manifest['seed']}",
            "expected_training_optimizer_steps="
            f"{run_manifest['expected_optimizer_steps']}",
            "padding_plan_validated=true",
            "logical_exposure_excludes_padding=true",
        ]
        _write_text(
            output_dir / "FULL_FINETUNE_MODEL_INIT_DEBUG_SUCCESS",
            "\n".join(marker_lines) + "\n",
        )
        print(
            json.dumps(
                {
                    "event": "full_finetune_model_init_debug_complete",
                    **aggregate,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    return 0


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    run_identity = required_run_identity()
    if script_args.loss_contract != S0_TARGET_AWARE_LOSS_CONTRACT:
        raise RuntimeError(
            "The full trainer requires the s0_target_aware_v2 loss"
        )

    method_match = re.fullmatch(
        r"oss120b-"
        r"(plain-base-tokenizer|ordinary-domain-bpe|atomic-plus-domain-bpe)-"
        r"full-finetune",
        run_identity["method_id"],
    )
    if method_match is None:
        raise RuntimeError(
            "The S0 method ID must select an OSS-120B full fine-tune run"
        )
    tokenizer_method = {
        "plain-base-tokenizer": "plain_base_tokenizer",
        "ordinary-domain-bpe": "ordinary_domain_bpe",
        "atomic-plus-domain-bpe": "atomic_plus_domain_bpe",
    }[method_match.group(1)]
    custom_tokenizer = tokenizer_method in {
        "ordinary_domain_bpe",
        "atomic_plus_domain_bpe",
    }
    if tokenizer_method not in {
        "plain_base_tokenizer",
        "ordinary_domain_bpe",
        "atomic_plus_domain_bpe",
    }:
        raise RuntimeError(
            f"The S0 method has an unknown tokenizer: {tokenizer_method!r}"
        )
    if bool(script_args.token_adapter_manifest) != custom_tokenizer:
        raise RuntimeError(
            "Only a custom S0 tokenizer requires a token adapter manifest"
        )

    dataset_path = Path(script_args.dataset_path).resolve()
    if _sha256_file(dataset_path) != run_identity["train_sha256"]:
        raise RuntimeError("The S0 train file identity changed")
    validation_path = Path(script_args.validation_dataset_path).resolve()
    if _sha256_file(validation_path) != run_identity["validation_sha256"]:
        raise RuntimeError("The S0 validation file identity changed")
    validation_answer_key_path = Path(
        script_args.validation_answer_key_path
    ).resolve()
    if (
        _sha256_file(validation_answer_key_path)
        != run_identity["validation_answer_key_sha256"]
    ):
        raise RuntimeError("The S0 validation answer key identity changed")
    token_manifest_path = (
        None
        if script_args.token_adapter_manifest is None
        else Path(script_args.token_adapter_manifest).resolve()
    )
    token_manifest = (
        None
        if token_manifest_path is None
        else load_token_manifest(token_manifest_path)
    )
    if token_manifest is not None:
        if token_manifest["method"] != tokenizer_method:
            raise RuntimeError(
                "The token manifest method differs from the S0 method"
            )
        if (
            token_manifest["manifest_sha256"]
            != run_identity["tokenizer_manifest_sha256"]
        ):
            raise RuntimeError("The S0 token manifest identity changed")
    s0_codec = (
        None
        if token_manifest_path is None
        else load_model_text_codec_for_token_manifest(token_manifest_path)
    )

    rank = _rank()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full fine-tuning requires a visible ROCm GPU")
    torch.cuda.set_device(local_rank)
    if (
        os.environ.get("ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES")
        != "GptOssMLP"
    ):
        raise RuntimeError("The ZeRO-3 MoE leaf class must be GptOssMLP")
    ds_config = _validate_deepspeed_contract(training_args)
    model_load_contract = _prepare_zero3_model_load(
        training_args,
        rank=rank,
        world_size=world_size,
    )

    output_dir = Path(training_args.output_dir).resolve()
    contract_dir = output_dir / "run_contract"
    if rank == 0:
        contract_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    if rank == 0:
        distributed_ready_marker = os.environ.get(
            "MENTOR_DISTRIBUTED_READY_MARKER"
        )
        if distributed_ready_marker:
            _write_text(
                Path(distributed_ready_marker),
                f"job_id={os.environ.get('SLURM_JOB_ID', 'unknown')}\n"
                f"world_size={world_size}\n",
            )
    dist.barrier()

    # Use the fixed train and validation data paths.
    training_args.remove_unused_columns = True
    training_args.dataloader_drop_last = False
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False
    }
    training_args.completion_only_loss = True
    training_args.packing = False
    training_args.save_strategy = "no"
    training_args.do_eval = True
    if training_args.eval_strategy.value != "epoch":
        raise RuntimeError("The S0 validation strategy must be epoch")
    if not training_args.eval_on_start:
        raise RuntimeError("S0 must validate before the first train step")
    if training_args.per_device_eval_batch_size != 1:
        raise RuntimeError("S0 requires one validation row per rank")
    dataset_kwargs = dict(training_args.dataset_kwargs or {})
    dataset_kwargs["skip_prepare_dataset"] = True
    training_args.dataset_kwargs = dataset_kwargs

    tokenizer_path = Path(script_args.tokenizer_path).resolve()
    tokenizer_hashes = tokenizer_artifact_hashes(tokenizer_path)
    tokenizer_identity_sha256 = _stable_sha256(tokenizer_hashes)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=script_args.local_files_only,
        trust_remote_code=True,
    )
    if token_manifest is not None:
        expected_length = token_manifest["final_tokenizer_length"]
        if len(tokenizer) != expected_length:
            raise RuntimeError(
                "The tokenizer length differs from the token manifest"
            )
        mismatched_rows = [
            str(row["content"])
            for row in token_manifest["tokens"]
            if tokenizer.convert_tokens_to_ids(str(row["content"]))
            != int(row["token_id"])
        ]
        if mismatched_rows:
            raise RuntimeError(
                "The tokenizer rows differ from the token manifest: "
                + ", ".join(mismatched_rows[:5])
            )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build all masks before the model uses GPU memory.
    train_dataset, _, train_identity = load_tokenized_dataset(
        str(dataset_path),
        tokenizer,
        rank,
        max_length=training_args.max_length,
        label="training",
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        loss_contract=script_args.loss_contract,
        s0_codec=s0_codec,
    )
    validation_dataset, _, validation_identity = load_tokenized_dataset(
        str(validation_path),
        tokenizer,
        rank,
        max_length=training_args.max_length,
        label="validation",
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        loss_contract=script_args.loss_contract,
        s0_codec=s0_codec,
        answer_key_path=str(validation_answer_key_path),
    )
    training_families = {
        str(value) for value in train_identity["question_families"]
    }
    if training_families != set(S0_FAMILIES):
        raise RuntimeError(
            "The S0 dataset must contain all three S0 families only"
        )
    validation_families = {
        str(value) for value in validation_identity["question_families"]
    }
    if validation_families != set(S0_FAMILIES):
        raise RuntimeError(
            "The S0 validation set must contain all three S0 families only"
        )
    if set(validation_identity["prompt_form_ids"]) != {"validation"}:
        raise RuntimeError(
            "The S0 validation set must use the validation prompt form"
        )

    global_batch_size = (
        world_size
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
    )
    epoch_count = float(training_args.num_train_epochs)
    if epoch_count <= 0 or not epoch_count.is_integer():
        raise RuntimeError(
            "The S0 exposure contract requires a positive whole epoch count"
        )
    num_train_epochs = int(epoch_count)
    steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
    expected_steps = steps_per_epoch * num_train_epochs
    if training_args.max_steps > 0:
        if training_args.max_steps != expected_steps:
            raise RuntimeError(
                f"max_steps={training_args.max_steps} differs from "
                f"the required {expected_steps} steps"
            )
    padding_plan = consumed_training_index_plan(
        len(train_dataset),
        total_steps=expected_steps,
        num_train_epochs=num_train_epochs,
        replica_count=world_size,
        per_device_train_batch_size=(
            training_args.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            training_args.gradient_accumulation_steps
        ),
        seed=training_args.seed,
        preserve_order=False,
    )
    exposure_args = {
        "run_id": run_identity["run_id"],
        "method_id": run_identity["method_id"],
        "run_config_sha256": run_identity["run_config_sha256"],
        "corpus_manifest_sha256": run_identity[
            "corpus_manifest_sha256"
        ],
        "train_sha256": run_identity["train_sha256"],
        "tokenizer_arm_manifest_sha256": run_identity[
            "tokenizer_arm_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": run_identity[
            "tokenizer_manifest_sha256"
        ],
        "record_ids": train_identity["record_ids"],
        "fact_ids": train_identity["fact_ids"],
        "question_families": train_identity["question_families"],
        "prompt_form_ids": train_identity["prompt_form_ids"],
        "consumed_indices": list(padding_plan["logical_indices"]),
        "distributed_padding_indices": list(
            padding_plan["distributed_padding_indices"]
        ),
        "seed": training_args.seed,
        "total_steps": expected_steps,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": (
            training_args.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": (
            training_args.gradient_accumulation_steps
        ),
        "data_parallel_size": world_size,
        "preserve_order": False,
        "padding_policy": padding_plan["padding_policy"],
        "exposure_scope": s0_exposure_scope(
            os.environ.get("S0_RUN_SCOPE", "").strip()
        ),
    }
    exposure_path = contract_dir / "training_exposure.json"
    if rank == 0 and not script_args.model_init_only:
        planned_exposure = build_training_exposure_manifest(
            **exposure_args,
            status="planned",
        )
        _write_json(exposure_path, planned_exposure)
        print(
            json.dumps(
                {
                    "event": "training_exposure_planned",
                    "selected_records": len(train_dataset),
                    "exposed_records": planned_exposure[
                        "logical_exposure"
                    ]["record_occurrences"],
                    "manifest_sha256": planned_exposure[
                        "manifest_sha256"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()

    collator = S0DataCollator(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=True,
        pad_to_multiple_of=8,
    )
    supervision = assert_s0_supervision(train_dataset, collator)
    validation_supervision = assert_s0_supervision(
        validation_dataset,
        collator,
    )
    if rank == 0:
        print(
            json.dumps(
                {"event": "s0_supervision_preflight", **supervision},
                sort_keys=True,
            ),
            flush=True,
        )

    # Load the complete model inside the active ZeRO-3 context.
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=script_args.local_files_only,
        trust_remote_code=True,
    )
    token_row_initialization = None
    if token_manifest_path is not None:
        token_row_initialization = initialize_full_finetune_token_rows(
            model,
            token_manifest_path,
        )
    model.config.use_cache = False
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    (
        total_parameters,
        trainable_parameters,
        parameter_objects,
    ) = _logical_parameter_count(model)
    zero3_parameter_objects = sum(
        1 for parameter in model.parameters() if hasattr(parameter, "ds_id")
    )
    if zero3_parameter_objects != parameter_objects:
        raise RuntimeError(
            "ZeRO-3 did not partition all parameters during the model load"
        )
    model_load_contract.update(
        {
            "all_parameter_objects_partitioned": True,
            "zero3_partitioned_parameter_objects": (
                zero3_parameter_objects
            ),
        }
    )
    if total_parameters != script_args.expected_model_parameters:
        raise RuntimeError(
            f"The model has {total_parameters} parameters, but the "
            f"contract requires {script_args.expected_model_parameters}"
        )
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "Full fine-tuning does not expose all parameters as trainable"
        )
    lora_names = [
        name
        for name, _ in model.named_parameters()
        if "lora" in name.casefold()
    ]
    if lora_names:
        raise RuntimeError(
            f"The full model contains LoRA parameters: {lora_names[:8]}"
        )

    distributed_padding = {
        "policy": padding_plan["padding_policy"],
        "data_parallel_size": world_size,
        "logical_rows_per_epoch": padding_plan[
            "logical_rows_per_epoch"
        ],
        "padded_rows_per_epoch": padding_plan[
            "padded_rows_per_epoch"
        ],
        "padding_rows_per_epoch": padding_plan[
            "padding_rows_per_epoch"
        ],
        "padding_source_positions": padding_plan[
            "padding_source_positions"
        ],
        "logical_record_occurrences": len(
            padding_plan["logical_indices"]
        ),
        "padding_record_occurrences": len(
            padding_plan["distributed_padding_indices"]
        ),
        "physical_record_occurrences": padding_plan[
            "physical_record_occurrences"
        ],
        "logical_exposure_excludes_padding": True,
        "seed": training_args.seed,
        "epoch_seed": "seed_plus_epoch",
    }
    tokenizer_contract = {
        "method": tokenizer_method,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_artifact_hashes": tokenizer_hashes,
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "token_manifest_path": (
            None
            if token_manifest_path is None
            else str(token_manifest_path)
        ),
        "token_manifest_file_sha256": (
            None
            if token_manifest_path is None
            else _sha256_file(token_manifest_path)
        ),
        "token_manifest_sha256": run_identity[
            "tokenizer_manifest_sha256"
        ],
        "s0_tokenizer_codec_manifest_sha256": (
            None if s0_codec is None else s0_codec.manifest_sha256
        ),
        "token_row_initialization": token_row_initialization,
    }
    run_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": (
            "model_init_debug_pending_aggregate"
            if script_args.model_init_only
            else "initialized"
        ),
        "stage": "S0",
        "training_mode": "full_finetune",
        "qualification_scope": (
            "rank_shell_python_zero3_model_init_only"
            if script_args.model_init_only
            else "full_parameter_training"
        ),
        **run_identity,
        **loss_contract_config(script_args.loss_contract),
        "model_path": str(Path(script_args.model_path).resolve()),
        "tokenizer_contract": tokenizer_contract,
        "logical_total_parameters": total_parameters,
        "logical_trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "parameter_objects": parameter_objects,
        "world_size": world_size,
        "global_batch_size": global_batch_size,
        "num_train_epochs": num_train_epochs,
        "seed": training_args.seed,
        "steps_per_epoch": steps_per_epoch,
        "expected_optimizer_steps": expected_steps,
        "dataset_rows": len(train_dataset),
        "validation_rows": len(validation_dataset),
        "distributed_padding": distributed_padding,
        "supervision_preflight": supervision,
        "validation_supervision_preflight": validation_supervision,
        "deepspeed": {
            "stage": ds_config["zero_optimization"]["stage"],
            "parameter_offload": ds_config[
                "zero_optimization"
            ]["offload_param"]["device"],
            "optimizer_offload": ds_config[
                "zero_optimization"
            ]["offload_optimizer"]["device"],
            "gather_16bit_on_save": ds_config[
                "zero_optimization"
            ]["stage3_gather_16bit_weights_on_model_save"],
            "moe_leaf_module": os.environ[
                "ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES"
            ],
        },
        "model_load_contract": model_load_contract,
        "model_load_seconds": time.time() - load_started,
        "evaluation_performed": False,
    }
    if rank == 0:
        _write_json(
            contract_dir / "full_finetune_manifest.json",
            run_manifest,
        )
        print(
            json.dumps(
                {"event": "full_finetune_preflight", **run_manifest},
                sort_keys=True,
            ),
            flush=True,
        )

    if script_args.model_init_only:
        return _run_model_init_qualification(
            script_args=script_args,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            contract_dir=contract_dir,
            output_dir=output_dir,
            model_load_contract=model_load_contract,
            run_identity=run_identity,
            run_manifest=run_manifest,
            total_parameters=total_parameters,
            trainable_parameters=trainable_parameters,
            parameter_objects=parameter_objects,
        )

    trainer = S0FullSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=None,
        loss_contract=script_args.loss_contract,
    )
    train_started = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_started
    if int(trainer.state.global_step) != expected_steps:
        raise RuntimeError(
            f"The trainer stopped at step {trainer.state.global_step}, "
            f"but the contract requires {expected_steps}"
        )
    validation_history = []
    for row in trainer.state.log_history:
        if "eval_loss" not in row:
            continue
        required_metrics = (
            "eval_loss",
            "eval_completion_loss",
            "eval_mapping_target_loss",
        )
        missing = [name for name in required_metrics if name not in row]
        if missing:
            raise RuntimeError(
                f"A validation event lacks metrics: {missing}"
            )
        validation_history.append(
            {
                "event": "validation",
                "epoch": row.get("epoch"),
                "global_step": int(row.get("step", 0)),
                "loss_contract": script_args.loss_contract,
                "loss": float(row["eval_loss"]),
                "completion_loss": float(row["eval_completion_loss"]),
                "mapping_target_loss": float(
                    row["eval_mapping_target_loss"]
                ),
                "elapsed_seconds": float(row.get("eval_runtime", 0.0)),
            }
        )
    expected_validation_events = num_train_epochs + 1
    if len(validation_history) != expected_validation_events:
        raise RuntimeError(
            f"The trainer recorded {len(validation_history)} validation events, "
            f"but the contract requires {expected_validation_events}"
        )
    if trainer.is_world_process_zero():
        _write_jsonl(
            output_dir / "validation_metrics.jsonl",
            validation_history,
        )
    trainer.accelerator.wait_for_everyone()
    metrics = dict(train_result.metrics)
    metrics["train_wall_seconds_observed"] = train_elapsed
    metrics["global_step"] = int(trainer.state.global_step)
    metrics["logical_trainable_parameters"] = trainable_parameters
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if trainer.is_world_process_zero():
        print(
            "Write the full model as bounded BF16 shards.",
            flush=True,
        )
    save_started = time.time()
    save_report = stream_zero3_safetensors(
        model=trainer.accelerator.unwrap_model(trainer.model),
        output_dir=output_dir,
        expected_total_size=script_args.expected_model_parameters * 2,
    )
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.model.config.save_pretrained(output_dir)
        generation_config = getattr(
            trainer.model,
            "generation_config",
            None,
        )
        if generation_config is not None:
            generation_config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        if token_manifest_path is not None:
            saved_contract_dir = contract_dir / "tokenizer_contract"
            saved_contract = copy_token_adapter_codec_artifacts(
                token_manifest_path,
                saved_contract_dir,
            )
            run_manifest["tokenizer_contract"].update(
                {
                    "saved_path": str(saved_contract_dir),
                    "saved_artifacts": saved_contract,
                }
            )
        torch.save(training_args, output_dir / "training_args.bin")
    trainer.accelerator.wait_for_everyone()
    save_elapsed = time.time() - save_started

    if trainer.is_world_process_zero():
        completed_exposure = build_training_exposure_manifest(
            **exposure_args,
            status="complete",
            completed_global_step=int(trainer.state.global_step),
        )
        logical_exposure = completed_exposure["logical_exposure"]
        expected_occurrences = (
            len(train_dataset) * num_train_epochs
        )
        if (
            logical_exposure["record_occurrences"]
            != expected_occurrences
            or logical_exposure[
                "all_eligible_train_rows_exposed"
            ] is not True
        ):
            raise RuntimeError(
                "The completed S0 logical exposure is incomplete"
            )
        _write_json(exposure_path, completed_exposure)
        run_manifest.update(
            {
                "status": "complete",
                "completed_optimizer_steps": int(
                    trainer.state.global_step
                ),
                "train_wall_seconds": train_elapsed,
                "consolidated_save_seconds": save_elapsed,
                "consolidated_save": save_report,
                "training_metrics": metrics,
                "validation_metrics": validation_history,
                "evaluation_performed": True,
                "exposure_manifest_sha256": completed_exposure[
                    "manifest_sha256"
                ],
            }
        )
        _write_json(
            contract_dir / "full_finetune_manifest.json",
            run_manifest,
        )
        _write_json(
            output_dir / "training_result.json",
            {
                "status": "complete",
                "global_step": int(trainer.state.global_step),
                "train_wall_seconds": train_elapsed,
                "consolidated_save_seconds": save_elapsed,
                "consolidated_save": save_report,
                "training_metrics": metrics,
                "evaluation_performed": True,
                "validation_metrics": validation_history,
                "question_family_counts": dict(
                    sorted(
                        Counter(
                            train_identity["question_families"]
                        ).items()
                    )
                ),
            },
        )
        print(
            json.dumps(
                {"event": "full_finetune_complete", **run_manifest},
                sort_keys=True,
            ),
            flush=True,
        )
    trainer.accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
