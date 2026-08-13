#!/usr/bin/env python3
"""Run native GPT-OSS data-parallel x tensor/expert-parallel S0 LoRA SFT.

This entry point intentionally does not use Trainer/Accelerate or stock PEFT.
Accelerate 1.10.1 re-wraps the local expert shards from the GPT-OSS native plan
as replicated DTensors, while PEFT 0.15.2 creates unsharded LoRA branches after
TP.  Both are incorrect for this model.  Instead, this script keeps the native
Transformers TP/EP plan, adds TP-aware attention-only LoRA branches, and runs a
small explicit optimization loop. Each node holds one eight-GPU TP/EP replica.
The script averages adapter gradients across data-parallel replicas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import accelerate
import numpy as np
import peft
import torch
import torch.distributed as dist
import transformers
import trl
from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from world_model_dp_tp_ep import (  # noqa: E402
    ATTENTION_LORA_TARGETS,
    WorldModelTopology,
    validate_gpt_oss_dimensions,
    validate_native_tp_plan,
    validate_tested_stack,
)
from utils.utils import build_prompt_completion_example  # noqa: E402
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
    assert_token_adapter_forward_contract,
    install_trainable_token_rows,
    save_token_adapter,
    token_adapter_parameters,
)
from tp_ep_autograd import (  # noqa: E402
    copy_to_expert_parallel_region,
    reduce_from_expert_parallel_region,
)


# Block 1 defines the S0 run identity, loss, and metric contract.
ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
TP_MANIFEST_NAME = "tp_adapter_manifest.json"
TRAINING_STATE_NAME = "training_state.json"
_WANDB_RUN = None

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
LOSS_CONTRACTS = (S0_TARGET_AWARE_LOSS_CONTRACT,)
S0_TARGET_FIELDS = {
    "human_symbol_to_ensembl": "gene_id",
    "human_ensembl_to_symbol": "gene_symbols",
    "human_ambiguous_symbol": "candidate_gene_ids",
}
S0_TARGET_AWARE_COMPLETION_WEIGHT = 0.5
S0_TARGET_AWARE_MAPPING_WEIGHT = 0.5


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
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(
                f"The launch contract has an invalid {environment_name}"
            )
        identity[key] = value
    return identity


def loss_contract_config(loss_contract: str) -> dict[str, Any]:
    """Return the fixed weights for the S0 loss contract."""

    if loss_contract == S0_TARGET_AWARE_LOSS_CONTRACT:
        return {
            "loss_contract": loss_contract,
            "completion_loss_weight": S0_TARGET_AWARE_COMPLETION_WEIGHT,
            "mapping_target_loss_weight": S0_TARGET_AWARE_MAPPING_WEIGHT,
            "mapping_target_fields": dict(S0_TARGET_FIELDS),
        }
    raise ValueError(f"Unknown loss contract: {loss_contract!r}")


def initialize_wandb(
    args: argparse.Namespace,
    topology: WorldModelTopology,
    *,
    output_dir: Path,
    s0_codec: Any | None,
):
    """Initialize the one authoritative rank-0 W&B run or fail closed."""

    run_scope = os.environ.get("S0_RUN_SCOPE", "").strip()
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    if mode != "online":
        raise RuntimeError(
            f"The {run_scope or 'unknown'} scope requires "
            f"WANDB_MODE=online, got {mode!r}"
        )
    load_dotenv(REPO_ROOT / ".env", override=False)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B tracking was requested but the wandb package is unavailable") from exc
    wandb_dir = Path(os.environ.get("WANDB_DIR", output_dir / "wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    tags = [value.strip() for value in os.environ.get("WANDB_TAGS", "").split(",") if value.strip()]
    run_identity = required_run_identity()
    init_kwargs: dict[str, Any] = {
        "project": os.environ.get("WANDB_PROJECT", "mentor-sft"),
        "group": os.environ.get("WANDB_RUN_GROUP"),
        "name": os.environ.get("WANDB_NAME"),
        "id": os.environ.get("WANDB_RUN_ID"),
        "resume": os.environ.get("WANDB_RESUME", "allow"),
        "mode": mode,
        "tags": tags,
        "dir": str(wandb_dir),
        "config": {
            "model_name": os.environ.get("MODEL_NAME", Path(args.model_path).name),
            **run_identity,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "run_scope": run_scope,
            "wandb_mode": mode,
            "world_size": topology.world_size,
            "tp_size": topology.tp_size,
            "ep_size": topology.ep_size,
            "global_batch_size": topology.global_batch_size,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
            **loss_contract_config(args.loss_contract),
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "eval_strategy": args.eval_strategy,
            "eval_on_start": args.eval_on_start,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "tokenizer_path": args.tokenizer_path,
            "token_adapter_manifest": args.token_adapter_manifest,
            "s0_tokenizer_codec_manifest_sha256": (
                None if s0_codec is None else s0_codec.manifest_sha256
            ),
            "seed": args.seed,
            "preserve_dataset_order": args.preserve_dataset_order,
            "autograd_multithreading": args.autograd_multithreading,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "wandb_version": wandb.__version__,
        },
    }
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity
    run = wandb.init(**init_kwargs)
    if run is None or getattr(run, "disabled", False):
        raise RuntimeError(f"wandb.init did not create an active {mode} run")
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("eval/*", step_metric="global_step")
    run.summary["tracking_contract"] = "rank0_online_fail_closed"
    return run


# Block 2 defines the topology state and the native TP/EP boundaries.
@dataclass(frozen=True)
class LoraSpec:
    r: int
    alpha: int
    dropout: float

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


@dataclass
class LoopState:
    global_step: int = 0
    epoch: int = 0
    batches_consumed_in_epoch: int = 0


class TPAwareLoraLinear(nn.Module):
    """A frozen native-TP linear plus a separately sharded LoRA branch."""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # Keep the native TP projection as the frozen base path.
        self.base_layer = base_layer
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False, dtype=dtype, device=device)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False, dtype=dtype, device=device)
        self.scaling = float(alpha / r)
        # A zero B matrix makes the first LoRA output equal the base output.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.base_layer(hidden_states) + self.lora_B(self.lora_A(hidden_states)) * self.scaling


def install_autograd_expert_gather() -> None:
    """Install autograd-safe GPT-OSS expert and router EP boundaries."""

    from transformers.integrations.tensor_parallel import (
        ALL_PARALLEL_STYLES,
        GatherParallel,
        RouterParallel,
    )

    # This gather keeps expert communication inside the autograd graph.
    class AutogradExpertGatherParallel(GatherParallel):
        @staticmethod
        def _prepare_input_fn(input_layouts, desired_input_layouts, module, inputs, device_mesh):
            group = device_mesh.get_group()
            module.expert_parallel_group = group
            tensor = inputs[0]
            if isinstance(tensor, DTensor):
                tensor = tensor.to_local()
            return (copy_to_expert_parallel_region(tensor, group), *inputs[1:])

        @staticmethod
        def _prepare_output_fn(output_layouts, use_local_output, module, outputs, device_mesh):
            group = device_mesh.get_group()
            if isinstance(outputs, torch.Tensor):
                return reduce_from_expert_parallel_region(outputs, group)
            if not isinstance(outputs, tuple) or not outputs or not isinstance(outputs[0], torch.Tensor):
                raise TypeError(
                    "Autograd expert gather requires a Tensor or Tensor-first tuple; "
                    f"got {type(outputs).__name__}"
                )
            return (reduce_from_expert_parallel_region(outputs[0], group), *outputs[1:])

        def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
            result = super().prepare_module_tp(module, device_mesh)
            module._mentor_autograd_expert_gather = True
            return result

    # This router restores derivatives from every expert rank.
    class AutogradRouterParallel(RouterParallel):
        @staticmethod
        def _prepare_input_fn(
            input_layouts, desired_input_layouts, module, inputs, device_mesh
        ):
            group = device_mesh.get_group()
            module.expert_parallel_group = group
            tensor = inputs[0]
            if isinstance(tensor, DTensor):
                raise NotImplementedError(
                    "Autograd router parallelism expects the replicated hidden "
                    "state as a rank-local Tensor"
                )
            # RouterParallel divides scores by expert rank.
            # This boundary rebuilds the complete route derivative.
            return copy_to_expert_parallel_region(tensor, group)

        def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module:
            result = super().prepare_module_tp(module, device_mesh)
            module._mentor_autograd_ep_router = True
            return result

    ALL_PARALLEL_STYLES["gather"] = AutogradExpertGatherParallel()
    ALL_PARALLEL_STYLES["ep_router"] = AutogradRouterParallel()


def assert_expert_gradient_parity(device_mesh, rank: int) -> None:
    """Require full hidden gradients from both rank-local EP branches."""

    import warnings

    group = device_mesh.get_group()
    # Compare the custom copy/reduce pair with its exact derivative sum.
    probe = torch.ones(1, device=torch.cuda.current_device(), requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expert_input = copy_to_expert_parallel_region(probe, group)
        router_input = copy_to_expert_parallel_region(probe, group)
        local_expert_output = expert_input * float(rank + 1)
        local_router_output = router_input * float(2 * (rank + 1))
        gathered_output = reduce_from_expert_parallel_region(
            local_expert_output + local_router_output, group
        )
        gathered_output.sum().backward()
    expected = 3 * device_mesh.size() * (device_mesh.size() + 1) / 2
    actual = None if probe.grad is None else float(probe.grad.item())
    if actual != expected:
        raise RuntimeError(
            "Expert/router gradient parity gate failed: expected the sum of "
            f"both rank-local derivative branches {expected}, got {actual}."
        )
    c10d_warnings = [
        str(item.message)
        for item in caught
        if "autograd kernel" in str(item.message).lower() or "c10d" in str(item.message).lower()
    ]
    if c10d_warnings:
        raise RuntimeError(f"Expert-parallel parity emitted unsafe c10d autograd warnings: {c10d_warnings}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    # Slurm may render task counts as ``8(x4)``.
    digits = "".join(character for character in raw if character.isdigit())
    if not digits:
        raise ValueError(f"Environment variable {name} is not an integer: {raw!r}")
    if "(" in raw:
        digits = raw.split("(", 1)[0]
    return int(digits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # These paths select the exact model, tokenizer, corpus, and output.
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--token_adapter_manifest")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--validation_dataset_path", required=True)
    parser.add_argument("--validation_answer_key_path", required=True)
    parser.add_argument("--output_dir", required=True)
    # These values define the parallel layout and batch size.
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--ep_size", type=int, default=8)
    parser.add_argument(
        "--preserve_dataset_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Consume the deterministic family-cycling subset order instead of reshuffling it.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # These values define the optimizer and loss contract.
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--loss_contract",
        choices=LOSS_CONTRACTS,
        default=S0_TARGET_AWARE_LOSS_CONTRACT,
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # These values control metric reports and checkpoint files.
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_strategy", choices=("epoch",), default="epoch")
    parser.add_argument(
        "--eval_on_start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save_strategy", choices=("no", "steps"), default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=900913)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument(
        "--autograd-multithreading",
        dest="autograd_multithreading",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--strict_tested_stack", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    # Reject an invalid run before the model uses GPU memory.
    positive = (
        "tp_size",
        "ep_size",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "per_device_eval_batch_size",
        "save_steps",
        "save_total_limit",
        "max_length",
        "lora_r",
        "lora_alpha",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max_steps must be -1 or positive")
    if args.lora_dropout != 0.0:
        raise ValueError(
            "TP-aware LoRA dropout is deliberately fixed at zero until identical-mask "
            "semantics across the TP group are validated."
        )
    if not args.bf16:
        raise ValueError("S0 GPT-OSS LoRA requires --bf16")
    if args.token_adapter_manifest and not Path(args.token_adapter_manifest).is_file():
        raise ValueError("--token_adapter_manifest does not exist")
    if not Path(args.validation_dataset_path).is_file():
        raise ValueError("--validation_dataset_path does not exist")
    if not Path(args.validation_answer_key_path).is_file():
        raise ValueError("--validation_answer_key_path does not exist")


def configure_topology(
    args: argparse.Namespace,
) -> tuple[WorldModelTopology, int, int, int]:
    # One node holds one complete TP/EP replica.
    rank = _env_int("RANK", _env_int("SLURM_PROCID", 0))
    world_size = _env_int("WORLD_SIZE", _env_int("SLURM_NTASKS", 1))
    local_rank = _env_int("LOCAL_RANK", _env_int("SLURM_LOCALID", 0))
    local_world_size = _env_int("LOCAL_WORLD_SIZE", _env_int("SLURM_NTASKS_PER_NODE", world_size))
    node_count = _env_int("SLURM_NNODES", _env_int("SLURM_JOB_NUM_NODES", 1))
    topology = WorldModelTopology(
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        per_replica_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    ).validate()
    if not torch.cuda.is_available():
        raise RuntimeError("S0 LoRA requires ROCm GPUs; torch.cuda.is_available() is false")
    visible_device_count = torch.cuda.device_count()
    if visible_device_count == 1:
        device_index = 0
    elif visible_device_count == local_world_size:
        device_index = local_rank
    else:
        raise RuntimeError(
            "Each S0 LoRA rank must see one GPU or all local GPUs; "
            f"visible={visible_device_count}, local_world_size={local_world_size}"
        )
    torch.cuda.set_device(device_index)
    return topology, rank, local_rank, device_index


def initialize_distributed(
    rank: int,
    world_size: int,
    device_index: int,
) -> None:
    # NCCL connects all model replicas and tensor shards.
    if not dist.is_initialized():
        timeout_seconds = _env_int("MENTOR_DIST_INIT_TIMEOUT_SECONDS", 600)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
            device_id=torch.device("cuda", device_index),
        )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("Initialized process group does not match the validated topology")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Block 3 creates the exact S0 token sequence and loss masks.
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
    """Build one homogeneous Arrow cache, then share it across local TP ranks."""

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

    # Rank 0 builds the token cache once. The other TP ranks use that cache.
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
        desc=f"Tokenizing {label} prompt/completion rows" if rank == 0 else None,
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


# Block 4 installs LoRA with the required tensor placements.
def _replicate_linear(module: nn.Linear, device_mesh) -> None:
    from transformers.integrations.tensor_parallel import ReplicateParallel

    for name, parameter in list(module.named_parameters(recurse=False)):
        replicated = distribute_tensor(parameter, device_mesh, [Replicate()], src_data_rank=0)
        module.register_parameter(name, nn.Parameter(replicated, requires_grad=True))
    ReplicateParallel().prepare_module_tp(module, device_mesh)


def _require_close(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
        difference = float((actual - expected).abs().max().item())
        raise RuntimeError(f"TP LoRA parity failed for {label}; max_abs_diff={difference}")


def assert_tp_lora_gradient_parity(device_mesh) -> None:
    """Compare both adapter layouts with an unsharded numerical reference."""

    device = torch.device("cuda", torch.cuda.current_device())
    rank = device_mesh.get_local_rank()
    width = 2 * device_mesh.size()
    lora_rank = 4
    scaling = 2.0

    # Test the q/k/v layout against one full reference layer.
    # The A matrix is replicated, and the B matrix uses column shards.
    torch.manual_seed(1103)
    q_base = nn.Linear(width, width, bias=False, device=device, dtype=torch.float32)
    q_base_weight = q_base.weight.detach().clone()
    parallelize_module(q_base, device_mesh, ColwiseParallel(use_local_output=True))
    q_wrapper = TPAwareLoraLinear(q_base, lora_rank, int(scaling * lora_rank), torch.float32, device)
    nn.init.normal_(q_wrapper.lora_B.weight, std=0.05)
    q_a = q_wrapper.lora_A.weight.detach().clone()
    q_b = q_wrapper.lora_B.weight.detach().clone()
    _replicate_linear(q_wrapper.lora_A, device_mesh)
    parallelize_module(
        q_wrapper,
        device_mesh,
        {"lora_B": ColwiseParallel(output_layouts=Shard(-1), use_local_output=True)},
    )
    q_wrapper.base_layer.weight.requires_grad_(False)
    q_input = torch.randn(3, width, device=device, requires_grad=True)
    q_output = q_wrapper(q_input)
    q_output.square().sum().backward()

    q_reference_input = q_input.detach().clone().requires_grad_(True)
    q_reference_a = q_a.detach().clone().requires_grad_(True)
    q_reference_b = q_b.detach().clone().requires_grad_(True)
    q_reference = torch.nn.functional.linear(q_reference_input, q_base_weight)
    q_reference = q_reference + torch.nn.functional.linear(
        torch.nn.functional.linear(q_reference_input, q_reference_a), q_reference_b
    ) * scaling
    q_reference.square().sum().backward()
    output_chunk = q_reference.chunk(device_mesh.size(), dim=-1)[rank]
    b_gradient_chunk = q_reference_b.grad.chunk(device_mesh.size(), dim=0)[rank]
    _require_close("qkv forward", q_output, output_chunk)
    _require_close("qkv replicated A gradient", q_wrapper.lora_A.weight.grad.to_local(), q_reference_a.grad)
    _require_close("qkv colwise B gradient", q_wrapper.lora_B.weight.grad.to_local(), b_gradient_chunk)

    # Test the output layout against one full reference layer.
    # The A matrix uses row shards, and the B matrix is replicated.
    torch.manual_seed(2207)
    o_base = nn.Linear(width, width, bias=False, device=device, dtype=torch.float32)
    o_base_weight = o_base.weight.detach().clone()
    parallelize_module(o_base, device_mesh, RowwiseParallel(use_local_output=True))
    o_wrapper = TPAwareLoraLinear(o_base, lora_rank, int(scaling * lora_rank), torch.float32, device)
    nn.init.normal_(o_wrapper.lora_B.weight, std=0.05)
    o_a = o_wrapper.lora_A.weight.detach().clone()
    o_b = o_wrapper.lora_B.weight.detach().clone()
    parallelize_module(
        o_wrapper,
        device_mesh,
        {"lora_A": RowwiseParallel(input_layouts=Shard(-1), use_local_output=True)},
    )
    _replicate_linear(o_wrapper.lora_B, device_mesh)
    o_wrapper.base_layer.weight.requires_grad_(False)
    full_o_input = torch.randn(3, width, device=device)
    local_o_input = full_o_input.chunk(device_mesh.size(), dim=-1)[rank].detach().clone().requires_grad_(True)
    o_output = o_wrapper(local_o_input)
    o_output.square().sum().backward()

    o_reference_a = o_a.detach().clone().requires_grad_(True)
    o_reference_b = o_b.detach().clone().requires_grad_(True)
    o_reference = torch.nn.functional.linear(full_o_input, o_base_weight)
    o_reference = o_reference + torch.nn.functional.linear(
        torch.nn.functional.linear(full_o_input, o_reference_a), o_reference_b
    ) * scaling
    o_reference.square().sum().backward()
    a_gradient_chunk = o_reference_a.grad.chunk(device_mesh.size(), dim=1)[rank]
    _require_close("o forward", o_output, o_reference)
    _require_close("o rowwise A gradient", o_wrapper.lora_A.weight.grad.to_local(), a_gradient_chunk)
    _require_close("o replicated B gradient", o_wrapper.lora_B.weight.grad.to_local(), o_reference_b.grad)

    dist.barrier(group=device_mesh.get_group())
    del q_wrapper, q_reference, o_wrapper, o_reference
    torch.cuda.empty_cache()


def inject_tp_lora(model, spec: LoraSpec, seed: int) -> list[str]:
    if spec.dropout != 0.0:
        raise ValueError("TP LoRA injection requires dropout=0")
    device_mesh = getattr(model, "_device_mesh", None)
    if device_mesh is None or device_mesh.size() != getattr(model, "_tp_size", None):
        raise RuntimeError("Model has no compatible native TP device mesh")
    # Freeze every base parameter before any LoRA module enters the model.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Select every attention projection and no expert parameter.
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in ATTENTION_LORA_TARGETS
    ]
    expected = model.config.num_hidden_layers * len(ATTENTION_LORA_TARGETS)
    if len(candidates) != expected:
        raise RuntimeError(f"Expected {expected} attention LoRA targets, found {len(candidates)}")

    torch.manual_seed(seed)
    injected = []
    for name, base_layer in candidates:
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA target {name} is {type(base_layer).__name__}, not nn.Linear")
        if not isinstance(base_layer.weight, DTensor):
            raise RuntimeError(f"LoRA target {name} was not sharded by the native TP plan")
        target = name.rsplit(".", 1)[-1]
        wrapper = TPAwareLoraLinear(
            base_layer,
            r=spec.r,
            alpha=spec.alpha,
            dtype=base_layer.weight.dtype,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        if target in {"q_proj", "k_proj", "v_proj"}:
            # q/k/v input is replicated and output is colwise-sharded.
            _replicate_linear(wrapper.lora_A, device_mesh)
            parallelize_module(
                wrapper,
                device_mesh,
                {"lora_B": ColwiseParallel(output_layouts=Shard(-1), use_local_output=True)},
            )
        elif target == "o_proj":
            # o_proj input is rowwise-sharded and output is replicated.
            parallelize_module(
                wrapper,
                device_mesh,
                {"lora_A": RowwiseParallel(input_layouts=Shard(-1), use_local_output=True)},
            )
            _replicate_linear(wrapper.lora_B, device_mesh)
        # Replace the native layer but keep its frozen base projection.
        parent_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attribute, wrapper)
        injected.append(name)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable) != expected * 2:
        raise RuntimeError(f"Expected {expected * 2} trainable LoRA tensors, found {len(trainable)}")
    if any(".lora_A." not in name and ".lora_B." not in name for name in trainable):
        raise RuntimeError(f"Non-LoRA parameters remain trainable: {trainable}")
    if any(not isinstance(parameter, DTensor) for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("Every trainable LoRA tensor must have an explicit TP placement")
    return injected


def _lora_parameters(model) -> list[tuple[str, DTensor]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (".lora_A." in name or ".lora_B." in name)
    ]


def _adapter_parameters(model) -> list[tuple[str, nn.Parameter]]:
    return [*_lora_parameters(model), *token_adapter_parameters(model)]


def _peft_key(name: str) -> str:
    return "base_model.model." + name


# Block 5 saves and restores the LoRA and token adapter state.
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _s0_tokenizer_codec_manifest_sha256(
    token_manifest_path: str | None,
) -> str | None:
    if not token_manifest_path:
        return None
    codec = load_model_text_codec_for_token_manifest(
        Path(token_manifest_path)
    )
    return None if codec is None else codec.manifest_sha256


def save_adapter(model, output_dir: Path, args: argparse.Namespace, topology: WorldModelTopology, rank: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "tp_adapter_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    adapter_parameters = _adapter_parameters(model)
    logical_trainable_parameters = sum(
        int(parameter.numel()) for _, parameter in adapter_parameters
    )
    # Save one local shard per rank for an exact train resume.
    local_state = {
        name: parameter.to_local().detach().cpu().contiguous()
        for name, parameter in adapter_parameters
    }
    save_file(local_state, shard_dir / f"rank-{rank:05d}.safetensors")

    # Gather full LoRA tensors for standard PEFT inference.
    full_state = {}
    for name, parameter in _lora_parameters(model):
        full = parameter.full_tensor()
        if rank == 0:
            full_state[_peft_key(name)] = full.detach().cpu().contiguous()
    if rank == 0:
        save_file(full_state, output_dir / ADAPTER_WEIGHTS_NAME)
        stable_model_path = os.environ.get("MODEL_PATH", str(args.model_path))
        _write_json(
            output_dir / ADAPTER_CONFIG_NAME,
            {
                "alpha_pattern": {},
                "auto_mapping": None,
                "base_model_name_or_path": stable_model_path,
                "bias": "none",
                "exclude_modules": None,
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "layer_replication": None,
                "layers_pattern": None,
                "layers_to_transform": None,
                "loftq_config": {},
                "lora_alpha": args.lora_alpha,
                "lora_bias": False,
                "lora_dropout": 0.0,
                "megatron_config": None,
                "megatron_core": "megatron.core",
                "modules_to_save": None,
                "peft_type": "LORA",
                "r": args.lora_r,
                "rank_pattern": {},
                "revision": None,
                "target_modules": sorted(ATTENTION_LORA_TARGETS),
                "task_type": "CAUSAL_LM",
                "use_dora": False,
                "use_rslora": False,
            },
        )
        # Bind the adapter to its model, corpus, tokenizer, and run contract.
        adapter_identity = {
            **required_run_identity(),
            "tokenizer_artifacts_sha256": args.tokenizer_artifacts_sha256,
        }
        s0_codec_hash = _s0_tokenizer_codec_manifest_sha256(
            args.token_adapter_manifest
        )
        if s0_codec_hash is not None:
            adapter_identity["s0_tokenizer_codec_manifest_sha256"] = s0_codec_hash
        _write_json(
            output_dir / TP_MANIFEST_NAME,
            {
                "format": "mentor-rl-s0-tp-lora-v1",
                "topology": asdict(topology),
                "base_model": stable_model_path,
                "lora": asdict(LoraSpec(args.lora_r, args.lora_alpha, args.lora_dropout)),
                "objective": loss_contract_config(args.loss_contract),
                "logical_trainable_parameters": logical_trainable_parameters,
                "rank_shards": topology.world_size,
                "consolidated_peft_adapter": ADAPTER_WEIGHTS_NAME,
                "token_adapter": bool(args.token_adapter_manifest),
                "identity": adapter_identity,
            },
        )
        # Reload the consolidated adapter and require exact tensor equality.
        reloaded = load_file(output_dir / ADAPTER_WEIGHTS_NAME, device="cpu")
        if set(reloaded) != set(full_state) or any(
            not torch.equal(reloaded[name], tensor) for name, tensor in full_state.items()
        ):
            raise RuntimeError("Consolidated PEFT adapter failed exact save/reload parity")
        from peft import LoraConfig

        parsed = LoraConfig.from_pretrained(output_dir)
        if set(parsed.target_modules or ()) != set(ATTENTION_LORA_TARGETS):
            raise RuntimeError("Consolidated adapter_config.json is not PEFT-compatible")
    # Custom tokenizers also save their two trainable token-row deltas.
    if args.token_adapter_manifest:
        save_token_adapter(
            model, output_dir, Path(args.token_adapter_manifest), rank
        )
    dist.barrier()


def load_adapter_shard(
    model,
    checkpoint: Path,
    topology: WorldModelTopology,
    rank: int,
    loss_contract: str,
    token_adapter_manifest: str | None,
    tokenizer_artifacts_sha256: str,
) -> None:
    manifest = json.loads((checkpoint / TP_MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != "mentor-rl-s0-tp-lora-v1":
        raise RuntimeError("The checkpoint has an unsupported S0 LoRA format")
    # A resume must use the same shard layout and the same artifact identity.
    saved_topology = manifest.get("topology", {})
    for field in ("world_size", "tp_size", "ep_size"):
        if saved_topology.get(field) != getattr(topology, field):
            raise RuntimeError(
                f"Checkpoint {field}={saved_topology.get(field)!r} does not match current {getattr(topology, field)}"
            )
    saved_objective = manifest.get("objective")
    expected_objective = loss_contract_config(loss_contract)
    if saved_objective != expected_objective:
        raise RuntimeError("The checkpoint loss contract does not match this run")
    expected_identity = {
        **required_run_identity(),
        "tokenizer_artifacts_sha256": tokenizer_artifacts_sha256,
    }
    s0_codec_hash = _s0_tokenizer_codec_manifest_sha256(
        token_adapter_manifest
    )
    if s0_codec_hash is not None:
        expected_identity["s0_tokenizer_codec_manifest_sha256"] = s0_codec_hash
    if manifest.get("identity") != expected_identity:
        raise RuntimeError(
            "Checkpoint model/tokenizer/token-adapter/codec identity does not match "
            "the current launch contract"
        )
    shard_path = checkpoint / "tp_adapter_shards" / f"rank-{rank:05d}.safetensors"
    state = load_file(shard_path, device="cpu")
    expected = dict(_adapter_parameters(model))
    if set(state) != set(expected):
        raise RuntimeError("Rank-local adapter checkpoint keys do not match the injected model")
    # Copy each saved local tensor into its installed adapter parameter.
    with torch.no_grad():
        for name, parameter in expected.items():
            local = parameter.to_local()
            if tuple(local.shape) != tuple(state[name].shape):
                raise RuntimeError(f"Adapter shard shape mismatch for {name}")
            local.copy_(state[name].to(local.device, dtype=local.dtype))
    dist.barrier()


def _optimizer_parameters(optimizer) -> list[torch.Tensor]:
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def save_optimizer_shard(optimizer, output_dir: Path, rank: int) -> None:
    # Store optimizer tensors in stable parameter order for this rank.
    parameters = _optimizer_parameters(optimizer)
    index = {id(parameter): position for position, parameter in enumerate(parameters)}
    states = {}
    for parameter, state in optimizer.state.items():
        encoded = {}
        for key, value in state.items():
            if isinstance(value, DTensor):
                encoded[key] = {"is_dtensor": True, "value": value.to_local().detach().cpu()}
            elif torch.is_tensor(value):
                encoded[key] = {
                    "is_dtensor": False,
                    "device_type": value.device.type,
                    "value": value.detach().cpu(),
                }
            else:
                encoded[key] = value
        states[index[id(parameter)]] = encoded
    groups = []
    offset = 0
    for group in optimizer.param_groups:
        item = {key: value for key, value in group.items() if key != "params"}
        item["params"] = list(range(offset, offset + len(group["params"])))
        offset += len(group["params"])
        groups.append(item)
    torch.save({"state": states, "param_groups": groups}, output_dir / f"optimizer-rank-{rank:05d}.pt")


def load_optimizer_shard(optimizer, checkpoint: Path, rank: int) -> None:
    payload = torch.load(
        checkpoint / f"optimizer-rank-{rank:05d}.pt", map_location="cpu", weights_only=False
    )
    # Rebuild each DTensor state with the parameter placement from this run.
    parameters = _optimizer_parameters(optimizer)
    optimizer.state.clear()
    for index, encoded in payload["state"].items():
        parameter = parameters[int(index)]
        restored = {}
        for key, value in encoded.items():
            if isinstance(value, dict) and "is_dtensor" in value:
                device = torch.device("cuda", torch.cuda.current_device())
                tensor = value["value"]
                if value["is_dtensor"]:
                    if not isinstance(parameter, DTensor):
                        raise RuntimeError(f"Optimizer expected a DTensor parameter for state {index}")
                    tensor = tensor.to(device)
                    tensor = DTensor.from_local(
                        tensor,
                        parameter.device_mesh,
                        parameter.placements,
                        run_check=False,
                        shape=parameter.shape,
                        stride=parameter.stride(),
                    )
                elif value.get("device_type") != "cpu":
                    tensor = tensor.to(device)
                restored[key] = tensor
            else:
                restored[key] = value
        optimizer.state[parameter] = restored
    for current, saved in zip(optimizer.param_groups, payload["param_groups"], strict=True):
        for key, value in saved.items():
            if key != "params":
                current[key] = value


def save_rng(output_dir: Path, rank: int) -> None:
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.random.get_rng_state(),
        },
        output_dir / f"rng-rank-{rank:05d}.pt",
    )


def load_rng(checkpoint: Path, rank: int) -> None:
    payload = torch.load(checkpoint / f"rng-rank-{rank:05d}.pt", map_location="cpu", weights_only=False)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch"])
    torch.cuda.random.set_rng_state(payload["cuda"], device=torch.cuda.current_device())


def checkpoint_paths(output_dir: Path) -> list[Path]:
    result = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if path.is_dir() and (path / "CHECKPOINT_COMPLETE").is_file():
            result.append(path)
    return sorted(result, key=lambda path: int(path.name.rsplit("-", 1)[1]))


def resolve_resume(value: str | None, output_dir: Path) -> Path | None:
    if not value:
        return None
    if value.lower() in {"auto", "true", "yes", "latest"}:
        paths = checkpoint_paths(output_dir)
        return paths[-1] if paths else None
    path = Path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    state: LoopState,
    output_dir: Path,
    args: argparse.Namespace,
    topology: WorldModelTopology,
    rank: int,
    *,
    final: bool = False,
) -> None:
    # The final state uses the output root. Interim states use step directories.
    destination = output_dir if final else output_dir / f"checkpoint-{state.global_step}"
    destination.mkdir(parents=True, exist_ok=True)
    save_adapter(model, destination, args, topology, rank)
    save_optimizer_shard(optimizer, destination, rank)
    save_rng(destination, rank)
    if rank == 0:
        torch.save(scheduler.state_dict(), destination / "scheduler.pt")
        _write_json(destination / TRAINING_STATE_NAME, asdict(state))
    dist.barrier()
    # Publish the complete marker only after every rank writes its files.
    if rank == 0 and not final:
        (destination / "CHECKPOINT_COMPLETE").write_text(
            f"global_step={state.global_step}\n", encoding="utf-8"
        )
    dist.barrier()
    # Keep only the newest complete interim checkpoints.
    if rank == 0 and not final:
        stale = checkpoint_paths(output_dir)[: -args.save_total_limit]
        for path in stale:
            shutil.rmtree(path)
    dist.barrier()


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint: Path,
    topology: WorldModelTopology,
    rank: int,
    loss_contract: str,
    token_adapter_manifest: str | None,
    tokenizer_artifacts_sha256: str,
) -> LoopState:
    load_adapter_shard(
        model,
        checkpoint,
        topology,
        rank,
        loss_contract,
        token_adapter_manifest,
        tokenizer_artifacts_sha256,
    )
    load_optimizer_shard(optimizer, checkpoint, rank)
    scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True))
    load_rng(checkpoint, rank)
    payload = json.loads((checkpoint / TRAINING_STATE_NAME).read_text(encoding="utf-8"))
    return LoopState(**payload)


def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to(torch.cuda.current_device(), non_blocking=True) for key, value in batch.items()}


def assert_identical_tp_batch(
    batch: dict[str, torch.Tensor], group, *, global_step: int, batch_index: int
) -> list[int]:
    """Check shape and content fingerprints before every model forward."""

    # Each TP rank must receive identical IDs, masks, and sequence lengths.
    checksum = torch.zeros(4, dtype=torch.int64, device=torch.cuda.current_device())
    for key_index, key in enumerate(sorted(batch)):
        tensor = batch[key].detach().to(torch.int64).reshape(-1)
        positions = torch.arange(1, tensor.numel() + 1, device=tensor.device, dtype=torch.int64)
        checksum[0] += tensor.numel()
        checksum[1] += tensor.sum()
        checksum[2] += (tensor * positions).sum()
        checksum[3] += (key_index + 1) * (tensor.square().sum() + len(key))
    minimum = checksum.clone()
    maximum = checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    if not torch.equal(minimum, maximum):
        raise RuntimeError(
            "TP ranks received different model inputs before forward at "
            f"global_step={global_step}, batch_index={batch_index}: "
            f"local={checksum.tolist()}, min={minimum.tolist()}, max={maximum.tolist()}"
        )
    return checksum.tolist()


def assert_first_step_gradient_contract(model, optimizer, group) -> None:
    """Require a frozen base and finite LoRA gradients on every TP rank."""

    # The first step detects an unfrozen base or a disconnected adapter.
    optimizer_parameters = {id(parameter) for parameter in _optimizer_parameters(optimizer)}
    local_errors = []
    for name, parameter in model.named_parameters():
        is_adapter = (
            ".lora_A." in name
            or ".lora_B." in name
            or name.endswith("token_input_delta")
            or name.endswith("token_output_delta")
        )
        if is_adapter:
            if not parameter.requires_grad or id(parameter) not in optimizer_parameters:
                local_errors.append(f"trainable adapter parameter missing from optimizer: {name}")
            if parameter.grad is None:
                local_errors.append(f"adapter parameter has no first-step gradient: {name}")
            else:
                gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                if not torch.isfinite(gradient).all():
                    local_errors.append(f"adapter parameter has non-finite gradient: {name}")
        else:
            if parameter.requires_grad:
                local_errors.append(f"base parameter is not frozen: {name}")
            if parameter.grad is not None:
                local_errors.append(f"frozen base parameter acquired a gradient: {name}")
            if id(parameter) in optimizer_parameters:
                local_errors.append(f"frozen base parameter is present in optimizer: {name}")
    gathered: list[list[str] | None] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local_errors, group=group)
    errors = [f"rank{index}: {error}" for index, items in enumerate(gathered) for error in (items or [])]
    if errors:
        raise RuntimeError("First-step frozen-base/gradient gate failed: " + "; ".join(errors[:20]))


def assert_tp_token_row_replica_parity(model, group, *, gradients: bool) -> int:
    """Require identical token-row replicas. Do not reduce their gradients."""

    checked = 0
    for name, parameter in token_adapter_parameters(model):
        value = parameter.grad if gradients else parameter
        kind = "gradient" if gradients else "parameter"
        if value is None:
            raise RuntimeError(f"Token-row {kind} is missing for {name}")
        if not isinstance(value, DTensor):
            raise RuntimeError(f"Token-row {kind} is not a DTensor for {name}")
        if value.placements != (Replicate(),):
            raise RuntimeError(
                f"Token-row {kind} is not replicated for {name}: {value.placements}"
            )
        local = value.to_local().detach()
        if not torch.isfinite(local).all():
            raise RuntimeError(f"Token-row {kind} is non-finite for {name}")
        minimum = local.clone()
        maximum = local.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(minimum, maximum):
            max_abs_diff = float((maximum - minimum).abs().max().item())
            raise RuntimeError(
                f"TP token-row {kind} replicas diverged for {name}; "
                f"max_abs_diff={max_abs_diff}"
            )
        checked += 1
    return checked


def assert_tp_replicated_lora_parity(model, group, *, gradients: bool) -> int:
    """Require identical gradients/parameters for every replicated LoRA tensor."""

    checked = 0
    for name, parameter in _lora_parameters(model):
        if parameter.placements != (Replicate(),):
            continue
        value = parameter.grad if gradients else parameter
        kind = "gradient" if gradients else "parameter"
        if value is None:
            raise RuntimeError(f"Replicated LoRA {kind} is missing for {name}")
        if not isinstance(value, DTensor) or value.placements != (Replicate(),):
            raise RuntimeError(
                f"Replicated LoRA {kind} has the wrong DTensor placement for {name}"
            )
        local = value.to_local().detach()
        if not torch.isfinite(local).all():
            raise RuntimeError(f"Replicated LoRA {kind} is non-finite for {name}")
        minimum = local.clone()
        maximum = local.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(minimum, maximum):
            max_abs_diff = float((maximum - minimum).abs().max().item())
            raise RuntimeError(
                f"TP replicated LoRA {kind}s diverged for {name}; "
                f"max_abs_diff={max_abs_diff}"
            )
        checked += 1
    if checked == 0:
        raise RuntimeError("No replicated LoRA tensors were available for TP parity")
    return checked


def assert_finite_lora_parameters(model, group) -> None:
    local_ok = all(
        torch.isfinite(parameter.to_local()).all().item()
        for _, parameter in _adapter_parameters(model)
    )
    status = torch.tensor(int(local_ok), device=torch.cuda.current_device())
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
    if status.item() != 1:
        raise RuntimeError("First optimizer step produced non-finite adapter parameters")


def average_lora_gradients(model, dp_group, dp_size: int) -> None:
    """Average corresponding local LoRA shards across model replicas."""

    for name, parameter in _adapter_parameters(model):
        if parameter.grad is None:
            raise RuntimeError(f"Adapter parameter has no gradient before DP reduction: {name}")
        gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM, group=dp_group)
        gradient.div_(dp_size)


def assert_dp_lora_parameter_parity(model, dp_group) -> None:
    """Fail if corresponding LoRA shards differ across model replicas."""

    for name, parameter in _adapter_parameters(model):
        local = parameter.to_local().detach().float()
        checksum = torch.stack((local.sum(), local.square().sum()))
        minimum, maximum = checksum.clone(), checksum.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=dp_group)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=dp_group)
        if not torch.allclose(minimum, maximum, rtol=1e-5, atol=1e-6):
            raise RuntimeError(f"DP replicas diverged for adapter shard {name}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


# Block 6 applies the S0 loss and synchronizes model replicas.
def compute_loss_objective(
    model,
    batch: dict[str, torch.Tensor],
    loss_contract: str,
) -> dict[str, Any]:
    """Compute one forward pass and the fixed S0 loss contract."""

    if loss_contract != S0_TARGET_AWARE_LOSS_CONTRACT:
        raise ValueError(f"Unknown loss contract: {loss_contract!r}")

    mapping_target_mask = batch.get("mapping_target_mask")
    model_batch = {
        key: value
        for key, value in batch.items()
        if key not in {"labels", "mapping_target_mask"}
    }
    labels = batch["labels"]
    outputs = model(**model_batch)
    logits = outputs.logits.float()
    # Each logit predicts the next token, so shift labels by one position.
    shift_labels = nn.functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    flat_token_loss = nn.functional.cross_entropy(
        logits.view(-1, model.config.vocab_size),
        shift_labels.view(-1).to(logits.device),
        ignore_index=-100,
        reduction="none",
    )
    token_loss = flat_token_loss.view_as(shift_labels)
    # Completion loss covers the complete assistant response.
    completion_positions = shift_labels != -100
    completion_tokens = int(completion_positions.sum().item())
    if completion_tokens == 0:
        raise RuntimeError("The batch has no completion target token")
    completion_loss = token_loss[completion_positions].mean()

    if mapping_target_mask is None or mapping_target_mask.shape != labels.shape:
        raise RuntimeError("The target-aware batch has no valid mapping target mask")
    # Mapping loss covers only the answer values named by the S0 family.
    target_labels = labels.masked_fill(~mapping_target_mask.to(torch.bool), -100)
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
        S0_TARGET_AWARE_COMPLETION_WEIGHT * completion_loss
        + S0_TARGET_AWARE_MAPPING_WEIGHT * mapping_target_loss
    )
    return {
        "loss": loss,
        "completion_loss": completion_loss,
        "mapping_target_loss": mapping_target_loss,
        "completion_tokens": completion_tokens,
        "mapping_target_tokens": mapping_target_tokens,
    }


def reduce_training_loss_window(
    *,
    loss_sum: float,
    completion_loss_sum: float,
    mapping_target_loss_sum: float,
    sample_count: int,
    mapping_sample_count: int,
    input_tokens: int,
    dp_group,
) -> dict[str, float]:
    """Average loss values across all data-parallel replicas."""

    # Sum local metric totals before rank 0 calculates global averages.
    values = torch.tensor(
        [
            loss_sum,
            completion_loss_sum,
            mapping_target_loss_sum,
            sample_count,
            mapping_sample_count,
            input_tokens,
        ],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=dp_group)
    reduced = values.tolist()
    total_samples = max(1.0, reduced[3])
    mapping_samples = max(1.0, reduced[4])
    return {
        "loss": reduced[0] / total_samples,
        "completion_loss": reduced[1] / total_samples,
        "mapping_target_loss": reduced[2] / mapping_samples,
        "mapping_sample_count": reduced[4],
        "input_tokens": reduced[5],
    }


@torch.no_grad()
def evaluate_validation(
    model,
    dataset,
    collator,
    args: argparse.Namespace,
    topology: WorldModelTopology,
    *,
    rank: int,
    dp_rank: int,
    dp_group,
    global_step: int,
    epoch: int,
    trigger: str,
    output_dir: Path,
    wandb_run=None,
) -> dict[str, Any]:
    """Measure the target-aware loss on the validation panel."""

    was_training = model.training
    model.eval()
    tp_group = model._device_mesh.get_group()
    sampler = range(dp_rank, len(dataset), topology.data_parallel_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collator,
        pin_memory=True,
    )
    torch.cuda.synchronize()
    started = time.monotonic()
    loss_sum = 0.0
    completion_loss_sum = 0.0
    mapping_loss_sum = 0.0
    sample_count = 0
    input_tokens = 0
    for batch_index, batch in enumerate(dataloader):
        batch = move_batch(batch)
        assert_identical_tp_batch(
            batch,
            tp_group,
            global_step=global_step,
            batch_index=batch_index,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            objective = compute_loss_objective(
                model,
                batch,
                args.loss_contract,
            )
        batch_size = int(batch["input_ids"].shape[0])
        loss_sum += objective["loss"].item() * batch_size
        completion_loss_sum += (
            objective["completion_loss"].item() * batch_size
        )
        mapping_loss_sum += (
            objective["mapping_target_loss"].item() * batch_size
        )
        sample_count += batch_size
        input_tokens += int(batch["attention_mask"].sum().item())
    reduced = reduce_training_loss_window(
        loss_sum=loss_sum,
        completion_loss_sum=completion_loss_sum,
        mapping_target_loss_sum=mapping_loss_sum,
        sample_count=sample_count,
        mapping_sample_count=sample_count,
        input_tokens=input_tokens,
        dp_group=dp_group,
    )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    model.train(was_training)
    payload = {
        "event": "validation",
        "trigger": trigger,
        "epoch": epoch,
        "global_step": global_step,
        "loss_contract": args.loss_contract,
        "loss": reduced["loss"],
        "completion_loss": reduced["completion_loss"],
        "mapping_target_loss": reduced["mapping_target_loss"],
        "examples": int(reduced["mapping_sample_count"]),
        "input_tokens": int(reduced["input_tokens"]),
        "elapsed_seconds": elapsed,
    }
    if rank == 0:
        print(json.dumps(payload, sort_keys=True), flush=True)
        append_jsonl(output_dir / "validation_metrics.jsonl", payload)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "global_step": global_step,
                    "eval/loss": payload["loss"],
                    "eval/completion_loss": payload["completion_loss"],
                    "eval/mapping_target_loss": payload[
                        "mapping_target_loss"
                    ],
                    "eval/examples": payload["examples"],
                    "eval/runtime": payload["elapsed_seconds"],
                }
            )
            wandb_run.summary["validation_loss_latest"] = payload["loss"]
            wandb_run.summary["validation_global_step_latest"] = global_step
    return payload


def train_loop(
    model,
    dataset,
    validation_dataset,
    collator,
    optimizer,
    scheduler,
    state: LoopState,
    total_steps: int,
    args: argparse.Namespace,
    topology: WorldModelTopology,
    output_dir: Path,
    rank: int,
    dp_rank: int,
    dp_group,
    wandb_run=None,
) -> LoopState:
    model.train()
    tp_group = model._device_mesh.get_group()
    running_loss = 0.0
    running_completion_loss = 0.0
    running_mapping_target_loss = 0.0
    running_loss_samples = 0
    running_mapping_samples = 0
    segment_start_step = state.global_step
    segment_start_time = time.monotonic()
    interval_start_time = segment_start_time
    interval_start_step = state.global_step
    interval_tokens = 0
    segment_tokens = 0
    throughput_path = output_dir / "throughput.jsonl"
    first_step_gate_pending = state.global_step == 0
    for epoch in range(state.epoch, args.num_train_epochs):
        # Build one deterministic global order and add declared DP padding.
        padded_epoch_indices, _ = (
            epoch_training_indices_with_replica_padding(
                len(dataset),
                args.seed + epoch,
                args.preserve_dataset_order,
                topology.data_parallel_size,
            )
        )
        # Each model replica receives one strided shard of the global order.
        epoch_indices = padded_epoch_indices[
            dp_rank::topology.data_parallel_size
        ]
        dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            sampler=epoch_indices,
            collate_fn=collator,
            pin_memory=True,
        )
        start_batch = state.batches_consumed_in_epoch if epoch == state.epoch else 0
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        accumulation_target = args.gradient_accumulation_steps
        for batch_index, batch in enumerate(dataloader):
            if batch_index < start_batch:
                continue
            batch = move_batch(batch)
            checksum = assert_identical_tp_batch(
                batch,
                tp_group,
                global_step=state.global_step,
                batch_index=batch_index,
            )
            # A short final window uses its actual accumulation count.
            if accumulated == 0:
                accumulation_target = min(
                    args.gradient_accumulation_steps, len(dataloader) - batch_index
                )
            input_tokens = int(batch["attention_mask"].sum().item())
            interval_tokens += input_tokens
            if rank == 0:
                segment_tokens += input_tokens
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective = compute_loss_objective(model, batch, args.loss_contract)
                loss = objective["loss"]
                scaled_loss = loss / accumulation_target
            with torch.autograd.set_multithreading_enabled(
                args.autograd_multithreading
            ):
                scaled_loss.backward()
            accumulated += 1
            running_loss += float(loss.detach().item())
            running_completion_loss += float(
                objective["completion_loss"].detach().item()
            )
            if objective["mapping_target_loss"] is not None:
                running_mapping_target_loss += float(
                    objective["mapping_target_loss"].detach().item()
                )
                running_mapping_samples += 1
            running_loss_samples += 1
            state.batches_consumed_in_epoch = batch_index + 1
            is_last_batch = batch_index + 1 == len(dataloader)
            if accumulated < accumulation_target and not is_last_batch:
                continue

            from torch.distributed._tensor.experimental import implicit_replication

            # Average adapter gradients across model replicas exactly once.
            average_lora_gradients(model, dp_group, topology.data_parallel_size)
            if first_step_gate_pending:
                # Check gradients before the first optimizer update.
                assert_first_step_gradient_contract(model, optimizer, tp_group)
                replicated_lora_gradient_replicas = (
                    assert_tp_replicated_lora_parity(
                        model, tp_group, gradients=True
                    )
                )
                token_row_gradient_replicas = assert_tp_token_row_replica_parity(
                    model, tp_group, gradients=True
                )
            with implicit_replication():
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer), args.max_grad_norm)
                optimizer.step()
            if first_step_gate_pending:
                # Check parameter parity after the first optimizer update.
                assert_finite_lora_parameters(model, tp_group)
                assert_dp_lora_parameter_parity(model, dp_group)
                replicated_lora_parameter_replicas = (
                    assert_tp_replicated_lora_parity(
                        model, tp_group, gradients=False
                    )
                )
                token_row_parameter_replicas = assert_tp_token_row_replica_parity(
                    model, tp_group, gradients=False
                )
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "event": "first_step_gate",
                                "status": "passed",
                                "batch_checksum": checksum,
                                "tp_token_row_gradient_replicas_checked": (
                                    token_row_gradient_replicas
                                ),
                                "tp_replicated_lora_gradient_replicas_checked": (
                                    replicated_lora_gradient_replicas
                                ),
                                "tp_replicated_lora_parameter_replicas_checked": (
                                    replicated_lora_parameter_replicas
                                ),
                                "tp_token_row_parameter_replicas_checked": (
                                    token_row_parameter_replicas
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.summary["first_step_gradient_gate"] = "passed"
                first_step_gate_pending = False
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            state.global_step += 1

            if state.global_step % args.logging_steps == 0:
                # Rank 0 records global loss and throughput totals.
                reduced_loss = reduce_training_loss_window(
                    loss_sum=running_loss,
                    completion_loss_sum=running_completion_loss,
                    mapping_target_loss_sum=running_mapping_target_loss,
                    sample_count=running_loss_samples,
                    mapping_sample_count=running_mapping_samples,
                    input_tokens=interval_tokens,
                    dp_group=dp_group,
                )
                if rank == 0:
                    torch.cuda.synchronize()
                    now = time.monotonic()
                    elapsed = now - interval_start_time
                    interval_steps = state.global_step - interval_start_step
                    payload = {
                        "event": "train",
                        "epoch": epoch,
                        "global_step": state.global_step,
                        "loss_contract": args.loss_contract,
                        "loss": reduced_loss["loss"],
                        "completion_loss": reduced_loss["completion_loss"],
                        "mapping_target_loss": (
                            reduced_loss["mapping_target_loss"]
                            if reduced_loss["mapping_sample_count"] > 0
                            else None
                        ),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "global_batch_size": topology.global_batch_size,
                        "elapsed_seconds": elapsed,
                        "steps_per_second": interval_steps / elapsed,
                        "input_tokens": int(reduced_loss["input_tokens"]),
                        "input_tokens_per_second": reduced_loss["input_tokens"] / elapsed,
                    }
                    print(json.dumps(payload, sort_keys=True), flush=True)
                    append_jsonl(throughput_path, payload)
                    if wandb_run is not None:
                        wandb_payload = {
                            "global_step": state.global_step,
                            "train/loss": payload["loss"],
                            "train/completion_loss": payload["completion_loss"],
                            "train/learning_rate": payload["learning_rate"],
                            "train/steps_per_second": payload["steps_per_second"],
                            "train/input_tokens_per_second": payload["input_tokens_per_second"],
                            "train/input_tokens_interval": payload["input_tokens"],
                        }
                        if payload["mapping_target_loss"] is not None:
                            wandb_payload["train/mapping_target_loss"] = payload[
                                "mapping_target_loss"
                            ]
                        wandb_run.log(wandb_payload)
                running_loss = 0.0
                running_completion_loss = 0.0
                running_mapping_target_loss = 0.0
                running_loss_samples = 0
                running_mapping_samples = 0
                interval_tokens = 0
                interval_start_step = state.global_step
                if rank == 0:
                    interval_start_time = now
            if args.save_strategy == "steps" and state.global_step % args.save_steps == 0:
                # Save a complete resume point at the declared interval.
                state.epoch = epoch
                save_checkpoint(model, optimizer, scheduler, state, output_dir, args, topology, rank)
            if state.global_step >= total_steps:
                state.epoch = epoch + (1 if is_last_batch else 0)
                validation = evaluate_validation(
                    model,
                    validation_dataset,
                    collator,
                    args,
                    topology,
                    rank=rank,
                    dp_rank=dp_rank,
                    dp_group=dp_group,
                    global_step=state.global_step,
                    epoch=state.epoch,
                    trigger="epoch_end",
                    output_dir=output_dir,
                    wandb_run=wandb_run,
                )
                segment_start_time += validation["elapsed_seconds"]
                interval_start_time += validation["elapsed_seconds"]
                if rank == 0:
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - segment_start_time
                    _write_json(
                        output_dir / "throughput_summary.json",
                        {
                            "segment_start_step": segment_start_step,
                            "segment_end_step": state.global_step,
                            "optimizer_steps": state.global_step - segment_start_step,
                            "elapsed_seconds": elapsed,
                            "input_tokens": segment_tokens,
                            "steps_per_second": (state.global_step - segment_start_step) / elapsed,
                            "input_tokens_per_second": segment_tokens / elapsed,
                            "topology": asdict(topology),
                        },
                    )
                return state
        state.epoch = epoch + 1
        state.batches_consumed_in_epoch = 0
        validation = evaluate_validation(
            model,
            validation_dataset,
            collator,
            args,
            topology,
            rank=rank,
            dp_rank=dp_rank,
            dp_group=dp_group,
            global_step=state.global_step,
            epoch=state.epoch,
            trigger="epoch_end",
            output_dir=output_dir,
            wandb_run=wandb_run,
        )
        segment_start_time += validation["elapsed_seconds"]
        interval_start_time += validation["elapsed_seconds"]
    if rank == 0:
        torch.cuda.synchronize()
        elapsed = time.monotonic() - segment_start_time
        _write_json(
            output_dir / "throughput_summary.json",
            {
                "segment_start_step": segment_start_step,
                "segment_end_step": state.global_step,
                "optimizer_steps": state.global_step - segment_start_step,
                "elapsed_seconds": elapsed,
                "input_tokens": segment_tokens,
                "steps_per_second": (state.global_step - segment_start_step) / max(elapsed, 1e-12),
                "input_tokens_per_second": segment_tokens / max(elapsed, 1e-12),
                "topology": asdict(topology),
            },
        )
    return state


# Block 7 validates the contract and runs the complete train job.
def main() -> None:
    global _WANDB_RUN
    args = parse_args()
    validate_cli(args)
    # Validate the launch identity before any distributed resource starts.
    run_identity = required_run_identity()
    method_match = re.fullmatch(
        r"(oss20b|oss120b)-"
        r"(plain-base-tokenizer|ordinary-domain-bpe|"
        r"atomic-plus-domain-bpe|fully-atomic-identifiers)-"
        r"lora-r(32|128|1024)",
        run_identity["method_id"],
    )
    if method_match is None:
        raise RuntimeError(
            "The S0 method ID must select one supported LoRA configuration"
        )
    tokenizer_method = {
        "plain-base-tokenizer": "plain_base_tokenizer",
        "ordinary-domain-bpe": "ordinary_domain_bpe",
        "atomic-plus-domain-bpe": "atomic_plus_domain_bpe",
        "fully-atomic-identifiers": "fully_atomic_identifiers",
    }[method_match.group(2)]
    if int(method_match.group(3)) != args.lora_r:
        raise RuntimeError(
            "The S0 method ID differs from the selected LoRA rank"
        )
    custom_tokenizer_methods = {
        "ordinary_domain_bpe",
        "atomic_plus_domain_bpe",
        "fully_atomic_identifiers",
    }
    custom_tokenizer = tokenizer_method in custom_tokenizer_methods

    supported_tokenizer_methods = {
        "plain_base_tokenizer",
        *custom_tokenizer_methods,
    }

    if tokenizer_method not in supported_tokenizer_methods:
        raise RuntimeError(
            f"The S0 method ID has an unknown tokenizer: {tokenizer_method!r}"
        )

    if bool(args.token_adapter_manifest) != custom_tokenizer:
        raise RuntimeError(
            "Only a custom S0 tokenizer requires --token_adapter_manifest"
        )
    # Plain tokenization has no codec or trainable token-row adapter.
    s0_codec = (
        None
        if not args.token_adapter_manifest
        else load_model_text_codec_for_token_manifest(
            Path(args.token_adapter_manifest)
        )
    )
    # Validate the process layout before the model loads.
    topology, rank, local_rank, device_index = configure_topology(args)
    if args.strict_tested_stack:
        validate_tested_stack(
            {
                "accelerate": accelerate.__version__,
                "peft": peft.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "trl": trl.__version__,
            }
        )
    # Start the process group only after all local checks pass.
    initialize_distributed(rank, topology.world_size, device_index)
    dist.barrier(device_ids=[device_index])
    if rank == 0:
        distributed_ready_marker = os.environ.get("MENTOR_DISTRIBUTED_READY_MARKER")
        if distributed_ready_marker:
            marker_path = Path(distributed_ready_marker)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                f"job_id={os.environ.get('SLURM_JOB_ID', 'unknown')}\n"
                f"world_size={topology.world_size}\n",
                encoding="utf-8",
            )
    seed_everything(args.seed)
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "topology_preflight",
                    **asdict(topology),
                    "data_parallel_size": topology.data_parallel_size,
                    "global_batch_size": topology.global_batch_size,
                    "local_rank": local_rank,
                    "device_index": device_index,
                    "visible_device_count": torch.cuda.device_count(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        _WANDB_RUN = initialize_wandb(
            args,
            topology,
            output_dir=output_dir,
            s0_codec=s0_codec,
        )
        _write_json(
            output_dir / "wandb_run.json",
            {
                "id": _WANDB_RUN.id,
                "name": _WANDB_RUN.name,
                "project": _WANDB_RUN.project,
                "entity": _WANDB_RUN.entity,
                "url": _WANDB_RUN.url,
                "mode": os.environ["WANDB_MODE"],
            },
        )
    dist.barrier()

    # Inspect the model shape and native TP plan before weight allocation.
    config = AutoConfig.from_pretrained(
        args.model_path, local_files_only=args.local_files_only, trust_remote_code=True
    )
    validate_gpt_oss_dimensions(config, args.tp_size)
    validate_native_tp_plan(config.base_model_tp_plan)
    install_autograd_expert_gather()
    from torch.distributed.device_mesh import init_device_mesh

    # The first mesh axis holds model replicas. The second axis holds TP shards.
    model_mesh = init_device_mesh(
        "cuda",
        (topology.data_parallel_size, args.tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    tp_mesh = model_mesh["tp"]
    dp_mesh = model_mesh["dp"]
    dp_group = dp_mesh.get_group()
    dp_rank = rank // args.tp_size
    tp_rank = rank % args.tp_size
    assert_expert_gradient_parity(tp_mesh, tp_rank)
    assert_tp_lora_gradient_parity(tp_mesh)
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "distributed_gradient_preflight",
                    "expert_autograd_pair": "passed",
                    "router_autograd_input": "passed",
                    "tp_lora_unsharded_parity": "passed",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    # Validate and tokenize the corpus before model weight allocation.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer_files = tokenizer_artifact_hashes(
        Path(args.tokenizer_path)
    )
    tokenizer_identity_sha256 = hashlib.sha256(
        json.dumps(
            tokenizer_files,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    args.tokenizer_artifacts_sha256 = tokenizer_identity_sha256
    dataset, _train_question_families, train_identities = load_tokenized_dataset(
        args.dataset_path,
        tokenizer,
        rank,
        max_length=args.max_length,
        label="training",
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        loss_contract=args.loss_contract,
        s0_codec=s0_codec,
    )
    validation_dataset, _, validation_identities = load_tokenized_dataset(
        args.validation_dataset_path,
        tokenizer,
        rank,
        max_length=args.max_length,
        label="validation",
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        loss_contract=args.loss_contract,
        s0_codec=s0_codec,
        answer_key_path=args.validation_answer_key_path,
    )
    per_replica_rows = math.ceil(len(dataset) / topology.data_parallel_size)
    batches_per_epoch = math.ceil(per_replica_rows / args.per_device_train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    requested_steps = updates_per_epoch * args.num_train_epochs
    total_steps = args.max_steps if args.max_steps > 0 else requested_steps
    if total_steps > requested_steps:
        raise ValueError(
            f"--max_steps={total_steps} exceeds the {requested_steps} updates available "
            "within --num_train_epochs; increase epochs explicitly."
        )
    training_families = {
        str(value) for value in train_identities["question_families"]
    }
    if training_families != set(S0_FAMILIES):
        raise RuntimeError("The S0 dataset must contain all three S0 families only")
    validation_families = {
        str(value) for value in validation_identities["question_families"]
    }
    if validation_families != set(S0_FAMILIES):
        raise RuntimeError(
            "The S0 validation set must contain all three S0 families only"
        )
    if set(validation_identities["prompt_form_ids"]) != {"validation"}:
        raise RuntimeError(
            "The S0 validation set must use the validation prompt form"
        )
    # Build the exact logical exposure and DP padding plan.
    padding_plan = consumed_training_index_plan(
        len(dataset),
        total_steps=total_steps,
        num_train_epochs=args.num_train_epochs,
        replica_count=topology.data_parallel_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        preserve_order=args.preserve_dataset_order,
    )
    consumed_indices = list(padding_plan["logical_indices"])
    exposure_manifest_args = {
        "run_id": run_identity["run_id"],
        "method_id": run_identity["method_id"],
        "run_config_sha256": run_identity["run_config_sha256"],
        "corpus_manifest_sha256": run_identity["corpus_manifest_sha256"],
        "train_sha256": run_identity["train_sha256"],
        "tokenizer_arm_manifest_sha256": run_identity[
            "tokenizer_arm_manifest_sha256"
        ],
        "tokenizer_manifest_sha256": run_identity[
            "tokenizer_manifest_sha256"
        ],
        "record_ids": train_identities["record_ids"],
        "fact_ids": train_identities["fact_ids"],
        "question_families": train_identities["question_families"],
        "prompt_form_ids": train_identities["prompt_form_ids"],
        "consumed_indices": consumed_indices,
        "distributed_padding_indices": list(
            padding_plan["distributed_padding_indices"]
        ),
        "seed": args.seed,
        "total_steps": total_steps,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "data_parallel_size": topology.data_parallel_size,
        "preserve_order": args.preserve_dataset_order,
        "padding_policy": padding_plan["padding_policy"],
        "exposure_scope": s0_exposure_scope(
            os.environ.get("S0_RUN_SCOPE", "").strip()
        ),
    }
    # Write the planned receipt before the first optimizer step.
    exposure_path = output_dir / "run_contract" / "training_exposure.json"
    if rank == 0:
        exposure_path.parent.mkdir(parents=True, exist_ok=True)
        planned_exposure = build_training_exposure_manifest(
            **exposure_manifest_args,
            status="planned",
        )
        _write_json(exposure_path, planned_exposure)
        print(
            json.dumps(
                {
                    "event": "training_exposure_planned",
                    "selected_records": len(dataset),
                    "exposed_records": planned_exposure[
                        "logical_exposure"
                    ]["record_occurrences"],
                    "exposed_facts": planned_exposure[
                        "logical_exposure"
                    ]["unique_fact_count"],
                    "all_eligible_train_rows_exposed": planned_exposure[
                        "logical_exposure"
                    ][
                        "all_eligible_train_rows_exposed"
                    ],
                    "manifest_sha256": planned_exposure["manifest_sha256"],
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
    supervision = assert_s0_supervision(dataset, collator)
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
        _WANDB_RUN.config.update(
            {
                "training_examples": len(dataset),
                "tokenizer_artifacts_sha256": tokenizer_identity_sha256,
                "completion_preflight_examples": supervision["examples_checked"],
                "mapping_target_preflight_tokens": supervision[
                    "mapping_target_tokens"
                ],
                "validation_examples": len(validation_dataset),
                "validation_completion_preflight_examples": (
                    validation_supervision["examples_checked"]
                ),
            },
            allow_val_change=True,
        )

    # Load the frozen base model with its native TP/EP plan.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        tp_plan="auto",
        device_mesh=tp_mesh,
    )
    expert_modules = [
        module for name, module in model.named_modules() if name.endswith(".mlp.experts")
    ]
    if len(expert_modules) != model.config.num_hidden_layers or any(
        not getattr(module, "_mentor_autograd_expert_gather", False)
        for module in expert_modules
    ):
        raise RuntimeError(
            "Native GPT-OSS expert modules did not receive the autograd-aware EP input/output pair"
        )
    router_modules = [
        module for name, module in model.named_modules() if name.endswith(".mlp.router")
    ]
    if len(router_modules) != model.config.num_hidden_layers or any(
        not getattr(module, "_mentor_autograd_ep_router", False)
        for module in router_modules
    ):
        raise RuntimeError(
            "Native GPT-OSS router modules did not receive the autograd-aware EP input boundary"
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # Add attention LoRA after the native plan places every base layer.
    injected = inject_tp_lora(
        model, LoraSpec(args.lora_r, args.lora_alpha, args.lora_dropout), args.seed
    )
    token_adapter = None
    token_adapter_forward_preflight = None
    # Add token-row deltas only for a custom S0 tokenizer.
    if args.token_adapter_manifest:
        token_adapter = install_trainable_token_rows(
            model, Path(args.token_adapter_manifest)
        )
        token_adapter_forward_preflight = assert_token_adapter_forward_contract(model)
    trainable = _adapter_parameters(model)
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "model_preflight",
                    "lora_modules": len(injected),
                    "trainable_tensors": len(trainable),
                    "trainable_global_parameters": sum(parameter.numel() for _, parameter in trainable),
                    "targets": list(ATTENTION_LORA_TARGETS),
                    "token_adapter": token_adapter,
                    "token_adapter_forward_preflight": (
                        token_adapter_forward_preflight
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # The optimizer receives LoRA and optional token-row parameters only.
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
    )
    warmup_steps = round(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if rank == 0:
        _WANDB_RUN.config.update(
            {"effective_total_steps": total_steps, "warmup_steps": warmup_steps},
            allow_val_change=True,
        )
    # Restore all adapter, optimizer, scheduler, RNG, and loop state together.
    resume = resolve_resume(args.resume_from_checkpoint, output_dir)
    if resume is not None:
        state = load_checkpoint(
            model,
            optimizer,
            scheduler,
            resume,
            topology,
            rank,
            args.loss_contract,
            args.token_adapter_manifest,
            args.tokenizer_artifacts_sha256,
        )
        if rank == 0:
            print(f"Resume S0 LoRA from {resume} at step {state.global_step}", flush=True)
    else:
        state = LoopState()
    if state.global_step > total_steps:
        raise RuntimeError(
            f"Checkpoint step {state.global_step} exceeds requested total steps {total_steps}"
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "training_start",
                    "dataset_rows": len(dataset),
                    "batches_per_epoch": batches_per_epoch,
                    "updates_per_epoch": updates_per_epoch,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                    **loss_contract_config(args.loss_contract),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if args.eval_on_start:
        evaluate_validation(
            model,
            validation_dataset,
            collator,
            args,
            topology,
            rank=rank,
            dp_rank=dp_rank,
            dp_group=dp_group,
            global_step=state.global_step,
            epoch=state.epoch,
            trigger="train_start",
            output_dir=output_dir,
            wandb_run=_WANDB_RUN if rank == 0 else None,
        )
    if state.global_step < total_steps:
        state = train_loop(
            model,
            dataset,
            validation_dataset,
            collator,
            optimizer,
            scheduler,
            state,
            total_steps,
            args,
            topology,
            output_dir,
            rank,
            dp_rank,
            dp_group,
            _WANDB_RUN if rank == 0 else None,
        )
    # Save the final inference adapter and the complete resume state.
    save_checkpoint(
        model, optimizer, scheduler, state, output_dir, args, topology, rank, final=True
    )
    if rank == 0:
        # Replace the planned receipt with the exact completed receipt.
        completed_exposure = build_training_exposure_manifest(
            **exposure_manifest_args,
            status="complete",
            completed_global_step=state.global_step,
        )
        _write_json(exposure_path, completed_exposure)
        print(
            json.dumps(
                {
                    "event": "training_exposure_completed",
                    "manifest_sha256": completed_exposure["manifest_sha256"],
                    "all_eligible_train_rows_exposed": completed_exposure[
                        "logical_exposure"
                    ]["all_eligible_train_rows_exposed"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        tokenizer.save_pretrained(output_dir)
        print(f"S0 LoRA finished at step {state.global_step}: {output_dir}", flush=True)
        _WANDB_RUN.summary["completed_optimizer_steps"] = state.global_step
        _WANDB_RUN.summary["run_status"] = "completed"
        _WANDB_RUN.finish(exit_code=0)
        _WANDB_RUN = None
    dist.barrier()
    # Rank 0 owns the TCPStore. Other ranks close their process groups first.
    # This order prevents false TCPStore errors after success.
    if rank == 0 and topology.world_size > 1:
        time.sleep(float(os.environ.get("MENTOR_RANK0_STORE_SETTLE_SECONDS", "5")))
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if _WANDB_RUN is not None:
            _WANDB_RUN.summary["run_status"] = "failed"
            _WANDB_RUN.summary["error_type"] = type(exc).__name__
            _WANDB_RUN.summary["error_message"] = str(exc)[:2000]
            _WANDB_RUN.finish(exit_code=1)
            _WANDB_RUN = None
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        raise
