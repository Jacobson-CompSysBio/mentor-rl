"""Define topology and startup contracts for GPT-OSS DP x TP/EP SFT.

The installed Accelerate 1.10.1 stack can train a native Transformers tensor-
parallel model, but it cannot compose *pure* data parallelism with tensor
parallelism. These checks do not import torch. The launcher can reject an
invalid allocation before it loads model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# Block 1 pins the software stack and the native GPT-OSS tensor plan.
TESTED_STACK = {
    "accelerate": "1.10.1",
    "peft": "0.15.2",
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "trl": "0.24.0",
}

ATTENTION_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

REQUIRED_GPT_OSS_TP_STYLES = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.experts": "gather",
    "layers.*.mlp.router": "ep_router",
    "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
    "layers.*.mlp.experts.down_proj": "grouped_gemm",
}


@dataclass(frozen=True)
class WorldModelTopology:
    """Define the ranks and effective batch dimensions for TP/EP."""

    world_size: int
    local_world_size: int
    node_count: int
    tp_size: int
    ep_size: int
    per_replica_batch_size: int
    gradient_accumulation_steps: int

    @property
    def data_parallel_size(self) -> int:
        return self.world_size // self.tp_size

    @property
    def global_batch_size(self) -> int:
        # All TP ranks use the same examples.
        # Only independent model replicas increase the global batch size.
        return (
            self.per_replica_batch_size
            * self.gradient_accumulation_steps
            * self.data_parallel_size
        )

    def validate(self) -> "WorldModelTopology":
        # Reject invalid positive dimensions before any process group starts.
        positive = {
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "tp_size": self.tp_size,
            "ep_size": self.ep_size,
            "per_replica_batch_size": self.per_replica_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.world_size % self.tp_size:
            raise ValueError(
                f"WORLD_SIZE={self.world_size} is not divisible by tp_size={self.tp_size}"
            )
        if self.ep_size != self.tp_size:
            raise ValueError(
                "Transformers 4.57.1 GPT-OSS uses one native device mesh for "
                f"attention TP and expert sharding; ep_size ({self.ep_size}) must "
                f"equal tp_size ({self.tp_size})."
            )
        if self.local_world_size != self.tp_size:
            raise ValueError(
                "Every node must contain exactly one complete TP/EP replica: "
                f"LOCAL_WORLD_SIZE={self.local_world_size}, tp_size={self.tp_size}."
            )
        if self.node_count != self.data_parallel_size:
            raise ValueError(
                "Expected one TP/EP replica per node: "
                f"node_count={self.node_count}, data_parallel_size={self.data_parallel_size}."
            )
        return self


def validate_gpt_oss_dimensions(config: object, tp_size: int) -> None:
    """Reject a GPT-OSS config that cannot use the requested native plan."""

    if getattr(config, "model_type", None) != "gpt_oss":
        raise ValueError(
            f"S0 TP/EP requires model_type='gpt_oss', got {getattr(config, 'model_type', None)!r}"
        )
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config:
        raise ValueError(
            "S0 TP/EP requires the BF16 GPT-OSS checkpoint; quantized "
            "MXFP4 training and adapter gradients are not validated."
        )
    dimensions = {
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_local_experts": getattr(config, "num_local_experts", None),
    }
    for name, value in dimensions.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"GPT-OSS config has invalid {name}={value!r}")
        if value % tp_size:
            raise ValueError(f"{name}={value} is not divisible by tp_size={tp_size}")


def validate_native_tp_plan(plan: Mapping[str, str]) -> None:
    """Ensure the installed model plan includes attention TP and expert sharding."""

    missing_or_wrong = {
        key: {"expected": style, "actual": plan.get(key)}
        for key, style in REQUIRED_GPT_OSS_TP_STYLES.items()
        if plan.get(key) != style
    }
    if missing_or_wrong:
        raise RuntimeError(
            "The GPT-OSS native TP plan is incompatible with S0: "
            f"{missing_or_wrong}"
        )


def parse_release(version: str) -> tuple[int, ...]:
    """Return the numeric release prefix (e.g. ``2.8.0+rocm`` -> ``(2,8,0)``)."""

    numeric = version.split("+", 1)[0].split("-", 1)[0]
    try:
        return tuple(int(part) for part in numeric.split("."))
    except ValueError as exc:
        raise ValueError(f"Could not parse package release {version!r}") from exc


def validate_tested_stack(versions: Mapping[str, str]) -> None:
    """Fail closed outside the local API versions inspected for this entry point."""

    errors = {}
    for package, tested in TESTED_STACK.items():
        actual = versions.get(package)
        if actual is None or parse_release(actual) != parse_release(tested):
            errors[package] = {"tested": tested, "actual": actual}
    if errors:
        raise RuntimeError(
            "TP/EP package versions differ from the inspected stack; review "
            f"native plans and checkpoint behavior before training: {errors}"
        )
