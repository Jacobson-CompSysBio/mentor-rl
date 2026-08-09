"""Define autograd-safe expert-parallel communication operations.

Expert parallelism partitions the experts and keeps the hidden state replicated.
Correct derivatives require a matched pair of communication operations.

``copy_to_expert_parallel_region`` returns its input in the forward pass. Its
backward pass sums gradients across the expert group.

``reduce_from_expert_parallel_region`` sums local expert outputs in the forward
pass. Its backward pass returns the gradient without another sum.

This pair follows the Megatron model-parallel convention. An autograd
all-reduce for the output would sum the same gradients twice.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _require_distributed(group: Any) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Expert-parallel communication requires an initialized "
            "torch.distributed process group."
        )
    # Resolve the group here. An autograd callback can hide a group error.
    dist.get_world_size(group=group)


class _CopyToExpertParallelRegion(torch.autograd.Function):
    """Forward identity; backward sum over the expert-parallel group."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        _require_distributed(group)
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Do not change storage that another autograd node owns.
        # ``contiguous`` can return its input, so make an explicit clone.
        reduced_gradient = gradient.contiguous().clone()
        dist.all_reduce(
            reduced_gradient,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return reduced_gradient, None


class _ReduceFromExpertParallelRegion(torch.autograd.Function):
    """Forward sum over the expert-parallel group; backward identity."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Any) -> torch.Tensor:
        _require_distributed(group)
        # Keep this collective outside autograd.
        # The custom backward method supplies the identity half of the pair.
        reduced_output = tensor.contiguous().clone()
        dist.all_reduce(
            reduced_output,
            op=dist.ReduceOp.SUM,
            group=group,
        )
        return reduced_output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


def copy_to_expert_parallel_region(
    tensor: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """Copy a replicated hidden state into the local expert computation.

    The forward value stays unchanged. The backward pass sums local derivatives
    so the prior replicated layer receives the complete MoE derivative.
    """

    return _CopyToExpertParallelRegion.apply(tensor, group)


def reduce_from_expert_parallel_region(
    tensor: torch.Tensor,
    group: Any = None,
) -> torch.Tensor:
    """Sum local expert contributions and replicate the logical MoE output.

    The backward pass returns its gradient. The paired input operation performs
    the required reduction for prior layers.
    """

    return _ReduceFromExpertParallelRegion.apply(tensor, group)


__all__ = [
    "copy_to_expert_parallel_region",
    "reduce_from_expert_parallel_region",
]
