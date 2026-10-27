"""Load one parent adapter into a new tensor-parallel run under torchrun."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import train_sft_dp_tp_ep as trainer  # noqa: E402

HIDDEN = 64
KV = 16
LAYERS = 2
SPEC = trainer.LoraSpec(4, 8, 0.0)
TOKENIZER_ARTIFACTS = "a" * 64


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        options = {"bias": False, "dtype": torch.bfloat16}
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, **options)
        self.k_proj = nn.Linear(HIDDEN, KV, **options)
        self.v_proj = nn.Linear(HIDDEN, KV, **options)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, **options)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        self.config = SimpleNamespace(num_hidden_layers=LAYERS)


def build_model(mesh, seed: int) -> TinyModel:
    """Build the same native TP layout that the trainer receives."""

    torch.manual_seed(0)
    model = TinyModel().cuda()
    plan = {}
    for layer in range(LAYERS):
        for name in ("q_proj", "k_proj", "v_proj"):
            plan[f"layers.{layer}.self_attn.{name}"] = ColwiseParallel()
        plan[f"layers.{layer}.self_attn.o_proj"] = RowwiseParallel()
    parallelize_module(model, mesh, plan)
    model._device_mesh = mesh
    model._tp_size = mesh.size()
    trainer.inject_tp_lora(model, SPEC, seed)
    return model


def main() -> None:
    output = Path(sys.argv[1])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    for environment_name in trainer.S0_IDENTITY_ENV.values():
        os.environ[environment_name] = f"test-{environment_name.lower()}"
    identity = trainer.required_run_identity()

    # Write a parent adapter whose tensors differ from a fresh LoRA start.
    model = build_model(mesh, seed=1)
    generator = torch.Generator().manual_seed(1234)
    parent = {
        trainer._peft_key(name): torch.randn(
            tuple(parameter.shape), generator=generator
        ).to(torch.bfloat16)
        for name, parameter in trainer._lora_parameters(model)
    }
    adapter_path = output / trainer.ADAPTER_WEIGHTS_NAME
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_file(parent, adapter_path)
        (output / trainer.TP_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "format": "mentor-rl-s0-tp-lora-v1",
                    "lora": asdict(SPEC),
                    "identity": {
                        **identity,
                        "tokenizer_artifacts_sha256": TOKENIZER_ARTIFACTS,
                    },
                }
            ),
            encoding="utf-8",
        )
    dist.barrier()
    args = SimpleNamespace(
        lora_r=SPEC.r,
        lora_alpha=SPEC.alpha,
        lora_dropout=SPEC.dropout,
        init_adapter_dir=str(output),
        init_adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        init_adapter_run_id=identity["run_id"],
        tokenizer_artifacts_sha256=TOKENIZER_ARTIFACTS,
    )

    trainer.initialize_adapter_from_checkpoint(model, output, args)

    # Each rank holds only its slice, and the gathered tensors equal the parent.
    sharded = 0
    for name, parameter in trainer._lora_parameters(model):
        gathered = parameter.full_tensor().detach().cpu()
        if not torch.equal(gathered, parent[trainer._peft_key(name)]):
            raise RuntimeError(f"Parent tensor changed for {name}")
        if parameter.to_local().numel() < parameter.numel():
            sharded += 1
    if sharded == 0:
        raise RuntimeError("No LoRA tensor was sharded across the TP group")

    # A changed adapter hash must stop the load before any tensor changes.
    fresh = build_model(mesh, seed=2)
    try:
        trainer.initialize_adapter_from_checkpoint(
            fresh,
            output,
            SimpleNamespace(**{**vars(args), "init_adapter_sha256": "0" * 64}),
        )
    except RuntimeError as error:
        if "differs from the launch contract" not in str(error):
            raise
    else:
        raise RuntimeError("A changed parent adapter hash was accepted")

    dist.barrier()
    if rank == 0:
        print(f"ADAPTER_INIT_PARITY_OK sharded_tensors={sharded}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
