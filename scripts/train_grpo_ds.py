### SCRIPT FROM: https://cloud.google.com/ai-hypercomputer/docs/tutorials/fsdp-llama4 ###
import os, sys
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)

from torch.distributed import get_rank, get_world_size

from transformers.models.llama4.modeling_llama4 import Llama4TextDecoderLayer
from trl import GRPOTrainer, GRPOConfig
from dataclasses import dataclass, field
from typing import Optional, Union
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import *
from utils.rewards import format_reward, correctness_reward

### wandb logging ###
load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

nnodes = int(os.environ.get("SLURM_NNODES", 1))
timeout = int(os.environ.get("SLURM_JOB_TIMEOUT", 0))
slurm_args = argparse.Namespace(nnodes=nnodes, timeout=timeout)

@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Hugging Face model ID from the Hub"})
    dataset_path: str = field(default="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/qa_pairs.json", metadata={"help": "Local dataset path"})
    run_inference_after_training: bool = field(default=False, metadata={"help": "Run sample inference on rank 0 after training"})
    dataset_subset_size: Optional[int] = field(default=None, metadata={"help": "Number of samples to use from the dataset for training. If None, uses the full dataset."})

    # grpo args
    max_completion_length: int = field(default=256, metadata={"help": "Max. generated tokens"})
    num_generations: int = field(default=8, metadata={"help": "GRPO group size per prompt"})

    # grpo: formatting reward
    max_think_chars: int = field(default=4000, metadata={"help": "Soft cap for <think> length"})

    # grpo: weights
    w_format: float = field(default=0.3, metadata={"help": "Weight for format reward"})
    w_task: float = field(default=1.0, metadata={"help": "Weight for domain reward"})

    # logging/output
    output_dir: str = field(default="runs/grpo-format", metadata={"help": "Output dir"})

@dataclass
class PeftArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})

@dataclass
class GrpoTrainingArguments:
    # Core trainer knobs (needed by your run-name function)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    output_dir: str = "runs/grpo-format"

    # Stability / logging
    seed: int = 900913
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    report_to: Union[str, List[str]] = "wandb"

    # GRPO specifics
    num_generations: int = 8
    max_completion_length: int = 256
    loss_type: str = "dapo"

    # Runtime / precision
    bf16: bool = True
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True
    dataloader_drop_last: bool = True
    deepspeed: Optional[str] = None  # path to ds_zero3.json

    # Optional extras
    warmup_steps: int = 0
    weight_decay: float = 0


def apply_chat_prompt(tokenizer):
    SYSTEM_PROMPT = (
        "You are a helpful biological chatbot. You will be given a biological question; "
        "return the final result wrapped in <answer></answer> and enclose any reasoning in <think></think>."
    )
    def _fmt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        # RL doesn't have answer included; instead, generation prompt is applies
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _fmt



def main():
    # extract args from classes
    parser = HfArgumentParser((ScriptArguments, PeftArguments, GrpoTrainingArguments))
    script_args, peft_args, grpo_args = parser.parse_args_into_dataclasses()

    # make run name
    training_args.run_name = make_run_name(script_args, peft_args, training_args, slurm_args)
    training_args.optim = "adamw_torch_fused"
    training_args.gradient_checkpointing = True

    # grpo args
    grpo_args.remove_unused_columns = False
    grpo_args.num_generations = script_args.num_generations
    grpo_args.max_completion_length = script_args.max_completion_length
    if not getattr(grpo_args, "loss_type", None):
        grpo_args.loss_type = "dapo" # good for loss stability 
    grpo_args.reward_weights = [script_args.w_format, script_args.w_task]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>", "<answer>", "</answer>"]})

    # load model (attn is sdpa)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # set up peft
    peft_config = LoraConfig(
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    
    # get rank, world size for distributed
    rank = get_rank()
    world_size = get_world_size()

    # load dataset
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")
    if script_args.dataset_subset_size is not None:
        dataset = dataset.select(range(script_args.dataset_subset_size))

    # keep gt answer in the row for correctness reward, but do not include it in the prompt text
    def _map(example):
        example["prompt"] = apply_chat_prompt(tokenizer, example["question"])
        return example
    dataset = dataset.map(_map, remove_columns=[])

    # rewards
    def format_reward_wrapper(completions, **kwargs):
        return format_reward(completions, max_think_chars=script_args.max_think_chars, **kwargs) 
    reward_funcs = [format_reward_wrapper, correctness_reward]

    # if using multiple GPUs, shard the dataset
    if world_size > 1:
        print(f"Sharding dataset for Rank {rank} of {world_size}.")
        dataset = dataset.shard(num_shards=world_size, index=rank)

    # init and run trainer
    print("Initializing GRPOTrainer...")
    report_to = grpo_args.report_to if isinstance(grpo_args.report_to, list) else [grpo_args.report_to]
    run_name = make_grpo_run_name(script_args, peft_args, grpo_args, slurm_args)

    trl_args = GRPOConfig(
        # basics
        output_dir=grpo_args.output_dir,
        run_name=run_name,
        seed=grpo_args.seed,
        report_to=report_to,

        # optimizer / schedule
        learning_rate=grpo_args.learning_rate,
        weight_decay=grpo_args.weight_decay,
        warmup_steps=grpo_args.warmup_steps,

        # batching
        per_device_train_batch_size=grpo_args.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_args.gradient_accumulation_steps,
        dataloader_drop_last=grpo_args.dataloader_drop_last,

        # logging / ckpt
        logging_steps=grpo_args.logging_steps,
        save_steps=grpo_args.save_steps,
        save_total_limit=grpo_args.save_total_limit,

        # runtime
        bf16=grpo_args.bf16,
        remove_unused_columns=grpo_args.remove_unused_columns,
        gradient_checkpointing=grpo_args.gradient_checkpointing,
        deepspeed=grpo_args.deepspeed,

        # GRPO specifics
        num_generations=grpo_args.num_generations,
        max_completion_length=grpo_args.max_completion_length,
        loss_type=grpo_args.loss_type,

        # epochs
        num_train_epochs=grpo_args.num_train_epochs,
    )
    # reward weights (order must match reward_funcs)
    trl_args.reward_weights = [script_args.w_format, script_args.w_task]

    trainer = GRPOTrainer(
        model=model,
        args=trl_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
        tokenizer=tokenizer,
        prompt_column="prompt",
        stop_sequences=["</answer>"]
    )

    trainer.train()
    trainer.save_model(script_args.output_dir)

if __name__ == "__main__":
    main()