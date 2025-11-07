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
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from transformers.models.llama4.modeling_llama4 import Llama4TextDecoderLayer
from trl import SFTTrainer
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import * 

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

@dataclass
class PeftArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})

@dataclass
class SftTrainingArguments(TrainingArguments):
    max_length: Optional[int] = field(default=2048, metadata={"help": "The maximum sequence length for SFTTrainer"})
    packing: Optional[bool] = field(default=False, metadata={"help": "Enable packing for SFTTrainer"})
    ddp_find_unused_parameters: Optional[bool] = field(default=True, metadata={"help": "When using FSDP activation checkpointing, this must be set to True"})

def build_formatting_func(tokenizer, train=True):
    SYSTEM_PROMPT = (
        "You are a helpful biological chatbot. You will be given a biological question; "
        "return the correct answer."
    )
    def _fmt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ] if train else [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        if train:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _fmt

def infer_with_zero3_gather(model, tokenizer, format, input):
    """
    Inference function that properly handles DeepSpeed ZeRO-3 partitioned weights.
    Uses deepspeed.zero.GatheredParameters context manager to gather weights for generation.
    """
    results = []
    
    with torch.no_grad():
        for example in input:
            formatted = format(example)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            
            # CRITICAL: Gather all model parameters for generation with ZeRO-3
            # This temporarily materializes the full model on each rank
            with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=None):
                output = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_ids = output[0][input_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)
    
    return results

def main():
    # extract args from classes
    parser = HfArgumentParser((ScriptArguments, PeftArguments, SftTrainingArguments))
    script_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    # get rank, world size for distributed
    rank = get_rank()
    world_size = get_world_size()

    # make run name
    training_args.run_name = make_run_name(script_args, peft_args, training_args, slurm_args)
    training_args.optim = "adamw_torch_fused"
    training_args.gradient_checkpointing = True

    # load tokenizer and ensure padding is set
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    formatting_func = build_formatting_func(tokenizer)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    model.use_cache = False
    model.config.use_cache = False
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    # set up peft
    peft_config = LoraConfig(
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    # load dataset
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # Subset dataset if specified
    if script_args.dataset_subset_size is not None:
        dataset = dataset.select(range(script_args.dataset_subset_size))
    else:
        if rank == 0:
            print(f"Using the full dataset with {len(dataset)} samples.")
    
    dataset = dataset.shuffle(seed=training_args.seed)
    if rank == 0:
        print(f"Dataset shuffled with seed: {training_args.seed}.")
    
    # if using multiple GPUs, shard the dataset
    if world_size > 1:
        if rank == 0:
            print(f"Sharding dataset for {world_size} ranks.")
        dataset = dataset.shard(num_shards=world_size, index=rank)

    # init trainer FIRST to initialize DeepSpeed
    if rank == 0:
        print(f"[rank {rank}] Initializing SFTTrainer...")
    training_args.report_to = ["wandb"]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    if rank == 0:
        print(f"[rank {rank}] Trainer initialized successfully.")
    
    # NOW perform pre-training inference with DeepSpeed-initialized model
    if rank == 0:
        print(f"\n[rank {rank}] Performing initial inference with ZeRO-3...\n", flush=True)
    
    inference_set = dataset.select(range(min(1, len(dataset))))
    format_infer = build_formatting_func(tokenizer, train=False)
    
    # Use the DeepSpeed-wrapped model from trainer
    trainer.model.eval()
    outputs = infer_with_zero3_gather(trainer.model, tokenizer, format_infer, inference_set)
    
    if rank == 0:
        print(f"\n[Rank {rank}] Initial inference results:", check_accuracy(
            outputs, list(inference_set["answer"])), "\n", flush=True)
    
    trainer.model.train()
    
    # Synchronize all ranks
    if world_size > 1:
        torch.distributed.barrier()

    # Run training
    if rank == 0:
        print(f"[rank {rank}] Beginning training...")
    trainer.train()

    if rank == 0:
        print(f"[rank {rank}] Training finished, saving model...")
    trainer.save_model(training_args.output_dir)

    # perform post-training inference with ZeRO-3
    if rank == 0:
        print(f"\n[Rank {rank}] Performing post-training inference with ZeRO-3...\n", flush=True)
    
    trainer.model.eval()
    outputs = infer_with_zero3_gather(trainer.model, tokenizer, format_infer, inference_set)
    
    if rank == 0:
        print(f"\n[Rank {rank}] Final inference results:", check_accuracy(outputs, list(inference_set["answer"])), "\n", flush=True)

    # Synchronize all ranks before exit
    if world_size > 1:
        torch.distributed.barrier()

if __name__ == "__main__":
    main()