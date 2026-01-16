import os, sys
import torch
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)

from accelerate import PartialState

from trl import SFTTrainer
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference import build_formatting_func, infer
from utils.utils import * 

### SLURM VARIABLES ###
load_dotenv()
os.environ["WANDB_PROJECT"] = "mentor-sft"
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

nnodes = int(os.environ.get("SLURM_NNODES", 1))
timeout = int(os.environ.get("SLURM_JOB_TIMEOUT", 0))
slurm_args = argparse.Namespace(nnodes=nnodes, timeout=timeout)

### ARGS ###
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

def main():

    ############################
    # MODEL LOADING / PRE-REQS #
    ############################
    # get rank, world_size
    state = PartialState()
    rank = state.process_index

    # extract args from classes
    parser = HfArgumentParser((ScriptArguments, PeftArguments, SftTrainingArguments))
    script_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    # make run name
    training_args.run_name = make_run_name(script_args, peft_args, training_args, slurm_args)
    training_args.optim = "adamw_torch_fused"
    training_args.gradient_checkpointing = True

    # load tokenizer and fix padding token
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Verify pad_token_id is within vocabulary range
    if tokenizer.pad_token_id >= tokenizer.vocab_size:
        if rank == 0:
            print(f"[WARNING] pad_token_id ({tokenizer.pad_token_id}) >= vocab_size ({tokenizer.vocab_size}), setting to eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    formatting_func = build_formatting_func(tokenizer)

    # load model (attn is eager for compatibility)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    model.use_cache = False  # needed for gradient checkpointing
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

    ###################    
    # DATASET LOADING #
    ###################

    # load dataset
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # Subset dataset if specified
    if script_args.dataset_subset_size is not None:
        dataset = dataset.select(range(script_args.dataset_subset_size))
    dataset = dataset.shuffle(seed=training_args.seed)

    # Slice a subset for post-training evaluation if requested
    if script_args.run_inference_after_training:
        sample_size = min(20, len(dataset))
        inf_ds = dataset.select(range(sample_size))
        inf_format = build_formatting_func(tokenizer, train=False)

    ############    
    # TRAINING #
    ############    

    if rank == 0:
        print(f"Initializing trainer...")
    # init and run trainer
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
        print(f"Trainer initialized successfully, beginning training...") 
    # Run training
    trainer.train()

    if rank == 0:
        print(f"Training complete. Saving...")
    trainer.save_model(training_args.output_dir)
    if rank == 0:
        print(f"Model saved to {training_args.output_dir}")

    if script_args.run_inference_after_training:
        ###########################   
        # POST TRAINING INFERENCE #
        ###########################
        if rank == 0:
            print("Running post-training inference...")
        post_outputs = infer(
            trainer.model,
            tokenizer,
            inf_format,
            inf_ds,
            trainer.accelerator
        )
        post_score = check_accuracy(post_outputs, list(inf_ds["answer"]))
        if isinstance(post_score, list):
            post_score = np.mean(post_score)
 
        if rank == 0:
            print(f"Post-training inference complete. Average Score={post_score:.2%}")
            print("Outputs:", post_outputs)

if __name__ == "__main__":
    main()
