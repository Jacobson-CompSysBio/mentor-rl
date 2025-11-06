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
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return _fmt

def infer(model, tokenizer, format, input):
    results = []
    with torch.no_grad():
        for example in input:
            formatted = format(example)
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]
            
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            new_ids = output[0][input_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)
    return results

def main():
    # extract args from classes
    parser = HfArgumentParser((ScriptArguments, PeftArguments, SftTrainingArguments))
    script_args, peft_args, training_args = parser.parse_args_into_dataclasses()

    # make run name
    training_args.run_name = make_run_name(script_args, peft_args, training_args, slurm_args)
    training_args.optim = "adamw_torch_fused"
    training_args.gradient_checkpointing = True

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    formatting_func = build_formatting_func(tokenizer)

    # load model (attn is sdpa)
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
    
    # get rank, world size for distributed
    rank = get_rank()
    world_size = get_world_size()

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
        print(f"Using the full dataset with {len(dataset)} samples.")

    print("[dbg] after subset", flush=True)

    dataset = dataset.shuffle(seed=training_args.seed)
    print(f"Dataset shuffled with seed: {training_args.seed}.")
    
    print("[dbg] after shuffle", flush=True)

    # if using multiple GPUs, shard the dataset
    if world_size > 1:
        print(f"Sharding dataset for Rank {rank} of {world_size}.")
        dataset = dataset.shard(num_shards=world_size, index=rank)
    
    print("[dbg] after shard", flush=True)
    
    # perform pre-training inference
    print(f"\n[rank {rank}] Performing initial inference...\n", flush=True)
    inference_set = dataset.select(range(20))
    format_infer = build_formatting_func(tokenizer, train=False)

    ds_inf_model = deepspeed.init_inference(model, dtype=torch.bfloat16).module
    if rank == 0:
        print("[dbg] max emb:", ds_inf_model.config.max_position_embeddings, flush=True)
    outputs = infer(ds_inf_model, tokenizer, format_infer, inference_set)
    print(f"\n[rank {rank}] Initial inference results:", check_accuracy(outputs, list(inference_set["answer"])), "\n", flush=True)
    del ds_inf_model

    # if rank == 0:
    #     print("\nPerforming initial inference...\n", flush=True)
    #     torch.cuda.empty_cache()
    #     model = AutoModelForCausalLM.from_pretrained(
    #         script_args.model_path,
    #         torch_dtype=torch.bfloat16,
    #     )
    #     model.eval()
        
    #     inference_set = dataset.select(range(20))
    #     format_infer = build_formatting_func(tokenizer, train=False)
        
    #     outputs = infer(model, tokenizer, format_infer, inference_set)
    #     print("\nInitial inference results:", check_accuracy(outputs, list(inference_set["answer"])), "\n")
    #     del model
    #     torch.distributed.barrier()
    # else:
    #     print(f"[rank {rank}] Waiting for completion of initial inference...")
    #     torch.distributed.barrier()

    # init and run trainer
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

    print(f"[rank {rank}] Trainer initialized successfully, beginning training...")
    trainer.train()

    print(f"[rank {rank}] Training finished, saving model...")
    trainer.save_model(training_args.output_dir)

    # perform post-training inference
    print(f"\n[rank {rank}] Performing post-training inference...\n", flush=True)
    del trainer
    # del model
    torch.cuda.empty_cache()

    # model = AutoModelForCausalLM.from_pretrained(
    #     script_args.model_path,
    #     torch_dtype=torch.bfloat16,
    # )
    # model = PeftModel.from_pretrained(model, training_args.output_dir)

    ds_inf_model = deepspeed.init_inference(model, dtype=torch.bfloat16).module
    outputs = infer(ds_inf_model, tokenizer, format_infer, inference_set)
    print(f"\n[rank {rank}] Final inference results:", check_accuracy(outputs, list(inference_set["answer"])), "\n", flush=True)

    # if rank  == 0:
    #     print("\nPerforming post-training inference...\n")
    #     del trainer
    #     del model
    #     torch.cuda.empty_cache()

    #     model = AutoModelForCausalLM.from_pretrained(
    #         script_args.model_path,
    #         torch_dtype=torch.bfloat16,
    #     )
    #     model = PeftModel.from_pretrained(model, training_args.output_dir)
    #     # model = model.merge_and_unload()
    #     model.eval()

    #     outputs = infer(model, tokenizer, format_infer, inference_set)
    #     print("\nFinal inference results:", check_accuracy(outputs, list(inference_set["answer"])), "\n")
    #     torch.distributed.barrier()
    # else:
    #     print(f"[rank {rank}] Waiting for completion of final inference...")
    #     torch.distributed.barrier()

if __name__ == "__main__":
    main()