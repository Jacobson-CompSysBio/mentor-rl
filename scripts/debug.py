import argparse
import os
from typing import Optional, Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

### wandb setup ###
load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
ds_cfg = HfDeepSpeedConfig("ds_zero3.json")

def parse_args():
    p = argparse.ArgumentParser(description="Debug TRL SFTTrainer for multinode w/ slurm")

    # required
    p.add_argument("--model_path", type=str, required=True,
                   help="Local path to the base model.")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to the dataset (json file).")    
    # optional
    p.add_argument("--output_dir", type=str, default="./checkpoints")

    # dataset options
    p.add_argument("--text_field", type=str, default=None,
                   help="Name of a single text field in your JSON (e.g., 'text'). "
                        "If provided, we train directly on this field.")
    p.add_argument("--prompt_field", type=str, default=None,
                   help="If you don't have a single text field, provide a prompt field name (e.g., 'prompt').")
    p.add_argument("--response_field", type=str, default=None,
                   help="If you don't have a single text field, provide a response field name (e.g., 'response').")
    p.add_argument("--max_length", type=int, default=256,
                   help="Maximum sequence length.")
    p.add_argument("--seed", type=int, default=42)
    
    # training hyperparameters
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine",
                   choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_strategy", type=str, default="epoch",
                   choices=["no", "epoch", "steps"])
    p.add_argument("--evaluation_strategy", type=str, default="epoch",
                   choices=["no", "epoch", "steps"])
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable gradient checkpointing to save memory.")
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 precision if available.")

    # performance/data
    p.add_argument("--dataset_num_proc", type=int, default=4,
                   help="Number of processes to use for dataset loading.")
    p.add_argument("--packing", action="store_true",
                   help="Enable sequence packing.")
    
    # attention kernels
    p.add_argument("--attn_imp", type=str, default=None,
                   choices=[None, "flash_attention_2", "sdpa", "eager"],
                   help="Attention implementation to use.")
    
    return p.parse_args()

def build_formatting_func(tokenizer, args):
    """
    Returns a formatting function if the dataset does not provide a single 'text' field.
    Tries, in order:
        1. If a sample has 'messages' (chat-style), use tokenizer.apply_chat_template
        2. If prompt/response fields exist, concatenate simply as: prompt + response
    """
    prompt_field = args.prompt_field
    response_field = args.response_field

    if args.text_field:
        return None # Let SFTTrainer read it directly
    
    def _fmt(example: Dict[str, Any]) -> str:
        # case 1: chat-style messages
        if "messages" in example and isinstance(example["messages"], (list, tuple)):
            try:
                return tokenizer.apply_chat_template(
                                                     example["messages"],
                                                     tokenize=False,
                                                     add_generation_prompt=False,
                                                     )
            except Exception:
                # fallback to naive join if template isn't available
                parts = []
                for m in example["messages"]:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    parts.append(f"[{role.upper()}]: {content}")
                return "\n".join(parts)
        # case 2: prompt/response fields
        if prompt_field and response_field and (prompt_field in example) and (response_field in example):
            prompt = example[prompt_field] if example[prompt_field] is not None else ""
            response = example[response_field] if example[response_field] is not None else ""
            return f"{prompt}{response}"

        
        # if none of the above, try common fallbacks
        for k in "text", "completion", "output":
            if k in example and isinstance(example[k], str):
                return example[k]
        
        raise ValueError(
            "Could not format example. Provide --text_field or --prompt_field and --response_field."
        )
    return _fmt

def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=True,
    )

    # many llama-style tokenizers don't have a pad token by default, so set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if args.attn_imp is not None:
        model_kwargs["attn_implementation"] = args.attn_imp
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    # gradient checkpointing and use_cache are usually mutually exclusive
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # load dataset
    data_files = {"train": args.data_path}
    ds = load_dataset("json", data_files=data_files)

    # build formatting function
    formatting_func = build_formatting_func(tokenizer, args)
    dataset_text_field = args.text_field if args.text_field else None

    # simple LM collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=["wandb"],
    )

    # build SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=collator,
        max_seq_length=args.max_length,
        packing=args.packing,
        dataset_text_field=dataset_text_field,
        formatting_func=formatting_func,
        dataset_num_proc=args.dataset_num_proc,
    )

    # train and save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)  
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print(f"Training complete. Model saved to: {args.output_dir}")

if __name__ == "__main__":
   main() 