### SCRIPT FROM: https://cloud.google.com/ai-hypercomputer/docs/tutorials/fsdp-llama4 ###
import os, sys
import torch
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
    # num_train_epochs: Optional[float] = field(default=3.0, metadata={"help": "Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training)"})

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
                eos_token_id=tokenizer.eos_token_id
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

    # set up FSDP
    training_args.fsdp = "full_shard"
    training_args.fsdp_config = {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": [Llama4TextDecoderLayer],
        "fsdp_state_dict_type": "FULL_STATE_DICT",
        "fsdp_offload_params": False,
        "fsdp_forward_prefetch": False,
        # grad checkpointing through fsdp
        "activation_checkpointing": True,
        "activation_checkpointing_reentrant": False,
    }

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    formatting_func = build_formatting_func(tokenizer)

    # load model (attn is sdpa)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.bfloat16,
    )
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
    
    # get rank, world size for distributed
    rank = get_rank()
    world_size = get_world_size()

    # load dataset
    dataset = load_dataset("json", data_files=script_args.dataset_path, split="train")

    # Subset dataset if specified
    if script_args.dataset_subset_size is not None:
        dataset = dataset.select(range(script_args.dataset_subset_size))
    else:
        print(f"Using the full dataset with {len(dataset)} samples.")

    dataset = dataset.shuffle(seed=training_args.seed)
    print(f"Dataset shuffled with seed: {training_args.seed}.")

    # if using multiple GPUs, shard the dataset
    if world_size > 1:
        print(f"Sharding dataset for Rank {rank} of {world_size}.")
        dataset = dataset.shard(num_shards=world_size, index=rank)

    # init and run trainer
    print("Initializing SFTTrainer...")
    training_args.report_to = ["wandb"]

    # turn off gradient checkpointing
    training_args.gradient_checkpointing = False
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    if rank == 0:
        inference_set = dataset.select(range(20))
        format_infer = build_formatting_func(tokenizer, train=False)
        
        print("Peforming initial inference...")
        outputs = infer(model, tokenizer, format_infer, inference_set)
        for inp, out in zip(inference_set, outputs):
            print(f"    Q: {inp['question']}")
            print(f" True: {inp['answer']}")
            print(f"Model: {out}")
            print()

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if rank  == 0:
        print("Performing post-training inference...")
        outputs = infer(model, tokenizer, format_infer, inference_set)
        for inp, out in zip(inference_set, outputs):
            print(f"    Q: {inp['question']}")
            print(f" True: {inp['answer']}")
            print(f"Model: {out}")
            print()

if __name__ == "__main__":
    main()