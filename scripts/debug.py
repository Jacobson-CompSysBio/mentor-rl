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

def build_formatting_func(tokenizer):
    SYSTEM_PROMPT = (
        "You are a helpful biological chatbot. You will be given a biological question; "
        "return the correct answer."
    )
    def _fmt(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return _fmt

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
        "use_orig_params": False,
        "sync_module_states": False,
        "limit_all_gathers": True,
        "fsdp_forward_prefetch": False,
        "fsdp_backward_prefetch": "BACKWARD_PRE",

        # grad checkpointing through fsdp
        "activation_checkpointing": True,
        "activation_checkpointing_reentrant": False,
    }

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path)
    formatting_func = build_formatting_func(tokenizer)

    # load model     
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False,
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
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()