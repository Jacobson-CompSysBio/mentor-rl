import os, sys
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)

from accelerate import Accelerator, DeepSpeedPlugin, PartialState
from torch.utils.data import DataLoader
import torch.distributed as dist

from trl import SFTTrainer
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import * 

### SLURM VARIABLES ###
load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
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
    use_ds_inference: bool = field(default=False, metadata={"help": "Use DeepSpeed init_inference for distributed generation (post-training recommended)"})
    ds_kernel_inject: bool = field(default=False, metadata={"help": "Try kernel injection for DS inference (set False on ROCm)"})

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

### FORMATTING/INFERENCE ###
def _collate_single(examples):
    """
    unwraps  example in a list from dataloader
    """
    return examples[0]

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

def infer(model, tokenizer, format_fn, dataset, accelerator=None):
    
    if accelerator is None:
        # load inference config
        local_cfg_dir = os.getenv("LOCAL_CONFIG_DIR")
        with open(local_cfg_dir + "/ds_zero3_inference.json") as f:
            ds_infer_cfg = json.load(f)
        ds_inference_plugin = DeepSpeedPlugin(hf_ds_config=ds_infer_cfg)
        inference_accelerator = Accelerator(deepspeed_plugin=ds_inference_plugin)
    else:
        inference_accelerator = accelerator
    inference_accelerator.print(f"Accelerator loaded.")

    # loading inference model
    #print("loading inference model")
    #inference_model = AutoModelForCausalLM.from_pretrained(
    #    model_path,
    #    dtype=torch.bfloat16,
    #    attn_implementation="eager"
    #)
    inference_model=model
    inference_model.eval()
    inference_accelerator.print("Inference model loaded.")

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=_collate_single)
    
    # accelerator.prepare model for distributed inference
    if accelerator is None: 
        inference_model, dataloader = inference_accelerator.prepare(inference_model, dataloader)
    else:
        dataloader = inference_accelerator.prepare(dataloader)

    results=[]
    with torch.no_grad():
        for i, example in enumerate(dataloader):
            formatted = format_fn(example)
            inputs = tokenizer(formatted, return_tensors="pt").to(inference_model.device)
            input_len = inputs["input_ids"].shape[1]

            #inference_accelerator.print(f"Generating completion {i}/{len(dataloader)}...")
            output = inference_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                synced_gpus=True
            )

            new_ids = output[0][input_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            results.append(text)
    
    # clear memory, free cache
    del inference_model, dataloader, inference_accelerator
    torch.cuda.empty_cache()
    return results

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

    ##########################    
    # PRE TRAINING INFERENCE #
    ##########################
    if rank == 0:
        print(f"Running pre-training inference...")
    inf_ds = dataset.select(range(20))
    inf_format = build_formatting_func(tokenizer, train=False)
    pre_outputs = infer(
        model,
        tokenizer,
        inf_format,
        inf_ds,
    ) 
    pre_score = check_accuracy(pre_outputs, list(inf_ds["answer"]))
    if isinstance(pre_score, list):
        pre_score = np.mean(pre_score)
 
    if rank == 0:
        print(f"Pre-training inference complete. Average Score={pre_score:.2%}")
        print("Outputs:", pre_outputs)

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

    ###########################   
    # POST TRAINING INFERENCE #
    ###########################
    if rank == 0:
        print(f"Running post-training inference...")
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
        print(f"Performance improvement: {(post_score - pre_score):.2%}")
        print("Outputs:", post_outputs)


if __name__ == "__main__":
    main()
