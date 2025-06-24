import os
import argparse
import sys
import traceback
import wandb
from pathlib import Path
from dotenv import load_dotenv
from accelerate import PartialState
from datasets import load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (
    Llama4ForConditionalGeneration, 
    TrainingArguments, 
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    )
from trl import (
    SFTConfig,
    SFTTrainer, 
    ModelConfig,
    clone_chat_template,
    get_kbit_device_map,
    get_quantization_config,
    )
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

#### environment variables ####
load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

######### paths + hyperparameters ###########
MODEL_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/"
MODEL_NAME = "Llama-4-Scout-17B-16E-Instruct"
DATA_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/data/"
RAW_JSON = "qa_pairs.json"
CACHE_DIR = Path("/mnt/bb/{}/tokenized_ds".format(os.environ["USER"]))
MAX_LEN = 1024
USE_PEFT = True

##### rank processing #####
state = PartialState()
is_main = state.is_main_process

##### set up cache #####
if not CACHE_DIR.exists():
    if state.is_local_main_process:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ##### data #####
        raw = load_dataset("json",
                            data_dir=DATA_DIR,
                            data_files=RAW_JSON,
                            )

        def format_dataset(row):
            # TRL wants each row in the format:
            # { messages: [ {role:r, content:c}, {role:r, content:c} ] }
            row_q = {"role": "user", "content": row["question"]}
            row_a = {"role": "assistant", "content": row["answer"]}
            row['messages'] = [row_q, row_a]
            return row

        raw = raw["train"].train_test_split(test_size=0.1)
        raw = raw.map(format_dataset,
                      batched=False,
                      remove_columns=["question", "answer"],
                      num_proc=32,
                      load_from_cache_file=False)
    
        # save pre-tokenized dataset so every rank can memory-map it
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_DIR, MODEL_NAME),
            use_fast=True,
        )

        def tokenize(row):
            tokens = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            return {
                "input_ids": tokens,
                "attention_mask": [1] * len(tokens)
            }

        tokenized = raw.map(
            tokenize,
            remove_columns=["messages"],
            num_proc=32,
        )
        tokenized.save_to_disk(CACHE_DIR)

state.wait_for_everyone()  # ensure all ranks have created the cache dir
dataset: DatasetDict = load_from_disk(CACHE_DIR, keep_in_memory=False)

##### model #####
model = Llama4ForConditionalGeneration.from_pretrained(
    os.path.join(MODEL_DIR, MODEL_NAME),
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.config.max_position_embeddings = MAX_LEN
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_DIR, MODEL_NAME),
    use_fast=True
)

lora_cfg = (
    LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    ) if USE_PEFT
    else None)

##### training #####
train_cfg = SFTConfig(
    bf16=True,
    remove_unused_columns=False,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    max_seq_length=MAX_LEN,
    logging_steps=10,
    dataloader_num_workers=0,
    fsdp_config=dict(
        cpu_ram_efficient_loading=True,
        sync_module_states=True,
        offload_params=True,
        activation_checkpointing=True,
        backward_prefetch="backward_pre",
        limit_all_gathers=True,
    ),
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=train_cfg,
    peft_config=lora_cfg,
)

try:
    trainer.train()
    if is_main:
        print("Training complete.")
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise