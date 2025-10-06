import os
import time
import argparse
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig, HfDeepSpeedConfig
from transformers import logging
logging.set_verbosity_error()
import deepspeed
from deepspeed import comm as dist
rank = dist.get_rank()
world_size = dist.get_world_size()

#### time tracking ####
script_start = time.perf_counter()
def min_elapsed():
    """Return elapsed time in minutes since the script started."""
    return (time.perf_counter() - script_start) / 60.0

#### environment variables ####
load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
ds_cfg = HfDeepSpeedConfig("ds_zero3.json")

######### paths + hyperparameters ###########
MODEL_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-4-Scout-17B-16E-Instruct"
DATA_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/"
RAW_JSON = "qa_pairs.json"
CACHE_DIR = Path("/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/tokenized_ds")
TOKEN_INFO = os.path.join(CACHE_DIR, "dataset_dict.json")
MAX_LEN = 256
USE_PEFT = True

##### rank processing #####
is_main = rank == 0

##### set up cache #####
def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def build_and_cache():
    """load dataset and cache the tokens"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{min_elapsed():.2f} min] Loading raw dataset from {DATA_DIR}...")
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
    print(f"[{min_elapsed():.2f} min] Formatting dataset...")
    raw = raw["train"].train_test_split(test_size=0.1)
    raw = raw.map(format_dataset,
                  batched=False,
                  remove_columns=["question", "answer"],
                  num_proc=32,
                  load_from_cache_file=False)

    # save pre-tokenized dataset so every rank can memory-map it
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
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

    print(f"[{min_elapsed():.2f} min] Tokenizing dataset...")
    tokenized = raw.map(
        tokenize,
        remove_columns=["messages"],
        num_proc=32,
    )
    tokenized.save_to_disk(CACHE_DIR)

# rank 0 builds the cache if it doesn't exist
if is_main and not CACHE_DIR.exists():
    print(f"[{min_elapsed():.2f} min] Caching dataset at {CACHE_DIR}")
    build_and_cache()
    print(f"[{min_elapsed():.2f} min] Dataset cached at {CACHE_DIR}")

# load the same on-disk dataset on all ranks
try:
    train_ds = load_from_disk(f"{CACHE_DIR}/train")
    test_ds = load_from_disk(f"{CACHE_DIR}/test")
    dataset = DatasetDict({"train": train_ds, "test": test_ds})
    print(f"[{min_elapsed():.2f} min] Dataset loaded from {CACHE_DIR}")
except Exception as e:
    if is_main:
       print(f"[{min_elapsed():.2f} min] Error loading dataset from {CACHE_DIR} ({e}); rebuilding...")
       build_and_cache()
    train_ds = load_from_disk(f"{CACHE_DIR}/train")
    test_ds = load_from_disk(f"{CACHE_DIR}/test")
    dataset = DatasetDict({"train": train_ds, "test": test_ds})

##### model #####
if is_main:
    print(f"[{min_elapsed():.2f} min] Loading model: {MODEL_DIR}")

def t(msg):
    torch.cuda.synchronize()
    print(f"[{min_elapsed():5.1f} min] {msg}", flush=True)

t("entering ds.zero.Init...")
#with deepspeed.zero.Init(config_dict_or_path="ds_zero3.json",
#                         dtype=torch.bfloat16):     
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
t("model loaded.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
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
if is_main:
    print(f"[{min_elapsed():.2f} min] LoRA config loaded.")

##### training #####
train_cfg = SFTConfig(
    deepspeed="ds_zero3.json",
    remove_unused_columns=False,
    num_train_epochs=1,
    max_seq_length=MAX_LEN,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    logging_steps=1,
    label_names=[],
    report_to=["wandb"],
    dataloader_num_workers=1,
)
if is_main:
    print(f"[{min_elapsed():.2f} min] Initializing Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=train_cfg,
    peft_config=lora_cfg,
)

try:
    if is_main:
        print(f"[{min_elapsed():.2f} min] Starting training...")
    trainer.train()
    if is_main:
        print(f"[{min_elapsed():.2f} min] Training complete.")
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise