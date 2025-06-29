import os
import psutil
import time
import argparse
import sys
import traceback
import wandb
from pathlib import Path
from torch.nn import Module
from dotenv import load_dotenv
from accelerate import (
    Accelerator,
    PartialState,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from datasets import load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (
    TrainerCallback,
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
from torch.distributed import is_initialized, get_rank

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

######### paths + hyperparameters ###########
MODEL_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/"
MODEL_NAME = "Llama-4-Scout-17B-16E-Instruct"
DATA_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/"
RAW_JSON = "qa_pairs.json"
CACHE_DIR = Path("/mnt/bb/{}/tokenized_ds".format(os.environ["USER"]))
MAX_LEN = 256
USE_PEFT = True

##### rank processing #####
accelerator = Accelerator()
dist_state = PartialState()
is_main = accelerator.is_main_process

#### memory tracking callbacks ####
MB = 1024 ** 2
t0 = time.perf_counter()

def _stamp(phase):
    if not dist_state.is_main_process:
        return
    gpu_alloc = torch.cuda.memory_allocated() / MB
    gpu_reserved = torch.cuda.memory_reserved() / MB
    cpu_rss = psutil.Process(os.getpid()).memory_info().rss / MB
    elapsed = (time.perf_counter() - t0) / 60.0
    print(f"[{elapsed:6.2f} min] {phase:<12} "
          f"GPU alloc {gpu_alloc:7.0f} MiB | "
          f"reserved {gpu_reserved:7.0f} | "
          f"CPU {cpu_rss:7.0f} MiB", flush=True)

def _mb(x):   # bytes → MiB
    return x / (1024 ** 2)

class MemTrace(TrainerCallback):
    """Print CPU & GPU memory every n steps (rank-0 only)."""
    def __init__(self, every=1):
        self.every = every

    def on_step_begin(self, args, state, control, **kw):
        if state.global_step % self.every != 0:
            return
        if not dist_state.is_main_process:
            return

        gpu_alloc = torch.cuda.memory_allocated()
        gpu_reserved = torch.cuda.memory_reserved()
        cpu_rss = psutil.Process(os.getpid()).memory_info().rss
        now = time.perf_counter() - script_start
        print(
            f"[{now/60:6.2f} min] "
            f"step {state.global_step:>6} | "
            f"GPU alloc { _mb(gpu_alloc):8.0f} MiB "
            f"(reserved { _mb(gpu_reserved):8.0f}) | "
            f"CPU RSS { _mb(cpu_rss):8.0f} MiB",
            flush=True,
        )

##### set up cache #####
if not CACHE_DIR.exists():
    if dist_state.is_local_main_process:
        print(f"[{min_elapsed():.2f} min] Cache directory does not exist, creating: {CACHE_DIR}")
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

        print(f"[{min_elapsed():.2f} min] Tokenizing dataset...")
        tokenized = raw.map(
            tokenize,
            remove_columns=["messages"],
            num_proc=32,
        )
        tokenized.save_to_disk(CACHE_DIR)

dist_state.wait_for_everyone()  # ensure all ranks have created the cache dir
print(f"[{min_elapsed():.2f} min] Cache directory ready: {CACHE_DIR}")
dataset: DatasetDict = load_from_disk(CACHE_DIR, keep_in_memory=False)

##### model #####
if is_main:
    print(f"[{min_elapsed():.2f} min] Loading model: {MODEL_NAME} from {MODEL_DIR}")

model = Llama4ForConditionalGeneration.from_pretrained(
    os.path.join(MODEL_DIR, MODEL_NAME),
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)
model.gradient_checkpointing_enable()

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
if is_main:
    print(f"[{min_elapsed():.2f} min] LoRA config loaded.")

#### establish hooks to track memory usage for debugging ####
# pick the very first transformer block
def first_transformer_block(root: Module) -> Module:
    for m in root.modules():
        # crude heuristic: has both self-attn *and* MLP submodules
        if hasattr(m, "self_attn") and hasattr(m, "mlp"):
            return m
    raise RuntimeError("could not find a transformer block")

first_block = first_transformer_block(model)

def fwd_pre(module, *_, **__):
    _stamp("fwd-PRE")

def fwd_post(module, *_, **__):
    _stamp("fwd-POST")

def bwd_hook(*_):
    _stamp("backward")

first_block.register_forward_pre_hook(fwd_pre)
first_block.register_forward_hook(fwd_post)
first_block.register_full_backward_hook(lambda *a, **k: bwd_hook())

##### training #####
train_cfg = SFTConfig(
    remove_unused_columns=False,
    bf16=True,
    gradient_checkpointing=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    max_seq_length=MAX_LEN,
    logging_steps=1,
    dataloader_num_workers=0,
    report_to=["wandb"],
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=train_cfg,
    peft_config=lora_cfg,
)
if dist_state.is_main_process:
    trainer.add_callback(MemTrace(every=1))

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