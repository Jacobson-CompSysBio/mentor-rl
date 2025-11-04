### SCRIPT FROM: https://cloud.google.com/ai-hypercomputer/docs/tutorials/fsdp-llama4 ###
import os, sys
import socket, time
import urllib.request, urllib.error
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
    GenerationConfig
)
import torch.distributed as dist
#from transformers.models.llama4.modeling_llama4 import Llama4TextDecoderLayer
from trl import GRPOTrainer, GRPOConfig
from dataclasses import dataclass, field
from typing import Optional, Union, List
from dotenv import load_dotenv
import multiprocessing as mp

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import *
from utils.rewards import format_reward, correctness_reward

### wandb logging ###
load_dotenv()
os.environ["WANDB_PROJECT"] = "mentor-rl"
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

### slurm script info ###
nnodes = int(os.environ.get("SLURM_NNODES", 1))
timeout = int(os.environ.get("SLURM_JOB_TIMEOUT", 0))
slurm_args = argparse.Namespace(nnodes=nnodes, timeout=timeout)

VLLM_HOST=os.getenv("VLLM_HOST_IP")
VLLM_PORT=os.getenv("VLLM_HTTP_PORT")

### sanity check ###
if "ROCR_VISIBLE_DEVICES" in os.environ and "HIP_VISIBLE_DEVICES" not in os.environ:
    os.environ["HIP_VISIBLE_DEVICES"] = os.environ["ROCR_VISIBLE_DEVICES"]
os.environ.setdefault("DS_ACCELERATOR", "cuda")

def _debug_preflight():
    import os, torch, platform
    print("\n==== PY PRE-FLIGHT ====")
    print("Host:", platform.node(), "PID:", os.getpid())
    print("Torch:", torch.__version__, "HIP:", getattr(torch.version, "hip", None))
    print("CUDA available?:", torch.cuda.is_available(), "num GPUs:", torch.cuda.device_count())
    print("Env HIP_VISIBLE_DEVICES:", os.environ.get("HIP_VISIBLE_DEVICES"))
    print("Env ROCR_VISIBLE_DEVICES:", os.environ.get("ROCR_VISIBLE_DEVICES"))
    # Try simple HIP query per rank
    try:
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            print(f"Rank sees device {i}: name={props.name}")
    except Exception as e:
        print("Device probe error:", repr(e))
    # Library versions that often matter here
    try:
        import deepspeed
        print("DeepSpeed:", getattr(deepspeed, "__version__", "unknown"))
    except Exception as e:
        print("DeepSpeed import error:", repr(e))
    try:
        import accelerate, transformers, vllm
        print("Accelerate:", accelerate.__version__)
        print("Transformers:", transformers.__version__)
        print("vLLM:", getattr(vllm, "__version__", "unknown"))
    except Exception as e:
        print("HF/vLLM import error:", repr(e))
    print("=======================\n")

_debug_preflight()

### Dataclasses for args
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
class GrpoTrainingArguments:
    # Core trainer knobs (needed by your run-name function)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    learning_rate: float = 2e-5
    output_dir: str = "runs/grpo-format"

    # Stability / logging
    seed: int = 900913
    logging_steps: int = 1
    save_steps: int = 500
    save_total_limit: int = 2
    report_to = ["wandb"]

    # GRPO specifics
    num_generations: int = 8
    max_completion_length: int = 256

    # Runtime / precision
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True
    dataloader_drop_last: bool = True
    deepspeed: Optional[str] = None  # path to ds_zero3.json

    # Optional extras
    warmup_steps: int = 0
    weight_decay: float = 0

    # grpo: formatting reward
    max_think_chars: int = field(default=4000, metadata={"help": "Soft cap for <think> length"})

    # grpo: weights
    w_format: float = field(default=0.3, metadata={"help": "Weight for format reward"})
    w_task: float = field(default=1.0, metadata={"help": "Weight for domain reward"})

### distributed helper function
def get_rank_world_size():
    # Use env first (works before PG init)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # If PG happens to be ready already, prefer that
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
    return rank, world

### chat template for GRPO
def build_prompt(question: str, tokenizer) -> str:
    SYSTEM_PROMPT = (
        "You are a helpful biological chatbot. You will be given a biological question; "
        "return the final result wrapped in <answer></answer> and enclose any reasoning in <think></think>."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def main():
    # verify vLLM server is reachable (rank 0 only)
    try:
        import requests
        if int(os.environ.get("RANK", "0")) == 0:
            url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            print(f"[CHECK] vLLM /v1/models OK: {r.status_code}")
    except Exception as e:
        print(f"[FATAL] Cannot reach vLLM server at {VLLM_HOST}:{VLLM_PORT}: {e}")
        sys.exit(2)

    # extract args from classes
    parser = HfArgumentParser((ScriptArguments, PeftArguments, GrpoTrainingArguments))
    script_args, peft_args, grpo_args = parser.parse_args_into_dataclasses()

    # grpo args
    grpo_args.remove_unused_columns = False

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model (attn is sdpa)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()           # important for LoRA + checkpointing
    model.config.use_cache = False               # for training
    model.generation_config.use_cache = False    # quiets messages during GRPO generation

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
    if script_args.dataset_subset_size is not None:
        dataset = dataset.select(range(script_args.dataset_subset_size))

    # keep gt answer in the row for correctness reward, but do not include it in the prompt text
    def _map(example):
        return {
            **example,
            "prompt": build_prompt(example["question"], tokenizer),
        } 
    dataset = dataset.map(_map, remove_columns=[])

    # rewards
    def format_reward_wrapper(completions, **kwargs):
        return format_reward(completions, max_think_chars=grpo_args.max_think_chars, **kwargs) 
    reward_funcs = [format_reward_wrapper, correctness_reward]

    rank, world_size = get_rank_world_size()
    print(f"[DIST] rank={rank} world={world_size} "
      f"LOCAL_RANK={os.getenv('LOCAL_RANK')} "
      f"RANK={os.getenv('RANK')} WORLD_SIZE={os.getenv('WORLD_SIZE')}")

    # if using multiple GPUs, shard the dataset
    if world_size > 1:
        print(f"Sharding dataset for Rank {rank} of {world_size}.")
        dataset = dataset.shard(num_shards=world_size, index=rank)

    # init and run trainer
    print("Initializing GRPOTrainer...")
    report_to = grpo_args.report_to if isinstance(grpo_args.report_to, list) else [grpo_args.report_to]
    run_name = make_grpo_run_name(script_args, peft_args, grpo_args, slurm_args)

    trl_args = GRPOConfig(
        # basics
        output_dir=grpo_args.output_dir,
        run_name=run_name,
        seed=grpo_args.seed,
        report_to=report_to,

        # optimizer / schedule
        learning_rate=grpo_args.learning_rate,
        weight_decay=grpo_args.weight_decay,
        warmup_steps=grpo_args.warmup_steps,

        # batching
        per_device_train_batch_size=grpo_args.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_args.gradient_accumulation_steps,
        dataloader_drop_last=grpo_args.dataloader_drop_last,

        # logging / ckpt
        logging_steps=grpo_args.logging_steps,
        save_steps=grpo_args.save_steps,
        save_total_limit=grpo_args.save_total_limit,

        # runtime
        remove_unused_columns=grpo_args.remove_unused_columns,
        gradient_checkpointing=grpo_args.gradient_checkpointing,
        deepspeed=grpo_args.deepspeed,
        bf16=False,
        
        # vLLM
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=VLLM_HOST,
        vllm_server_port=VLLM_PORT,

        # GRPO specifics
        num_generations=grpo_args.num_generations,
        max_completion_length=grpo_args.max_completion_length,

        # epochs
        num_train_epochs=grpo_args.num_train_epochs,
    )
    # reward weights (order must match reward_funcs)
    trl_args.reward_weights = [grpo_args.w_format, grpo_args.w_task]

    # configs for GRPO generations
    trl_args.generation_kwargs = {
        "max_new_tokens": grpo_args.max_completion_length,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.7,
        "use_cache": False,                        # <-- explicit: silences the message
        "pad_token_id": tokenizer.pad_token_id,    # <-- important for batching
        "eos_token_id": tokenizer.convert_tokens_to_ids("</answer>"),
    }
 
    trl_args.vllm_engine_kwargs = {
        "tensor_parallel_size": 8,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "disable_log_stats": True,
        "tensor_parallel_size": 8,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "disable_log_stats": True,
        "distributed_executor_backend": "ray",
        "disable_custom_all_reduce": True,
    }

    trainer = GRPOTrainer(
        model=model,
        args=trl_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )

    # Make sure the final wrapped model has the input-grad hook
    if hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()

    trainer.train()
    trainer.save_model(script_args.output_dir)

if __name__ == "__main__":
    main()