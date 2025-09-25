import os, random
import sys
import traceback
import socket
from pathlib import Path
from torch.nn import Module
from dotenv import load_dotenv
from accelerate import (
    Accelerator,
    PartialState,
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map
)
import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from datasets import load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
import torch
import torch.distributed as dist
from transformers import (
    TrainerCallback,
    Llama4ForConditionalGeneration, 
    TrainingArguments, 
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    )
from transformers.integrations import HfDeepSpeedConfig
from trl import (
    SFTConfig,
    SFTTrainer, 
    ModelConfig,
    clone_chat_template,
    get_kbit_device_map,
    get_quantization_config,
    )
from torch.distributed import is_initialized, get_rank

#### environment variables ####
def get_env():
    # Works with: srun -N... --ntasks-per-node=8 --gpus-per-task=1
    env = {}
    env["MASTER_ADDR"] = os.environ.get("MASTER_ADDR") or os.environ.get("SLURM_LAUNCH_NODE_IPADDR") or os.environ.get("SLURM_LAUNCH_NODE") or socket.gethostname()
    env["MASTER_PORT"] = os.environ.get("MASTER_PORT") or "6000"
    env["WORLD_SIZE"] = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)
    env["RANK"] = int(os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or 0)
    env["LOCAL_RANK"] = int(os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or 0)
    return env

def setup_dist(timeout_min=10):
    env = get_env()
    os.environ["MASTER_ADDR"] = env["MASTER_ADDR"]
    os.environ["MASTER_PORT"] = str(env["MASTER_PORT"])
    os.environ["WORLD_SIZE"] = str(env["WORLD_SIZE"])
    os.environ["RANK"] = str(env["RANK"])
    os.environ["LOCAL_RANK"] = str(env["LOCAL_RANK"])

    # ROCm: backend="nccl" is correct (RCCL)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=env["WORLD_SIZE"],
        rank=env["RANK"],
        timeout=torch.timedelta(minutes=timeout_min),
    )
    torch.cuda.set_device(env["LOCAL_RANK"])
    return env

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_main_process(rank): return rank == 0

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
 
def main():
    env = setup_dist()
    rank, world_size, local_rank = env["RANK"], env["WORLD_SIZE"], env["LOCAL_RANK"]
    device = torch.device("cuda", local_rank)

    if is_main_process(rank):
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        print(f"WORLD_SIZE: {world_size}")
        print(f"LOCAL_RANK: {local_rank}")
        print(f"MODEL_DIR: {MODEL_DIR}")
        print(f"CACHE_DIR: {CACHE_DIR}")