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

# PATCH VLLMClient to work with standard vLLM server
"""
TRL's vllm_mode="server" expects special endpoints (/get_world_size/, 
/init_communicator/, /generate/, etc.) that are only available in TRL's custom 
vLLM server (trl vllm-serve). Standard vLLM serve doesn't have these endpoints.
 
Since we're using standard vLLM with Ray for multi-node inference (TP+PP),
we patch VLLMClient to:
1. Skip weight sync (vLLM uses initial weights)
2. Use /v1/completions instead of /generate/ for text generation
"""
def _patch_vllm_client():
    """Patch VLLMClient to work with standard vLLM server (no weight sync)."""
    try:
        from trl.extras import vllm_client
        import requests
        
        # Store original init_communicator
        original_init_communicator = vllm_client.VLLMClient.init_communicator
        
        def patched_init_communicator(self, device=0):
            """Skip weight synchronization - standard vLLM doesn't support it."""
            print("[PATCH] VLLMClient.init_communicator called - SKIPPING (using standard vLLM)")
            print("[PATCH] Weight synchronization is disabled. vLLM will use initial model weights.")
            # Set minimal attributes that TRL might check
            self.rank = 0
            self.world_size = 1
        
        vllm_client.VLLMClient.init_communicator = patched_init_communicator
        
        # Also patch update_named_param if it exists
        if hasattr(vllm_client.VLLMClient, 'update_named_param'):
            def patched_update_named_param(self, name, weights):
                # Silently skip - don't spam logs during training
                return True
            vllm_client.VLLMClient.update_named_param = patched_update_named_param
        
        # Patch close_communicator
        if hasattr(vllm_client.VLLMClient, 'close_communicator'):
            def patched_close_communicator(self):
                pass  # No-op
            vllm_client.VLLMClient.close_communicator = patched_close_communicator
        
        # Patch reset_prefix_cache
        if hasattr(vllm_client.VLLMClient, 'reset_prefix_cache'):
            def patched_reset_prefix_cache(self):
                pass  # Standard vLLM doesn't have this endpoint
            vllm_client.VLLMClient.reset_prefix_cache = patched_reset_prefix_cache
        
        # ================================================================
        # CRITICAL: Patch generate() to use /v1/completions instead of /generate/
        # ================================================================
        original_generate = vllm_client.VLLMClient.generate
        
        def patched_generate(
            self,
            prompts,
            images=None,
            n=1,
            repetition_penalty=1.0,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            max_tokens=16,
            truncate_prompt_tokens=None,
            guided_decoding_regex=None,
            generation_kwargs=None,
        ):
            """
            Use standard vLLM /v1/completions endpoint instead of TRL's /generate/.
            Returns dict with prompt_ids, completion_ids, and logprobs.
            """
            import requests
            from transformers import AutoTokenizer
            
            # Get model name from vLLM server
            models_url = f"{self.base_url}/v1/models"
            try:
                models_resp = requests.get(models_url, timeout=30)
                models_resp.raise_for_status()
                model_name = models_resp.json()["data"][0]["id"]
            except Exception as e:
                print(f"[PATCH] Failed to get model name: {e}")
                raise
            
            # Build request for /v1/completions
            url = f"{self.base_url}/v1/completions"
            
            all_prompt_ids = []
            all_completion_ids = []
            all_logprobs = []
            
            # Load tokenizer for encoding/decoding (cache it on self)
            if not hasattr(self, '_tokenizer'):
                # Use the model path from the model name
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer = self._tokenizer
            
            # Process each prompt
            for prompt in prompts:
                # Convert max_new_tokens to max_tokens if provided in generation_kwargs
                effective_max_tokens = max_tokens
                if generation_kwargs and 'max_new_tokens' in generation_kwargs:
                    effective_max_tokens = generation_kwargs['max_new_tokens']
                
                # Ensure max_tokens is positive (vLLM requires > 0)
                if effective_max_tokens <= 0:
                    print(f"[PATCH] WARNING: max_tokens={effective_max_tokens}, using default 128")
                    effective_max_tokens = 128
                
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": effective_max_tokens,
                    "n": n,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "logprobs": 1,  # Request logprobs
                    "echo": False,  # Don't echo prompt
                }
                
                if top_k > 0:
                    payload["top_k"] = top_k
                if min_p > 0:
                    payload["min_p"] = min_p
                
                # Filter out HuggingFace-specific kwargs that vLLM doesn't support
                IGNORED_KWARGS = {
                    'max_new_tokens', 'pad_token_id', 'eos_token_id', 'use_cache',
                    'do_sample', 'num_return_sequences', 'output_scores',
                    'return_dict_in_generate', 'synced_gpus', 'attention_mask',
                }
                if generation_kwargs:
                    filtered_kwargs = {k: v for k, v in generation_kwargs.items() if k not in IGNORED_KWARGS}
                    payload.update(filtered_kwargs)
                
                try:
                    response = requests.post(url, json=payload, timeout=300)
                    response.raise_for_status()
                    result = response.json()
                except Exception as e:
                    print(f"[PATCH] Completion request failed: {e}")
                    raise Exception(f"Request failed: {response.status_code if 'response' in dir() else 'N/A'}, {str(e)}")
                
                # Extract results for each completion (n completions per prompt)
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                
                for choice in result.get("choices", []):
                    completion_text = choice.get("text", "")
                    
                    # Extract logprobs and token_ids from vLLM response
                    choice_logprobs = []
                    completion_tokens = []
                    
                    if choice.get("logprobs"):
                        logprobs_data = choice["logprobs"]
                        # Get token logprobs
                        if logprobs_data.get("token_logprobs"):
                            choice_logprobs = [lp if lp is not None else 0.0 for lp in logprobs_data["token_logprobs"]]
                        # Get actual token IDs from vLLM (preferred) or tokens
                        if logprobs_data.get("tokens"):
                            # vLLM returns the actual tokens, convert to IDs
                            # This ensures consistency with logprobs count
                            tokens = logprobs_data["tokens"]
                            try:
                                converted = tokenizer.convert_tokens_to_ids(tokens)
                                # Handle both single value and list returns
                                if converted is None:
                                    completion_tokens = []
                                elif isinstance(converted, int):
                                    completion_tokens = [converted]
                                elif isinstance(converted, list):
                                    # Filter out None values and replace with unk_token_id
                                    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
                                    completion_tokens = [t if t is not None else unk_id for t in converted]
                                else:
                                    completion_tokens = []
                            except Exception:
                                completion_tokens = []
                    
                    # Fallback: if we didn't get tokens from logprobs, re-tokenize the text
                    if not completion_tokens:
                        completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
                    
                    # Safety: ensure completion_tokens is a list of ints
                    if not isinstance(completion_tokens, list):
                        completion_tokens = list(completion_tokens) if completion_tokens else []
                    
                    # Ensure logprobs match completion_tokens length
                    if len(choice_logprobs) != len(completion_tokens):
                        # Adjust logprobs to match token count
                        if len(choice_logprobs) < len(completion_tokens):
                            # Pad with zeros
                            choice_logprobs.extend([0.0] * (len(completion_tokens) - len(choice_logprobs)))
                        else:
                            # Truncate
                            choice_logprobs = choice_logprobs[:len(completion_tokens)]
                    
                    # Final fallback: if still no logprobs, use zeros
                    if not choice_logprobs:
                        choice_logprobs = [0.0] * len(completion_tokens)
                    
                    all_prompt_ids.append(prompt_tokens)
                    all_completion_ids.append(completion_tokens)
                    all_logprobs.append(choice_logprobs)
            
            return {
                "prompt_ids": all_prompt_ids,
                "completion_ids": all_completion_ids,
                "logprobs": all_logprobs,
            }
        
        vllm_client.VLLMClient.generate = patched_generate
        
        print("[PATCH] VLLMClient patched for standard vLLM server compatibility")
        print("[PATCH] - init_communicator: SKIPPED")
        print("[PATCH] - generate: Using /v1/completions instead of /generate/")
        
    except ImportError as e:
        print(f"[PATCH] Could not patch VLLMClient: {e}")

# Apply patch immediately on import
_patch_vllm_client()

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

VLLM_HOST=os.getenv("HEAD_IP")
VLLM_PORT=int(os.getenv("VLLM_HTTP_PORT", "51001"))  # convert to int for gRPC/HTTP

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

def _verify_vllm():
    # verify vLLM server is reachable and completions endpoint works (rank 0 only)
    try:
        import requests
        if int(os.environ.get("RANK", "0")) == 0:
            url = f"http://{VLLM_HOST}:{int(VLLM_PORT)}/v1/models"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            models = r.json()
            print(f"[CHECK] vLLM /v1/models OK: {r.status_code} {models}")
            if not models.get("data"):
                print("[FATAL] No models loaded in vLLM server!")
                sys.exit(2)
            model_id = models["data"][0]["id"]

            # Now test completions (first request may be slow due to kernel compilation)
            completions_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/completions"
            payload = {
                "model": model_id,
                "prompt": "Hello, world! Say hi:",
                "max_tokens": 10,
                "temperature": 0
            }
            print("[CHECK] Testing /v1/completions (may take 30-60s for first request)...")
            resp = requests.post(completions_url, json=payload, timeout=120)
            print(f"[CHECK] vLLM /v1/completions: {resp.status_code} {resp.text}")
            if "choices" not in resp.text:
                print("[FATAL] vLLM completions endpoint did not return choices!")
                sys.exit(2)
    except Exception as e:
        print(f"[FATAL] Cannot reach vLLM server or completions endpoint at {VLLM_HOST}:{VLLM_PORT}: {e}")
        sys.exit(2)

_debug_preflight()
_verify_vllm()

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
    per_device_train_batch_size: int = 4  # Increased: more prompts per vLLM request
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
    num_generations: int = 4   # Reduced: fewer generations per prompt (4 is often sufficient)
    max_completion_length: int = 128  # Reduced: faster generation if task allows

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
        torch_dtype=torch.bfloat16,
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
        bf16=True,  # Must match model dtype (bfloat16)
        
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
 
    # vLLM engine kwargs (used when TRL spawns vLLM directly, not in server mode)
    # These are kept for reference but vllm_mode="server" uses external server
    trl_args.vllm_engine_kwargs = {
        "tensor_parallel_size": 8,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "disable_log_stats": True,
        "distributed_executor_backend": "ray",
        "disable_custom_all_reduce": True,
        "trust_remote_code": True,
        "max_model_len": 4096,
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