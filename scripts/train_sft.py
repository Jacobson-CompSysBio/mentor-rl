import os, sys
import json
import struct
import torch
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
from accelerate.utils import gather_object

from trl import SFTTrainer
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference import infer
from evaluate_pretrajectory_sft_predictions import (
    evaluate_prediction_rows,
    load_canonical_objects,
    render_html_report,
    write_json as write_exact_json,
)
from utils.utils import * 

### SLURM VARIABLES ###
load_dotenv()
os.environ["WANDB_PROJECT"] = "mentor-sft"
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

nnodes = int(os.environ.get("SLURM_NNODES", 1))
ntasks = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", nnodes)))
timeout = int(os.environ.get("SLURM_JOB_TIMEOUT", 0))
slurm_args = argparse.Namespace(nnodes=nnodes, ntasks=ntasks, timeout=timeout)

### ARGS ###
@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Hugging Face model ID from the Hub"})
    dataset_path: str = field(default="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/qa_pairs.json", metadata={"help": "Local dataset path"})
    eval_dataset_path: Optional[str] = field(default=None, metadata={"help": "Optional validation dataset JSONL path for eval loss"})
    holdout_dataset_path: Optional[str] = field(default=None, metadata={"help": "Optional held-out dataset JSONL path for post-training predictions"})
    run_inference_after_training: bool = field(default=False, metadata={"help": "Run sample inference after training"})
    dataset_subset_size: Optional[int] = field(default=None, metadata={"help": "Number of samples to use from the dataset for training. If None, uses the full dataset."})
    eval_dataset_subset_size: Optional[int] = field(default=None, metadata={"help": "Number of validation samples to use. If None, uses the full validation dataset."})
    holdout_sample_size: int = field(default=20, metadata={"help": "Number of held-out examples to generate after training"})
    holdout_max_new_tokens: int = field(default=96, metadata={"help": "Maximum new tokens for held-out generation"})
    holdout_temperature: float = field(default=0.0, metadata={"help": "Sampling temperature for held-out generation when holdout_do_sample is true"})
    holdout_top_p: float = field(default=1.0, metadata={"help": "Top-p value for held-out generation when holdout_do_sample is true"})
    holdout_do_sample: bool = field(default=False, metadata={"help": "Sample held-out predictions. Defaults to deterministic generation."})
    holdout_enable_thinking: bool = field(default=True, metadata={"help": "Use thinking-capable chat-template generation for held-out predictions."})
    holdout_reasoning_effort: Optional[str] = field(default="low", metadata={"help": "Reasoning effort for held-out generation when the chat template supports it."})
    holdout_max_total_tokens: Optional[int] = field(default=1536, metadata={"help": "Optional cap on prompt tokens plus generated tokens during holdout inference."})
    holdout_output_path: Optional[str] = field(default=None, metadata={"help": "Optional JSONL path for held-out predictions"})
    holdout_report_path: Optional[str] = field(default=None, metadata={"help": "Optional JSON path for held-out summary metrics"})
    holdout_exact_report_path: Optional[str] = field(default=None, metadata={"help": "Optional JSON path for exact graph-answer holdout metrics"})
    holdout_html_report_path: Optional[str] = field(default=None, metadata={"help": "Optional HTML path for exact graph-answer holdout review"})
    local_files_only: bool = field(default=True, metadata={"help": "Only load model/tokenizer files from local disk/cache"})

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

def configure_device(rank: int) -> None:
    if "ROCR_VISIBLE_DEVICES" in os.environ and "HIP_VISIBLE_DEVICES" not in os.environ:
        os.environ["HIP_VISIBLE_DEVICES"] = os.environ["ROCR_VISIBLE_DEVICES"]
    os.environ.setdefault("DS_ACCELERATOR", "cuda")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    print(
        f"[rank{rank}] preflight: local_rank={local_rank} "
        f"cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')} "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}",
        flush=True,
    )
    if not torch.cuda.is_available():
        print(f"[rank{rank}] WARNING: torch.cuda.is_available() is False; this rank will not train on GPU.", flush=True)
        return
    device_count = torch.cuda.device_count()
    device_index = local_rank if local_rank < device_count else 0
    torch.cuda.set_device(device_index)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(
        f"[rank{rank}] using cuda device {torch.cuda.current_device()}: {props.name}",
        flush=True,
    )

def count_safetensors_tensors(path: Path) -> int:
    with path.open("rb") as handle:
        header_len_bytes = handle.read(8)
        if len(header_len_bytes) != 8:
            raise RuntimeError(f"Invalid safetensors file: {path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header = json.loads(handle.read(header_len))
    return len([key for key in header if key != "__metadata__"])

def validate_peft_adapter(output_dir: str) -> None:
    output_path = Path(output_dir)
    adapter_path = output_path / "adapter_model.safetensors"
    if adapter_path.exists():
        tensor_count = count_safetensors_tensors(adapter_path)
        if tensor_count == 0:
            raise RuntimeError(
                f"Saved PEFT adapter has no tensors: {adapter_path}. "
                "For ZeRO-3 saves, enable stage3_gather_16bit_weights_on_model_save."
            )
        print(
            f"Validated PEFT adapter: {adapter_path} contains {tensor_count} tensors.",
            flush=True,
        )
        return

    adapter_bin = output_path / "adapter_model.bin"
    if adapter_bin.exists() and adapter_bin.stat().st_size > 1024:
        print(f"Validated PEFT adapter: {adapter_bin} exists.", flush=True)
        return

    raise RuntimeError(f"No usable PEFT adapter found in {output_path}")

def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

def _load_sft_dataset(path: str):
    dataset = load_dataset("json", data_files=path, split="train")
    if "system" not in dataset.column_names:
        dataset = dataset.map(lambda x: {"system": SYSTEM_PROMPT})
    return dataset

def _select_first_n(dataset, n: Optional[int], label: str, rank: int):
    if n is None:
        if rank == 0:
            print(f"Using the full {label} dataset with {len(dataset)} samples.")
        return dataset
    count = min(n, len(dataset))
    if rank == 0 and count < n:
        print(f"[WARNING] Requested {n} {label} samples, but only {len(dataset)} are available.")
    return dataset.select(range(count))

def _prepare_training_dataset(path: str, subset_size: Optional[int], seed: int, rank: int):
    dataset = _load_sft_dataset(path)
    dataset = _select_first_n(dataset, subset_size, "training", rank)
    dataset = dataset.shuffle(seed=seed)
    if rank == 0:
        print(f"Training dataset shuffled with seed: {seed}.")
    return dataset

def _prepare_eval_dataset(path: Optional[str], subset_size: Optional[int], seed: int, rank: int):
    if path is None:
        return None
    dataset = _load_sft_dataset(path)
    if subset_size is not None:
        dataset = dataset.shuffle(seed=seed)
    dataset = _select_first_n(dataset, subset_size, "validation", rank)
    if rank == 0:
        print(f"Loaded validation dataset from {path} with {len(dataset)} samples.")
    return dataset

def _prepare_holdout_dataset(script_args, train_dataset, seed: int, rank: int):
    if script_args.holdout_dataset_path is not None:
        dataset = _load_sft_dataset(script_args.holdout_dataset_path).shuffle(seed=seed)
        source = script_args.holdout_dataset_path
    else:
        dataset = train_dataset
        source = script_args.dataset_path

    sample_size = min(script_args.holdout_sample_size, len(dataset))
    if rank == 0:
        print(f"Preparing {sample_size} holdout samples from {source}.")
    dataset = dataset.select(range(sample_size))
    if "_sample_idx" in dataset.column_names:
        dataset = dataset.remove_columns(["_sample_idx"])
    return dataset.add_column("_sample_idx", list(range(sample_size)))

def _default_holdout_paths(training_args, script_args):
    output_dir = Path(training_args.output_dir)
    predictions_path = Path(script_args.holdout_output_path) if script_args.holdout_output_path else output_dir / "holdout_predictions.jsonl"
    report_path = Path(script_args.holdout_report_path) if script_args.holdout_report_path else output_dir / "holdout_report.json"
    exact_report_path = Path(script_args.holdout_exact_report_path) if script_args.holdout_exact_report_path else output_dir / "holdout_exact_report.json"
    html_report_path = Path(script_args.holdout_html_report_path) if script_args.holdout_html_report_path else output_dir / "holdout_review.html"
    return predictions_path, report_path, exact_report_path, html_report_path

def _write_holdout_artifacts(predictions_path: Path, report_path: Path, rows: list[dict], mean_score: float) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "mean_overlap_score": float(mean_score),
        "sample_count": len(rows),
        "predictions_path": str(predictions_path),
    }
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

def _resolve_resume_checkpoint(resume_from_checkpoint: Optional[str], output_dir: str, rank: int) -> Optional[str]:
    if not resume_from_checkpoint:
        return None

    value = str(resume_from_checkpoint).strip()
    auto_values = {"1", "true", "yes", "auto", "latest"}
    if value.lower() in auto_values:
        latest = get_last_checkpoint(output_dir)
        if latest is None:
            if rank == 0:
                print(f"No checkpoint found in {output_dir}; starting fresh.")
            return None
        if rank == 0:
            print(f"Auto-resuming from latest checkpoint: {latest}")
        return latest

    checkpoint_path = Path(value)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Requested resume checkpoint does not exist: {checkpoint_path}")
    if rank == 0:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    return str(checkpoint_path)

def main():

    ############################
    # MODEL LOADING / PRE-REQS #
    ############################
    # extract args from classes
    parser = HfArgumentParser((ScriptArguments, PeftArguments, SftTrainingArguments))
    script_args, peft_args, training_args = parser.parse_args_into_dataclasses()
    rank, world_size = get_rank_world_size()
    configure_device(rank)

    # make run name
    training_args.run_name = make_run_name(script_args, peft_args, training_args, slurm_args)
    training_args.optim = "adamw_torch_fused"
    training_args.gradient_checkpointing = True

    # load tokenizer and fix padding token
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_path,
        local_files_only=script_args.local_files_only,
        trust_remote_code=True,
    )
    
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
        local_files_only=script_args.local_files_only,
        trust_remote_code=True,
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

    dataset = _prepare_training_dataset(
        script_args.dataset_path,
        script_args.dataset_subset_size,
        training_args.seed,
        rank,
    )
    eval_dataset = _prepare_eval_dataset(
        script_args.eval_dataset_path,
        script_args.eval_dataset_subset_size,
        training_args.seed,
        rank,
    )
    if eval_dataset is not None:
        training_args.do_eval = True

    run_holdout = script_args.run_inference_after_training
    if run_holdout:
        inf_ds = _prepare_holdout_dataset(script_args, dataset, training_args.seed + 1729, rank)
        inf_format = build_formatting_func(
            tokenizer,
            train=False,
            enable_thinking=script_args.holdout_enable_thinking,
            reasoning_effort=script_args.holdout_reasoning_effort,
        )

    if world_size > 1:
        if rank == 0:
            print(f"Using distributed sampler across {world_size} ranks.")
    
    ############    
    # TRAINING #
    ############    

    if rank == 0:
        print(f"Initializing trainer...")
    # init and run trainer
    if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true"}:
        training_args.report_to = []
    else:
        training_args.report_to = ["wandb"]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )

    if rank == 0:
        print(f"Trainer initialized successfully, beginning training...") 
    resume_checkpoint = _resolve_resume_checkpoint(
        getattr(training_args, "resume_from_checkpoint", None),
        training_args.output_dir,
        rank,
    )

    # Run training
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if eval_dataset is not None:
        if rank == 0:
            print("Running final validation evaluation...")
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        if rank == 0:
            print(f"Final validation metrics: {eval_metrics}")

    if rank == 0:
        print(f"Training complete. Saving...")
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        validate_peft_adapter(training_args.output_dir)
        print(f"Model saved to {training_args.output_dir}")
    trainer.accelerator.wait_for_everyone()

    if run_holdout:
        ###########################   
        # POST TRAINING HOLDOUT   #
        ###########################
        if rank == 0:
            print("Running post-training holdout inference...")
        post_records = infer(
            trainer.model,
            tokenizer,
            inf_format,
            inf_ds,
            trainer.accelerator,
            max_new_tokens=script_args.holdout_max_new_tokens,
            temperature=script_args.holdout_temperature,
            top_p=script_args.holdout_top_p,
            do_sample=script_args.holdout_do_sample,
            max_total_tokens=script_args.holdout_max_total_tokens,
            enable_thinking=script_args.holdout_enable_thinking,
            reasoning_effort=script_args.holdout_reasoning_effort,
            return_indices=True,
        )
        gathered_records = list(flatten_gathered_objects(gather_object(post_records)))
        predictions_by_index = {}
        for record in gathered_records:
            if not isinstance(record, dict):
                continue
            idx = int(record["idx"])
            if 0 <= idx < len(inf_ds) and idx not in predictions_by_index:
                predictions_by_index[idx] = record["prediction"]

        missing = [idx for idx in range(len(inf_ds)) if idx not in predictions_by_index]
        if missing and rank == 0:
            print(f"[WARNING] Missing post-training holdout predictions for sample indices: {missing}")

        ordered_indices = sorted(predictions_by_index)
        all_outputs = [predictions_by_index[idx] for idx in ordered_indices]
        answers = [inf_ds[idx]["answer"] for idx in ordered_indices]
        
        post_scores = check_accuracy(all_outputs, answers)
        post_score = float(np.mean(post_scores)) if post_scores else 0.0
 
        if rank == 0:
            rows = []
            scores_by_index = {idx: float(score) for idx, score in zip(ordered_indices, post_scores)}
            for idx in ordered_indices:
                example = inf_ds[idx]
                rows.append(
                    {
                        "idx": int(idx),
                        "system": example.get("system", SYSTEM_PROMPT),
                        "question": example["question"],
                        "answer": example["answer"],
                        "prediction": predictions_by_index[idx],
                        "overlap_score": scores_by_index[idx],
                        "metadata": example.get("metadata", {}),
                    }
                )
            holdout_output_path, holdout_report_path, holdout_exact_report_path, holdout_html_report_path = _default_holdout_paths(training_args, script_args)
            _write_holdout_artifacts(holdout_output_path, holdout_report_path, rows, post_score)
            canonical_objects_path = None
            if script_args.holdout_dataset_path is not None:
                candidate_path = Path(script_args.holdout_dataset_path).resolve().parent / "canonical_objects.jsonl"
                if candidate_path.exists():
                    canonical_objects_path = candidate_path
            canonical_objects = load_canonical_objects(canonical_objects_path)
            exact_report = evaluate_prediction_rows(rows, canonical_objects_by_id=canonical_objects)
            write_exact_json(holdout_exact_report_path, exact_report)
            holdout_html_report_path.parent.mkdir(parents=True, exist_ok=True)
            holdout_html_report_path.write_text(
                render_html_report(exact_report, {row["idx"]: row for row in rows}),
                encoding="utf-8",
            )
            print(f"Post-training holdout complete. Average overlap score={post_score:.2%}")
            print(f"Holdout predictions written to {holdout_output_path}")
            print(f"Holdout report written to {holdout_report_path}")
            print(f"Holdout exact report written to {holdout_exact_report_path}")
            print(f"Holdout HTML review written to {holdout_html_report_path}")
            try:
                import wandb
                table = wandb.Table(
                    columns=[
                        "idx",
                        "mixture_bucket",
                        "view_type",
                        "system",
                        "question",
                        "answer",
                        "prediction",
                        "overlap_score",
                    ]
                )
                for row in rows:
                    metadata = row.get("metadata") or {}
                    table.add_data(
                        row["idx"],
                        metadata.get("mixture_bucket", ""),
                        metadata.get("view_type", ""),
                        row["system"],
                        row["question"],
                        row["answer"],
                        row["prediction"],
                        row["overlap_score"],
                    )
                trainer.log(
                    {
                        "holdout/overlap_score": post_score,
                        "holdout/num_samples": len(rows),
                        "holdout/exact_graph_fact_pass_rate": exact_report["summary"].get("exact_graph_fact_pass_rate"),
                        "holdout/mean_id_recall": exact_report["summary"].get("mean_id_recall"),
                        "holdout/mean_layer_recall": exact_report["summary"].get("mean_layer_recall"),
                        "holdout/predictions": table,
                    }
                )
            except Exception as exc:
                print(f"[WARNING] Failed to log post-training holdout to wandb: {exc}")
    trainer.accelerator.wait_for_everyone()
    cleanup_distributed()

if __name__ == "__main__":
    main()
