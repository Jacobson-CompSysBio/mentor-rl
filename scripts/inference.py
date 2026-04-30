import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import build_formatting_func, check_accuracy, flatten_gathered_objects


def _collate_single(examples):
    return examples[0]


def _load_deepspeed_config(config_path: Optional[str] = None):
    if config_path is None:
        local_cfg_dir = os.getenv("LOCAL_CONFIG_DIR")
        if local_cfg_dir is None:
            raise RuntimeError("LOCAL_CONFIG_DIR must be set when --deepspeed_config is not provided.")
        config_path = os.path.join(local_cfg_dir, "ds_zero3_inference.json")

    with open(config_path) as f:
        return json.load(f)


def _create_inference_accelerator(config_path: Optional[str] = None):
    ds_infer_cfg = _load_deepspeed_config(config_path)
    hf_ds_config = HfDeepSpeedConfig(ds_infer_cfg)
    ds_inference_plugin = DeepSpeedPlugin(hf_ds_config=ds_infer_cfg)
    return Accelerator(deepspeed_plugin=ds_inference_plugin), hf_ds_config


def infer(
    model,
    tokenizer,
    format_fn,
    dataset,
    accelerator=None,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    return_indices=False,
    prepare_model=False,
):
    if accelerator is None:
        inference_accelerator, _ = _create_inference_accelerator()
        prepare_model = True
    else:
        inference_accelerator = accelerator
    inference_accelerator.print("Accelerator loaded.")

    inference_model = model
    inference_model.eval()
    inference_accelerator.print("Inference model loaded.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_single,
    )

    # accelerator.prepare model for distributed inference
    if prepare_model:
        inference_model, dataloader = inference_accelerator.prepare(inference_model, dataloader)
    else:
        dataloader = inference_accelerator.prepare(dataloader)

    results = []
    with torch.no_grad():
        for _, example in enumerate(dataloader):
            formatted = format_fn(example)
            inputs = tokenizer(formatted, return_tensors="pt").to(inference_accelerator.device)
            input_len = inputs["input_ids"].shape[1]

            output = inference_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                synced_gpus=True,
            )

            new_ids = output[0][input_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            if return_indices:
                results.append({"idx": int(example["_sample_idx"]), "prediction": text})
            else:
                results.append(text)

    # clear memory, free cache
    inference_accelerator.wait_for_everyone()
    del inference_model, dataloader, inference_accelerator
    torch.cuda.empty_cache()
    return results


@dataclass
class InferenceArguments:
    model_path: str = field(metadata={"help": "Hugging Face model ID or local path to the model"})
    dataset_path: str = field(
        default="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/qa_pairs.json",
        metadata={"help": "Local dataset path"},
    )
    dataset_subset_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to use from the dataset. If None, uses the full dataset."},
    )
    sample_size: int = field(
        default=20,
        metadata={"help": "Number of examples to run inference on."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed used to shuffle the dataset before selecting samples."},
    )
    max_new_tokens: int = field(
        default=50,
        metadata={"help": "Maximum new tokens to generate for each example."},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for generation."},
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p nucleus sampling value for generation."},
    )
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the DeepSpeed ZeRO-3 inference config."},
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON path for rank 0 to write inference outputs and score."},
    )

def run_inference(args: InferenceArguments):
    accelerator, ds_config_ref = _create_inference_accelerator(args.deepspeed_config)
    rank = accelerator.process_index

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token_id >= tokenizer.vocab_size:
        if rank == 0:
            print(
                f"[WARNING] pad_token_id ({tokenizer.pad_token_id}) >= vocab_size ({tokenizer.vocab_size}), "
                "setting to eos_token_id"
            )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    format_fn = build_formatting_func(tokenizer, train=False)

    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    if args.dataset_subset_size is not None:
        dataset = dataset.select(range(args.dataset_subset_size))
    dataset = dataset.shuffle(seed=args.seed)

    sample_size = min(args.sample_size, len(dataset))
    inf_ds = dataset.select(range(sample_size))
    inf_ds = inf_ds.add_column("_sample_idx", list(range(sample_size)))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = True
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    local_records = infer(
        model,
        tokenizer,
        format_fn,
        inf_ds,
        accelerator=accelerator,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        return_indices=True,
        prepare_model=True,
    )

    gathered_records = list(flatten_gathered_objects(gather_object(local_records)))
    predictions_by_index = {}
    for record in gathered_records:
        if not isinstance(record, dict):
            continue
        idx = int(record["idx"])
        if 0 <= idx < sample_size and idx not in predictions_by_index:
            predictions_by_index[idx] = record["prediction"]

    missing = [idx for idx in range(sample_size) if idx not in predictions_by_index]
    if missing and rank == 0:
        print(f"[WARNING] Missing predictions for sample indices: {missing}")

    ordered_indices = sorted(predictions_by_index)
    outputs = [predictions_by_index[idx] for idx in ordered_indices]
    answers = [inf_ds[idx]["answer"] for idx in ordered_indices]

    score = check_accuracy(outputs, answers)
    if isinstance(score, list):
        score = np.mean(score)

    _ = ds_config_ref
    return outputs, score


def main():
    parser = HfArgumentParser((InferenceArguments,))
    script_args = parser.parse_args_into_dataclasses()[0]
    rank = int(os.environ.get("RANK", "0"))

    outputs, score = run_inference(script_args)

    if rank == 0:
        print(f"Inference complete. Average Score={score:.2%}")
        print("Outputs:", outputs)
        if script_args.output_path is not None:
            output_dir = os.path.dirname(script_args.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(script_args.output_path, "w") as f:
                json.dump({"score": float(score), "outputs": outputs}, f, indent=2)
            print(f"Inference results written to {script_args.output_path}")


if __name__ == "__main__":
    main()
