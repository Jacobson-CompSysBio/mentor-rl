import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from accelerate import Accelerator, DeepSpeedPlugin, PartialState
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.utils import check_accuracy


def _collate_single(examples):
    """
    Unwrap a single example from the dataloader list.
    """
    return examples[0]


def build_formatting_func(tokenizer, train=True):
    system_prompt = (
        "You are a helpful biological chatbot. You will be given a biological question; "
        "return the correct answer."
    )

    def _fmt(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ] if train else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ]
        if train:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return _fmt


def infer(
    model,
    tokenizer,
    format_fn,
    dataset,
    accelerator=None,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
):
    if accelerator is None:
        local_cfg_dir = os.getenv("LOCAL_CONFIG_DIR")
        if local_cfg_dir is None:
            raise RuntimeError("LOCAL_CONFIG_DIR must be set to use the default inference accelerator.")
        with open(os.path.join(local_cfg_dir, "ds_zero3_inference.json")) as f:
            ds_infer_cfg = json.load(f)
        ds_inference_plugin = DeepSpeedPlugin(hf_ds_config=ds_infer_cfg)
        inference_accelerator = Accelerator(deepspeed_plugin=ds_inference_plugin)
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
    if accelerator is None:
        inference_model, dataloader = inference_accelerator.prepare(inference_model, dataloader)
    else:
        dataloader = inference_accelerator.prepare(dataloader)

    results = []
    with torch.no_grad():
        for _, example in enumerate(dataloader):
            formatted = format_fn(example)
            inputs = tokenizer(formatted, return_tensors="pt").to(inference_model.device)
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
            results.append(text)

    # clear memory, free cache
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


def run_inference(args: InferenceArguments):
    state = PartialState()
    rank = state.process_index

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    model.use_cache = False
    model.config.use_cache = False
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    outputs = infer(
        model,
        tokenizer,
        format_fn,
        inf_ds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    score = check_accuracy(outputs, list(inf_ds["answer"]))
    if isinstance(score, list):
        score = np.mean(score)

    return outputs, score


def main():
    parser = HfArgumentParser((InferenceArguments,))
    script_args = parser.parse_args_into_dataclasses()[0]
    state = PartialState()
    rank = state.process_index

    outputs, score = run_inference(script_args)

    if rank == 0:
        print(f"Inference complete. Average Score={score:.2%}")
        print("Outputs:", outputs)


if __name__ == "__main__":
    main()
