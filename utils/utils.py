import argparse
import itertools
import os
import string

import torch.distributed as dist
from transformers import HfArgumentParser

SYSTEM_PROMPT = (
    "You are a helpful biological chatbot. You will be given a biological question; "
    "return the correct answer."
)


def build_prompt_completion_example(example):
    """Convert one graph-QA record to TRL's conversational prompt/completion form.

    Keeping the assistant turn in ``completion`` (rather than rendering the whole
    conversation into a single text field) lets ``SFTTrainer`` construct a
    completion mask and exclude the system/user prompt from the language-model
    loss.
    """

    system_prompt = example.get("system") or SYSTEM_PROMPT
    question = example.get("question")
    answer = example.get("answer")
    for field_name, value in (
        ("system", system_prompt),
        ("question", question),
        ("answer", answer),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"SFT record field {field_name!r} must be a non-empty string")

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "completion": [{"role": "assistant", "content": answer}],
    }


def _plain_list(value):
    """Return tensors/arrays/lists as plain Python lists for mask checks."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _first_batch_row(value, field_name):
    value = _plain_list(value)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Completion-only preflight received an empty {field_name}")
    if isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError(
                f"Completion-only preflight expected a one-example {field_name} batch, got {len(value)} rows"
            )
        value = value[0]
    return value


def assert_completion_only_supervision(trainer, max_examples=8):
    """Fail fast unless the trainer's actual collator masks every prompt token.

    This checks post-tokenization examples from ``trainer.train_dataset`` and
    labels emitted by ``trainer.data_collator``.  It therefore guards the exact
    path used for training, including chat templating, truncation, and packing,
    instead of merely trusting a configuration flag.
    """

    if getattr(trainer, "completion_only_loss", None) is not True:
        raise RuntimeError(
            "SFT completion-only preflight failed: trainer.completion_only_loss is not True"
        )

    dataset = trainer.train_dataset
    try:
        dataset_size = len(dataset)
    except TypeError:
        dataset_size = None

    if dataset_size == 0:
        raise RuntimeError("SFT completion-only preflight failed: training dataset is empty")

    if dataset_size is None:
        examples = list(itertools.islice(iter(dataset), max_examples))
    else:
        examples = [dataset[index] for index in range(min(max_examples, dataset_size))]
    if not examples:
        raise RuntimeError("SFT completion-only preflight failed: no training examples were inspectable")

    checked_prompt_tokens = 0
    checked_completion_tokens = 0
    for example_index, example in enumerate(examples):
        if "completion_mask" not in example:
            raise RuntimeError(
                "SFT completion-only preflight failed: tokenized example "
                f"{example_index} has no completion_mask; use prompt/completion dataset columns"
            )

        completion_mask = _plain_list(example["completion_mask"])
        if not isinstance(completion_mask, list) or not completion_mask:
            raise RuntimeError(
                f"SFT completion-only preflight failed: example {example_index} has an empty completion_mask"
            )
        if any(value not in (0, 1, False, True) for value in completion_mask):
            raise RuntimeError(
                f"SFT completion-only preflight failed: example {example_index} has a non-binary completion_mask"
            )

        batch = trainer.data_collator([example])
        if "labels" not in batch or "input_ids" not in batch:
            raise RuntimeError(
                "SFT completion-only preflight failed: data collator did not return input_ids and labels"
            )
        labels = _first_batch_row(batch["labels"], "labels")
        input_ids = _first_batch_row(batch["input_ids"], "input_ids")
        token_count = len(completion_mask)
        if len(labels) < token_count or len(input_ids) < token_count:
            raise RuntimeError(
                "SFT completion-only preflight failed: collated sequence is shorter than its completion mask"
            )

        prompt_positions = [index for index, flag in enumerate(completion_mask) if not flag]
        completion_positions = [index for index, flag in enumerate(completion_mask) if flag]
        if not prompt_positions:
            raise RuntimeError(
                f"SFT completion-only preflight failed: example {example_index} has no prompt tokens"
            )
        if not completion_positions:
            raise RuntimeError(
                "SFT completion-only preflight failed: example "
                f"{example_index} has no completion tokens after tokenization/truncation"
            )

        leaked_positions = [index for index in prompt_positions if labels[index] != -100]
        if leaked_positions:
            raise RuntimeError(
                "SFT completion-only preflight failed: prompt/system/user tokens have trainable labels "
                f"in example {example_index} at positions {leaked_positions[:8]}"
            )

        masked_completion_positions = [index for index in completion_positions if labels[index] == -100]
        if masked_completion_positions:
            raise RuntimeError(
                "SFT completion-only preflight failed: assistant completion tokens are masked "
                f"in example {example_index} at positions {masked_completion_positions[:8]}"
            )
        wrong_completion_labels = [
            index for index in completion_positions if labels[index] != input_ids[index]
        ]
        if wrong_completion_labels:
            raise RuntimeError(
                "SFT completion-only preflight failed: assistant labels differ from input tokens "
                f"in example {example_index} at positions {wrong_completion_labels[:8]}"
            )

        checked_prompt_tokens += len(prompt_positions)
        checked_completion_tokens += len(completion_positions)

    return {
        "examples_checked": len(examples),
        "prompt_tokens_masked": checked_prompt_tokens,
        "completion_tokens_trainable": checked_completion_tokens,
    }


def build_formatting_func(
    tokenizer,
    train=True,
    *,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
):
    def _fmt(example):
        system_prompt = example.get("system") or SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ]
        if train:
            messages.append({"role": "assistant", "content": example["answer"]})

        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": not train,
        }
        if not train and enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if reasoning_effort is not None:
            template_kwargs["reasoning_effort"] = reasoning_effort

        return tokenizer.apply_chat_template(messages, **template_kwargs)

    return _fmt

def flatten_gathered_objects(values):
    for value in values:
        if isinstance(value, list):
            yield from flatten_gathered_objects(value)
        else:
            yield value

### RUN NAME
def make_run_name(script_args, peft_args, training_args, slurm_args):
    
    # parse model, dataset names
    model_name = script_args.model_path.split("/")[-2]
    dataset_name = script_args.dataset_path.split("/")[-1].split(".")[0]

    # gbs
    world_size = int(getattr(slurm_args, "ntasks", getattr(slurm_args, "nnodes", 1)))
    gbs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    parts = [
        model_name,
        dataset_name,
        f"{peft_args.lora_r}lora",
        f"{gbs}gbs",
        f"{training_args.per_device_train_batch_size}mbs",
        f"{slurm_args.nnodes}nodes",
        f"{world_size}ranks",
        f"{slurm_args.timeout}timeout",
        f"{training_args.gradient_accumulation_steps}acc",
        f"{training_args.num_train_epochs}ep",
        f"{training_args.learning_rate}lr",
    ]
    if training_args.max_steps and training_args.max_steps > 0:
        parts.append(f"{training_args.max_steps}steps")
    if script_args.dataset_subset_size:
        parts.append(f"{script_args.dataset_subset_size}subset")
    run_label = os.environ.get("MENTOR_RUN_LABEL")
    if run_label:
        parts.insert(0, run_label)
    return "-".join(parts)

def make_grpo_run_name(script_args, peft_args, grpo_args, slurm_args):
    """
    Make a name for GRPO wandb runs with necessary params
    """
    # parse model, dataset
    model_name = script_args.model_path.split("/")[-2]
    dataset_name = script_args.dataset_path.split("/")[-1].split(".")[0]

    # global batch size calc
    gbs = grpo_args.per_device_train_batch_size * grpo_args.gradient_accumulation_steps * slurm_args.nnodes

    # compose name
    run_name = (
        f"{model_name}-{dataset_name}-"
        f"{peft_args.lora_r}lora-"
        f"{gbs}gbs-{grpo_args.per_device_train_batch_size}mbs-"
        f"{slurm_args.nnodes}nodes-{slurm_args.timeout}timeout-"
        f"{grpo_args.gradient_accumulation_steps}acc-"
        f"{grpo_args.num_train_epochs}ep-"
        f"{grpo_args.learning_rate}lr-"
        f"{grpo_args.num_generations}ngen-"
        f"{grpo_args.max_completion_length}mcl-"
        #f"{grpo_args.loss_type}loss-"
        f"{grpo_args.w_format}wf-{grpo_args.w_task}wt"
    )
    
    return run_name

def get_rank_world_size():
    # Use env first (works before PG init)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # If PG happens to be ready already, prefer that
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world = dist.get_world_size()
    return rank, world

### INFERENCE
def check_accuracy(
    preds: list[str],
    targets: list[str]
) -> list[float]:
    # split into unique non-trivial words
    pred_w = [set(_clean_and_split(p)) - _TRIVIAL for p in preds]
    target_w = [set(_clean_and_split(t)) - _TRIVIAL for t in targets]

    # extract words present in both preds and targets
    overlap = [p & t for p, t in zip(pred_w, target_w)]

    # compute ratio of present to total words
    accuracy = [len(o) / len(t) for o, t in zip(overlap, target_w)]

    return accuracy

def check_numeric_accuracy(
    preds: list[str],
    targets: list[float]
) -> list[float]:
    # extract numbers from preds as floats
    pred_n = [_extract_num(p) for p in preds]

    # check similarity with targets
    similiarty = [
        [_inv_sq_sim(q, t) for q in p]
        for p, t in zip(pred_n, targets)
    ]

    # take most accurate prediction
    accuracy = [max(s) for s in similiarty]

    return accuracy

_TRIVIAL = {
    "it", "its", "they", "their",
    "that", "this", "which", "is",
    "are", "were", "be", "to",
    "a", "an", "the", "some",
    "as", "and", "also",
}

_NOPUNC = str.maketrans("", "", string.punctuation)

def _clean_and_split(
    s: str
) -> list[str]:
    return (
        s.lower() # all lowercase
         .translate(_NOPUNC) # remove punctuation
         .split() # split words by whitespace
    )

def _extract_num(
    s: str
) -> list[float]:
    nums = []
    for w in s.split():
        try:
            nums.append(float(w))
        except:
            pass
    return nums

def _inv_sq_sim(
    a: float,
    b: float
) -> float:
    return 1 / ((a-b)**2 + 1)
