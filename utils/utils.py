import itertools
import os

import torch.distributed as dist

from runtime.world_model_training import (
    build_world_model_prompt_messages,
    serialize_sft_answer,
)

### FORMAT RECORDS FOR TRL
def build_prompt_completion_example(example):
    """Convert one S0 record to the TRL prompt and completion format."""

    system_prompt = example.get("system")
    question = example.get("question")
    answer = serialize_sft_answer(example.get("answer"))

    return {
        "answer": answer,
        "prompt": build_world_model_prompt_messages(
            system=system_prompt,
            question=question,
            metadata=example.get("metadata"),
            context=example.get("context"),
            in_context_examples=example.get("in_context_examples"),
        ),
        "completion": [{"role": "assistant", "content": answer}],
    }


def _plain_list(value):
    """Convert a tensor, array, or list to a Python list."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _first_batch_row(value, field_name):
    """Return the only row from one test batch."""

    value = _plain_list(value)
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Completion-only check received an empty {field_name}")
    if isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError(
                "Completion-only check expected one row in "
                f"{field_name}, but it received {len(value)} rows"
            )
        value = value[0]
    return value


def assert_completion_only_supervision(trainer, max_examples=8):
    """Confirm that the data collator masks all prompt tokens."""

    if getattr(trainer, "completion_only_loss", None) is not True:
        raise RuntimeError(
            "Completion-only check failed: trainer.completion_only_loss is not True"
        )

    dataset = trainer.train_dataset
    try:
        dataset_size = len(dataset)
    except TypeError:
        dataset_size = None

    if dataset_size == 0:
        raise RuntimeError("Completion-only check failed: the train dataset is empty")

    if dataset_size is None:
        examples = list(itertools.islice(iter(dataset), max_examples))
    else:
        examples = [
            dataset[index]
            for index in range(min(max_examples, dataset_size))
        ]
    if not examples:
        raise RuntimeError("Completion-only check failed: no examples are available")

    checked_prompt_tokens = 0
    checked_completion_tokens = 0
    for example_index, example in enumerate(examples):
        if "completion_mask" not in example:
            raise RuntimeError(
                "Completion-only check failed: tokenized example "
                f"{example_index} has no completion_mask"
            )

        completion_mask = _plain_list(example["completion_mask"])
        if not isinstance(completion_mask, list) or not completion_mask:
            raise RuntimeError(
                "Completion-only check failed: example "
                f"{example_index} has an empty completion_mask"
            )
        if any(value not in (0, 1, False, True) for value in completion_mask):
            raise RuntimeError(
                "Completion-only check failed: example "
                f"{example_index} has a nonbinary completion_mask"
            )

        batch = trainer.data_collator([example])
        if "labels" not in batch or "input_ids" not in batch:
            raise RuntimeError(
                "Completion-only check failed: the data collator omitted labels or input_ids"
            )
        labels = _first_batch_row(batch["labels"], "labels")
        input_ids = _first_batch_row(batch["input_ids"], "input_ids")
        token_count = len(completion_mask)
        if len(labels) < token_count or len(input_ids) < token_count:
            raise RuntimeError(
                "Completion-only check failed: the collated sequence is too short"
            )

        prompt_positions = [
            index for index, flag in enumerate(completion_mask) if not flag
        ]
        completion_positions = [
            index for index, flag in enumerate(completion_mask) if flag
        ]
        if not prompt_positions:
            raise RuntimeError(
                "Completion-only check failed: example "
                f"{example_index} has no prompt tokens"
            )
        if not completion_positions:
            raise RuntimeError(
                "Completion-only check failed: example "
                f"{example_index} has no completion tokens"
            )

        leaked_positions = [
            index for index in prompt_positions if labels[index] != -100
        ]
        if leaked_positions:
            raise RuntimeError(
                "Completion-only check failed: prompt tokens have labels in example "
                f"{example_index} at positions {leaked_positions[:8]}"
            )

        masked_completion_positions = [
            index for index in completion_positions if labels[index] == -100
        ]
        if masked_completion_positions:
            raise RuntimeError(
                "Completion-only check failed: completion tokens have masks in example "
                f"{example_index} at positions {masked_completion_positions[:8]}"
            )

        wrong_completion_labels = [
            index
            for index in completion_positions
            if labels[index] != input_ids[index]
        ]
        if wrong_completion_labels:
            raise RuntimeError(
                "Completion-only check failed: completion labels differ from input tokens "
                f"in example {example_index} at positions {wrong_completion_labels[:8]}"
            )

        checked_prompt_tokens += len(prompt_positions)
        checked_completion_tokens += len(completion_positions)

    return {
        "examples_checked": len(examples),
        "prompt_tokens_masked": checked_prompt_tokens,
        "completion_tokens_trainable": checked_completion_tokens,
    }


def build_formatting_func(tokenizer, train=True):
    def _fmt(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["question"]},
        ]
        if train:
            messages.append({"role": "assistant", "content": example["answer"]})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not train,
        )

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
    gbs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * slurm_args.nnodes
    run_name = f"{model_name}-{dataset_name}-{peft_args.lora_r}lora-{gbs}gbs-{training_args.per_device_train_batch_size}mbs" \
        f"-{slurm_args.nnodes}nodes-{slurm_args.timeout}timeout-" \
        f"-{training_args.gradient_accumulation_steps}acc-{training_args.num_train_epochs}ep" \
        f"-{training_args.learning_rate}lr"
    return run_name

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
