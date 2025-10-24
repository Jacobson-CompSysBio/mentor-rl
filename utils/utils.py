import argparse
from transformers import HfArgumentParser
import string

def make_run_name(script_args, peft_args, training_args, slurm_args):
    model_name = script_args.model_path.split("/")[-2]
    dataset_name = script_args.dataset_path.split("/")[-1].split(".")[0]
    gbs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * slurm_args.nnodes
    run_name = f"{model_name}-{dataset_name}-{peft_args.lora_r}lora-{gbs}gbs-{training_args.per_device_train_batch_size}mbs" \
        f"-{slurm_args.nnodes}nodes-{slurm_args.timeout}timeout-" \
        f"-{training_args.gradient_accumulation_steps}acc-{training_args.num_train_epochs}ep" \
        f"-{training_args.learning_rate}lr"
    return run_name

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