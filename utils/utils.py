import argparse
from transformers import HfArgumentParser

def make_run_name(script_args, peft_args, training_args, slurm_args):
    model_name = script_args.model_path.split("/")[-2]
    dataset_name = script_args.dataset_path.split("/")[-1].split(".")[0]
    gbs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * slurm_args.nnodes
    run_name = f"{model_name}-{dataset_name}-{peft_args.lora_r}lora-{gbs}gbs-{training_args.per_device_train_batch_size}mbs" \
        f"-{slurm_args.nnodes}nodes-{slurm_args.timeout}timeout-" \
        f"-{training_args.gradient_accumulation_steps}acc-{training_args.num_train_epochs}ep" \
        f"-{training_args.learning_rate}lr"
    return run_name