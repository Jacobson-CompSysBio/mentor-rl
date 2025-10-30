import argparse
from transformers import HfArgumentParser

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