import argparse
from transformers import HfArgumentParser

def make_run_name(script_args, peft_args, training_args):
    model_name = script_args.model_path.split("/")[-2]
    dataset_name = script_args.dataset_path.split("/")[-1].split(".")[0]
    run_name = f"{model_name}-{dataset_name}-lora{peft_args.lora_r}-bs{training_args.per_device_train_batch_size}-acc{training_args.gradient_accumulation_steps}-ep{training_args.num_train_epochs}-lr{training_args.learning_rate}"
    return run_name