import os
import sys
import traceback
import wandb
from dotenv import load_dotenv
from accelerate import PartialState, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import DistributedType
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import Llama4ForConditionalGeneration, TrainingArguments
from trl import SFTConfig, SFTTrainer
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy

MODEL_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/"
MODEL_NAME = "Llama-4-Scout-17B-16E-Instruct"
DATA_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/data/"

# train config #
USE_PEFT = True 
################

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_PROJECT"] = "mentor-sft"

def format_dataset(row):
    # TRL wants each row in the format:
    # { messages: [ {role:r, content:c}, {role:r, content:c} ] }
    row_q = {"role": "user", "content": row["question"]}
    row_a = {"role": "assistant", "content": row["answer"]}
    row['messages'] = [row_q, row_a]
    return row

def load_data_formatted(file: str):
    raw = load_dataset("json", data_dir=DATA_DIR, data_files=file)
    split = raw["train"].train_test_split(test_size=0.1)
    formatted = split.map(format_dataset, remove_columns=split['train'].column_names)
    return formatted

def main():
    # get main process state 
    state = PartialState()
    is_main = state.process_index == 0
    checkpoint = os.path.join(MODEL_DIR, MODEL_NAME)

    # init llama with empty weights and dispatch
    with init_empty_weights():
        model = Llama4ForConditionalGeneration.from_pretrained(
                checkpoint,
                torch_dtype=torch.bfloat16,
                )
    model = load_checkpoint_and_dispatch(
            model, checkpoint,
            device_map={"": "cpu"},
            no_split_module_classes=["LlamaDecoderLayer"]
            )

    # peft loading
    if USE_PEFT:
        if is_main:
            print("using PEFT...")
        target = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=target,
                task_type=TaskType.CAUSAL_LM,
                )
        model = get_peft_model(model, lora_config)
        if is_main:
            print("trainable params:", model.print_trainable_parameters())

    if is_main:
        print("model loaded.")

    # load dataset
    if is_main:
        print("loading dataset...")
    dataset = load_dataset("json", data_dir=DATA_DIR, data_files="qa_pairs.json")
    dataset = dataset.map(format_dataset, remove_columns=['question', 'answer'])
    if is_main:
        print("dataset loaded. initializing wandb...")

    # set max position embeddings to avoid OOM
    model.config.max_position_embeddings=1024
    fsdp_cfg = dict(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=transformer_auto_wrap_policy,
            offload_params=True,
            use_orig_params=True,
            cpu_ram_efficient_loading=True,
            sync_module_states=True,
            backward_prefetch="backward_pre",
            limit_all_gathers=True,
            )

    train_config = SFTConfig(
        use_cpu=True,
        fsdp=["full_shard","auto_wrap"],
        fsdp_config=fsdp_cfg,
        bf16=True,
        report_to="wandb",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_seq_length=1024,
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=True,
    )

    if is_main:
        print("training model...")
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        args=train_config,
        #peft_config=lora_config if USE_PEFT else None
    )
    trainer.train()
    if is_main:
        print("training complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
