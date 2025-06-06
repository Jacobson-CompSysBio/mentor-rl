import os
import sys
import traceback
import wandb
from dotenv import load_dotenv
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, TaskType
import torch
from transformers import Llama4ForConditionalGeneration, TrainingArguments
from trl import SFTConfig, SFTTrainer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

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
    # get main process 
    state = PartialState()
    is_main = state.process_index == 0

    # load dataset
    if is_main:
        print("loading dataset...")
    dataset = load_dataset("json", data_dir=DATA_DIR, data_files="qa_pairs.json")
    dataset = dataset.map(format_dataset, remove_columns=['question', 'answer'])
    if is_main:
        print("dataset loaded. initializing wandb...")

    # init wandb on main 
    if is_main:
        wandb.init(project="mentor-sft",
                   entity=os.getenv("WANDB_ENTITY")
                   )
    if is_main:
        print("loading model...")
    lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
    )
    # device_string = PartialState().process_index
    model = Llama4ForConditionalGeneration.from_pretrained(
            os.path.join(MODEL_DIR, MODEL_NAME),
            torch_dtype=torch.bfloat16,
    )

    # set max position embeddings to avoid OOM
    model.config.max_position_embeddings=1024

    if is_main:
        print("model loaded.")

    train_config = SFTConfig(
        fsdp_config={'cpu_ram_efficient_loading': True,
                     'sync_module_states': True,
                     'offload_params': True,
                     'mixed_precision': "bf16",
                     'activation_checkpointing': True,
                     'min_num_params': 1e9,
                     'backward_prefetch': 'backward_pre',
                     'limit_all_gathers': True
                     },
        report_to="wandb",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_seq_length=1024,
        logging_strategy="steps",
        logging_steps=10,
        remove_unused_columns=True,
        fsdp=fsdp_config
    )

    if is_main:
        print("training model...")
    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        args=train_config,
        peft_config=lora_config if USE_PEFT else None
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
