import os
import sys
import traceback

from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, TaskType
import torch
from transformers import Llama4ForConditionalGeneration
from trl import SFTConfig, SFTTrainer

MODEL_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/"
MODEL_NAME = "Llama-4-Scout-17B-16E-Instruct"
DATA_DIR = "/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/data/"

# train config #
USE_PEFT = True
################


def main():
    lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
    )
    device = PartialState().process_index
    model = Llama4ForConditionalGeneration.from_pretrained(
            os.path.join(MODEL_DIR, MODEL_NAME),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": device},
            # flash attention?
    )
    dataset = load_data_formatted("qa_pairs.json")
    train_config = SFTConfig(
        output_dir=MODEL_DIR,
        learning_rate=1e-4,
        num_train_epochs=2,
        max_length=512,
        warmup_steps=1000,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=["full_shard", "offload", "auto_wrap"],
        report_to="wandb",
        run_name="mentor-sft",
        max_steps=1 # FOR DEBUG
    )
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=train_config,
        peft_config=lora_config if USE_PEFT else None
    )
    trainer.train()


def format_dataset(row):
    # TRL wants each row in the format:
    # { messages: [ {role:r, content:c}, {role:r, content:c} ] }
    row_q = {"role": "user", "content": row["question"]}
    row_a = {"role": "assistant", "content": row["answer"]}
    row_formatted = {"messages": [row_q, row_a]}
    return row_formatted


def load_data_formatted(file: str):
    raw = load_dataset("json", data_dir=DATA_DIR, data_files=file)
    split = raw["train"].train_test_split(test_size=0.1)
    formatted = split.map(format_dataset, remove_columns=split['train'].column_names)
    return formatted


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
