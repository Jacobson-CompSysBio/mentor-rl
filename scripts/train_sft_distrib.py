# import necessary packages
import sys, os
import torch 
import wandb
import traceback
import numpy as np
from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorForSeq2Seq,
                          DataCollatorForLanguageModeling,
                          Llama4ForConditionalGeneration,
                          get_scheduler)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.plugins import FullyShardedDataParallelPlugin


# set proper root path
sys.path.append(str(Path(__file__).resolve().parent.parent))

load_dotenv('/lustre/orion/syb111/proj-shared/Projects/mentor-rl/.env')
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
token = os.getenv("HUGGINGFACE_TOKEN")

MODEL_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/'
MODEL_NAME = 'Llama-4-Scout-17B-16E-Instruct'
SAVE_NAME = 'Llama4-Scout-EnsemblSFT' 
DATA_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/data/'

accelerator.print(f"Getting training config...")
# set up train config
train_config = {
        "model": MODEL_NAME,
        "dataset": "ensembl",
        "peft": True,
        "micro_batch_size": 1,
        "num_epochs": 2,
        "max_lr": 1e-4,
        "grad_accum_steps": 32,
        }

def main():
    # init accelerator for distributed training
    accelerator=Accelerator(
            gradient_accumulation_steps=training_config["grad_accum_steps"],
            mixed_precision="bf16",
            fsdp_plugin=FullyShardedDataParallelPlugin(),
            )
    device = accelerator.device
    is_main = accelerator.is_main_process
    if is_main:
        accelerator.print(f"Accelerator distributed type: {accelerator.state.distributed_type}")

    accelerator.print(f"Loading dataset")
    # load dataset (we actually don't want to split this data since we want to ingest all ensembls)
    raw_dataset = load_dataset('json', data_dir=DATA_DIR, data_files="qa_pairs.json")
    dataset = raw_dataset['train'].train_test_split(test_size=0.1)
    accelerator.print("Data loaded.")

    # load tokenizer and model
    accelerator.print(f"Loading model: {train_config['model']}")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR + MODEL_NAME), token=token)
    model = Llama4ForConditionalGeneration.from_pretrained(os.path.join(MODEL_DIR + MODEL_NAME),
                                                           torch_dtype=torch.bfloat16)
    print(f"Model loaded.")

    # apply peft (optional)
    if train_config["peft"]:
        accelerator.print("Initializing with LoRA...")
        lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"]
                )
        model = get_peft_model(model, lora_config)
        accelerator.print("LoRA applied.")

    ## PREPROCESSING
    # convert to chat format
    def format_chat(row):
        row_json_inp = [{"role": "user", "content": row["question"]}]
        row_json_out = [{"role": "assistant", "content": row["answer"]}]
        row["prompt"] = tokenizer.apply_chat_template(row_json_inp, tokenize=False)
        row["response"] = tokenizer.apply_chat_template(row_json_out, tokenize=False)
        return row

    # tokenize data
    def preprocess_data(example):
        # concat to get full text
        full_text = example["prompt"] + example["response"]
        # tokenize
        tokenized = tokenizer(full_text,
                              truncation=True,
                              max_length=256,
                              add_special_tokens=False
                              )
        # loss masking
        prompt_tokenized = tokenizer(example["prompt"],
                                     truncation=True,
                                     max_length=512,
                                     add_special_tokens=False
                                     )
        prompt_length = len(prompt_tokenized["input_ids"])
        labels = tokenized["input_ids"].copy()
        labels[:prompt_length] = [-100] * prompt_length
        tokenized['labels'] = labels

        return tokenized
    
    # format dataset
    accelerator.print("Preprocessing data..")
    formatted_dataset = dataset.map(format_chat,
                                    remove_columns=dataset['train'].column_names)
    tokenized_dataset = formatted_dataset.map(preprocess_data,
                                              remove_columns=["prompt", "response"])

    accelerator.print("Dataset processed.")

    ## MODELING
    # create dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, return_tensors="pt")
    train_dataloader = DataLoader(tokenized_dataset['train'],
                                  collate_fn=data_collator,
                                  batch_size=train_config['batch_size']
                                  )
    val_dataloader = DataLoader(tokenized_dataset['test'],
                                collate_fn=data_collator,
                                batch_size=train_config['batch_size']
                                )

    # init optimizers
    optimizer = AdamW(model.parameters(), lr=train_config["max_lr"])

    num_epochs = train_config['num_epochs']
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
            'linear',
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps=num_training_steps
            )

    # prepare objects for distribution
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    
    if is_main:
        print("Initializing wandb...")
        # init wandb
        wandb.init(project="mentor-sft",
                   entity=os.environ["WANDB_ENTITY"],
                   reinit=True,
                   config = train_config)
    
    accelerator.print("Training...")
    global_step = 0
    # train loop
    for epoch in range(num_epochs):
        
        # training
        model.train()
        epoch_train_loss = 0.0
        accum_loss = 0.0
        micro_batches = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model)
                # map to cuda
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                # fwd
                outputs = model(**batch)
                loss = outputs.loss
                accum_loss += loss.item()
                micro_batches += 1

                # backward
                accelerator.backward(loss)
            
            # track + step opt
            if accelerator.sync_gradients():
                # step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # log
                avg_loss = accum_loss / micro_batches
                wandb.log({
                    "train_loss": avg_loss,
                    "g_batch": global_step + 1,
                    })
        # eval
        model.eval()
        val_loss_total = 0.0
        val_count = 0
        for batch in val_dataloader:
            # map to cuda
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # fwd
            with torch.no_grad():
                outputs = model(**batch)
            
            # track
            loss = outputs.loss.item()
            val_loss_total += loss
            val_count += 1
        avg_val_loss = 
        
        # Log one averaged validation loss per epoch
        wandb.log({
            "val_loss": avg_val_loss,
            "epoch": epoch + 1
            })
        accelerator.print(f"Epoch {epoch+1}: train_loss = {avg_loss:.4f}, val_loss = {avg_val_loss:.4f}")

    ## CHECKPOINTING
    accelerator.print("Training complete! Saving...")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(MODEL_DIR + SAVE_NAME))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR + SAVE_NAME))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
