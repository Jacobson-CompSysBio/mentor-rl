import os
import sys
import re
import copy
import random
import itertools
import traceback
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import wandb

# Accelerate gives you full multi‑GPU (data parallel) and can work with model parallel setups.
from accelerate import Accelerator

# Transformers and PEFT imports.
from transformers import (
    AutoTokenizer,
    Llama4ForConditionalGeneration,
    LoraConfig,
    TaskType
)
from peft import get_peft_model

# Custom dataset utilities.
from utils.dataset import BasicEdgePredDataset, TransformedDataset

# =====================
# UTILITY FUNCTIONS
# =====================
def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

# Load environment variables.
load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
token = os.getenv("HUGGINGFACE_TOKEN")

# Model and data paths.
MODEL_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/'
MODEL_NAME = 'Llama-4-Scout-17B-16E-Instruct'
DATA_DIR = '../data/test/edge_tests.tsv'
log_dir = "../logs/"
checkpoint_dir = "../checkpoints/"

# Training configuration.
training_config = {
    'num_iterations': 1,
    'num_steps': 100,
    'batch_size': 1,              # Effective per-process batch size.
    'num_generations': 8,         # More generations per prompt.
    'max_completion_length': 512, # Longer generated sequences.
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1
}

# =====================
# PROMPT & ANSWER FUNCTIONS
# =====================
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it, answering 'yes' or 'no'. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
)

def extract_answer(text):
    parts = text.split("<answer>")
    if len(parts) < 2:
        return ""
    last_part = parts[-1]
    if "</answer>" not in last_part:
        return ""
    answer = last_part.split("</answer>")[0].strip().lower()
    return "" if answer == "..." or not answer else answer

def build_prompt(messages):
    return "\n".join(msg["content"].strip() for msg in messages)

# =====================
# DATASET PREPARATION
# =====================
def format_edgelist(filepath):
    edges = []
    with open(filepath, 'r') as f:
        # Skip header (first two lines).
        for line in itertools.islice(f, 2, None):
            split_text = line.strip().split()
            if len(split_text) >= 2:
                node1, node2 = split_text[:2]
                edges.append(f'(node {node1}, connected to, node {node2})')
    return ("You are given a graph in the form of triplets, e.g. (node 1, connected to, node 2). "
            "Answer the question related to the graph. This is the graph: " + ', '.join(edges))

def prepare_dataset(example):
    desc = format_edgelist(example["desc"])
    answer = example["answer"].replace('[','').replace(']','').replace('\'','').strip().lower()
    question = example["question"].replace("graph_n50_", "").replace('[','').replace(']','').replace('\'','')
    prompt_str = build_prompt([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{desc} {question}"}
    ])
    return {"prompt": prompt_str, "answer": answer}

# =====================
# EVALUATION FUNCTION
# =====================
def evaluate_model(model, tokenizer, eval_examples, accelerator):
    model.eval()
    correct = 0
    total = len(eval_examples)
    accelerator.print("="*50)
    accelerator.print("EVALUATION ON", total, "EXAMPLES")
    accelerator.print("="*50)

    for example in eval_examples:
        full_prompt = example["prompt"] if not isinstance(example["prompt"], list) else example["prompt"][0]
        expected = example["answer"] if not isinstance(example["answer"], list) else example["answer"][0]

        tokenized = tokenizer(full_prompt, return_tensors="pt")
        inputs = tokenized.input_ids.to(accelerator.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            predicted = extract_answer(response)
            if predicted == expected:
                correct += 1
            accelerator.print("\nPrompt:")
            accelerator.print(full_prompt)
            accelerator.print("\nExpected Answer:")
            accelerator.print(expected)
            accelerator.print("\nExtracted Answer:")
            accelerator.print(predicted)
            accelerator.print("\nFull Generated Response:")
            accelerator.print(response)
            accelerator.print("\nCorrect:", "✓" if predicted == expected else "✗")
            accelerator.print("-"*50)
        except Exception as e:
            accelerator.print("\nFailed to parse model output for prompt:")
            accelerator.print(full_prompt)
            accelerator.print("Error:", e)
    accuracy = (correct / total) * 100
    accelerator.print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    accelerator.print("="*50)
    model.train()
    return accuracy

# =====================
# REWARD FUNCTIONS
# =====================
def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        r_str = r.strip().lower() if r is not None else ""
        a_str = a.strip().lower() if a is not None else ""
        if r_str == a_str:
            rewards.append(2.0)
        elif a_str in r_str:
            rewards.append(1.5)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        think_open = response.count("<think>")
        think_close = response.count("</think>")
        answer_open = response.count("<answer>")
        answer_close = response.count("</answer>")
        trailing_text = response.split("</answer>")[-1].strip() if "</answer>" in response else ""
        if (think_open == 1 and think_close == 1 and answer_open == 1 and answer_close == 1 and trailing_text == ""
            and response.strip().endswith("</answer>")):
            score += 1.5
        else:
            score += 0.2 if (think_open == 1 and think_close == 1) else -0.1 * (abs(think_open-1)+abs(think_close-1))
            score += 0.2 if (answer_open == 1 and answer_close == 1) else -0.1 * (abs(answer_open-1)+abs(answer_close-1))
            if trailing_text:
                score -= 0.5
        rewards.append(score)
    return rewards

def combined_reward(prompts, completions, answer):
    c_scores = correctness_reward(prompts, completions, answer)
    f_scores = format_reward(completions)
    return [c + f for c, f in zip(c_scores, f_scores)]

# =====================
# GRPO FUNCTIONS (with generation restructured for smaller mini-batches)
# =====================
def selective_log_softmax(logits, input_ids):
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    seq_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (seq_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Process each prompt individually to limit per-call memory.
    Returns concatenated prompt and completion tensors.
    """
    all_pid, all_pmask, all_cid, all_cmask = [], [], [], []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        pid = inputs["input_ids"].to(model.device)
        pmask = inputs["attention_mask"].to(model.device)
        plen = pid.size(1)
        # Repeat prompt for num_generations.
        pid_gen = pid.repeat_interleave(num_generations, dim=0)
        pmask_gen = pmask.repeat_interleave(num_generations, dim=0)
        torch.cuda.empty_cache()
        # Save settings and disable gradient checkpointing temporarily.
        orig_cache = model.config.use_cache
        grad_ckpt_enabled = getattr(model, "is_gradient_checkpointing", False)
        model.config.use_cache = True
        if grad_ckpt_enabled:
            model.gradient_checkpointing_disable()
        try:
            outputs = model.generate(
                pid_gen,
                attention_mask=pmask_gen,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        except Exception as e:
            print(f"[ERROR] Generation error for prompt {i}: {e}")
            model.config.use_cache = orig_cache
            if grad_ckpt_enabled:
                model.gradient_checkpointing_enable()
            raise
        model.config.use_cache = orig_cache
        if grad_ckpt_enabled:
            model.gradient_checkpointing_enable()
        if outputs is None or outputs.ndim != 2:
            raise ValueError(f"[ERROR] Unexpected outputs for prompt {i}: {outputs}")
        cid = outputs[:, plen:]
        if cid.size(1) == 0:
            raise ValueError(f"[ERROR] No completion tokens for prompt {i}")
        cmask = create_completion_mask(cid, tokenizer.eos_token_id)
        all_pid.append(pid_gen)
        all_pmask.append(pmask_gen)
        all_cid.append(cid)
        all_cmask.append(cmask)
        print(f"[DEBUG] Prompt {i} processed: prompt length = {plen}, completion shape = {cid.shape}")
    return torch.cat(all_pid, dim=0), torch.cat(all_pmask, dim=0), torch.cat(all_cid, dim=0), torch.cat(all_cmask, dim=0)

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    if isinstance(batch_samples, dict):
        prompts = batch_samples["prompt"]
        answers = batch_samples["answer"]
    else:
        prompts = [sample["prompt"] for sample in batch_samples]
        answers = [sample["answer"] for sample in batch_samples]
    with torch.no_grad():
        pid, pmask, cid, cmask = generate_completions(model, tokenizer, prompts, num_generations, max_completion_length)
        input_ids = torch.cat([pid, cid], dim=1)
        attention_mask = torch.cat([pmask, cmask], dim=1)
        logits_to_keep = cid.size(1)
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in cid]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": cmask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

def grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta=0.01, epsilon=0.2):
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    cmask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(
            prompts=rollout_data["repeated_prompts"],
            completions=rollout_data["formatted_completions"],
            answer=rollout_data["repeated_answers"]
        ),
        dtype=torch.float32,
        device=token_log_probs.device
    )
    bs = rollout_data["batch_size"]
    ngen = rollout_data["num_generations"]
    rewards = rewards.view(bs, ngen)
    avg_reward = rewards.mean().item()
    mean_rewards = rewards.mean(dim=1).repeat_interleave(ngen)
    std_rewards = rewards.std(dim=1).repeat_interleave(ngen)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    per_token_loss = surrogate_loss - beta * kl
    loss = -((per_token_loss * cmask).sum(dim=1) / cmask.sum(dim=1)).mean()
    return loss, avg_reward

def train_with_grpo(model, tokenizer, train_data, num_iterations=1, num_steps=500,
                    batch_size=4, num_generations=4, max_completion_length=128,
                    beta=0.1, learning_rate=5e-6, mu=3, epsilon=0.2, reward_function=None, accelerator=None):
    for iteration in range(num_iterations):
        accelerator.print(f"\nIteration {iteration+1}/{num_iterations}")
        # Create a reference model for KL penalty.
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        accelerator.print("Reference model created.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()
        for step, batch in enumerate(train_data):
            if step >= num_steps:
                break
            with torch.no_grad():
                rollout_data = generate_rollout_data(model, ref_model, tokenizer, batch, num_generations, max_completion_length)
            for grpo_iter in range(mu):
                loss, avg_reward = grpo_loss(model, ref_model, rollout_data, tokenizer, reward_function, beta, epsilon)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "iteration": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })
                accelerator.print(
                    f"\rIteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                    f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}, reward: {avg_reward:.4f}",
                    end="",
                    flush=True
                )
            accelerator.print()
    return model

# =====================
# MAIN FUNCTION (using Accelerator for full model & data parallelism)
# =====================
def main():
    accelerator = Accelerator()  # Initializes distributed setup.
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # For full model parallelism, you can supply a max_memory dictionary.
    max_memory = {i: "60GB" for i in range(8)}

    accelerator.print("Downloading model...")
    model = Llama4ForConditionalGeneration.from_pretrained(
        os.path.join(MODEL_DIR, MODEL_NAME),
        device_map="auto",  # Allows model parallelism.
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        token=token
    )
    accelerator.print("Model downloaded.")

    # Setup LoRA / QLoRA.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    accelerator.print("Wrapping model with QLoRA adapters...")
    model = get_peft_model(model, lora_config)
    accelerator.print("Model wrapped with QLoRA adapters.")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, MODEL_NAME), padding_side="left", token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False              # Disable caching during training.
    model.gradient_checkpointing_enable()       # Enable gradient checkpointing for memory saving.
    model.train()
    model.config.eos_token_id = tokenizer.eos_token_id

    # Build transformed dataset.
    transformed_dataset = TransformedDataset(BasicEdgePredDataset(DATA_DIR), prepare_dataset)
    eval_size = 10
    eval_idxs = list(range(eval_size))
    train_idxs = list(range(eval_size, len(transformed_dataset)))
    eval_dataset = Subset(transformed_dataset, eval_idxs)
    train_dataset = Subset(transformed_dataset, train_idxs)
    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    # Prepare model, optimizer, and data loaders for distributed training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"])
    model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    accelerator.print("\nInitial model evaluation before finetuning:")
    pre_acc = evaluate_model(model, tokenizer, eval_loader, accelerator)
    accelerator.print(f"Pre-GRPO Accuracy: {pre_acc:.2f}%")

    accelerator.print("\nStarting RL fine-tuning using GRPO...")
    wandb.init(project=os.environ["WANDB_PROJECT"],
               dir=os.path.join(log_dir, "grpo"),
               entity=os.environ["WANDB_ENTITY"],
               reinit=True)
    accelerator.print("Weights & Biases initialized.")

    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_data=train_loader,
        reward_function=combined_reward,
        **training_config,
        accelerator=accelerator
    )

    wandb.finish()
    accelerator.print("Training completed and wandb run finished.")

    accelerator.print("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_acc = evaluate_model(model, tokenizer, eval_loader, accelerator)
    accelerator.print(f"Post-GRPO Accuracy: {post_acc:.2f}%")
    accelerator.print(f"Accuracy improvement: {post_acc - pre_acc:.2f}")

    accelerator.print("\nSaving GRPO fine-tuned model...")
    # Make sure to unwrap the model (only the main process should save)
    model_to_save = accelerator.unwrap_model(model)
    model_to_save.save_pretrained(os.path.join(MODEL_DIR, "GRPO-EdgePred-Vanilla"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "GRPO-EdgePred-Vanilla"))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
