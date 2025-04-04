import os
import random
import re
import itertools
import copy
import sys
import traceback

import torch
import torch.nn.functional as F
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb  # Assumes wandb is installed

# For mixed precision
from torch.cuda.amp import autocast
from accelerate import Accelerator, FullyShardedDataParallelPlugin

sys.path.append(str(Path(__file__).resolve().parent.parent))

# custom
from utils.dataset import BasicEdgePredDataset, TransformedDataset


# ---------------------------
# SEEDING & ENVIRONMENT SETUP
# ---------------------------
def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "default_project")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY", "default_entity")
token = os.getenv("HUGGINGFACE_TOKEN", "")

MODEL_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/'
MODEL_NAME = 'Llama-3.3-70B-Instruct'
DATA_DIR = '../data/test/edge_tests.tsv'

output_dir = "edgelist_model"
log_dir = "../logs/"
checkpoint_dir = "../checkpoints/"

training_config = {
    'num_iterations': 1,
    'num_steps': 250,
    'batch_size': 1,  # adjust for your available GPUs
    'num_generations': 4,  # reduce if GPUs have less VRAM
    'max_completion_length': 512,  # reduce if GPUs have less VRAM
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1
}

def main():
    # Initialize Accelerate with FSDP enabled
    accelerator = Accelerator(
        fsdp_plugin=FullyShardedDataParallelPlugin()
    )

    # Use accelerator.print instead of raw print
    def print_main(*args, **kwargs):
        accelerator.print(*args, **kwargs)

    # ---------------------------
    # FORMATTING + EXTRACTION FUNCTIONS
    # ---------------------------
    SYSTEM_PROMPT = """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it, answering "yes" or "no".
    The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.
    """
    FORMAT_PATTERN = re.compile(r"^<think>.*?</think><answer>.*?</answer>$", re.DOTALL | re.VERBOSE)
    ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>")

    def extract_answer(text):
        parts = text.split("<answer>")
        if len(parts) < 2:
            return ""
        last_part = parts[-1]
        if "</answer>" not in last_part:
            return ""
        answer = last_part.split("</answer>")[0].strip().lower()
        return "" if answer == "..." or not answer else answer

    # ---------------------------
    # DATASET PREP FUNCTIONS
    # ---------------------------
    def format_edgelist(filepath):
        """
        Adjust 'lines[2:]' if your TSV file has only 1 header line or none.
        If your file is empty or too short, edges will also be empty.
        """
        with open(filepath, 'r') as f:
            # If the file has 2 header lines, skip them:
            lines = f.readlines()[2:]
        edges = []
        for line in lines:
            split_text = line.strip().split()
            if len(split_text) >= 2:
                node1, node2 = split_text[:2]
                edges.append(f'(node {node1}, connected to, node {node2})')
        formatted_edgelist = (
            "You are given a graph in the form of triplets, e.g. (node 1, connected to, node 2). "
            "Answer the question related to the graph. "
            "This is the graph: " + ', '.join(edges)
        )
        return formatted_edgelist

    remove_chars = str.maketrans('', '', "[]'")

    def prepare_dataset(example):
        desc = format_edgelist(example["desc"])
        answer = example["answer"].translate(remove_chars).strip().lower()
        question = example["question"].translate(remove_chars).replace("graph_n50_", "").strip()
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc + " " + question} 
        ])
        return {"prompt": prompt_str, "answer": answer}

    def build_prompt(messages):
        return "\n".join(msg["content"].strip() for msg in messages)

    # ---------------------------
    # BATCHED EVALUATION FUNCTIONS
    # ---------------------------
    def evaluate_model(model, tokenizer, eval_dataset, device, batch_size=8):
        model.eval()
        prompts, expected_answers = [], []
        for example in eval_dataset:
            full_prompt = example["prompt"]
            expected = example["answer"]
            prompts.append(full_prompt)
            expected_answers.append(expected)

        if len(prompts) == 0:
            print_main("No evaluation samples found; skipping evaluation.")
            return 0.0

        tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        inputs = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        with torch.no_grad():
            with autocast():
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

        correct = 0
        for prompt, expected, response in zip(prompts, expected_answers, responses):
            predicted = extract_answer(response)
            is_correct = (predicted == expected)
            correct += int(is_correct)
            print_main("\nPrompt:\n", prompt)
            print_main("\nExpected Answer:\n", expected)
            print_main("\nExtracted Answer:\n", predicted)
            print_main("\nFull Generated Response:\n", response)
            print_main("\nCorrect:", "✓" if is_correct else "✗")
            print_main("-"*50)

        accuracy = (correct / len(prompts)) * 100
        print_main(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(prompts)})")
        print_main("="*50)
        model.train()
        return accuracy

    # ---------------------------
    # REWARD FUNCTIONS
    # ---------------------------
    def correctness_reward(prompts, completions, answer, **kwargs):
        responses = [completion[0]['content'] for completion in completions]
        extracted = [extract_answer(r) for r in responses]
        rewards = []
        for r, a in zip(extracted, answer):
            r_str, a_str = (r or "").strip().lower(), (a or "").strip().lower()
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
            think_open_count = response.count("<think>")
            think_close_count = response.count("</think>")
            answer_open_count = response.count("<answer>")
            answer_close_count = response.count("</answer>")
            trailing_text = ""
            if "</answer>" in response:
                trailing_text = response.split("</answer>")[-1].strip()
            if (think_open_count == 1 and think_close_count == 1 and 
                answer_open_count == 1 and answer_close_count == 1 and
                trailing_text == "" and response.strip().endswith("</answer>")):
                score += 1.5
            else:
                # partial credit
                if think_open_count == 1 and think_close_count == 1:
                    score += 0.2
                else:
                    score -= 0.1 * (abs(think_open_count - 1) + abs(think_close_count - 1))

                if answer_open_count == 1 and answer_close_count == 1:
                    score += 0.2
                else:
                    score -= 0.1 * (abs(answer_open_count - 1) + abs(answer_close_count - 1))

                if trailing_text:
                    score -= 0.5
            rewards.append(score)
        return rewards

    def combined_reward(prompts, completions, answer):
        correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
        format_scores = format_reward(completions=completions)
        return [c + f for c, f in zip(correctness_scores, format_scores)]

    # ---------------------------
    # GRPO TRAINING FUNCTIONS
    # ---------------------------
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
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        return (sequence_indices <= eos_idx.unsqueeze(1)).int()

    def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(accelerator.device)
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)

        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

        with autocast():
            outputs = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=max_completion_length,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False,
            )
        completion_ids = outputs[:, prompt_length:]
        completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
        if isinstance(batch_samples, dict):
            prompts = batch_samples["prompt"]
            answers = batch_samples["answer"]
        else:
            prompts = [sample["prompt"] for sample in batch_samples]
            answers = [sample["answer"] for sample in batch_samples]

        with torch.no_grad():
            prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
                model, tokenizer, prompts, num_generations, max_completion_length
            )
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

            with autocast():
                old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
                ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)

        formatted_completions = [[{'content': tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
        repeated_prompts = [p for p in prompts for _ in range(num_generations)]
        repeated_answers = [a for a in answers for _ in range(num_generations)]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
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
        completion_mask = rollout_data["completion_mask"]
        logits_to_keep = rollout_data["logits_to_keep"]
        old_log_probs = rollout_data["old_log_probs"]
        ref_log_probs = rollout_data["ref_log_probs"]

        with autocast():
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

        batch_size = rollout_data["batch_size"]
        num_generations = rollout_data["num_generations"]
        rewards = rewards.view(batch_size, num_generations)
        avg_reward = rewards.mean().item()

        mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
        std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
        advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        surrogate_loss = torch.min(surr1, surr2)

        # This "kl" line is not the typical KL expression, but we'll keep your code as is:
        kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
        per_token_loss = surrogate_loss - beta * kl

        loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss, avg_reward

    def train_with_grpo(
        model,
        tokenizer,
        train_loader,
        num_iterations=1,
        num_steps=500,
        batch_size=4,
        num_generations=4,
        max_completion_length=128,
        beta=0.1,
        learning_rate=5e-6,
        mu=3,
        epsilon=0.2,
        reward_function=None
    ):
        logging_interval = 5

        # Prepare model & optimizer just once outside the loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

        for iteration in range(num_iterations):
            print_main(f"\nIteration {iteration+1}/{num_iterations}")

            # Create a frozen reference model
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            print_main("Reference model created.")

            model.train()

            step = 0
            for batch_samples in train_loader:
                if step >= num_steps:
                    break
                step += 1

                # Rollout data from the current policy
                with torch.no_grad():
                    rollout_data = generate_rollout_data(
                        model,
                        ref_model,
                        tokenizer,
                        batch_samples,
                        num_generations,
                        max_completion_length
                    )

                # Perform GRPO updates
                for grpo_iter in range(mu):
                    loss, avg_reward = grpo_loss(
                        model,
                        ref_model,
                        rollout_data,
                        tokenizer,
                        reward_function,
                        beta=beta,
                        epsilon=epsilon
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()

                    if grpo_iter % logging_interval == 0:
                        wandb.log({
                            "loss": loss.item(),
                            "average_reward": avg_reward,
                            "iteration": iteration + 1,
                            "step": step,
                            "grpo_iter": grpo_iter + 1
                        })
                        print_main(
                            f"Iteration {iteration+1}/{num_iterations}, "
                            f"Step {step}/{num_steps}, "
                            f"GRPO iter {grpo_iter+1}/{mu}, "
                            f"loss: {loss.item():.4f}, "
                            f"reward: {avg_reward:.4f}"
                        )
        return model

    def optimize_model_memory(model):
        model.train()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()
        return model

    device = accelerator.device
    print_main(f"Using primary device: {device}")

    print_main("Loading model...")
    # Remove device_map='auto' so FSDP can fully shard
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(MODEL_DIR, MODEL_NAME),
        torch_dtype=torch.bfloat16,
        use_auth_token=token,       # <--- use_auth_token instead of token=
        use_safetensors=True
    )
    
    # Configure LoRA / QLoRA
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=8,                         
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    print_main("Wrapping model with QLoRA adapters...")
    model = get_peft_model(model, lora_config)
    print_main("Model loaded & LoRA applied.")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, MODEL_NAME),
        padding_side="left",
        use_auth_token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    model.config.eos_token_id = tokenizer.eos_token_id

    # Prepare the dataset (ensure your dataset returns dicts with "desc", "question", "answer")
    transformed_dataset = TransformedDataset(BasicEdgePredDataset(DATA_DIR), prepare_dataset)
    eval_size = 10 
    eval_idxs = list(range(eval_size))
    train_idxs = list(range(eval_size, len(transformed_dataset)))

    eval_dataset = Subset(transformed_dataset, eval_idxs)
    train_dataset = Subset(transformed_dataset, train_idxs)

    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    print_main("\nInitial model evaluation before finetuning:")
    pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_dataset, device)
    print_main(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = optimize_model_memory(model)

    print_main("\nStarting RL fine-tuning using GRPO...")
    wandb.init(
        project=os.environ["WANDB_PROJECT"], 
        dir=os.path.join(log_dir, "grpo"),
        entity=os.environ["WANDB_ENTITY"], 
        reinit=True
    )
    print_main("Weights & Biases initialized.")

    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        reward_function=combined_reward,
        **training_config
    )

    wandb.finish()
    print_main("Training completed and wandb run finished.")

    print_main("\nFinal model evaluation after GRPO RL fine-tuning:")
    post_grpo_accuracy = evaluate_model(model, tokenizer, eval_dataset, device)
    print_main(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
    print_main(f"Accuracy improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}")

    # Unwrap model from FSDP before saving (best practice)
    final_model = accelerator.unwrap_model(model)

    print_main("\nSaving GRPO fine-tuned model...")
    final_model.save_pretrained(os.path.join(MODEL_DIR, "GRPO-EdgePred-Vanilla"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "GRPO-EdgePred-Vanilla"))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
