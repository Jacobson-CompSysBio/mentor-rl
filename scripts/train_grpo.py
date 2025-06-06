# Import necessary libraries

# Basic Python libraries for various operations
import random
import itertools
import copy
import re
import os, sys, glob
import numpy as np
import wandb
import traceback
import sys
from dotenv import load_dotenv
from pathlib import Path

# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

# Hugging Face libraries for transformer models
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Llama4ForConditionalGeneration,
    Llama4ForCausalLM
)
from peft import LoraConfig, get_peft_model, TaskType

# append root `mentor-rl` folder to dir
sys.path.append(str(Path(__file__).resolve().parent.parent))

# custom
from utils.dataset import BasicEdgePredDataset, TransformedDataset

def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Call the function to set random seed for reproducibility
set_random_seed(42)


load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
token = os.getenv("HUGGINGFACE_TOKEN")

MODEL_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/'
MODEL_NAME = 'Llama-4-Scout-17B-16E-Instruct' 
DATA_DIR = '../data/test/edge_tests.tsv'

output_dir = "edgelist_model"
log_dir = "../logs/"
checkpoint_dir = "../checkpoints/"

training_config = {
    'num_iterations': 1,
    'num_steps': 100,
    'batch_size': 1, # 3 for 4 gpus
    'num_generations': 2, # reduce if you have GPUs with less VRAM
    'max_completion_length': 64, # reduce if you have GPUs with less VRAM
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.1
}

def main():
    # -----------------------------------------
    ## FORMATTING + ANSWER EXTRACTION FUNCTIONS
    # -----------------------------------------
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

    # -----------------------
    # DATASET PREP FUNCTIONS
    # -----------------------
    def format_edgelist(filepath):
        edges = []
        with open(filepath, 'r') as f:
            # Skip header (first two lines)
            for line in itertools.islice(f, 2, None):
                split_text = line.strip().split()
                if len(split_text) >= 2:
                    node1, node2 = split_text[:2]
                    edges.append(f'(node {node1}, connected to, node {node2})')

        formatted_edgelist = "You are given a graph in the form of triplets," \
        "e.g. (node 1, connected to, node 2). Answer the question related to the graph." \
        "This is the graph: " + ', '.join(edges)
        return formatted_edgelist

    def prepare_dataset(example):
        
        # format desc, question, answer
        desc = format_edgelist(example["desc"])
        answer = example["answer"]
        question = example["question"]
        question = question.replace("graph_n50_", "")
        for x in ['[', ']', '\'']:
            answer = answer.replace(x, '')
            question = question.replace(x, '')
        answer = answer.strip().lower()

        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc + " " + question} 
        ])

        formatted_example = {
            "prompt": prompt_str,
            "answer": answer 
        }
        return formatted_example

    def build_prompt(messages):
        return "\n".join([msg["content"].strip() for msg in messages])

    # ---------------
    # EVAL FUNCTIONS
    # ---------------
    def evaluate_model(model, tokenizer, eval_examples, device):
        model.eval()
        correct = 0
        total = len(eval_examples)
        print("\n" + "="*50)
        print("EVALUATION ON", total, "EXAMPLES")
        print("="*50)

        for example in eval_examples:
            full_prompt = example["prompt"][0] if isinstance(example["prompt"], list) else example["prompt"]
            expected = example["answer"][0] if isinstance(example["answer"], list) else example["answer"]

            tokenized = tokenizer(full_prompt, return_tensors="pt").to(device)
            inputs = tokenized.input_ids.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_eos_token_id=tokenizer.eos_token_id  # Force EOS at the end
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                is_correct = False
                predicted = extract_answer(response)
                if predicted == expected:
                    is_correct = True
                    correct += 1

                print("\nPrompt:")
                print(full_prompt)
                print("\nExpected Answer:")
                print(expected)
                print("\nExtracted Answer:")
                print(predicted)
                print("\nFull Generated Response:")
                print(response)
                print("\nCorrect:", "✓" if is_correct else "✗")
                print("-"*50)

            except Exception as e:
                print("\nFailed to parse model output for prompt:")
                print(full_prompt)
                print("Error:", e)
                print("-"*50)

        accuracy = (correct / total) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
        print("="*50)
        model.train()
        return accuracy

    # -----------------
    # REWARD FUNCTIONS
    # -----------------
    def correctness_reward(prompts, completions, answer, **kwargs):
        responses = [completion[0]['content'] for completion in completions]
        extracted = [extract_answer(r) for r in responses]
        rewards = []
        for r, a in zip(extracted, answer):
            # If no answer was extracted, default to an empty string.
            r_str = r if r is not None else ""
            a_str = a if a is not None else ""
            r_str = r_str.strip().lower()
            a_str = a_str.strip().lower()
            # exact match
            if r_str == a_str:
                rewards.append(2.0)
            # if the answer is a substring of the response
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
            
            # Count occurrences of each tag
            think_open_count = response.count("<think>")
            think_close_count = response.count("</think>")
            answer_open_count = response.count("<answer>")
            answer_close_count = response.count("</answer>")
            
            # Check for extra text after the closing answer tag
            trailing_text = ""
            if "</answer>" in response:
                trailing_text = response.split("</answer>")[-1].strip()
            
            # Reward full points if exactly one set is present and no extra text exists
            if (think_open_count == 1 and think_close_count == 1 and 
                answer_open_count == 1 and answer_close_count == 1 and
                trailing_text == "" and response.strip().endswith("</answer>")):
                score += 1.5
            else:
                # Apply partial rewards/penalties based on deviations
                # Reward for having at least one pair
                if think_open_count == 1 and think_close_count == 1:
                    score += 0.2
                else:
                    score -= 0.1 * (abs(think_open_count - 1) + abs(think_close_count - 1))
                
                if answer_open_count == 1 and answer_close_count == 1:
                    score += 0.2
                else:
                    score -= 0.1 * (abs(answer_open_count - 1) + abs(answer_close_count - 1))
                
                # Penalize extra text after </answer>
                if trailing_text:
                    score -= 0.5
            rewards.append(score)

        return rewards

    def combined_reward(prompts, completions, answer):
        correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
        format_scores = format_reward(completions=completions)
        combined_rewards = []
        for c_score, f_score in zip(correctness_scores, format_scores):
            combined_rewards.append(c_score + f_score)
        return combined_rewards

    # ------------------------
    # GRPO TRAINING FUNCTIONS
    # ------------------------
    def selective_log_softmax(logits, input_ids):
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

    def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
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
        # Tokenize each prompt using the tokenizer and pad to the longest sequence
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
        prompt_ids = inputs["input_ids"].to(model.device)
        prompt_mask = inputs["attention_mask"].to(model.device)
        prompt_length = prompt_ids.size(1)
        
        # Debug print: log the shape and a snippet of prompt_ids
        print(f"[DEBUG] prompt_ids shape: {prompt_ids.shape}")
        print(f"[DEBUG] prompt_mask shape: {prompt_mask.shape}")
        
        # Repeat each prompt num_generations times
        prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
        
        print(f"[DEBUG] Repeated prompt_ids shape: {prompt_ids.shape}")
        
        # --- Temporarily enable caching for generation to avoid RoPE issues ---
        original_use_cache = model.config.use_cache
        gradient_checkpointing_enabled = getattr(model, "is_gradient_checkpointing", False)
        model.config.use_cache = True
        if gradient_checkpointing_enabled:
            print("[DEBUG] Disabling gradient checkpointing for generation")
            model.gradient_checkpointing_disable()
        print(f"[DEBUG] model.config.use_cache set to True for generation")
        
        try:
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
        except Exception as e:
            print("[ERROR] Exception occurred during model.generate call:")
            print(e)
            # Restore original cache setting even if generation fails.
            model.config.use_cache = original_use_cache
            raise

        # Restore the original configuration after generation
        model.config.use_cache = original_use_cache
        if gradient_checkpointing_enabled:
            print("[DEBUG] Re-enabling gradient checkpointing after generation")
            model.gradient_checkpointing_enable()
        print(f"[DEBUG] model.config.use_cache restored to {original_use_cache}")

        # Check for non-empty outputs and the expected dimensions.
        if outputs is None:
            raise ValueError("[ERROR] model.generate returned None.")

        if outputs.ndim != 2:
            raise ValueError(f"[ERROR] Unexpected outputs shape: {outputs.shape}. Expected 2D tensor.")

        print(f"[DEBUG] outputs shape from generate(): {outputs.shape}")

        # Assume that the new tokens are the ones after the original prompt length.
        completion_ids = outputs[:, prompt_length:]
        
        # Check that the completion_ids have nonzero length
        if completion_ids.size(1) == 0:
            raise ValueError("[ERROR] No completion tokens generated (completion_ids has zero length).")

        # Generate completion mask based on eos_token_id.
        completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
        
        print(f"[DEBUG] completion_ids shape: {completion_ids.shape}")
        print(f"[DEBUG] completion_mask shape: {completion_mask.shape}")

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
        prompts = batch_samples["prompt"] if isinstance(batch_samples, dict) else [sample["prompt"] for sample in batch_samples]
        answers = batch_samples["answer"] if isinstance(batch_samples, dict) else [sample["answer"] for sample in batch_samples]

        with torch.no_grad():
            prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
                model, tokenizer, prompts, num_generations, max_completion_length
            )
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)

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

        kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
        per_token_loss = surrogate_loss - beta * kl

        loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return loss, avg_reward

    def train_with_grpo(
        model,
        tokenizer,
        train_data,
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
        """
        Quick PPO/GRPO approach (no separate old_model copy):
          - Generate rollouts once per batch, storing old log probs.
          - Perform multiple gradient updates, overwriting the console log each time.
        """

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration+1}/{num_iterations}")

            # If you still want a reference model for KL, copy it here:
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            print("Reference model created.")

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            model.train()

            for step, batch_samples in enumerate(train_data):
                if step >= num_steps:
                    break

                # 1) Generate rollouts ONCE per batch, storing old_log_probs in memory.
                with torch.no_grad():
                    rollout_data = generate_rollout_data(
                        model,
                        ref_model,
                        tokenizer,
                        batch_samples,
                        num_generations,
                        max_completion_length
                    )

                # 2) Do multiple PPO/GRPO gradient updates
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
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()

                    # Log to W&B if needed
                    wandb.log({
                        "loss": loss.item(),
                        "average_reward": avg_reward,
                        "iteration": iteration + 1,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1
                    })

                    # ———————————————
                    # Overwrite the console line:
                    print(
                        f"\rIteration {iteration+1}/{num_iterations}, "
                        f"Step {step+1}/{num_steps}, "
                        f"GRPO iter {grpo_iter+1}/{mu}, "
                        f"loss: {loss.item():.4f}, "
                        f"reward: {avg_reward:.4f}",
                        end="",
                        flush=True
                    )
                # After finishing the mu inner loop, print a final newline so the
                # next batch's updates start on a fresh line:
                print()

        return model

    # -----
    # MAIN
    # -----
    if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        print(f"Using primary device: {device}")

        print("Downloading model...")
        model = Llama4ForConditionalGeneration.from_pretrained(
                MODEL_DIR + MODEL_NAME,
                #attn_implementation="flex_attention",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=token
                ) 
        print("Model downloaded")

        # Configure LoRA / QLoRA
        lora_config = LoraConfig(
            r=8,                         
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
        )
        print("Wrapping model with QLoRA adapters...")
        model = get_peft_model(model, lora_config)
        print("Model downloaded")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR + MODEL_NAME, padding_side="left", token=token)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False         # <--- Must disable use_cache for gradient checkpointing
        model.gradient_checkpointing_enable()  # <--- Turn on gradient checkpointing
        model.train()
        model.config.eos_token_id = tokenizer.eos_token_id

        transformed_dataset = TransformedDataset(BasicEdgePredDataset(DATA_DIR), prepare_dataset)
        eval_size = 10 
        eval_idxs = list(range(eval_size))
        train_idxs = list(range(eval_size, len(transformed_dataset)))

        eval_dataset = Subset(transformed_dataset, eval_idxs)
        train_dataset = Subset(transformed_dataset, train_idxs)

        train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

        print("\nInitial model evaluation before finetuning:")
        pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_loader, device)
        print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

        print("\nStarting RL fine-tuning using GRPO...")
        wandb.init(project=os.environ["WANDB_PROJECT"], 
                   dir = log_dir + "grpo",
                   entity=os.environ["WANDB_ENTITY"], 
                   reinit=True)
        print("Weights & Biases initialized.")

        model = train_with_grpo(
            model=model,
            tokenizer=tokenizer,
            train_data=train_loader,
            reward_function=combined_reward,
            **training_config
        )

        wandb.finish()
        print("Training completed and wandb run finished.")

        print("\nFinal model evaluation after GRPO RL fine-tuning:")
        post_grpo_accuracy = evaluate_model(model, tokenizer, eval_loader, device)
        print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")
        print(f"Accuracy improvement: {post_grpo_accuracy - pre_grpo_accuracy:.2f}")

        print("\nSaving GRPO fine-tuned model...")
        model.save_pretrained(MODEL_DIR + "GRPO-EdgePred-Vanilla")
        tokenizer.save_pretrained(MODEL_DIR + "GRPO-EdgePred-Vanilla")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush
        raise
