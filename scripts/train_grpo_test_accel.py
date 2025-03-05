# Import necessary libraries
# Basic Python libraries for various operations
import random
import copy
import re
import os
import numpy as np
import wandb
from dotenv import load_dotenv
from DGXutils import GetLowestGPU

# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Hugging Face libraries for transformer models
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import Accelerator, DeepSpeedPlugin
from deepspeed.runtime.zero.stage3 import GatheredParameters

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # # Ensure deterministic behavior in cuDNN (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# Call the function to set random seed for reproducibility
set_random_seed(42)

load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ds config
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "fp16": {"enabled": True}  # or fp16 if supported; note both fp16 and bfloat16 use 16 bits
}

# -----------------------------------------
## FORMATTING + ANSWER EXTRACTION FUNCTIONS
# -----------------------------------------
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_answer_from_model_output(text):
   """
   Extracts the value from the last <answer> tag in the text.

   Args:
       text (str): The model-generated text containing XML-style <answer> tags.

   Returns:
       str or None: The content inside the <answer> tags, or None if no valid answer is found.

   Explanation:
       1. Splits the text on the <answer> tag to isolate content after the tag.
       2. Checks if at least one <answer> tag exists in the text.
       3. For the last <answer> segment:
          - Verifies it contains a closing </answer> tag.
          - Extracts only the content between the tags.
       4. Returns None if the answer is empty (just "...") or if tags are missing.
   """
   # Split on <answer> and take everything after the last occurrence
   parts = text.split("<answer>")
   if len(parts) < 2:  # No <answer> tag found
       return None
   last_part = parts[-1]

   # Extract content up to </answer>
   if "</answer>" not in last_part:
       return None
   answer = last_part.split("</answer>")[0].strip()
   return None if answer == "..." else answer

def extract_answer_from_dataset(text):
   """
   Extracts the answer from the GSM8K dataset examples.

   Args:
       text (str): The dataset example text containing a question and answer.

   Returns:
       str or None: The extracted answer part after the '####' delimiter, or None if not found.

   Explanation:
       1. Checks if the text contains the '####' delimiter that separates question from answer.
       2. If found, splits the text at this delimiter and returns the second part (the answer).
       3. The answer is stripped of leading/trailing whitespace.
       4. Returns None if no delimiter is present.
   """
   if "####" not in text:
       return None
   return text.split("####")[1].strip()

# -----------------------
## DATASET PREP FUNCTIONS
# -----------------------
def build_prompt(messages):
   """
   Build a single prompt string from a list of messages.

   Args:
       messages (list): A list of message dictionaries, each with 'role' and 'content' keys.

   Returns:
       str: A concatenated string of all message contents.
   """
   return "\n".join([msg["content"].strip() for msg in messages])

def prepare_dataset(example):
    """
    prepare a gsm8k observation for training with string prompts

    Args:
        dataset (DatasetDict): a dataset containing examples with "question" and "text" keys
    
    Returns:
        list: a list of formatted examples, each containing a prompt string and an answer
    """

    # load data

    # loop through examples, format, add to new dataset
    prompt_str = build_prompt([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ])
    formatted_example = {
        "prompt": prompt_str,
        "answer": extract_answer_from_dataset(example["answer"])
    }
    return formatted_example

# Define a custom collate function to batch examples
def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]
    return {"prompt": prompts, "answer": answers}

# ---------------
## EVAL FUNCTIONS
# ---------------
def extract_last_number(text):
   """
   Extracts the last number appearing in the text.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The last number in the text, or None if no number is found.
   """

   text = text.replace('$', '').replace('%', '')
   pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
   match = re.search(pattern, text)
   return float(match.group(1)) if match else None

def extract_single_number(text):
   """
   Extracts a single number from text if exactly one number is present.

   Args:
       text (str): The text to extract a number from.

   Returns:
       float or None: The single number in the text, or None if zero or multiple numbers are found.
   """

   numbers = re.findall(r'-?\d*\.?\d+', text)
   return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_examples, device):
   """
   Evaluates the model on a set of examples and prints detailed results.

   Args:
       model: The language model to evaluate.
       tokenizer: The tokenizer for encoding inputs and decoding outputs.
       eval_examples (list): List of evaluation examples, each containing "prompt" and "answer".
       device: The device (CPU or GPU) to run evaluation on.

   Returns:
       float: The accuracy percentage (correct predictions / total examples * 100).
   """
   device = next(model.parameters()).device
   model.eval()
   correct = 0
   total = len(eval_examples)
   print("\n" + "="*50)
   print("EVALUATION ON", total, "EXAMPLES")
   print("="*50)

   for example in eval_examples:
       # Get the prompt and expected answer
       full_prompt = example["prompt"]
       expected = example["answer"]

       # Tokenize and generate response
       inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
       with torch.no_grad():
           with GatheredParameters(list(model.parameters()), enabled=True):
            outputs = model.generate(
                    inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    forced_eos_token_id=tokenizer.eos_token_id,
                    early_stopping=False,
                )
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)

       try:
           # Extract answer and check correctness
           predicted = extract_answer_from_model_output(response)

           # Try different matching methods
           if predicted == expected:  # Exact match
               is_correct = True
           else:
               # Try single number matching
               pred_num = extract_single_number(str(predicted))
               exp_num = extract_single_number(str(expected))
               if pred_num is not None and exp_num is not None and pred_num == exp_num:
                   is_correct = True
               else:
                   # Try last number matching
                   pred_num = extract_last_number(str(predicted))
                   exp_num = extract_last_number(str(expected))
                   is_correct = (pred_num is not None and exp_num is not None and
                               pred_num == exp_num)

           # Update counter for correct answers
           if is_correct:
               correct += 1

           # Print evaluation details
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

   # Calculate and print final accuracy
   accuracy = (correct / total) * 100
   print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
   print("="*50)

   # Return model to training mode
   model.train()
   return accuracy

# -----------------
## REWARD FUNCTIONS
# -----------------
def correctness_reward(prompts, completions, answer, **kwargs):
   """
   Assigns a reward based on the correctness of the model's answer.

   Args:
       prompts (list): List of input prompts.
       completions (list): List of model completions, each containing content.
       answer (list): List of expected answers.
       **kwargs: Additional keyword arguments.

   Returns:
       list: List of numerical rewards for each completion.

   Explanation:
       1. Extracts the content from each completion.
       2. Extracts the answer portion from each response using extract_answer_from_model_output.
       3. Assigns rewards based on matching criteria:
          - 2.0 points for an exact match
          - 1.5 points for numeric equivalence (when values match but format differs)
          - 0.0 points for incorrect answers
       4. Tracks completion lengths for analysis.
   """
   responses = [completion[0]['content'] for completion in completions]
   extracted = [extract_answer_from_model_output(r) for r in responses]
   rewards = []
   for r, a in zip(extracted, answer):
       if r == a:  # Exact match case
           rewards.append(2.0)
       else:
           # Try numeric equivalence
           r_num = extract_single_number(str(r))
           a_num = extract_single_number(str(a))
           if r_num is not None and a_num is not None and r_num == a_num:
               rewards.append(1.5)
           else:
               rewards.append(0.0)
   # Log completion lengths
   completion_lengths = [len(response.split()) for response in responses]
   return rewards

def format_reward(completions, **kwargs):
   """
   Assigns a reward for adhering to the desired XML format.

   Args:
       completions (list): List of model completions, each containing content.
       **kwargs: Additional keyword arguments.

   Returns:
       list: List of format compliance scores for each completion.

   Explanation:
       1. Extracts the content from each completion.
       2. Evaluates format compliance by checking for required XML tags:
          - 0.2 points for each tag present (<reasoning>, </reasoning>, <answer>, </answer>)
          - Maximum score of 0.8 for perfect format compliance
       3. Stores and returns the format compliance scores.
   """
   responses = [completion[0]['content'] for completion in completions]
   rewards = []
   format_scores = []
   for response in responses:
       score = 0.0
       if "<reasoning>" in response: score += 0.2
       if "</reasoning>" in response: score += 0.2
       if "<answer>" in response: score += 0.2
       if "</answer>" in response: score += 0.2
       rewards.append(score)
       format_scores.append(score)
   return rewards

def combined_reward(prompts, completions, answer):
   """
   Combines correctness and format rewards.

   Args:
       prompts (list[str]): List of prompt texts
       completions (list[list[dict]]): List of completion dictionaries
       answer (list[str]): List of expected answers

   Returns:
       list[float]: Combined rewards for each prompt-completion pair

   Explanation:
       1. Calculates separate rewards for correctness and format compliance.
       2. Combines the rewards with the following weights:
          - Correctness score range: 0.0 to 2.0
          - Format score range: 0.0 to 0.8
          - Total possible range: 0.0 to 2.8
       3. Returns the combined reward for each example.
   """
   # Get individual rewards
   correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
   format_scores = format_reward(completions=completions)

   # Combine rewards - correctness is weighted more heavily
   combined_rewards = []
   for c_score, f_score in zip(correctness_scores, format_scores):
       # Correctness score range: 0.0 to 2.0
       # Format score range: 0.0 to 0.8
       # Total range: 0.0 to 2.8
       combined_rewards.append(c_score + f_score)

   return combined_rewards

# ------------------------
## GRPO TRAINING FUNCTIONS
# ------------------------
def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.

    Args:
        logits (torch.Tensor): The raw logits output from the model.
        input_ids (torch.Tensor): The token IDs for which we want the log probabilities.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Applies log softmax to convert logits to log probabilities over the vocabulary.
        2. Uses gather to extract only the log probabilities corresponding to the input_ids.
        3. Removes the extra dimension to match the original shape of input_ids.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    Computes the log probabilities for a batch of tokens.

    Args:
        model: The language model.
        input_ids (torch.Tensor): Token IDs for input sequences.
        attention_mask (torch.Tensor): Attention mask for input sequences.
        logits_to_keep (int): Number of tokens to keep from the end of the sequence.

    Returns:
        torch.Tensor: Log probabilities of the selected tokens.

    Explanation:
        1. Gets logits from the model for the input sequence.
        2. Selects logits for all tokens except the last one (as we predict next tokens).
        3. Selects only the last 'logits_to_keep' tokens from both logits and input_ids.
        4. Computes log probabilities for these tokens using selective_log_softmax.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.

    Args:
        completion_ids (torch.Tensor): Token IDs of the generated completions.
        eos_token_id (int): The ID of the end-of-sequence token.

    Returns:
        torch.Tensor: A binary mask with 1s for valid tokens and 0s after the EOS token.

    Explanation:
        1. Identifies positions where EOS tokens occur in each sequence.
        2. Finds the index of the first EOS token in each sequence.
        3. Creates a mask where positions before and including the first EOS are 1, others are 0.
        4. If no EOS token is found in a sequence, all positions are set to 1.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generates multiple completions for each prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer for encoding and decoding text.
        prompts (list): List of text prompts.
        num_generations (int): Number of completions to generate per prompt.
        max_completion_length (int): Maximum number of tokens to generate.

    Returns:
        tuple: Containing prompt IDs, prompt mask, completion IDs, and completion mask.

    Explanation:
        1. Encodes the prompts and moves them to the appropriate device.
        2. Repeats each prompt num_generations times to generate multiple completions.
        3. Generates completions using the model with specified parameters.
        4. Extracts the completion IDs (excluding the prompt tokens).
        5. Creates a mask for the completions using create_completion_mask.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    print(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    with GatheredParameters(list(model.parameters()), enabled=True):
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )
    print(f"Output batch size: {outputs.size(0)}, Device after model: {outputs.device}")
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    Generates data for GRPO rollouts including completions and log probabilities.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        tokenizer: The tokenizer for encoding and decoding text.
        batch_samples (list): Batch of training samples.
        num_generations (int): Number of completions to generate per sample.
        max_completion_length (int): Maximum completion length.

    Returns:
        dict: Dictionary containing all data needed for GRPO updates.

    Explanation:
        1. Extracts prompts and expected answers from the batch samples.
        2. Generates completions using the current policy model.
        3. Combines prompt and completion tokens.
        4. Computes log probabilities from both the policy model and reference model.
        5. Formats completions for reward calculation.
        6. Repeats prompts and answers to match the number of generated completions.
        7. Returns all data needed for GRPO loss calculation.
    """
    device = next(model.parameters()).device
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]
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
    """
    Computes the GRPO loss for updating the policy model.

    Args:
        model: The policy model being trained.
        ref_model: The reference model for KL divergence calculation.
        rollout_data (dict): Data generated by generate_rollout_data.
        tokenizer: The tokenizer for encoding and decoding text.
        reward_function: Function that calculates rewards for completions.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter for PPO.

    Returns:
        torch.Tensor: The GRPO loss to be minimized.

    Explanation:
        1. Computes current token log probabilities using the policy model.
        2. Calculates the probability ratio between current and old policies.
        3. Computes rewards using the provided reward_function.
        4. Calculates advantages by standardizing rewards within each prompt.
        5. Computes the PPO surrogate objective with clipping.
        6. Calculates the KL divergence between reference and policy models.
        7. Combines surrogate loss and KL penalty.
        8. Averages the loss across all tokens and batches.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device
    )
    #print(f"Rewards: {rewards}")  # Debug rewards
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)
    avg_reward = rewards.mean().item()
    print("Average Reward:", avg_reward)

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

# --------------------------------------------------
## MODIFIED TRAIN FUNCTION (DATALOADER + ACCELERATE)
# --------------------------------------------------

def train_with_grpo(accelerator,
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
                    mu=3, epsilon=0.2, 
                    reward_function=None):
    """
    This function is your original working code (train_with_grpo_static)
    with an added outer loop for iterative GRPO updates per the pseudocode.

    Args:
        model: The language model to train.
        tokenizer: The tokenizer for encoding and decoding text.
        train_data (list): Training dataset.
        num_iterations (int): Number of outer iterations (reference model updates).
        num_steps (int): Number of batch updates per iteration.
        batch_size (int): Number of prompts per batch.
        num_generations (int): Number of completions per prompt.
        max_completion_length (int): Maximum token length for completions.
        beta (float): KL penalty coefficient.
        learning_rate (float): Learning rate for optimizer.
        mu (int): Number of policy updates per batch.
        epsilon (float): PPO clipping parameter.
        reward_function: Function that calculates rewards for completions.
        device_ids (list): List of GPU device IDs for DataParallel.

    Returns:
        The trained model.

    Explanation:
        1. For each outer iteration:
           - Creates a reference model as a deep copy of the current policy model.
           - Reinitializes the optimizer for the policy model.
           - For each training step:
             a. Samples a batch of examples from the training data.
             b. Generates rollout data including completions and log probabilities.
             c. For mu iterations:
                i. Computes the GRPO loss.
                ii. Updates the policy model using gradient descent.
           - Monitors GPU memory usage and prints progress information.
    """
    
    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # prepare model, optimizer, and dataloader with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Outer loop: iterative GRPO updates.
    for iteration in range(num_iterations):
        accelerator.print(f"\nIteration {iteration+1}/{num_iterations}")

        # create ref model for kl penalty
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        base_model = getattr(accelerator.unwrap_model(model), "base_model", accelerator.unwrap_model(model))
        ref_model.load_state_dict(base_model.state_dict())
        ref_model.to(next(model.parameters()).device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        # Inner loop: your original training steps.
        for step, batch in enumerate(train_loader):
            batch_samples = [{"prompt": p, "answer": a} for p, a in zip(batch["prompt"], batch["answer"])]
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model,
                    ref_model,
                    tokenizer,
                    batch_samples,
                    num_generations,
                    max_completion_length
                )
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
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                # Log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "iteration": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })
                accelerator.print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                      f"GRPO iter {grpo_iter+1}/{mu}, loss: {loss.item():.4f}")
    return model.module

# -------------
## TRAIN CONFIG
# -------------
def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.

    Args:
        model: The language model to optimize.

    Returns:
        The optimized model.

    Explanation:
        1. Sets the model to training mode.
        2. Disables KV caching to save memory.
        3. Enables gradient checkpointing to trade computation for memory.
        4. Ensures that input embeddings require gradients:
           - Either uses the built-in method if available.
           - Or adds a forward hook to the input embeddings layer.
        5. Returns the optimized model ready for memory-efficient training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

# -----
## MAIN
# -----

if __name__ == "__main__":
    # Main execution
    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(ds_config))
    device = accelerator.device

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "math_solver_model"

    accelerator.print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    accelerator.print("Model downloaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # lora
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # adjust depending on your model architecture
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    accelerator.print("LoRA integrated. Trainable parameters:")
    accelerator.print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # build gsm8k dataset and preprocess
    gsm8k = load_dataset("openai/gsm8k", "main")["train"]
    data = gsm8k.map(prepare_dataset).remove_columns(["question"])
    size_of_eval_data = 1  # for quick evaluation
    eval_data = data.select(range(size_of_eval_data))
    train_data = data.select(range(size_of_eval_data, len(data)))

    accelerator.print("\nStarting RL fine-tuning using GRPO...")
    # This config was tested on a 8xA100 node, where each A100 is has 80GB of VRAM
    training_config = {
        'num_iterations': 1,
        'num_steps': 500,
        'num_generations': 4, # reduce if you have GPUs with less VRAM
        'max_completion_length': 200, # reduce if you have GPUs with less VRAM
        'beta': 0.04,
        'learning_rate': 5e-6,
        'mu': 1,
        'epsilon': 0.1,
        'batch_size': 1,
    } 
    train_loader = DataLoader(train_data, batch_size=training_config["batch_size"], collate_fn=collate_fn, shuffle=True, num_workers=0)


    # initial eval
    if accelerator.is_main_process:
        accelerator.print("\nInitial model evaluation before finetuning:")
        pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
        accelerator.print(f"Pre-GRPO Accuracy: {pre_grpo_accuracy:.2f}%")

    model = optimize_model_memory(model)

    # Initialize Weights & Biases
    if accelerator.is_main_process:
        wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
    accelerator.print("Weights & Biases initialized.")

    model = train_with_grpo(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        reward_function=combined_reward,
        **training_config
    )

    if accelerator.is_main_process:
        wandb.finish()
    accelerator.print("Training completed and wandb run finished.")

    
    if accelerator.is_main_process:
        accelerator.print("\nFinal model evaluation after GRPO RL fine-tuning:")
        post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
        accelerator.print(f"Post-GRPO Accuracy: {post_grpo_accuracy:.2f}%")

    accelerator.print("\nSaving GRPO fine-tuned model...")
    if accelerator.is_main_process:
        model.save_pretrained("grpo_finetuned_model")
        tokenizer.save_pretrained("grpo_finetuned_model")