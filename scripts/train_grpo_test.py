# imports 
import random
import copy
import re
import os
import sys
import numpy as np
import wandb
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from DGXutils import GetLowestGPU

def set_random_seed(seed: int=42):
    """
    Set random seed for reproducibility across python, numpy, pytorch

    Args:
        seed (int): random seed value
    
    Returns:
        None
    """

    # set seed for python random module
    random.seed(seed)

    # set seed for numpy
    np.random.seed(seed)

    # set seed for pytorch
    torch.manual_seed(seed)

    # set seed for torch.cuda
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set the seed
set_random_seed(42)

# set wandb logging variables
load_dotenv()
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")

# -----------------------------------------
## FORMATTING + ANSWER EXTRACTION FUNCTIONS
# -----------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

def extract_answer_from_model_output(text):
    """
    Extracts value from the last <answer> tag in the text

    Args:
        text (str): model-generated text containing XML-style <answer> tags
    
    Returns:
        str or None: extracted answer from the last <answer> tag, or None if no <answer> tags are found
    """

    # split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:
        # No <answer> tag found
        return None
    last_part = parts [-1]

    # extract the content up to </answer>
    if "</answer>" not in last_part:
        return None
    answer = last_part.split("</answer>")[0].strip() 
    return None if answer == "..." else answer

def extract_answer_from_dataset(text):
    """
    Extracts answer from gsm8k dataset examples

    Args:
        text (str): dataset example text containing a question and answer
    
    Returns:
        str or None: extracted answer after '####' delimiter, or None if no answer is found
    """
    if '####' not in text:
        return None
    return text.split('####')[1].strip()

# -----------------------
## DATASET PREP FUNCTIONS
# -----------------------
def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.

    Args:
        messages (list): a list of message dictionaries, each with "role" and "content" keys
    
    Returns:
        str: a concatenated string of all message contents
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

# build gsm8k dataset and preprocess
gsm8k = load_dataset("openai/gsm8k", "main")["train"]
data = gsm8k.map(prepare_dataset).remove_columns(["question"])

# ---------------
## EVAL FUNCTIONS
# ---------------
def extract_last_number(text):
    """
    Extracts the last number appearing in the text.

    Args:
        text (str): the text to extract a number from
    
    Returns:
        float or None: the last number in the text, or None if no number is found.
    """ 

    # remove $, % from text
    text = text.replace('$', '').replace('%', '')

    # regex to find an int, fraction, or decimal appearing at the end of the text
    pattern = r'(?:^|\s)(\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(2)) if match else None

def extract_single_number(text):
    """
    Extracts a single number from the text if exactly one number is present.
    
    Args:
        text (str): The text to extract a number from.
    
    Returns:
        float or None: the single number in the text, or none if zero or multiple
    """

    # regex to find a number in the text
    numbers = re.findall(r'-?\d*\.?\d+', text)

    # return the number if exactly one is found
    return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_examples, device):
    """
    Evaluates the model on a set of examples and prints the detailed results.

    Args:
        model: the language model to evaluate
        tokenizer: tokenizer for encoding inputs, decoding outputs
        eval_examples (list): list of evaluation examples, each containing a "prompt" and "answer"
        device: the device to run the model on

    Returns:
        float: accuracy percentage (correct predictions / total examples * 100)
    """
    
    # initialize variables
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print(f"EVALUATING ON {total} EXAMPLES")
    print("="*50)

    for example in eval_examples:
        # get prompt, expected answer
        full_prompt = example["prompt"]
        expected = example["answer"]

        # tokenize and generate response
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_eos_token_id=tokenizer.eos_token_id,
                early_stopping=False
                )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # extact answer and check correctness
            predicted = extract_answer_from_model_output(response)

            # try different match methods
            if predicted == expected: # Exact math
                is_correct = True
            
            # single number matching
            else:
                pred_num = extract_single_number(str(predicted))
                exp_num = extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    # try last number matching
                    pred_num = extract_last_number(str(predicted))
                    exp_num = extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and pred_num == exp_num)
            
            # update counter if correct
            if is_correct:
                correct += 1
            
            # print results
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:" "YES" if is_correct else "NO")
            print("-"*50)
            
        except Exception as e:
            print("\nFailed to parse model output for prompt:")
            print(full_prompt)
            print("Error:", e)
            print("-"*50)
    
    # calculate and print final accuracy
    accuracy = correct / total * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)

    # put back in train mode
    model.train()
    return accuracy

# -----------------
## REWARD FUNCTIONS
# -----------------
def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.

    Args:
        prompts (list): list of input prompts
        completions (list): list of model-generated completions
        answer (list): list of expected answers
        **kwargs: additional keyword arguments
    
    Returns:
        list: a list of numerical rewards for each completion
    
    Rewards:
        2.0 points for an exact match
        1.5 points for numeric equivalense (values match but format differs)
        0.0 points for incorrect answers
    """

    # extract answers from model completions
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for r, a in zip(extracted, answer):
        # exact match
        if r == a: 
            rewards.append(2.0)
        # try numeric equivalence
        else:
            r_num = extract_single_number(str(r))
            a_num = extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    
    # log completion lengths
    completion_lengths = [len(r) for r in responses]
    return rewards

def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.
    
    Args:
        completions (list): list of model completions, each containing content.
        **kwargs: additional keyword arguments
    
    Returns:
        list: a list of format compliance scores for each completion
    
    Rewards:
        0.2 points for each tag 
        0.8 points maximum score
    """

    # extract responses
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    format_scores = []

    # score responses
    for response in responses:
        score = 0.0
        if "<think>" in response:
            score += 0.2
        if "</think>" in response:
            score += 0.2
        if "<answer>" in response:
            score += 0.2
        if "</answer>" in response:
            score += 0.2
        rewards.append(score)
        format_scores.append(score)
    return rewards

def combined_reward(prompts, completions, answer):
    """
    Combines correctness and format rewards for each completion.

    Args:
        prompts (list[str]): list of prompt texts
        completions (list[list[dict]]): list of model completions
        answer (list[str]): list of expected answers
    
    Returns:
        list[float]: list of combined rewards
    
    Rewards:
        Correctness score range: 0.0 to 2.0
        Format score range: 0.0 to 0.8
        Total possible range: 0.0 to 2.8
    """

    # get individual rewards
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)

    # combine rewards
    combined_rewards = [c + f for c, f in zip(correctness_scores, format_scores)]
    return combined_rewards

# ------------------------
## GRPO TRAINING FUNCTIONS
# ------------------------
def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.

    Args:
        logits (torch.Tensor): raw logits output from the model
        input_ids (torch.Tensor): token IDs for which we want log probabilities

    Returns:
        torch.Tensor: log probabilities for the selected tokens
    """

    # get log probabilities for all tokens
    log_probs = F.log_softmax(logits, dim=-1)

    # select only log probs for input tokens
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))

def compute_log_probs(model, input_ids, attention_mask, logits_to_keep):
    """
    Computes log probabilities for a batch of tokens

    Args:
        model: the language model
        input_ids (torch.Tensor): token ids for the input sequence
        attention_mask (torch.Tensor): mask for the input sequence
        logits_to_keep (list): number of tokens to keep from the end of the sequence

    Returns:
        torch.Tensor: log probabilities for the selected tokens
    """

    # get logits from the model for the input sequence
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :] # last logit is a predicted token

    # select only the "logits_to_keep" tokens from inputs/logits sequence
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]

    # compute log probabilities for the selected tokens
    return selective_log_softmax(logits, input_ids)

def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after EOS

    Args:
        completion_ids (torch.Tensor): token ids of the generated completion
        eos_token_id (int): the token id for the end-of-sequence token
    
    Returns:
        torch.Tensor: binary mask with 1s for valid tokens, 0s for tokens after EOS
    """

    # find where eos occurs
    is_eos = completion_ids == eos_token_id

    # create a tensor matching the input shape with the eos masked 
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)

    # mask before and including eos
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)

    # return mask
    return (sequence_indices <= eos_idx.unsqueeze(-1)).int()

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generates multiple completions for each prompt.

    Args:
        model: the language model
        tokenizer: the tokenizer for encoding and decoding text
        prompts (list): list of text prompts
        num_generations (int): number of completions to generate per prompt
        max_completion_length (int): maximum number of tokens to generate

    Returns:
        tuple: containing prompts IDs, prompt mask, completion IDs, and completion mask
    """

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    
    # move to device
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    print(f"Input batch size: {prompt_ids.size(0)}, Device before model: {prompt_ids.device}")

    # get prompt length
    prompt_length = prompt_ids.size(1)

    # expand prompts for multiple generations
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # generate
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

    # split completions and mask
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)
    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    Generates data for GRPO rollouts including completions and log probabilities.

    Args:
        model: the policy model being trained
        ref_model: the reference model for KL divergence calculation
        tokenizer: the tokenizer for encoding and decoding text
        batch_samples (list): batch of training samples
        num_generations (int): number of completions to generate per prompt
        max_completion_length (int): maximum completion length

    Returns:
        dict: dictionary containing all data needed for GRPO updates
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # extract prompts and answers
    prompts = [sample["prompt"] if isinstance(sample, dict) else sample[0] for sample in batch_samples]
    answers = [sample["answer"] if isinstance(sample, dict) else sample[1] for sample in batch_samples]

    # generate completions
    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        
        # compute log probabilities for current + ref model
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        old_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probs(ref_model, input_ids, attention_mask, logits_to_keep)
    
    # format completions, prompts, answers
    formatted_completions = [[{"content": tokenizer.decode(ids, skip_special_tokens=True)}] for ids in completion_ids]
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
    Computes GRPO loss for updating the policy model.

    Args:
        model: the policy model being trained
        ref_model: reference model for KL divergence calculation
        rollout_data (dict): data generated by generate_rollout_data
        tokenizer: tokenizer for encoding and decoding text
        reward_function: function that calculates rewards for completions
        beta (float): KL penalty coefficient
        epsilon (float): clipping parameter for PPO
    
    Returns:
        torch.Tensor: GRPO loss to be optimized
    """

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get data from policy rollout
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]

    # get new log probs from policy model
    token_log_probs = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

    # get ratio of new to old log probs
    ratio = torch.exp(token_log_probs - old_log_probs)

    # calculate rewards with the given reward_function
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], 
        completions=rollout_data["formatted_completions"],
        answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=device
    )

    # get batch size, num generations, and reshape rewards with them
    batch_size = rollout_data["batch_size"]
    num_generations = rollout_data["num_generations"]
    rewards = rewards.view(batch_size, num_generations)

    # get mean reward
    avg_reward = rewards.mean().item()
    print(f"Average reward: {avg_reward:.2f}")

    # calculate advantages
    mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)
    std_rewards = rewards.std(dim=1).repeat_interleave(num_generations)
    advantages = ((rewards.view(-1) - mean_rewards) / (std_rewards + 1e-4)).unsqueeze(1)

    # calculate surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    surrogate_loss = -torch.min(surr1, surr2)

    # calculate kl penalty
    kl = torch.exp(ref_log_probs - token_log_probs) - (ref_log_probs - token_log_probs) - 1
    
    # calculate loss per token
    per_token_loss = surrogate_loss + beta * kl
    loss = -((per_token_loss * completion_mask).sum(dim=1)) / completion_mask.sum(dim=1).mean()

    return loss, avg_reward

def train_with_grpo(model, tokenizer,
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
                    reward_function=None,
                    device_ids=None
                    ):
    """
    Full training wrapper for GRPO

    Args:
        model: the language model to train
        tokenizer: the tokenizer for encoding and decoding text
        train_data (list): training dataset
        num_iterations (int): number of outer iterations (epochs/ref model updates)
        num_steps (int): number of batch updates per iteration
        batch_size (int): number of samples per batch
        max_completion_length (int): maxmum token length for completions
        beta (float): kl penalty coefficient
        learning_rate (float): optimizer learning rate
        mu (int): number of policy updates per batch
        epsilon (float): PPO clipping parameter
        reward_function (function): function that calculates rewards for completions
        device_ids (list): list of GPU device IDs to use for DataParallel
    """

    # make sure we have multiple device ids if we're using DataParallel
    # assert device_ids is not None and len(device_ids) > 1, "This code needs at least 2 GPU cores to run!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=device_ids)
    print(f"Model wrapped with DataParallel on devices: {device_ids}")
    # model.to(device)

    # outer loop: iterative grpo updates (epochs)
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        # create a ref model with deepcopy, set to eval
        ref_model = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model)
        ref_model.eval()
        # ref_model.to(device)
        for param in ref_model.parameters():
            param.requires_grad = False
        print("Reference model created.")

        # re-init optimizer for this iteration
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()

        # inner loop
        for step in range(num_steps):

            # randomly sample from data to get batch
            batch_samples = random.sample(train_data, batch_size)

            # generate rollout data
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    model.module if isinstance(model, nn.DataParallel) else model,
                    ref_model,
                    tokenizer,
                    batch_samples,
                    num_generations,
                    max_completion_length
                )
            
            # policy updates
            for grpo_iter in range(mu):
                loss, avg_reward = grpo_loss(
                    model.module if isinstance(model, nn.DataParallel) else model,
                    ref_model,
                    rollout_data,
                    tokenizer,
                    reward_function,
                    beta=beta,
                    epsilon=epsilon
                )

                # zero gradients, backward, step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                # log to wandb
                wandb.log({
                    "loss": loss.item(),
                    "average_reward": avg_reward,
                    "epoch": iteration + 1,
                    "step": step + 1,
                    "grpo_iter": grpo_iter + 1
                })

                # print
                print(f"Iteration {iteration + 1}/{num_iterations}, Step {step + 1}/{num_steps}, GRPO Iteration {grpo_iter + 1}/{mu}, loss: {loss.item():.4f}")

    return model.module if isinstance(model, nn.DataParallel) else model

# -------------
## TRAIN CONFIG
# -------------
def optimize_model_memory(model):
    """
    Optimizes model memory during training

    Args:
        model: language model to optimize

    Returns:
        the optimized model
    """

    # set to train, caching off
    model.train()
    model.config.use_cache = False

    # ensure inputs require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        # forward hook applies requires_grad to inputs
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model

# -----
## MAIN
# -----

device = GetLowestGPU()
print(f"Using primary device: {device}")

model_name = "meta-llama/Llama-3.1-8B-Instruct"
output_dir = "logs/grpo-test"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    )
print(f"{model_name} loaded.")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# detect available gpus
num_gpus = 2 # only going to use 2 gpus
print(f"Number of GPUs: {num_gpus}")
device_ids = list(range(num_gpus)) if num_gpus > 1 else None

# prepare dataset
gsm8k = load_dataset("openai/gsm8k", "main")["train"]
data = gsm8k.map(prepare_dataset).remove_columns(["question"])
eval_size = 30
eval_data = data[:eval_size]
train_data = data[eval_size:]

# eval before training
print("\nEvaluation before fine-tuning:")
pre_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
print(f"Pre-GRPO accuracy: {pre_grpo_accuracy:.2f}%")

# apply mem-optimizations
model = optimize_model_memory(model)

print(f"\n Starting RL fine-tune with GRPO")
train_config = {
    'num_iterations':1,
    'num_steps': 500,
    'batch_size': 7,
    'num_generations': 12,
    'max_completion_length': 400,
    'beta': 0.04,
    'learning_rate': 5e-6,
    'mu': 1,
    'epsilon': 0.2,
}

# init wandb
wandb.init(project=os.environ["WANDB_PROJECT"], reinit=True)
print("wandb initialized.")

# train with GRPO
model = train_with_grpo(
    model=model,
    tokenizer=tokenizer,
    train_data=train_data,
    reward_function=combined_reward,
    device_ids=device_ids,
    **train_config
)

# clean up, eval, and save
wandb.finish()
print("Training complete. Evaluating model...")

print("\nFinal eval after fine-tuning:")
post_grpo_accuracy = evaluate_model(model, tokenizer, eval_data, device)
print(f"Post-GRPO accuracy: {post_grpo_accuracy:.2f}%")

print("\nSaving fine-tuned model...")
model.save_pretrained("../checkpoints/grpo-test")
tokenizer.save_pretrained("../checkpoints/grpo-test")
print("Model saved.")

