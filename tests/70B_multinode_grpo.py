# imports
import argparse

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

MODEL_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/'
MODEL_NAME = 'Llama-3.3-70B-Instruct'
DATA_DIR = '/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/mentor-rl/data/test/edge_tests.tsv'

output_dir = "edgelist_model"
log_dir = "../logs/"
checkpoint_dir = "../checkpoints/"

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

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

    # -----------------
    # REWARD FUNCTIONS
    # -----------------
    def correctness_reward(prompts, completions, **kwargs):
        answer = kwargs["answer"]
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
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    # -----
    ## GRPO
    # -----

    # load dataset from path
    dataset = TransformedDataset(BasicEdgePredDataset(DATA_DIR), prepare_dataset)
    
    # training args
    training_args = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            use_vllm=True,
            vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),
            )
     
if __name__=="__main__":
    main()
