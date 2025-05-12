# imports
import argparse

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

def main():
    # arg parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_horst", type=str, default="", help="The server IP")
    args = parser.parse_args()

    # load dataset
    dataset = load_dataset("trl-lib/tldr", split="train")

    # dummy reward function: count number of unique characters
    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c)) for c in completions]
    
    # set up training arfs
    training_args = GRPOConfig(
            output_dir="grpo-test",
            per_device_train_batch_size=4,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            use_vllm=True,
            vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", ".")
            )

    trainer = GRPOTrainer(model=MODEL_PATH,
                          args=training_args,
                          reward_funcs=reward_num_unique_chars,
                          train_dataset=dataset
                          )
    trainer.train()

if __name__=="__main__":
    main()
