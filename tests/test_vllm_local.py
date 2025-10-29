import os
# vLLM will set this itself if needed, but setting early avoids warnings.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-3.2-1B-Instruct",
        dtype="auto",                 # lets vLLM pick bf16/fp16 appropriately
        tokenizer_mode="auto",        # default; included for clarity
        trust_remote_code=True,       # harmless for local HF-style repos
    )
    params = SamplingParams(max_tokens=64, temperature=0.0)
    outs = llm.generate(["Say 'vLLM is alive' and nothing else."], sampling_params=params)
    print(outs[0].outputs[0].text)

if __name__ == "__main__":
    main()