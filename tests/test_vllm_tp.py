import os
import sys
from vllm import LLM, SamplingParams

def run_test():
    # Force Ray out of the picture entirely
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    # Unset the conflicting variable just in case
    if "ROCR_VISIBLE_DEVICES" in os.environ:
        del os.environ["ROCR_VISIBLE_DEVICES"]

    print(f"--- Testing vLLM on ROCm {os.environ.get('ROCM_VERSION', 'Unknown')} ---")
    print(f"--- Python: {sys.executable} ---")

    try:
        # Initialize vLLM with Tensor Parallelism = 2
        llm = LLM(
            model=os.environ["MODEL_PATH"],
            tensor_parallel_size=2, 
            gpu_memory_utilization=0.8,
            max_model_length=2048,
            enforce_eager=True,
            distributed_executor_backend="mp" 
        )
        
        print("--- Engine Initialized Successfully! ---")

        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")

        print("--- SUCCESS: vLLM + NCCL is working locally ---")

    except Exception as e:
        print("\n\nCRITICAL FAILURE:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    run_test()