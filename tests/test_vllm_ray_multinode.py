# tests/test_vllm_ray_multinode.py
# Use AsyncLLMEngine (required for pipeline parallel) with v0 engine
# v1 engine's Ray Compiled DAG fails on multi-node ROCm
import asyncio
import os
import ray
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

def main():
    # Connect to the already-running Ray cluster; DO NOT pass runtime_env
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    # Pick sizes that use all 16 GPUs across 2 nodes
    tp_size = 8
    pp_size = 2

    engine_args = AsyncEngineArgs(
        model=os.environ["MODEL_PATH"],
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
        max_model_len=4096,
    )

    async def run():
        # Use from_engine_args to create AsyncLLMEngine (v0 style)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        params = SamplingParams(temperature=0.0, max_tokens=8)
        
        # Generate using async interface
        results = engine.generate("Hello world", params, request_id="smoke")
        async for output in results:
            final_output = output
        
        print("✓ vLLM init OK. Sample:", repr(final_output.outputs[0].text))
        
        # Cleanup
        engine.shutdown_background_loop()

    asyncio.run(run())

if __name__ == "__main__":
    main()
