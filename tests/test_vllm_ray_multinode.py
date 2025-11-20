import asyncio
import os
import ray
import socket
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

def test_multinode():
    """
    Test vLLM multi-node setup on Frontier.
    """
    
    # Check what IP Python sees
    hostname = socket.gethostname()
    try:
        detected_ip = socket.gethostbyname(hostname)
        print(f"=== Python IP Detection ===")
        print(f"Hostname: {hostname}")
        print(f"Detected IP: {detected_ip}")
        print("===========================\n")
    except:
        print("Could not detect IP from hostname\n")
    
    # CRITICAL: Include LD_LIBRARY_PATH in runtime env to propagate AWS-OFI-RCCL
    aws_ofi_lib = os.environ.get("AWS_OFI_RCCL_LIB", "/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-nccl/lib")
    rocm_lib = os.environ.get("ROCM_HOME", "/opt/rocm-6.3.1") + "/lib"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    
    # Ensure AWS-OFI-RCCL is first in the path
    new_ld_path = f"{aws_ofi_lib}:{rocm_lib}:{current_ld_path}"
    
    # Complete Frontier environment for Ray workers
    frontier_env_vars = {
        # Libfabric / Slingshot
        "FI_PROVIDER": "cxi",
        "FI_MR_CACHE_MONITOR": "kdreg2",
        "FI_CXI_DEFAULT_CQ_SIZE": "131072",
        "FI_CXI_DEFAULT_TX_SIZE": "2048",
        "FI_CXI_RX_MATCH_MODE": "hybrid",
        "FI_CXI_ATS": "0",
        
        # NCCL Configuration - USE PLUGIN
        "NCCL_NET": "AWS",
        "NCCL_NET_GDR_LEVEL": "3",
        "NCCL_IGNORE_DISABLED_P2P": "1",
        "NCCL_CROSS_NIC": "1",
        "NCCL_SOCKET_IFNAME": "hsn0",
        "NCCL_SOCKET_FAMILY": "AF_INET",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "NCCL_PROTO": "Simple",
        "NCCL_IB_DISABLE": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_TIMEOUT": "3600",
        "NCCL_COLLNET_ENABLE": "0",
        
        # Gloo fallback
        "GLOO_SOCKET_FAMILY": "AF_INET",
        "GLOO_SOCKET_IFNAME": "hsn0",
        
        # Paths - CRITICAL
        "VLLM_NCCL_SO_PATH": os.environ.get("VLLM_NCCL_SO_PATH", "/opt/rocm-6.3.1/lib/librccl.so"),
        "LD_LIBRARY_PATH": new_ld_path,  # CRITICAL: Propagate full path
        
        # Ray/GPU visibility
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        
        # vLLM v0 engine
        "VLLM_USE_V1": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ATTENTION_BACKEND": "ROCM_FLASH",
    }

    # Initialize Ray
    if not ray.is_initialized():
        print("=== Initializing Ray ===")
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            runtime_env={"env_vars": frontier_env_vars}
        )
    
    # Verify cluster
    print(f"=== Ray Cluster Info ===")
    print(f"Ray Node IP: {ray.util.get_node_ip_address()}")
    print(f"Available nodes: {len(ray.nodes())}")
    print(f"Available GPUs: {ray.cluster_resources().get('GPU', 0)}")
    
    print("\n=== Ray Node Details ===")
    for i, node in enumerate(ray.nodes()):
        node_resources = node.get('Resources', {})
        node_labels = [k for k in node_resources.keys() if k.startswith('node:')]
        print(f"  Node {i}: Alive={node['Alive']}, Labels={node_labels}")
    print("========================\n")
    
    # Configuration: TP=8, PP=2 for pipeline parallelism across nodes
    tp_size = 8
    pp_size = 2
    
    print(f"=== Initializing vLLM ===")
    print(f"  Config: TP={tp_size}, PP={pp_size}")
    print(f"  Total GPUs: {tp_size * pp_size}")
    print(f"  LD_LIBRARY_PATH propagated: {aws_ofi_lib} (first)")
    
    try:
        output_text = asyncio.run(
            _run_pipeline_test(tp_size=tp_size, pp_size=pp_size))
        print("\n=== ✓ vLLM Engine Initialized! ===")
        print(f"Output: {output_text!r}")
        print("\n=== ✓ Test Passed! ===")
        return True

    except Exception as e:
        print(f"\n=== ✗ Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def _run_pipeline_test(tp_size: int, pp_size: int) -> str:
    """Launch an AsyncLLMEngine so pipeline parallelism is enabled."""
    engine_args = AsyncEngineArgs(
        model=os.environ.get("MODEL_PATH"),
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        disable_log_stats=True,
        max_model_len=4096,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0, max_tokens=10)
    request_id = "pipeline-test"
    prompt = "Hello world"

    final_output = None
    async for request_output in engine.generate(prompt=prompt,
                                                sampling_params=sampling_params,
                                                request_id=request_id):
        final_output = request_output

    if not final_output or not final_output.outputs:
        raise RuntimeError("AsyncLLMEngine returned no outputs.")

    return final_output.outputs[0].text


if __name__ == "__main__":
    success = test_multinode()
    exit(0 if success else 1)
