#!/bin/bash
#SBATCH --job-name=vllm-cxi
#SBATCH -A SYB114
#SBATCH -N 2
#SBATCH -t 0:15:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH -q debug
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- Load Modules ---
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

# --- Activate Conda ---
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

# --- Set Paths ---
export ROCM_HOME=/opt/rocm-6.3.1

# ============================================================================
# Transport Configuration for Multi-Node vLLM on Frontier
# ============================================================================
# ISSUE SUMMARY:
# - GPU memory registration on Frontier CXI fails with "ioctl error 25"
# - This is a driver-level issue with GPU Direct RDMA registration
# - AWS-OFI-NCCL SENDRECV protocol doesn't fully bypass the issue
# - SOLUTION: Use reliable socket transport (proven to work multi-node)
#
# TRANSPORT OPTIONS (set TRANSPORT env var):
#   "socket"     - Pure socket-based (DEFAULT - reliable, proven to work)
#   "ofi-sendrecv" - CXI with SENDRECV protocol (experimental, may still fail)
#   "ofi-rdma"   - CXI with GPU Direct RDMA (likely to fail on Frontier)
#
# For multi-node deployments on Frontier, use TRANSPORT=socket
# ============================================================================

TRANSPORT="${TRANSPORT:-socket}"  # Default to proven socket transport

if [[ "${TRANSPORT}" == "socket" ]]; then
    echo ">>> Using socket transport (NCCL OFI disabled)"
    # Socket-based transport - known to work reliably multi-node
    unset NCCL_NET
    unset NCCL_NET_PLUGIN
    export FI_PROVIDER=cxi  # For local libfabric use only
    export FI_MR_CACHE_MONITOR=kdreg2

elif [[ "${TRANSPORT}" == "ofi-sendrecv" ]]; then
    echo ">>> Using OFI/CXI with SENDRECV protocol (experimental)"
    # Attempt CXI with SENDRECV to bypass GPU Direct RDMA
    # WARNING: This may still fail with ioctl error 25 on Frontier
    export NCCL_NET_PLUGIN=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-nccl/lib/librccl-net.so
    export NCCL_NET=OFI
    export FI_PROVIDER=cxi
    export FI_MR_CACHE_MONITOR=kdreg2
    export FI_HMEM_ROCR_USE_DMABUF=0
    export FI_CXI_RDZV_GET_MIN=0
    export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
    export FI_CXI_DEFAULT_CQ_SIZE=262144
    export FI_CXI_OFLOW_BUF_SIZE=12582912
    export FI_CXI_DEFAULT_TX_SIZE=16384
    
    # Force SENDRECV protocol
    export OFI_NCCL_PROTOCOL=SENDRECV
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_OFI_DISABLE_GDR=1
    # Libfabric-level GPU memory disable
    export FI_HMEM_DISABLE_P2P=1
    export FI_CXI_DISABLE_DMABUF_ROCR=1
    export FI_CXI_DISABLE_HOST_REGISTER=1

elif [[ "${TRANSPORT}" == "ofi-rdma" ]]; then
    echo ">>> Using OFI/CXI with GPU Direct RDMA (NOT RECOMMENDED - likely to fail)"
    # GPU Direct RDMA - expected to fail with ioctl error 25 on Frontier
    export NCCL_NET_PLUGIN=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-nccl/lib/librccl-net.so
    export NCCL_NET=OFI
    export FI_PROVIDER=cxi
    export FI_MR_CACHE_MONITOR=kdreg2
    export FI_HMEM_ROCR_USE_DMABUF=0
    export OFI_NCCL_PROTOCOL=RDMA
    export NCCL_NET_GDR_LEVEL=SYS
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export FI_CXI_DISABLE_HOST_REGISTER=0
else
    echo "ERROR: Unknown TRANSPORT=${TRANSPORT}"
    exit 1
fi

# --- RCCL Configuration ---
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=3600
export NCCL_ALGO=Tree

# Debug - verbose for initial testing
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

export VLLM_NCCL_SO_PATH=${ROCM_HOME}/lib/librccl.so

# --- Gloo rendezvous ---
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_IPV6=0

# --- PyTorch Distributed / c10d ---
export TP_SOCKET_IFNAME=hsn0
export TORCH_DISTRIBUTED_DEBUG=INFO

# --- Visible GPUs and Ray "don't touch" ---
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export GPU_MAX_HW_QUEUES=2

# --- vLLM configuration ---
export VLLM_USE_V1=0
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_SPMD_WORKER=0

# ============================================================================
# Configuration Variables
# ============================================================================
# MODE: "test" (run test script), "vllm-serve" (vLLM serve), "trl-serve" (TRL vllm-serve)
export MODE="${MODE:-test}"

# Model path
export MODEL_PATH="${MODEL_PATH:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-3.2-1B-Instruct}"
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: MODEL_PATH (${MODEL_PATH}) not found"; exit 1; }

# Parallelism configuration
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-2}"
export DP_SIZE="${DP_SIZE:-1}"

# Server configuration (for serve modes)
export VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.8}"

# --- Ray cluster ports ---
export RAY_PORT=${RAY_PORT:-6379}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-7}

# --- Determine nodes/IPs on the hsn network ---
mapfile -t NODELIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE=${NODELIST[0]}
WORKER_NODES=("${NODELIST[@]:1}")

resolve_hsn_ip() { getent hosts "$1-hsn0" | awk '{print $1}'; }
HEAD_IP=$(resolve_hsn_ip "${HEAD_NODE}")
[[ -n "${HEAD_IP}" ]] || { echo "ERROR: failed to resolve head IP"; exit 1; }
export MASTER_ADDR=${HEAD_IP}
export VLLM_HOST_IF=hsn0

echo "============================================"
echo "Ray Cluster Configuration"
echo "============================================"
echo "Head node: ${HEAD_NODE}"
echo "Head IP: ${HEAD_IP}"
echo "Num nodes: ${SLURM_JOB_NUM_NODES}"
echo "MODE: ${MODE}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "TP_SIZE: ${TP_SIZE}, PP_SIZE: ${PP_SIZE}, DP_SIZE: ${DP_SIZE}"
echo "TRANSPORT: ${TRANSPORT}"
echo "NCCL_NET: ${NCCL_NET:-<unset>}"
echo "NCCL_NET_PLUGIN: ${NCCL_NET_PLUGIN:-<unset>}"
echo "FI_PROVIDER: ${FI_PROVIDER:-<unset>}"
if [[ -n "${OFI_NCCL_PROTOCOL:-}" ]]; then
    echo "OFI_NCCL_PROTOCOL: ${OFI_NCCL_PROTOCOL}"
fi
echo "============================================"

# ============================================================================
# Start Ray Cluster with VNI options
# ============================================================================

# Start Ray head via srun with VNI options
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
     --network=single_node_vni,job_vni \
     --export=ALL \
     bash -c "
         echo 'HEAD: Starting Ray with CXI config'
         ulimit -n 65536
         ray stop --force >/dev/null 2>&1 || true
         ray start --head \
             --node-ip-address='${HEAD_IP}' \
             --port=${RAY_PORT} \
             --dashboard-port=${RAY_DASHBOARD_PORT} \
             --num-cpus=${RAY_NUM_CPUS} \
             --num-gpus=8 \
             --block
     " &

# Start Ray workers via srun with VNI options
for n in "${WORKER_NODES[@]}"; do
    WIP=$(resolve_hsn_ip "${n}") || true
    [[ -n "${WIP}" ]] || { echo "ERROR: failed to resolve ${n} IP"; exit 1; }
    
    srun --nodes=1 --ntasks=1 -w "${n}" \
         --network=single_node_vni,job_vni \
         --export=ALL \
         bash -c "
             echo 'WORKER ${n}: Starting Ray with CXI config'
             ulimit -n 65536
             ray stop --force >/dev/null 2>&1 || true
             ray start \
                 --address='${HEAD_IP}':${RAY_PORT} \
                 --node-ip-address='${WIP}' \
                 --num-cpus=${RAY_NUM_CPUS} \
                 --num-gpus=8 \
                 --block
         " &
done

# --- Wait for Ray cluster to be ready ---
for i in {1..30}; do
    if ray status --address="${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
        echo "Ray cluster is up."
        break
    fi
    sleep 2
done

ray status --address="${HEAD_IP}:${RAY_PORT}" || true

# ============================================================================
# Run the selected mode
# ============================================================================

case "${MODE}" in
    test)
        echo "Running vLLM multi-node test..."
        python tests/test_vllm_ray_multinode.py
        ;;
    
    vllm-serve)
        echo "Starting vLLM serve..."
        # vllm serve uses tensor_parallel_size and pipeline_parallel_size
        # Note: vllm serve doesn't support pipeline parallelism directly via CLI
        # Use distributed_executor_backend=ray for multi-node
        vllm serve "${MODEL_PATH}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --distributed-executor-backend ray \
            --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --enforce-eager \
            --trust-remote-code
        ;;
    
    trl-serve)
        echo "Starting TRL vllm-serve..."
        # TRL's vllm-serve is simpler and works well with TRL's training scripts
        trl vllm-serve \
            --model "${MODEL_PATH}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --tensor_parallel_size "${TP_SIZE}" \
            --data_parallel_size "${DP_SIZE}" \
            --gpu_memory_utilization "${GPU_MEMORY_UTIL}" \
            --max_model_len "${MAX_MODEL_LEN}" \
            --enforce_eager \
            --trust_remote_code
        ;;
    
    *)
        echo "ERROR: Unknown MODE '${MODE}'. Use 'test', 'vllm-serve', or 'trl-serve'"
        exit 1
        ;;
esac

echo "Done."

# --- Cleanup on exit ---
trap "ray stop --force >/dev/null 2>&1 || true" EXIT
