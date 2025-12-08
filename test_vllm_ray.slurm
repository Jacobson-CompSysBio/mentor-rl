#!/bin/bash
#SBATCH --job-name=vllm-frontier
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
# WORKING NCCL CONFIGURATION (Socket-based transport - NO OFI)
# ============================================================================
# This configuration bypasses the OFI/CXI fi_domain() VNI allocation issue
# by using RCCL's built-in Socket transport for inter-node communication.
# Successfully tested with 2-node, 16-GPU all_reduce and broadcast operations.
#
# Trade-off: Socket transport is slower than native OFI/CXI, but it's functional
# and allows vLLM multi-node inference to work on Frontier.
# ============================================================================

# DON'T force OFI - let RCCL use its internal socket transport
unset NCCL_NET
unset NCCL_NET_PLUGIN

# Disable InfiniBand (not used on Slingshot)
export NCCL_IB_DISABLE=1

# Use Slingshot NIC for socket communication
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET

# RCCL tuning
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=3600

# Debug level while testing (change to WARN for production)
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
# CRITICAL: Disable v1 engine because Ray Compiled DAG tensor channel fails
# on multi-node with ROCm/RCCL socket transport. The v0 engine works without
# needing Ray's NCCL-based tensor channel.
export VLLM_USE_V1=0

# Disable Ray Compiled DAG (not needed with v0 engine, but set explicitly)
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_SPMD_WORKER=0

# --- Model path ---
export MODEL_PATH="${MODEL_PATH:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-3.2-1B-Instruct}"
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: MODEL_PATH (${MODEL_PATH}) not found"; exit 1; }

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

# NOTE: Use VLLM_HOST_IF instead of VLLM_HOST_IP - each node will automatically
# detect its own IP from the specified network interface. This is cleaner than
# trying to set per-node VLLM_HOST_IP values.
export VLLM_HOST_IF=hsn0

# --- Start Ray head/worker (simple) ---
# Note: VLLM_HOST_IF is set globally, so all vLLM actors will automatically
# get the correct hsn0 IP address for their node.
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" --export=ALL \
  bash -c "echo 'HEAD: VLLM_HOST_IF='\$VLLM_HOST_IF; ulimit -n 65536; ray stop --force >/dev/null 2>&1 || true; ray start --head --node-ip-address='${HEAD_IP}' --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-cpus=${RAY_NUM_CPUS} --num-gpus=8 --block" &

for n in "${WORKER_NODES[@]}"; do
  WIP=$(resolve_hsn_ip "${n}") || true
  [[ -n "${WIP}" ]] || { echo "ERROR: failed to resolve ${n} IP"; exit 1; }
  srun --nodes=1 --ntasks=1 -w "${n}" --export=ALL \
    bash -c "echo 'WORKER ${n}: VLLM_HOST_IF='\$VLLM_HOST_IF; ulimit -n 65536; ray stop --force >/dev/null 2>&1 || true; ray start --address='${HEAD_IP}':${RAY_PORT} --node-ip-address='${WIP}' --num-cpus=${RAY_NUM_CPUS} --num-gpus=8 --block" &
done


# --- Wait for GCS to be up instead of a blind sleep ---
for i in {1..30}; do
  if ray status --address="${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
    echo "Ray is up."
    break
  fi
  sleep 2
done

ray status --address="${HEAD_IP}:${RAY_PORT}" || true

# --- Run the test on the head node ---
python tests/test_vllm_ray_multinode.py

# --- Cleanup on exit ---
trap "ray stop --force >/dev/null 2>&1 || true" EXIT