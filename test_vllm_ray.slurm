#!/bin/bash
#SBATCH --job-name=vllm-multi-test
#SBATCH -A SYB114
#SBATCH -N 2
#SBATCH -t 0:15:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH -q debug
#SBATCH --gpus-per-node=8
#SBATCH -o logs/test-multi-%j.out
#SBATCH -e logs/test-multi-%j.err

set -euo pipefail

# --- 1. Load Modules ---
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1

# --- 2. Activate Conda ---
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

# --- 3. Environment Setup ---
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_ATS=0

# CRITICAL: AWS-OFI-RCCL Plugin Path
export AWS_OFI_RCCL_LIB=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-nccl/lib

# NCCL Configuration - UPDATED
export NCCL_NET=AWS  # Load AWS-OFI-RCCL plugin
export NCCL_NET_GDR_LEVEL=3
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO  # Changed to INFO to see plugin loading
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=3600
export NCCL_COLLNET_ENABLE=0  # aws-ofi-nccl does not ship a CollNet plugin

# Gloo for CPU fallback
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=hsn0

# Library Linking - CRITICAL: AWS-OFI-RCCL must come BEFORE ROCm
export ROCM_HOME=/opt/rocm-6.3.1
export VLLM_NCCL_SO_PATH=${ROCM_HOME}/lib/librccl.so

# CRITICAL: Put AWS-OFI-RCCL first in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${AWS_OFI_RCCL_LIB}:${ROCM_HOME}/lib:${LD_LIBRARY_PATH}

# Verify the plugin exists
if [ ! -f "${AWS_OFI_RCCL_LIB}/librccl-net.so" ]; then
    echo "ERROR: AWS-OFI-RCCL plugin not found at ${AWS_OFI_RCCL_LIB}/librccl-net.so"
    exit 1
fi
echo "Found AWS-OFI-RCCL plugin at: ${AWS_OFI_RCCL_LIB}/librccl-net.so"

# GPU Visibility
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_DEVICE_ORDINAL=0,1,2,3,4,5,6,7

# Ray Configuration
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0

# vLLM Configuration
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=ROCM_FLASH

# Model Path
export MODEL_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-3.2-1B-Instruct"

# --- 4. Get Node Information ---
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -i | awk '{print $1}')
port=6379

echo "=== Node Configuration ==="
echo "Head node: $head_node"
echo "Head IP: $head_node_ip"
echo "Worker nodes: ${nodes_array[@]:1}"
echo "AWS-OFI-RCCL: $AWS_OFI_RCCL_LIB"
echo "=========================="

# --- 5. Start Ray Head ---
echo ""
echo "Starting Ray Head on $head_node ($head_node_ip)"
srun --export=ALL --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
        --node-ip-address="$head_node_ip" \
        --port=$port \
        --num-cpus=7 \
        --num-gpus=8 \
        --block &

sleep 15

# --- 6. Start Ray Workers ---
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Starting Ray Workers on remaining $worker_num node(s)"

for ((i=1; i<${#nodes_array[@]}; i++)); do
    node_i=${nodes_array[$i]}
    node_i_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname -i | awk '{print $1}')
    
    echo "  Starting worker on $node_i ($node_i_ip)"
    srun --export=ALL --nodes=1 --ntasks=1 -w "$node_i" \
        ray start \
            --address="$head_node_ip:$port" \
            --node-ip-address="$node_i_ip" \
            --num-cpus=7 \
            --num-gpus=8 \
            --block &
done

sleep 15

# --- 7. Verify Ray Cluster ---
export RAY_ADDRESS="$head_node_ip:$port"
echo ""
echo "=== Ray Cluster Status ==="
ray status
echo "=========================="

# --- 8. Run the Test ---
echo ""
echo "--- Running vLLM Multinode Test ---"
python tests/test_vllm_ray_multinode.py

# --- 9. Cleanup ---
echo ""
echo "Stopping Ray cluster..."
ray stop --force
