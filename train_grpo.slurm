#!/bin/bash
#SBATCH --job-name=grpo-ds-debug
#SBATCH -A SYB114
#SBATCH -N 4 
#SBATCH -t 0:45:00
#SBATCH -p batch 
#SBATCH -C nvme
#SBATCH -q debug
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail
set -x

# load modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0

export CC=cc
export CXX=CC
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC

# NOTE: We do NOT load aws-ofi-rccl since we're using socket transport.
# The OFI plugin would try to use Slingshot CXI which fails without VNI allocation.
# Socket transport is slower but works without special network setup.
# export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

# load conda, activate env
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

echo "--- Slurm Job Started ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"

# NOTE: MASTER_ADDR/PORT for training will be set later, specific to training nodes
# Don't export globally here as it confuses vLLM which uses its own HEAD_IP
TRAIN_MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
TRAIN_MASTER_ADDR=$(echo ${TRAIN_MASTER_ADDR} | sed 's/^\([^\.]*\)/\1-hsn0/')
TRAIN_MASTER_PORT=29500

# --- Define Paths ---
MODEL_NAME="Llama-3.2-1B-Instruct"  # Change to your large model

LOCAL_SSD_PATH="/mnt/bb/${USER}/job_${SLURM_JOB_ID}"
LOCAL_MODEL_DIR="${LOCAL_SSD_PATH}/model/${MODEL_NAME}/"
LOCAL_DATA_DIR="${LOCAL_SSD_PATH}/data"
LOCAL_CONFIG_DIR="${LOCAL_SSD_PATH}/config"

VENV_PATH="/lustre/orion/syb111/world-shared/environments/pytorch-rocm"
MODEL_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/${MODEL_NAME}"
DATA_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/"
DS_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/config/"

export HF_HOME="/mnt/bb/$USER/hf_cache_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="/mnt/bb/$USER/ds_cache_${SLURM_JOB_ID}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# --- STAGE 1: Stage Data to Local SSD on Each Node ---
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  echo \"--- Staging on node: \$(hostname) ---\"
  mkdir -p ${LOCAL_SSD_PATH} ${LOCAL_MODEL_DIR} ${LOCAL_DATA_DIR} ${LOCAL_CONFIG_DIR}
  echo 'Python check:'; which python; python -c 'import torch,sys; print(torch.__version__, sys.executable)'
  echo 'Copying model weights...'
  rsync -a --info=progress2 \"${MODEL_PATH}/\" \"${LOCAL_MODEL_DIR}\"
  echo 'Copying dataset...'
  rsync -a --info=progress2 ${DATA_PATH} ${LOCAL_DATA_DIR}
  echo 'Copying DS config...'
  rsync -a --info=progress2 ${DS_PATH} ${LOCAL_CONFIG_DIR}
  mkdir -p ${LOCAL_SSD_PATH}/hf_cache
  echo '--- Staging on \$(hostname) complete ---'
"
echo "--- Staging complete on all nodes ---"

echo "--- Launching Distributed Training with RCCL Plugin ---"

# ========================================
# CRITICAL: RCCL/NCCL Environment Setup for Multi-Node
# ========================================
# This configuration uses RCCL's built-in Socket transport for inter-node
# communication. It bypasses the OFI/CXI fi_domain() VNI allocation issue
# that occurs when SLINGSHOT_VNIS is not allocated in the job environment.
# Trade-off: Socket transport is slower than native OFI/CXI, but it works.
# ========================================

# DON'T let RCCL load external network plugins - use built-in socket transport
unset NCCL_NET
unset NCCL_NET_PLUGIN

# Disable InfiniBand (not used on Slingshot)
export NCCL_IB_DISABLE=1

# Use Slingshot NIC for socket communication
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET

# RCCL tuning for socket transport
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=3600

# Gloo rendezvous
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_IPV6=0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# For distributed
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-6.3.1
export NCCL_DEBUG=WARN  # Change to INFO for debugging
export HSA_XNACK=0
export DS_ACCELERATOR=cuda
export OMP_NUM_THREADS=8

# Point vLLM to the system RCCL
export VLLM_NCCL_SO_PATH=${ROCM_HOME}/lib/librccl.so

# Per-rank, node-local caches
export LOCAL_CACHE_BASE="${LOCAL_SSD_PATH}/caches"
mkdir -p "${LOCAL_CACHE_BASE}"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False"

# ========================================
# Node Allocation Strategy for Large Models
# ========================================
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NUM_NODES=${#NODELIST[@]}

# For 100B+ models, use more nodes for vLLM
# Example: 8 nodes total -> 2 for training, 6 for vLLM
TRAIN_NODE_COUNT=$(( $SLURM_NNODES / 2 ))
VLLM_NODE_COUNT=$((NUM_NODES - TRAIN_NODE_COUNT))

TRAIN_NODES=("${NODELIST[@]:0:${TRAIN_NODE_COUNT}}")
VLLM_NODES=("${NODELIST[@]:${TRAIN_NODE_COUNT}:${VLLM_NODE_COUNT}}")

TRAIN_NODELIST_CSV="$(IFS=,; echo "${TRAIN_NODES[*]}")"
VLLM_NODELIST_CSV="$(IFS=,; echo "${VLLM_NODES[*]}")"
VLLM_HEAD_NODE="${VLLM_NODES[0]}"

echo "=== Node Allocation ==="
echo "Total nodes: ${NUM_NODES}"
echo "Training nodes (${TRAIN_NODE_COUNT}): ${TRAIN_NODELIST_CSV}"
echo "vLLM nodes (${VLLM_NODE_COUNT}): ${VLLM_NODELIST_CSV}"
echo "vLLM head: ${VLLM_HEAD_NODE}"

# ========================================
# vLLM Configuration for Large Models
# ========================================
export VLLM_HTTP_PORT=${VLLM_HTTP_PORT:-51001}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# For 100B+ models:
# - TP size: 8 (one node)
# - PP size: number of vLLM nodes
export TENSOR_PARALLEL_SIZE=8
export PIPELINE_PARALLEL_SIZE=${VLLM_NODE_COUNT}

echo "=== vLLM Parallelism Config ==="
echo "Tensor Parallel: ${TENSOR_PARALLEL_SIZE}"
echo "Pipeline Parallel: ${PIPELINE_PARALLEL_SIZE}"
echo "Total GPUs for inference: $((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))"

# Resolve IPs
export HEAD_IP=$(getent hosts ${VLLM_HEAD_NODE}-hsn0 | awk '{print $1}')
echo "[INFO] VLLM_HEAD=${VLLM_HEAD_NODE} | HEAD_IP=${HEAD_IP} (hsn0)"


# Proxy settings
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90
export NO_PROXY="localhost,127.0.0.1,${TRAIN_MASTER_ADDR},${HEAD_IP},${VLLM_NODELIST_CSV//,/,},*.olcf.ornl.gov"
export no_proxy="$NO_PROXY"

export OPENAI_BASE_URL="http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1"
export OPENAI_API_KEY="dummy"

# ========================================
# Ray Configuration for Multi-Node
# ========================================
export RAY_NODE_IP_ADDRESS=${HEAD_IP}
export RAY_PORT=${RAY_PORT:-6379}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Critical: Don't let Ray override device visibility
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

# vLLM v0 engine (required to avoid Ray Compiled DAG issues on multi-node ROCm)
export VLLM_USE_V1=0
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_SPMD_WORKER=0

# Use hsn0 interface for vLLM communication
export VLLM_HOST_IF=hsn0

# Critical: set RAY_CARRY_OVER_ENV_VARS to propagate env vars

# ========================================
# Start Ray Cluster on All vLLM Nodes
# ========================================

# Ray HEAD
srun --overlap -N1 -n1 -w "${VLLM_HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    # 1. EXPLICITLY SOURCE CONDA (Crucial for Ray to find python libs)
    source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
    source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

    # 2. Socket-based RCCL transport configuration (no OFI)
    unset NCCL_NET
    unset NCCL_NET_PLUGIN
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=hsn0
    export NCCL_SOCKET_FAMILY=AF_INET
    export NCCL_CROSS_NIC=1
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=4
    export NCCL_P2P_DISABLE=0
    export GLOO_SOCKET_IFNAME=hsn0
    export GLOO_SOCKET_FAMILY=AF_INET
    export GLOO_IPV6=0
    
    # Ray setup
    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    export ROCR_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"

    # vLLM v0 engine settings
    export VLLM_USE_V1=0
    export VLLM_USE_RAY_COMPILED_DAG=0
    export VLLM_USE_RAY_SPMD_WORKER=0
    export VLLM_HOST_IF=hsn0
    export VLLM_HOST_IP=${HEAD_IP}
    export RAY_NODE_IP_ADDRESS=${HEAD_IP}
    
    ulimit -n 65536
    ray stop --force >/dev/null 2>&1 || true
    ray start --head \
      --node-ip-address=${HEAD_IP} \
      --port=${RAY_PORT} \
      --dashboard-port=${RAY_DASHBOARD_PORT} \
      --num-gpus=8 \
      --block 
  " &

sleep 10  # Give head time to start

# Ray WORKERS (all remaining vLLM nodes)
for ((i=1; i<${VLLM_NODE_COUNT}; i++)); do
  WORKER_NODE="${VLLM_NODES[$i]}"
  WORKER_IP=$(getent hosts ${WORKER_NODE}-hsn0 | awk '{print $1}')  
  echo "[INFO] Starting Ray worker on ${WORKER_NODE} (hsn0 ${WORKER_IP})"
  
  srun --overlap -N1 -n1 -w "$WORKER_NODE" --export=ALL \
    bash --noprofile --norc -c "
      set -euo pipefail
      # 1. EXPLICITLY SOURCE CONDA
      source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
      source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

      # 2. Socket-based RCCL transport configuration (no OFI)
      unset NCCL_NET
      unset NCCL_NET_PLUGIN
      export NCCL_IB_DISABLE=1
      export NCCL_SOCKET_IFNAME=hsn0
      export NCCL_SOCKET_FAMILY=AF_INET
      export NCCL_CROSS_NIC=1
      export NCCL_SOCKET_NTHREADS=4
      export NCCL_NSOCKS_PERTHREAD=4
      export NCCL_P2P_DISABLE=0
      export GLOO_SOCKET_IFNAME=hsn0
      export GLOO_SOCKET_FAMILY=AF_INET
      export GLOO_IPV6=0

      export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
      export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
      export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
      export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      export CUDA_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
      export ROCR_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"

      # vLLM v0 engine settings
      export VLLM_USE_V1=0
      export VLLM_USE_RAY_COMPILED_DAG=0
      export VLLM_USE_RAY_SPMD_WORKER=0
      export VLLM_HOST_IF=hsn0
      export VLLM_HOST_IP=${WORKER_IP}
      export RAY_NODE_IP_ADDRESS=${WORKER_IP}
      
      ulimit -n 65536
      ray start --address=${HEAD_IP}:${RAY_PORT} \
        --node-ip-address=${WORKER_IP} \
        --num-gpus=8 \
        --block
    " &
done

echo "##################################################"
echo "RAY CLUSTER STARTED. LAUNCHING VLLM SERVER"
echo "##################################################"

sleep 5

# ========================================
# vLLM Serve with Pipeline Parallelism
# ========================================
# Socket-based RCCL env vars to propagate to Ray workers
export VLLM_CARRY_OVER_ENV_VARS="NCCL_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,NCCL_IB_DISABLE,NCCL_CROSS_NIC,NCCL_SOCKET_NTHREADS,NCCL_NSOCKS_PERTHREAD,NCCL_P2P_DISABLE,GLOO_SOCKET_IFNAME,GLOO_SOCKET_FAMILY,GLOO_IPV6,VLLM_USE_V1,VLLM_USE_RAY_COMPILED_DAG,VLLM_USE_RAY_SPMD_WORKER,VLLM_HOST_IF,ROCM_HOME,HSA_XNACK,PYTORCH_ROCM_ARCH,VLLM_NCCL_SO_PATH"

echo "vLLM will propagate these env vars to Ray workers: $VLLM_CARRY_OVER_ENV_VARS"

srun --overlap -N1 -n1 -w "$VLLM_HEAD_NODE" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail 
    
    # Socket-based RCCL transport configuration (no OFI)
    unset NCCL_NET
    unset NCCL_NET_PLUGIN
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME=hsn0
    export NCCL_SOCKET_FAMILY=AF_INET
    export NCCL_CROSS_NIC=1
    export NCCL_SOCKET_NTHREADS=4
    export NCCL_NSOCKS_PERTHREAD=4
    export GLOO_SOCKET_IFNAME=hsn0
    export GLOO_IPV6=0
    
    # vLLM v0 engine (avoids Ray Compiled DAG issues on multi-node ROCm)
    export VLLM_USE_V1=0
    export VLLM_USE_RAY_COMPILED_DAG=0
    export VLLM_USE_RAY_SPMD_WORKER=0
    export VLLM_HOST_IF=hsn0
    
    vllm serve '${LOCAL_MODEL_DIR}' \
      --host 0.0.0.0 \
      --port ${VLLM_HTTP_PORT} \
      --distributed-executor-backend ray \
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
      --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
      --dtype bfloat16 \
      --tokenizer '${LOCAL_MODEL_DIR}' \
      --max-model-len 256 \
      --enforce-eager \
      --disable-custom-all-reduce \
      --gpu-memory-utilization 0.90
  " &

echo "##################################################"
echo "VLLM SERVED. PROBING SERVER..."
echo "##################################################"

TRAIN_PROBE_NODE=${VLLM_HEAD_NODE}
echo "[INFO] Waiting quietly for vLLM at http://${HEAD_IP}:${VLLM_HTTP_PORT}"
srun --overlap -N1 -n1 --export=ALL -w "${TRAIN_PROBE_NODE}" bash -c "
  set -e
  
  echo '[1/4] Testing /health endpoint...'
  for i in {1..180}; do
    if curl -sf http://${HEAD_IP}:${VLLM_HTTP_PORT}/health 2>/dev/null; then
      echo \"[HEALTH CHECK] Health check passed on attempt \$i\"
      break
    fi
    echo \"[HEALTH CHECK] vLLM not ready yet (attempt \$i/180), sleeping 10s...\"
    sleep 10
  done 
  
  echo ''
  echo '[2/4] Testing /v1/models endpoint...'
  if MODELS=\$(curl -sf http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1/models 2>/dev/null); then
    if echo \"\$MODELS\" | grep -q '\"object\":\"list\"'; then
      echo '[HEALTH CHECK] Models endpoint working'
      # Extract the actual model ID from the response
      MODEL_ID=\$(echo \"\$MODELS\" | python3 -c \"import sys, json; data = json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '')\" 2>/dev/null)
      if [ -z \"\$MODEL_ID\" ]; then
        echo '[HEALTH CHECK] Could not extract model ID from response'
        echo \"  Response: \$MODELS\"
        exit 1
      fi
      echo \"  Available models: \$MODEL_ID\"
    else
      echo '[HEALTH CHECK] Models endpoint returned unexpected payload'
      exit 1
    fi
  else
    echo '[HEALTH CHECK] Models endpoint failed (connection or HTTP error)'
    exit 1
  fi
  
  echo ''
  echo '[3/4] Testing /v1/completions with a simple prompt...'
  RESPONSE=\$(curl -sf http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      \"model\": \"'\$MODEL_ID'\",
      \"prompt\": \"Hello, world! Say hi:\",
      \"max_tokens\": 10,
      \"temperature\": 0
    }' 2>/dev/null)
  
  if echo \"\$RESPONSE\" | grep -q '\"choices\"'; then
    echo '[HEALTH CHECK] Completion endpoint working'
    # Extract and show the generated text
    GENERATED=\$(echo \"\$RESPONSE\" | grep -o '\"text\":\"[^\"]*\"' | head -1)
    echo \"  Sample output: \$GENERATED\"
  else
    echo '[HEALTH CHECK] Completion endpoint failed'
    echo \"  Response: \$RESPONSE\"
    exit 1
  fi
  
  echo ''
  echo '[4/4] Verifying OPENAI_BASE_URL environment variable...'
  echo \"  OPENAI_BASE_URL=\${OPENAI_BASE_URL:-NOT SET}\"
  
  if [ -z \"\${OPENAI_BASE_URL:-}\" ]; then
    echo '✗ OPENAI_BASE_URL is not set!'
    exit 1
  fi
  
  echo '[HEALTH CHECK] PASSED'
  echo '[HEALTH CHECK] vLLM server is ready for TRL training!'
"

echo "##################################################"
echo "SERVER READY. LAUNCHING TRL TRAINING..."
echo "##################################################"

# Launch training - set MASTER_ADDR for training nodes
srun --cpu-bind=none --accel-bind=g --kill-on-bad-exit=1 --export=ALL \
  -N ${#TRAIN_NODES[@]} \
  --ntasks-per-node=8 \
  -w "${TRAIN_NODELIST_CSV}" \
  bash -c '
  
  # Set MASTER_ADDR for torch distributed (training nodes only)
  export MASTER_ADDR="'"${TRAIN_MASTER_ADDR}"'"  
  export HEAD_IP="'"${HEAD_IP}"'"  
  export VLLM_HTTP_PORT="'"${VLLM_HTTP_PORT}"'"
  export MASTER_PORT="'"${TRAIN_MASTER_PORT}"'"
  
  export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
  echo "[RANK ${SLURM_PROCID}] ROCR=${ROCR_VISIBLE_DEVICES} HIP=${HIP_VISIBLE_DEVICES}"

  export AMDSMI_GPU_METRICS_CACHE_MS=5000
  if [ "${SLURM_LOCALID}" != "0" ]; then export WANDB_MODE=disabled; fi

  python \
    "'${SLURM_SUBMIT_DIR}'"/scripts/train_grpo.py \
      --model_path="'${LOCAL_MODEL_DIR}'" \
      --output_dir="'${LOCAL_SSD_PATH}'/checkpoints/" \
      --dataset_path="'${LOCAL_DATA_DIR}'/qa_pairs.json" \
      --deepspeed="'${LOCAL_CONFIG_DIR}'/ds_zero3.json" \
      --seed=900913 \
      --num_train_epochs=1 \
      --per_device_train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --learning_rate=2e-5 \
      --logging_steps=1 \
      --lora_r=16 \
      --lora_alpha=32 \
      --lora_dropout=0.05 \
      --run_inference_after_training
'

# Cleanup
echo "Training complete. Shutting down..."
ray stop --force || true