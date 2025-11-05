#!/bin/bash
#SBATCH --job-name=grpo-ds-debug
#SBATCH -A SYB114
#SBATCH -N 4
#SBATCH -t 00:15:00
#SBATCH -p batch 
#SBATCH -C nvme
#SBATCH -q debug
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out # Out Path
#SBATCH -e logs/%x-%j.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

set -euo pipefail
set -x

# load modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0

# export compilers
export CC=cc
export CXX=CC
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC

# load aws-ofi-rccl
export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

# load conda, activate env
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

echo "--- Slurm Job Started ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$(echo ${MASTER_ADDR} | sed 's/^\([^\.]*\)/\1-hsn0/')
export MASTER_PORT=29500

# --- Define Paths ---
MODEL_NAME="Llama-3.2-1B-Instruct"

LOCAL_SSD_PATH="/mnt/bb/${USER}/job_${SLURM_JOB_ID}"
LOCAL_ENV_DIR="${LOCAL_SSD_PATH}/venv"
LOCAL_CACHE_DIR="${LOCAL_SSD_PATH}/hf_cache"
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
  echo '--- Staging on node: $(hostname) ---'

  mkdir -p ${LOCAL_SSD_PATH} ${LOCAL_MODEL_DIR} ${LOCAL_ENV_DIR} ${LOCAL_DATA_DIR} ${LOCAL_CONFIG_DIR}

  echo 'Python check:'; which python; python -c 'import torch,sys; print(torch.__version__, sys.executable)'

  echo 'Copying model weights...'
  rsync -a --info=progress2 "${MODEL_PATH}/" "${LOCAL_MODEL_DIR}"

  echo 'Copying dataset...'
  rsync -a --info=progress2 ${DATA_PATH} ${LOCAL_DATA_DIR}

  echo 'Copying DS config...'
  rsync -a --info=progress2 ${DS_PATH} ${LOCAL_CONFIG_DIR}

  mkdir -p ${LOCAL_SSD_PATH}/hf_cache

  echo '--- Staging on $(hostname) complete ---'
"
echo "--- Staging complete on all nodes ---"

echo "--- Launching Distributed Training with RCCL Plugin ---"

# rccl settings
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.

export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export NCCL_OFI_USE_NICLIST=hsn0
export GLOO_SOCKET_IFNAME=hsn0

# for distributed
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-6.3.1
export NCCL_DEBUG=INFO # Un-comment to diagnose NCCL issues if needed
export HSA_XNACK=0
export DS_ACCELERATOR=cuda
export OMP_NUM_THREADS=8

# accelerate / transformer / deepspeed logs
#export ACCELERATE_LOG_LEVEL=info                  # HF Accelerate init logs
#export TRANSFORMERS_VERBOSITY=info
#export DEEPSPEED_LOG_LEVEL=debug  

# --- Per-rank, node-local caches to avoid Triton/Inductor races ---
export LOCAL_CACHE_BASE="${LOCAL_SSD_PATH}/caches"
mkdir -p "${LOCAL_CACHE_BASE}"

# Optional: keep HIP allocator log quiet (unrelated warning you saw)
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False"
# wandb

# pick nodes; use half for weight updates, half for generation
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NUM_NODES=${#NODELIST[@]}
TRAIN_NODES=("${NODELIST[@]:0:2}")
VLLM_NODES=("${NODELIST[@]:2:2}")
TRAIN_NODELIST_CSV="$(IFS=,; echo "${TRAIN_NODES[*]}")"
VLLM_NODELIST_CSV="$(IFS=,; echo "${VLLM_NODES[*]}")"
VLLM_HEAD_NODE="${VLLM_NODES[0]}"
VLLM_WORKER_NODE="${VLLM_NODES[1]}"

# vllm serve settings
export VLLM_HTTP_PORT=${VLLM_HTTP_PORT:-51001}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-DEBUG}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NUM_PP_NODES=${#VLLM_NODES[@]}

# resolve ipv4 of head node
HEAD_IP=$(srun -N1 -n1 -w "$VLLM_HEAD_NODE" bash --noprofile --norc -lc 'hostname -I | awk "{print \$1}"')
WORKER_IP=$(srun -N1 -n1 -w "$VLLM_WORKER_NODE" bash --noprofile --norc -lc 'hostname -I | awk "{print \$1}"')
echo "[debug] VLLM_HEAD=${VLLM_HEAD_NODE} | HEAD_IP=${HEAD_IP} | VLLM_WORKER=${VLLM_WORKER_NODE} | WORKER_IP=${WORKER_IP}"

# proxy
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${HEAD_IP},${VLLM_HEAD_NODE},${VLLM_WORKER_NODE},*.olcf.ornl.gov"
export no_proxy="$NO_PROXY"

# trl client looks at "OPENAI_BASE_URL"
export OPENAI_BASE_URL="http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1"
export OPENAI_API_KEY="dummy" 

# start Ray cluster on two vLLM nodes, then run vLLM with Ray
export RAY_NODE_IP_ADDRESS=${HEAD_IP}
export RAY_PORT=${RAY_PORT:-6379}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# don't let ray set these - manually set them
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# device masking to avoid HIP ordinal errors
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
export ROCR_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"

# Ray HEAD (background srun step)
srun --overlap -N1 -n1 -w "${VLLM_HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    ulimit -n 65536

    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    export ROCR_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    echo 'CUDA Devices: ${CUDA_VISIBLE_DEVICES} | HIP Devices: ${HIP_VISIBLE_DEVICES}'

    export VLLM_HOST_IP=${HEAD_IP}
    export RAY_NODE_IP_ADDRESS=${HEAD_IP}
    ray stop --force >/dev/null 2>&1 || true
    ray start --head \
      --node-ip-address=${HEAD_IP} \
      --port=${RAY_PORT} \
      --dashboard-port=${RAY_DASHBOARD_PORT} \
      --num-gpus=8 \
      --disable-usage-stats \
      --block 
  " &

# Ray WORKER (background srun step)
srun --overlap -N1 -n1 -w "$VLLM_WORKER_NODE" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    ulimit -n 65536

    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    export ROCR_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    echo 'CUDA Devices: ${CUDA_VISIBLE_DEVICES} | HIP Devices: ${HIP_VISIBLE_DEVICES}'

    export VLLM_HOST_IP=${WORKER_IP}
    export RAY_NODE_IP_ADDRESS=${WORKER_IP}
    ray stop --force >/dev/null 2>&1 || true
    ray start --address=${HEAD_IP}:${RAY_PORT} \
      --node-ip-address=${WORKER_IP} \
      --num-gpus=8 \
      --disable-usage-stats \
      --block
  " &

# Wait for Ray to see both nodes, all gpus, before starting vllm
until srun -N1 -n1 -w "${VLLM_HEAD_NODE}" bash --noprofile --norc -c '
  addr="'"${HEAD_IP}"'":"'"${RAY_PORT}"'"
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  unset ROCR_VISIBLE_DEVICES
  ray status --address="$addr" >/tmp/ray_status.txt 2>/dev/null || true
  if grep -qE "GPU[^0-9]*16" /tmp/ray_status.txt; then
    echo READY
  else
    echo NOT_READY
  fi
  exit 0
  ' | grep -q READY; do
  sleep 2
done

# vLLM serve on Ray HEAD (background srun step)
srun --overlap -N1 -n1 -w "$VLLM_HEAD_NODE" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail 

    export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
    export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    export ROCR_VISIBLE_DEVICES=\"\${HIP_VISIBLE_DEVICES}\"
    echo 'CUDA Devices: ${CUDA_VISIBLE_DEVICES} | HIP Devices: ${HIP_VISIBLE_DEVICES}'

    export RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}
    export VLLM_HOST_IP=${HEAD_IP}
    echo \"[vLLM] TP_SIZE=8, PP_SIZE=${NUM_PP_NODES} using Ray at \${RAY_ADDRESS}\"

    vllm serve '${LOCAL_MODEL_DIR}' \
      --host 0.0.0.0 \
      --port ${VLLM_HTTP_PORT} \
      --distributed-executor-backend ray \
      --tensor-parallel-size 8 --pipeline-parallel-size '${NUM_PP_NODES}' \
      --dtype bfloat16 \
      --tokenizer '${LOCAL_MODEL_DIR}' \
      --max-model-len 8192 \
      --enforce-eager \
      --disable-custom-all-reduce \
      --disable-log-stats
  " &

TRAIN_PROBE_NODE=${TRAIN_NODES[0]}
echo "[INFO] Waiting quietly for vLLM at http://${HEAD_IP}:${VLLM_HTTP_PORT}"
srun --overlap -N1 -n1 -w "${TRAIN_PROBE_NODE}" bash -lc ' 
  set -e 
  URL="http://'"${HEAD_IP}"':'"${VLLM_HTTP_PORT}"'/v1/models" 
  for i in $(seq 1 900); do 
  # -s : silent, -o /dev/null : discard body, -f : fail on non-200, 2>/dev/null : drop errors 
  if curl -sf -o /dev/null --connect-timeout 2 "$URL" 2>/dev/null; then 
    echo "[OK] vLLM is ready" 
    exit 0 
  fi 
  sleep 2 
  done 
  echo "[ERROR] vLLM did not become ready in 900s" 
  exit 7 
'

# launch actual trl
srun --cpu-bind=none --accel-bind=g --kill-on-bad-exit=1 --export=ALL \
  -N ${#TRAIN_NODES[@]} \
  --ntasks-per-node=8 \
  -w "${TRAIN_NODELIST_CSV}" \
  bash -c '
  
  export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
  echo "[RANK ${SLURM_PROCID}] ROCR=${ROCR_VISIBLE_DEVICES} HIP=${HIP_VISIBLE_DEVICES}"

  export AMDSMI_GPU_METRICS_CACHE_MS=5000
  if [ "${SLURM_LOCALID}" != "0" ]; then export WANDB_MODE=disabled; fi

  # --- Launch the training ---
  python \
    "'${SLURM_SUBMIT_DIR}'"/scripts/train_grpo_ds.py \
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