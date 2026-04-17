#!/bin/bash
#SBATCH --job-name=mentor-traj-gen
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

module load craype-accel-amd-gfx90a
module load rocm/6.3.1

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

REPO_ROOT="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl"
MODEL_PATH="${MODEL_PATH:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/gpt-oss-120b-bf16}"
TASKS_PATH="${TASKS_PATH:-${REPO_ROOT}/data/corum_corpus/tasks.train.jsonl}"
STORE_DIR="${STORE_DIR:-${REPO_ROOT}/data/humannet_multiplex_store}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/corum_trajectories/gpt_oss_120b_job_${SLURM_JOB_ID}}"
ENCODINGS_SHARED_DIR="${ENCODINGS_SHARED_DIR:-${REPO_ROOT}/data/tiktoken_encodings}"

MAX_TASKS="${MAX_TASKS:-8}"
TASK_SHARD_INDEX="${TASK_SHARD_INDEX:-0}"
TASK_SHARD_COUNT="${TASK_SHARD_COUNT:-1}"
MAX_STEPS="${MAX_STEPS:-4}"
N_ACT="${N_ACT:-4}"
N_VER="${N_VER:-2}"
GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.8}"
GENERATOR_TOP_P="${GENERATOR_TOP_P:-0.95}"
GENERATOR_MAX_COMPLETION_TOKENS="${GENERATOR_MAX_COMPLETION_TOKENS:-700}"
GENERATOR_REASONING_EFFORT="${GENERATOR_REASONING_EFFORT:-low}"
GENERATOR_TIMEOUT_SECONDS="${GENERATOR_TIMEOUT_SECONDS:-900}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-51001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

STAGE_MODEL_TO_NVME="${STAGE_MODEL_TO_NVME:-1}"
LOCAL_NVME_ROOT="${LOCAL_NVME_ROOT:-/mnt/bb/${USER}/${SLURM_JOB_ID}}"
LOCAL_ENCODINGS_DIR="${LOCAL_NVME_ROOT}/tiktoken_encodings"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-safetensors}"
VLLM_SAFETENSORS_LOAD_STRATEGY="${VLLM_SAFETENSORS_LOAD_STRATEGY:-eager}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_HEALTH_MAX_WAIT_SECONDS="${VLLM_HEALTH_MAX_WAIT_SECONDS:-5400}"
VLLM_HEALTH_POLL_SECONDS="${VLLM_HEALTH_POLL_SECONDS:-10}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-auto}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${ENCODINGS_SHARED_DIR}"

mapfile -t NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${NODE_LIST[0]}"

resolve_hsn_ip() {
  local node_name="$1"
  getent hosts "${node_name}-hsn0" | awk '{print $1}'
}

HEAD_IP="$(resolve_hsn_ip "${HEAD_NODE}")"
if [[ -z "${HEAD_IP}" ]]; then
  log "Failed to resolve head node HSN IP for ${HEAD_NODE}"
  exit 1
fi

ensure_encoding_file() {
  local target_path="$1"
  local url="$2"
  if [[ -f "${target_path}" ]]; then
    return 0
  fi
  log "Downloading missing encoding file: ${target_path}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "${url}" -o "${target_path}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${target_path}" "${url}"
  else
    log "Neither curl nor wget is available to download ${url}"
    exit 1
  fi
}

ensure_encoding_file \
  "${ENCODINGS_SHARED_DIR}/o200k_base.tiktoken" \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
ensure_encoding_file \
  "${ENCODINGS_SHARED_DIR}/cl100k_base.tiktoken" \
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

SERVE_MODEL_PATH="${MODEL_PATH}"
SERVE_ENCODINGS_DIR="${ENCODINGS_SHARED_DIR}"
if [[ "${STAGE_MODEL_TO_NVME}" == "1" ]]; then
  if [[ -d /mnt/bb || -d "/mnt/bb/${USER}" ]]; then
    STAGED_MODEL_PATH="${LOCAL_NVME_ROOT}/$(basename "${MODEL_PATH}")"
    mkdir -p "${LOCAL_NVME_ROOT}"
    log "Staging model to node-local storage: ${STAGED_MODEL_PATH}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete --info=stats2 "${MODEL_PATH}/" "${STAGED_MODEL_PATH}/"
    else
      cp -a "${MODEL_PATH}" "${STAGED_MODEL_PATH}"
    fi
    SERVE_MODEL_PATH="${STAGED_MODEL_PATH}"
    mkdir -p "${LOCAL_ENCODINGS_DIR}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete "${ENCODINGS_SHARED_DIR}/" "${LOCAL_ENCODINGS_DIR}/"
    else
      cp -a "${ENCODINGS_SHARED_DIR}/." "${LOCAL_ENCODINGS_DIR}/"
    fi
    SERVE_ENCODINGS_DIR="${LOCAL_ENCODINGS_DIR}"
  else
    log "Node-local NVMe path was not found. Using the source model path."
  fi
fi

if [[ "${DISTRIBUTED_EXECUTOR_BACKEND}" == "auto" ]]; then
  if [[ "${SLURM_NNODES}" -gt 1 || "${PIPELINE_PARALLEL_SIZE}" -gt 1 ]]; then
    EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND="ray"
  else
    EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND="mp"
  fi
else
  EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND}"
fi

log "HEAD_NODE=${HEAD_NODE}"
log "HEAD_IP=${HEAD_IP}"
log "MODEL_PATH=${MODEL_PATH}"
log "SERVE_MODEL_PATH=${SERVE_MODEL_PATH}"
log "SERVE_ENCODINGS_DIR=${SERVE_ENCODINGS_DIR}"
log "TASKS_PATH=${TASKS_PATH}"
log "OUT_DIR=${OUT_DIR}"
log "TASK_SHARD_INDEX=${TASK_SHARD_INDEX}"
log "TASK_SHARD_COUNT=${TASK_SHARD_COUNT}"
log "GENERATOR_REASONING_EFFORT=${GENERATOR_REASONING_EFFORT}"
log "GENERATOR_TIMEOUT_SECONDS=${GENERATOR_TIMEOUT_SECONDS}"
log "EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND=${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND}"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
  jobs -pr | xargs -r kill || true
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
unset ROCR_VISIBLE_DEVICES || true
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export VLLM_USE_V1=0
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_SPMD_WORKER=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_IPV6=0
export VLLM_HOST_IF=hsn0
export VLLM_CARRY_OVER_ENV_VARS="NCCL_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,NCCL_IB_DISABLE,NCCL_CROSS_NIC,NCCL_SOCKET_NTHREADS,NCCL_NSOCKS_PERTHREAD,GLOO_SOCKET_IFNAME,GLOO_SOCKET_FAMILY,GLOO_IPV6,VLLM_USE_V1,VLLM_USE_RAY_COMPILED_DAG,VLLM_USE_RAY_SPMD_WORKER,VLLM_HOST_IF"

if [[ "${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND}" == "ray" ]]; then
  log "Starting Ray head on ${HEAD_NODE}"
  srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
    bash --noprofile --norc -c "
      set -euo pipefail
      source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
      source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
      ulimit -n 65536
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      export HIP_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}
      unset ROCR_VISIBLE_DEVICES || true
      ray stop --force >/dev/null 2>&1 || true
      export VLLM_HOST_IP=${HEAD_IP}
      export RAY_NODE_IP_ADDRESS=${HEAD_IP}
      ray start --head \
        --node-ip-address=${HEAD_IP} \
        --port=${RAY_PORT} \
        --dashboard-port=${RAY_DASHBOARD_PORT} \
        --num-gpus=8 \
        --block
    " &

  sleep 10

  for WORKER_NODE in "${NODE_LIST[@]:1}"; do
    WORKER_IP="$(resolve_hsn_ip "${WORKER_NODE}")"
    if [[ -z "${WORKER_IP}" ]]; then
      log "Failed to resolve worker node HSN IP for ${WORKER_NODE}"
      exit 1
    fi
    log "Starting Ray worker on ${WORKER_NODE}"
    srun --overlap -N1 -n1 -w "${WORKER_NODE}" --export=ALL \
      bash --noprofile --norc -c "
        set -euo pipefail
        source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
        source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
        ulimit -n 65536
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export HIP_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}
        unset ROCR_VISIBLE_DEVICES || true
        export VLLM_HOST_IP=${WORKER_IP}
        export RAY_NODE_IP_ADDRESS=${WORKER_IP}
        ray stop --force >/dev/null 2>&1 || true
        ray start \
          --address=${HEAD_IP}:${RAY_PORT} \
          --node-ip-address=${WORKER_IP} \
          --num-gpus=8 \
          --block
      " &
  done

  sleep 15
  ray status --address="${HEAD_IP}:${RAY_PORT}" || true
else
  log "Skipping Ray startup because the backend is ${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND}."
fi

VLLM_ENFORCE_EAGER_FLAG=""
if [[ "${VLLM_ENFORCE_EAGER}" == "1" ]]; then
  VLLM_ENFORCE_EAGER_FLAG="--enforce-eager"
fi

log "Starting vLLM on ${HEAD_NODE}"
srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
    source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HIP_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}
    unset ROCR_VISIBLE_DEVICES || true
    export TIKTOKEN_ENCODINGS_BASE='${SERVE_ENCODINGS_DIR}'
    export VLLM_HOST_IP=${HEAD_IP}
    vllm serve '${SERVE_MODEL_PATH}' \
      --host 0.0.0.0 \
      --port ${VLLM_HTTP_PORT} \
      --distributed-executor-backend ${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND} \
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
      --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
      --dtype bfloat16 \
      --tokenizer '${SERVE_MODEL_PATH}' \
      --load-format ${VLLM_LOAD_FORMAT} \
      --safetensors-load-strategy ${VLLM_SAFETENSORS_LOAD_STRATEGY} \
      --max-model-len ${MAX_MODEL_LEN} \
      --disable-custom-all-reduce \
      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
      ${VLLM_ENFORCE_EAGER_FLAG}
  " &

log "Waiting for vLLM to become healthy at http://${HEAD_IP}:${VLLM_HTTP_PORT}"
HEALTH_READY=0
MAX_HEALTH_ATTEMPTS=$(( (VLLM_HEALTH_MAX_WAIT_SECONDS + VLLM_HEALTH_POLL_SECONDS - 1) / VLLM_HEALTH_POLL_SECONDS ))
for attempt in $(seq 1 "${MAX_HEALTH_ATTEMPTS}"); do
  if curl -sf "http://${HEAD_IP}:${VLLM_HTTP_PORT}/health" >/dev/null 2>&1; then
    HEALTH_READY=1
    break
  fi
  if (( attempt % 6 == 0 )); then
    ELAPSED_SECONDS=$(( attempt * VLLM_HEALTH_POLL_SECONDS ))
    log "vLLM is still loading after ${ELAPSED_SECONDS}s."
  fi
  sleep "${VLLM_HEALTH_POLL_SECONDS}"
done

if [[ "${HEALTH_READY}" != "1" ]]; then
  log "vLLM did not become healthy within ${VLLM_HEALTH_MAX_WAIT_SECONDS}s."
  exit 1
fi

if ! curl -sf "http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1/models" | tee "${OUT_DIR}/served_models.json"; then
  log "vLLM became healthy, but /v1/models did not return successfully."
  exit 1
fi

log "Launching trajectory generation"
srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
    source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export HIP_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}
    unset ROCR_VISIBLE_DEVICES || true
    cd '${REPO_ROOT}'
    export PYTHONPATH='${REPO_ROOT}':\${PYTHONPATH:-}
    python scripts/generate_trajectories.py \
      --tasks-path '${TASKS_PATH}' \
      --out-dir '${OUT_DIR}' \
      --store-dir '${STORE_DIR}' \
      --candidate-source model_vllm \
      --generator-api-base 'http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1' \
      --generator-temperature ${GENERATOR_TEMPERATURE} \
      --generator-top-p ${GENERATOR_TOP_P} \
      --generator-max-completion-tokens ${GENERATOR_MAX_COMPLETION_TOKENS} \
      --generator-reasoning-effort ${GENERATOR_REASONING_EFFORT} \
      --generator-timeout-seconds ${GENERATOR_TIMEOUT_SECONDS} \
      --max-tasks ${MAX_TASKS} \
      --task-shard-index ${TASK_SHARD_INDEX} \
      --task-shard-count ${TASK_SHARD_COUNT} \
      --max-steps ${MAX_STEPS} \
      --n-act ${N_ACT} \
      --n-ver ${N_VER}
  "

log "Trajectory generation finished: ${OUT_DIR}"
