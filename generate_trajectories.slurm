#!/bin/bash
#SBATCH --job-name=mentor-traj-gen
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -p extended
##SBATCH -q debug
#SBATCH -C nvme
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
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

REPO_ROOT="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl"
MODELS_ROOT="${MODELS_ROOT:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models}"
MODEL_PATH_INPUT="${MODEL_PATH:-gpt-oss-120b}"
TASKS_PATH="${TASKS_PATH:-${REPO_ROOT}/data/gw_dendrogram_corpus/tasks.train.jsonl}"
STORE_DIR="${STORE_DIR:-${REPO_ROOT}/data/humannet_multiplex_store}"
ENCODINGS_SHARED_DIR="${ENCODINGS_SHARED_DIR:-${REPO_ROOT}/data/tiktoken_encodings}"
SCRATCH="${SCRATCH:-/lustre/orion/syb114/scratch/${USER}}"
SIF="${SIF:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/vllm0141rocm72.sif}"

abspath_dir() {
  local path="$1"
  (
    cd "${path}" >/dev/null 2>&1
    pwd -P
  )
}

resolve_model_path() {
  local model_spec="$1"
  if [[ -d "${model_spec}" ]]; then
    abspath_dir "${model_spec}"
    return 0
  fi
  if [[ -d "${MODELS_ROOT}/${model_spec}" ]]; then
    abspath_dir "${MODELS_ROOT}/${model_spec}"
    return 0
  fi
  return 1
}

slugify_model_name() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//'
}

infer_model_uses_tiktoken() {
  local model_name
  model_name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "${model_name}" == *"gpt-oss"* ]]; then
    echo "1"
  else
    echo "0"
  fi
}

infer_default_tensor_parallel_size() {
  local model_name
  model_name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  case "${model_name}" in
    *"llama-3.2-1b"*|*"llama_3.2_1b"*)
      echo "1"
      ;;
    *)
      echo "8"
      ;;
  esac
}

infer_default_max_model_len() {
  local model_name
  model_name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "${model_name}" == *"gpt-oss"* ]]; then
    echo "16384"
  else
    echo "16384"
  fi
}

if ! MODEL_PATH="$(resolve_model_path "${MODEL_PATH_INPUT}")"; then
  log "Could not resolve MODEL_PATH='${MODEL_PATH_INPUT}'."
  log "Pass either an absolute directory or a model directory name under ${MODELS_ROOT}."
  if [[ -d "${MODELS_ROOT}" ]]; then
    log "Available models under ${MODELS_ROOT}:"
    find "${MODELS_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' | sort
  fi
  exit 1
fi

MODEL_BASENAME="$(basename "${MODEL_PATH}")"
MODEL_SLUG="$(slugify_model_name "${MODEL_BASENAME}")"
MODEL_USES_TIKTOKEN="$(infer_model_uses_tiktoken "${MODEL_BASENAME}")"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/gw_dendrogram_trajectories/${MODEL_SLUG}_job_${SLURM_JOB_ID}}"

if [[ ! -f "${SIF}" ]]; then
  log "Missing vLLM Apptainer image: ${SIF}"
  log "Set SIF=/path/to/vllm0141rocm72.sif or copy the working smoke-test image into ${SCRATCH}/containers."
  exit 1
fi

MAX_TASKS="${MAX_TASKS:-8}"
TASK_SHARD_INDEX="${TASK_SHARD_INDEX:-0}"
TASK_SHARD_COUNT="${TASK_SHARD_COUNT:-1}"
MAX_STEPS="${MAX_STEPS:-6}"
N_ACT="${N_ACT:-6}"
N_VER="${N_VER:-2}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-2}"
GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.8}"
GENERATOR_TOP_P="${GENERATOR_TOP_P:-0.95}"
GENERATOR_MAX_COMPLETION_TOKENS="${GENERATOR_MAX_COMPLETION_TOKENS:-4096}"
GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS="${GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS:-2048}"
GENERATOR_API_MODE="${GENERATOR_API_MODE:-auto}"
GENERATOR_REASONING_EFFORT="${GENERATOR_REASONING_EFFORT:-medium}"
ACTOR_SAMPLING_STRATEGY="${ACTOR_SAMPLING_STRATEGY:-verbalized}"
SELECTION_POLICY="${SELECTION_POLICY:-task_quality}"
SELECTION_SCORE_EPSILON="${SELECTION_SCORE_EPSILON:-0.02}"
PAIR_MINING_STRATEGY="${PAIR_MINING_STRATEGY:-quality_balanced}"
TOOL_COVERAGE_RETRY_COUNT="${TOOL_COVERAGE_RETRY_COUNT:-2}"
RECOVERY_RWR_TOP_K="${RECOVERY_RWR_TOP_K:-500}"
MYGENE_CACHE_PATH="${MYGENE_CACHE_PATH:-${OUT_DIR}/mygene_cache.json}"
ENRICHMENT_CACHE_PATH="${ENRICHMENT_CACHE_PATH:-${OUT_DIR}/gprofiler_enrichment_cache.json}"
ENRICHMENT_BACKGROUND_PATH="${ENRICHMENT_BACKGROUND_PATH:-${REPO_ROOT}/data/gw_dendrogram_corpus/modules.jsonl}"
ALLOW_NETWORK_MYGENE="${ALLOW_NETWORK_MYGENE:-1}"
ALLOW_NETWORK_ENRICHMENT="${ALLOW_NETWORK_ENRICHMENT:-1}"
GENERATOR_TIMEOUT_SECONDS="${GENERATOR_TIMEOUT_SECONDS:-3600}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
SMOKE_TEST_ONLY="${SMOKE_TEST_ONLY:-0}"
SMOKE_TASK_INDEX="${SMOKE_TASK_INDEX:-0}"
SMOKE_TASK_ID="${SMOKE_TASK_ID:-}"
SMOKE_TASK_IDS="${SMOKE_TASK_IDS:-}"
SMOKE_SEED="${SMOKE_SEED:-0}"
SMOKE_N_ACT="${SMOKE_N_ACT:-1}"
SMOKE_GENERATOR_TEMPERATURE="${SMOKE_GENERATOR_TEMPERATURE:-0.2}"
SMOKE_MAX_COMPLETION_TOKENS="${SMOKE_MAX_COMPLETION_TOKENS:-2048}"
SMOKE_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS="${SMOKE_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS:-2048}"
SMOKE_REASONING_EFFORT="${SMOKE_REASONING_EFFORT:-low}"
SMOKE_ACTOR_SAMPLING_STRATEGY="${SMOKE_ACTOR_SAMPLING_STRATEGY:-${ACTOR_SAMPLING_STRATEGY}}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-51001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$(infer_default_tensor_parallel_size "${MODEL_BASENAME}")}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(infer_default_max_model_len "${MODEL_BASENAME}")}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
CALCULATE_KV_SCALES="${CALCULATE_KV_SCALES:-0}"

STAGE_MODEL_TO_NVME="${STAGE_MODEL_TO_NVME:-1}"
LOCAL_NVME_ROOT="${LOCAL_NVME_ROOT:-/mnt/bb/${USER}/${SLURM_JOB_ID}}"
LOCAL_ENCODINGS_DIR="${LOCAL_NVME_ROOT}/tiktoken_encodings"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-safetensors}"
VLLM_SAFETENSORS_LOAD_STRATEGY="${VLLM_SAFETENSORS_LOAD_STRATEGY:-eager}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-0}"
VLLM_USE_TRITON_FLASH_ATTN="${VLLM_USE_TRITON_FLASH_ATTN:-0}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_HEALTH_MAX_WAIT_SECONDS="${VLLM_HEALTH_MAX_WAIT_SECONDS:-5400}"
VLLM_HEALTH_POLL_SECONDS="${VLLM_HEALTH_POLL_SECONDS:-10}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-auto}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${ENCODINGS_SHARED_DIR}"

MAX_TASKS_ARG=()
if [[ -n "${MAX_TASKS}" && "${MAX_TASKS}" != "all" && "${MAX_TASKS}" != "ALL" ]]; then
  MAX_TASKS_ARG=(--max-tasks "${MAX_TASKS}")
fi
MYGENE_NETWORK_ARG=()
if [[ "${ALLOW_NETWORK_MYGENE}" == "1" ]]; then
  MYGENE_NETWORK_ARG=(--allow-network-mygene)
fi
ENRICHMENT_NETWORK_ARG=()
if [[ "${ALLOW_NETWORK_ENRICHMENT}" == "1" ]]; then
  ENRICHMENT_NETWORK_ARG=(--allow-network-enrichment)
fi
ENRICHMENT_BACKGROUND_ARG=()
if [[ -n "${ENRICHMENT_BACKGROUND_PATH}" ]]; then
  ENRICHMENT_BACKGROUND_ARG=(--enrichment-background-path "${ENRICHMENT_BACKGROUND_PATH}")
fi

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

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
export APPTAINERENV_TRITON_CACHE_DIR="/tmp/triton-${SLURM_JOB_ID}/cache/"
export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:-${SCRATCH},/lustre/orion/syb111,/lustre/orion/syb114,/mnt/bb}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

if ! command -v apptainer >/dev/null 2>&1; then
  log "apptainer is not available on PATH."
  exit 1
fi

log "Validating vLLM Apptainer image: ${SIF}"
if ! apptainer exec --fakeroot --writable-tmpfs "${SIF}" /bin/sh -lc 'command -v vllm >/dev/null'; then
  log "Failed to start the vLLM Apptainer image: ${SIF}"
  log "The image may be truncated or corrupt. Recopy it before staging the model."
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

if [[ "${MODEL_USES_TIKTOKEN}" == "1" ]]; then
  ensure_encoding_file \
    "${ENCODINGS_SHARED_DIR}/o200k_base.tiktoken" \
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
  ensure_encoding_file \
    "${ENCODINGS_SHARED_DIR}/cl100k_base.tiktoken" \
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
else
  log "Skipping tiktoken encoding bootstrap for ${MODEL_BASENAME}"
fi

SERVE_MODEL_PATH="${MODEL_PATH}"
SERVE_ENCODINGS_DIR=""
if [[ "${MODEL_USES_TIKTOKEN}" == "1" ]]; then
  SERVE_ENCODINGS_DIR="${ENCODINGS_SHARED_DIR}"
fi
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
    if [[ "${MODEL_USES_TIKTOKEN}" == "1" ]]; then
      mkdir -p "${LOCAL_ENCODINGS_DIR}"
      if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete "${ENCODINGS_SHARED_DIR}/" "${LOCAL_ENCODINGS_DIR}/"
      else
        cp -a "${ENCODINGS_SHARED_DIR}/." "${LOCAL_ENCODINGS_DIR}/"
      fi
      SERVE_ENCODINGS_DIR="${LOCAL_ENCODINGS_DIR}"
    fi
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
log "MODELS_ROOT=${MODELS_ROOT}"
log "MODEL_PATH_INPUT=${MODEL_PATH_INPUT}"
log "MODEL_PATH=${MODEL_PATH}"
log "MODEL_BASENAME=${MODEL_BASENAME}"
log "MODEL_SLUG=${MODEL_SLUG}"
log "MODEL_USES_TIKTOKEN=${MODEL_USES_TIKTOKEN}"
log "SIF=${SIF}"
log "SERVE_MODEL_PATH=${SERVE_MODEL_PATH}"
log "SERVE_ENCODINGS_DIR=${SERVE_ENCODINGS_DIR}"
log "TASKS_PATH=${TASKS_PATH}"
log "OUT_DIR=${OUT_DIR}"
log "TASK_SHARD_INDEX=${TASK_SHARD_INDEX}"
log "TASK_SHARD_COUNT=${TASK_SHARD_COUNT}"
log "MAX_TASKS_ARG=${MAX_TASKS_ARG[*]:-(full shard)}"
log "TASK_CONCURRENCY=${TASK_CONCURRENCY}"
log "GENERATOR_API_MODE=${GENERATOR_API_MODE}"
log "GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS=${GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS}"
log "GENERATOR_REASONING_EFFORT=${GENERATOR_REASONING_EFFORT}"
log "ACTOR_SAMPLING_STRATEGY=${ACTOR_SAMPLING_STRATEGY}"
log "SELECTION_POLICY=${SELECTION_POLICY}"
log "SELECTION_SCORE_EPSILON=${SELECTION_SCORE_EPSILON}"
log "PAIR_MINING_STRATEGY=${PAIR_MINING_STRATEGY}"
log "TOOL_COVERAGE_RETRY_COUNT=${TOOL_COVERAGE_RETRY_COUNT}"
log "RECOVERY_RWR_TOP_K=${RECOVERY_RWR_TOP_K}"
log "GENERATOR_TIMEOUT_SECONDS=${GENERATOR_TIMEOUT_SECONDS}"
log "RUN_SMOKE_TEST=${RUN_SMOKE_TEST}"
log "SMOKE_TEST_ONLY=${SMOKE_TEST_ONLY}"
log "SMOKE_TASK_INDEX=${SMOKE_TASK_INDEX}"
log "SMOKE_TASK_IDS=${SMOKE_TASK_IDS}"
log "SMOKE_GENERATOR_TEMPERATURE=${SMOKE_GENERATOR_TEMPERATURE}"
log "SMOKE_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS=${SMOKE_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS}"
log "SMOKE_ACTOR_SAMPLING_STRATEGY=${SMOKE_ACTOR_SAMPLING_STRATEGY}"
log "SMOKE_TASK_ID=${SMOKE_TASK_ID}"
log "SMOKE_N_ACT=${SMOKE_N_ACT}"
log "SMOKE_SEED=${SMOKE_SEED}"
log "SMOKE_MAX_COMPLETION_TOKENS=${SMOKE_MAX_COMPLETION_TOKENS}"
log "SMOKE_REASONING_EFFORT=${SMOKE_REASONING_EFFORT}"
log "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
log "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
log "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
log "ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL}"
log "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"
log "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
log "VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER}"
log "VLLM_USE_V1=${VLLM_USE_V1}"
log "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
log "VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER}"
log "VLLM_USE_TRITON_FLASH_ATTN=${VLLM_USE_TRITON_FLASH_ATTN}"
log "EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND=${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND}"

if [[ "${SMOKE_TEST_ONLY}" == "1" && "${RUN_SMOKE_TEST}" != "1" ]]; then
  log "SMOKE_TEST_ONLY=1 requires RUN_SMOKE_TEST=1."
  exit 1
fi

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
export VLLM_USE_V1
export VLLM_WORKER_MULTIPROC_METHOD
export VLLM_USE_RAY_COMPILED_DAG=0
export VLLM_USE_RAY_SPMD_WORKER=0
export VLLM_ROCM_USE_AITER
export VLLM_USE_TRITON_FLASH_ATTN
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
export VLLM_CARRY_OVER_ENV_VARS="NCCL_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,NCCL_IB_DISABLE,NCCL_CROSS_NIC,NCCL_SOCKET_NTHREADS,NCCL_NSOCKS_PERTHREAD,GLOO_SOCKET_IFNAME,GLOO_SOCKET_FAMILY,GLOO_IPV6,VLLM_USE_V1,VLLM_WORKER_MULTIPROC_METHOD,VLLM_USE_RAY_COMPILED_DAG,VLLM_USE_RAY_SPMD_WORKER,VLLM_HOST_IF,VLLM_ROCM_USE_AITER,VLLM_USE_TRITON_FLASH_ATTN,SAFETENSORS_FAST_GPU"

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

VLLM_ENABLE_CHUNKED_PREFILL_FLAG="--enable-chunked-prefill"
if [[ "${ENABLE_CHUNKED_PREFILL}" != "1" ]]; then
  VLLM_ENABLE_CHUNKED_PREFILL_FLAG="--no-enable-chunked-prefill"
fi

VLLM_ENABLE_PREFIX_CACHING_FLAG="--enable-prefix-caching"
if [[ "${ENABLE_PREFIX_CACHING}" != "1" ]]; then
  VLLM_ENABLE_PREFIX_CACHING_FLAG="--no-enable-prefix-caching"
fi

VLLM_CALCULATE_KV_SCALES_FLAG=""
if [[ "${CALCULATE_KV_SCALES}" == "1" ]]; then
  VLLM_CALCULATE_KV_SCALES_FLAG="--calculate-kv-scales"
fi

APPTAINER_CMD=(apptainer exec --fakeroot --writable-tmpfs "${SIF}")
SERVER_LOG="${OUT_DIR}/vllm_server.log"
VLLM_SERVE_ARGS=(
  vllm serve "${SERVE_MODEL_PATH}"
  --host 0.0.0.0
  --port "${VLLM_HTTP_PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

if [[ "${MODEL_USES_TIKTOKEN}" != "1" ]]; then
  VLLM_SERVE_ARGS+=(
    --distributed-executor-backend "${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND}"
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
    --dtype bfloat16
    --tokenizer "${SERVE_MODEL_PATH}"
    --load-format "${VLLM_LOAD_FORMAT}"
    --safetensors-load-strategy "${VLLM_SAFETENSORS_LOAD_STRATEGY}"
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
    --tool-call-parser openai
    --enable-auto-tool-choice
    --disable-custom-all-reduce
    ${VLLM_ENABLE_CHUNKED_PREFILL_FLAG}
    ${VLLM_ENABLE_PREFIX_CACHING_FLAG}
  )
  if [[ -n "${VLLM_CALCULATE_KV_SCALES_FLAG}" ]]; then
    VLLM_SERVE_ARGS+=("${VLLM_CALCULATE_KV_SCALES_FLAG}")
  fi
  if [[ -n "${VLLM_ENFORCE_EAGER_FLAG}" ]]; then
    VLLM_SERVE_ARGS+=("${VLLM_ENFORCE_EAGER_FLAG}")
  fi
fi

log "Starting vLLM on ${HEAD_NODE}"
/usr/bin/time -v srun --overlap -N1 -n1 -c56 -G8 -w "${HEAD_NODE}" \
  --ntasks-per-node=1 --gpu-bind=closest --export=ALL \
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONPATH -u PYTHONHOME \
  -u LD_LIBRARY_PATH -u ROCM_PATH -u ROCM_HOME -u HIP_PATH \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  VLLM_HOST_IP="${HEAD_IP}" TIKTOKEN_ENCODINGS_BASE="${SERVE_ENCODINGS_DIR}" \
  "${APPTAINER_CMD[@]}" \
  "${VLLM_SERVE_ARGS[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

unset all_proxy ftp_proxy http_proxy https_proxy no_proxy

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
    tail -20 "${SERVER_LOG}" || true
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "vLLM server exited before becoming healthy."
    tail -200 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep "${VLLM_HEALTH_POLL_SECONDS}"
done

if [[ "${HEALTH_READY}" != "1" ]]; then
  log "vLLM did not become healthy within ${VLLM_HEALTH_MAX_WAIT_SECONDS}s."
  tail -200 "${SERVER_LOG}" || true
  exit 1
fi

if ! curl -sf "http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1/models" | tee "${OUT_DIR}/served_models.json"; then
  log "vLLM became healthy, but /v1/models did not return successfully."
  exit 1
fi

SERVED_MODEL_ID="$(
  python3 - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${OUT_DIR@Q} + "/served_models.json").read_text())
print(payload["data"][0]["id"])
PY
)"
log "SERVED_MODEL_ID=${SERVED_MODEL_ID}"

RUN_FREEZE_PATH="${OUT_DIR}/run_freeze.json"
python3 - <<PY
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(${REPO_ROOT@Q})

def git(args):
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None

payload = {
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "repo": {
        "root": str(repo_root),
        "git_sha": git(["rev-parse", "HEAD"]),
        "git_status_short": git(["status", "--short"]),
    },
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "node_count": os.environ.get("SLURM_NNODES"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
    },
    "model": {
        "model_path_input": ${MODEL_PATH_INPUT@Q},
        "model_path": ${MODEL_PATH@Q},
        "model_basename": ${MODEL_BASENAME@Q},
        "model_slug": ${MODEL_SLUG@Q},
        "model_uses_tiktoken": ${MODEL_USES_TIKTOKEN@Q},
        "serve_model_path": ${SERVE_MODEL_PATH@Q},
        "served_model_id": ${SERVED_MODEL_ID@Q},
    },
    "task_selection": {
        "tasks_path": ${TASKS_PATH@Q},
        "max_tasks": ${MAX_TASKS@Q},
        "task_shard_index": ${TASK_SHARD_INDEX@Q},
        "task_shard_count": ${TASK_SHARD_COUNT@Q},
        "task_shard_strategy": "sha256_module_key_mod",
    },
    "generation": {
        "max_steps": ${MAX_STEPS@Q},
        "n_act": ${N_ACT@Q},
        "n_ver": ${N_VER@Q},
        "task_concurrency": ${TASK_CONCURRENCY@Q},
        "generator_api_mode": ${GENERATOR_API_MODE@Q},
        "generator_temperature": ${GENERATOR_TEMPERATURE@Q},
        "generator_top_p": ${GENERATOR_TOP_P@Q},
        "generator_max_completion_tokens": ${GENERATOR_MAX_COMPLETION_TOKENS@Q},
        "generator_actor_rationale_max_completion_tokens": ${GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS@Q},
        "generator_reasoning_effort": ${GENERATOR_REASONING_EFFORT@Q},
        "actor_sampling_strategy": ${ACTOR_SAMPLING_STRATEGY@Q},
        "selection_policy": ${SELECTION_POLICY@Q},
        "selection_score_epsilon": ${SELECTION_SCORE_EPSILON@Q},
        "pair_mining_strategy": ${PAIR_MINING_STRATEGY@Q},
        "tool_coverage_retry_count": ${TOOL_COVERAGE_RETRY_COUNT@Q},
        "recovery_rwr_top_k": ${RECOVERY_RWR_TOP_K@Q},
        "mechanism_evidence_score_weights": {
            "specific_claim": {
                "grounding": 0.25,
                "consensus": 0.25,
                "specificity": 0.18,
                "network_agreement": 0.12,
                "stability": 0.10,
                "cross_tool_agreement": 0.10,
                "unsupported_claim_multiplier": 0.35
            },
            "abstention": {
                "abstention": 0.60,
                "network_agreement": 0.20,
                "grounding": 0.20
            },
            "evidence_strength": {
                "mygene_scale": 0.75,
                "network_scale": 0.50,
                "network_without_consensus_scale": 0.35,
                "unsupported_threshold": 0.05
            }
        },
        "mygene_cache_path": ${MYGENE_CACHE_PATH@Q},
        "enrichment_cache_path": ${ENRICHMENT_CACHE_PATH@Q},
        "enrichment_background_path": ${ENRICHMENT_BACKGROUND_PATH@Q},
        "allow_network_mygene": ${ALLOW_NETWORK_MYGENE@Q},
        "allow_network_enrichment": ${ALLOW_NETWORK_ENRICHMENT@Q},
        "generator_timeout_seconds": ${GENERATOR_TIMEOUT_SECONDS@Q},
    },
    "smoke": {
        "run_smoke_test": ${RUN_SMOKE_TEST@Q},
        "smoke_test_only": ${SMOKE_TEST_ONLY@Q},
        "smoke_task_index": ${SMOKE_TASK_INDEX@Q},
        "smoke_task_id": ${SMOKE_TASK_ID@Q},
        "smoke_task_ids": ${SMOKE_TASK_IDS@Q},
        "smoke_n_act": ${SMOKE_N_ACT@Q},
        "smoke_seed": ${SMOKE_SEED@Q},
        "smoke_actor_sampling_strategy": ${SMOKE_ACTOR_SAMPLING_STRATEGY@Q},
    },
    "vllm": {
        "head_node": ${HEAD_NODE@Q},
        "head_ip": ${HEAD_IP@Q},
        "http_port": ${VLLM_HTTP_PORT@Q},
        "tensor_parallel_size": ${TENSOR_PARALLEL_SIZE@Q},
        "pipeline_parallel_size": ${PIPELINE_PARALLEL_SIZE@Q},
        "max_model_len": ${MAX_MODEL_LEN@Q},
        "max_num_batched_tokens": ${MAX_NUM_BATCHED_TOKENS@Q},
        "max_num_seqs": ${MAX_NUM_SEQS@Q},
        "gpu_memory_utilization": ${GPU_MEMORY_UTILIZATION@Q},
        "effective_distributed_executor_backend": ${EFFECTIVE_DISTRIBUTED_EXECUTOR_BACKEND@Q},
        "enable_chunked_prefill": ${ENABLE_CHUNKED_PREFILL@Q},
        "enable_prefix_caching": ${ENABLE_PREFIX_CACHING@Q},
    },
}

Path(${RUN_FREEZE_PATH@Q}).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
log "Wrote run freeze metadata: ${RUN_FREEZE_PATH}"

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
  SMOKE_TASK_DESCRIPTORS=()
  if [[ -n "${SMOKE_TASK_IDS}" ]]; then
    IFS=',' read -r -a REQUESTED_SMOKE_TASK_IDS <<< "${SMOKE_TASK_IDS}"
    for RAW_SMOKE_TASK_ID in "${REQUESTED_SMOKE_TASK_IDS[@]}"; do
      TRIMMED_SMOKE_TASK_ID="${RAW_SMOKE_TASK_ID#"${RAW_SMOKE_TASK_ID%%[![:space:]]*}"}"
      TRIMMED_SMOKE_TASK_ID="${TRIMMED_SMOKE_TASK_ID%"${TRIMMED_SMOKE_TASK_ID##*[![:space:]]}"}"
      if [[ -n "${TRIMMED_SMOKE_TASK_ID}" ]]; then
        SMOKE_TASK_DESCRIPTORS+=("id:${TRIMMED_SMOKE_TASK_ID}")
      fi
    done
  elif [[ -n "${SMOKE_TASK_ID}" ]]; then
    SMOKE_TASK_DESCRIPTORS+=("id:${SMOKE_TASK_ID}")
  else
    SMOKE_TASK_DESCRIPTORS+=("index:${SMOKE_TASK_INDEX}")
  fi

  for SMOKE_DESCRIPTOR in "${SMOKE_TASK_DESCRIPTORS[@]}"; do
    SMOKE_SELECTOR="${SMOKE_DESCRIPTOR%%:*}"
    SMOKE_VALUE="${SMOKE_DESCRIPTOR#*:}"
    SMOKE_TASK_FLAGS=()
    if [[ "${SMOKE_SELECTOR}" == "id" ]]; then
      SMOKE_TASK_FLAGS+=(--task-id "${SMOKE_VALUE}")
      log "Running preflight generator smoke test for task id: ${SMOKE_VALUE}"
    else
      SMOKE_TASK_FLAGS+=(--task-index "${SMOKE_VALUE}")
      log "Running preflight generator smoke test for task index: ${SMOKE_VALUE}"
    fi

    if ! srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
      bash --noprofile --norc -c "
        set -euo pipefail
        source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
        source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        export HIP_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}
        unset ROCR_VISIBLE_DEVICES || true
        cd '${REPO_ROOT}'
        export PYTHONPATH='${REPO_ROOT}':\${PYTHONPATH:-}
        python scripts/smoke_test_generator.py \
          --tasks-path '${TASKS_PATH}' \
          --store-dir '${STORE_DIR}' \
          --mygene-cache-path '${MYGENE_CACHE_PATH}' \
          --enrichment-cache-path '${ENRICHMENT_CACHE_PATH}' \
          ${MYGENE_NETWORK_ARG[*]@Q} \
          ${ENRICHMENT_NETWORK_ARG[*]@Q} \
          ${ENRICHMENT_BACKGROUND_ARG[*]@Q} \
          --generator-api-base 'http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1' \
          --generator-model '${SERVED_MODEL_ID}' \
          --generator-api-mode ${GENERATOR_API_MODE} \
          --n-act ${SMOKE_N_ACT} \
          --n-ver 1 \
          --seed ${SMOKE_SEED} \
          --generator-temperature ${SMOKE_GENERATOR_TEMPERATURE} \
          --generator-max-completion-tokens ${SMOKE_MAX_COMPLETION_TOKENS} \
          --generator-actor-rationale-max-completion-tokens ${SMOKE_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS} \
          --generator-reasoning-effort ${SMOKE_REASONING_EFFORT} \
          --actor-sampling-strategy ${SMOKE_ACTOR_SAMPLING_STRATEGY} \
          ${SMOKE_TASK_FLAGS[*]@Q}
      "; then
      log "Preflight generator smoke test failed. See the candidate dump above for the raw response and generator errors."
      exit 1
    fi
  done

  if [[ "${SMOKE_TEST_ONLY}" == "1" ]]; then
    log "Smoke test finished successfully; exiting because SMOKE_TEST_ONLY=1."
    exit 0
  fi
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
      --mygene-cache-path '${MYGENE_CACHE_PATH}' \
      --enrichment-cache-path '${ENRICHMENT_CACHE_PATH}' \
      ${MYGENE_NETWORK_ARG[*]@Q} \
      ${ENRICHMENT_NETWORK_ARG[*]@Q} \
      ${ENRICHMENT_BACKGROUND_ARG[*]@Q} \
      --candidate-source model_vllm \
      --generator-api-base 'http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1' \
      --generator-model '${SERVED_MODEL_ID}' \
      --generator-api-mode ${GENERATOR_API_MODE} \
      --generator-temperature ${GENERATOR_TEMPERATURE} \
      --generator-top-p ${GENERATOR_TOP_P} \
      --generator-max-completion-tokens ${GENERATOR_MAX_COMPLETION_TOKENS} \
      --generator-actor-rationale-max-completion-tokens ${GENERATOR_ACTOR_RATIONALE_MAX_COMPLETION_TOKENS} \
      --generator-reasoning-effort ${GENERATOR_REASONING_EFFORT} \
      --actor-sampling-strategy ${ACTOR_SAMPLING_STRATEGY} \
      --selection-policy ${SELECTION_POLICY} \
      --selection-score-epsilon ${SELECTION_SCORE_EPSILON} \
      --pair-mining-strategy ${PAIR_MINING_STRATEGY} \
      --tool-coverage-retry-count ${TOOL_COVERAGE_RETRY_COUNT} \
      --recovery-rwr-top-k ${RECOVERY_RWR_TOP_K} \
      --generator-timeout-seconds ${GENERATOR_TIMEOUT_SECONDS} \
      ${MAX_TASKS_ARG[*]@Q} \
      --task-shard-index ${TASK_SHARD_INDEX} \
      --task-shard-count ${TASK_SHARD_COUNT} \
      --max-steps ${MAX_STEPS} \
      --n-act ${N_ACT} \
      --n-ver ${N_VER} \
      --task-concurrency ${TASK_CONCURRENCY}
  "

log "Trajectory generation finished: ${OUT_DIR}"
