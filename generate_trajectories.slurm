#!/bin/bash
#SBATCH --job-name=mentor-traj-gen
#SBATCH -A SYB114
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

module load craype-accel-amd-gfx90a

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

REPO_ROOT="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl"
MODEL_PATH="${MODEL_PATH:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/gpt-oss-120b-bf16}"
TASKS_PATH="${TASKS_PATH:-${REPO_ROOT}/data/corum_corpus/tasks.train.jsonl}"
STORE_DIR="${STORE_DIR:-${REPO_ROOT}/data/humannet_multiplex_store}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/corum_trajectories/gpt_oss_120b_job_${SLURM_JOB_ID}}"

MAX_TASKS="${MAX_TASKS:-8}"
MAX_STEPS="${MAX_STEPS:-4}"
N_ACT="${N_ACT:-4}"
N_VER="${N_VER:-2}"
GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.8}"
GENERATOR_TOP_P="${GENERATOR_TOP_P:-0.95}"
GENERATOR_MAX_COMPLETION_TOKENS="${GENERATOR_MAX_COMPLETION_TOKENS:-700}"

RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-51001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-${SLURM_NNODES}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUT_DIR}"

mapfile -t NODE_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${NODE_LIST[0]}"

resolve_hsn_ip() {
  local node_name="$1"
  getent hosts "${node_name}-hsn0" | awk '{print $1}'
}

HEAD_IP="$(resolve_hsn_ip "${HEAD_NODE}")"
if [[ -z "${HEAD_IP}" ]]; then
  echo "Failed to resolve head node HSN IP for ${HEAD_NODE}"
  exit 1
fi

echo "HEAD_NODE=${HEAD_NODE}"
echo "HEAD_IP=${HEAD_IP}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASKS_PATH=${TASKS_PATH}"
echo "OUT_DIR=${OUT_DIR}"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
  jobs -pr | xargs -r kill || true
}
trap cleanup EXIT

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
export ROCR_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}"
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

echo "Starting Ray head on ${HEAD_NODE}"
srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
    source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
    ulimit -n 65536
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
    echo "Failed to resolve worker node HSN IP for ${WORKER_NODE}"
    exit 1
  fi
  echo "Starting Ray worker on ${WORKER_NODE}"
  srun --overlap -N1 -n1 -w "${WORKER_NODE}" --export=ALL \
    bash --noprofile --norc -c "
      set -euo pipefail
      source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
      source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
      ulimit -n 65536
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

echo "Starting vLLM on ${HEAD_NODE}"
srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
    source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
    export VLLM_HOST_IP=${HEAD_IP}
    vllm serve '${MODEL_PATH}' \
      --host 0.0.0.0 \
      --port ${VLLM_HTTP_PORT} \
      --distributed-executor-backend ray \
      --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
      --pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
      --dtype bfloat16 \
      --tokenizer '${MODEL_PATH}' \
      --max-model-len ${MAX_MODEL_LEN} \
      --disable-custom-all-reduce \
      --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
  " &

echo "Waiting for vLLM to become healthy at http://${HEAD_IP}:${VLLM_HTTP_PORT}"
for attempt in $(seq 1 180); do
  if curl -sf "http://${HEAD_IP}:${VLLM_HTTP_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

curl -sf "http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1/models" | tee "${OUT_DIR}/served_models.json"

echo "Launching trajectory generation"
srun --overlap -N1 -n1 -w "${HEAD_NODE}" --export=ALL \
  bash --noprofile --norc -c "
    set -euo pipefail
    source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
    source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
    cd '${REPO_ROOT}'
    python scripts/generate_trajectories.py \
      --tasks-path '${TASKS_PATH}' \
      --out-dir '${OUT_DIR}' \
      --store-dir '${STORE_DIR}' \
      --candidate-source model_vllm \
      --generator-api-base 'http://${HEAD_IP}:${VLLM_HTTP_PORT}/v1' \
      --generator-temperature ${GENERATOR_TEMPERATURE} \
      --generator-top-p ${GENERATOR_TOP_P} \
      --generator-max-completion-tokens ${GENERATOR_MAX_COMPLETION_TOKENS} \
      --max-tasks ${MAX_TASKS} \
      --max-steps ${MAX_STEPS} \
      --n-act ${N_ACT} \
      --n-ver ${N_VER}
  "

echo "Trajectory generation finished: ${OUT_DIR}"
