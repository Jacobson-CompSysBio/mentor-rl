#!/bin/bash
# Train the pinned S2 corpus from its verified S1 parent on Frontier.
# CONTRACT_ONLY=1 validates inputs without Slurm or GPU work.
# PREPARE_DATA_ONLY=1 builds the shared token cache without Slurm or GPUs.
# Use scripts/submit_world_model_s2.sh to seal the config hash before queueing.
#SBATCH --job-name=s2_v1_184n
#SBATCH -A SYB114
#SBATCH -N 184
#SBATCH -t 10:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail

die() { echo "[ERROR] $*" >&2; exit 2; }
require_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer."
}
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
[[ -f world_model_s2.slurm ]] || die "Run this launcher from the repository root."
export REPO_ROOT PYTHONDONTWRITEBYTECODE=1
S2_PYTHON="${S2_PYTHON:-/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python}"
[[ -x "${S2_PYTHON}" ]] || die "The pinned Python environment is absent."
RUN_CONFIG="${RUN_CONFIG:-config/world_model_s2_120b_training_v1_184n.json}"
CONTRACT_ONLY="${CONTRACT_ONLY:-0}"
PREPARE_DATA_ONLY="${PREPARE_DATA_ONLY:-0}"
S2_RESUME_CHECKPOINT="${S2_RESUME_CHECKPOINT:-}"
[[ "${CONTRACT_ONLY}" =~ ^[01]$ && "${PREPARE_DATA_ONLY}" =~ ^[01]$ ]] || \
  die "CONTRACT_ONLY and PREPARE_DATA_ONLY must be zero or one."
[[ "${CONTRACT_ONLY}${PREPARE_DATA_ONLY}" != "11" ]] || \
  die "Select only one CPU mode."
[[ -f "${RUN_CONFIG}" ]] || die "The run config is absent."
actual_config_sha256="$(sha256sum "${RUN_CONFIG}" | awk '{print $1}')"
if [[ -n "${EXPECTED_RUN_CONFIG_SHA256:-}" ]]; then
  [[ "${actual_config_sha256}" == "${EXPECTED_RUN_CONFIG_SHA256}" ]] || \
    die "The queued run config identity changed."
elif [[ "${CONTRACT_ONLY}" == 0 && "${PREPARE_DATA_ONLY}" == 0 ]]; then
  die "EXPECTED_RUN_CONFIG_SHA256 is required; use the submission helper."
fi
CONTRACT_CHECKER="${REPO_ROOT}/scripts/validate_world_model_s2_train_contract.py"
contract_env="$(mktemp "${TMPDIR:-/tmp}/mentor-s2-contract.XXXXXX")"
trap 'rm -f -- "${contract_env}"' EXIT
"${S2_PYTHON}" "${CONTRACT_CHECKER}" "${REPO_ROOT}" "${RUN_CONFIG}" > "${contract_env}"
set -a
# shellcheck disable=SC1090
source "${contract_env}"
set +a
[[ "${S0_RUN_CONFIG_SHA256}" == "${actual_config_sha256}" ]] || die "The config changed during validation."
[[ "${REQUESTED_PARTITION}" == batch && "${REQUESTED_WALLTIME}" == 10:00:00 ]] || \
  die "This launcher requires the agreed batch partition and ten-hour request."
[[ "${NUM_NODES}" == 184 && "${RANKS_PER_NODE}" == 8 && "${WORLD_SIZE_EXPECTED}" == 1472 ]] || \
  die "This launcher requires 184 nodes with eight ranks per node."

# Keep the expensive CPU tokenization cache shared across preparation and GPUs.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROCM_PATH=/opt/rocm-6.3.1 ROCM_HOME=/opt/rocm-6.3.1
export HF_DATASETS_CACHE="${REPO_ROOT}/data/world_model_v4/cache/s2_training_v1_184n/arrow"
export S2_DATA_PREPARATION_DIR="${REPO_ROOT}/data/world_model_v4/cache/s2_training_v1_184n/preparation"
resolve_resume_identity() {
  "${S2_PYTHON}" - "${S2_RESUME_CHECKPOINT}" "${OUTPUT_ROOT}" "${S0_RUN_CONFIG_SHA256}" > "${contract_env}" <<'PYRESUME'
import json
from pathlib import Path
import re
import shlex
import sys
checkpoint=Path(sys.argv[1]).resolve()
output=checkpoint.parent
if not re.fullmatch(r"checkpoint-[0-9]+", checkpoint.name) or output.parent != Path(sys.argv[2]).resolve():
    raise SystemExit("Resume must name a rolling checkpoint under this run's output root")
for name in ("CHECKPOINT_COMPLETE", "training_state.json", "tp_adapter_manifest.json", "scheduler.pt"):
    if not (checkpoint/name).is_file():
        raise SystemExit(f"Incomplete resume checkpoint: {name}")
manifest=json.loads((checkpoint/"tp_adapter_manifest.json").read_text())
if manifest.get("topology",{}).get("world_size") != 1472:
    raise SystemExit("Resume requires the original 184-node topology")
identity=manifest.get("identity",{})
if identity.get("run_config_sha256") != sys.argv[3]:
    raise SystemExit("Resume config identity differs from the pinned run")
if not identity.get("run_id"):
    raise SystemExit("Resume checkpoint has no original run ID")
for key,value in {"OUTPUT_DIR":str(output),"S0_RUN_ID":identity["run_id"],"S2_RESUME_CHECKPOINT":str(checkpoint)}.items():
    print(f"{key}={shlex.quote(value)}")
PYRESUME
    # shellcheck disable=SC1090
    source "${contract_env}"
}

if [[ "${CONTRACT_ONLY}" == 1 ]]; then
  [[ -z "${S2_RESUME_CHECKPOINT}" ]] || resolve_resume_identity
  echo "S2 contract valid: 184 nodes, batch, 10:00:00, batch 184, 1371 updates."
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Scheduled CPU contract check complete. No GPU training ran."
  else
    echo "No GPU work or Slurm job ran."
  fi
  exit 0
fi

execution_id="${SLURM_JOB_ID:-CPU-PREPARE}"
S0_RUN_ID="${S0_METHOD_ID}-slurm-${execution_id}"
OUTPUT_DIR="${OUTPUT_ROOT}/slurm-${execution_id}"
if [[ "${PREPARE_DATA_ONLY}" == 1 ]]; then
  [[ -z "${S2_RESUME_CHECKPOINT}" ]] || die "CPU cache preparation does not resume a checkpoint."
  OUTPUT_DIR="${S2_DATA_PREPARATION_DIR}"
else
  require_positive_integer SLURM_JOB_ID "${SLURM_JOB_ID:-}"
  [[ "${SLURM_NNODES:-}" == "${NUM_NODES}" && "${SLURM_NTASKS:-}" == "${WORLD_SIZE_EXPECTED}" ]] || \
    die "The allocation node/rank count differs from the pinned contract."
  allocated_local_ranks="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}"
  allocated_local_ranks="${allocated_local_ranks%%(*}"
  [[ "${allocated_local_ranks}" == "${RANKS_PER_NODE}" ]] || die "The local rank count differs from the contract."
  [[ "${SLURM_JOB_PARTITION:-}" == batch ]] || die "The allocation must use batch."
  [[ -s "${S2_DATA_PREPARATION_DIR}/data_preparation.json" ]] || \
    die "Prepare the token cache first with scripts/submit_world_model_s2.sh --prepare."
  "${S2_PYTHON}" - "${S2_DATA_PREPARATION_DIR}/data_preparation.json" "${S0_RUN_CONFIG_SHA256}" <<'PYCACHE'
import json
from pathlib import Path
import sys
receipt=json.loads(Path(sys.argv[1]).read_text())
if (receipt.get('status')!='complete' or receipt.get('run_config_sha256')!=sys.argv[2]
        or not receipt.get('cache_files') or any(not Path(p).is_file() for p in receipt['cache_files'])):
    raise SystemExit('Prepared S2 cache is stale or incomplete; run --prepare again')
PYCACHE
  if [[ -n "${S2_RESUME_CHECKPOINT}" ]]; then
    # Reuse the original output and W&B identity; full state is topology-bound.
    resolve_resume_identity
  else
    [[ ! -e "${OUTPUT_DIR}" ]] || die "The new run output already exists."
  fi
fi
export S0_RUN_ID OUTPUT_DIR
TRAIN_MODEL_PATH="${MODEL_PATH}"

# Build the same arguments for CPU cache preparation and distributed training.
build_train_args() {
  train_args=(
    --model_path "${TRAIN_MODEL_PATH}" --tokenizer_path "${TOKENIZER_PATH}"
    --dataset_path "${S0_TRAIN_PATH}" --validation_dataset_path "${S0_VALIDATION_PATH}"
    --corpus_manifest_path "${S2_CORPUS_MANIFEST_PATH}"
    --output_dir "${OUTPUT_DIR}" --num_train_epochs "${TRAIN_NUM_EPOCHS}"
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
    --learning_rate "${LEARNING_RATE}" --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
    --warmup_ratio "${WARMUP_RATIO}" --loss_contract "${LOSS_CONTRACT}"
    --weight_decay "${WEIGHT_DECAY}" --max_grad_norm "${MAX_GRAD_NORM}"
    --logging_steps "${LOGGING_STEPS}" --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
    --save_strategy "${SAVE_STRATEGY}" --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}" --eval_strategy "${EVAL_STRATEGY}"
    --eval_steps "${EVAL_STEPS}" --max_length "${MAX_LENGTH}" --seed "${SEED}"
    --tp_size "${TP_SIZE}" --ep_size "${EP_SIZE}"
    --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}"
    --init_adapter_dir "${INIT_ADAPTER_DIR}" --init_adapter_sha256 "${INIT_ADAPTER_SHA256}"
    --init_adapter_run_id "${INIT_ADAPTER_RUN_ID}"
    --diagnostic_snapshot_steps "${DIAGNOSTIC_SNAPSHOT_STEPS}"
    --throughput_calibration_updates "${THROUGHPUT_CALIBRATION_UPDATES}"
    --expected_optimizer_updates "${TRAIN_TOTAL_STEPS}"
    --expected_data_parallel_size "${DATA_PARALLEL_SIZE}"
  )
  [[ "${SEGMENT_LOSS_TELEMETRY}" != 1 ]] || train_args+=(--segment_loss_telemetry)
  [[ "${EVAL_ON_START}" != 1 ]] || train_args+=(--eval_on_start)
  [[ "${PRESERVE_DATASET_ORDER}" != 0 ]] || train_args+=(--no-preserve_dataset_order)
  [[ "${AUTOGRAD_MULTITHREADING}" != 1 ]] || train_args+=(--autograd-multithreading)
  [[ "${LOCAL_FILES_ONLY}" != 1 ]] || train_args+=(--local_files_only)
  [[ "${STRICT_TESTED_STACK}" != 1 ]] || train_args+=(--strict_tested_stack)
  [[ "${BF16}" != 1 ]] || train_args+=(--bf16)
  [[ "${GPU_PREFLIGHT}" != 1 ]] || train_args+=(--gpu_preflight)
  [[ -z "${S2_RESUME_CHECKPOINT}" ]] || train_args+=(--resume_from_checkpoint "${S2_RESUME_CHECKPOINT}")
}
if [[ "${PREPARE_DATA_ONLY}" == 1 ]]; then
  module load rocm/6.3.1
  export ROCM_PATH=/opt/rocm-6.3.1 ROCM_HOME=/opt/rocm-6.3.1 OMP_NUM_THREADS=1
  mkdir -p "${HF_DATASETS_CACHE}" "${OUTPUT_DIR}"
  build_train_args
  "${S2_PYTHON}" "${TRAINER_PATH}" "${train_args[@]}" --prepare-data-only
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Scheduled S2 CPU token cache preparation complete. No GPU training ran."
  else
    echo "S2 shared CPU token cache prepared. No Slurm job ran."
  fi
  exit 0
fi

mkdir -p "${OUTPUT_DIR}/run_contract" "${OUTPUT_DIR}/wandb"
RUN_CONTRACT_DIR="${OUTPUT_DIR}/run_contract"
export RUN_CONTRACT_DIR
launch_receipt_dir="${RUN_CONTRACT_DIR}"
if [[ -n "${S2_RESUME_CHECKPOINT}" ]]; then
  launch_receipt_dir="${RUN_CONTRACT_DIR}/continuations/slurm-${SLURM_JOB_ID}"
  mkdir -p "${launch_receipt_dir}"
fi
cp "${S0_RUN_CONFIG}" "${launch_receipt_dir}/run_config.json"
printf '%s\n' "job_id=${SLURM_JOB_ID}" "run_id=${S0_RUN_ID}" \
  "run_config_sha256=${S0_RUN_CONFIG_SHA256}" "training_code_sha256=${S0_TRAINING_CODE_SHA256}" \
  "nodes=184" "ranks_per_node=8" "world_size=1472" "partition=batch" "walltime=10:00:00" \
  "global_batch_size=184" "total_steps=1371" "resume_checkpoint=${S2_RESUME_CHECKPOINT}" \
  > "${launch_receipt_dir}/launch.env"
if [[ -n "${S2_SUBMISSION_RECEIPT:-}" ]]; then
  [[ -s "${S2_SUBMISSION_RECEIPT}" ]] || die "The submission receipt is absent."
  cp "${S2_SUBMISSION_RECEIPT}" "${launch_receipt_dir}/submission_preparation.json"
fi
BASE_MODEL_VALIDATOR="${CONTRACT_CHECKER}"
S0_PREFLIGHT_ATTEMPTS=3
export BASE_MODEL_VALIDATOR

# Load the pinned Frontier software stack.
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

export ROCM_SMI_DISABLE=1 AMD_SMI_DISABLE=1 RSMI_LOGGING=0
export AMDSMI_GPU_METRICS_CACHE_MS=0
ulimit -c 0

# Remove only ROCm-SMI state that the current user owns.
echo "Clear user-owned ROCm-SMI state before the first torch import."
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  --cpu-bind=none bash -c '
  set -euo pipefail
  shopt -s nullglob
  removed=0
  skipped=0
  for path in /dev/shm/rocm_smi* /dev/shm/amd_smi*; do
    if [[ ! -O "${path}" ]]; then
      ((skipped += 1))
      continue
    fi
    if rm -f -- "${path}"; then
      ((removed += 1))
    else
      echo "[$(hostname)] Could not remove owned ROCm-SMI state: ${path}" >&2
    fi
  done
  echo "[$(hostname)] ROCm-SMI state removed=${removed} skipped=${skipped}"
'

# Load W&B credentials after the ROCm-SMI cleanup.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi
# Set the Frontier network, cache, and train environment.
export CC=cc CXX=CC CMAKE_C_COMPILER=cc CMAKE_CXX_COMPILER=CC
export ROCM_PATH=/opt/rocm-6.3.1 ROCM_HOME=/opt/rocm-6.3.1 PYTORCH_ROCM_ARCH=gfx90a HSA_XNACK=0
export DS_ACCELERATOR=cuda PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export AWS_OFI_RCCL_ROOT="${AWS_OFI_RCCL_ROOT:-/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl}"
[[ -d "${AWS_OFI_RCCL_ROOT}/lib" ]] || die "aws-ofi-rccl is absent."
export LD_LIBRARY_PATH="${AWS_OFI_RCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export FI_PROVIDER=cxi FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
export FI_CXI_DEFAULT_TX_SIZE="${FI_CXI_DEFAULT_TX_SIZE:-2048}"
export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-hybrid}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_SOCKET_IFNAME=hsn0 NCCL_OFI_USE_NICLIST=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=hsn0 GLOO_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_MOE_LAYER_CLS_NAMES=GptOssMLP
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
export WANDB_MODE="${S0_TRACKING_MODE}"
# The run contract owns the W&B workspace. Credentials only authenticate the user.
export WANDB_ENTITY="${S0_WANDB_ENTITY}"
export WANDB_PROJECT="mentor-sft"
export WANDB_RUN_GROUP="${S0_CONFIG_ID}"
export WANDB_NAME="${S0_RUN_ID}"
export WANDB_RUN_ID="${S0_RUN_ID}"
export WANDB_RESUME=allow WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-180}"
export http_proxy="${http_proxy:-http://proxy.ccs.ornl.gov:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"

# Select the head node and one deterministic first port.
mapfile -t NODELIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#NODELIST[@]}" -eq "${NUM_NODES}" ]] || \
  die "The allocated node list is incomplete."
HEAD_NODE="${NODELIST[0]}"
MASTER_ADDR="$(getent hosts "${HEAD_NODE}-hsn0" | awk 'NR == 1 {print $1}')"
if [[ -z "${MASTER_ADDR}" ]]; then
  MASTER_ADDR="$(
    srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" hostname -I | awk '{print $1}'
  )"
fi
[[ -n "${MASTER_ADDR}" ]] || die "The distributed master address is absent."
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MASTER_ADDR MASTER_PORT
export MENTOR_DIST_INIT_TIMEOUT_SECONDS="${MENTOR_DIST_INIT_TIMEOUT_SECONDS:-600}"
export MENTOR_DISTRIBUTED_READY_MARKER="${RUN_CONTRACT_DIR}/DISTRIBUTED_READY"

# Keep each verified base model in the user burst buffer.
LOCAL_MODEL_CACHE_ROOT="${LOCAL_MODEL_CACHE_ROOT:-/mnt/bb/${USER}/mentor_model_cache}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_CACHE_ROOT}/${MODEL_ID}-${BASE_MODEL_ARTIFACT_SHA256:0:16}"
[[ "${LOCAL_MODEL_CACHE_ROOT}" == "/mnt/bb/${USER}/"* ]] || \
  die "The local model root must stay below the user burst buffer."
[[ "${LOCAL_MODEL_DIR}" == "${LOCAL_MODEL_CACHE_ROOT}/"* ]] || \
  die "The local model path must stay below its root."
export LOCAL_MODEL_CACHE_ROOT LOCAL_MODEL_DIR MODEL_PATH MODEL_ID

# Stage one model copy on each node in bounded waves.
stage_model_wave() {
  local first="$1"
  local count="$2"
  local hosts
  hosts="$(IFS=,; echo "${NODELIST[*]:first:count}")"
  echo "Stage the model on nodes $((first + 1))-$((first + count))."
  srun --overlap --nodes="${count}" --ntasks="${count}" \
    --ntasks-per-node=1 -w "${hosts}" --cpu-bind=none bash -c '
    set -euo pipefail
    marker="${LOCAL_MODEL_DIR}/.mentor_model_identity_sha256"
    receipt="${LOCAL_MODEL_DIR}/.mentor_base_model_artifact.json"
    mkdir -p "${LOCAL_MODEL_CACHE_ROOT}"
    exec 9>"${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.stage.lock"
    flock 9
    # Reuse a cache only after its full artifact check passes.
    if [[ -e "${LOCAL_MODEL_DIR}" ]]; then
      [[ -s "${marker}" ]] || {
        echo "[ERROR] The model cache has no identity marker." >&2
        exit 2
      }
      [[ "$(<"${marker}")" == "${BASE_MODEL_ARTIFACT_SHA256}" ]] || {
        echo "[ERROR] The model cache marker changed." >&2
        exit 2
      }
      [[ -s "${LOCAL_MODEL_DIR}/config.json" ]] || {
        echo "[ERROR] The model cache has no config.json." >&2
        exit 2
      }
      python "${BASE_MODEL_VALIDATOR}" --validate-base-model \
        "${LOCAL_MODEL_DIR}" \
        "${BASE_MODEL_ARTIFACT_SHA256}" \
        "${BASE_MODEL_WEIGHT_SHARD_COUNT}" "${receipt}"
      echo "[$(hostname)] Use the verified model cache."
      exit 0
    fi
    # Copy into a temporary path before the final atomic move.
    staging="$(
      mktemp -d \
        "${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.stage.${SLURM_JOB_ID}.XXXXXX"
    )"
    cleanup_staging() {
      [[ -z "${staging:-}" || ! -e "${staging}" ]] || rm -rf -- "${staging}"
    }
    trap cleanup_staging EXIT
    rsync -a "${MODEL_PATH}/" "${staging}/"
    python "${BASE_MODEL_VALIDATOR}" --validate-base-model "${staging}" \
      "${BASE_MODEL_ARTIFACT_SHA256}" \
      "${BASE_MODEL_WEIGHT_SHARD_COUNT}" \
      "${staging}/.mentor_base_model_artifact.json"
    printf "%s\n" "${BASE_MODEL_ARTIFACT_SHA256}" \
      > "${staging}/.mentor_model_identity_sha256"
    mv "${staging}" "${LOCAL_MODEL_DIR}"
    staging=""
    echo "[$(hostname)] The model cache is ready."
  '
}

# Stage the production allocation in one wave. Four-node waves made cold
# staging consume most of the ten-hour allocation; prior 184-node S1 runs
# established that allocation-wide staging is viable on Frontier.
MODEL_STAGE_WAVE_SIZE="${MODEL_STAGE_WAVE_SIZE:-${NUM_NODES}}"
require_positive_integer MODEL_STAGE_WAVE_SIZE "${MODEL_STAGE_WAVE_SIZE}"
(( MODEL_STAGE_WAVE_SIZE <= NUM_NODES )) || \
  die "MODEL_STAGE_WAVE_SIZE exceeds the node count."
for ((first = 0; first < NUM_NODES; first += MODEL_STAGE_WAVE_SIZE)); do
  count="${MODEL_STAGE_WAVE_SIZE}"
  remaining="$((NUM_NODES - first))"
  (( remaining >= count )) || count="${remaining}"
  stage_model_wave "${first}" "${count}"
done
TRAIN_MODEL_PATH="${LOCAL_MODEL_DIR}"
export TRAIN_MODEL_PATH

# Use separate caches for each job and rank.
export HF_HOME="/mnt/bb/${USER}/mentor_hf/${SLURM_JOB_ID}"

export TORCH_EXTENSIONS_DIR="/mnt/bb/${USER}/mentor_torch_extensions/torch280_rocm631"
export TRITON_CACHE_DIR="/mnt/bb/${USER}/mentor_triton/${SLURM_JOB_ID}"

# Import torch once on each node after the cleanup.
echo "Check the Python and ROCm stack on each node."
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  --gpus-per-task=1 --cpu-bind=none --gpu-bind=closest \
  python -c 'import socket, torch, transformers, peft, accelerate, datasets, trl; assert torch.cuda.is_available(); print("python_rocm_ready host={} torch={} gpu_count={}".format(socket.gethostname(), torch.__version__, torch.cuda.device_count()), flush=True)'

# Write and check one receipt for every allocated rank.
rank_receipt_dir="${RUN_CONTRACT_DIR}/rank_preflight"
mkdir -p "${rank_receipt_dir}"
export rank_receipt_dir
echo "Check all allocated ranks."
run_rank_preflight() {
  srun --nodes="${NUM_NODES}" --ntasks="${WORLD_SIZE_EXPECTED}" \
    --ntasks-per-node="${RANKS_PER_NODE}" --gpus-per-task=1 \
    --cpu-bind=none --gpu-bind=closest --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    gpu_count="$(python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())")"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${SLURM_PROCID}" "${SLURM_LOCALID}" "$(hostname)" "${gpu_count}" \
      "${ROCR_VISIBLE_DEVICES:-<unset>}" \
      "${HIP_VISIBLE_DEVICES:-<unset>}" \
      "${CUDA_VISIBLE_DEVICES:-<unset>}" \
      > "${rank_receipt_dir}/rank-${SLURM_PROCID}.tsv"
  '
}

# A Slurm controller timeout can reject a step before any rank starts. Retry
# this read-only check and discard partial receipts between attempts.
preflight_status=1
for ((preflight_attempt = 1; \
      preflight_attempt <= S0_PREFLIGHT_ATTEMPTS; \
      preflight_attempt++)); do
  rm -f -- "${rank_receipt_dir}"/rank-*.tsv
  if run_rank_preflight; then
    preflight_status=0
    break
  else
    preflight_status=$?
  fi
  if (( preflight_attempt < S0_PREFLIGHT_ATTEMPTS )); then
    echo "[WARN] Rank preflight attempt ${preflight_attempt} failed with status ${preflight_status}; retrying." >&2
    sleep "$((5 * preflight_attempt))"
  fi
done
[[ "${preflight_status}" -eq 0 ]] || \
  die "The rank preflight failed with status ${preflight_status}."
python - "${rank_receipt_dir}" "${WORLD_SIZE_EXPECTED}" \
  "${RANKS_PER_NODE}" "${NUM_NODES}" <<'PY'
from collections import defaultdict
from pathlib import Path
import sys

root = Path(sys.argv[1])
world_size = int(sys.argv[2])
local_world_size = int(sys.argv[3])
node_count = int(sys.argv[4])
receipts = []
for path in root.glob("rank-*.tsv"):
    fields = path.read_text(encoding="utf-8").strip().split("\t")
    if len(fields) != 7:
        raise SystemExit("A rank GPU receipt is invalid")
    rank, local_rank, host, gpu_count, rocr, hip, cuda = fields
    receipts.append(
        (int(rank), int(local_rank), host, int(gpu_count), rocr, hip, cuda)
    )
if sorted(rank for rank, *_ in receipts) != list(range(world_size)):
    raise SystemExit("The rank receipt set is incomplete or duplicated")
by_host = defaultdict(set)
by_host_gpu = defaultdict(set)
by_host_device_counts = defaultdict(set)
for _, local_rank, host, gpu_count, rocr, _, _ in receipts:
    if gpu_count not in {1, local_world_size}:
        raise SystemExit(
            "Each train rank must see one GPU or all local GPUs"
        )
    by_host[host].add(local_rank)
    by_host_gpu[host].add(rocr)
    by_host_device_counts[host].add(gpu_count)
if len(by_host) != node_count:
    raise SystemExit("The rank receipts have the wrong node count")
expected_local = set(range(local_world_size))
if any(local_ranks != expected_local for local_ranks in by_host.values()):
    raise SystemExit("The local rank receipt set is incomplete or duplicated")
if any(len(counts) != 1 for counts in by_host_device_counts.values()):
    raise SystemExit("The visible GPU count differs across local ranks")
for host, counts in by_host_device_counts.items():
    if next(iter(counts)) == 1 and (
        "<unset>" in by_host_gpu[host]
        or len(by_host_gpu[host]) != local_world_size
    ):
        raise SystemExit(
            "The one-GPU rank bindings are incomplete or duplicated"
        )
PY


build_train_args

# Launch one process for each selected rank.
launch_training() {
  srun --nodes="${NUM_NODES}" --ntasks="${WORLD_SIZE_EXPECTED}" \
    --ntasks-per-node="${RANKS_PER_NODE}" --gpus-per-task=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-7}" --cpu-bind=none \
    --gpu-bind=closest --kill-on-bad-exit=1 bash -c '
    set -euo pipefail
    trainer="$1"
    shift
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export LOCAL_RANK="${SLURM_LOCALID}"
    export LOCAL_WORLD_SIZE="${RANKS_PER_NODE}"
    [[ "${WORLD_SIZE}" == "${WORLD_SIZE_EXPECTED}" ]] || {
      echo "[ERROR] The world rank count changed." >&2
      exit 2
    }
    [[ "${LOCAL_WORLD_SIZE}" == "${RANKS_PER_NODE}" ]] || {
      echo "[ERROR] The local rank count changed." >&2
      exit 2
    }
    export TP_GROUP_INDEX="$((RANK / TP_SIZE))"
    export TP_GROUP_RANK="$((RANK % TP_SIZE))"
    export EP_GROUP_INDEX="${TP_GROUP_INDEX}"
    export EP_GROUP_RANK="${TP_GROUP_RANK}"
    export HF_HOME="${HF_HOME}/rank_${RANK}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank_${RANK}"
    mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}"
    if [[ "${RANK}" != "0" ]]; then
      export WANDB_MODE=disabled
    fi
    echo "rank_start rank=${RANK} local_rank=${LOCAL_RANK} host=$(hostname) attempt=${MENTOR_LAUNCH_ATTEMPT}" >&2
    python "${trainer}" "$@"
  ' bash "${TRAINER_PATH}" "${train_args[@]}"
}

# Retry once on a new port only before distributed readiness.
echo "Start ${S0_METHOD_ID}."
train_status=1
for launch_attempt in 1 2; do
  export MENTOR_LAUNCH_ATTEMPT="${launch_attempt}"
  if launch_training; then
    train_status=0
    break
  else
    train_status=$?
  fi
  if [[ -e "${MENTOR_DISTRIBUTED_READY_MARKER}" || \
        "${launch_attempt}" -eq 2 ]]; then
    break
  fi
  MASTER_PORT="$((MASTER_PORT + 1))"
  export MASTER_PORT
  echo "[WARN] The distributed start failed before readiness. Retry once on port ${MASTER_PORT}." >&2
done
[[ "${train_status}" -eq 0 ]] || \
  die "The train launch failed with status ${train_status}."

# Require the final model files and the completed exposure receipt.
completed_files=(
  "${OUTPUT_DIR}/adapter_config.json"
  "${OUTPUT_DIR}/adapter_model.safetensors"
  "${OUTPUT_DIR}/tp_adapter_manifest.json"
  "${OUTPUT_DIR}/validation_metrics.jsonl"
  "${RUN_CONTRACT_DIR}/training_exposure.json"
)
for completed in "${completed_files[@]}"; do
  [[ -s "${completed}" ]] || \
    die "The completed output lacks ${completed}."
done

python - "${RUN_CONTRACT_DIR}/training_exposure.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("status") != "complete":
    raise SystemExit("The train exposure receipt is not complete")
PY
echo "The S2 train run is complete: ${OUTPUT_DIR}"
