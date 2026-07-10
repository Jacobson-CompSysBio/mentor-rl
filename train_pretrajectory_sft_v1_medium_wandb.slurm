#!/bin/bash
#SBATCH --job-name=pretraj_sft_train
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate
#SBATCH --signal=B:USR1@600

# Defaults to data/pretrajectory_sft/v1_medium. Swap datasets with:
# sbatch --export=ALL,PRETRAJ_DATASET_NAME=v2_250k,PRETRAJ_RUN_SUFFIX=v2_250k train_pretrajectory_sft_v1_medium_wandb.slurm
# Add TRAIN_MAX_STEPS=<n> for bounded smoke runs without changing the launcher.
# For a 120B multi-node smoke, override Slurm resources on the sbatch command line, e.g.:
# sbatch -N 4 -p extended -t 06:00:00 --export=ALL,MODEL_NAME=gpt-oss-120b-bf16,PRETRAJ_DATASET_NAME=v3_250k,PRETRAJ_RUN_SUFFIX=v3_250k_120b_smoke,TRAIN_MAX_STEPS=100,RUN_HOLDOUT_AFTER_TRAINING=False train_pretrajectory_sft_v1_medium_wandb.slurm
# For staged v4 curriculum training, set PRETRAJ_CURRICULUM_STAGE to one of:
# stage1_entity_schema, stage2_topology_priors, stage3_open_book_vectors,
# stage4_module_world_model, stage5_structured_tools, stage6_blend.
# Example:
# sbatch -N 4 -p extended -t 06:00:00 --export=ALL,MODEL_NAME=gpt-oss-120b-bf16,PRETRAJ_DATASET_NAME=v4_spec_120b,PRETRAJ_CURRICULUM_STAGE=stage1_entity_schema,PRETRAJ_RUN_SUFFIX=v4_120b_stage1,AUTO_RESUME=1 train_pretrajectory_sft_v1_medium_wandb.slurm
# For restartable runs, use a stable PERSISTENT_OUTPUT_DIR and AUTO_RESUME=1.

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0

export CC=cc
export CXX=CC
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC
export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:${LD_LIBRARY_PATH:-}

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

export MODEL_NAME="${MODEL_NAME:-gpt-oss-20b-bf16}"
export MODEL_PATH="${MODEL_PATH:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/${MODEL_NAME}}"

if [ -z "${PRETRAJ_DATASET_NAME:-}" ]; then
  if [ -n "${PRETRAJ_DATASET_DIR:-}" ]; then
    PRETRAJ_DATASET_NAME="$(basename "${PRETRAJ_DATASET_DIR}")"
  else
    PRETRAJ_DATASET_NAME="v1_medium"
  fi
fi
export PRETRAJ_DATASET_NAME
export PRETRAJ_DATASET_DIR="${PRETRAJ_DATASET_DIR:-${SLURM_SUBMIT_DIR}/data/pretrajectory_sft/${PRETRAJ_DATASET_NAME}}"
export PRETRAJ_RUN_SUFFIX="${PRETRAJ_RUN_SUFFIX:-${PRETRAJ_DATASET_NAME}}"
export PRETRAJ_CURRICULUM_STAGE="${PRETRAJ_CURRICULUM_STAGE:-}"

if [ -n "${PRETRAJ_CURRICULUM_STAGE}" ]; then
  export PRETRAJ_STAGE_DATASET_DIR="${PRETRAJ_DATASET_DIR}/curriculum/${PRETRAJ_CURRICULUM_STAGE}"
  export DATASET_PATH="${DATASET_PATH:-${PRETRAJ_STAGE_DATASET_DIR}/train.jsonl}"
  export VAL_DATASET_PATH="${VAL_DATASET_PATH:-${PRETRAJ_STAGE_DATASET_DIR}/val.jsonl}"
  export HOLDOUT_DATASET_PATH="${HOLDOUT_DATASET_PATH:-${PRETRAJ_STAGE_DATASET_DIR}/test.jsonl}"
else
  export DATASET_PATH="${DATASET_PATH:-${PRETRAJ_DATASET_DIR}/train.jsonl}"
  export VAL_DATASET_PATH="${VAL_DATASET_PATH:-${PRETRAJ_DATASET_DIR}/val.jsonl}"
  export HOLDOUT_DATASET_PATH="${HOLDOUT_DATASET_PATH:-${PRETRAJ_DATASET_DIR}/test.jsonl}"
fi
export DS_CONFIG="${DS_CONFIG:-${SLURM_SUBMIT_DIR}/config/ds_zero3_lora_no_offload.json}"
export PERSISTENT_OUTPUT_DIR="${PERSISTENT_OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/checkpoints/pretrajectory_sft_${PRETRAJ_RUN_SUFFIX}_${SLURM_JOB_ID}}"
export NUM_NODES="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
export HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"

export TRAIN_NUM_EPOCHS="${TRAIN_NUM_EPOCHS:-1}"
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
if [[ "${MODEL_NAME}" == *"120b"* || "${MODEL_NAME}" == *"120B"* ]]; then
  DEFAULT_LEARNING_RATE="5e-6"
else
  DEFAULT_LEARNING_RATE="1e-5"
fi
export LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LEARNING_RATE}}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
export EVAL_DATASET_SUBSET_SIZE="${EVAL_DATASET_SUBSET_SIZE:-}"
export LOGGING_STEPS="${LOGGING_STEPS:-10}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export HOLDOUT_SAMPLE_SIZE="${HOLDOUT_SAMPLE_SIZE:-64}"
export HOLDOUT_MAX_NEW_TOKENS="${HOLDOUT_MAX_NEW_TOKENS:-192}"
export HOLDOUT_ENABLE_THINKING="${HOLDOUT_ENABLE_THINKING:-True}"
export HOLDOUT_REASONING_EFFORT="${HOLDOUT_REASONING_EFFORT:-low}"
export HOLDOUT_MAX_TOTAL_TOKENS="${HOLDOUT_MAX_TOTAL_TOKENS:-1536}"
export RUN_HOLDOUT_AFTER_TRAINING="${RUN_HOLDOUT_AFTER_TRAINING:-True}"
export SEED="${SEED:-900913}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-True}"
export AUTO_RESUME="${AUTO_RESUME:-0}"
export RESUME_SEARCH_DIR="${RESUME_SEARCH_DIR:-${PERSISTENT_OUTPUT_DIR}}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export CHECKPOINT_SYNC_ENABLED="${CHECKPOINT_SYNC_ENABLED:-1}"
export CHECKPOINT_SYNC_ON_EXIT="${CHECKPOINT_SYNC_ON_EXIT:-1}"
export CHECKPOINT_SYNC_INTERVAL_SECONDS="${CHECKPOINT_SYNC_INTERVAL_SECONDS:-0}"
export STAGE_MODEL_ON_ALL_NODES="${STAGE_MODEL_ON_ALL_NODES:-1}"

export LOCAL_SSD_PATH="${LOCAL_SSD_PATH:-/mnt/bb/${USER}/pretraj_sft_${PRETRAJ_RUN_SUFFIX}_${SLURM_JOB_ID}}"
export LOCAL_MODEL_DIR="${LOCAL_SSD_PATH}/model/${MODEL_NAME}"
export LOCAL_OUTPUT_DIR="${LOCAL_SSD_PATH}/checkpoints"
export LOCAL_CACHE_BASE="${LOCAL_SSD_PATH}/caches"
export WANDB_DIR="${LOCAL_SSD_PATH}/wandb"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${LOCAL_CACHE_BASE}/hf"
export HF_DATASETS_CACHE="${LOCAL_CACHE_BASE}/datasets"
export TRITON_CACHE_DIR="${LOCAL_CACHE_BASE}/triton"
export TORCH_EXTENSIONS_DIR="${LOCAL_CACHE_BASE}/torch_extensions"

export http_proxy=http://proxy.ccs.ornl.gov:3128
export https_proxy="${http_proxy}"
export WANDB_HTTP_TIMEOUT=90
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-pretrajectory_sft_${PRETRAJ_DATASET_NAME}}"
export WANDB_TAGS="${WANDB_TAGS:-pretrajectory_sft,${PRETRAJ_DATASET_NAME},full,${MODEL_NAME}}"
export MENTOR_RUN_LABEL="${MENTOR_RUN_LABEL:-pretraj-${PRETRAJ_RUN_SUFFIX}-${SLURM_JOB_ID}}"

export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_OFI_USE_NICLIST=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-6.3.1
export HSA_XNACK=0
export DS_ACCELERATOR=cuda
export PYTORCH_ALLOC_CONF="expandable_segments:False"

find_latest_checkpoint() {
  local root="$1"
  if [ ! -d "${root}" ]; then
    return 1
  fi
  find "${root}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sed 's/^checkpoint-//' \
    | sort -n \
    | tail -n 1 \
    | awk -v root="${root}" 'NF {print root "/checkpoint-" $1}'
}

if [ -z "${RESUME_FROM_CHECKPOINT}" ] && [ "${AUTO_RESUME}" = "1" ]; then
  latest_checkpoint="$(find_latest_checkpoint "${RESUME_SEARCH_DIR}" || true)"
  if [ -n "${latest_checkpoint}" ]; then
    export RESUME_FROM_CHECKPOINT="${latest_checkpoint}"
  else
    export RESUME_FROM_CHECKPOINT="auto"
  fi
fi

sync_outputs() {
  local reason="${1:-manual}"
  if [ "${CHECKPOINT_SYNC_ENABLED}" != "1" ]; then
    return 0
  fi
  mkdir -p "${PERSISTENT_OUTPUT_DIR}"
  echo "--- Syncing outputs (${reason}) to ${PERSISTENT_OUTPUT_DIR} ---"
  if [ "${NUM_NODES}" -gt 1 ]; then
    srun --overlap --cpu-bind=none --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 bash -c '
      set +e
      if [ -d "${LOCAL_OUTPUT_DIR}" ]; then
        rsync -a --ignore-errors "${LOCAL_OUTPUT_DIR}/" "${PERSISTENT_OUTPUT_DIR}/"
      fi
    '
  elif [ -d "${LOCAL_OUTPUT_DIR}" ]; then
    rsync -a --ignore-errors "${LOCAL_OUTPUT_DIR}/" "${PERSISTENT_OUTPUT_DIR}/" || true
  fi
  if [ -d "${WANDB_DIR}" ]; then
    mkdir -p "${PERSISTENT_OUTPUT_DIR}/wandb"
    rsync -a --ignore-errors "${WANDB_DIR}/" "${PERSISTENT_OUTPUT_DIR}/wandb/" || true
  fi
}

SYNC_LOOP_PID=""
sync_outputs_safe() {
  sync_outputs "$1" || echo "[WARNING] Output sync failed during $1."
}

start_sync_loop() {
  if [ "${CHECKPOINT_SYNC_INTERVAL_SECONDS}" -gt 0 ]; then
    (
      trap '' TERM
      while true; do
        sleep "${CHECKPOINT_SYNC_INTERVAL_SECONDS}"
        sync_outputs_safe "periodic"
      done
    ) &
    SYNC_LOOP_PID="$!"
    echo "Started checkpoint sync loop with interval ${CHECKPOINT_SYNC_INTERVAL_SECONDS}s (PID ${SYNC_LOOP_PID})."
  fi
}

stop_sync_loop() {
  if [ -n "${SYNC_LOOP_PID}" ]; then
    kill "${SYNC_LOOP_PID}" 2>/dev/null || true
    wait "${SYNC_LOOP_PID}" 2>/dev/null || true
    SYNC_LOOP_PID=""
  fi
}

on_usr1() {
  echo "Received Slurm USR1 pre-timeout signal; syncing checkpoint outputs."
  sync_outputs_safe "slurm-usr1"
}

on_term() {
  echo "Received TERM; syncing checkpoint outputs before exit."
  sync_outputs_safe "term"
  exit 143
}

on_exit() {
  local exit_code="$?"
  stop_sync_loop
  if [ "${CHECKPOINT_SYNC_ON_EXIT}" = "1" ]; then
    sync_outputs_safe "exit-${exit_code}"
  fi
  exit "${exit_code}"
}

trap on_usr1 USR1
trap on_term TERM
trap on_exit EXIT

echo "--- Pre-trajectory SFT W&B run started ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Head node: ${HEAD_NODE}"
echo "Node count: ${NUM_NODES}"
echo "Dataset name: ${PRETRAJ_DATASET_NAME}"
echo "Dataset dir: ${PRETRAJ_DATASET_DIR}"
echo "Curriculum stage: ${PRETRAJ_CURRICULUM_STAGE:-none}"
echo "Dataset: ${DATASET_PATH}"
echo "Validation dataset: ${VAL_DATASET_PATH}"
echo "Holdout dataset: ${HOLDOUT_DATASET_PATH}"
echo "Output: ${PERSISTENT_OUTPUT_DIR}"
echo "W&B group: ${WANDB_RUN_GROUP}"
echo "W&B tags: ${WANDB_TAGS}"
echo "Epochs: ${TRAIN_NUM_EPOCHS}"
echo "Max steps: ${TRAIN_MAX_STEPS:-unset}"
echo "Eval steps: ${EVAL_STEPS}"
echo "Eval dataset subset size: ${EVAL_DATASET_SUBSET_SIZE:-full}"
echo "Save strategy: ${SAVE_STRATEGY}"
echo "Save steps: ${SAVE_STEPS}"
echo "Save total limit: ${SAVE_TOTAL_LIMIT}"
echo "Auto resume: ${AUTO_RESUME}"
echo "Resume search dir: ${RESUME_SEARCH_DIR}"
echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT:-unset}"
echo "Checkpoint sync interval seconds: ${CHECKPOINT_SYNC_INTERVAL_SECONDS}"
echo "Holdout sample size: ${HOLDOUT_SAMPLE_SIZE}"
echo "Run holdout after training: ${RUN_HOLDOUT_AFTER_TRAINING}"
echo "Holdout thinking: ${HOLDOUT_ENABLE_THINKING}"
echo "Holdout reasoning effort: ${HOLDOUT_REASONING_EFFORT}"
echo "Holdout max total tokens: ${HOLDOUT_MAX_TOTAL_TOKENS}"

for required_dataset_file in "${DATASET_PATH}" "${VAL_DATASET_PATH}" "${HOLDOUT_DATASET_PATH}"; do
  if [ ! -f "${required_dataset_file}" ]; then
    echo "[ERROR] Required dataset file is missing: ${required_dataset_file}" >&2
    exit 2
  fi
done

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 -w "${HEAD_NODE}" bash -c "
  set -euo pipefail
  echo 'Python check:'; which python; python -c 'import torch, transformers, trl, datasets, peft, wandb; print(torch.__version__, transformers.__version__)'
"

echo "--- Staging model to local NVMe ---"
if [ "${STAGE_MODEL_ON_ALL_NODES}" = "1" ]; then
  srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 bash -c '
    set -euo pipefail
    mkdir -p "${LOCAL_MODEL_DIR}" "${LOCAL_OUTPUT_DIR}" "${LOCAL_CACHE_BASE}" "${WANDB_DIR}"
    echo "[$(hostname)] staging model to ${LOCAL_MODEL_DIR}"
    rsync -a --info=stats2 "${MODEL_PATH}/" "${LOCAL_MODEL_DIR}/"
  '
else
  mkdir -p "${LOCAL_MODEL_DIR}" "${LOCAL_OUTPUT_DIR}" "${LOCAL_CACHE_BASE}" "${WANDB_DIR}"
  rsync -a --info=stats2 "${MODEL_PATH}/" "${LOCAL_MODEL_DIR}/"
fi

MASTER_ADDR="$(ip -4 addr show hsn0 | awk '/inet /{print $2}' | cut -d/ -f1 | head -n 1)"
if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi
export MASTER_ADDR
export MASTER_PORT=29500

echo "--- Launching SFT run ---"
start_sync_loop
srun --cpu-bind=none --accel-bind=g --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export AMDSMI_GPU_METRICS_CACHE_MS=5000
  if [ "${SLURM_PROCID}" != "0" ]; then
    export WANDB_MODE=disabled
  fi
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
  fi

  train_args=(
    --model_path="${LOCAL_MODEL_DIR}/" \
    --output_dir="${LOCAL_OUTPUT_DIR}" \
    --dataset_path="${DATASET_PATH}" \
    --eval_dataset_path="${VAL_DATASET_PATH}" \
    --holdout_dataset_path="${HOLDOUT_DATASET_PATH}" \
    --holdout_sample_size="${HOLDOUT_SAMPLE_SIZE}" \
    --holdout_max_new_tokens="${HOLDOUT_MAX_NEW_TOKENS}" \
    --holdout_enable_thinking="${HOLDOUT_ENABLE_THINKING}" \
    --holdout_reasoning_effort="${HOLDOUT_REASONING_EFFORT}" \
    --holdout_max_total_tokens="${HOLDOUT_MAX_TOTAL_TOKENS}" \
    --holdout_output_path="${LOCAL_OUTPUT_DIR}/holdout_predictions.jsonl" \
    --holdout_report_path="${LOCAL_OUTPUT_DIR}/holdout_report.json" \
    --holdout_exact_report_path="${LOCAL_OUTPUT_DIR}/holdout_exact_report.json" \
    --holdout_html_report_path="${LOCAL_OUTPUT_DIR}/holdout_review.html" \
    --run_inference_after_training="${RUN_HOLDOUT_AFTER_TRAINING}" \
    --local_files_only "${LOCAL_FILES_ONLY}" \
    --deepspeed="${DS_CONFIG}" \
    --seed="${SEED}" \
    --num_train_epochs="${TRAIN_NUM_EPOCHS}" \
    --per_device_train_batch_size="${TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size="${EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
    --learning_rate="${LEARNING_RATE}" \
    --lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
    --warmup_ratio="${WARMUP_RATIO}" \
    --eval_strategy=steps \
    --eval_steps="${EVAL_STEPS}" \
    --logging_steps="${LOGGING_STEPS}" \
    --save_strategy="${SAVE_STRATEGY}" \
    --save_steps="${SAVE_STEPS}" \
    --save_total_limit="${SAVE_TOTAL_LIMIT}" \
    --bf16 \
    --max_length="${MAX_LENGTH}" \
    --ddp_find_unused_parameters False \
    --lora_r="${LORA_R}" \
    --lora_alpha="${LORA_ALPHA}" \
    --lora_dropout="${LORA_DROPOUT}"
  )
  if [ -n "${TRAIN_MAX_STEPS}" ]; then
    train_args+=(--max_steps="${TRAIN_MAX_STEPS}")
  fi
  if [ -n "${EVAL_DATASET_SUBSET_SIZE}" ]; then
    train_args+=(--eval_dataset_subset_size="${EVAL_DATASET_SUBSET_SIZE}")
  fi
  if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    train_args+=(--resume_from_checkpoint="${RESUME_FROM_CHECKPOINT}")
  fi
  python "${SLURM_SUBMIT_DIR}/scripts/train_sft.py" "${train_args[@]}"
'
stop_sync_loop

echo "--- Copying full-run outputs ---"
sync_outputs "final"

echo "--- Cleaning local NVMe ---"
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 rm -rf "${LOCAL_SSD_PATH}"

echo "--- Pre-trajectory SFT W&B run finished ---"
echo "Persistent output: ${PERSISTENT_OUTPUT_DIR}"
trap - EXIT
