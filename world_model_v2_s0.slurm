#!/bin/bash
# Run one S0 method from one exact run contract.
#
# Set RUN_CONFIG and S0_METHOD_ID for each submission.
# Use a method-level run_settings object for a run-specific value.
# Slurm adds its job ID to the run ID and output path.
# Create logs/ before submission.
# Slurm uses these defaults for LoRA. Override tasks and GPUs for full fine-tune.

#SBATCH --job-name=wm2_s0
#SBATCH -A SYB114
#SBATCH -N 16
#SBATCH -t 8:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

die() {
  echo "[ERROR] $*" >&2
  exit 2
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
    die "${name} must be a positive integer."
}

# Find the repository from the Slurm submission directory.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
[[ -f "${REPO_ROOT}/world_model_v2_s0.slurm" ]] || \
  die "Submit this file from the repository root."
export REPO_ROOT PYTHONDONTWRITEBYTECODE=1

# Read one mode, one config, and one method from the submission.
JOB_MODE="${JOB_MODE:-train}"
RUN_CONFIG="${RUN_CONFIG:?RUN_CONFIG is required}"
S0_METHOD_ID="${S0_METHOD_ID:?S0_METHOD_ID is required}"
CONTRACT_ONLY="${CONTRACT_ONLY:-0}"

# Train mode includes teacher-forced validation loss.
# Test mode generates answers and then measures exact mapping accuracy.
case "${JOB_MODE}" in
  train|test) ;;
  validation)
    die "Validation loss runs during train. Use JOB_MODE=test for exact evaluation."
    ;;
  *) die "JOB_MODE must be train, validation, or test." ;;
esac
[[ "${CONTRACT_ONLY}" == "0" || "${CONTRACT_ONLY}" == "1" ]] || \
  die "CONTRACT_ONLY must be zero or one."

contract_env="$(mktemp "${TMPDIR:-/tmp}/wm2-s0-contract.XXXXXX")"
TEST_MODEL_STAGING_DIR=""

# Remove the temporary contract file after either mode exits.
# Remove an incomplete test model copy only from the default cache root.
cleanup_launcher() {
  rm -f -- "${contract_env}"
  if [[ -n "${TEST_MODEL_STAGING_DIR}" && \
        "${TEST_MODEL_STAGING_DIR}" == "/mnt/bb/${USER}/mentor_model_cache/"* ]]; then
    rm -rf -- "${TEST_MODEL_STAGING_DIR}"
  fi
}
trap cleanup_launcher EXIT

# Test flow:
# 1. Validate the config, data, tokenizer, model, and checkpoint identities.
# 2. Copy the verified base model to the node-local cache.
# 3. Build an answer-free bundle and generate one answer for each test row.
# 4. Score the answers and append metrics to the original W&B train run.
# Test mode exits before the distributed train path.
if [[ "${JOB_MODE}" == "test" ]]; then
  CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
  CONTRACT_CHECKER="${REPO_ROOT}/scripts/validate_world_model_v2_s0_eval_contract.py"
  BASE_MODEL_VALIDATOR="${REPO_ROOT}/scripts/validate_world_model_v2_s0_train_contract.py"
  [[ -f "${CONTRACT_CHECKER}" ]] || die "The test contract checker is absent."
  [[ -f "${BASE_MODEL_VALIDATOR}" ]] || \
    die "The base-model checker is absent."
  python "${CONTRACT_CHECKER}" "${REPO_ROOT}" "${RUN_CONFIG}" \
    "${S0_METHOD_ID}" "${CHECKPOINT_PATH}" > "${contract_env}"

  # Load only shell-safe values that the validated test contract owns.
  # shellcheck disable=SC1090
  source "${contract_env}"
  export S0_EVAL_CONFIG S0_EVAL_CONFIG_SHA256 S0_EVALUATION_ID
  export S0_METHOD_ID S0_TOKENIZER_METHOD TOKENIZER_PATH TOKENIZER_MANIFEST
  export S0_TOKENIZER_MANIFEST_SHA256 MODEL_ID MODEL_PATH
  export MODEL_IDENTITY_SHA256 BASE_MODEL_ARTIFACT_SHA256
  export BASE_MODEL_WEIGHT_SHARD_COUNT S0_CORPUS_MANIFEST_SHA256
  export S0_EVALUATOR_MANIFEST S0_EVALUATOR_MANIFEST_SHA256
  export S0_TEST_QUESTIONS S0_TEST_QUESTIONS_SHA256
  export S0_TEST_ANSWER_KEY S0_TEST_ANSWER_KEY_SHA256 S0_TEST_PANEL_ID
  export S0_TEST_ROWS CHECKPOINT_PATH S0_TRAIN_RUN_ID
  export S0_EVAL_OUTPUT_ROOT S0_EVAL_NUM_NODES S0_EVAL_TASKS_PER_NODE
  export S0_EVAL_GPUS_PER_NODE S0_MAX_NEW_TOKENS S0_MAX_TOTAL_TOKENS
  export S0_REASONING_EFFORT S0_EVAL_SEED
  export S0_MINIMUM_FAMILY_ACCURACY S0_TARGET_FAMILY_ACCURACY
  export S0_WANDB_ENTITY S0_WANDB_PROJECT

  # Use the Slurm job ID for one unique and immutable result path.
  if [[ "${CONTRACT_ONLY}" == "1" ]]; then
    execution_id="${SLURM_JOB_ID:-JOB-ID}"
  else
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Run test mode as a Slurm job."
    require_positive_integer SLURM_JOB_ID "${SLURM_JOB_ID}"
    execution_id="${SLURM_JOB_ID}"
  fi
  S0_EVAL_RUN_ID="${S0_METHOD_ID}-test-slurm-${execution_id}"
  OUTPUT_DIR="${S0_EVAL_OUTPUT_ROOT}/slurm-${execution_id}"
  export S0_EVAL_RUN_ID OUTPUT_DIR

  # Print the selected identities before any costly work starts.
  echo "Test: ${S0_EVAL_RUN_ID}"
  echo "Method: ${S0_METHOD_ID}"
  echo "Checkpoint: ${CHECKPOINT_PATH}"
  echo "Resources: nodes=${S0_EVAL_NUM_NODES} tasks_per_node=${S0_EVAL_TASKS_PER_NODE} gpus_per_node=${S0_EVAL_GPUS_PER_NODE}"
  echo "W&B entity: ${S0_WANDB_ENTITY}"
  echo "W&B project: ${S0_WANDB_PROJECT}"

  # One process can use all eight GPUs through the Transformers device map.
  # Reject a job if its allocation differs from this topology.
  if [[ "${CONTRACT_ONLY}" == "0" ]]; then
    [[ "${SLURM_NNODES:-}" == "${S0_EVAL_NUM_NODES}" ]] || \
      die "The allocated node count differs from the test contract."
    [[ "${SLURM_NTASKS:-}" == "${S0_EVAL_TASKS_PER_NODE}" ]] || \
      die "The allocated task count differs from the test contract."
    allocated_tasks_per_node="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}"
    allocated_tasks_per_node="${allocated_tasks_per_node%%(*}"
    [[ "${allocated_tasks_per_node}" == "${S0_EVAL_TASKS_PER_NODE}" ]] || \
      die "The allocated local task count differs from the test contract."
  fi

  # This hash identifies all code that can affect prompts, output, or scores.
  evaluation_code_files=(
    runtime/world_model_training.py
    runtime/world_model_s0.py
    runtime/world_model_s0_tokenizer.py
    runtime/world_model_domain_bpe.py
    runtime/world_model_token_adapter.py
    scripts/build_world_model_v2_s0_generation_bundle.py
    scripts/evaluate_world_model_v2_s0.py
    scripts/run_world_model_v2_s0_exact_generation.py
    scripts/validate_world_model_v2_s0_eval_contract.py
    scripts/validate_world_model_v2_s0_train_contract.py
    world_model_v2_s0.slurm
  )
  S0_EVALUATION_CODE_SHA256="$({
    for path in "${evaluation_code_files[@]}"; do
      [[ -f "${path}" ]] || die "A test code file is absent: ${path}"
      printf '%s  %s\n' "$(sha256sum "${path}" | awk '{print $1}')" "${path}"
    done
  } | sha256sum | awk '{print $1}')"
  export S0_EVALUATION_CODE_SHA256

  # Contract-only mode stops before it creates output or loads the model.
  if [[ "${CONTRACT_ONLY}" == "1" ]]; then
    echo "The selected test inputs and config are valid. No Slurm job ran."
    exit 0
  fi

  # Do not reuse an output path because that can mix two test results.
  # Copy the immutable inputs so that each result has a local audit trail.
  [[ ! -e "${OUTPUT_DIR}" ]] || die "The test output already exists: ${OUTPUT_DIR}"
  umask 077
  mkdir -p "${OUTPUT_DIR}/run_contract" "${OUTPUT_DIR}/wandb"
  RUN_CONTRACT_DIR="${OUTPUT_DIR}/run_contract"
  GENERATION_BUNDLE_DIR="${OUTPUT_DIR}/generation_bundle"
  GENERATION_OUTPUT_DIR="${OUTPUT_DIR}/generation"
  REPORT_OUTPUT_DIR="${OUTPUT_DIR}/report"
  export RUN_CONTRACT_DIR GENERATION_BUNDLE_DIR
  export GENERATION_OUTPUT_DIR REPORT_OUTPUT_DIR
  cp "${S0_EVAL_CONFIG}" "${RUN_CONTRACT_DIR}/test_config.json"
  cp "${S0_EVALUATOR_MANIFEST}" \
    "${RUN_CONTRACT_DIR}/evaluator_manifest.json"
  cp "${TOKENIZER_MANIFEST}" \
    "${RUN_CONTRACT_DIR}/tokenizer_manifest.json"
  cp "${CHECKPOINT_PATH}/run_contract/training_exposure.json" \
    "${RUN_CONTRACT_DIR}/training_exposure.json"
  printf '%s\n' \
    "job_id=${SLURM_JOB_ID}" \
    "evaluation_id=${S0_EVALUATION_ID}" \
    "evaluation_run_id=${S0_EVAL_RUN_ID}" \
    "method_id=${S0_METHOD_ID}" \
    "train_run_id=${S0_TRAIN_RUN_ID}" \
    "checkpoint_path=${CHECKPOINT_PATH}" \
    "test_config_sha256=${S0_EVAL_CONFIG_SHA256}" \
    "test_panel_id=${S0_TEST_PANEL_ID}" \
    "test_questions_sha256=${S0_TEST_QUESTIONS_SHA256}" \
    "test_answer_key_sha256=${S0_TEST_ANSWER_KEY_SHA256}" \
    "tokenizer_manifest_sha256=${S0_TOKENIZER_MANIFEST_SHA256}" \
    "model_identity_sha256=${MODEL_IDENTITY_SHA256}" \
    "base_model_artifact_sha256=${BASE_MODEL_ARTIFACT_SHA256}" \
    "evaluation_code_sha256=${S0_EVALUATION_CODE_SHA256}" \
    "wandb_entity=${S0_WANDB_ENTITY}" \
    "wandb_project=${S0_WANDB_PROJECT}" \
    "output_dir=${OUTPUT_DIR}" \
    > "${RUN_CONTRACT_DIR}/launch.env"

  # Use the same pinned Frontier stack as the train job.
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
  # This step prevents a stale state error at the first torch import.
  echo "Clear user-owned ROCm-SMI state before the first torch import."
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpu-bind=none bash -c '
    set -euo pipefail
    shopt -s nullglob
    for path in /dev/shm/rocm_smi* /dev/shm/amd_smi*; do
      [[ ! -O "${path}" ]] || rm -f -- "${path}"
    done
  '

  # Load optional project credentials before the costly model copy.
  # W&B can also use the saved login from the user account.
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi

  # The test contract owns the W&B destination.
  # The scorer uses S0_TRAIN_RUN_ID to resume the original train run.
  # Remove inherited run values so that they cannot select another run.
  unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_RESUME
  export WANDB_MODE=online
  export WANDB_ENTITY="${S0_WANDB_ENTITY}"
  export WANDB_PROJECT="${S0_WANDB_PROJECT}"
  export WANDB_DIR="${OUTPUT_DIR}/wandb"
  export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-180}"
  export http_proxy="${http_proxy:-http://proxy.ccs.ornl.gov:3128}"
  export https_proxy="${https_proxy:-${http_proxy}}"

  # Keep all model libraries offline and use node-local caches.
  export CC=cc CXX=CC CMAKE_C_COMPILER=cc CMAKE_CXX_COMPILER=CC
  export ROCM_HOME=/opt/rocm-6.3.1 PYTORCH_ROCM_ARCH=gfx90a HSA_XNACK=0
  export PYTORCH_ALLOC_CONF="expandable_segments:False"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-56}"
  export HF_HOME="/mnt/bb/${USER}/mentor_hf/${SLURM_JOB_ID}/test"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export TORCH_EXTENSIONS_DIR="/mnt/bb/${USER}/mentor_torch_extensions/torch280_rocm631"
  export TRITON_CACHE_DIR="/mnt/bb/${USER}/mentor_triton/${SLURM_JOB_ID}/test"
  mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"

  # The base-model hash covers its metadata and all weight shards.
  # Reuse only a node-local cache that passes this complete check.
  LOCAL_MODEL_CACHE_ROOT="${LOCAL_MODEL_CACHE_ROOT:-/mnt/bb/${USER}/mentor_model_cache}"
  LOCAL_MODEL_DIR="${LOCAL_MODEL_CACHE_ROOT}/${MODEL_ID}-${BASE_MODEL_ARTIFACT_SHA256:0:16}"
  [[ "${LOCAL_MODEL_CACHE_ROOT}" == "/mnt/bb/${USER}/"* ]] || \
    die "The local model root must stay below the user burst buffer."
  [[ "${LOCAL_MODEL_DIR}" == "${LOCAL_MODEL_CACHE_ROOT}/"* ]] || \
    die "The local model path must stay below its root."
  mkdir -p "${LOCAL_MODEL_CACHE_ROOT}"
  exec 9>"${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.stage.lock"
  # The lock prevents two jobs from writing the same node-local cache.
  flock 9
  if [[ -e "${LOCAL_MODEL_DIR}" ]]; then
    # Check every cached file before the model process uses the cache.
    [[ -s "${LOCAL_MODEL_DIR}/.mentor_model_identity_sha256" ]] || \
      die "The model cache has no identity marker."
    [[ "$(<"${LOCAL_MODEL_DIR}/.mentor_model_identity_sha256")" == \
      "${BASE_MODEL_ARTIFACT_SHA256}" ]] || \
      die "The model cache marker changed."
    python "${BASE_MODEL_VALIDATOR}" --validate-base-model \
      "${LOCAL_MODEL_DIR}" "${BASE_MODEL_ARTIFACT_SHA256}" \
      "${BASE_MODEL_WEIGHT_SHARD_COUNT}" \
      "${LOCAL_MODEL_DIR}/.mentor_base_model_artifact.json"
  else
    # Copy to a temporary path and publish the cache only after validation.
    TEST_MODEL_STAGING_DIR="$(
      mktemp -d \
        "${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.test.${SLURM_JOB_ID}.XXXXXX"
    )"
    rsync -a "${MODEL_PATH}/" "${TEST_MODEL_STAGING_DIR}/"
    python "${BASE_MODEL_VALIDATOR}" --validate-base-model \
      "${TEST_MODEL_STAGING_DIR}" "${BASE_MODEL_ARTIFACT_SHA256}" \
      "${BASE_MODEL_WEIGHT_SHARD_COUNT}" \
      "${TEST_MODEL_STAGING_DIR}/.mentor_base_model_artifact.json"
    printf '%s\n' "${BASE_MODEL_ARTIFACT_SHA256}" \
      > "${TEST_MODEL_STAGING_DIR}/.mentor_model_identity_sha256"
    mv "${TEST_MODEL_STAGING_DIR}" "${LOCAL_MODEL_DIR}"
    TEST_MODEL_STAGING_DIR=""
  fi
  flock -u 9

  # Confirm that the one task can use all eight GPUs and each required package.
  # This check runs before the script loads model weights into GPU memory.
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gpus-per-task="${S0_EVAL_GPUS_PER_NODE}" --cpu-bind=none \
    --accel-bind=none python -c \
    'import torch, transformers, peft, accelerate, wandb; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print("test_python_rocm_ready", flush=True)'

  # The bundle builder reads public questions and writes only safe prompt fields.
  # It does not copy the private answer key into the bundle.
  python scripts/build_world_model_v2_s0_generation_bundle.py \
    --evaluator-manifest "${S0_EVALUATOR_MANIFEST}" \
    --evaluator-manifest-sha256 "${S0_EVALUATOR_MANIFEST_SHA256}" \
    --output-dir "${GENERATION_BUNDLE_DIR}"

  # Give the model process only the public bundle and verified model inputs.
  # Remove the answer-key path and W&B key from its environment.
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --gpus-per-task="${S0_EVAL_GPUS_PER_NODE}" --cpu-bind=none \
    --accel-bind=none env \
    -u WANDB_API_KEY \
    -u S0_TEST_ANSWER_KEY \
    -u S0_TEST_ANSWER_KEY_SHA256 \
    -u S0_EVALUATOR_MANIFEST \
    WANDB_MODE=disabled \
    python scripts/run_world_model_v2_s0_exact_generation.py \
      --generation-bundle "${GENERATION_BUNDLE_DIR}" \
      --base-model-path "${LOCAL_MODEL_DIR}" \
      --checkpoint-path "${CHECKPOINT_PATH}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --tokenizer-manifest "${TOKENIZER_MANIFEST}" \
      --output-dir "${GENERATION_OUTPUT_DIR}" \
      --method-id "${S0_METHOD_ID}" \
      --corpus-manifest-sha256 "${S0_CORPUS_MANIFEST_SHA256}" \
      --tokenizer-manifest-sha256 "${S0_TOKENIZER_MANIFEST_SHA256}" \
      --model-identity-sha256 "${MODEL_IDENTITY_SHA256}" \
      --max-new-tokens "${S0_MAX_NEW_TOKENS}" \
      --max-total-tokens "${S0_MAX_TOTAL_TOKENS}" \
      --reasoning-effort "${S0_REASONING_EFFORT}" \
      --seed "${S0_EVAL_SEED}"

  # The model process exits before the scorer reads the private answer key.
  # Append exact metrics to the W&B run from the checkpoint receipt.
  python scripts/evaluate_world_model_v2_s0.py \
    --evaluator-manifest "${S0_EVALUATOR_MANIFEST}" \
    --evaluator-manifest-sha256 "${S0_EVALUATOR_MANIFEST_SHA256}" \
    --predictions "${GENERATION_OUTPUT_DIR}/predictions.jsonl" \
    --generation-manifest "${GENERATION_OUTPUT_DIR}/generation_manifest.json" \
    --tokenizer-manifest "${TOKENIZER_MANIFEST}" \
    --output-dir "${REPORT_OUTPUT_DIR}" \
    --minimum-family-accuracy "${S0_MINIMUM_FAMILY_ACCURACY}" \
    --target-family-accuracy "${S0_TARGET_FAMILY_ACCURACY}" \
    --publish-wandb

  # Require the answer-free bundle, raw outputs, exact report, and W&B receipt.
  completed_files=(
    "${GENERATION_BUNDLE_DIR}/manifest.json"
    "${GENERATION_BUNDLE_DIR}/questions.jsonl"
    "${GENERATION_OUTPUT_DIR}/generation_plan.json"
    "${GENERATION_OUTPUT_DIR}/predictions.jsonl"
    "${GENERATION_OUTPUT_DIR}/generation_manifest.json"
    "${REPORT_OUTPUT_DIR}/exact_mapping_report.json"
    "${REPORT_OUTPUT_DIR}/test_metrics.json"
    "${REPORT_OUTPUT_DIR}/wandb_test_run.json"
  )
  for completed in "${completed_files[@]}"; do
    [[ -s "${completed}" ]] || die "The completed test lacks ${completed}."
  done
  echo "The exact S0 test is complete: ${OUTPUT_DIR}"
  exit 0
fi

# Resolve one method. The run contract owns all train values.
CONTRACT_CHECKER="${REPO_ROOT}/scripts/validate_world_model_v2_s0_train_contract.py"
[[ -f "${CONTRACT_CHECKER}" ]] || die "The train contract checker is absent."
python "${CONTRACT_CHECKER}" "${REPO_ROOT}" "${RUN_CONFIG}" "${S0_METHOD_ID}" > "${contract_env}"

# shellcheck disable=SC1090
source "${contract_env}"
export S0_RUN_CONFIG S0_RUN_CONFIG_SHA256 S0_CONFIG_ID S0_RUN_SCOPE
export S0_METHOD_ID S0_TRAIN_MODE S0_TOKENIZER_METHOD
export S0_TOKENIZER_ARM_ROOT S0_TOKENIZER_ARM_MANIFEST_SHA256
export S0_TOKENIZER_MANIFEST_SHA256 TOKENIZER_PATH TOKEN_ADAPTER_MANIFEST
export MODEL_ID MODEL_PATH MODEL_IDENTITY_SHA256
export BASE_MODEL_ARTIFACT_SCHEMA BASE_MODEL_ARTIFACT_SHA256
export BASE_MODEL_WEIGHT_SHARD_COUNT S0_CORPUS_ROOT
export S0_CORPUS_MANIFEST_SHA256 S0_TRAIN_PATH S0_TRAIN_SHA256
export S0_TRAIN_ROWS S0_VALIDATION_PATH S0_VALIDATION_SHA256
export S0_VALIDATION_ROWS S0_VALIDATION_ANSWER_KEY_PATH
export S0_VALIDATION_ANSWER_KEY_SHA256 S0_EVALUATOR_MANIFEST
export TRAINER_PATH OUTPUT_ROOT
export NUM_NODES GPUS_PER_NODE RANKS_PER_NODE WORLD_SIZE_EXPECTED
export TP_SIZE EP_SIZE DATA_PARALLEL_SIZE MAX_LENGTH
export S0_TRAINING_CODE_SHA256=""
S0_TRACKING_MODE=online
S0_WANDB_ENTITY=jail-ai
export S0_TRACKING_MODE S0_WANDB_ENTITY

# Add the scheduler identity to this execution and its output path.
if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  execution_id="${SLURM_JOB_ID:-JOB-ID}"
else
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "Run this mode as a Slurm job."
  require_positive_integer SLURM_JOB_ID "${SLURM_JOB_ID}"
  execution_id="${SLURM_JOB_ID}"
fi
S0_RUN_ID="${S0_METHOD_ID}-slurm-${execution_id}"
OUTPUT_DIR="${OUTPUT_ROOT}/slurm-${execution_id}"
export S0_RUN_ID OUTPUT_DIR

echo "Run: ${S0_RUN_ID}"
echo "Method: ${S0_METHOD_ID}"
echo "Trainer: ${S0_TRAIN_MODE}"
echo "Resources: nodes=${NUM_NODES} ranks_per_node=${RANKS_PER_NODE}"
echo "Train values: epochs=${TRAIN_NUM_EPOCHS} steps=${TRAIN_MAX_STEPS} global_batch=${GLOBAL_BATCH_SIZE} learning_rate=${LEARNING_RATE}"
echo "W&B mode: ${S0_TRACKING_MODE}"
echo "W&B entity: ${S0_WANDB_ENTITY}"

# Require an allocation that matches the selected topology.
if [[ "${CONTRACT_ONLY}" == "0" ]]; then
  [[ "${SLURM_NNODES:-}" == "${NUM_NODES}" ]] || \
    die "The allocated node count differs from the run contract."
  [[ "${SLURM_NTASKS:-}" == "${WORLD_SIZE_EXPECTED}" ]] || \
    die "The allocated rank count differs from the run contract."
  allocated_ranks_per_node="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}"
  allocated_ranks_per_node="${allocated_ranks_per_node%%(*}"
  [[ "${allocated_ranks_per_node}" == "${RANKS_PER_NODE}" ]] || \
    die "The allocated local rank count differs from the run contract."
fi

# Require each file that the selected method uses.
required_files=(
  "${S0_RUN_CONFIG}"
  "${S0_CORPUS_ROOT}/manifest.json"
  "${S0_TRAIN_PATH}"
  "${S0_VALIDATION_PATH}"
  "${S0_VALIDATION_ANSWER_KEY_PATH}"
  "${S0_EVALUATOR_MANIFEST}"
  "${S0_TOKENIZER_ARM_ROOT}/manifest.json"
  "${S0_TOKENIZER_ARM_ROOT}/audit_report.json"
  "${S0_TOKENIZER_ARM_ROOT}/tokenizer_manifest.json"
  "${TOKENIZER_PATH}/tokenizer.json"
  "${MODEL_PATH}/config.json"
  "${MODEL_PATH}/model.safetensors.index.json"
  "${TRAINER_PATH}"
)
[[ -z "${TOKEN_ADAPTER_MANIFEST}" ]] || \
  required_files+=("${TOKEN_ADAPTER_MANIFEST}")
[[ "${S0_TRAIN_MODE}" != "full_finetune" ]] || \
  required_files+=("${DS_CONFIG}")
for required in "${required_files[@]}"; do
  [[ -f "${required}" ]] || die "A required file is absent: ${required}"
done

# Verify the config and model metadata before the model stage.
[[ "$(sha256sum "${S0_RUN_CONFIG}" | awk '{print $1}')" == \
  "${S0_RUN_CONFIG_SHA256}" ]] || die "The run config identity changed."
if [[ "${S0_TRAIN_MODE}" == "full_finetune" ]]; then
  [[ "$(sha256sum "${DS_CONFIG}" | awk '{print $1}')" == \
    "${DS_CONFIG_SHA256_EXPECTED}" ]] || \
    die "The DeepSpeed config identity changed."
fi

model_identity_inputs=(
  "${MODEL_PATH}/config.json"
  "${MODEL_PATH}/model.safetensors.index.json"
)
[[ ! -f "${MODEL_PATH}/generation_config.json" ]] || \
  model_identity_inputs+=("${MODEL_PATH}/generation_config.json")
actual_model_identity="$({
  for path in "${model_identity_inputs[@]}"; do
    printf '%s  %s\n' \
      "$(sha256sum "${path}" | awk '{print $1}')" \
      "$(basename "${path}")"
  done
} | sha256sum | awk '{print $1}')"
[[ "${actual_model_identity}" == "${MODEL_IDENTITY_SHA256}" ]] || \
  die "The model metadata identity changed."

# Validate the corpus and tokenizer with the shared checker.
python "${CONTRACT_CHECKER}" --validate-inputs

# Hash the direct code inputs for this method.
training_code_files=(
  runtime/world_model_training.py
  runtime/world_model_s0.py
  runtime/world_model_s0_tokenizer.py
  runtime/world_model_domain_bpe.py
  runtime/world_model_token_adapter.py
  scripts/validate_world_model_v2_s0_train_contract.py
  utils/utils.py
  world_model_v2_s0.slurm
)
if [[ "${S0_TRAIN_MODE}" == "lora" ]]; then
  training_code_files+=(
    scripts/train_sft_dp_tp_ep.py
    scripts/world_model_dp_tp_ep.py
    scripts/tp_ep_autograd.py
  )
else
  training_code_files+=(scripts/train_sft_full_zero3.py)
fi
S0_TRAINING_CODE_SHA256="$({
  for path in "${training_code_files[@]}"; do
    [[ -f "${path}" ]] || die "A train code file is absent: ${path}"
    printf '%s  %s\n' "$(sha256sum "${path}" | awk '{print $1}')" "${path}"
  done
} | sha256sum | awk '{print $1}')"
export S0_TRAINING_CODE_SHA256

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  echo "The selected inputs and config are valid. No Slurm job ran."
  exit 0
fi

# Create a new output and copy its immutable input receipts.
[[ ! -e "${OUTPUT_DIR}" ]] || die "The output already exists: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/run_contract" "${OUTPUT_DIR}/wandb"
RUN_CONTRACT_DIR="${OUTPUT_DIR}/run_contract"
export RUN_CONTRACT_DIR
cp "${S0_RUN_CONFIG}" "${RUN_CONTRACT_DIR}/run_config.json"
cp "${S0_CORPUS_ROOT}/manifest.json" \
  "${RUN_CONTRACT_DIR}/dataset_manifest.json"
cp "${S0_EVALUATOR_MANIFEST}" \
  "${RUN_CONTRACT_DIR}/evaluator_manifest.json"
cp "${S0_TOKENIZER_ARM_ROOT}/manifest.json" \
  "${RUN_CONTRACT_DIR}/tokenizer_arm_manifest.json"
cp "${S0_TOKENIZER_ARM_ROOT}/tokenizer_manifest.json" \
  "${RUN_CONTRACT_DIR}/tokenizer_manifest.json"
if [[ "${S0_TRAIN_MODE}" == "full_finetune" ]]; then
  cp "${DS_CONFIG}" "${RUN_CONTRACT_DIR}/deepspeed_config.json"
fi

BASE_MODEL_VALIDATOR="${CONTRACT_CHECKER}"
export BASE_MODEL_VALIDATOR

# Record the selected run values before the job starts.
printf '%s\n' \
  "job_id=${SLURM_JOB_ID}" \
  "config_id=${S0_CONFIG_ID}" \
  "run_id=${S0_RUN_ID}" \
  "method_id=${S0_METHOD_ID}" \
  "train_mode=${S0_TRAIN_MODE}" \
  "run_config_sha256=${S0_RUN_CONFIG_SHA256}" \
  "corpus_manifest_sha256=${S0_CORPUS_MANIFEST_SHA256}" \
  "train_sha256=${S0_TRAIN_SHA256}" \
  "validation_sha256=${S0_VALIDATION_SHA256}" \
  "validation_rows=${S0_VALIDATION_ROWS}" \
  "validation_answer_key_sha256=${S0_VALIDATION_ANSWER_KEY_SHA256}" \
  "tokenizer_arm_manifest_sha256=${S0_TOKENIZER_ARM_MANIFEST_SHA256}" \
  "tokenizer_manifest_sha256=${S0_TOKENIZER_MANIFEST_SHA256}" \
  "model_identity_sha256=${MODEL_IDENTITY_SHA256}" \
  "base_model_artifact_sha256=${BASE_MODEL_ARTIFACT_SHA256}" \
  "training_code_sha256=${S0_TRAINING_CODE_SHA256}" \
  "nodes=${NUM_NODES}" \
  "ranks_per_node=${RANKS_PER_NODE}" \
  "world_size=${WORLD_SIZE_EXPECTED}" \
  "epochs=${TRAIN_NUM_EPOCHS}" \
  "max_steps=${TRAIN_MAX_STEPS}" \
  "global_batch_size=${GLOBAL_BATCH_SIZE}" \
  "learning_rate=${LEARNING_RATE}" \
  "eval_strategy=${EVAL_STRATEGY}" \
  "eval_on_start=${EVAL_ON_START}" \
  "per_device_eval_batch_size=${EVAL_BATCH_SIZE}" \
  "wandb_mode=${S0_TRACKING_MODE}" \
  "wandb_entity=${S0_WANDB_ENTITY}" \
  "output_dir=${OUTPUT_DIR}" \
  > "${RUN_CONTRACT_DIR}/launch.env"

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
export ROCM_HOME=/opt/rocm-6.3.1 PYTORCH_ROCM_ARCH=gfx90a HSA_XNACK=0
export DS_ACCELERATOR=cuda PYTORCH_ALLOC_CONF="expandable_segments:False"
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
export DS_CONFIG_SHA256_EXPECTED
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-7}"
export WANDB_MODE="${S0_TRACKING_MODE}"
# The run contract owns the W&B workspace. Credentials only authenticate the user.
export WANDB_ENTITY="${S0_WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT:-mentor-sft}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${S0_CONFIG_ID}}"
export WANDB_NAME="${WANDB_NAME:-${S0_RUN_ID}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${S0_RUN_ID}}"
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

if [[ -z "${MODEL_STAGE_WAVE_SIZE:-}" ]]; then
  MODEL_STAGE_WAVE_SIZE=4
  if (( NUM_NODES < MODEL_STAGE_WAVE_SIZE )); then
    MODEL_STAGE_WAVE_SIZE="${NUM_NODES}"
  fi
fi
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
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="/mnt/bb/${USER}/mentor_torch_extensions/torch280_rocm631"
export TRITON_CACHE_DIR="/mnt/bb/${USER}/mentor_triton/${SLURM_JOB_ID}"

# Import torch once on each node after the cleanup.
echo "Check the Python and ROCm stack on each node."
srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 \
  --gpus-per-task=1 --cpu-bind=none --accel-bind=g \
  python -c 'import socket, torch, transformers, peft, accelerate, datasets, trl; assert torch.cuda.is_available(); print("python_rocm_ready host={} torch={} gpu_count={}".format(socket.gethostname(), torch.__version__, torch.cuda.device_count()), flush=True)'

# Write and check one receipt for every allocated rank.
rank_receipt_dir="${RUN_CONTRACT_DIR}/rank_preflight"
mkdir -p "${rank_receipt_dir}"
export rank_receipt_dir
echo "Check all allocated ranks."
srun --nodes="${NUM_NODES}" --ntasks="${WORLD_SIZE_EXPECTED}" \
  --ntasks-per-node="${RANKS_PER_NODE}" --gpus-per-task=1 \
  --cpu-bind=none --accel-bind=g --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  printf "%s\t%s\t%s\n" \
    "${SLURM_PROCID}" "${SLURM_LOCALID}" "$(hostname)" \
    > "${rank_receipt_dir}/rank-${SLURM_PROCID}.tsv"
'
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
    rank, local_rank, host = path.read_text(encoding="utf-8").strip().split("\t")
    receipts.append((int(rank), int(local_rank), host))
if sorted(rank for rank, _, _ in receipts) != list(range(world_size)):
    raise SystemExit("The rank receipt set is incomplete or duplicated")
by_host = defaultdict(set)
for _, local_rank, host in receipts:
    by_host[host].add(local_rank)
if len(by_host) != node_count:
    raise SystemExit("The rank receipts have the wrong node count")
expected_local = set(range(local_world_size))
if any(local_ranks != expected_local for local_ranks in by_host.values()):
    raise SystemExit("The local rank receipt set is incomplete or duplicated")
PY

# Build the arguments that both trainers share.
train_args=(
  --model_path "${TRAIN_MODEL_PATH}"
  --tokenizer_path "${TOKENIZER_PATH}"
  --dataset_path "${S0_TRAIN_PATH}"
  --validation_dataset_path "${S0_VALIDATION_PATH}"
  --validation_answer_key_path "${S0_VALIDATION_ANSWER_KEY_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --num_train_epochs "${TRAIN_NUM_EPOCHS}"
  --max_steps "${TRAIN_MAX_STEPS}"
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --warmup_ratio "${WARMUP_RATIO}"
  --loss_contract "${LOSS_CONTRACT}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --logging_steps "${LOGGING_STEPS}"
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
  --save_strategy "${SAVE_STRATEGY}"
  --max_length "${MAX_LENGTH}"
  --seed "${SEED}"
)
[[ -z "${TOKEN_ADAPTER_MANIFEST}" ]] || \
  train_args+=(--token_adapter_manifest "${TOKEN_ADAPTER_MANIFEST}")

# Add the arguments for the selected train mode.
if [[ "${S0_TRAIN_MODE}" == "lora" ]]; then
  train_args+=(
    --tp_size "${TP_SIZE}"
    --ep_size "${EP_SIZE}"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --eval_strategy "${EVAL_STRATEGY}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
  )
  if [[ "${EVAL_ON_START}" == "1" ]]; then
    train_args+=(--eval_on_start)
  else
    train_args+=(--no-eval_on_start)
  fi
  if [[ "${PRESERVE_DATASET_ORDER}" == "1" ]]; then
    train_args+=(--preserve_dataset_order)
  else
    train_args+=(--no-preserve_dataset_order)
  fi
  if [[ "${AUTOGRAD_MULTITHREADING}" == "1" ]]; then
    train_args+=(--autograd-multithreading)
  else
    train_args+=(--no-autograd-multithreading)
  fi
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    train_args+=(--local_files_only)
  else
    train_args+=(--no-local_files_only)
  fi
  if [[ "${STRICT_TESTED_STACK}" == "1" ]]; then
    train_args+=(--strict_tested_stack)
  else
    train_args+=(--no-strict_tested_stack)
  fi
  [[ "${BF16}" != "1" ]] || train_args+=(--bf16)
else
  train_args+=(
    --deepspeed "${DS_CONFIG}"
    --expected_model_parameters "${EXPECTED_MODEL_PARAMETERS}"
    --local_files_only "$([[ "${LOCAL_FILES_ONLY}" == "1" ]] && echo true || echo false)"
    --bf16 "$([[ "${BF16}" == "1" ]] && echo true || echo false)"
    --gradient_checkpointing true
    --ddp_find_unused_parameters false
    --dataloader_num_workers 0
    --data_seed "${SEED}"
    --eval_strategy "${EVAL_STRATEGY}"
    --eval_on_start "$([[ "${EVAL_ON_START}" == "1" ]] && echo true || echo false)"
    --packing false
    --report_to wandb
    --run_name "${WANDB_NAME}"
  )
fi

# Launch one process for each selected rank.
launch_training() {
  srun --nodes="${NUM_NODES}" --ntasks="${WORLD_SIZE_EXPECTED}" \
    --ntasks-per-node="${RANKS_PER_NODE}" --gpus-per-task=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-7}" --cpu-bind=none \
    --accel-bind=g --kill-on-bad-exit=1 bash -c '
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
    if [[ "${S0_TRAIN_MODE}" == "lora" ]]; then
      export TP_GROUP_INDEX="$((RANK / TP_SIZE))"
      export TP_GROUP_RANK="$((RANK % TP_SIZE))"
      export EP_GROUP_INDEX="${TP_GROUP_INDEX}"
      export EP_GROUP_RANK="${TP_GROUP_RANK}"
    fi
    export HF_HOME="${HF_HOME}/rank_${RANK}"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/rank_${RANK}"
    mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"
    if [[ "${RANK}" != "0" ]]; then
      export WANDB_MODE=disabled
    fi
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
      export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
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
if [[ "${S0_TRAIN_MODE}" == "lora" ]]; then
  completed_files=(
    "${OUTPUT_DIR}/adapter_config.json"
    "${OUTPUT_DIR}/adapter_model.safetensors"
    "${OUTPUT_DIR}/tp_adapter_manifest.json"
    "${OUTPUT_DIR}/validation_metrics.jsonl"
    "${RUN_CONTRACT_DIR}/training_exposure.json"
  )
else
  completed_files=(
    "${OUTPUT_DIR}/config.json"
    "${OUTPUT_DIR}/model.safetensors.index.json"
    "${OUTPUT_DIR}/validation_metrics.jsonl"
    "${RUN_CONTRACT_DIR}/full_finetune_manifest.json"
    "${RUN_CONTRACT_DIR}/training_exposure.json"
  )
fi
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
echo "The S0 train run is complete: ${OUTPUT_DIR}"
