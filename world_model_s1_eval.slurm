#!/bin/bash
# Run exact native-transcript S1 evaluation for one completed production model.
#
# Example:
#   sbatch \
#     --export=ALL,RUN_CONFIG=config/world_model_s1_120b_32n_test_v10_structured95_512n.json,\
#S1_METHOD_ID=oss120b-plain-base-tokenizer-lora-r32,\
#CHECKPOINT_PATH=data/world_model_v4/checkpoints/s1_v10_120b_32n/\
#oss120b-plain-base-tokenizer-lora-r32/slurm-JOBID \
#     world_model_s1_eval.slurm

#SBATCH --job-name=wm_s1_eval
#SBATCH -A SYB114
#SBATCH -N 512
#SBATCH -t 16:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
[[ -f "${REPO_ROOT}/world_model_s1_eval.slurm" ]] || \
  die "Submit world_model_s1_eval.slurm from the repository root."
export REPO_ROOT PYTHONDONTWRITEBYTECODE=1

RUN_CONFIG="${RUN_CONFIG:?RUN_CONFIG is required}"
S1_METHOD_ID="${S1_METHOD_ID:?S1_METHOD_ID is required}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?CHECKPOINT_PATH is required}"
CONTRACT_ONLY="${CONTRACT_ONLY:-0}"
S1_GENERATION_ATTEMPTS="${S1_GENERATION_ATTEMPTS:-2}"
S1_RESUME_SHARDS_ROOT="${S1_RESUME_SHARDS_ROOT:-}"
PUBLISH_WANDB="${PUBLISH_WANDB:-1}"
[[ "${CONTRACT_ONLY}" == "0" || "${CONTRACT_ONLY}" == "1" ]] || \
  die "CONTRACT_ONLY must be zero or one."
[[ "${PUBLISH_WANDB}" == "0" || "${PUBLISH_WANDB}" == "1" ]] || \
  die "PUBLISH_WANDB must be zero or one."
require_positive_integer S1_GENERATION_ATTEMPTS "${S1_GENERATION_ATTEMPTS}"

contract_env="$(mktemp "${TMPDIR:-/tmp}/wm-s1-eval-contract.XXXXXX")"
cleanup_contract() {
  rm -f -- "${contract_env}"
}
trap cleanup_contract EXIT

python scripts/validate_world_model_s1_eval_contract.py \
  "${REPO_ROOT}" "${RUN_CONFIG}" "${S1_METHOD_ID}" \
  "${CHECKPOINT_PATH}" > "${contract_env}"

# shellcheck disable=SC1090
source "${contract_env}"

export S1_EVAL_CONFIG S1_EVAL_CONFIG_SHA256 S1_EVALUATION_ID
export S1_EVALUATION_CONTRACT S1_METHOD_ID S1_TOKENIZER_METHOD
export TOKENIZER_PATH TOKENIZER_MANIFEST S1_TOKENIZER_MANIFEST_SHA256
export S1_TOKENIZER_ARTIFACTS_SHA256 S1_TOKENIZER_ARM_MANIFEST_SHA256
export MODEL_ID MODEL_PATH MODEL_IDENTITY_SHA256
export BASE_MODEL_ARTIFACT_SHA256 BASE_MODEL_WEIGHT_SHARD_COUNT
export S1_CORPUS_MANIFEST_SHA256 S1_EVALUATOR_MANIFEST
export S1_EVALUATOR_MANIFEST_SHA256 S1_TEST_QUESTIONS
export S1_TEST_QUESTIONS_SHA256 S1_TEST_ANSWER_KEY
export S1_TEST_ANSWER_KEY_SHA256 S1_TEST_PANEL_ID S1_TEST_ROWS
export CHECKPOINT_PATH S1_CHECKPOINT_ARTIFACT_SHA256 S1_TRAIN_RUN_ID
export S1_EVAL_OUTPUT_ROOT
export S1_EVAL_NUM_NODES S1_EVAL_TASKS_PER_NODE S1_EVAL_WORLD_SIZE
export S1_EVAL_GPUS_PER_NODE S1_MAX_NEW_TOKENS S1_MAX_TOTAL_TOKENS
export S1_MAX_TOOL_CALL_TOKENS S1_MAX_FINAL_PROMPT_TOKENS
export S1_REASONING_EFFORT S1_HISTORY_POLICY S1_ENABLE_THINKING S1_EVAL_SEED
export S1_EVAL_BATCH_SIZE S1_SOURCE_INDEX S1_SOURCE_INDEX_SHA256
export S1_WANDB_ENTITY S1_WANDB_PROJECT
export S1_MINIMUM_VALID_TRAJECTORY S1_MINIMUM_REASONING_PRESENT
export S1_MINIMUM_TOOL_FOLLOWUP_THINKING_PRESENT
export S1_MINIMUM_EXACT_TOOL_SELECTION S1_MINIMUM_EXACT_TOOL_ARGUMENTS
export S1_MINIMUM_TOOL_PAYLOAD_CONSISTENCY
export S1_MINIMUM_FINAL_ANSWER_ACCURACY
export S1_MINIMUM_SOURCE_TRACE_ENVELOPE_ACCURACY
export S1_MINIMUM_FAMILY_ACCURACY
export S1_MAXIMUM_UNEXPECTED_DIRECT_ANSWER_RATE

if [[ -n "${S1_RESUME_SHARDS_ROOT}" ]]; then
  [[ -d "${S1_RESUME_SHARDS_ROOT}" ]] || \
    die "The resume shards root is absent: ${S1_RESUME_SHARDS_ROOT}"
  S1_RESUME_SHARDS_ROOT="$(realpath "${S1_RESUME_SHARDS_ROOT}")"
  [[ "${S1_RESUME_SHARDS_ROOT}" == "${S1_EVAL_OUTPUT_ROOT}/"* ]] || \
    die "The resume shards root must belong to this evaluation output root."
fi

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  execution_id="${SLURM_JOB_ID:-JOB-ID}"
else
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "Run evaluation as a Slurm job."
  require_positive_integer SLURM_JOB_ID "${SLURM_JOB_ID}"
  execution_id="${SLURM_JOB_ID}"
fi
S1_EVAL_RUN_ID="${S1_METHOD_ID}-test-slurm-${execution_id}"
OUTPUT_DIR="${S1_EVAL_OUTPUT_ROOT}/slurm-${execution_id}"
export S1_EVAL_RUN_ID OUTPUT_DIR S1_GENERATION_ATTEMPTS

echo "S1 evaluation: ${S1_EVAL_RUN_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Panel rows: ${S1_TEST_ROWS}"
echo "Reasoning history: ${S1_HISTORY_POLICY}"
echo "Resources: nodes=${S1_EVAL_NUM_NODES} tasks=${S1_EVAL_WORLD_SIZE}"

if [[ "${CONTRACT_ONLY}" == "0" ]]; then
  [[ "${SLURM_NNODES:-}" == "${S1_EVAL_NUM_NODES}" ]] || \
    die "The allocated node count differs from the S1 eval contract."
  [[ "${SLURM_NTASKS:-}" == "${S1_EVAL_WORLD_SIZE}" ]] || \
    die "The allocated task count differs from the S1 eval contract."
  allocated_tasks="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}"
  allocated_tasks="${allocated_tasks%%(*}"
  [[ "${allocated_tasks}" == "${S1_EVAL_TASKS_PER_NODE}" ]] || \
    die "The allocated tasks per node differ from the S1 eval contract."
fi

evaluation_code_files=(
  runtime/tools.py
  runtime/world_model_training.py
  runtime/world_model/contract.py
  runtime/world_model/source_index.py
  runtime/world_model/sources.py
  runtime/world_model/tools.py
  scripts/build_world_model_v2_s0_generation_bundle.py
  scripts/evaluate_world_model_v2_s0.py
  scripts/run_world_model_v2_s0_exact_generation.py
  scripts/validate_world_model_v2_s0_train_contract.py
  scripts/world_model_s1_eval_common.py
  scripts/build_world_model_s1_generation_bundle.py
  scripts/run_world_model_s1_exact_generation.py
  scripts/merge_world_model_s1_generation_shards.py
  scripts/evaluate_world_model_s1.py
  scripts/validate_world_model_s1_eval_contract.py
  world_model_s1_eval.slurm
)
S1_EVALUATION_CODE_SHA256="$({
  for path in "${evaluation_code_files[@]}"; do
    [[ -f "${path}" ]] || die "An S1 evaluation code file is absent: ${path}"
    printf '%s  %s\n' "$(sha256sum "${path}" | awk '{print $1}')" "${path}"
  done
} | sha256sum | awk '{print $1}')"
export S1_EVALUATION_CODE_SHA256

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  echo "The S1 evaluation config, panel, and checkpoint are valid."
  exit 0
fi

[[ ! -e "${OUTPUT_DIR}" ]] || die "The S1 eval output already exists."
umask 077
mkdir -p "${OUTPUT_DIR}/run_contract" "${OUTPUT_DIR}/wandb"
RUN_CONTRACT_DIR="${OUTPUT_DIR}/run_contract"
GENERATION_BUNDLE_DIR="${OUTPUT_DIR}/generation_bundle"
GENERATION_SHARDS_DIR="${OUTPUT_DIR}/generation_shards"
GENERATION_OUTPUT_DIR="${OUTPUT_DIR}/generation"
REPORT_OUTPUT_DIR="${OUTPUT_DIR}/report"
export RUN_CONTRACT_DIR GENERATION_BUNDLE_DIR GENERATION_SHARDS_DIR
export GENERATION_OUTPUT_DIR REPORT_OUTPUT_DIR S1_RESUME_SHARDS_ROOT

cp "${S1_EVAL_CONFIG}" "${RUN_CONTRACT_DIR}/test_config.json"
cp "${S1_EVALUATOR_MANIFEST}" \
  "${RUN_CONTRACT_DIR}/evaluator_manifest.json"
cp "${TOKENIZER_MANIFEST}" "${RUN_CONTRACT_DIR}/tokenizer_manifest.json"
cp "${CHECKPOINT_PATH}/run_contract/training_exposure.json" \
  "${RUN_CONTRACT_DIR}/training_exposure.json"
printf '%s\n' \
  "job_id=${SLURM_JOB_ID}" \
  "evaluation_id=${S1_EVALUATION_ID}" \
  "evaluation_run_id=${S1_EVAL_RUN_ID}" \
  "method_id=${S1_METHOD_ID}" \
  "train_run_id=${S1_TRAIN_RUN_ID}" \
  "checkpoint_path=${CHECKPOINT_PATH}" \
  "checkpoint_artifact_sha256=${S1_CHECKPOINT_ARTIFACT_SHA256}" \
  "test_config_sha256=${S1_EVAL_CONFIG_SHA256}" \
  "model_identity_sha256=${MODEL_IDENTITY_SHA256}" \
  "base_model_artifact_sha256=${BASE_MODEL_ARTIFACT_SHA256}" \
  "corpus_manifest_sha256=${S1_CORPUS_MANIFEST_SHA256}" \
  "evaluator_manifest_sha256=${S1_EVALUATOR_MANIFEST_SHA256}" \
  "test_panel_id=${S1_TEST_PANEL_ID}" \
  "test_questions_sha256=${S1_TEST_QUESTIONS_SHA256}" \
  "test_answer_key_sha256=${S1_TEST_ANSWER_KEY_SHA256}" \
  "source_index_sha256=${S1_SOURCE_INDEX_SHA256}" \
  "tokenizer_arm_manifest_sha256=${S1_TOKENIZER_ARM_MANIFEST_SHA256}" \
  "tokenizer_manifest_sha256=${S1_TOKENIZER_MANIFEST_SHA256}" \
  "tokenizer_artifacts_sha256=${S1_TOKENIZER_ARTIFACTS_SHA256}" \
  "evaluation_code_sha256=${S1_EVALUATION_CODE_SHA256}" \
  "history_policy=${S1_HISTORY_POLICY}" \
  "world_size=${S1_EVAL_WORLD_SIZE}" \
  "output_dir=${OUTPUT_DIR}" \
  > "${RUN_CONTRACT_DIR}/launch.env"

module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

export ROCM_SMI_DISABLE=1 AMD_SMI_DISABLE=1 RSMI_LOGGING=0
export AMDSMI_GPU_METRICS_CACHE_MS=0
export CC=cc CXX=CC CMAKE_C_COMPILER=cc CMAKE_CXX_COMPILER=CC
export ROCM_HOME=/opt/rocm-6.3.1 PYTORCH_ROCM_ARCH=gfx90a HSA_XNACK=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-56}"
export HF_HOME="/mnt/bb/${USER}/mentor_hf/${SLURM_JOB_ID}/s1-eval"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="/mnt/bb/${USER}/mentor_torch_extensions/torch280_rocm631"
export TRITON_CACHE_DIR="/mnt/bb/${USER}/mentor_triton/${SLURM_JOB_ID}/s1-eval"
ulimit -c 0

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi
unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_RESUME
export WANDB_MODE=online WANDB_ENTITY="${S1_WANDB_ENTITY}"
export WANDB_PROJECT="${S1_WANDB_PROJECT}" WANDB_DIR="${OUTPUT_DIR}/wandb"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-180}"
export http_proxy="${http_proxy:-http://proxy.ccs.ornl.gov:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"

# Keep one content-addressed copy on each node-local burst buffer.
LOCAL_MODEL_CACHE_ROOT="${LOCAL_MODEL_CACHE_ROOT:-/mnt/bb/${USER}/mentor_model_cache}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_CACHE_ROOT}/${MODEL_ID}-${BASE_MODEL_ARTIFACT_SHA256:0:16}"
BASE_MODEL_VALIDATOR="${REPO_ROOT}/scripts/validate_world_model_v2_s0_train_contract.py"
export LOCAL_MODEL_CACHE_ROOT LOCAL_MODEL_DIR BASE_MODEL_VALIDATOR
[[ "${LOCAL_MODEL_DIR}" == "/mnt/bb/${USER}/"* ]] || \
  die "The local model path must stay in the user burst buffer."

srun --nodes="${S1_EVAL_NUM_NODES}" --ntasks="${S1_EVAL_WORLD_SIZE}" \
  --ntasks-per-node=1 --cpu-bind=none bash -c '
  set -euo pipefail
  marker="${LOCAL_MODEL_DIR}/.mentor_model_identity_sha256"
  receipt="${LOCAL_MODEL_DIR}/.mentor_base_model_artifact.json"
  mkdir -p "${LOCAL_MODEL_CACHE_ROOT}"
  exec 9>"${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.stage.lock"
  flock 9
  if [[ -d "${LOCAL_MODEL_DIR}" && -s "${marker}" && \
        "$(<"${marker}")" == "${BASE_MODEL_ARTIFACT_SHA256}" ]]; then
    python "${BASE_MODEL_VALIDATOR}" --validate-base-model \
      "${LOCAL_MODEL_DIR}" "${BASE_MODEL_ARTIFACT_SHA256}" \
      "${BASE_MODEL_WEIGHT_SHARD_COUNT}" "${receipt}"
    exit 0
  fi
  if [[ -e "${LOCAL_MODEL_DIR}" ]]; then
    echo "[ERROR] An unverified model cache already occupies ${LOCAL_MODEL_DIR}" >&2
    exit 2
  fi
  staging="$(mktemp -d "${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.s1-eval.XXXXXX")"
  rsync -a "${MODEL_PATH}/" "${staging}/"
  python "${BASE_MODEL_VALIDATOR}" --validate-base-model \
    "${staging}" "${BASE_MODEL_ARTIFACT_SHA256}" \
    "${BASE_MODEL_WEIGHT_SHARD_COUNT}" \
    "${staging}/.mentor_base_model_artifact.json"
  printf "%s\n" "${BASE_MODEL_ARTIFACT_SHA256}" \
    > "${staging}/.mentor_model_identity_sha256"
  mv "${staging}" "${LOCAL_MODEL_DIR}"
'

python scripts/build_world_model_s1_generation_bundle.py \
  --evaluator-manifest "${S1_EVALUATOR_MANIFEST}" \
  --evaluator-manifest-sha256 "${S1_EVALUATOR_MANIFEST_SHA256}" \
  --source-index "${S1_SOURCE_INDEX}" \
  --source-index-sha256 "${S1_SOURCE_INDEX_SHA256}" \
  --output-dir "${GENERATION_BUNDLE_DIR}"

S1_GENERATION_BUNDLE_SHA256="$(
  python - "${GENERATION_BUNDLE_DIR}/manifest.json" <<'PY'
import sys
from pathlib import Path

from scripts.world_model_s1_eval_common import (
    read_json,
    require_sha256,
    stable_sha256,
)

bundle = read_json(Path(sys.argv[1]))
claimed = require_sha256(bundle.get("bundle_sha256"), "S1 generation bundle")
identity = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
if stable_sha256(identity) != claimed:
    raise SystemExit("The S1 generation bundle failed its identity check.")
print(claimed)
PY
)"
export S1_GENERATION_BUNDLE_SHA256
printf '%s\n' \
  "generation_bundle_sha256=${S1_GENERATION_BUNDLE_SHA256}" \
  >> "${RUN_CONTRACT_DIR}/launch.env"

mkdir -p "${GENERATION_SHARDS_DIR}"
srun --label --kill-on-bad-exit=0 \
  --nodes="${S1_EVAL_NUM_NODES}" --ntasks="${S1_EVAL_WORLD_SIZE}" \
  --ntasks-per-node=1 --gpus-per-task="${S1_EVAL_GPUS_PER_NODE}" \
  --cpu-bind=none --accel-bind=none bash -c '
  set -euo pipefail
  shard_name="$(printf "shard-%05d" "${SLURM_PROCID}")"
  shard_dir="${GENERATION_SHARDS_DIR}/${shard_name}"
  for ((attempt = 1; attempt <= S1_GENERATION_ATTEMPTS; attempt++)); do
    resume_args=()
    if (( attempt > 1 )) && [[ -d "${shard_dir}" ]]; then
      resume_args+=(--resume-shard-dir "${shard_dir}")
    elif [[ -n "${S1_RESUME_SHARDS_ROOT}" && \
            -d "${S1_RESUME_SHARDS_ROOT}/${shard_name}" ]]; then
      resume_args+=(
        --resume-shard-dir "${S1_RESUME_SHARDS_ROOT}/${shard_name}"
      )
    fi
    if env \
      -u WANDB_API_KEY \
      -u HF_TOKEN \
      -u HF_HUB_TOKEN \
      -u HUGGING_FACE_HUB_TOKEN \
      -u OPENAI_API_KEY \
      -u ANTHROPIC_API_KEY \
      -u AWS_ACCESS_KEY_ID \
      -u AWS_SECRET_ACCESS_KEY \
      -u AWS_SESSION_TOKEN \
      -u S1_TEST_ANSWER_KEY \
      -u S1_TEST_ANSWER_KEY_SHA256 \
      -u S1_EVALUATOR_MANIFEST \
      -u http_proxy \
      -u https_proxy \
      -u all_proxy \
      -u no_proxy \
      -u HTTP_PROXY \
      -u HTTPS_PROXY \
      -u ALL_PROXY \
      -u NO_PROXY \
      WANDB_MODE=disabled \
      python scripts/run_world_model_s1_exact_generation.py \
        --generation-bundle "${GENERATION_BUNDLE_DIR}" \
        --base-model-path "${LOCAL_MODEL_DIR}" \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --tokenizer-manifest "${TOKENIZER_MANIFEST}" \
        --output-dir "${shard_dir}" \
        --method-id "${S1_METHOD_ID}" \
        --corpus-manifest-sha256 "${S1_CORPUS_MANIFEST_SHA256}" \
        --tokenizer-manifest-sha256 "${S1_TOKENIZER_MANIFEST_SHA256}" \
        --model-identity-sha256 "${MODEL_IDENTITY_SHA256}" \
        --base-model-artifact-sha256 "${BASE_MODEL_ARTIFACT_SHA256}" \
        --max-new-tokens "${S1_MAX_NEW_TOKENS}" \
        --max-total-tokens "${S1_MAX_TOTAL_TOKENS}" \
        --max-tool-call-tokens "${S1_MAX_TOOL_CALL_TOKENS}" \
        --max-final-prompt-tokens "${S1_MAX_FINAL_PROMPT_TOKENS}" \
        --enable-thinking \
        --reasoning-effort "${S1_REASONING_EFFORT}" \
        --history-policy "${S1_HISTORY_POLICY}" \
        --seed "${S1_EVAL_SEED}" \
        --batch-size "${S1_EVAL_BATCH_SIZE}" \
        --shard-index "${SLURM_PROCID}" \
        --shard-count "${S1_EVAL_WORLD_SIZE}" \
        --shard-output \
        "${resume_args[@]}"; then
      exit 0
    else
      status=$?
    fi
    (( attempt < S1_GENERATION_ATTEMPTS )) || exit "${status}"
  done
'

env -u WANDB_API_KEY -u S1_TEST_ANSWER_KEY -u S1_TEST_ANSWER_KEY_SHA256 \
  python scripts/merge_world_model_s1_generation_shards.py \
    --generation-bundle "${GENERATION_BUNDLE_DIR}" \
    --shards-root "${GENERATION_SHARDS_DIR}" \
    --output-dir "${GENERATION_OUTPUT_DIR}" \
    --expected-shard-count "${S1_EVAL_WORLD_SIZE}"

publish_args=()
[[ "${PUBLISH_WANDB}" != "1" ]] || publish_args+=(--publish-wandb)
python scripts/evaluate_world_model_s1.py \
  --evaluator-manifest "${S1_EVALUATOR_MANIFEST}" \
  --evaluator-manifest-sha256 "${S1_EVALUATOR_MANIFEST_SHA256}" \
  --predictions "${GENERATION_OUTPUT_DIR}/predictions.jsonl" \
  --generation-manifest "${GENERATION_OUTPUT_DIR}/generation_manifest.json" \
  --tokenizer-manifest "${TOKENIZER_MANIFEST}" \
  --output-dir "${REPORT_OUTPUT_DIR}" \
  --expected-train-run-id "${S1_TRAIN_RUN_ID}" \
  --expected-method-id "${S1_METHOD_ID}" \
  --expected-checkpoint-identity-sha256 \
    "${S1_CHECKPOINT_ARTIFACT_SHA256}" \
  --expected-base-model-identity-sha256 "${MODEL_IDENTITY_SHA256}" \
  --expected-base-model-artifact-sha256 "${BASE_MODEL_ARTIFACT_SHA256}" \
  --expected-tokenizer-manifest-sha256 \
    "${S1_TOKENIZER_MANIFEST_SHA256}" \
  --expected-tokenizer-artifacts-sha256 \
    "${S1_TOKENIZER_ARTIFACTS_SHA256}" \
  --expected-source-index-sha256 "${S1_SOURCE_INDEX_SHA256}" \
  --expected-generation-bundle-sha256 \
    "${S1_GENERATION_BUNDLE_SHA256}" \
  --minimum-valid-trajectory "${S1_MINIMUM_VALID_TRAJECTORY}" \
  --minimum-reasoning-present "${S1_MINIMUM_REASONING_PRESENT}" \
  --minimum-tool-followup-thinking-present \
    "${S1_MINIMUM_TOOL_FOLLOWUP_THINKING_PRESENT}" \
  --minimum-exact-tool-selection "${S1_MINIMUM_EXACT_TOOL_SELECTION}" \
  --minimum-exact-tool-arguments "${S1_MINIMUM_EXACT_TOOL_ARGUMENTS}" \
  --minimum-tool-payload-consistency \
    "${S1_MINIMUM_TOOL_PAYLOAD_CONSISTENCY}" \
  --minimum-final-answer-accuracy "${S1_MINIMUM_FINAL_ANSWER_ACCURACY}" \
  --minimum-source-trace-envelope-accuracy \
    "${S1_MINIMUM_SOURCE_TRACE_ENVELOPE_ACCURACY}" \
  --minimum-family-accuracy "${S1_MINIMUM_FAMILY_ACCURACY}" \
  --maximum-unexpected-direct-answer-rate \
    "${S1_MAXIMUM_UNEXPECTED_DIRECT_ANSWER_RATE}" \
  --gate-profile "${S1_GATE_PROFILE}" \
  --minimum-strict-rate "${S1_MINIMUM_STRICT_RATE:-1}" \
  --minimum-target-rate "${S1_MINIMUM_TARGET_RATE:-1}" \
  "${publish_args[@]}"

completed_files=(
  "${GENERATION_BUNDLE_DIR}/manifest.json"
  "${GENERATION_BUNDLE_DIR}/questions.jsonl"
  "${GENERATION_OUTPUT_DIR}/generation_plan.json"
  "${GENERATION_OUTPUT_DIR}/predictions.jsonl"
  "${GENERATION_OUTPUT_DIR}/generation_manifest.json"
  "${REPORT_OUTPUT_DIR}/exact_s1_report.json"
  "${REPORT_OUTPUT_DIR}/test_metrics.json"
)
for completed in "${completed_files[@]}"; do
  [[ -s "${completed}" ]] || die "The completed S1 eval lacks ${completed}."
done
if [[ "${PUBLISH_WANDB}" == "1" ]]; then
  [[ -s "${REPORT_OUTPUT_DIR}/wandb_test_run.json" ]] || \
    die "The completed S1 eval lacks its W&B receipt."
fi
promotion_gate="$(python - "${REPORT_OUTPUT_DIR}/test_metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    metrics = json.load(source)
passed = metrics.get("gate", {}).get("passed")
if not isinstance(passed, bool):
    raise SystemExit("The S1 metrics lack a Boolean promotion gate result.")
print("PASS" if passed else "FAIL")
PY
)"
echo "S1 promotion gate: ${promotion_gate}"
echo "The exact native S1 evaluation artifacts are complete: ${OUTPUT_DIR}; promotion gate: ${promotion_gate}"
