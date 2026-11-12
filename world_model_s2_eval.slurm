#!/bin/bash
# Evaluate the completed S2 final checkpoint on the frozen structured panel.
#
# One-node real-checkpoint preflight:
#   sbatch -N1 -p batch -t 02:00:00 \
#     --export=ALL,PREFLIGHT_ONLY=1 world_model_s2_eval.slurm
# Full evaluation (normally submitted afterok on the preflight):
#   sbatch world_model_s2_eval.slurm
#
# The full run generates 512 shards on 512 nodes. It requests eight spare
# nodes: every node stages the model, a node that cannot stage a verified
# copy is left out, and a shard that fails on its node is rerun once on a
# spare or finished node. Job 5892323 measured shards of up to 3.1 hours,
# so eight hours covers staging, the slowest shard, and one rerun.

#SBATCH --job-name=wm_s2_eval
#SBATCH -A SYB114
#SBATCH -N 520
#SBATCH -t 08:00:00
#SBATCH -p batch
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
[[ -f "${REPO_ROOT}/world_model_s2_eval.slurm" ]] || \
  die "Submit world_model_s2_eval.slurm from the repository root."
export REPO_ROOT PYTHONDONTWRITEBYTECODE=1

RUN_CONFIG="${RUN_CONFIG:-config/world_model_s2_120b_test_v1_512n.json}"
S2_METHOD_ID="${S2_METHOD_ID:-oss120b-plain-base-tokenizer-lora-r32}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-data/world_model_v4/checkpoints/s2_v1_184n/oss120b-plain-base-tokenizer-lora-r32/slurm-5856263}"
CONTRACT_ONLY="${CONTRACT_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PUBLISH_WANDB="${PUBLISH_WANDB:-1}"
S2_GENERATION_ATTEMPTS="${S2_GENERATION_ATTEMPTS:-2}"
S2_RESUME_SHARDS_ROOT="${S2_RESUME_SHARDS_ROOT:-}"
S2_PYTHON="${S2_PYTHON:-/lustre/orion/syb111/world-shared/environments/pytorch-rocm/bin/python}"
# Only the repair pass assigns a shard explicitly; normal tasks use SLURM_PROCID.
unset S2_SHARD_INDEX

[[ "${CONTRACT_ONLY}" =~ ^[01]$ ]] || \
  die "CONTRACT_ONLY must be zero or one."
[[ "${PREFLIGHT_ONLY}" =~ ^[01]$ ]] || \
  die "PREFLIGHT_ONLY must be zero or one."
[[ "${PUBLISH_WANDB}" =~ ^[01]$ ]] || \
  die "PUBLISH_WANDB must be zero or one."
[[ "${CONTRACT_ONLY}${PREFLIGHT_ONLY}" != "11" ]] || \
  die "CONTRACT_ONLY and PREFLIGHT_ONLY are mutually exclusive."
require_positive_integer S2_GENERATION_ATTEMPTS "${S2_GENERATION_ATTEMPTS}"
[[ -x "${S2_PYTHON}" ]] || die "The pinned Python environment is absent."

contract_env="$(mktemp "${TMPDIR:-/tmp}/wm-s2-eval-contract.XXXXXX")"
cleanup_contract() {
  rm -f -- "${contract_env}"
}
trap cleanup_contract EXIT

"${S2_PYTHON}" scripts/validate_world_model_s2_eval_contract.py \
  "${REPO_ROOT}" "${RUN_CONFIG}" "${S2_METHOD_ID}" \
  "${CHECKPOINT_PATH}" > "${contract_env}"

# shellcheck disable=SC1090
source "${contract_env}"

export S2_EVAL_CONFIG S2_EVAL_CONFIG_SHA256 S2_EVALUATION_ID
export S2_EVALUATION_CONTRACT S2_METHOD_ID S2_TOKENIZER_METHOD
export TOKENIZER_PATH TOKENIZER_MANIFEST S2_TOKENIZER_MANIFEST_SHA256
export S2_TOKENIZER_ARTIFACTS_SHA256 S2_TOKENIZER_ARM_MANIFEST_SHA256
export MODEL_ID MODEL_PATH MODEL_IDENTITY_SHA256
export BASE_MODEL_ARTIFACT_SHA256 BASE_MODEL_WEIGHT_SHARD_COUNT
export S2_CORPUS_MANIFEST_SHA256 S2_RUNTIME_CONTRACT_SHA256
export S2_EVALUATOR_MANIFEST
export S2_EVALUATOR_MANIFEST_SHA256 S2_TEST_QUESTIONS
export S2_TEST_QUESTIONS_SHA256 S2_TEST_ANSWER_KEY
export S2_TEST_ANSWER_KEY_SHA256 S2_TEST_PANEL_ID S2_TEST_ROWS
export CHECKPOINT_PATH S2_CHECKPOINT_ARTIFACT_SHA256 S2_TRAIN_RUN_ID
export S2_EVAL_OUTPUT_ROOT
export S2_EVAL_NUM_NODES S2_EVAL_TASKS_PER_NODE S2_EVAL_WORLD_SIZE
export S2_EVAL_GPUS_PER_NODE S2_MAX_NEW_TOKENS S2_MAX_TOTAL_TOKENS
export S2_MAX_TOOL_CALL_TOKENS S2_MAX_FINAL_PROMPT_TOKENS
export S2_MAXIMUM_TOOL_CALLS S2_REASONING_EFFORT S2_HISTORY_POLICY
export S2_ENABLE_THINKING S2_EVAL_SEED S2_EVAL_BATCH_SIZE
export S2_GATE_PROFILE S2_WANDB_ENTITY S2_WANDB_PROJECT

if [[ -n "${S2_RESUME_SHARDS_ROOT}" ]]; then
  [[ "${PREFLIGHT_ONLY}" == "0" ]] || \
    die "PREFLIGHT_ONLY does not accept S2_RESUME_SHARDS_ROOT."
  [[ -d "${S2_RESUME_SHARDS_ROOT}" ]] || \
    die "The resume shards root is absent: ${S2_RESUME_SHARDS_ROOT}"
  S2_RESUME_SHARDS_ROOT="$(realpath "${S2_RESUME_SHARDS_ROOT}")"
  [[ "${S2_RESUME_SHARDS_ROOT}" == "${S2_EVAL_OUTPUT_ROOT}/"* ]] || \
    die "The resume shards root must belong to this evaluation output root."
fi

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  execution_id="${SLURM_JOB_ID:-JOB-ID}"
else
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "Run evaluation as a Slurm job."
  require_positive_integer SLURM_JOB_ID "${SLURM_JOB_ID}"
  execution_id="${SLURM_JOB_ID}"
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  EVAL_ALLOCATION_NODES=1
  EVAL_ALLOCATION_TASKS=1
  S2_EVAL_RUN_ID="${S2_METHOD_ID}-s2-preflight-slurm-${execution_id}"
  OUTPUT_DIR="${S2_EVAL_OUTPUT_ROOT}/preflight-slurm-${execution_id}"
  PUBLISH_WANDB=0
else
  EVAL_ALLOCATION_NODES="${S2_EVAL_NUM_NODES}"
  EVAL_ALLOCATION_TASKS="${S2_EVAL_WORLD_SIZE}"
  S2_EVAL_RUN_ID="${S2_METHOD_ID}-s2-test-slurm-${execution_id}"
  OUTPUT_DIR="${S2_EVAL_OUTPUT_ROOT}/slurm-${execution_id}"
fi
export EVAL_ALLOCATION_NODES EVAL_ALLOCATION_TASKS
export S2_EVAL_RUN_ID OUTPUT_DIR S2_GENERATION_ATTEMPTS

echo "S2 evaluation: ${S2_EVAL_RUN_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Panel rows: ${S2_TEST_ROWS}"
echo "Mode: $([[ "${PREFLIGHT_ONLY}" == "1" ]] && echo preflight || echo full)"
echo "Resources: nodes=${EVAL_ALLOCATION_NODES} tasks=${EVAL_ALLOCATION_TASKS}"

if [[ "${CONTRACT_ONLY}" == "0" ]]; then
  require_positive_integer SLURM_NNODES "${SLURM_NNODES:-}"
  (( SLURM_NNODES >= EVAL_ALLOCATION_NODES )) || \
    die "This S2 eval mode needs at least ${EVAL_ALLOCATION_NODES} nodes."
  [[ "${SLURM_NTASKS:-}" == "${SLURM_NNODES}" ]] || \
    die "The S2 eval allocation must have one task per node."
  allocated_tasks="${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}"
  allocated_tasks="${allocated_tasks%%(*}"
  [[ "${allocated_tasks}" == "${S2_EVAL_TASKS_PER_NODE}" ]] || \
    die "The allocated tasks per node differ from the S2 eval contract."
  echo "Spare nodes: $(( SLURM_NNODES - EVAL_ALLOCATION_NODES ))"
fi

evaluation_code_files=(
  runtime/world_model_training.py
  runtime/world_model/contract.py
  runtime/world_model/contract.yaml
  runtime/world_model/sources.py
  runtime/world_model/s2_3_tokens.py
  runtime/world_model/s2_production_adapter.py
  scripts/build_world_model_s2_eval_bundle.py
  scripts/build_world_model_s2_generation_bundle.py
  scripts/evaluate_world_model_s2.py
  scripts/merge_world_model_s2_generation_shards.py
  scripts/run_world_model_s2_exact_generation.py
  scripts/run_world_model_s1_exact_generation.py
  scripts/run_world_model_v2_s0_exact_generation.py
  scripts/validate_world_model_s2_eval_contract.py
  scripts/validate_world_model_v2_s0_train_contract.py
  scripts/world_model_s1_eval_common.py
  world_model_s2_eval.slurm
)
S2_EVALUATION_CODE_SHA256="$({
  for path in "${evaluation_code_files[@]}"; do
    [[ -f "${path}" ]] || die "An S2 evaluation code file is absent: ${path}"
    printf '%s  %s\n' "$(sha256sum "${path}" | awk '{print $1}')" "${path}"
  done
} | sha256sum | awk '{print $1}')"
export S2_EVALUATION_CODE_SHA256

if [[ "${CONTRACT_ONLY}" == "1" ]]; then
  echo "The S2 evaluation config, panel, and completed checkpoint are valid."
  exit 0
fi

# One task generates one shard. It resumes its own checkpointed rows, or
# the same shard under S2_RESUME_SHARDS_ROOT. A task that still fails marks
# its node, so the repair pass reruns the shard on a different node.
RUN_SHARD_SCRIPT='
  set -euo pipefail
  shard_index="${S2_SHARD_INDEX:-${SLURM_PROCID}}"
  shard_name="$(printf "shard-%05d" "${shard_index}")"
  shard_dir="${GENERATION_SHARDS_DIR}/${shard_name}"
  status=1
  for ((attempt = 1; attempt <= S2_GENERATION_ATTEMPTS; attempt++)); do
    resume_args=()
    if [[ -s "${shard_dir}/partial_generation_manifest.json" ]]; then
      resume_args+=(--resume-shard-dir "${shard_dir}")
    elif [[ -n "${S2_RESUME_SHARDS_ROOT}" && \
            -d "${S2_RESUME_SHARDS_ROOT}/${shard_name}" ]]; then
      resume_args+=(
        --resume-shard-dir "${S2_RESUME_SHARDS_ROOT}/${shard_name}"
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
      -u S2_TEST_ANSWER_KEY \
      -u S2_TEST_ANSWER_KEY_SHA256 \
      -u S2_EVALUATOR_MANIFEST \
      -u http_proxy \
      -u https_proxy \
      -u all_proxy \
      -u no_proxy \
      -u HTTP_PROXY \
      -u HTTPS_PROXY \
      -u ALL_PROXY \
      -u NO_PROXY \
      WANDB_MODE=disabled \
      python scripts/run_world_model_s2_exact_generation.py \
        --generation-bundle "${GENERATION_BUNDLE_DIR}" \
        --base-model-path "${LOCAL_MODEL_DIR}" \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --tokenizer-manifest "${TOKENIZER_MANIFEST}" \
        --output-dir "${shard_dir}" \
        --method-id "${S2_METHOD_ID}" \
        --corpus-manifest-sha256 "${S2_CORPUS_MANIFEST_SHA256}" \
        --tokenizer-manifest-sha256 "${S2_TOKENIZER_MANIFEST_SHA256}" \
        --model-identity-sha256 "${MODEL_IDENTITY_SHA256}" \
        --base-model-artifact-sha256 "${BASE_MODEL_ARTIFACT_SHA256}" \
        --max-new-tokens "${S2_MAX_NEW_TOKENS}" \
        --max-total-tokens "${S2_MAX_TOTAL_TOKENS}" \
        --max-tool-call-tokens "${S2_MAX_TOOL_CALL_TOKENS}" \
        --max-final-prompt-tokens "${S2_MAX_FINAL_PROMPT_TOKENS}" \
        --enable-thinking \
        --reasoning-effort "${S2_REASONING_EFFORT}" \
        --history-policy "${S2_HISTORY_POLICY}" \
        --seed "${S2_EVAL_SEED}" \
        --batch-size "${S2_EVAL_BATCH_SIZE}" \
        --shard-index "${shard_index}" \
        --shard-count "${S2_EVAL_WORLD_SIZE}" \
        --shard-output \
        "${resume_args[@]}"; then
      exit 0
    else
      status=$?
    fi
  done
  touch "${FAILED_NODES_DIR}/${SLURMD_NODENAME}"
  exit "${status}"
'

# Print the index of every shard without a completed generation manifest.
incomplete_shards() {
  local index
  for ((index = 0; index < S2_EVAL_WORLD_SIZE; index++)); do
    [[ -s "${GENERATION_SHARDS_DIR}/$(printf "shard-%05d" "${index}")/generation_manifest.json" ]] || \
      echo "${index}"
  done
}

# Rerun each incomplete shard once, in parallel, on a staged node that has
# not failed a shard. Spare nodes come first. A shard that fails again
# stops the run; its checkpointed rows stay available for resubmission.
repair_failed_shards() {
  local missing=() nodes=() pids=() node index pid
  mapfile -t missing < <(incomplete_shards)
  (( ${#missing[@]} )) || return 0
  for node in "${STAGED_NODES[@]:EVAL_ALLOCATION_NODES}" \
              "${STAGED_NODES[@]:0:EVAL_ALLOCATION_NODES}"; do
    [[ -e "${FAILED_NODES_DIR}/${node}" ]] || nodes+=("${node}")
  done
  echo "Repairing ${#missing[@]} S2 shards: ${missing[*]}"
  (( ${#missing[@]} <= ${#nodes[@]} )) || \
    die "Too few healthy nodes remain to repair ${#missing[@]} S2 shards."
  for index in "${!missing[@]}"; do
    S2_SHARD_INDEX="${missing[index]}" S2_GENERATION_ATTEMPTS=1 \
      srun --label --kill-on-bad-exit=0 --nodes=1 --ntasks=1 \
        --nodelist="${nodes[index]}" --ntasks-per-node=1 \
        --gpus-per-task="${S2_EVAL_GPUS_PER_NODE}" \
        --cpu-bind=none --accel-bind=none bash -c "${RUN_SHARD_SCRIPT}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
  mapfile -t missing < <(incomplete_shards)
  (( ${#missing[@]} == 0 )) || die "S2 shards ${missing[*]} failed again." \
    "Resubmit with S2_RESUME_SHARDS_ROOT=${GENERATION_SHARDS_DIR}."
}

run_full_evaluation() {
srun --label --kill-on-bad-exit=0 \
  --nodes="${S2_EVAL_NUM_NODES}" --ntasks="${S2_EVAL_WORLD_SIZE}" \
  --nodelist="${GENERATION_NODES_FILE}" \
  --ntasks-per-node=1 --gpus-per-task="${S2_EVAL_GPUS_PER_NODE}" \
  --cpu-bind=none --accel-bind=none bash -c "${RUN_SHARD_SCRIPT}" || \
  echo "[WARN] Some S2 shards failed; the repair pass reruns them." >&2
repair_failed_shards

env -u WANDB_API_KEY -u S2_TEST_ANSWER_KEY -u S2_TEST_ANSWER_KEY_SHA256 \
  python scripts/merge_world_model_s2_generation_shards.py \
    --generation-bundle "${GENERATION_BUNDLE_DIR}" \
    --shards-root "${GENERATION_SHARDS_DIR}" \
    --output-dir "${GENERATION_OUTPUT_DIR}" \
    --expected-shard-count "${S2_EVAL_WORLD_SIZE}"

publish_args=()
[[ "${PUBLISH_WANDB}" != "1" ]] || publish_args+=(--publish-wandb)
python scripts/evaluate_world_model_s2.py \
  --evaluator-manifest "${S2_EVALUATOR_MANIFEST}" \
  --evaluator-manifest-sha256 "${S2_EVALUATOR_MANIFEST_SHA256}" \
  --predictions "${GENERATION_OUTPUT_DIR}/predictions.jsonl" \
  --generation-manifest "${GENERATION_OUTPUT_DIR}/generation_manifest.json" \
  --output-dir "${REPORT_OUTPUT_DIR}" \
  --gate-profile "${S2_GATE_PROFILE}" \
  --expected-train-run-id "${S2_TRAIN_RUN_ID}" \
  --expected-method-id "${S2_METHOD_ID}" \
  --expected-checkpoint-identity-sha256 \
    "${S2_CHECKPOINT_ARTIFACT_SHA256}" \
  --expected-base-model-identity-sha256 "${MODEL_IDENTITY_SHA256}" \
  --expected-base-model-artifact-sha256 "${BASE_MODEL_ARTIFACT_SHA256}" \
  --expected-tokenizer-manifest-sha256 \
    "${S2_TOKENIZER_MANIFEST_SHA256}" \
  --expected-tokenizer-artifacts-sha256 \
    "${S2_TOKENIZER_ARTIFACTS_SHA256}" \
  --expected-generation-bundle-sha256 \
    "${S2_GENERATION_BUNDLE_SHA256}" \
  "${publish_args[@]}"

completed_files=(
  "${GENERATION_BUNDLE_DIR}/manifest.json"
  "${GENERATION_BUNDLE_DIR}/questions.jsonl"
  "${GENERATION_BUNDLE_DIR}/tool_fixtures.jsonl"
  "${GENERATION_OUTPUT_DIR}/generation_plan.json"
  "${GENERATION_OUTPUT_DIR}/predictions.jsonl"
  "${GENERATION_OUTPUT_DIR}/generation_manifest.json"
  "${REPORT_OUTPUT_DIR}/report.json"
  "${REPORT_OUTPUT_DIR}/metrics.json"
)
for completed in "${completed_files[@]}"; do
  [[ -s "${completed}" ]] || die "The completed S2 eval lacks ${completed}."
done
if [[ "${PUBLISH_WANDB}" == "1" ]]; then
  [[ -s "${REPORT_OUTPUT_DIR}/wandb_receipt.json" ]] || \
    die "The completed S2 eval lacks its W&B receipt."
fi

headline="$(python - "${REPORT_OUTPUT_DIR}/metrics.json" "${S2_TEST_ROWS}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    metrics = json.load(source)
gate = metrics.get("gate", {})
overall = metrics.get("overall", {})
if (metrics.get("complete_coverage") is not True
        or overall.get("row_count") != int(sys.argv[2])
        or gate.get("profile") != "report_only_v1"
        or gate.get("passed") is not None):
    raise SystemExit("The S2 report-only completion contract failed.")
print(overall.get("structured_accuracy_rate"))
PY
)"
echo "S2 structured accuracy: ${headline}"
echo "The complete report-only S2 evaluation is at ${OUTPUT_DIR}"
}

[[ ! -e "${OUTPUT_DIR}" ]] || die "The S2 eval output already exists."
umask 077
mkdir -p "${OUTPUT_DIR}/run_contract"
if [[ "${PUBLISH_WANDB}" == "1" ]]; then
  mkdir -p "${OUTPUT_DIR}/wandb"
fi
RUN_CONTRACT_DIR="${OUTPUT_DIR}/run_contract"
GENERATION_BUNDLE_DIR="${OUTPUT_DIR}/generation_bundle"
GENERATION_SHARDS_DIR="${OUTPUT_DIR}/generation_shards"
GENERATION_OUTPUT_DIR="${OUTPUT_DIR}/generation"
REPORT_OUTPUT_DIR="${OUTPUT_DIR}/report"
export RUN_CONTRACT_DIR GENERATION_BUNDLE_DIR GENERATION_SHARDS_DIR
export GENERATION_OUTPUT_DIR REPORT_OUTPUT_DIR S2_RESUME_SHARDS_ROOT

cp "${S2_EVAL_CONFIG}" "${RUN_CONTRACT_DIR}/test_config.json"
cp "${S2_EVALUATOR_MANIFEST}" \
  "${RUN_CONTRACT_DIR}/evaluator_manifest.json"
cp "${TOKENIZER_MANIFEST}" "${RUN_CONTRACT_DIR}/tokenizer_manifest.json"
cp "${CHECKPOINT_PATH}/run_contract/training_exposure.json" \
  "${RUN_CONTRACT_DIR}/training_exposure.json"
printf '%s\n' \
  "job_id=${SLURM_JOB_ID}" \
  "mode=$([[ "${PREFLIGHT_ONLY}" == "1" ]] && echo preflight || echo full)" \
  "evaluation_id=${S2_EVALUATION_ID}" \
  "evaluation_run_id=${S2_EVAL_RUN_ID}" \
  "method_id=${S2_METHOD_ID}" \
  "train_run_id=${S2_TRAIN_RUN_ID}" \
  "checkpoint_path=${CHECKPOINT_PATH}" \
  "checkpoint_artifact_sha256=${S2_CHECKPOINT_ARTIFACT_SHA256}" \
  "test_config_sha256=${S2_EVAL_CONFIG_SHA256}" \
  "model_identity_sha256=${MODEL_IDENTITY_SHA256}" \
  "base_model_artifact_sha256=${BASE_MODEL_ARTIFACT_SHA256}" \
  "corpus_manifest_sha256=${S2_CORPUS_MANIFEST_SHA256}" \
  "runtime_contract_sha256=${S2_RUNTIME_CONTRACT_SHA256}" \
  "evaluator_manifest_sha256=${S2_EVALUATOR_MANIFEST_SHA256}" \
  "test_panel_id=${S2_TEST_PANEL_ID}" \
  "test_questions_sha256=${S2_TEST_QUESTIONS_SHA256}" \
  "test_answer_key_sha256=${S2_TEST_ANSWER_KEY_SHA256}" \
  "tokenizer_arm_manifest_sha256=${S2_TOKENIZER_ARM_MANIFEST_SHA256}" \
  "tokenizer_manifest_sha256=${S2_TOKENIZER_MANIFEST_SHA256}" \
  "tokenizer_artifacts_sha256=${S2_TOKENIZER_ARTIFACTS_SHA256}" \
  "evaluation_code_sha256=${S2_EVALUATION_CODE_SHA256}" \
  "world_size=${EVAL_ALLOCATION_TASKS}" \
  "allocated_nodes=${SLURM_NNODES}" \
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
export HF_HOME="/mnt/bb/${USER}/mentor_hf/${SLURM_JOB_ID}/s2-eval"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="/mnt/bb/${USER}/mentor_torch_extensions/torch280_rocm631"
export TRITON_CACHE_DIR="/mnt/bb/${USER}/mentor_triton/${SLURM_JOB_ID}/s2-eval"
ulimit -c 0

unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_RESUME
if [[ "${PUBLISH_WANDB}" == "1" ]]; then
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
  [[ -n "${WANDB_API_KEY:-}" ]] || \
    die "WANDB_API_KEY is required before the full 512-node evaluation."
  unset WANDB_RUN_ID WANDB_NAME WANDB_RUN_GROUP WANDB_RESUME
  export WANDB_MODE=online WANDB_ENTITY="${S2_WANDB_ENTITY}"
  export WANDB_PROJECT="${S2_WANDB_PROJECT}" WANDB_DIR="${OUTPUT_DIR}/wandb"
  export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-180}"
  export http_proxy="${http_proxy:-http://proxy.ccs.ornl.gov:3128}"
  export https_proxy="${https_proxy:-${http_proxy}}"
else
  unset WANDB_API_KEY
  export WANDB_MODE=disabled
fi

# Keep one content-addressed model copy in every node-local burst buffer.
# This single allocation-wide srun stages all nodes, spares included,
# concurrently. Frontier kills a whole step when one task fails, so the step
# opts out: a node that cannot stage a verified copy in two attempts is
# left out, and the run continues if enough nodes remain. Job 5867865 lost
# all 512 nodes to one node's read error while hashing its copy.
LOCAL_MODEL_CACHE_ROOT="${LOCAL_MODEL_CACHE_ROOT:-/mnt/bb/${USER}/mentor_model_cache}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_CACHE_ROOT}/${MODEL_ID}-${BASE_MODEL_ARTIFACT_SHA256:0:16}"
BASE_MODEL_VALIDATOR="${REPO_ROOT}/scripts/validate_world_model_v2_s0_train_contract.py"
STAGED_NODES_DIR="${RUN_CONTRACT_DIR}/staged_nodes"
FAILED_NODES_DIR="${RUN_CONTRACT_DIR}/failed_generation_nodes"
GENERATION_NODES_FILE="${RUN_CONTRACT_DIR}/generation_nodes"
export LOCAL_MODEL_CACHE_ROOT LOCAL_MODEL_DIR BASE_MODEL_VALIDATOR
export STAGED_NODES_DIR FAILED_NODES_DIR
[[ "${LOCAL_MODEL_DIR}" == "/mnt/bb/${USER}/"* ]] || \
  die "The local model path must stay in the user burst buffer."
mkdir -p "${STAGED_NODES_DIR}" "${FAILED_NODES_DIR}"

srun --kill-on-bad-exit=0 --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" \
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
  else
    if [[ -e "${LOCAL_MODEL_DIR}" ]]; then
      echo "[ERROR] An unverified model cache occupies ${LOCAL_MODEL_DIR}" >&2
      exit 2
    fi
    staging=""
    cleanup_staging() {
      [[ -z "${staging}" || ! -e "${staging}" ]] || rm -rf -- "${staging}"
    }
    trap cleanup_staging EXIT
    for attempt in 1 2; do
      staging="$(mktemp -d \
        "${LOCAL_MODEL_CACHE_ROOT}/.${MODEL_ID}.s2-eval.${SLURM_JOB_ID}.XXXXXX")"
      if rsync -a "${MODEL_PATH}/" "${staging}/" && \
         python "${BASE_MODEL_VALIDATOR}" --validate-base-model \
           "${staging}" "${BASE_MODEL_ARTIFACT_SHA256}" \
           "${BASE_MODEL_WEIGHT_SHARD_COUNT}" \
           "${staging}/.mentor_base_model_artifact.json"; then
        break
      fi
      cleanup_staging
      staging=""
      echo "[WARN] Model staging attempt ${attempt} failed on ${SLURMD_NODENAME}." >&2
      (( attempt < 2 )) || exit 1
    done
    printf "%s\n" "${BASE_MODEL_ARTIFACT_SHA256}" \
      > "${staging}/.mentor_model_identity_sha256"
    mv "${staging}" "${LOCAL_MODEL_DIR}"
    staging=""
  fi
  touch "${STAGED_NODES_DIR}/${SLURMD_NODENAME}"
' || echo "[WARN] Model staging failed on some nodes; they are left out." >&2

mapfile -t STAGED_NODES < <(
  find "${STAGED_NODES_DIR}" -mindepth 1 -maxdepth 1 -printf "%f\n" | sort
)
echo "Staged nodes: ${#STAGED_NODES[@]} of ${SLURM_NNODES}"
(( ${#STAGED_NODES[@]} >= EVAL_ALLOCATION_NODES )) || \
  die "Only ${#STAGED_NODES[@]} nodes staged the base model;" \
    "this mode needs ${EVAL_ALLOCATION_NODES}."
printf "%s\n" "${STAGED_NODES[@]:0:EVAL_ALLOCATION_NODES}" \
  > "${GENERATION_NODES_FILE}"

python scripts/build_world_model_s2_generation_bundle.py \
  --evaluator-manifest "${S2_EVALUATOR_MANIFEST}" \
  --evaluator-manifest-sha256 "${S2_EVALUATOR_MANIFEST_SHA256}" \
  --output-dir "${GENERATION_BUNDLE_DIR}"

S2_GENERATION_BUNDLE_SHA256="$(
  python - "${GENERATION_BUNDLE_DIR}/manifest.json" <<'PY'
import sys
from pathlib import Path

from scripts.world_model_s1_eval_common import read_json, require_sha256, stable_sha256

bundle = read_json(Path(sys.argv[1]))
claimed = require_sha256(bundle.get("bundle_sha256"), "S2 generation bundle")
identity = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
if stable_sha256(identity) != claimed:
    raise SystemExit("The S2 generation bundle failed its identity check.")
print(claimed)
PY
)"
export S2_GENERATION_BUNDLE_SHA256
printf '%s\n' \
  "generation_bundle_sha256=${S2_GENERATION_BUNDLE_SHA256}" \
  >> "${RUN_CONTRACT_DIR}/launch.env"

mkdir -p "${GENERATION_SHARDS_DIR}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  # Row 3100 has the panel's maximum four-call fixture, so a model that
  # follows the target exercises every iterative generation turn. Row 11421
  # has the panel's longest prompt, 6,350 tokens. Job 5892323 could not
  # prefill any prompt above 3,812 tokens.
  S2_PREFLIGHT_LONG_ROW_ID="s2prod_e16d4ce1f1ddb81075d6ccbc288a8bc9"
  export S2_PREFLIGHT_LONG_ROW_ID
  python - "${GENERATION_BUNDLE_DIR}" <<'PY'
import json
from itertools import islice
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
with (root / "questions.jsonl").open(encoding="utf-8") as source:
    question = json.loads(next(islice(source, 3100, None)))
with (root / "tool_fixtures.jsonl").open(encoding="utf-8") as source:
    fixture = json.loads(next(islice(source, 3100, None)))
if question.get("row_id") != fixture.get("row_id") or len(fixture.get("steps", [])) != 4:
    raise SystemExit("The frozen S2 maximum-path preflight fixture changed.")
with (root / "questions.jsonl").open(encoding="utf-8") as source:
    long_question = json.loads(next(islice(source, 11421, None)))
if long_question.get("row_id") != os.environ["S2_PREFLIGHT_LONG_ROW_ID"]:
    raise SystemExit("The frozen S2 longest-prompt preflight row changed.")
print(json.dumps({"preflight_rows": [3100, 11421]}, sort_keys=True))
PY

  for S2_PREFLIGHT_SHARD_INDEX in 3100 11421; do
  PREFLIGHT_SHARD_DIR="${GENERATION_SHARDS_DIR}/$(printf "shard-%05d" "${S2_PREFLIGHT_SHARD_INDEX}")"
  export S2_PREFLIGHT_SHARD_INDEX PREFLIGHT_SHARD_DIR

  # Striding a row index by the complete row count selects exactly one row.
  srun --label --kill-on-bad-exit=1 --nodes=1 --ntasks=1 \
    --nodelist="${GENERATION_NODES_FILE}" \
    --ntasks-per-node=1 --gpus-per-task="${S2_EVAL_GPUS_PER_NODE}" \
    --cpu-bind=none --accel-bind=none bash -c '
    set -euo pipefail
    env \
      -u WANDB_API_KEY \
      -u HF_TOKEN \
      -u HF_HUB_TOKEN \
      -u HUGGING_FACE_HUB_TOKEN \
      -u OPENAI_API_KEY \
      -u ANTHROPIC_API_KEY \
      -u AWS_ACCESS_KEY_ID \
      -u AWS_SECRET_ACCESS_KEY \
      -u AWS_SESSION_TOKEN \
      -u S2_TEST_ANSWER_KEY \
      -u S2_TEST_ANSWER_KEY_SHA256 \
      -u S2_EVALUATOR_MANIFEST \
      -u http_proxy \
      -u https_proxy \
      -u all_proxy \
      -u no_proxy \
      -u HTTP_PROXY \
      -u HTTPS_PROXY \
      -u ALL_PROXY \
      -u NO_PROXY \
      WANDB_MODE=disabled \
      python scripts/run_world_model_s2_exact_generation.py \
        --generation-bundle "${GENERATION_BUNDLE_DIR}" \
        --base-model-path "${LOCAL_MODEL_DIR}" \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --tokenizer-path "${TOKENIZER_PATH}" \
        --tokenizer-manifest "${TOKENIZER_MANIFEST}" \
        --output-dir "${PREFLIGHT_SHARD_DIR}" \
        --method-id "${S2_METHOD_ID}" \
        --corpus-manifest-sha256 "${S2_CORPUS_MANIFEST_SHA256}" \
        --tokenizer-manifest-sha256 "${S2_TOKENIZER_MANIFEST_SHA256}" \
        --model-identity-sha256 "${MODEL_IDENTITY_SHA256}" \
        --base-model-artifact-sha256 "${BASE_MODEL_ARTIFACT_SHA256}" \
        --max-new-tokens "${S2_MAX_NEW_TOKENS}" \
        --max-total-tokens "${S2_MAX_TOTAL_TOKENS}" \
        --max-tool-call-tokens "${S2_MAX_TOOL_CALL_TOKENS}" \
        --max-final-prompt-tokens "${S2_MAX_FINAL_PROMPT_TOKENS}" \
        --enable-thinking \
        --reasoning-effort "${S2_REASONING_EFFORT}" \
        --history-policy "${S2_HISTORY_POLICY}" \
        --seed "${S2_EVAL_SEED}" \
        --batch-size 1 \
        --shard-index "${S2_PREFLIGHT_SHARD_INDEX}" \
        --shard-count "${S2_TEST_ROWS}" \
        --shard-output
  '

  python - \
    "${GENERATION_BUNDLE_DIR}" \
    "${PREFLIGHT_SHARD_DIR}" \
    "${RUN_CONTRACT_DIR}/preflight-${S2_PREFLIGHT_SHARD_INDEX}.json" <<'PY'
import json
import os
from pathlib import Path
import sys

from scripts.world_model_s1_eval_common import (
    read_json, read_jsonl, sha256_file, stable_sha256, write_json,
)

bundle_root, shard_root, receipt_path = map(Path, sys.argv[1:])
bundle = read_json(bundle_root / "manifest.json")
questions = read_jsonl(bundle_root / "questions.jsonl")
plan = read_json(shard_root / "generation_plan.json")
manifest = read_json(shard_root / "generation_manifest.json")
predictions = read_jsonl(shard_root / "predictions.jsonl")
claimed = manifest.get("manifest_sha256")
identity = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
row_count = int(os.environ["S2_TEST_ROWS"])
shard_index = int(os.environ["S2_PREFLIGHT_SHARD_INDEX"])
expected = {
    "schema_version": "mentor-rl-world-model-s2-prediction-shard-v1",
    "record_count": 1,
    "full_record_count": row_count,
    "shard_index": shard_index,
    "shard_count": row_count,
    "shard_strategy": "strided_row_order_v1",
    "train_run_id": os.environ["S2_TRAIN_RUN_ID"],
    "method_id": os.environ["S2_METHOD_ID"],
    "checkpoint_identity_sha256": os.environ["S2_CHECKPOINT_ARTIFACT_SHA256"],
    "evaluator_manifest_sha256": os.environ["S2_EVALUATOR_MANIFEST_SHA256"],
    "generation_bundle_sha256": bundle["bundle_sha256"],
    "reads_private_answer_keys": False,
}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit("The S2 preflight shard identity changed.")
if stable_sha256(identity) != claimed:
    raise SystemExit("The S2 preflight shard manifest identity failed.")
if sha256_file(shard_root / "predictions.jsonl") != manifest.get("predictions_sha256"):
    raise SystemExit("The S2 preflight predictions identity failed.")
if (len(predictions) != 1
        or predictions[0].get("row_id") != questions[shard_index].get("row_id")
        or predictions[0].get("network_used_false") is not True):
    raise SystemExit("The S2 preflight did not generate exactly its strided row.")
if (plan.get("record_count") != 1
        or plan.get("reads_private_answer_keys") is not False
        or plan.get("generation_config", {}).get("network_policy")
        != "python_audit_deny_outbound_socket_v1"):
    raise SystemExit("The S2 preflight generation plan is incomplete or unsafe.")
first_prompt_tokens = predictions[0]["generations"][0]["prompt_tokens"]
if (predictions[0].get("row_id") == os.environ["S2_PREFLIGHT_LONG_ROW_ID"]
        and (type(first_prompt_tokens) is not int or first_prompt_tokens < 6000)):
    raise SystemExit("The S2 longest-prompt preflight did not prefill its long prompt.")
write_json(receipt_path, {
    "schema_version": "mentor-rl-world-model-s2-eval-preflight-v2",
    "status": "complete",
    "row_id": predictions[0]["row_id"],
    "shard_index": shard_index,
    "first_turn_prompt_tokens": first_prompt_tokens,
    "tool_call_count": predictions[0]["tool_call_count"],
    "stop_reason": predictions[0]["stop_reason"],
    "elapsed_seconds": manifest["elapsed_seconds"],
    "generation_manifest_sha256": claimed,
    "checkpoint_identity_sha256": expected["checkpoint_identity_sha256"],
    "generation_bundle_sha256": bundle["bundle_sha256"],
})
PY
  [[ -s "${RUN_CONTRACT_DIR}/preflight-${S2_PREFLIGHT_SHARD_INDEX}.json" ]] || \
    die "The S2 preflight lacks its row ${S2_PREFLIGHT_SHARD_INDEX} receipt."
  done
  echo "The real-checkpoint two-row S2 preflight is complete: ${OUTPUT_DIR}"
  echo "No scoring or W&B publication ran."
  exit 0
fi

run_full_evaluation
