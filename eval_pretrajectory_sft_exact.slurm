#!/bin/bash
#SBATCH --job-name=pretraj_exact_eval
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

# Example:
# sbatch --export=ALL eval_pretrajectory_sft_exact.slurm
#
# Common overrides:
# sbatch --export=ALL,SAMPLE_SIZE=512,ENABLE_THINKING=False eval_pretrajectory_sft_exact.slurm
# sbatch --export=ALL,MODEL_NAME=gpt-oss-20b-bf16,ADAPTER_PATH=checkpoints/<adapter>,SAMPLE_SIZE=256 eval_pretrajectory_sft_exact.slurm
# Evaluate the base model by exporting an explicitly empty adapter path:
# sbatch --export=ALL,MODEL_NAME=gpt-oss-20b-bf16,ADAPTER_PATH= eval_pretrajectory_sft_exact.slurm

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
# Use '-' rather than ':-' so ADAPTER_PATH= remains empty for base-model eval.
export ADAPTER_PATH="${ADAPTER_PATH-}"

export PRETRAJ_DATASET_NAME="${PRETRAJ_DATASET_NAME:-v5_curriculum_patchcheck}"
export PRETRAJ_DATASET_DIR="${PRETRAJ_DATASET_DIR:-${SLURM_SUBMIT_DIR}/data/pretrajectory_sft/${PRETRAJ_DATASET_NAME}}"
export PRETRAJ_CURRICULUM_STAGE="${PRETRAJ_CURRICULUM_STAGE:-stage6_blend}"
export PRETRAJ_SPLIT="${PRETRAJ_SPLIT:-test}"
export DATASET_PATH="${DATASET_PATH:-${PRETRAJ_DATASET_DIR}/curriculum/${PRETRAJ_CURRICULUM_STAGE}/${PRETRAJ_SPLIT}.jsonl}"
export CANONICAL_OBJECTS_PATH="${CANONICAL_OBJECTS_PATH:-${PRETRAJ_DATASET_DIR}/canonical_objects.jsonl}"
if [[ -z "${DATASET_AUDIT_PATH:-}" ]]; then
    if [[ -f "${PRETRAJ_DATASET_DIR}/audit_report_contract_v3.json" ]]; then
        export DATASET_AUDIT_PATH="${PRETRAJ_DATASET_DIR}/audit_report_contract_v3.json"
    else
        export DATASET_AUDIT_PATH="${PRETRAJ_DATASET_DIR}/audit_report.json"
    fi
fi

export SAMPLE_SIZE="${SAMPLE_SIZE:-256}"
export SEED="${SEED:-1729}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-5120}"
export ENABLE_THINKING="${ENABLE_THINKING:-False}"
export REASONING_EFFORT="${REASONING_EFFORT:-}"
export DO_SAMPLE="${DO_SAMPLE:-False}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-True}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
export MAX_REPORT_EXAMPLES="${MAX_REPORT_EXAMPLES:-120}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
if [ -n "${ADAPTER_PATH}" ]; then
  DEFAULT_EXACT_EVAL_ROOT="${ADAPTER_PATH}"
  EVALUATION_TARGET="adapter"
else
  DEFAULT_EXACT_EVAL_ROOT="${SLURM_SUBMIT_DIR}/checkpoints/pretrajectory_base_model_${MODEL_NAME}"
  EVALUATION_TARGET="base_model"
fi
export EXACT_EVAL_OUTDIR="${EXACT_EVAL_OUTDIR:-${DEFAULT_EXACT_EVAL_ROOT}/exact_eval_${SLURM_JOB_ID}}"
export EVALUATION_TARGET
export STAGE_MODEL="${STAGE_MODEL:-1}"

export LOCAL_SSD_PATH="${LOCAL_SSD_PATH:-/mnt/bb/${USER}/pretraj_exact_eval_${SLURM_JOB_ID}}"
export LOCAL_MODEL_DIR="${LOCAL_SSD_PATH}/model/${MODEL_NAME}"
export LOCAL_CACHE_BASE="${LOCAL_SSD_PATH}/caches"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${LOCAL_CACHE_BASE}/hf"
export HF_DATASETS_CACHE="${LOCAL_CACHE_BASE}/datasets"
export TRITON_CACHE_DIR="${LOCAL_CACHE_BASE}/triton"
export TORCH_EXTENSIONS_DIR="${LOCAL_CACHE_BASE}/torch_extensions"

export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-6.3.1
export HSA_XNACK=0
export DS_ACCELERATOR=cuda
export PYTORCH_ALLOC_CONF="expandable_segments:False"

cleanup() {
  rm -rf "${LOCAL_SSD_PATH}" || true
}
trap cleanup EXIT

echo "--- Pre-trajectory exact eval started ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Evaluation target: ${EVALUATION_TARGET}"
echo "Adapter path: ${ADAPTER_PATH:-<none>}"
echo "Dataset: ${DATASET_PATH}"
echo "Canonical objects: ${CANONICAL_OBJECTS_PATH}"
echo "Dataset audit: ${DATASET_AUDIT_PATH}"
echo "Output dir: ${EXACT_EVAL_OUTDIR}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Thinking: ${ENABLE_THINKING}"

required_paths=("${MODEL_PATH}" "${DATASET_PATH}" "${CANONICAL_OBJECTS_PATH}" "${DATASET_AUDIT_PATH}")
if [ -n "${ADAPTER_PATH}" ]; then
  required_paths+=("${ADAPTER_PATH}")
fi
for required_path in "${required_paths[@]}"; do
  if [ ! -e "${required_path}" ]; then
    echo "[ERROR] Missing required path: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${LOCAL_MODEL_DIR}" "${LOCAL_CACHE_BASE}" "${EXACT_EVAL_OUTDIR}" logs

if [ "${STAGE_MODEL}" = "1" ]; then
  echo "--- Staging model to local NVMe ---"
  rsync -a --info=stats2 "${MODEL_PATH}/" "${LOCAL_MODEL_DIR}/"
  export EVAL_MODEL_PATH="${LOCAL_MODEL_DIR}"
else
  export EVAL_MODEL_PATH="${MODEL_PATH}"
fi

echo "--- Running exact eval ---"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpu-bind=none --accel-bind=g bash -c '
  set -euo pipefail
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
  fi
  adapter_args=()
  if [ -n "${ADAPTER_PATH}" ]; then
    adapter_args=(--adapter-path "${ADAPTER_PATH}")
  fi
  python "${SLURM_SUBMIT_DIR}/scripts/run_pretrajectory_sft_exact_eval.py" \
    --model-path "${EVAL_MODEL_PATH}" \
    --model-source-path "${MODEL_PATH}" \
    "${adapter_args[@]}" \
    --dataset-path "${DATASET_PATH}" \
    --canonical-objects "${CANONICAL_OBJECTS_PATH}" \
    --dataset-audit "${DATASET_AUDIT_PATH}" \
    --out-dir "${EXACT_EVAL_OUTDIR}" \
    --sample-size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --enable-thinking "${ENABLE_THINKING}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    --do-sample "${DO_SAMPLE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --local-files-only "${LOCAL_FILES_ONLY}" \
    --trust-remote-code "${TRUST_REMOTE_CODE}" \
    --max-report-examples "${MAX_REPORT_EXAMPLES}" \
    --progress-every "${PROGRESS_EVERY}"
'

echo "--- Pre-trajectory exact eval finished ---"
echo "Output dir: ${EXACT_EVAL_OUTDIR}"
