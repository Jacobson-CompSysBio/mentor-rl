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
# sbatch --export=ALL,MODEL_NAME=gpt-oss-20b-bf16,ADAPTER_PATH=checkpoints/<adapter> eval_pretrajectory_sft_exact.slurm
# Literal training-prompt recall (diagnostic fallback):
# sbatch --export=ALL,EVALUATION_REGIME=seen_row_recall,MODEL_NAME=gpt-oss-20b-bf16,ADAPTER_PATH=checkpoints/<adapter> eval_pretrajectory_sft_exact.slurm
# Legacy official-holdout behavior (not the default closed-book diagnostic):
# sbatch --export=ALL,EVALUATION_REGIME=official_holdout,SAMPLE_SIZE=120 eval_pretrajectory_sft_exact.slurm
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

export PRETRAJ_DATASET_NAME="v5_curriculum_patchcheck"
export PRETRAJ_DATASET_DIR="${SLURM_SUBMIT_DIR}/data/pretrajectory_sft/v5_curriculum_patchcheck"
export PRETRAJ_EXPECTED_PLAN_HASH="7a1e3a6c7b67c8dd8dd528c78a5ca653bce4ecbf430c3afc6480bd4ffb676556"
export PRETRAJ_EXPECTED_CONTENT_HASH="4e3d299d916c2c4149643d8ec5f6c4759bea9eff31d9d4a3219850e39b2981fc"
export CANONICAL_OBJECTS_PATH="${PRETRAJ_DATASET_DIR}/canonical_objects.jsonl"
export DATASET_AUDIT_PATH="${PRETRAJ_DATASET_DIR}/audit_report_contract_v3.json"

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
export MINIMUM_DIAGNOSTIC_FAMILY_EXACT="${MINIMUM_DIAGNOSTIC_FAMILY_EXACT:-0.80}"
export EVALUATION_REGIME="${EVALUATION_REGIME:-seen_fact_heldout_rendering}"
if [ -n "${ADAPTER_PATH}" ]; then
  DEFAULT_EXACT_EVAL_ROOT="${ADAPTER_PATH}"
  EVALUATION_TARGET="adapter"
else
  DEFAULT_EXACT_EVAL_ROOT="${SLURM_SUBMIT_DIR}/checkpoints/pretrajectory_base_model_${MODEL_NAME}"
  EVALUATION_TARGET="base_model"
fi
if [ "${EVALUATION_REGIME}" = "official_holdout" ]; then
  DEFAULT_EVAL_DIRNAME="exact_eval_${SLURM_JOB_ID}"
else
  DEFAULT_EVAL_DIRNAME="${EVALUATION_REGIME}_eval_${SLURM_JOB_ID}"
fi
export EXACT_EVAL_OUTDIR="${EXACT_EVAL_OUTDIR:-${DEFAULT_EXACT_EVAL_ROOT}/${DEFAULT_EVAL_DIRNAME}}"
export EVALUATION_TARGET
export STAGE_MODEL="${STAGE_MODEL:-1}"

DEFAULT_MAPPING_FAMILIES="entity_symbol_to_ensembl,entity_ensembl_to_symbol,ambiguous_alias_resolution"
export RETENTION_SUITE_MANIFEST=""
case "${EVALUATION_REGIME}" in
  official_holdout|unseen_fact_generalization)
    export PRETRAJ_CURRICULUM_STAGE="${PRETRAJ_CURRICULUM_STAGE:-stage1_entity_schema}"
    case "${PRETRAJ_CURRICULUM_STAGE}" in
      stage1_entity_schema|stage2_topology_priors|stage3_open_book_vectors|stage4_module_world_model|stage5_structured_tools|stage6_blend) ;;
      *)
        echo "[ERROR] Unsupported v5 curriculum stage: ${PRETRAJ_CURRICULUM_STAGE}" >&2
        exit 2
        ;;
    esac
    export PRETRAJ_SPLIT="${PRETRAJ_SPLIT:-test}"
    case "${PRETRAJ_SPLIT}" in
      train|val|test) ;;
      *)
        echo "[ERROR] Unsupported v5 split: ${PRETRAJ_SPLIT}" >&2
        exit 2
        ;;
    esac
    export DATASET_PATH="${PRETRAJ_DATASET_DIR}/curriculum/${PRETRAJ_CURRICULUM_STAGE}/${PRETRAJ_SPLIT}.jsonl"
    export SAMPLE_SIZE="${SAMPLE_SIZE:-120}"
    export SAMPLES_PER_FAMILY="0"
    export QUESTION_FAMILIES="${QUESTION_FAMILIES:-}"
    ;;
  seen_row_recall)
    export PRETRAJ_CURRICULUM_STAGE="stage1_entity_schema"
    export PRETRAJ_SPLIT="train"
    export DATASET_PATH="${PRETRAJ_DATASET_DIR}/curriculum/stage1_entity_schema/train.jsonl"
    export SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
    export SAMPLES_PER_FAMILY="${SAMPLES_PER_FAMILY:-32}"
    export QUESTION_FAMILIES="${QUESTION_FAMILIES:-${DEFAULT_MAPPING_FAMILIES}}"
    ;;
  seen_fact_heldout_rendering)
    export PRETRAJ_CURRICULUM_STAGE="stage1_entity_schema"
    export PRETRAJ_SPLIT="train"
    export SAMPLE_SIZE="${SAMPLE_SIZE:-0}"
    export SAMPLES_PER_FAMILY="${SAMPLES_PER_FAMILY:-32}"
    export QUESTION_FAMILIES="${QUESTION_FAMILIES:-${DEFAULT_MAPPING_FAMILIES}}"
    export RETENTION_SUITE_DIR="${RETENTION_SUITE_DIR:-${EXACT_EVAL_OUTDIR}/retention_suite}"
    export DATASET_PATH="${RETENTION_SUITE_DIR}/seen_fact_heldout_rendering.jsonl"
    export RETENTION_SUITE_MANIFEST="${RETENTION_SUITE_DIR}/suite_manifest.json"
    ;;
  *)
    echo "[ERROR] Unsupported evaluation regime: ${EVALUATION_REGIME}" >&2
    exit 2
    ;;
esac

if [ "${SAMPLES_PER_FAMILY}" -gt 0 ] && [ "${SAMPLE_SIZE}" -gt 0 ]; then
  echo "[ERROR] SAMPLE_SIZE and SAMPLES_PER_FAMILY cannot both be positive." >&2
  exit 2
fi

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

mkdir -p "${LOCAL_MODEL_DIR}" "${LOCAL_CACHE_BASE}" "${EXACT_EVAL_OUTDIR}" logs
if [ "${EVALUATION_REGIME}" = "seen_fact_heldout_rendering" ]; then
  echo "--- Building audited seen-fact/held-out-rendering suite ---"
  python "${SLURM_SUBMIT_DIR}/scripts/build_pretrajectory_sft_retention_suite.py" \
    --dataset-root "${PRETRAJ_DATASET_DIR}" \
    --out-dir "${RETENTION_SUITE_DIR}" \
    --seed "${SEED}" \
    --samples-per-family "${SAMPLES_PER_FAMILY}" \
    --families "${QUESTION_FAMILIES}"
fi

echo "--- Pre-trajectory exact eval started ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Evaluation target: ${EVALUATION_TARGET}"
echo "Evaluation regime: ${EVALUATION_REGIME}"
echo "Adapter path: ${ADAPTER_PATH:-<none>}"
echo "Dataset: ${DATASET_PATH}"
echo "Canonical objects: ${CANONICAL_OBJECTS_PATH}"
echo "Dataset audit: ${DATASET_AUDIT_PATH}"
echo "Output dir: ${EXACT_EVAL_OUTDIR}"
echo "Sample size: ${SAMPLE_SIZE}"
echo "Samples per family: ${SAMPLES_PER_FAMILY}"
echo "Question families: ${QUESTION_FAMILIES:-<all>}"
echo "Retention suite manifest: ${RETENTION_SUITE_MANIFEST:-<none>}"
echo "Thinking: ${ENABLE_THINKING}"

required_paths=("${MODEL_PATH}" "${DATASET_PATH}" "${CANONICAL_OBJECTS_PATH}" "${DATASET_AUDIT_PATH}")
if [ -n "${ADAPTER_PATH}" ]; then
  required_paths+=("${ADAPTER_PATH}")
fi
if [ -n "${RETENTION_SUITE_MANIFEST}" ]; then
  required_paths+=("${RETENTION_SUITE_MANIFEST}")
fi
for required_path in "${required_paths[@]}"; do
  if [ ! -e "${required_path}" ]; then
    echo "[ERROR] Missing required path: ${required_path}" >&2
    exit 2
  fi
done

python -c 'import json, os; p=os.environ["DATASET_AUDIT_PATH"]; r=json.load(open(p)); assert r.get("schema_version") == "pretrajectory-sft-audit-v3", f"wrong audit schema: {p}"; assert r.get("dataset_schema_version") == "pretrajectory-sft-v3", f"wrong dataset schema: {p}"; assert r.get("passed") is True and r.get("fatal_error_count") == 0, f"dataset audit did not pass: {p}"; assert r.get("plan_hash") == os.environ["PRETRAJ_EXPECTED_PLAN_HASH"], f"unexpected v5 plan hash: {p}"; assert r.get("content_hash") == os.environ["PRETRAJ_EXPECTED_CONTENT_HASH"], f"unexpected v5 content hash: {p}"'

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
  retention_suite_args=()
  if [ -n "${RETENTION_SUITE_MANIFEST}" ]; then
    retention_suite_args=(--retention-suite-manifest "${RETENTION_SUITE_MANIFEST}")
  fi
  python "${SLURM_SUBMIT_DIR}/scripts/run_pretrajectory_sft_exact_eval.py" \
    --model-path "${EVAL_MODEL_PATH}" \
    --model-source-path "${MODEL_PATH}" \
    "${adapter_args[@]}" \
    --dataset-path "${DATASET_PATH}" \
    --canonical-objects "${CANONICAL_OBJECTS_PATH}" \
    --dataset-audit "${DATASET_AUDIT_PATH}" \
    --source-dataset-root "${PRETRAJ_DATASET_DIR}" \
    --evaluation-regime "${EVALUATION_REGIME}" \
    --question-families "${QUESTION_FAMILIES}" \
    --samples-per-family "${SAMPLES_PER_FAMILY}" \
    "${retention_suite_args[@]}" \
    --minimum-diagnostic-family-exact "${MINIMUM_DIAGNOSTIC_FAMILY_EXACT}" \
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
