#!/bin/bash
#SBATCH -J mentor-loe-corpus
#SBATCH -A SYB114
#SBATCH -p extended
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH --gpus-per-node=8
#SBATCH -C nvme
#SBATCH -o logs/rwr-loe-corpus-%A_%a.out
#SBATCH -e logs/rwr-loe-corpus-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl}"
cd "${REPO_ROOT}"
mkdir -p logs

module load craype-accel-amd-gfx90a rocm/6.3.1

if [[ "${LOE_SKIP_CONDA:-0}" != "1" ]]; then
  if [[ ! -f /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh ]]; then
    echo "Missing Frontier conda bootstrap script." >&2
    exit 1
  fi
  source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
  conda activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
fi

LOE_MODE="${LOE_MODE:-prewarm}"
LOE_SHARD_COUNT="${LOE_SHARD_COUNT:-1}"
LOE_SHARD_INDEX="${LOE_SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
LOE_SEED="${LOE_SEED:-42}"
LOE_MODULE_SELECTION_METHOD="${LOE_MODULE_SELECTION_METHOD:-elbow}"
if [[ -z "${LOE_MATERIALIZE_WORKERS:-}" ]]; then
  if [[ "${LOE_MODE}" == "materialize" || "${LOE_MODE}" == "all" ]]; then
    LOE_MATERIALIZE_WORKERS=16
  else
    LOE_MATERIALIZE_WORKERS=1
  fi
fi

STORE_DIR="${STORE_DIR:-data/runtime/full_brain_multiplex_store}"
RWR_HPC_FLIST="${RWR_HPC_FLIST:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/data/full_brain_flist.tsv}"
RWR_HPC_BUILD_DIR="${RWR_HPC_BUILD_DIR:-external/rwr_hpc/build_frontier}"
RWR_HPC_CACHE_DIR="${RWR_HPC_CACHE_DIR:-data/runtime/rwr_hpc_cache}"
LOE_RANK_CACHE_DIR="${LOE_RANK_CACHE_DIR:-data/runtime/rwr_loe_full_brain_rank_cache}"
MENTOR_CORPUS_DIR="${MENTOR_CORPUS_DIR:-data/gw_dendrogram_corpus_full_brain}"
OUT_DIR="${OUT_DIR:-data/rwr_loe_corpus_full_brain}"

JOB_ID="${SLURM_JOB_ID:-local}"
SCRATCH_BASE="${SCRATCH:-${TMPDIR:-/tmp}}"
LOE_SCRATCH_DIR="${LOE_SCRATCH_DIR:-${SCRATCH_BASE}/mentor_rl_loe/${JOB_ID}/${LOE_SHARD_INDEX}}"
mkdir -p "${LOE_SCRATCH_DIR}"

LOE_RWR_TASKS="${LOE_RWR_TASKS:-8}"
if [[ -z "${LOE_RWR_LAUNCHER_PREFIX:-}" ]]; then
  LOE_RWR_LAUNCHER_PREFIX="srun -N1 -n ${LOE_RWR_TASKS} --gpus-per-node=8 --gpu-bind=closest"
fi

MAX_GENES_ARG=()
if [[ -n "${LOE_MAX_GENES:-}" ]]; then
  MAX_GENES_ARG=(--max-genes "${LOE_MAX_GENES}")
fi

FORCE_ARG=()
if [[ "${LOE_FORCE:-0}" == "1" ]]; then
  FORCE_ARG=(--force)
fi

CHECKPOINT_ARG=()
if [[ -n "${LOE_CHECKPOINT_DIR:-}" ]]; then
  CHECKPOINT_ARG=(--checkpoint-dir "${LOE_CHECKPOINT_DIR}")
fi

RANK_CONTEXT_ARG=()
if [[ -n "${LOE_RANK_CACHE_CONTEXT_DIR:-}" ]]; then
  RANK_CONTEXT_ARG=(--rank-cache-context-dir "${LOE_RANK_CACHE_CONTEXT_DIR}")
fi

python scripts/build_rwr_loe_corpus.py \
  --mode "${LOE_MODE}" \
  --store-dir "${STORE_DIR}" \
  --rwr-hpc-flist "${RWR_HPC_FLIST}" \
  --rwr-hpc-build-dir "${RWR_HPC_BUILD_DIR}" \
  --rwr-hpc-cache-dir "${RWR_HPC_CACHE_DIR}" \
  --rank-cache-dir "${LOE_RANK_CACHE_DIR}" \
  --scratch-dir "${LOE_SCRATCH_DIR}" \
  --mentor-corpus-dir "${MENTOR_CORPUS_DIR}" \
  --out-dir "${OUT_DIR}" \
  --seed "${LOE_SEED}" \
  --module-selection-method "${LOE_MODULE_SELECTION_METHOD}" \
  --materialize-workers "${LOE_MATERIALIZE_WORKERS}" \
  --shard-index "${LOE_SHARD_INDEX}" \
  --shard-count "${LOE_SHARD_COUNT}" \
  --rwr-launcher-prefix "${LOE_RWR_LAUNCHER_PREFIX}" \
  --rwr-hpc-edgelist-has-headers \
  "${CHECKPOINT_ARG[@]}" \
  "${RANK_CONTEXT_ARG[@]}" \
  "${MAX_GENES_ARG[@]}" \
  "${FORCE_ARG[@]}" \
  --json
