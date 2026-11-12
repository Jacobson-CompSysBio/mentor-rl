#!/bin/bash
# Build the shared S2 token cache on one CPU node before allocating GPUs.
# Submit after a successful world_model_s2_check.slurm job and pass the same
# EXPECTED_RUN_CONFIG_SHA256. Eight processes preserve corpus row order.
#SBATCH --job-name=s2_prepare
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -p extended
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail

die() { echo "[ERROR] $*" >&2; exit 2; }
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 && \
   "${SLURM_CPUS_PER_TASK:-}" == 8 && "${SLURM_JOB_PARTITION:-}" == extended ]] || \
  die "S2 preparation requires one node/task and eight CPUs on extended."
: "${EXPECTED_RUN_CONFIG_SHA256:?Pass the pinned config SHA when submitting.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
[[ -f world_model_s2.slurm ]] || die "Submit from the repository root."
export RUN_CONFIG="${RUN_CONFIG:-config/world_model_s2_120b_training_v1_184n.json}"
export CONTRACT_ONLY=0 PREPARE_DATA_ONLY=1 OMP_NUM_THREADS=1
export S2_PREPARATION_WORKERS="${SLURM_CPUS_PER_TASK}"
# Cache contents are independent of a later training checkpoint continuation.
export S2_RESUME_CHECKPOINT=""
exec bash world_model_s2.slurm
