#!/bin/bash
# Check the pinned S2 training contract in a one-node CPU allocation.
# Submit from the repository root after creating logs/. Pass the config SHA
# from submission time as EXPECTED_RUN_CONFIG_SHA256.
#SBATCH --job-name=s2_check
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:15:00
#SBATCH -p extended
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail

die() { echo "[ERROR] $*" >&2; exit 2; }
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 && \
   "${SLURM_CPUS_PER_TASK:-}" == 1 && "${SLURM_JOB_PARTITION:-}" == extended ]] || \
  die "The S2 check requires one node/task/CPU on extended."
: "${EXPECTED_RUN_CONFIG_SHA256:?Pass the pinned config SHA when submitting.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"
[[ -f world_model_s2.slurm ]] || die "Submit from the repository root."
export RUN_CONFIG="${RUN_CONFIG:-config/world_model_s2_120b_training_v1_184n.json}"
export CONTRACT_ONLY=1 PREPARE_DATA_ONLY=0 OMP_NUM_THREADS=1
exec bash world_model_s2.slurm
