#!/bin/bash
# Pair selected completed v9 adapters with validation losses and rollouts.
# Required: RUN_CONFIG, CHECKPOINT_PATH, OUTPUT_DIR, RUN_CONFIG_SHA256,
# DIAGNOSTIC_CODE_MANIFEST, and DIAGNOSTIC_CODE_MANIFEST_SHA256.
# Each worker reuses its node's base model across all five adapters.
#SBATCH --job-name=s1_v9_checkpoint_diag
#SBATCH -A SYB114
#SBATCH -N 32
#SBATCH -t 16:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate
#SBATCH --no-requeue

set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
[[ "${SLURM_NNODES:-}" == 32 ]] || { echo 'Expected 32 diagnostic nodes.' >&2; exit 2; }
: "${RUN_CONFIG:?}" "${CHECKPOINT_PATH:?}" "${OUTPUT_DIR:?}" "${RUN_CONFIG_SHA256:?}"
: "${DIAGNOSTIC_CODE_MANIFEST:?}" "${DIAGNOSTIC_CODE_MANIFEST_SHA256:?}"
printf '%s  %s\n' "${RUN_CONFIG_SHA256}" "${RUN_CONFIG}" | sha256sum --check --strict --quiet
printf '%s  %s\n' "${DIAGNOSTIC_CODE_MANIFEST_SHA256}" "${DIAGNOSTIC_CODE_MANIFEST}" | sha256sum --check --strict --quiet
sha256sum --check --strict --quiet "${DIAGNOSTIC_CODE_MANIFEST}"

module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset PYTHONPATH USE_TORCH WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP
export ROCM_SMI_DISABLE=1 AMD_SMI_DISABLE=1 RSMI_LOGGING=0 AMDSMI_GPU_METRICS_CACHE_MS=0
export CC=cc CXX=CC CMAKE_C_COMPILER=cc CMAKE_CXX_COMPILER=CC
export ROCM_HOME=/opt/rocm-6.3.1 PYTORCH_ROCM_ARCH=gfx90a HSA_XNACK=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-56}"
export HF_HOME="/mnt/bb/${USER}/mentor_hf/${SLURM_JOB_ID}/checkpoint-diagnostic"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_EXTENSIONS_DIR="/mnt/bb/${USER}/mentor_torch_extensions/torch280_rocm631"
export TRITON_CACHE_DIR="/mnt/bb/${USER}/mentor_triton/${SLURM_JOB_ID}/checkpoint-diagnostic"
export WANDB_MODE=online
ulimit -c 0

python scripts/diagnose_world_model_checkpoints.py prepare \
  --config "${RUN_CONFIG}" --checkpoint-path "${CHECKPOINT_PATH}" --output-dir "${OUTPUT_DIR}"

srun --label --kill-on-bad-exit=1 --nodes=32 --ntasks=32 --ntasks-per-node=1 \
  --gpus-per-task=8 --cpu-bind=none --accel-bind=none \
  env -u WANDB_API_KEY -u HF_TOKEN -u HF_HUB_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
  -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
  python -c '
import json, os, pathlib, subprocess, sys
plan = json.loads((pathlib.Path(os.environ["OUTPUT_DIR"]) / "plan.json").read_text())
spec = plan["training"]["model"]
source = pathlib.Path(spec["path"]).resolve()
cache = pathlib.Path("/mnt/bb") / os.environ["USER"] / "mentor_model_cache" / (source.name + "-" + spec["base_model_artifact_sha256"][:16])
marker = cache / ".mentor_model_identity_sha256"
if not marker.is_file() or marker.read_text().strip() != spec["base_model_artifact_sha256"]:
    cache = pathlib.Path("/mnt/bb") / os.environ["USER"] / "mentor_checkpoint_diagnostic_models" / os.environ["SLURM_JOB_ID"]
    cache.mkdir(parents=True, exist_ok=False)
    subprocess.run(["rsync", "-a", str(source) + "/", str(cache) + "/"], check=True)
args = [sys.executable, "scripts/diagnose_world_model_checkpoints.py", "worker",
        "--config", os.environ["RUN_CONFIG"], "--checkpoint-path", os.environ["CHECKPOINT_PATH"],
        "--output-dir", os.environ["OUTPUT_DIR"], "--shard-index", os.environ["SLURM_PROCID"],
        "--base-model-cache-path", str(cache)]
os.execv(sys.executable, args)
'
python scripts/diagnose_world_model_checkpoints.py merge \
  --config "${RUN_CONFIG}" --checkpoint-path "${CHECKPOINT_PATH}" --output-dir "${OUTPUT_DIR}"
