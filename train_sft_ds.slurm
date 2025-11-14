#!/bin/bash
#SBATCH --job-name=120b-inf-debug
#SBATCH -A SYB114
#SBATCH -N 4
#SBATCH -t 12:00:00
#SBATCH -p extended
##SBATCH -q debug
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out # Out Path
#SBATCH -e logs/%x-%j.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

set -euo pipefail
set -x

# load modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0

# export compilers
export CC=cc
export CXX=CC
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC

# load aws-ofi-rccl
export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

# load conda, activate env
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

echo "--- Slurm Job Started ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"

# --- Define Paths ---
MODEL_NAME="gpt-oss-120b-bf16"

export LOCAL_SSD_PATH="/mnt/bb/${USER}/job_${SLURM_JOB_ID}"
export LOCAL_ENV_DIR="${LOCAL_SSD_PATH}/venv"
export LOCAL_CACHE_DIR="${LOCAL_SSD_PATH}/hf_cache"
export LOCAL_MODEL_DIR="${LOCAL_SSD_PATH}/model/${MODEL_NAME}/"
export LOCAL_DATA_DIR="${LOCAL_SSD_PATH}/data"
export LOCAL_CONFIG_DIR="${LOCAL_SSD_PATH}/config"

VENV_PATH="/lustre/orion/syb111/world-shared/environments/pytorch-rocm"
MODEL_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/${MODEL_NAME}"
DATA_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/data/"
DS_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/config/"
CHECKPOINT_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/mentor-rl/checkpoints"

# hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="/mnt/bb/$USER/hf_cache_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="/mnt/bb/$USER/ds_cache_${SLURM_JOB_ID}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# --- STAGE 1: Stage Data to Local SSD on Each Node ---
srun --kill-on-bad-exit=1 --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  echo '--- Staging on node: $(hostname) ---'

  mkdir -p ${LOCAL_SSD_PATH} ${LOCAL_MODEL_DIR} ${LOCAL_ENV_DIR} ${LOCAL_DATA_DIR} ${LOCAL_CONFIG_DIR}

  echo 'Python check:'; which python; python -c 'import torch,sys; print(torch.__version__, sys.executable)'

  echo 'Copying model weights...'
  rsync -a --info=progress2 "${MODEL_PATH}/" "${LOCAL_MODEL_DIR}"

  echo 'Copying dataset...'
  rsync -a --info=progress2 ${DATA_PATH} ${LOCAL_DATA_DIR}

  echo 'Copying DS config...'
  rsync -a --info=progress2 ${DS_PATH} ${LOCAL_CONFIG_DIR}

  mkdir -p ${LOCAL_SSD_PATH}/hf_cache

  echo '--- Staging on $(hostname) complete ---'
"
echo "--- Staging complete on all nodes ---"

# --- STAGE 2: Run the Training Job ---
echo "--- Launching Distributed Training with RCCL Plugin ---"
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500

# rccl settings
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.

export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
export NCCL_SOCKET_IFNAME=hsn0        # NCCL/RCCL will use the high speed network to coordinate startup.
export NCCL_OFI_USE_NICLIST=hsn0
export GLOO_SOCKET_IFNAME=hsn0

# for distributed
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-6.3.1
export NCCL_DEBUG=WARN # Un-comment to diagnose NCCL issues if needed
export HSA_XNACK=0
export DS_ACCELERATOR=cuda

# --- Per-rank, node-local caches to avoid Triton/Inductor races ---
export LOCAL_CACHE_BASE="${LOCAL_SSD_PATH}/caches"
mkdir -p "${LOCAL_CACHE_BASE}"

# Optional: keep HIP allocator log quiet (unrelated warning you saw)
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False"

# wandb
export https_proxy=http://proxy.ccs.ornl.gov:3128
export http_proxy=$https_proxy
export WANDB_HTTP_TIMEOUT=90

srun --cpu-bind=none --accel-bind=g --kill-on-bad-exit=1 bash -c '

  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID

  export AMDSMI_GPU_METRICS_CACHE_MS=5000
  if [ "${SLURM_LOCALID}" != "0" ]; then export WANDB_MODE=disabled; fi

  # --- Launch the training ---
  python \
    "'${SLURM_SUBMIT_DIR}'"/scripts/train_sft_ds.py \
      --model_path="'${LOCAL_MODEL_DIR}'" \
      --output_dir="'${LOCAL_SSD_PATH}'/checkpoints/" \
      --dataset_path="'${LOCAL_DATA_DIR}'/qa_pairs.json" \
      --deepspeed="'${LOCAL_CONFIG_DIR}'/ds_zero3.json" \
      --seed=900913 \
      --num_train_epochs=10 \
      --per_device_train_batch_size=1 \
      --gradient_accumulation_steps=4 \
      --learning_rate=2e-5 \
      --logging_steps=1 \
      --bf16 \
      --lora_r=16 \
      --lora_alpha=32 \
      --lora_dropout=0.05 \
      --run_inference_after_training
'

# --- STAGE 3: Copy Final Results Back to Persistent Storage ---
echo "--- Copying final results from local SSD to shared storage ---"
PERSISTENT_OUTPUT_DIR="${CHECKPOINT_PATH}/${MODEL_NAME}_job_${SLURM_JOB_ID}"
mkdir -p "$PERSISTENT_OUTPUT_DIR"

# Only copy from the head node where trl has combined the results
srun --nodes=1 --ntasks=1 -w "$head_node" \
  rsync -a --info=progress2 "${LOCAL_SSD_PATH}/checkpoints/" "${PERSISTENT_OUTPUT_DIR}/"

# --- STAGE 4: Cleanup ---
echo "--- Cleaning up local SSD on all nodes ---"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf ${LOCAL_SSD_PATH}"

echo "--- Slurm Job Finished ---"