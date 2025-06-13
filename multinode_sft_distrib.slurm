#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J mentor-sft
#SBATCH -N 2 
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH -C nvme
#SBATCH --gres=gpu:8
#SBATCH -o logs/%x.out # Out Path
#SBATCH -e logs/%x.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=krusepi@ornl.gov

# exit on errors
set -e

# activate modules and software
module load  PrgEnv-gnu/8.6.0  rocm/6.3.1  craype-accel-amd-gfx90a
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

# caches and paths
export HF_HOME=/mnt/bb/$USER/model_cache
export HF_DATASETS_CACHE=/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/data/
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# wandb settings
export WANDB_SETTINGS="init_timeout=30"
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DISABLE_SERVICE=true

# nccl and rocm tuning
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1
export NCCL_COLLNET_ENABLE=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export FI_MR_CACHE_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ROCM_RESOURCES_OOM_FALLBACK=1

# don't mask gpus
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# set gpu, node, process vars
export LOCAL_RANK=$SLURM_LOCALID
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
WORLD=$((GPUS_PER_NODE * NNODES))

# get port/address
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$(shuf -i 20000-40000 -n 1)

# create launcher var 
LAUNCHER="accelerate launch \
	--main_process_ip $MASTER_ADDR \
	--main_process_port $MASTER_PORT \
	--machine_rank \$SLURM_PROCID \
	--num_processes $GPUS_PER_NODE \
	--num_machines $NNODES \
        --dynamo_backend no \
	"

# create program var
PROG="scripts/fsdp_train_sft_hf.py"

# combine launcher + program to make the command
export CMD="$LAUNCHER $PROG"

# srun call
srun --export=ALL,HTTPS_PROXY,http_proxy, SSL_CERT_DIR --ntasks=$NNODES bash -c "$CMD"
