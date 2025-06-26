#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J mentor-sft-ds
#SBATCH -N 2 
#SBATCH -t 2:00:00
#SBATCH -p extended
#SBATCH -q debug
#SBATCH -C nvme
#SBATCH --gres=gpu:8
#SBATCH -o logs/%x.out # Out Path
#SBATCH -e logs/%x.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

# exit on errors
set -eo pipefail

####### ENVIRONMENT #######

# env file
. env.sh

# activate modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

# activate environment and set the cache home to nvme 
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

export HF_HOME=/mnt/bb/$USER/hf_cache_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/mnt/bb/$USER/ds_cache_$SLURM_JOB_ID
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# no buffering when printing output
export PYTHONUNBUFFERED=1

# force crash on nccl issues like hanging broadcast
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

####### DISTRIBUTED ########
GPUS_PER_NODE=8
WORLD_SIZE=$(( $SLURM_NNODES * $GPUS_PER_NODE))

# get port/address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

####### LAUNCHING ########
export LAUNCHER="accelerate launch \
	--config_file ds_config.yaml \
	--main_process_ip $MASTER_ADDR \
	--main_process_port $MASTER_PORT \
	--machine_rank \$SLURM_PROCID \
	--num_processes $WORLD_SIZE \
	--num_machines $SLURM_NNODES \
	"
export PROG="scripts/train_sft_ds.py"
export CMD="$LAUNCHER $PROG"

# srun call
srun --jobid $SLURM_JOBID bash -c "$CMD"
