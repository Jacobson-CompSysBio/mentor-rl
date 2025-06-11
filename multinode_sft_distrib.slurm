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

# env file
. env.sh

# activate modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

# activate environment and set the cache home to nvme 
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/
export HF_HOME=/mnt/bb/$USER/model_cache
export HF_DATASETS_CACHE=/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/data/

# no buffering when printing output
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# force crash on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# set gpu, node, process vars
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# get port/address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# run training on first N nodes
# accelerate launch --config_file "fsdp_config.yaml" --num_processes 8 train_sft_hf.py

# create launcher var 
export LAUNCHER="accelerate launch \
	--main_process_ip $MASTER_ADDR \
	--main_process_port $MASTER_PORT \
	--machine_rank \$SLURM_PROCID \
	--num_processes $NUM_PROCESSES \
	--num_machines $NNODES \
        --dynamo_backend no \
	"

# create program var
export PROG="scripts/fsdp_train_sft_hf.py"

# combine launcher + program to make the command
export CMD="$LAUNCHER $PROG"

# srun call
srun --jobid $SLURM_JOBID bash -c "$CMD"
