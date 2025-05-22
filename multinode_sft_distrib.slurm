#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J 70B-multinode-grpo
#SBATCH -N 1 
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH --gres=gpu:8
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# exit on errors
set -x -e

# activate modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

# activate env
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

# no buffering when printing output
export PYTHONUNBUFFERED=1

# force crash on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

# set gpu, node, process vars
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# get port/address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# run training on first N nodes
accelerate launch --config_file "../fsdp_config.yaml" --num_processes 8 train_sft_hf.py

# create launcher var 
export LAUNCHER="accelerate launch \
	--config_file fsdp_config.yaml \
	--main_process_ip $MASTER_ADDR \
	--main_process_port $MASTER_PORT \
	--machine_rank \$SLURM_PROCID \
	--num_process $NUM_PROCESS \
	--num_machines $NNODES \
	"

# create program var
export PROG="scripts\train_sft_hf.py"

# combine launcher + program to make the command
export CMD="$LAUNCHER $PROGRAM"

# srun call
srun --jobid $SLURM_JOBID bash -c "$CMD"
