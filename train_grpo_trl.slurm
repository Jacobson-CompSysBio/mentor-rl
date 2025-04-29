#!/bin/bash
#SBATCH --job-name=train_grpo_trl
#SBATCH -N 8
#SBATCH -A SYB114
#SBATCH -p batch
#SBATCH --reservation=hackathon1
#SBATCH -q debug
##SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=64 # max amount of cpus per task / change to 56 if we remove -S 0 flag
#SBATCH -S 0 # turn of "low noise mode"
#SBATCH -C nvme
#SBATCH -o slurm/%j.out
#SBATCH -e slurm/%j.err
#SBATCH --time=2:00:00
#SBATCH --tasks-per-node=1
##SBATCH -d afterany:3342165

. env.sh

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# don't worry about RCCL right now (yet) - will be slower
#export NCCL_SOCKET_IFNAME=hsn 
#export NCCL_NET_GDR_LEVEL=3
export JOB_MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# wandb settings
export WANDB_PROJECT=mentor-rl
export WANDB_RESUME=allow # for >24hr training

# broadcast the conda environment to all nodes
. sbcast_env.sh /lustre/orion/syb111/world-shared/environments/pytorch-rocm/ 

# broadcast the model cache
. sbcast_model.sh model_cache
export HF_HOME=/mnt/bb/$USER/model_cache

MODEL=


