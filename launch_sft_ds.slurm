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

#### CACHE ####
#MODEL_DIR="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models"
#MODEL_NAME="Llama-4-Scout-17B-16E-Instruct"
#if [[ $SLURM_LOCALID -eq 0 ]]; then
#	rsync -a "$MODEL_DIR/$MODEL_NAME" "/mnt/bb/$USER/llama4"
#fi
export HF_HOME=/mnt/bb/$USER/hf_cache_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/mnt/bb/$USER/ds_cache_$SLURM_JOB_ID
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# no buffering when printing output
export PYTHONUNBUFFERED=1

# force crash on nccl issues like hanging broadcast
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

# deepsepeed
export DS_REPORT_ENV=0
export DS_BUILD_AIO=0          

####### DISTRIBUTED ########
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=8
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

####### LAUNCHING ########
export LAUNCHER="accelerate launch \
	--config_file ds_config.yaml \
	--deepspeed_multinode_launcher standard \
	--main_process_ip $MASTER_ADDR \
	--main_process_port $MASTER_PORT \
	--machine_rank \$SLURM_PROCID \
	--num_processes $NUM_PROCESSES \
	--num_machines $NNODES \
	"
export PROG="scripts/train_sft_ds.py"
export CMD="$LAUNCHER $PROG"

#srun call
srun bash -c "$CMD"