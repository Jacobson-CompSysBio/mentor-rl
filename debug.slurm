#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J mentor-sft-ds
#SBATCH -N 2
#SBATCH -t 0:45:00
#SBATCH -p batch
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
export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

# activate environment and set the cache home to nvme 
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/

#### CACHE ####
export HF_HOME=/mnt/bb/$USER/hf_cache_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/mnt/bb/$USER/ds_cache_$SLURM_JOB_ID
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# no buffering when printing output
export PYTHONUNBUFFERED=1

# nccl setup
export NCCL_NET=OFI
export FI_PROVIDER=cxi
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export FI_MR_CACHE_MONITOR=kdreg2     # Required to avoid a deadlock.
export FI_CXI_DEFAULT_CQ_SIZE=131072  # Ask the network stack to allocate additional space to process message completions.
export FI_CXI_DEFAULT_TX_SIZE=2048    # Ask the network stack to allocate additional space to hold pending outgoing messages.
export FI_CXI_RX_MATCH_MODE=hybrid    # Allow the network stack to transition to software mode if necessary.
export NCCL_NET_GDR_LEVEL=3           # Typically improves performance, but remove this setting if you encounter a hang/crash.
export NCCL_CROSS_NIC=1               # On large systems, this NCCL setting has been found to improve performance
#export CUDA_LAUNCH_BLOCKING=1     

####### DISTRIBUTED ########
# --- distributed env ---
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=6000
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"


####### LAUNCHING ########
MODEL_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/Llama-4-Scout-17B-16E-Instruct"
DATA_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/projects/llms/data/qa_pairs.json"
PROMPT_FIELD="question"
RESPONSE_FIELD="answer"

accelerate launch --config_file ds_config.yaml debug.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --prompt_field $PROMPT_FIELD \
    --response_field $RESPONSE_FIELD \
    --bf16 \
    --gradient_checkpointing \