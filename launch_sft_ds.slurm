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
export HF_HOME=/mnt/bb/$USER/hf_cache_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/mnt/bb/$USER/ds_cache_$SLURM_JOB_ID
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# no buffering when printing output
export PYTHONUNBUFFERED=1

# force crash on nccl issues like hanging broadcast
export HSA_FORCE_FINE_GRAIN_PCIE=1   # required for fast inter-GPU P2P DMA
export RCCL_P2P_LEVEL=LOC            # keep P2P inside a GCD pair when >1 node
export RCCL_NET_GDR_LEVEL=2          # allow GPUDirect on Slingshot
export NCCL_P2P_DISABLE=0            # re-enable P2P if you disabled it earlier
export NCCL_DEBUG=INFO               # short NCCL/RCCL traces
export NCCL_DEBUG_SUBSYS=COLL        # only collectives
export RCCL_DEBUG=INFO               # ROCm equivalent
export RCCL_ENABLE_SHARP=0           # disables experimental SHArP if present
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=hsn0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_LAUNCH_BLOCKING=1

# deepsepeed
export DS_REPORT_ENV=0
export DS_BUILD_AIO=0          

####### DISTRIBUTED ########
# recommended: use 8 tasks per node instead of 1
GPUS_PER_NODE=8
MASTER_ADDR=$(hostname -i)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

####### LAUNCHING ########
# how does deepspeed define machines? change parameters to figure this out
export LAUNCHER="accelerate launch \
	--config_file ds_config.yaml \
	--deepspeed_multinode_launcher standard \
	--num_processes $WORLD_SIZE \
	--num_machines $NNODES \
	--machine_rank \$SLURM_PROCID \
	--main_process_ip $MASTER_ADDR \
	--main_process_port $MASTER_PORT \
	"
export PROG="scripts/train_sft_ds.py"
export CMD="$LAUNCHER $PROG"

#srun call
echo "Launching model..."
srun bash -c "$CMD"

# two nodes with 16 tasks --> will change the device_id point of view; might need to configure the device_ids in the .py
# srun -N2 -n16 -c7 --gpus-per-task=1 --gpu-bind=closest \
#     bash -c "$CMD" 
# consider downgrading torch - might work better with FA2