#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J 70B-multinode-grpo
#SBATCH -N 5 # 5 nodes
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -C nvme
#SBATCH --gres=gpu:8
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# exit on errors
set -e

# activate modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# activate conda
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh

# activate env
source /lustre/orion/syb111/proj-shared/Environments/frontier/pytorch-rocm/activate.rc

# no buffering when printing output
export PYTHONUNBUFFERED=1

# get model
MODEL=/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/Llama3-3.3-70B-Instruct

# get list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# assign first 4 nodes for training, 5th node for vllm
TRAIN_NODES="${NODELIST[@]:0:4}"
VLLM_NODE="${NODELIST[4]}"

# run training on first 4 nodes
srun --nodes=4 --ntasks=4 --nodelist=${TRAIN_NODES} accelerate launch \
	--config_file fsdp_config.yaml \
	--num_processes 32 \
	--num_machines 4 \
	--main_process_ip ${TRAIN_NODES[0]} \
	--machine_rank $SLURM_PROCID \
	--rdzv_backend c10d \
	tests/70B_multinode_grpo.py \
	--server_ip $VLLM_NODE &

# run vllm server on 5th node
srun --nodes=1 --ntasks=1 --nodelist=${VLLM_NODE} trl vllm-serve --model $MODEL --tensor_parallel_size 8 &

wait
