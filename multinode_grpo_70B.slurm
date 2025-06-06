#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J 70B-multinode-grpo
#SBATCH -N 5
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
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

# activate conda
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

# no buffering when printing output
export PYTHONUNBUFFERED=1

# choose main port
export MASTER_PORT=29500

# get model
MODEL=/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/Llama3-3.3-70B-Instruct

# get list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# assign first N nodes for training, N+1 node for vllm
TRAIN_NODES="${NODELIST[@]:0:4}"
VLLM_NODE="${NODELIST[4]}"

echo "$TRAIN_NODES"
echo "$VLLM_NODE"

# run training on first N nodes
srun --nodes=4 --ntasks=4 --nodelist="${NODELIST[@]:0:4}" accelerate launch \
	--config_file fsdp_config.yaml \
	--num_processes 32 \
	--num_machines 4 \
	--main_process_ip ${NODELIST[0]} \
	--machine_rank $SLURM_PROCID \
	--rdzv_backend c10d \
	tests/70B_multinode_grpo.py \
	--server_ip $VLLM_NODE &

# run vllm server on N+1 node
srun --nodes=1 --ntasks=1 --nodelist="${NODELIST[4]}" trl vllm-serve --model "$MODEL" --tensor_parallel_size 8 &

wait
