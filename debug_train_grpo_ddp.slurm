#!/bin/bash
#SBATCH --job-name=train_grpo_trl
#SBATCH -N 4
#SBATCH -A SYB114
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --gres=gpu:8
##SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=64 # max amount of cpus per task / change to 56 if we remove -S 0 flag
#SBATCH -S 0 # turn of "low noise mode"
#SBATCH -C nvme
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --time=2:00:00
#SBATCH --tasks-per-node=1
##SBATCH -d afterany:3342165

# load environment
. env.sh

# set number of threads for each openmp process
export OMP_NUM_THREADS=8

# don't buffer the stdout
export PYTHONUNBUFFERED=1

# overwrite ray's hip device check - it wasn't finding any, causing error
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# get the address for the master node on our job
JOB_MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
JOB_MASTER_ADDR=$(echo ${JOB_MASTER_ADDR} | sed 's/^\([^\.]*\)/\1-hsn0/')

# wandb settings
export WANDB_PROJECT=mentor-rl
export WANDB_RESUME=allow # for >24hr training

# broadcast the conda environment to all nodes
#. sbcast_env.sh /lustre/orion/syb111/world-shared/environments/pytorch-rocm

# broadcast the model cache
#. sbcast_model.sh model_cache

# activate environment and set the cache home to nvme 
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm/
export HF_HOME=/mnt/bb/$USER/model_cache

# model path
MODEL=/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/Llama-4-Scout-17B-16E-Instruct/

# split the nodes into a ML and a vLLM half
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_NODES=${#NODELIST[@]}
SPLIT=$((NUM_NODES / 2))
NODES_FIRST_SPLIT=$(printf -v joined '%s,' "${NODELIST[@]:0:SPLIT}"; echo "${joined%,}")

# the second split consists of a head node and n-1 worker nodes
HEAD_NODE=${NODELIST[SPLIT]}
HEAD_NODE_IP=$(echo ${HEAD_NODE} | sed 's/^\([^\.]*\)/\1-hsn0/')
WORKER_NODES=$(printf -v joined '%s,' "${NODELIST[@]:SPLIT+1}"; echo "${joined%,}")

# start ray cluster
export PORT=6379
RAY_HEAD_CMD="
export HIP_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES} ;
unset ROCR_VISIBLE_DEVICES ;
export VLLM_HOST_IP=\$(hostname | sed 's/^\([^\.]*\)/\1-hsn0/') ;
ray start --head --node-ip-address=\${VLLM_HOST_IP} --port=${PORT} --num-cpus=64 --num-gpus=8 --block &
trl vllm-serve --model=$MODEL --data-parallel-size=$((NUM_NODES-SPLIT)) --tensor-parallel-size 8 --max-model-len 32768 &
wait
"

export LD_PRELOAD=/usr/lib64/libstdc++.so.6
srun -w $HEAD_NODE --exclusive -n 1 --nodes=1 --ntasks-per-node=1 --gpus-per-task=8 bash -c "${RAY_HEAD_CMD}" &
srun -w $WORKER_NODES --exclusive --nodes=$((NUM_NODES-SPLIT-1)) --ntasks-per-node=1 bash -c \
"
export VLLM_HOST_IP=\$(hostname | sed 's/^\([^\.]*\)/\1-hsn0/') ;
ray start --node-ip-address=\${VLLM_HOST_IP} --address=${HEAD_NODE_IP}:${PORT} --num-cpus=64 --num-gpus=8 --block
" &


# ML task
CMD=$(tr -d "\n" << EOF
echo "[DEBUG] testing CMD"
export NCCL_DEBUG=info ;
export HIP_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES} ; 
unset ROCR_VISIBLE_DEVICES ;
accelerate launch
    --multi_gpu
    --use_fsdp 
    --mixed_precision=no
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP
    --fsdp_sharding_strategy=FULL_SHARD
    --fsdp_state_dict_type=SHARDED_STATE_DICT
    --fsdp_activation_checkpointing=true
    --num_machines=${SPLIT}
    --num_processes=$((${SPLIT}*8))
    --machine_rank=\${SLURM_NODEID}
    --main_process_ip=${JOB_MASTER_ADDR}
    --main_process_port=29500
    --gpu_ids=0,1,2,3,4,5,6,7
    --dynamo_backend=no 
    scripts/train_grpo_trl.py --server_ip=${HEAD_NODE_IP} &
wait
EOF
)
srun -w $NODES_FIRST_SPLIT --nodes=$SPLIT --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=64 bash -c "$CMD" & 
wait
