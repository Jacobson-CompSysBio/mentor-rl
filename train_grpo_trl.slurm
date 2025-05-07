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

MODEL=/lustre/orion/syb111/proj-shared/Personal/krusepi/llms/models/Llama-4-Scout-17B-16E-Instruct/

CMD=$(tr -d "\n" << EOF
accelerate launch
    --use_fsdp 
    --mixed_precision=no
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP
    --fsdp_sharding_strategy=FULL_SHARD
    --fsdp_state_dict_type=SHARDED_STATE_DICT
    --fsdp_activation_checkpointing=true
    --num_machines=${SLURM_NNODES}
    --num_processes=$((${SLURM_NNODES}*7))
    --machine_rank=\${SLURM_NODEID}
    --main_process_ip=${JOB_MASTER_ADDR}
    --main_process_port=29500
    --gpu_ids=0,1,2,3,4,5,6
    --dynamo_backend=no
train_grpo_trl.py &
VLLM_PORT=29601 ROCR_VISIBLE_DEVICES=7 trl vllm-serve --model ${MODEL} --host=127.0.0.1 --port=8001 --timeout_keep_alive=1200 1> /dev/null &
wait
EOF)

srun -c $CMD 
