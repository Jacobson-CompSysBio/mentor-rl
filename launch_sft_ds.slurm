#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J mentor-sft-ds
#SBATCH -N 2
#SBATCH -t 0:45:00
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
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

# sbcast conda env to node-local NVMe
sbcast -pf env.tar.gz /mnt/bb/$USER/env.tar.gz
srun -N"${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "/mnt/bb/${USER}/env"
srun -N"${SLURM_NNODES}" --ntasks-per-node=1 -c56 tar --use-compress-program=pigz -xf "/mnt/bb/${USER}/env.tar.gz" -C "/mnt/bb/${USER}/env"
source "/mnt/bb/$USER/env/bin/activate" "/mnt/bb/${USER}/env"
srun -N"${SLURM_NNODES}" --ntasks-per-node=1 conda-unpack

#### CACHE ####
export HF_HOME="/mnt/bb/$USER/hf_cache_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="/mnt/bb/$USER/ds_cache_${SLURM_JOB_ID}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES=1

######## COMMUNICATION / FABRIC (Frontier: Slingshot/CXI via OFI) ########
export NCCL_DEBUG=INFO

# If anything falls back to TCP/Gloo, keep it on the high-speed NIC
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export DEEPSPEED_COMM_SOCKET_IFNAME=hsn0


######## MASTER / RENDEZVOUS ########
# Stable rendezvous port and master IP taken from first node in allocation
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# Process counts
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"   # fallback to 8 if not set

######## LAUNCH ########
# Build accelerate command (DeepSpeed multi-node)
LAUNCHER="accelerate launch \
  --config_file ds_config.yaml \
  --deepspeed_multinode_launcher standard \
  --num_processes ${GPUS_PER_NODE} \
  --num_machines ${SLURM_NNODES} \
  --machine_rank \$SLURM_PROCID \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT}"

PROG="scripts/train_sft_ds.py"
CMD="${LAUNCHER} ${PROG}"

echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}  NNODES=${SLURM_NNODES}"
echo "Launching model..."

# Exactly one accelerator launcher per node; it locally spawns $GPUS_PER_NODE ranks
srun -N "${SLURM_NNODES}" -n "${SLURM_NNODES}" --ntasks-per-node=1 \
     --gpu-bind=closest \
     --export=ALL,MASTER_ADDR="${MASTER_ADDR}",MASTER_PORT="${MASTER_PORT}" \
     bash -c "${CMD}"

####### LAUNCHING ########
## how does deepspeed define machines? change parameters to figure this out
#export LAUNCHER="accelerate launch \
#	--config_file ds_config.yaml \
#	--deepspeed_multinode_launcher standard \
#	--num_processes $WORLD_SIZE \
#	--num_machines $NNODES \
#	--machine_rank \$SLURM_PROCID \
#	--main_process_ip $MASTER_ADDR \
#	--main_process_port $MASTER_PORT \
#	"
#export PROG="scripts/train_sft_ds.py"
#export CMD="$LAUNCHER $PROG"
#
##srun call
#echo "Launching model..."
#srun bash -c "$CMD"


#srun -N ${NNODES} -n ${WORLD_SIZE} \
#	--gpus-per-task=1 \
#	--ntasks-per-node=${GPUS_PER_NODE} \
#	--cpus-per-task=7 \
#	--gpu-bind=closest --cpu-bind=cores \
#	accelerate launch \
#		--config_file ds_config.yaml \
#		--num_processes 1 \
#		--num_machines 1 \
#		${PROG}

