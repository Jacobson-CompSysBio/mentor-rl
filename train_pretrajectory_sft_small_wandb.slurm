#!/bin/bash
#SBATCH --job-name=pretraj_sft_small_wandb
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --open-mode=truncate

set -euo pipefail

module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a
module load gcc/12.2.0

export CC=cc
export CXX=CC
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC
export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:${LD_LIBRARY_PATH:-}

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

MODEL_NAME="gpt-oss-20b-bf16"
MODEL_PATH="/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/${MODEL_NAME}"
export DATASET_PATH="${SLURM_SUBMIT_DIR}/data/pretrajectory_sft/v1_medium/train.jsonl"
export VAL_DATASET_PATH="${SLURM_SUBMIT_DIR}/data/pretrajectory_sft/v1_medium/val.jsonl"
export HOLDOUT_DATASET_PATH="${SLURM_SUBMIT_DIR}/data/pretrajectory_sft/v1_medium/test.jsonl"
export DS_CONFIG="${SLURM_SUBMIT_DIR}/config/ds_zero3_lora_no_offload.json"
export PERSISTENT_OUTPUT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/pretrajectory_sft_small_wandb_${SLURM_JOB_ID}"

export LOCAL_SSD_PATH="/mnt/bb/${USER}/pretraj_sft_small_wandb_${SLURM_JOB_ID}"
export LOCAL_MODEL_DIR="${LOCAL_SSD_PATH}/model/${MODEL_NAME}"
export LOCAL_OUTPUT_DIR="${LOCAL_SSD_PATH}/checkpoints"
export LOCAL_CACHE_BASE="${LOCAL_SSD_PATH}/caches"
export WANDB_DIR="${LOCAL_SSD_PATH}/wandb"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${LOCAL_CACHE_BASE}/hf"
export HF_DATASETS_CACHE="${LOCAL_CACHE_BASE}/datasets"
export TRITON_CACHE_DIR="${LOCAL_CACHE_BASE}/triton"
export TORCH_EXTENSIONS_DIR="${LOCAL_CACHE_BASE}/torch_extensions"

export http_proxy=http://proxy.ccs.ornl.gov:3128
export https_proxy="${http_proxy}"
export WANDB_HTTP_TIMEOUT=90
export WANDB_MODE=online
export WANDB_RUN_GROUP="pretrajectory_sft_v1_medium"
export WANDB_TAGS="pretrajectory_sft,v1_medium,small_reporting,gpt_oss_20b_bf16"
export MENTOR_RUN_LABEL="pretraj-small-wandb-${SLURM_JOB_ID}"

export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_OFI_USE_NICLIST=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=WARN
export PYTORCH_ROCM_ARCH=gfx90a
export ROCM_HOME=/opt/rocm-6.3.1
export HSA_XNACK=0
export DS_ACCELERATOR=cuda
export PYTORCH_ALLOC_CONF="expandable_segments:False"

echo "--- Pre-trajectory SFT small W&B run started ---"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node List: ${SLURM_JOB_NODELIST}"
echo "Dataset: ${DATASET_PATH}"
echo "Validation dataset: ${VAL_DATASET_PATH}"
echo "Holdout dataset: ${HOLDOUT_DATASET_PATH}"
echo "Output: ${PERSISTENT_OUTPUT_DIR}"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c "
  set -euo pipefail
  mkdir -p '${LOCAL_MODEL_DIR}' '${LOCAL_OUTPUT_DIR}' '${LOCAL_CACHE_BASE}' '${WANDB_DIR}'
  echo 'Python check:'; which python; python -c 'import torch, transformers, trl, datasets, peft, wandb; print(torch.__version__, transformers.__version__)'
  echo 'Staging model to local NVMe...'
  rsync -a --info=progress2 '${MODEL_PATH}/' '${LOCAL_MODEL_DIR}/'
"

MASTER_ADDR="$(ip -4 addr show hsn0 | awk '/inet /{print $2}' | cut -d/ -f1 | head -n 1)"
if [ -z "${MASTER_ADDR}" ]; then
  MASTER_ADDR="$(hostname -I | awk '{print $1}')"
fi
export MASTER_ADDR
export MASTER_PORT=29500

echo "--- Launching small tracked SFT run ---"
srun --cpu-bind=none --accel-bind=g --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export AMDSMI_GPU_METRICS_CACHE_MS=5000
  if [ "${SLURM_PROCID}" != "0" ]; then
    export WANDB_MODE=disabled
  fi
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
  fi

  python "${SLURM_SUBMIT_DIR}/scripts/train_sft.py" \
    --model_path="${LOCAL_MODEL_DIR}/" \
    --output_dir="${LOCAL_OUTPUT_DIR}" \
    --dataset_path="${DATASET_PATH}" \
    --dataset_subset_size=1024 \
    --eval_dataset_path="${VAL_DATASET_PATH}" \
    --eval_dataset_subset_size=500 \
    --holdout_dataset_path="${HOLDOUT_DATASET_PATH}" \
    --holdout_sample_size=32 \
    --holdout_max_new_tokens=128 \
    --holdout_output_path="${LOCAL_OUTPUT_DIR}/holdout_predictions.jsonl" \
    --holdout_report_path="${LOCAL_OUTPUT_DIR}/holdout_report.json" \
    --run_inference_after_training=True \
    --local_files_only True \
    --deepspeed="${DS_CONFIG}" \
    --seed=900913 \
    --max_steps=50 \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-5 \
    --lr_scheduler_type=cosine \
    --warmup_steps=5 \
    --eval_strategy=steps \
    --eval_steps=25 \
    --logging_steps=5 \
    --save_strategy=no \
    --bf16 \
    --max_length=1024 \
    --ddp_find_unused_parameters False \
    --lora_r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.05
'

echo "--- Copying small-run outputs ---"
mkdir -p "${PERSISTENT_OUTPUT_DIR}"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 rsync -a --info=progress2 "${LOCAL_OUTPUT_DIR}/" "${PERSISTENT_OUTPUT_DIR}/"
if [ -d "${WANDB_DIR}" ]; then
  mkdir -p "${PERSISTENT_OUTPUT_DIR}/wandb"
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 rsync -a --info=progress2 "${WANDB_DIR}/" "${PERSISTENT_OUTPUT_DIR}/wandb/"
fi

echo "--- Cleaning local NVMe ---"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 rm -rf "${LOCAL_SSD_PATH}"

echo "--- Pre-trajectory SFT small W&B run finished ---"
echo "Persistent output: ${PERSISTENT_OUTPUT_DIR}"
