#!/bin/bash
#SBATCH --job-name=dequantize_gpt_oss
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -o logs/%x-%j.out # Out Path
#SBATCH -e logs/%x-%j.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

export LD_LIBRARY_PATH=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

python utils/bf16_model.py