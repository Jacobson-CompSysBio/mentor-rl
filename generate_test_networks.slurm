#!/bin/bash
#SBATCH --job-name=network_gen
#SBATCH -A SYB114
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH -q debug
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -o logs/%x-%j.out # Out Path
#SBATCH -e logs/%x-%j.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err

# load conda, activate env
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

# --- Define Inputs ---
NUM_NETWORKS=6
NUM_NODES="5 6 5 5 7 8"
NUM_EDGES="12 13 14 15 16 20"
OUTPUT_DIR="/lustre/orion/syb114/proj-shared/Personal/smithkp/sandbox/mentor-rl/data/test_multiplex10/"
PREFIX="network"

# Create JSON representation of multiplex
python -m scripts.generate_random_networks \
  --num_networks ${NUM_NETWORKS} \
  --num_nodes ${NUM_NODES} \
  --num_edges ${NUM_EDGES} \
  --output_dir ${OUTPUT_DIR} \
  --prefix ${PREFIX}
