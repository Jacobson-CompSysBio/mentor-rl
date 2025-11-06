#!/bin/bash
#SBATCH --job-name=mentor_rl_data_gen
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

# --- Define Paths ---
DATA_PATH="/lustre/orion/syb114/proj-shared/Personal/smithkp/sandbox/mentor-rl/data/test_multiplex10/"

# --- Define Files ---
flist="/lustre/orion/syb114/proj-shared/Personal/smithkp/sandbox/mentor-rl/data/test_multiplex10/multiplex10.flist"
multiplex_json="mentor_rl_test_multiplex10.json"
distance_matrix="/lustre/orion/syb114/proj-shared/Personal/smithkp/sandbox/mentor-rl/data/test_multiplex10/spearman_dist_matrix.tsv"
qas_filename="mentor_rl_test_distance_qas10.json"

# Create JSON representation of multiplex
python -m scripts.textualize_network \
  --flist $flist \
  --output_dir $DATA_PATH \
  --output_file $multiplex_json

# prepend data path to json file name
multiplex_json="${DATA_PATH}${multiplex_json}"

# Create QAs from distance matrix and JSON
python -m scripts.distance_qa_generator \
  --distance_matrix $distance_matrix \
  --multiplex $multiplex_json \
  --output_dir $DATA_PATH \
  --output_file $qas_filename
