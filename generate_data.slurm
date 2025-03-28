#!/bin/bash
#SBATCH -A syb114 # Project Name
#SBATCH -J grpo_edge_pred_training # Job Name
#SBATCH -N 1 # Number of Nodes
#SBATCH -p extended # extended queue for longer walltime
#SBATCH -t 00:10:00 # Walltime
#SBATCH -o logs/%x.out # Out Path
#SBATCH -e logs/%x.err # Err Path
#SBATCH --open-mode=truncate # Overwrite .out/.err
#SBATCH --mail-type=BEGIN 
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=smithp@ornl.gov

# exit on errors
set -e

# activate modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# set working directory to "scripts" for proper relative path function
cd /lustre/orion/syb111/proj-shared/Projects/mentor-rl/scripts

# activate conda
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh 
echo "conda activated"

# activate env
source /lustre/orion/syb111/proj-shared/Environments/frontier/pytorch-rocm/activate.rc
echo "environment activated"

# run train.py
python edge_data_generator.py
echo "script complete"
