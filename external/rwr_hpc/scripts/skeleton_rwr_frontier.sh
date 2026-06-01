#!/bin/bash
#SBATCH -A <project_name>
#SBATCH -J <job_name>
#SBATCH -o logs/-%x.o
#SBATCH -e logs/-%x.e
#SBATCH -t <run_time>
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email>

module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm/6.2.4
export MPICH_GPU_SUPPORT_ENABLED=1

ROOT_DIR="<root directory of install>"
APP=${ROOT_DIR}"/build/apps/rwr/rwr_wrapper"`

flist="<full_flist_path>"
out_dir="<full_ouput_path>" # Optional
seed_file="<full_seed_path>" # Optional

OMP_NUM_THREADS=7 srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest $APP -f $flist -s $seed_file -o $out_dir
# OMP_NUM_THREADS=7 srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest $APP -f $flist -o $out_dir
