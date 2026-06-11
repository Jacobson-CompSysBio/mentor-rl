#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J compile_rwr_hpc
#SBATCH -o compile_rwr-%x.o
#SBATCH -e compile_rwr-%x.e
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smithkp@ornl.gov

module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm/6.2.4
export MPICH_GPU_SUPPORT_ENABLED=1

cd "/lustre/orion/syb114/proj-shared/Personal/smithkp/sandbox/rwr_hpc/"

rm -rf build_frontier

mkdir build_frontier && cd build_frontier

cmake -DENABLE_LLVM_COVERAGE=OFF -DBUILD_TESTS=OFF -DUSE_HIP=ON -DUSE_OPENMP=ON -DUSE_MPI=ON -DCMAKE_CXX_COMPILER=/opt/rocm-6.2.4/bin/hipcc ..

make -j
