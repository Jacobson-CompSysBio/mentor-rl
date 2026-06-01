#!/bin/bash
#SBATCH -A SYB114
#SBATCH -J compile_rwr_hpc
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -c 8

set -euo pipefail

module load cray-mpich
module load craype-accel-amd-gfx90a
module load rocm/6.2.4
export MPICH_GPU_SUPPORT_ENABLED=1

if [[ -n "${MENTOR_RL_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "${MENTOR_RL_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/external/rwr_hpc" ]]; then
  REPO_ROOT="$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
fi

RWR_DIR="${REPO_ROOT}/external/rwr_hpc"
if [[ ! -d "${RWR_DIR}" ]]; then
  echo "Could not find RWR++ source at ${RWR_DIR}" >&2
  echo "Submit from the mentor-rl repo root or set MENTOR_RL_ROOT." >&2
  exit 1
fi

BUILD_DIR="${RWR_DIR}/build_frontier"
SOURCE_STAGE="${RWR_DIR}/build_frontier_source"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

rm -rf "${BUILD_DIR}" "${SOURCE_STAGE}"
mkdir -p "${SOURCE_STAGE}"

ln -s "${RWR_DIR}/apps" "${SOURCE_STAGE}/apps"
ln -s "${RWR_DIR}/libs" "${SOURCE_STAGE}/libs"
cp "${RWR_DIR}/CMakeLists_frontier.txt" "${SOURCE_STAGE}/CMakeLists.txt"

cmake -S "${SOURCE_STAGE}" -B "${BUILD_DIR}" \
  -DENABLE_LLVM_COVERAGE=OFF \
  -DBUILD_TESTS=OFF \
  -DUSE_HIP=ON \
  -DUSE_OPENMP=ON \
  -DUSE_MPI=ON \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.2.4/bin/hipcc

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
