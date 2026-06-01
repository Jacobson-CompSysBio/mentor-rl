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
module load xpmem/1.0.1-1.5_1_gfb6998056825
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

CLI11_DIR="${REPO_ROOT}/external/cli11"
if [[ ! -f "${CLI11_DIR}/CMakeLists.txt" || ! -f "${CLI11_DIR}/include/CLI/CLI.hpp" ]]; then
  echo "Could not find vendored CLI11 source at ${CLI11_DIR}" >&2
  echo "Expected CLI11 v2.3.2 under external/cli11 for offline Frontier builds." >&2
  exit 1
fi

XPMEM_LIB="${XPMEM_LIB:-/opt/xpmem/lib64/libxpmem.so}"
if [[ ! -f "${XPMEM_LIB}" ]]; then
  echo "Could not find XPMEM library at ${XPMEM_LIB}" >&2
  echo "Set XPMEM_LIB to the correct libxpmem.so path." >&2
  exit 1
fi

MPI_GTL_HSA_LIB="${MPI_GTL_HSA_LIB:-}"
if [[ -z "${MPI_GTL_HSA_LIB}" ]]; then
  MPI_GTL_DIR_FROM_ENV="${PE_MPICH_GTL_DIR_amd_gfx90a:-}"
  MPI_GTL_DIR_FROM_ENV="${MPI_GTL_DIR_FROM_ENV#-L}"
  for candidate in \
    "${CRAY_MPICH_ROOTDIR:-}/gtl/lib/libmpi_gtl_hsa.so" \
    "${MPI_GTL_DIR_FROM_ENV}/libmpi_gtl_hsa.so" \
    /opt/cray/pe/mpich/*/gtl/lib/libmpi_gtl_hsa.so; do
    if [[ -f "${candidate}" ]]; then
      MPI_GTL_HSA_LIB="${candidate}"
      break
    fi
  done
fi
if [[ ! -f "${MPI_GTL_HSA_LIB}" ]]; then
  echo "Could not find MPI GTL HSA library." >&2
  echo "Set MPI_GTL_HSA_LIB to the correct libmpi_gtl_hsa.so path." >&2
  exit 1
fi

BUILD_DIR="${RWR_DIR}/build_frontier"
SOURCE_STAGE="${RWR_DIR}/build_frontier_source"
BUILD_JOBS="${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

rm -rf "${BUILD_DIR}" "${SOURCE_STAGE}"
mkdir -p "${SOURCE_STAGE}"

cp -a "${RWR_DIR}/apps" "${SOURCE_STAGE}/apps"
ln -s "${RWR_DIR}/libs" "${SOURCE_STAGE}/libs"
cp "${RWR_DIR}/CMakeLists_frontier.txt" "${SOURCE_STAGE}/CMakeLists.txt"
find "${SOURCE_STAGE}/apps" -name CMakeLists.txt -print0 | xargs -0 sed -i \
  -e "s#/usr/lib64/libxpmem\\.so#${XPMEM_LIB}#g" \
  -e "s#/opt/cray/pe/mpich/8\\.1\\.30/gtl/lib/libmpi_gtl_hsa\\.so#${MPI_GTL_HSA_LIB}#g"

cmake -S "${SOURCE_STAGE}" -B "${BUILD_DIR}" \
  -DENABLE_LLVM_COVERAGE=OFF \
  -DBUILD_TESTS=OFF \
  -DUSE_HIP=ON \
  -DUSE_OPENMP=ON \
  -DUSE_MPI=ON \
  -DFETCHCONTENT_SOURCE_DIR_CLI11="${CLI11_DIR}" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.2.4/bin/hipcc

cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
