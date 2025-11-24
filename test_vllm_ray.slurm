#!/bin/bash
#SBATCH --job-name=vllm-frontier
#SBATCH -A SYB114
#SBATCH -N 2
#SBATCH -t 0:15:00
#SBATCH -p batch
#SBATCH -C nvme
#SBATCH -q debug
#SBATCH --gpus-per-node=8
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# --- Load Modules ---
module load PrgEnv-gnu/8.6.0
module load rocm/6.3.1
module load craype-accel-amd-gfx90a

# --- Activate Conda ---
source /lustre/orion/syb111/proj-shared/Environments/source_miniconda_frontier.sh
source activate /lustre/orion/syb111/world-shared/environments/pytorch-rocm

# --- Set Paths ---
export ROCM_HOME=/opt/rocm-6.3.1
export LIBFABRIC_DIR=/opt/cray/libfabric/1.22.0/lib64
# use the rocm rccl fork on frontier
export AWS_OFI_PLUGIN_DIR=/lustre/orion/syb111/proj-shared/Personal/krusepi/packages/aws-ofi-rccl/lib
# locate cxi lib dir
export CXI_LIB_DIR=$(ls -d /opt/cray/pe/cxi/*/lib64 2>/dev/null | tail -n1 || true)
if [[ -z "${CXI_LIB_DIR}" && -d /opt/cray/pe/cxi/default/lib64 ]]; then
    export CXI_LIB_DIR=/opt/cray/pe/cxi/default/lib64
fi

# sanity checks
[[ -f "${AWS_OFI_PLUGIN_DIR}/librccl-net.so" ]] || { echo "ERROR: ${AWS_OFI_PLUGIN_DIR}/librccl-nest.so not found"; exit 1; }
[[ -d "${CXI_LIB_DIR}" ]] || { echo "ERROR: CXI lib dir not found"; exit 1; }

# --- Networking / OFI / Slingshot ---
export FI_PROVIDER=cxi
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_ATS=0
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=131072
export FI_CXI_RX_MATCH_MODE=1

# --- RCCL / OFI Plugin ---
export VLLM_NCCL_SO_PATH=${ROCM_HOME}/lib/librccl.so
export NCCL_NET=OFI
export NCCL_COLLNET_DISABLE=1
export NCCL_TIMEOUT=3600
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1 # slingshot ≠ ib
export NCCL_P2P_DISABLE=0

# critical for cxi: disable HMEM path
export NCCL_OFI_USE_HMEM=0
export NCCL_OFI_HMEM_DISABLE=1

# debug level while testing (drop to WARN later)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# --- Gloo rendezvous ---
export GLOO_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_FAMILY=AF_INET
export GLOO_IPV6=0

# --- Visible GPUs and Ray "don't touch" ---
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCR_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}
export CUDA_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# --- Library search order: plugin -> cxi -> libfabric -> rocm ---
export LD_LIBRARY_PATH=${AWS_OFI_PLUGIN_DIR}:${CXI_LIB_DIR}:${LIBFABRIC_DIR}:${ROCM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# link check to show libcxi/libfabric/amdhip64
ldd ${AWS_OFI_PLUGIN_DIR}/librccl-net.so | egrep 'libcxi|libfabric|amdhip64|rccl' || true

# --- Model path ---
export MODEL_PATH="${MODEL_PATH:-/lustre/orion/syb111/proj-shared/Personal/krusepi/projects/llms/models/Llama-3.2-1B-Instruct}"
[[ -d "${MODEL_PATH}" ]] || { echo "ERROR: MODEL_PATH (${MODEL_PATH}) not found"; exit 1; }

# --- Ray cluster ports ---
export RAY_PORT=${RAY_PORT:-6379}
export RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-7}

# --- Determine nodes/IPs on the hsn network ---
mapfile -t NODELIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE=${NODELIST[0]}
WORKER_NODES=("${NODELIST[@]:1}")

resolve_hsn_ip() { getent hosts "$1-hsn0" | awk '{print $1}'; }
HEAD_IP=$(resolve_hsn_ip "${HEAD_NODE}")
[[ -n "${HEAD_IP}" ]] || { echo "ERROR: failed to resolve head IP"; exit 1; }
export MASTER_ADDR=${HEAD_IP}

# --- Start Ray head/worker (simple) ---
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  bash -lc "ulimit -n 65536; ray stop --force >/dev/null 2>&1 || true; ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --dashboard-port=${RAY_DASHBOARD_PORT} --num-cpus=${RAY_NUM_CPUS} --num-gpus=8 --block" &

for n in "${WORKER_NODES[@]}"; do
  WIP=$(resolve_hsn_ip "${n}") || true
  [[ -n "${WIP}" ]] || { echo "ERROR: failed to resolve ${n} IP"; exit 1; }
  srun --nodes=1 --ntasks=1 -w "${n}" \
    bash -lc "ulimit -n 65536; ray stop --force >/dev/null 2>&1 || true; ray start --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WIP} --num-cpus=${RAY_NUM_CPUS} --num-gpus=8 --block" &
done

sleep 12
ray status --address="${HEAD_IP}:${RAY_PORT}" || true

# --- Run the test on the head node ---
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  bash -lc "python tests/test_vllm_ray_multinode_min.py"

# --- Cleanup on exit ---
trap "srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks=${SLURM_JOB_NUM_NODES} bash -lc 'ray stop --force' >/dev/null 2>&1 || true" EXIT