module load rocm/6.3.1
module load PrgEnv-gnu

#export HF_HOME=/lustre/orion/stf006/scratch/glaser
#export TORCH_EXTENSIONS_DIR=/lustre/orion/stf006/scratch/glaser
export HTTP_PROXY=http://proxy.ccs.ornl.gov:3128
export HTTPS_PROXY=http://proxy.ccs.ornl.gov:3128
export NO_PROXY="127.0.0.1,localhost,localhost.localdomain,.frontier.olcf.ornl.gov"
#export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export CC=`which gcc`
export CXX=`which g++`

export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export VLLM_CACHE_ROOT=/mnt/bb/${USER}/vllm_cache
export TRITON_CACHE_DIR=/mnt/bb/${USER}/triton_cache
export WANDB_CACHE_DIR=/mnt/bb/${USER}/wandb_cache
#export VLLM_WORKER_MULTIPROC_METHOD=spawn

#. $WORLDWORK/stf006/glaser/miniconda3/etc/profile.d/conda.sh
#conda activate grpo

