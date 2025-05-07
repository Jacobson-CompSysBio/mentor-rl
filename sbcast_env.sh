ENV_NAME=$1
ENV_BASE=`basename ${ENV_NAME}`
TAR_FILE=${ENV_BASE}.tar.bz2
NNODES=${SLURM_NNODES}



if [ ! -f /mnt/bb/${USER}/${TAR_FILE} ]; then
    echo "broadcasting ${TAR_FILE}"
    TIMEFORMAT="%Rs"
#    { time sbcast -pf /lustre/orion/stf006/world-shared/glaser/${TAR_FILE} /mnt/bb/${USER}/${TAR_FILE}; } 2>&1
#
#    if [ ! "$?" == "0" ]; then
#        # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes
#        echo "SBCAST failed!"
#        exit 1
#
#    fi
    { time srun -N ${NNODES} --ntasks-per-node 1 cp ${TAR_FILE} /mnt/bb/${USER}/${TAR_FILE}; } 2>&1
    echo "creating local directory"
    srun -N ${NNODES} --ntasks-per-node 1 mkdir /mnt/bb/${USER}/${ENV_BASE}
    echo "activating global environment"
    . /lustre/orion/stf006/world-shared/glaser/miniconda3/etc/profile.d/conda.sh
    conda activate ${ENV_NAME} # needed for lbzip2
    echo "untaring"
    { time srun -N ${NNODES} --ntasks-per-node 1 --cpus-per-task=64 tar -I lbzip2 -xf /mnt/bb/${USER}/${TAR_FILE} -C /mnt/bb/${USER}/${ENV_BASE}; } 2>&1
fi

echo "activating local environment"
conda activate /mnt/bb/${USER}/${ENV_BASE}
echo "unpacking"
{ time srun -N ${NNODES} --ntasks-per-node 1 conda-unpack; } 2>&1

