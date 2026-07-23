#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --time=00:15:00

set -e # Exit immediately if a pipeline returns a non-zero status.

# Usage: sbatch experiment_job.sh <experiment_dir> <start_iter> <communication_socket>
EXP_DIR=$1
ITER=$2
SOCK=$3

# CUDA_SOURCE_PATH=/path/to/cuda/setup/script.sh
# CONDA_PATH=/path/to/conda/directory

if [ -v CUDA_SOURCE_PATH ]; then
    source $CUDA_SOURCE_PATH
else
    echo "CUDA_SOURCE_PATH not set, CUDA must already be configured"
fi

if [ -v CONDA_PATH ]; then
    source $CONDA_PATH/etc/profile.d/conda.sh
else
    echo "CONDA_PATH not set, conda must already be active"
fi

conda activate nccl

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Wait up to 60 seconds for NFS to sync the generated directory and source files
TIMEOUT=60
ELAPSED=0
while [ ! -f "$EXP_DIR/experiment.cu" ]; do
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Error: Timeout ($TIMEOUT seconds) waiting for NFS to sync $EXP_DIR/experiment.cu" >&2
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

cd $EXP_DIR

nvcc experiment.cu experiment.pb.cc -o experiment.bin \
    -ccbin /usr/bin/gcc -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -std=c++20 -lnccl -lzmq -lprotobuf \
    -labsl_log_internal_check_op -labsl_log_internal_message

srun --unbuffered -n 1 ./experiment.bin $ITER $SOCK
