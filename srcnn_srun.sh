#!/bin/bash
#SBATCH -A cli138
#SBATCH -J srcnn
#SBATCH -o logs/srcnn-%j.out
#SBATCH -e logs/srcnn-%j.err
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 02:00:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1

# Master address and port
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500   # <-- YOU MUST ADD THIS

# Fix MIOpen cache
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# Run 1 task (1 GPU)
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
     --export=ALL,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,HIP_VISIBLE_DEVICES=$SLURM_LOCALID \
     python3 -u SRCNN_frontier.py \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --base-dir "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data" \
        --dir-elev "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM" \
        --exp "SRCNN_v1" \
        --var "tmax_dy" \
        --year-start 1980 \
        --year-end 1981 \
        --epochs 50 \
        --batch-size 8 \
        --amp
