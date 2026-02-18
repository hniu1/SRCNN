#!/bin/bash
#SBATCH -A cli138
#SBATCH -J srcnn
#SBATCH -o ./logs/srcnn-%j.out
#SBATCH -e ./logs/srcnn-%j.err
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1        # 1 task per GPU
#SBATCH --gpus-per-task=1          # exactly 1 GPU per rank
#SBATCH --cpus-per-task=6          # CPU cores per rank
#SBATCH -t 00:30:00

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

# Fix MIOpen cache
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# Downscaling mode:
# - 1to0p25: low=1degto0p25deg, high=0p25deg this mode is not yet fully supported yet as it requires different srcnn structure
# - 0p25to0p0416: low=0p25degto0p0416deg, high=trim
DOWNSCALE_MODE=${DOWNSCALE_MODE:-0p25to0p0416}

# Run 1 task (1 GPU)
srun \
     python3 -u SRCNN_frontier.py \
        --master_addr $MASTER_ADDR \
        --master_port=3442 \
        --base-dir "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/data" \
        --dir-elev "/lustre/orion/proj-shared/cli138/dr6/NA-Downscaling/DEM" \
        --exp "SRCNN_v1" \
      --downscale-mode "$DOWNSCALE_MODE" \
        --var "tmax_dy" \
        --year-start 1980 \
        --year-end 1981 \
        --epochs 50 \
      --batch-size 8
