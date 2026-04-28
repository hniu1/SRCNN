#!/bin/bash
#SBATCH -A cli138
#SBATCH -J srcnn-infer
#SBATCH -o ./logs/srcnn-infer-%j.out
#SBATCH -e ./logs/srcnn-infer-%j.err
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

conda activate /lustre/orion/proj-shared/cli138/7hn/envs/torch_rocm

export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_IB_DISABLE=1

# Fix MIOpen cache to avoid shared sqlite cache I/O issues on Frontier
export MIOPEN_USER_DB_PATH="/tmp/miopen-cache-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
rm -rf "$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Runtime knobs (override at submit time, e.g. EXP=SRCNN_v1 SPLIT=val sbatch srcnn_infer_srun.sh)
# EXP: experiment name used to locate cache/checkpoint directories.
# SPLIT: dataset split to run inference on (train or val).
# BATCH_SIZE: inference batch size per GPU.
# NUM_WORKERS: DataLoader CPU worker processes (0 is safest on Frontier).
# USE_AMP: set 1 to enable AMP/bfloat16 inference, 0 to disable.
# OUTPUT_PREFIX: prefix for metrics/scaled artifact names.
EXP=${EXP:-SRCNN_v1}
SPLIT=${SPLIT:-val}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
USE_AMP=${USE_AMP:-0}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-infer}

PATH_OUTPUT=${PATH_OUTPUT:-./output/${EXP}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./checkpoints_${EXP}}

EXTRA_ARGS=""
if [ "$USE_AMP" -eq 1 ]; then
  EXTRA_ARGS="$EXTRA_ARGS --amp"
fi

srun python3 -u srcnn_infer.py \
  --exp "$EXP" \
  --split "$SPLIT" \
  --path-output "$PATH_OUTPUT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-prefix "$OUTPUT_PREFIX" \
  $EXTRA_ARGS
