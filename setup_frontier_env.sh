#!/bin/bash
set -e

###############################################
# Frontier AI Environment Setup Script
###############################################

# ==== USER CONFIGURATION ====
ENV_PATH="xxxxxx/xxxx"   # change if needed
REQ_FILE="requirements.txt"               # must NOT contain torch, torchvision, torchaudio
PYTHON_VERSION="3.12"
TORCH_VER="2.8.0"
VISION_VER="0.23.0"
AUDIO_VER="2.8.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.4"

echo "=============================================================="
echo " 1. Loading Frontier-required modules"
echo "=============================================================="

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

echo "=============================================================="
echo " 2. Creating Conda environment at: $ENV_PATH"
echo "=============================================================="

conda create -y -p "$ENV_PATH" python=$PYTHON_VERSION -c conda-forge

echo "=============================================================="
echo " 3. Activating environment"
echo "=============================================================="

# Note: using 'source' ensures this works in scripts
source activate "$ENV_PATH"

echo "=============================================================="
echo " 4. Installing PyTorch ROCm wheels"
echo "=============================================================="

pip install \
    torch==$TORCH_VER \
    torchvision==$VISION_VER \
    torchaudio==$AUDIO_VER \
    --index-url "$TORCH_INDEX_URL"

echo "=============================================================="
echo " 5. (Optional) Installing mpi4py using Frontier-safe build"
echo "=============================================================="

if grep -q "mpi4py" "$REQ_FILE"; then
    echo "Detected mpi4py in requirements.txt — installing safely..."

    module unload rocm

    MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

    module load rocm
else
    echo "Skipping mpi4py (not in requirements.txt)"
fi

echo "=============================================================="
echo " 6. Installing remaining Python packages"
echo "=============================================================="

if [[ -f "$REQ_FILE" ]]; then
    # Remove torch packages from requirements automatically (safety)
    grep -v -E "torch|torchvision|torchaudio" "$REQ_FILE" > tmp_req.txt

    pip install -r tmp_req.txt
    rm tmp_req.txt
else
    echo "WARNING: $REQ_FILE not found. Skipping pip install."
fi

echo "=============================================================="
echo " INSTALLATION COMPLETE!"
echo " To activate the environment:"
echo ""
echo "     source activate $ENV_PATH"
echo ""
echo "=============================================================="
