#!/bin/bash

# Prepare a clean Pointcept/PTv3 environment aligned to the official Pointcept
# CUDA 12.4 / PyTorch 2.5 stack.
#
# Run this on the cluster login node (or any shell where Python is available).
# It intentionally does not submit a SLURM job because it creates a private
# environment on shared storage.
#
# Usage:
#   chmod +x slurm/run_prepare_ptv3_pointcept_env.sh
#   bash slurm/run_prepare_ptv3_pointcept_env.sh
#
# Optional overrides:
#   PTV3_ENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env
#   POINTCEPT_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept
#   PYTHON_VERSION=3.10

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
POINTCEPT_ROOT="${POINTCEPT_ROOT:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept}"
ENV_DIR="${PTV3_ENV_DIR:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
USE_CONDA=0

if ! command -v conda >/dev/null 2>&1; then
  CONDA_SH_CANDIDATES=(
    "${CONDA_SH:-}"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/opt/miniconda3/etc/profile.d/conda.sh"
    "/usr/share/miniconda3/etc/profile.d/conda.sh"
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
  )

  for candidate in "${CONDA_SH_CANDIDATES[@]}"; do
    if [ -n "${candidate:-}" ] && [ -f "$candidate" ]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
fi

if command -v conda >/dev/null 2>&1; then
  USE_CONDA=1
fi

if [ "$USE_CONDA" -eq 0 ] && ! command -v python3 >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load python/native >/dev/null 2>&1 || true
  fi
fi

if [ "$USE_CONDA" -eq 0 ] && ! command -v python3 >/dev/null 2>&1; then
  cat <<'EOF'
Neither conda nor python3 is available in this shell.
Try loading the cluster Python module first, for example:
  module avail | grep -i python
  module load python/native
Then rerun the script.
EOF
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"

if [ "$USE_CONDA" -eq 1 ]; then
  eval "$(conda shell.bash hook)"

  if [ -d "$ENV_DIR" ]; then
    echo "Environment directory already exists: $ENV_DIR"
    echo "Remove it manually if you want a fresh rebuild, then rerun."
    exit 1
  fi

  echo "Creating conda env: $ENV_DIR"
  conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION" pip ninja
  conda activate "$ENV_DIR"

  python -m pip install --upgrade pip setuptools wheel

  echo "Installing core PyTorch + CUDA stack..."
  conda install -y -c pytorch -c nvidia \
    pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4

  echo "Installing Pointcept/PyG dependencies..."
  conda install -y -c conda-forge \
    h5py pyyaml tensorboard tensorboardx wandb yapf addict einops scipy \
    plyfile termcolor timm ftfy regex tqdm matplotlib black open3d peft

  conda install -y -c pyg \
    torch-cluster torch-scatter torch-sparse

  python -m pip install torch-geometric spconv-cu124
else
  if [ -d "$ENV_DIR" ]; then
    echo "Environment directory already exists: $ENV_DIR"
    echo "Remove it manually if you want a fresh rebuild, then rerun."
    exit 1
  fi

  echo "Creating venv env: $ENV_DIR"
  python3 -m venv "$ENV_DIR"
  # shellcheck disable=SC1090
  source "$ENV_DIR/bin/activate"

  python -m pip install --upgrade pip setuptools wheel

  echo "Installing core PyTorch + CUDA stack from wheels..."
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0

  echo "Installing PyG wheels..."
  python -m pip install --no-build-isolation \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

  echo "Installing Pointcept/PyPI dependencies..."
  python -m pip install \
    h5py pyyaml tensorboard tensorboardx wandb yapf addict einops scipy \
    plyfile termcolor timm ftfy regex tqdm matplotlib black open3d peft \
    spconv-cu124
fi

python - <<'PY'
import torch
import torch_geometric
import spconv.pytorch as spconv
print("torch:", torch.__version__)
print("torch.cuda:", torch.version.cuda)
print("torch_geometric:", torch_geometric.__version__)
print("spconv import OK:", spconv is not None)
PY

echo
echo "PTv3 clean env is ready."
echo "Next steps:"
if [ "$USE_CONDA" -eq 1 ]; then
  echo "  1) conda activate '$ENV_DIR'"
else
  echo "  1) source '$ENV_DIR/bin/activate'"
fi
echo "  2) cd '$PROJECT_ROOT'"
echo "  3) run a short smoke:"
echo "     BACKEND=ptv3 FORWARD_SMOKE=1 SBATCH_NODELIST_OVERRIDE=<node> bash slurm/run_diagnose_real_backends.sh"
