#!/bin/bash

# Build pointnet2_ops inside the cluster venv using the local CUDA module.
#
# This job is useful for enabling the real Uni3D backend, which depends on
# pointnet2_ops compiling successfully against the cluster CUDA toolchain.
#
# Submit with:
#   chmod +x slurm/run_build_pointnet2_ops.sh
#   sbatch slurm/run_build_pointnet2_ops.sh
#
# Optional overrides:
#   CUDA_MODULE=cuda/12.9
#   CUDA_ROOT=/usr/local/nvidia/cuda/12.9

#SBATCH --job-name=pnet2_build
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ellorwaizner.nir@mail.huji.ac.il

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
source "$PROJECT_ROOT/slurm/config.sh"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/nvidia/cuda/12.9}"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
  fi
fi

module purge
module load "$CUDA_MODULE"

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"

python3 -c "import torch; print('torch.cuda=', torch.version.cuda)"

python3 -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

python3 -c "import pointnet2_ops; print('pointnet2_ops OK')"
