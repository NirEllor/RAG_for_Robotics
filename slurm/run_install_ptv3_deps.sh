#!/bin/bash

# Build the Python dependencies that PTv3 needs when no binary wheel is
# available for the current torch/CUDA combination.
#
# Submit with:
#   chmod +x slurm/run_install_ptv3_deps.sh
#   sbatch slurm/run_install_ptv3_deps.sh
#
# This job intentionally keeps compilation serial and lowers optimization to
# reduce peak memory usage during source builds.

#SBATCH --job-name=ptv3_deps
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ellorwaizner.nir@mail.huji.ac.il

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
source "$PROJECT_ROOT/slurm/config.sh"

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
module load cuda/12.9

export CUDA_HOME="${CUDA_HOME:-/usr/local/nvidia/cuda/12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export MAX_JOBS="${MAX_JOBS:-1}"
export CFLAGS="${CFLAGS:--O0 -g0}"
export CXXFLAGS="${CXXFLAGS:--O0 -g0}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export TMPDIR="$TMP_ROOT"
export TMP="$TMP_ROOT"
export TEMP="$TMP_ROOT"
export TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$(python3 - <<'PY'
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"{major}.{minor}")
else:
    print("8.0")
PY
)"
fi
export TORCH_CUDA_ARCH_LIST

echo "CUDA_HOME=$CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
python3 -c "import torch; print('torch=', torch.__version__); print('torch.cuda=', torch.version.cuda)"

python3 -m pip install --no-build-isolation --no-cache-dir --verbose \
  git+https://github.com/rusty1s/pytorch_scatter.git \
  git+https://github.com/rusty1s/pytorch_sparse.git \
  git+https://github.com/rusty1s/pytorch_cluster.git

python3 -m pip install --no-cache-dir torch-geometric

python3 - <<'PY'
import torch_scatter
import torch_sparse
import torch_cluster
import torch_geometric
print("PTv3 deps OK")
PY
