#!/bin/bash

# Diagnose real-backend loading for Uni3D / PTv3 on a GPU node.
# Usage:
#   bash slurm/run_diagnose_real_backends.sh

set -euo pipefail

# Use an absolute repo path so SLURM's spool copy does not break config loading.
PROJECT_ROOT="${PROJECT_ROOT:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics}"
source "$PROJECT_ROOT/slurm/config.sh"

DIAG_MEM_GB="${DIAG_MEM_GB:-48G}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/nvidia/cuda/12.9}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.6}"
BACKEND="${BACKEND:-both}"
FORWARD_SMOKE="${FORWARD_SMOKE:-1}"

sbatch $GPU_NODE_ARGS --mem="$DIAG_MEM_GB" -c2 --time=02:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="diag_3d_backends" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && module load \"$CUDA_MODULE\" && export CUDA_HOME=\"$CUDA_ROOT\" && export PATH=\"\$CUDA_HOME/bin:\$PATH\" && export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\" && export TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" TMPDIR=\"$TMP_ROOT\" TMP=\"$TMP_ROOT\" TEMP=\"$TMP_ROOT\" TORCH_EXTENSIONS_DIR=\"$TORCH_EXTENSIONS_DIR\" && rm -rf \"$TORCH_EXTENSIONS_DIR\"/pointnet2_ops* \"$TORCH_EXTENSIONS_DIR\"/_ptv3_runtime* && python3 scripts/diagnose_real_backends.py --backend \"$BACKEND\" $( [ \"$FORWARD_SMOKE\" = \"1\" ] && printf -- \"--forward-smoke\" )'"
