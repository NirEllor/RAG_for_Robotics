#!/bin/bash

# Submit retrieval evaluation as a SLURM job.
# Usage:
#   bash slurm/run_evaluate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask}"
ANNOTATIONS="${ANNOTATIONS_OVERRIDE:-$PROJECT_ROOT/configs/evaluation/rlbench_reach_target_hand_labels.json}"
METHODS="${METHODS_OVERRIDE:-random pose_descriptor rgb_histogram global_color geometry_only uni3d ptv3}"
KS="${KS_OVERRIDE:-1 2 3}"
EVAL_MEM_GB="${EVAL_MEM_GB:-32G}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-06:00:00}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/nvidia/cuda/12.9}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.6}"
SBATCH_NODELIST_OVERRIDE="${SBATCH_NODELIST_OVERRIDE:-}"

NODE_FLAG=""
[ -n "$SBATCH_NODELIST_OVERRIDE" ] && NODE_FLAG="--nodelist=$SBATCH_NODELIST_OVERRIDE"

sbatch $GPU_NODE_ARGS $NODE_FLAG --mem="$EVAL_MEM_GB" -c2 --time="$EVAL_TIME_LIMIT" \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="retr_eval" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && module load \"$CUDA_MODULE\" && export CUDA_HOME=\"$CUDA_ROOT\" && export PATH=\"\$CUDA_HOME/bin:\$PATH\" && export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\" && export TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" TMPDIR=\"$TMP_ROOT\" TMP=\"$TMP_ROOT\" TEMP=\"$TMP_ROOT\" TORCH_EXTENSIONS_DIR=\"$TORCH_EXTENSIONS_DIR\" && rm -rf \"$TORCH_EXTENSIONS_DIR\"/pointnet2_ops* \"$TORCH_EXTENSIONS_DIR\"/_ptv3_runtime* && python3 scripts/evaluate_retrieval.py --dataset-root \"$DATASET_ROOT\" --annotations \"$ANNOTATIONS\" --methods $METHODS --ks $KS'"
