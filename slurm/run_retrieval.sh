#!/bin/bash

# Submit the retrieval MVP as a SLURM job.
# Usage:
#   bash slurm/run_retrieval.sh
#   bash slurm/run_retrieval.sh uni3d
#   bash slurm/run_retrieval.sh pose_descriptor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ENCODER="${1:-pose_descriptor}"
DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask}"
K="${K_OVERRIDE:-1}"

case "$ENCODER" in
  uni3d|ptv3|point_transformer_v3)
    NODE_ARGS="$GPU_NODE_ARGS"
    ;;
  *)
    NODE_ARGS="$CPU_NODE_ARGS"
    ;;
esac

sbatch $NODE_ARGS --mem=12G -c2 --time=06:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="retr_$ENCODER" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/run_retrieval_mvp.py --dataset-root \"$DATASET_ROOT\" --encoder \"$ENCODER\" --k \"$K\"'"

