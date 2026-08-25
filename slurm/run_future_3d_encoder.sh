#!/bin/bash

# GPU-ready template for future Uni3D / Point Transformer V3 runs.
# This script is intentionally a thin launcher so we can keep the future
# encoder jobs consistent with the current retrieval pipeline.
#
# Usage:
#   bash slurm/run_future_3d_encoder.sh uni3d
#   bash slurm/run_future_3d_encoder.sh ptv3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ENCODER="${1:-uni3d}"
DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask}"
K="${K_OVERRIDE:-1}"

case "$ENCODER" in
  uni3d|ptv3|point_transformer_v3)
    ;;
  *)
    echo "Unsupported encoder: $ENCODER"
    echo "Use uni3d or ptv3."
    exit 1
    ;;
esac

sbatch $GPU_NODE_ARGS --mem=30G -c4 --time=12:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="3d_$ENCODER" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/run_retrieval_mvp.py --dataset-root \"$DATASET_ROOT\" --encoder \"$ENCODER\" --k \"$K\"'"

