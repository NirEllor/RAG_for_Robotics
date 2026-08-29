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

sbatch $CPU_NODE_ARGS --mem="$EVAL_MEM_GB" -c2 --time=06:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="retr_eval" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/evaluate_retrieval.py --dataset-root \"$DATASET_ROOT\" --annotations \"$ANNOTATIONS\" --methods $METHODS --ks $KS'"
