#!/bin/bash

# Run controlled robustness evaluation on an exported dataset.
# Query observations are perturbed while the retrieval candidate database stays clean.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_ROOT_OVERRIDE="${DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask_subset8}"
ROBUSTNESS_OUTPUT_DIR="${ROBUSTNESS_OUTPUT_DIR:-$OUTPUT_ROOT/evaluation/robustness/v2_multitask_subset8}"
ROBUSTNESS_MEM_GB="${ROBUSTNESS_MEM_GB:-64G}"
ROBUSTNESS_CPUS="${ROBUSTNESS_CPUS:-4}"
ROBUSTNESS_METHODS="${ROBUSTNESS_METHODS:-pose_descriptor rgb_histogram global_color geometry_only uni3d ptv3}"
ROBUSTNESS_CONDITIONS="${ROBUSTNESS_CONDITIONS:-viewpoint occlusion geometry_noise}"
ROBUSTNESS_KS="${ROBUSTNESS_KS:-1 3}"
ROBUSTNESS_MAX_QUERIES="${ROBUSTNESS_MAX_QUERIES:-0}"
SBATCH_NODELIST_OVERRIDE="${SBATCH_NODELIST_OVERRIDE:-}" 

NODE_FLAG=""
[ -n "$SBATCH_NODELIST_OVERRIDE" ] && NODE_FLAG="--nodelist=$SBATCH_NODELIST_OVERRIDE"

sbatch $GPU_NODE_ARGS $NODE_FLAG --gres=gpu:1 --mem="$ROBUSTNESS_MEM_GB" -c"$ROBUSTNESS_CPUS" --time=18:00:00 \
  --job-name="retr_robust" \
  --wrap "bash -lc 'cd "$PROJECT_ROOT" && source "$VENV_DIR/bin/activate" && $COPPELIASIM_ENV python3 scripts/evaluate_robustness.py --dataset-root "$DATASET_ROOT_OVERRIDE" --output-dir "$ROBUSTNESS_OUTPUT_DIR" --methods $ROBUSTNESS_METHODS --conditions $ROBUSTNESS_CONDITIONS --ks $ROBUSTNESS_KS --max-queries $ROBUSTNESS_MAX_QUERIES'"
