#!/bin/bash

# Convenience launcher:
# 1) download/extract the RLBench raw mirror on the cluster
# 2) build the processed multitask dataset after that job finishes
#
# Usage:
#   ./slurm/run_prepare_cluster_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

FETCH_JOB_ID=$(sbatch $CPU_NODE_ARGS --parsable --mem=24G -c4 --time=2-00:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="fetch_rlbench" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/fetch_rlbench_mirror.py --stage-dir \"$RAW_STAGE_DIR\" --extract-root \"$RAW_DATA_ROOT\" --workers 4'")

echo "Submitted raw fetch job: $FETCH_JOB_ID"

BUILD_JOB_ID=$(sbatch $CPU_NODE_ARGS --parsable --dependency=afterok:$FETCH_JOB_ID --mem=20G -c4 --time=12:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="build_ds" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/build_multitask_dataset.py --config \"$PROJECT_ROOT/configs/dataset/rlbench_multitask.yaml\" --dataset-root \"$DATA_ROOT/processed/v2_multitask\" --overwrite'")

echo "Submitted build job after fetch: $BUILD_JOB_ID"
