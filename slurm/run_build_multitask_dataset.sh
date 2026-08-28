#!/bin/bash

# Submit the multi-task dataset build as a SLURM job.
# This is CPU-heavy and does not require GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CONFIG_PATH="${CONFIG_OVERRIDE:-$PROJECT_ROOT/configs/dataset/rlbench_multitask.yaml}"
OUTPUT_DATASET_ROOT="${OUTPUT_DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask}"
RESUME_FLAG=""
[ "${BUILD_DATASET_RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
TASK_START_FLAG=""
[ -n "${TASK_START_INDEX:-}" ] && TASK_START_FLAG="--task-start-index ${TASK_START_INDEX}"
TASK_END_FLAG=""
[ -n "${TASK_END_INDEX:-}" ] && TASK_END_FLAG="--task-end-index ${TASK_END_INDEX}"

sbatch $CPU_NODE_ARGS --mem=20G -c4 --time=12:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="build_ds" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/build_multitask_dataset.py --config \"$CONFIG_PATH\" --dataset-root \"$OUTPUT_DATASET_ROOT\" --overwrite $RESUME_FLAG $TASK_START_FLAG $TASK_END_FLAG'"
