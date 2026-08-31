#!/bin/bash

# Submit the full 19-task multitask dataset build as a SLURM job.
# This is CPU-heavy and does not require GPU.
#
# Usage:
#   bash slurm/run_build_multitask_full_dataset.sh
#
# Optional overrides:
#   OUTPUT_DATASET_ROOT_OVERRIDE=/path/to/output/root
#   BUILD_MEM_GB=24G
#   BUILD_CPUS=4
#   BUILD_DATASET_RESUME=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CONFIG_PATH="${CONFIG_OVERRIDE:-$PROJECT_ROOT/configs/dataset/rlbench_multitask_full.yaml}"
OUTPUT_DATASET_ROOT="${OUTPUT_DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask_full}"
BUILD_MEM_GB="${BUILD_MEM_GB:-24G}"
BUILD_CPUS="${BUILD_CPUS:-4}"
RESUME_FLAG=""
[ "${BUILD_DATASET_RESUME:-0}" = "1" ] && RESUME_FLAG="--resume"
TASK_START_FLAG=""
[ -n "${TASK_START_INDEX:-}" ] && TASK_START_FLAG="--task-start-index ${TASK_START_INDEX}"
TASK_END_FLAG=""
[ -n "${TASK_END_INDEX:-}" ] && TASK_END_FLAG="--task-end-index ${TASK_END_INDEX}"

sbatch $CPU_NODE_ARGS --mem="$BUILD_MEM_GB" -c"$BUILD_CPUS" --time=18:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="build_full_ds" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && $COPPELIASIM_ENV python3 scripts/build_multitask_dataset.py --config \"$CONFIG_PATH\" --dataset-root \"$OUTPUT_DATASET_ROOT\" --overwrite $RESUME_FLAG $TASK_START_FLAG $TASK_END_FLAG'"
