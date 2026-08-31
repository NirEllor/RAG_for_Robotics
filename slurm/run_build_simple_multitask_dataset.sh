#!/bin/bash

# Build a smaller multitask dataset that prioritizes simpler tasks and skips
# the already-covered reach_target task.
#
# This is a thin wrapper over build_multitask_dataset.py using task indices
# from the existing multitask config. It keeps only tasks 2-7 from the current
# config order:
#   open_drawer
#   slide_block_to_color_target
#   close_jar
#   stack_blocks
#   place_shape_in_shape_sorter
#   light_bulb_in
#   insert_onto_square_peg
#
# Usage:
#   bash slurm/run_build_simple_multitask_dataset.sh
#
# Optional overrides:
#   OUTPUT_DATASET_ROOT_OVERRIDE=/path/to/output/root
#   BUILD_MEM_GB=20G
#   BUILD_CPUS=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

CONFIG_PATH="${CONFIG_OVERRIDE:-$PROJECT_ROOT/configs/dataset/rlbench_multitask.yaml}"
OUTPUT_DATASET_ROOT="${OUTPUT_DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask_simple}"
BUILD_MEM_GB="${BUILD_MEM_GB:-20G}"
BUILD_CPUS="${BUILD_CPUS:-4}"

sbatch $CPU_NODE_ARGS --mem="$BUILD_MEM_GB" -c"$BUILD_CPUS" --time=12:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="build_simple_ds" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && $COPPELIASIM_ENV python3 scripts/build_multitask_dataset.py --config \"$CONFIG_PATH\" --dataset-root \"$OUTPUT_DATASET_ROOT\" --overwrite --task-start-index 2 --task-end-index 7'"
