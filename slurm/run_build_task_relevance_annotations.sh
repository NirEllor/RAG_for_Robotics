#!/bin/bash

# Submit task-level relevance annotation generation as a SLURM job.
# The default target is the exported subset-8 multitask dataset.
#
# Usage:
#   chmod +x slurm/run_build_task_relevance_annotations.sh
#   bash slurm/run_build_task_relevance_annotations.sh
#
# Optional overrides:
#   DATASET_ROOT_OVERRIDE=/path/to/exported/dataset
#   ANNOTATIONS_OUTPUT_OVERRIDE=/path/to/output.json
#   ANNOTATION_TASKS_OVERRIDE="task_a task_b ..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask_subset8}"
ANNOTATIONS_OUTPUT="${ANNOTATIONS_OUTPUT_OVERRIDE:-$PROJECT_ROOT/configs/evaluation/rlbench_multitask_subset8_task_labels.json}"
TASKS="${ANNOTATION_TASKS_OVERRIDE:-}"
CPU_MEM_GB="${ANNOTATION_MEM_GB:-4G}"
CPU_CPUS="${ANNOTATION_CPUS:-2}"

TASK_FLAGS=""
if [ -n "$TASKS" ]; then
  # shellcheck disable=SC2086
  TASK_FLAGS="--task-name $TASKS"
fi

sbatch $CPU_NODE_ARGS --mem="$CPU_MEM_GB" -c"$CPU_CPUS" --time=01:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="build_ann" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/build_task_relevance_annotations.py --dataset-root \"$DATASET_ROOT\" --output \"$ANNOTATIONS_OUTPUT\" $TASK_FLAGS'"
