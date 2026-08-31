#!/bin/bash

# Evaluate retrieval on the full 19-task multitask dataset.
#
# Expected dataset root:
#   data/processed/v2_multitask_full
#
# This is the most comprehensive single-dataset evaluation we have.
#
# Usage:
#   bash slurm/run_evaluate_full_multitask.sh
#
# Optional overrides:
#   VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env
#   SBATCH_NODELIST_OVERRIDE=silico-013
#   EVAL_MEM_GB=64G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_ROOT_OVERRIDE="${DATASET_ROOT_OVERRIDE:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_data/processed/v2_multitask_full}"
ANNOTATIONS_OVERRIDE="${ANNOTATIONS_OVERRIDE:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics/configs/evaluation/rlbench_v2_multitask_full_task_labels.json}"
METHODS_OVERRIDE="${METHODS_OVERRIDE:-random pose_descriptor rgb_histogram global_color geometry_only uni3d ptv3}"
KS_OVERRIDE="${KS_OVERRIDE:-1 2 3}"
EVAL_MEM_GB="${EVAL_MEM_GB:-64G}"
SBATCH_NODELIST_OVERRIDE="${SBATCH_NODELIST_OVERRIDE:-}"

export DATASET_ROOT_OVERRIDE
export ANNOTATIONS_OVERRIDE
export METHODS_OVERRIDE
export KS_OVERRIDE
export EVAL_MEM_GB
export SBATCH_NODELIST_OVERRIDE

bash "$SCRIPT_DIR/run_evaluate.sh"
