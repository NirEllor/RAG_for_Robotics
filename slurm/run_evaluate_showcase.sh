#!/bin/bash

# Showcase retrieval evaluation over the subset-8 dataset.
# Runs the full set of baseline + learned methods that we have validated so far.
#
# Usage:
#   bash slurm/run_evaluate_showcase.sh
#
# Optional overrides:
#   VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env
#   SBATCH_NODELIST_OVERRIDE=silico-009
#   EVAL_MEM_GB=64G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_ROOT_OVERRIDE="${DATASET_ROOT_OVERRIDE:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_data/processed/v2_multitask_subset8}"
ANNOTATIONS_OVERRIDE="${ANNOTATIONS_OVERRIDE:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics/configs/evaluation/rlbench_multitask_subset8_task_labels.json}"
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
