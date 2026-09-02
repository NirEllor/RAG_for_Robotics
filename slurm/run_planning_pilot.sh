#!/bin/bash
# Validate or execute the small RLBench replay pilot.
#SBATCH --job-name=rag_plan_pilot
#SBATCH --partition=short
#SBATCH --nodelist=silico-013
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
WORK_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/ptv3-pointcept-env}"
DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$WORK_ROOT/RAG_for_Robotics_data/processed/v2_multitask_full}"
OUTPUT_DIR="${PLANNING_OUTPUT_DIR:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/planning_pilot/reach_target}"

source "$VENV_DIR/bin/activate"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$WORK_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
cd "$PROJECT_ROOT"
PYTHONPATH=src python scripts/run_rlbench_planning_pilot.py \
  --dataset-root "$DATASET_ROOT" \
  --task "${PLANNING_TASK:-reach_target}" \
  --episodes "${PLANNING_EPISODES:-5}" \
  --output-dir "$OUTPUT_DIR" \
  ${PLANNING_EXECUTE:+--execute}
