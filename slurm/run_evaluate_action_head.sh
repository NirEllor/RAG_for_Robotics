#!/bin/bash
# Evaluate saved action-head embeddings without rerunning Uni3D.
#SBATCH --job-name=rag_head_eval
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
WORK_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/ptv3-pointcept-env}"
DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$WORK_ROOT/RAG_for_Robotics_data/processed/v2_multitask_subset8}"
PROJECTED="${PROJECTED_EMBEDDINGS_OVERRIDE:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/action_head/uni3d_subset8/projected_embeddings.npz}"
ANNOTATIONS="${ANNOTATIONS_OVERRIDE:-$PROJECT_ROOT/configs/evaluation/rlbench_multitask_subset8_task_labels.json}"
BASELINE="${BASELINE_SUMMARY_OVERRIDE:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/retrieval_all/v2_multitask_subset8/summary_metrics.csv}"
OUTPUT_DIR="${ACTION_HEAD_EVAL_OUTPUT_DIR:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/action_head/uni3d_subset8_eval}"

source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
python3 scripts/evaluate_projected_embeddings.py \
  --dataset-root "$DATASET_ROOT" \
  --projected-embeddings "$PROJECTED" \
  --annotations "$ANNOTATIONS" \
  --baseline-summary "$BASELINE" \
  --output-dir "$OUTPUT_DIR"
