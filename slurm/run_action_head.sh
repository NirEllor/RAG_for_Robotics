#!/bin/bash
# Train the frozen-backbone trajectory-aware projection head.
#SBATCH --job-name=rag_action_head
#SBATCH --partition=short
#SBATCH --nodelist=silico-013
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
WORK_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/ptv3-pointcept-env}"
DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$WORK_ROOT/RAG_for_Robotics_data/processed/v2_multitask_subset8}"
OUTPUT_DIR="${ACTION_HEAD_OUTPUT_DIR:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/action_head/uni3d_subset8}"

source "$VENV_DIR/bin/activate"
export UNI3D_REPO_ROOT="${UNI3D_REPO_ROOT:-$WORK_ROOT/Uni3D}"
export UNI3D_CHECKPOINT="${UNI3D_CHECKPOINT:-$WORK_ROOT/Uni3D/checkpoints/uni3d-g/modelzoo/uni3d-g/model.pt}"
export UNI3D_USE_REAL=1
export UNI3D_DEVICE=cuda
cd "$PROJECT_ROOT"
PYTHONPATH=src python scripts/train_action_aware_projection.py \
  --dataset-root "$DATASET_ROOT" \
  --encoder "${ACTION_HEAD_ENCODER:-uni3d}" \
  --output-dir "$OUTPUT_DIR"
