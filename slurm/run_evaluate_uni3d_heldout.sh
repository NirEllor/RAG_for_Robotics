#!/bin/bash
# Fair Uni3D baseline: test queries retrieved against train candidates.
#SBATCH --job-name=rag_uni3d_test
#SBATCH --partition=short
#SBATCH --nodelist=silico-013
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
WORK_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/ptv3-pointcept-env}"
DATASET_ROOT="${DATASET_ROOT_OVERRIDE:-$WORK_ROOT/RAG_for_Robotics_data/processed/v2_multitask_subset8}"
ANNOTATIONS="${ANNOTATIONS_OVERRIDE:-$PROJECT_ROOT/configs/evaluation/rlbench_multitask_subset8_task_labels.json}"
OUTPUT_DIR="${UNI3D_HELDOUT_OUTPUT_DIR:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/retrieval_heldout/uni3d_subset8}"

source "$VENV_DIR/bin/activate"
export UNI3D_REPO_ROOT="${UNI3D_REPO_ROOT:-$WORK_ROOT/Uni3D}"
export UNI3D_CHECKPOINT="${UNI3D_CHECKPOINT:-$WORK_ROOT/Uni3D/checkpoints/uni3d-g/modelzoo/uni3d-g/model.pt}"
export UNI3D_USE_REAL=1
export UNI3D_DEVICE=cuda
cd "$PROJECT_ROOT"

python3 scripts/evaluate_retrieval.py \
  --dataset-root "$DATASET_ROOT" \
  --annotations "$ANNOTATIONS" \
  --methods uni3d \
  --ks 1 2 3 \
  --query-split test \
  --candidate-split train \
  --output-dir "$OUTPUT_DIR"
