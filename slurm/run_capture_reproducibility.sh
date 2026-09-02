#!/bin/bash
# Capture reproducibility metadata inside a real GPU allocation.
# This script intentionally does not depend on slurm/config.sh.
#SBATCH --job-name=rag_repro
#SBATCH --partition=short
#SBATCH --nodelist=silico-013
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
WORK_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl"
VENV_DIR="${VENV_DIR:-$WORK_ROOT/ptv3-pointcept-env}"
OUTPUT_DIR="${REPRO_OUTPUT_DIR:-$WORK_ROOT/RAG_for_Robotics_outputs/evaluation/reproducibility}"
mkdir -p "$OUTPUT_DIR"
source "$VENV_DIR/bin/activate"

{
  echo "date: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "python: $(python --version 2>&1)"
  echo "repo_commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  echo "pointcept_release: $(git -C "$WORK_ROOT/Pointcept" describe --tags --always 2>/dev/null || true)"
  echo "modules:"
  if type module >/dev/null 2>&1; then module list 2>&1; else echo "module command unavailable"; fi
  echo "nvidia_smi:"
  if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi unavailable"; fi
  echo "torch_gpu:"
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
  echo "pip_freeze:"
  python -m pip freeze
} > "$OUTPUT_DIR/environment.txt"

sha256sum \
  "$WORK_ROOT/Uni3D/checkpoints/uni3d-g/modelzoo/uni3d-g/model.pt" \
  "$WORK_ROOT/Pointcept/scannet-semseg-pt-v3m1-0-base/model/model_best.pth" \
  > "$OUTPUT_DIR/checkpoint_sha256.txt"
echo "Reproducibility outputs written to: $OUTPUT_DIR"
