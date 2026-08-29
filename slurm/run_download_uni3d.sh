#!/bin/bash

# Download the official Uni3D checkpoint as a SLURM job.
#
# Submit with:
#   chmod +x slurm/run_download_uni3d.sh
#   sbatch slurm/run_download_uni3d.sh
#
# Required at submit/runtime:
#   export HF_TOKEN=hf_...
#
# Optional overrides:
#   UNI3D_REPO_ID=BAAI/Uni3D
#   UNI3D_REPO_SUBPATH=modelzoo/uni3d-g/model.pt
#   UNI3D_CHECKPOINT_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Uni3D/checkpoints/uni3d-g

#SBATCH --job-name=uni3d_dl
#SBATCH --partition=short
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ellorwaizner.nir@mail.huji.ac.il

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

UNI3D_REPO_ID="${UNI3D_REPO_ID:-BAAI/Uni3D}"
UNI3D_REPO_SUBPATH="${UNI3D_REPO_SUBPATH:-modelzoo/uni3d-g/model.pt}"
UNI3D_CHECKPOINT_DIR="${UNI3D_CHECKPOINT_DIR:-$WORK_ROOT/Uni3D/checkpoints/uni3d-g}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set."
  echo "Export a Hugging Face read token before submitting this job."
  exit 1
fi

mkdir -p "$UNI3D_CHECKPOINT_DIR"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

export HF_HOME="$HF_HOME"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TORCH_HOME="$TORCH_HOME"

hf download "$UNI3D_REPO_ID" "$UNI3D_REPO_SUBPATH" \
  --local-dir "$UNI3D_CHECKPOINT_DIR" \
  --max-workers 1

