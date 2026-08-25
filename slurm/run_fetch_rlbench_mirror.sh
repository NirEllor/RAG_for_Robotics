#!/bin/bash

# Submit the RLBench raw mirror download/extract as a SLURM job.
# This job assumes no data already exists on the cluster and recreates it from
# Hugging Face into the repository-local ignored data path.
#
# Usage:
#   ./slurm/run_fetch_rlbench_mirror.sh
#   ./slurm/run_fetch_rlbench_mirror.sh --keep-archives
#   ./slurm/run_fetch_rlbench_mirror.sh --workers 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

WORKERS="${RLBENCH_WORKERS_OVERRIDE:-4}"
KEEP_ARCHIVES_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      shift
      WORKERS="${1:-4}"
      ;;
    --keep-archives)
      KEEP_ARCHIVES_FLAG="--keep-archives"
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--workers N] [--keep-archives]"
      exit 1
      ;;
  esac
  shift
done

sbatch $CPU_NODE_ARGS --mem=24G -c4 --time=2-00:00:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="fetch_rlbench" \
  --wrap "bash -lc 'cd \"$PROJECT_ROOT\" && source \"$VENV_DIR/bin/activate\" && export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" && python3 scripts/fetch_rlbench_mirror.py --stage-dir \"$RAW_STAGE_DIR\" --extract-root \"$RAW_DATA_ROOT\" --workers ${WORKERS:-4} ${KEEP_ARCHIVES_FLAG}'"
