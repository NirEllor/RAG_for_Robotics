#!/bin/bash

# PTv3 forward-crash debug launcher.
# This submits three short GPU jobs:
#   1) PTv3 smoke on node A
#   2) PTv3 smoke on node B
#   3) standalone spconv smoke on node C
#
# The goal is to quickly distinguish:
#   - node-specific CUDA/kernel issues
#   - generic spconv runtime issues
#   - PTv3-specific forward crashes
#
# Usage:
#   chmod +x slurm/run_ptv3_forward_crash_debug.sh
#   bash slurm/run_ptv3_forward_crash_debug.sh
#
# Optional overrides:
#   PTV3_DEBUG_NODE_A=silico-013
#   PTV3_DEBUG_NODE_B=silico-009
#   PTV3_DEBUG_NODE_C=silico-011
#   CUDA_MODULE=cuda/12.9
#   CUDA_ROOT=/usr/local/nvidia/cuda/12.9

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
source "$PROJECT_ROOT/slurm/config.sh"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/nvidia/cuda/12.9}"
PTV3_DEBUG_NODE_A="${PTV3_DEBUG_NODE_A:-silico-013}"
PTV3_DEBUG_NODE_B="${PTV3_DEBUG_NODE_B:-silico-009}"
PTV3_DEBUG_NODE_C="${PTV3_DEBUG_NODE_C:-silico-011}"
PTV3_STRICT_RELEASE="${PTV3_STRICT_RELEASE:-0}"
SPCONV_MEM="${SPCONV_MEM:-12G}"
SPCONV_TIME="${SPCONV_TIME:-00:20:00}"

echo "[1/3] Submitting PTv3 smoke on node A: $PTV3_DEBUG_NODE_A"
BACKEND=ptv3 \
FORWARD_SMOKE=1 \
SBATCH_NODELIST_OVERRIDE="$PTV3_DEBUG_NODE_A" \
PTV3_STRICT_RELEASE="$PTV3_STRICT_RELEASE" \
bash "$PROJECT_ROOT/slurm/run_diagnose_real_backends.sh"

echo "[2/3] Submitting PTv3 smoke on node B: $PTV3_DEBUG_NODE_B"
BACKEND=ptv3 \
FORWARD_SMOKE=1 \
SBATCH_NODELIST_OVERRIDE="$PTV3_DEBUG_NODE_B" \
PTV3_STRICT_RELEASE="$PTV3_STRICT_RELEASE" \
bash "$PROJECT_ROOT/slurm/run_diagnose_real_backends.sh"

echo "[3/3] Submitting standalone spconv smoke on node C: $PTV3_DEBUG_NODE_C"
SPCONV_JOB_ID=$(
  sbatch --parsable \
    --partition=short --gres=gpu:1 --nodelist="$PTV3_DEBUG_NODE_C" \
    --mem="$SPCONV_MEM" -c2 --time="$SPCONV_TIME" \
    --mail-type=END,FAIL --mail-user="$EMAIL" \
    --job-name="spconv_smoke" \
    --wrap "bash -lc '
      cd \"$PROJECT_ROOT\" &&
      source \"$VENV_DIR/bin/activate\" &&
      if ! command -v module >/dev/null 2>&1; then
        if [ -f /etc/profile.d/modules.sh ]; then
          source /etc/profile.d/modules.sh
        elif [ -f /usr/share/lmod/lmod/init/bash ]; then
          source /usr/share/lmod/lmod/init/bash
        fi
      fi &&
      module purge &&
      module load \"$CUDA_MODULE\" &&
      export CUDA_HOME=\"$CUDA_ROOT\" &&
      export PATH=\"\$CUDA_HOME/bin:\$PATH\" &&
      export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\" &&
      export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST:-7.5;8.6}\" &&
      export MAX_JOBS=1 &&
      export CFLAGS=\"-O0 -g0\" &&
      export CXXFLAGS=\"-O0 -g0\" &&
      export TMPDIR=\"$TMP_ROOT\" &&
      export TMP=\"$TMP_ROOT\" &&
      export TEMP=\"$TMP_ROOT\" &&
      export TORCH_EXTENSIONS_DIR=\"$TORCH_EXTENSIONS_DIR\" &&
      python3 scripts/spconv_smoke.py
    '"
)

echo "Submitted spconv smoke job: $SPCONV_JOB_ID"
echo "Watch logs with: tail -f slurm-<jobid>.out"
