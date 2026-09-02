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
DERIVED_FLAG=""
if [ "${PLANNING_ALLOW_DERIVED:-0}" = "1" ]; then
  DERIVED_FLAG="--allow-derived-actions"
fi

source "$VENV_DIR/bin/activate"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-$WORK_ROOT/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
if ! command -v Xvfb >/dev/null 2>&1; then
  echo "ERROR: Xvfb is required for the CoppeliaSim headless OpenGL pilot." >&2
  exit 2
fi
DISPLAY_NUM=$((100 + ${SLURM_JOB_ID:-$$} % 900))
Xvfb ":$DISPLAY_NUM" +extension GLX +extension RANDR +extension RENDER +iglx \
  -screen 0 1280x1024x24 -nolisten tcp -ac >/tmp/rag_xvfb_${DISPLAY_NUM}.log 2>&1 &
XVFB_PID=$!
trap 'kill "$XVFB_PID" 2>/dev/null || true' EXIT
export DISPLAY=":$DISPLAY_NUM"
export QT_QPA_PLATFORM="xcb"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT/platforms"
unset QT_PLUGIN_PATH
export QT_XCB_GL_INTEGRATION="xcb_glx"
export QT_OPENGL="desktop"
export QT_XCB_FORCE_SOFTWARE_OPENGL="1"
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3
export LIBGL_ALWAYS_INDIRECT=0
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/RLBench:$PROJECT_ROOT/third_party/PyRep:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
if command -v glxinfo >/dev/null 2>&1; then
  for _ in 1 2 3 4 5; do
    glxinfo -B >/tmp/rag_glxinfo_${DISPLAY_NUM}.log 2>&1 && break
    sleep 1
  done
  if ! grep -q "direct rendering\|OpenGL renderer string" /tmp/rag_glxinfo_${DISPLAY_NUM}.log; then
    echo "ERROR: Xvfb started, but GLX preflight failed:" >&2
    cat /tmp/rag_glxinfo_${DISPLAY_NUM}.log >&2
    exit 3
  fi
  grep -E "direct rendering|OpenGL renderer|OpenGL version" /tmp/rag_glxinfo_${DISPLAY_NUM}.log || true
fi
python3 - <<'PY'
import _cffi_backend
import gymnasium
import natsort
import rlbench
import pyrep
print("cffi/RLBench/PyRep import preflight OK")
PY
python3 scripts/run_rlbench_planning_pilot.py \
  --dataset-root "$DATASET_ROOT" \
  --task "${PLANNING_TASK:-reach_target}" \
  --episodes "${PLANNING_EPISODES:-5}" \
  --output-dir "$OUTPUT_DIR" \
  ${PLANNING_EXECUTE:+--execute} \
  $DERIVED_FLAG
