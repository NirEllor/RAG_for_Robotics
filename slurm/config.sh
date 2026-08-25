#!/bin/bash

# Shared SLURM configuration for the RAG_for_Robotics cluster jobs.
# Source this file from other run_*.sh scripts.

set -euo pipefail

PROJECT_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics"
VENV_DIR="$PROJECT_ROOT/.venv"

# Keep caches private to this user, not global to the cluster.
WORK_ROOT="/cs/labs/raananf/ellorw.nir/3d_cv_dl"
CACHE_ROOT="$WORK_ROOT/.cache"
PIP_CACHE_DIR="$CACHE_ROOT/pip"
HF_HOME="$CACHE_ROOT/huggingface"
TORCH_HOME="$CACHE_ROOT/torch"

EMAIL="ellorwaizner.nir@mail.huji.ac.il"

# Default cluster settings; individual scripts can override them.
CPU_NODE_ARGS=""
GPU_NODE_ARGS="--gres=gpu:1"

# Default data and outputs locations on the cluster.
DATA_ROOT="$WORK_ROOT/RAG_for_Robotics_data"
OUTPUT_ROOT="$WORK_ROOT/RAG_for_Robotics_outputs"

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$DATA_ROOT" "$OUTPUT_ROOT"

PYTHON_BIN="$VENV_DIR/bin/python3"

export PROJECT_ROOT VENV_DIR CACHE_ROOT PIP_CACHE_DIR HF_HOME TORCH_HOME
export DATA_ROOT OUTPUT_ROOT EMAIL CPU_NODE_ARGS GPU_NODE_ARGS PYTHON_BIN
