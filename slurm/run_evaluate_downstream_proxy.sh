#!/bin/bash

# Evaluate offline transfer of the top retrieved trajectory.
# This is a lightweight CPU job and does not execute a simulator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

DATASET_ROOT_OVERRIDE="${DATASET_ROOT_OVERRIDE:-$DATA_ROOT/processed/v2_multitask_full}"
RETRIEVAL_DIR_OVERRIDE="${RETRIEVAL_DIR_OVERRIDE:-$OUTPUT_ROOT/evaluation/retrieval_full/v2_multitask_full}"
DOWNSTREAM_OUTPUT_DIR="${DOWNSTREAM_OUTPUT_DIR:-$OUTPUT_ROOT/evaluation/downstream/v2_multitask_full}"

source "$VENV_DIR/bin/activate"

python3 scripts/evaluate_downstream_proxy.py \
  --dataset-root "$DATASET_ROOT_OVERRIDE" \
  --retrieval-dir "$RETRIEVAL_DIR_OVERRIDE" \
  --output-dir "$DOWNSTREAM_OUTPUT_DIR"
