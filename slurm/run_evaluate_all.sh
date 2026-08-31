#!/bin/bash

# Full retrieval evaluation sweep over all exported datasets under data/processed.
#
# Runs all currently supported methods on each dataset root that contains a
# manifest.parquet file. For datasets without a dedicated annotations JSON, the
# evaluation code falls back to task_name grouping inferred from the manifest.
#
# Usage:
#   bash slurm/run_evaluate_all.sh
#
# Optional overrides:
#   VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env
#   SBATCH_NODELIST_OVERRIDE=silico-009
#   EVAL_MEM_GB=64G
#   EVAL_TIME=12:00:00
#   EVAL_METHODS_OVERRIDE="random pose_descriptor rgb_histogram global_color geometry_only uni3d ptv3"
#   EVAL_KS_OVERRIDE="1 2 3"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

METHODS="${EVAL_METHODS_OVERRIDE:-random pose_descriptor rgb_histogram global_color geometry_only uni3d ptv3}"
KS="${EVAL_KS_OVERRIDE:-1 2 3}"
EVAL_MEM_GB="${EVAL_MEM_GB:-64G}"
EVAL_TIME="${EVAL_TIME:-12:00:00}"
SBATCH_NODELIST_OVERRIDE="${SBATCH_NODELIST_OVERRIDE:-}"

NODE_ARGS=()
if [ -n "$SBATCH_NODELIST_OVERRIDE" ]; then
  NODE_ARGS+=(--nodelist="$SBATCH_NODELIST_OVERRIDE")
fi

DATASET_ROOTS=()
if [ -n "${EVAL_DATASET_ROOTS_OVERRIDE:-}" ]; then
  # shellcheck disable=SC2206
  DATASET_ROOTS=($EVAL_DATASET_ROOTS_OVERRIDE)
else
  while IFS= read -r -d '' dataset_root; do
    DATASET_ROOTS+=("$dataset_root")
  done < <(find "$DATA_ROOT/processed" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [ "${#DATASET_ROOTS[@]}" -eq 0 ]; then
  echo "No dataset roots found under $DATA_ROOT/processed."
  exit 1
fi

sbatch "${NODE_ARGS[@]}" $GPU_NODE_ARGS --mem="$EVAL_MEM_GB" -c2 --time="$EVAL_TIME" \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="retr_eval_all" \
  --wrap "bash -lc '
    cd \"$PROJECT_ROOT\" &&
    source \"$VENV_DIR/bin/activate\" &&
    module load \"${CUDA_MODULE:-cuda/12.9}\" &&
    export CUDA_HOME=\"${CUDA_ROOT:-/usr/local/nvidia/cuda/12.9}\" &&
    export PATH=\"\$CUDA_HOME/bin:\$PATH\" &&
    export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\" &&
    export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST:-7.5;8.6}\" &&
    export PIP_CACHE_DIR=\"$PIP_CACHE_DIR\" HF_HOME=\"$HF_HOME\" TORCH_HOME=\"$TORCH_HOME\" TMPDIR=\"$TMP_ROOT\" TMP=\"$TMP_ROOT\" TEMP=\"$TMP_ROOT\" TORCH_EXTENSIONS_DIR=\"$TORCH_EXTENSIONS_DIR\" &&
    rm -rf \"$TORCH_EXTENSIONS_DIR\"/pointnet2_ops* \"$TORCH_EXTENSIONS_DIR\"/_ptv3_runtime* &&
    METHODS=\"$METHODS\" KS=\"$KS\" DATASETS=\"${DATASET_ROOTS[*]}\" PROJECT_ROOT=\"$PROJECT_ROOT\" python3 - <<\"PY\"
import os
from pathlib import Path
import subprocess
import sys

project_root = Path(os.environ[\"PROJECT_ROOT\"])
dataset_roots = [Path(item) for item in os.environ[\"DATASETS\"].split()]
methods = os.environ[\"METHODS\"].split()
ks = os.environ[\"KS\"].split()
output_root = project_root / \"outputs\" / \"evaluation\" / \"retrieval_all\"
config_dir = project_root / \"configs\" / \"evaluation\"

annotation_overrides = {
    \"v1_reach_target\": config_dir / \"rlbench_reach_target_hand_labels.json\",
}

for dataset_root in dataset_roots:
    dataset_name = dataset_root.name
    annotations = annotation_overrides.get(dataset_name, config_dir / f\"{dataset_name}_task_labels.json\")
    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        \"scripts/evaluate_retrieval.py\",
        \"--dataset-root\",
        str(dataset_root),
        \"--annotations\",
        str(annotations),
        \"--methods\",
        *methods,
        \"--ks\",
        *ks,
        \"--output-dir\",
        str(out_dir),
    ]

    print(\"=\" * 80)
    print(f\"Running dataset: {dataset_name}\")
    print(f\"Dataset root: {dataset_root}\")
    print(f\"Annotations: {annotations if annotations.exists() else 'inferred from manifest task_name groups'}\")
    print(f\"Methods: {methods}\")
    print(f\"K values: {ks}\")
    print(f\"Output dir: {out_dir}\")
    print(\"Command:\")
    print(\" \".join(cmd))
    print(\"=\" * 80)
    subprocess.run(cmd, check=True)
PY
  '"
