#!/bin/bash

# Minimal PTv3 real-backend smoke test.
# This job loads the official PTv3 checkpoint, normalizes the checkpoint keys,
# constructs the real backend, and runs one tiny forward pass on synthetic data.
#
# Usage:
#   chmod +x slurm/run_ptv3_smoke.sh
#   sbatch slurm/run_ptv3_smoke.sh
#
# Optional overrides:
#   PTV3_SMOKE_NODE_ARGS="--partition=short --gres=gpu:1 --nodelist=silico-013"
#   PTV3_REPO_ROOT=/path/to/PointTransformerV3
#   PTV3_CHECKPOINT=/path/to/model_best.pth

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

PTV3_SMOKE_NODE_ARGS="${PTV3_SMOKE_NODE_ARGS:---partition=short --gres=gpu:1 --nodelist=silico-013}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/nvidia/cuda/12.9}"
PTV3_REPO_ROOT="${PTV3_REPO_ROOT:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/PointTransformerV3}"
PTV3_CHECKPOINT="${PTV3_CHECKPOINT:-/cs/labs/raananf/ellorw.nir/3d_cv_dl/PointTransformerV3/checkpoints/scannet-semseg-pt-v3m1-0-base/scannet-semseg-pt-v3m1-0-base/model/model_best.pth}"

sbatch $PTV3_SMOKE_NODE_ARGS --mem=32G -c2 --time=00:30:00 \
  --mail-type=END,FAIL --mail-user="$EMAIL" \
  --job-name="ptv3_smoke" \
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
    rm -rf \"$TORCH_EXTENSIONS_DIR\"/pointnet2_ops* \"$TORCH_EXTENSIONS_DIR\"/_ptv3_runtime* &&
    export PTV3_REPO_ROOT=\"$PTV3_REPO_ROOT\" &&
    export PTV3_CHECKPOINT=\"$PTV3_CHECKPOINT\" &&
    export PTV3_USE_REAL=1 &&
    export PTV3_DEVICE=cuda &&
    python3 - <<\"PY\"
from pathlib import Path
import os
import torch

ckpt_path = Path(os.environ[\"PTV3_CHECKPOINT\"])
print(\"[1] checkpoint exists:\", ckpt_path.exists())
checkpoint = torch.load(ckpt_path, map_location=\"cpu\", weights_only=False)
print(\"[2] checkpoint type:\", type(checkpoint))
if isinstance(checkpoint, dict):
    keys = list(checkpoint.keys())
    print(\"[3] top-level keys:\", keys[:20])
    state_dict = checkpoint.get(\"state_dict\", checkpoint)
    print(\"[4] num tensors:\", len(state_dict) if isinstance(state_dict, dict) else -1)
    if isinstance(state_dict, dict):
        prefixed = sum(1 for key in state_dict.keys() if isinstance(key, str) and key.startswith(\"module.\"))
        print(\"[5] keys with module. prefix:\", prefixed)
        print(\"[6] first tensor keys:\", list(state_dict.keys())[:20])

from action_retrieval.retrieval.encoders import PointTransformerV3Encoder

enc = PointTransformerV3Encoder()
print(\"[7] encoder device:\", enc.device)
print(\"[8] encoder checkpoint:\", enc.checkpoint)
model = enc._get_real_backend()
print(\"[9] backend loaded:\", model is not None)
if model is None:
    raise SystemExit(2)

sampled = torch.randn(128, 6, dtype=torch.float32, device=enc.device)
coord = sampled[:, :3].contiguous()
feat = sampled.contiguous()
batch = torch.zeros((sampled.shape[0],), dtype=torch.long, device=enc.device)
data_dict = {\"coord\": coord, \"feat\": feat, \"batch\": batch, \"grid_size\": enc.grid_size}
print(\"[10] running forward with synthetic input...\")
with torch.inference_mode():
    output = model(data_dict)
print(\"[11] forward output type:\", type(output))
if isinstance(output, dict):
    print(\"[12] output keys:\", list(output.keys())[:20])
else:
    print(\"[12] has feat attr:\", hasattr(output, \"feat\"))
PY
  '"
