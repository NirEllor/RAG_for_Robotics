# Cluster Reproduction Environment

The generic [`requirements.txt`](requirements.txt) describes the project-level
Python dependencies. The validated real-backend experiments used a separate
Cluster environment because CUDA and native extensions are hardware-specific.

## Required external assets

| Component | Location on Cluster |
| --- | --- |
| Uni3D checkout | `/cs/labs/raananf/ellorw.nir/3d_cv_dl/Uni3D` |
| Uni3D checkpoint | `/cs/labs/raananf/ellorw.nir/3d_cv_dl/Uni3D/checkpoints/uni3d-g/modelzoo/uni3d-g/model.pt` |
| Pointcept checkout | `/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept` |
| Pointcept release | `v1.5.2`, commit prefix `ad653ee` |
| PTv3 checkpoint | `/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept/scannet-semseg-pt-v3m1-0-base/model/model_best.pth` |

Use a compatible GPU node such as `silico-013` (RTX3060), not GTX1080 nodes
when the installed PyTorch build excludes compute capability 6.1.

## Capture the actual environment

Run inside a Cluster allocation, because `nvidia-smi` is generally unavailable
on the login shell:

```bash
mkdir -p "$OUTPUT_ROOT/evaluation/reproducibility"
{
  date -Is
  hostname
  python3 --version
  python3 -m pip freeze
  nvidia-smi || true
  module list 2>&1 || true
  git -C "$PROJECT_ROOT" rev-parse HEAD
  git -C /cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept describe --tags --always
} > "$OUTPUT_ROOT/evaluation/reproducibility/environment.txt"
```

The final report should include this file and SHA256 hashes of both
checkpoints. Native packages such as `spconv`, `cumm`, and `pointnet2_ops`
must be installed for the selected CUDA/PyTorch combination and should be
recorded by `pip freeze`.
