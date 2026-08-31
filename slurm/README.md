# SLURM Helpers

These scripts submit jobs from the cluster login node without touching the
global Python installation.

## What lives where

- Code: clone the repository normally with Git.
- Python environment: create a private `.venv` inside the clone.
- Caches: keep them under `/cs/labs/raananf/ellorw.nir/3d_cv_dl/.cache`.
- Data: keep shared or regenerated data outside the repo, for example under
  `/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_data`.
- Outputs: keep results under
  `/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs`.

## Recommended workflow

1. Clone the repository on the cluster.
2. Create and activate the private virtual environment.
3. Install package dependencies inside the venv only.
4. Point the dataset and output roots to cluster storage.
5. Submit jobs through the `run_*.sh` helpers.

After every `git pull`, rerun:

```bash
python3 -m pip install -e .
```

That keeps the venv aligned with the current `pyproject.toml` dependencies,
including `huggingface_hub`.

## Data connection strategy

There are three practical options:

1. Re-download or re-export the data on the cluster.
2. Copy only the processed dataset from the local machine to cluster storage.
3. Keep the raw mirror local on the laptop and only move the processed
   `data/processed/...` outputs that you actually need.

For most experiments, the best compromise is:

- use the laptop for raw ingestion/debugging,
- use the cluster for large experiments,
- transfer only the processed dataset and important outputs.

## If the cluster starts from zero

Use the following order:

1. `./slurm/run_fetch_rlbench_mirror.sh`
2. `./slurm/run_build_multitask_dataset.sh`
3. `./slurm/run_evaluate.sh`

Or, if you want a single convenience command:

1. `./slurm/run_prepare_cluster_data.sh`

That will download the raw RLBench mirror on the cluster, then build the
processed dataset after the download job completes successfully.

The fetch job writes directly into:

- `RAG_for_Robotics/data/rlbench/raw`

That path is ignored by Git, so it is safe to use on the cluster without
polluting the repository history.

## GPU-ready future encoders

The current codebase already has a retrieval interface that can be extended with
`uni3d` and `ptv3` selectors. The `slurm/run_future_3d_encoder.sh` script is a
cluster launcher for those backends; it can be used now with the CPU-friendly
proxy encoders and later swapped to real checkpoints without changing the job
shape. The main evaluation helper also includes `uni3d` and `ptv3` in its
default method list.

For a presentation-friendly full run on the subset-8 dataset, use:

1. `./slurm/run_evaluate_showcase.sh`

That wrapper evaluates `random`, `pose_descriptor`, `rgb_histogram`,
`global_color`, `geometry_only`, `uni3d`, and `ptv3` on the subset-8 dataset
with the current cluster-safe environment setup.

For a sweep across every exported dataset root under `data/processed`, use:

1. `./slurm/run_evaluate_all.sh`

That wrapper iterates over each dataset root that contains a `manifest.parquet`
file and runs the full method set on each one, using inferred task grouping
when a dataset-specific annotations JSON is not available.

For a smaller subset that skips `reach_target` and focuses on simpler tasks,
use:

1. `./slurm/run_build_simple_multitask_dataset.sh`
2. `./slurm/run_evaluate_simple_subset.sh`

Those wrappers keep the simpler RLBench tasks from the multitask config and
write to `data/processed/v2_multitask_simple` by default.

For PTv3 specifically, we are pinning the real backend to the Pointcept
`v1.5.2` release family. The encoder checks the cloned repository's Git tag or
commit and warns/fails early if the code checkout does not match the expected
release. That avoids the common failure mode of mixing a checkpoint from one
PTv3 release with model code from another.

Set `PTV3_REPO_LAYOUT=pointcept` when `PTV3_REPO_ROOT` points at a Pointcept
checkout, or leave it as `auto` to let the loader infer the layout from the
repo contents. The standalone `PointTransformerV3` repository still works via
`PTV3_REPO_LAYOUT=standalone`.

## Download jobs for real checkpoints

If you want the real pretrained backbones, use the download helpers first:

1. `./slurm/run_download_uni3d.sh`
2. `./slurm/run_download_ptv3.sh`

Both jobs expect `HF_TOKEN` to be exported before submission and will store
weights under cluster-local checkpoint directories.

## Building pointnet2_ops for Uni3D

Uni3D's real backend depends on `pointnet2_ops`. If the pip build fails with a
CUDA mismatch or the compiler gets killed, use:

1. `./slurm/run_build_pointnet2_ops.sh`

The job loads `cuda/12.9`, caps parallel compilation with `MAX_JOBS=1`, and
sets `TORCH_CUDA_ARCH_LIST` to a conservative default that works well on the
cluster. Once this job succeeds, Uni3D should stop falling back to the proxy
backend.

## Building PTv3 PyG dependencies

If PTv3 import fails because `torch_scatter`, `torch_sparse`, or
`torch_cluster` are missing, use:

1. `./slurm/run_install_ptv3_deps.sh`

The job uses a GPU node, forces serial compilation, and lowers compiler
optimization to reduce peak memory during source builds.

## Clean PTv3 environment

If the current Python stack keeps crashing inside `spconv/cumm` during PTv3
forward smoke tests, build a clean environment that follows the official
Pointcept CUDA 12.4 / PyTorch 2.5 stack:

1. `./slurm/run_prepare_ptv3_pointcept_env.sh`

That helper creates a private conda environment, installs the Pointcept
dependency set when conda is available. If conda is not installed on the
cluster shell, it falls back to `python/native` plus a private virtualenv and
installs the same wheel-based stack there. In both cases it finishes with a
quick import smoke test. After that, run the PTv3 diagnosis helper again from
the new environment.

The environment also includes `pyarrow`/`fastparquet` so `pandas.read_parquet`
can load dataset manifests without additional manual installs.

To point the existing SLURM helpers at the clean PTv3 env, export:

```bash
VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env
```

before launching `run_diagnose_real_backends.sh` or any other helper that
activates the project Python environment.
