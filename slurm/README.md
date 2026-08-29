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
