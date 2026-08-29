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
