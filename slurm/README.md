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

## GPU-ready future encoders

The current codebase already has a retrieval interface that can be extended with
`uni3d` and `ptv3` selectors. The `slurm/run_future_3d_encoder.sh` script is a
placeholder launcher for when those backends are implemented.

