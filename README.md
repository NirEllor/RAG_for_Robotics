# Action-Aware 3D Retrieval for Robotic Manipulation

**Research objective:** Do pretrained 3D representations retrieve demonstrations that are geometrically
similar and action-compatible for robotic manipulation, and are they more robust than image representations
under viewpoint, appearance, noise, and partial-observation changes?

**Status:** dataset ingestion, real-backend retrieval evaluation, robustness testing, and offline downstream transfer are complete; final report packaging is in progress.

## Quick Start

See [CLAUDE.md](CLAUDE.md) for detailed setup instructions and project context. The specification is in
[3d_retrieval_experiment_infrastructure_spec.md](3d_retrieval_experiment_infrastructure_spec.md).

### For Users: Environment Setup

1. For CPU baselines, install dependencies with `pip install -e .`.
2. For Uni3D/PTv3 cluster evaluation, use `slurm/run_prepare_ptv3_pointcept_env.sh` and the clean Pointcept environment described in `slurm/README.md`.
3. Use the full-dataset and robustness SLURM wrappers to reproduce the locked experiments.
4. See [ENVIRONMENT_REPRODUCTION.md](ENVIRONMENT_REPRODUCTION.md) for the validated CUDA/PyTorch stack and asset paths.

### For Developers / Future Claude Sessions

- Read [CLAUDE.md](CLAUDE.md) for phase status and project constraints.
- The spec ([3d_retrieval_experiment_infrastructure_spec.md](3d_retrieval_experiment_infrastructure_spec.md))
  is authoritative; this README is a summary.
- Each phase is gated by the previous phase's acceptance tests. Do not skip phases.

### Helpful Project Docs

- [PROJECT_DICTIONARY.md](PROJECT_DICTIONARY.md) - plain-language terms and file guide.
- [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) - how retrieval is evaluated.
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - near-term and mid-term milestones.
- [FINAL_SUBMISSION_CHECKLIST.md](FINAL_SUBMISSION_CHECKLIST.md) - assignment requirements and evidence.
- [ENVIRONMENT_REPRODUCTION.md](ENVIRONMENT_REPRODUCTION.md) - Cluster environment capture and checkpoints.

## Repository Structure

## Code Provenance and Attribution

The project-specific code is the code in `src/`, `scripts/`, `configs/`,
`slurm/`, and `tests/`. It implements dataset export and validation, retrieval
interfaces, baseline encoders, ranking metrics, robustness perturbations,
projection-head experiments, reproducibility capture, and the RLBench pilot
harness.

The directories and packages used as external dependencies include the Uni3D
repository, the Pointcept/PTv3 repository, RLBench, PyRep, CoppeliaSim, PyTorch,
PyTorch Geometric, spconv, cumm, and related Python packages. Their model
architectures, native kernels, and pretrained checkpoints are upstream assets;
we do not present them as code authored in this project. The adapters, runtime
configuration, checkpoint loading/remapping, experiment scripts, and evaluation
logic surrounding them are project code. Upstream licenses and citations must
remain included when redistributing or submitting the code.

```
project/
├── README.md                          (this file)
├── CLAUDE.md                          (project context & setup instructions)
├── LICENSE                            (MIT)
├── pyproject.toml                     (package metadata)
├── requirements.txt                   (dependencies)
├── 3d_retrieval_experiment_infrastructure_spec.md
├── configs/                           (Hydra configuration files)
├── src/action_retrieval/              (main package)
├── scripts/                           (entry-point scripts)
├── tests/                             (unit & integration tests)
├── data/                              (datasets, ignored by git)
├── outputs/                           (experiment results, ignored by git)
└── third_party/                       (external dependencies)
```

## Dependencies

See `requirements.txt` for the full list. Main dependencies:
- PyTorch (CPU baselines; CUDA build for real 3D backends)
- Hydra & OmegaConf (configuration)
- pandas, numpy, scipy, scikit-learn (data & math)
- matplotlib (visualization)
- RLBench, PyRep, CoppeliaSim (Cluster/WSL2 simulator dependencies)

Install with:
```bash
pip install -e .
```

## Reproducing the Report Results on the Cluster

The commands below describe the validated Cluster workflow. They assume the
repository is checked out at:

```text
/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics
```

The exported datasets and large checkpoints are stored outside Git. The
expected roots are:

```text
DATA_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_data
OUTPUT_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs
UNI3D_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Uni3D
POINTCEPT_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept
VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env
```

### 1. Prepare and Validate the Environment

The real PTv3 evaluation uses the clean Pointcept-aligned environment. The
installation script supports the Cluster's conda setup when available and
otherwise creates a venv:

```bash
cd /cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics
bash slurm/run_prepare_ptv3_pointcept_env.sh
source /cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env/bin/activate
```

Important packages in the validated stack include PyTorch 2.5.0 with CUDA
12.4, torchvision 0.20.0, torch-geometric, spconv-cu124, torch-scatter,
torch-sparse, torch-cluster, timm, open3d, pyarrow, fastparquet, easydict,
and the Pointcept/Uni3D support packages. RLBench execution additionally uses
gymnasium, cffi, natsort, PyRep, and CoppeliaSim. The complete project-level
and Cluster lists are in `requirements.txt` and `requirements-cluster.txt`.

Use a compatible GPU node such as `silico-013` for real 3D evaluation. GTX1080
nodes have compute capability 6.1 and are incompatible with the validated
PyTorch CUDA build.

### 2. Verify External Checkpoints

```bash
test -f "$UNI3D_ROOT/checkpoints/uni3d-g/modelzoo/uni3d-g/model.pt"
test -f "$POINTCEPT_ROOT/scannet-semseg-pt-v3m1-0-base/model/model_best.pth"

cd "$POINTCEPT_ROOT"
git describe --tags --always
```

The PTv3 checkout must be Pointcept `v1.5.2` (commit prefix `ad653ee`). Set the
runtime variables explicitly before real-backend jobs:

```bash
export UNI3D_REPO_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Uni3D
export UNI3D_CHECKPOINT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Uni3D/checkpoints/uni3d-g/modelzoo/uni3d-g/model.pt
export UNI3D_USE_REAL=1
export UNI3D_DEVICE=cuda
export PTV3_REPO_ROOT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept
export PTV3_CHECKPOINT=/cs/labs/raananf/ellorw.nir/3d_cv_dl/Pointcept/scannet-semseg-pt-v3m1-0-base/model/model_best.pth
export PTV3_USE_REAL=1
export PTV3_DEVICE=cuda
```

Run the short real-backend smoke before a long evaluation:

```bash
BACKEND=ptv3 FORWARD_SMOKE=1 \
  SBATCH_NODELIST_OVERRIDE=silico-013 \
  bash slurm/run_diagnose_real_backends.sh
```

### 3. Full Retrieval Evaluation

The full evaluation uses 19 tasks, 1,804 episodes, all seven retrieval methods,
and `k=1,2,3`:

```bash
cd /cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics
source /cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env/bin/activate

EVAL_MEM_GB=64G \
EVAL_TIME_LIMIT=24:00:00 \
SBATCH_NODELIST_OVERRIDE=silico-013 \
bash slurm/run_evaluate_full_multitask.sh
```

Outputs are written to:

```text
$OUTPUT_ROOT/evaluation/retrieval_full/v2_multitask_full
```

The directory contains `summary_metrics.csv`, `per_query_metrics.csv`,
`evaluation.json`, and `summary_report.md`.

### 4. Robustness Evaluation

This evaluates query-only viewpoint, occlusion, and geometry-noise perturbations
while keeping the candidate database clean:

```bash
ROBUSTNESS_MEM_GB=64G \
SBATCH_NODELIST_OVERRIDE=silico-013 \
bash slurm/run_evaluate_robustness.sh
```

Outputs are written to `$OUTPUT_ROOT/evaluation/robustness/v2_multitask_subset8`.

### 5. Action-Aware Projection Head

The MLP trains on frozen Uni3D embeddings and trajectory-state signatures. It
does not fine-tune Uni3D and does not generate actions:

```bash
VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env \
DATASET_ROOT_OVERRIDE=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_data/processed/v2_multitask_subset8 \
sbatch slurm/run_action_head.sh
```

For the fair held-out evaluation, use test queries and train candidates:

```bash
ACTION_HEAD_QUERY_SPLIT=test \
ACTION_HEAD_CANDIDATE_SPLIT=train \
BASELINE_SUMMARY_OVERRIDE=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs/evaluation/retrieval_heldout/uni3d_subset8/summary_metrics.csv \
ACTION_HEAD_EVAL_OUTPUT_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs/evaluation/action_head/uni3d_subset8_heldout_final \
bash slurm/run_evaluate_action_head.sh
```

### 6. Offline Downstream Proxy

```bash
VENV_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/ptv3-pointcept-env \
DATASET_ROOT_OVERRIDE=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_data/processed/v2_multitask_full \
RETRIEVAL_DIR_OVERRIDE=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs/evaluation/retrieval_full/v2_multitask_full \
DOWNSTREAM_OUTPUT_DIR=/cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs/evaluation/downstream/v2_multitask_full \
bash slurm/run_evaluate_downstream_proxy.sh
```

This is an offline trajectory-transfer proxy, not a learned-planner success
benchmark.

### 7. Report Figures

Figures are generated from existing CSV files without recomputing embeddings:

```bash
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID:-$$}
mkdir -p "$MPLCONFIGDIR"
python3 scripts/plot_report_figures.py \
  --evaluation-root /cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs/evaluation \
  --output-dir /cs/labs/raananf/ellorw.nir/3d_cv_dl/RAG_for_Robotics_outputs/evaluation/report_figures
```

The output contains PNG and SVG versions of the full-dataset, subset-8,
robustness, and held-out projection-head figures.

### 8. Reproducibility Snapshot

Capture the actual node environment from inside a SLURM allocation rather than
from the login shell:

```bash
sbatch slurm/run_capture_reproducibility.sh
```

This records Python packages, node/runtime information, repository revisions,
and checkpoint hashes under `$OUTPUT_ROOT/evaluation/reproducibility`.

## Current Phase

**Current work: final report and reproducibility packaging**

- Phase 0 smoke test passes via the saved-demo fallback path.
- The repo now supports both saved RLBench demos and raw RLBench mirror ingestion.
- `scripts/build_multitask_dataset.py` can build a multi-task dataset from the configured sources.
- Retrieval baselines and evaluation scripts already exist.
- `scripts/train_action_aware_projection.py` trains a small trajectory-aware head while freezing the backbone.
- `scripts/run_rlbench_planning_pilot.py` validates or executes a deliberately small RLBench trajectory-replay pilot.

The current practical focus is to turn the completed experiments into a
reproducible final report with explicit limitations and result provenance.

## License

MIT. See [LICENSE](LICENSE) for details.
