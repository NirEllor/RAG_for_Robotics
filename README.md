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

### For Developers / Future Claude Sessions

- Read [CLAUDE.md](CLAUDE.md) for phase status and constraints (CPU-only, no GPU).
- The spec ([3d_retrieval_experiment_infrastructure_spec.md](3d_retrieval_experiment_infrastructure_spec.md))
  is authoritative; this README is a summary.
- Each phase is gated by the previous phase's acceptance tests. Do not skip phases.

### Helpful Project Docs

- [PROJECT_DICTIONARY.md](PROJECT_DICTIONARY.md) - plain-language terms and file guide.
- [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) - how retrieval is evaluated.
- [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) - near-term and mid-term milestones.

## Repository Structure

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
- PyTorch (CPU)
- Hydra & OmegaConf (configuration)
- pandas, numpy, scipy, scikit-learn (data & math)
- matplotlib (visualization)
- RLBench, PyRep, CoppeliaSim (simulator; WSL2 only)

Install with:
```bash
pip install -e .
```

## Current Phase

**Current work: final report and reproducibility packaging**

- Phase 0 smoke test passes via the saved-demo fallback path.
- The repo now supports both saved RLBench demos and raw RLBench mirror ingestion.
- `scripts/build_multitask_dataset.py` can build a multi-task dataset from the configured sources.
- Retrieval baselines and evaluation scripts already exist.

The current practical focus is to turn the completed experiments into a
reproducible final report with explicit limitations and result provenance.

## License

MIT. See [LICENSE](LICENSE) for details.
