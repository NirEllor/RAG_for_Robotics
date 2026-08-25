# Action-Aware 3D Retrieval for Robotic Manipulation

**Research objective:** Do pretrained 3D representations retrieve demonstrations that are geometrically
similar and action-compatible for robotic manipulation, and are they more robust than image representations
under viewpoint, appearance, noise, and partial-observation changes?

**Status:** dataset ingestion, retrieval baselines, and evaluation are in progress.

## Quick Start

See [CLAUDE.md](CLAUDE.md) for detailed setup instructions and project context. The specification is in
[3d_retrieval_experiment_infrastructure_spec.md](3d_retrieval_experiment_infrastructure_spec.md).

### For Users: Environment Setup

1. Inside WSL2 Ubuntu 24.04, install dependencies and CoppeliaSim 4.1 (see CLAUDE.md for details).
2. Run `scripts/smoke_env.py` to verify the environment and generate your first test episode.
3. Report the result back before proceeding to Phase 1 (dataset generation).

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

**Current work: Dataset expansion and retrieval evaluation**

- Phase 0 smoke test passes via the saved-demo fallback path.
- The repo now supports both saved RLBench demos and raw RLBench mirror ingestion.
- `scripts/build_multitask_dataset.py` can build a multi-task dataset from the configured sources.
- Retrieval baselines and evaluation scripts already exist.

The current practical focus is to finish dataset scaling, then keep improving
retrieval and evaluation.

## License

MIT. See [LICENSE](LICENSE) for details.
