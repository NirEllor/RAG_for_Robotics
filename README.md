# Action-Aware 3D Retrieval for Robotic Manipulation

**Research objective:** Do pretrained 3D representations retrieve demonstrations that are geometrically
similar and action-compatible for robotic manipulation, and are they more robust than image representations
under viewpoint, appearance, noise, and partial-observation changes?

**Status:** Phase 0 scaffolding (environment setup in progress).

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

**Phase 0: Environment Feasibility**

- Repo skeleton created (config/src/scripts/tests layout).
- Smoke-test script (`scripts/smoke_env.py`) ready.
- Awaiting user confirmation that WSL2/CoppeliaSim setup succeeded.

Once Phase 0 is confirmed working, Phase 1 (dataset generation) can begin.

## License

MIT. See [LICENSE](LICENSE) for details.
