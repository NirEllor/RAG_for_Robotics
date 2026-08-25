# Action-Aware 3D Retrieval for Robotic Manipulation

## Research Objective

This project investigates whether pretrained 3D representations retrieve robot demonstrations that are
geometrically similar and action-compatible for manipulation planning, and whether such retrieval is more
robust than image representations under viewpoint, appearance, noise, and partial-observation changes.
Secondarily, we study whether lightweight action-aware adaptation of frozen 3D encodings improves
retrieval of transferable demonstrations. All work follows the detailed implementation specification in
`3d_retrieval_experiment_infrastructure_spec.md` (the authoritative source of truth for design decisions,
phased build order, and acceptance criteria).

## Hardware Constraints & Implications

**This machine has no GPU.** Investigation confirmed: no NVIDIA hardware present (integrated Intel UHD
Graphics only); no CUDA toolkit; native Windows Python 3.12 has torch 2.5.1+cpu (cuda.is_available()=False);
WSL2 Ubuntu 24.04.2 also has no GPU passthrough. Everything—both CoppeliaSim rendering and all deep learning
inference—will run CPU-only on this machine.

Consequences:
- We use only small pretrained models (DINOv2-ViT-S, not ViT-B or ViT-L; lightweight 3D encoders like
  Uni3D's smallest checkpoint if feasible, else fall back to an explicit pose descriptor + PCA-on-points).
- The MVP dataset is small (15-20 ReachTarget episodes, not hundreds).
- Heavy operations (CoppeliaSim rendering, foundation-model inference) must not stack their memory
  footprints; we keep them in separate script invocations.
- WSL2's 7.6GB RAM allocation is a real bottleneck; if a 3D foundation encoder cannot fit within that
  budget after loading the dataset and PyRep, we gracefully fall back to the geometric descriptor.
- CPU inference is slow; this is a research-validation pipeline, not a production deployment, so we
  optimize for correctness and reproducibility over speed.

## Resolved Design Decisions (Spec Section 23)

1. **Live generation vs. saved demonstrations:** Attempt live RLBench simulation first (ReachTarget
   episodes are short). If CoppeliaSim cannot render headlessly in WSL2 (a known pain point), fall back
   to importing pre-generated saved demonstrations (e.g., HuggingFace mirror `hqfang/rlbench-18-tasks` or
   user-provided demos, using a `saved_importer.py` backend per spec Phase 0 fallback).
2. **Initial task:** `ReachTarget` only (spec's own recommendation; simplest scene, fewest dependencies).
   `PushButton` as secondary once MVP is stable. `PickAndLift` deferred (grasp/orientation is "target
   project" tier, not MVP).
3. **Observation modality:** single camera (`front_rgb`), 128×128 resolution (small enough to fit comfortably
   on disk and in memory; multi-camera fusion stays configurable but is not computed by default).
4. **3D input representation:** Primary MVP baseline is explicit pose/geometric descriptor (spec 9.1);
   target-crop point clouds (256-512 pts, target-local frame) as the 3D foundation-model input if/when a
   real 3D encoder is available. Full fused-multi-view XYZRGB scenes are stretch ablations.
5. **3D encoder:** Attempt Uni3D's smallest checkpoint first (ViT-point-tokenized, minimal custom CUDA).
   Timebox this to Phase 2 (~half a day); if CPU inference is infeasible or memory budget is exceeded,
   fall back explicitly to PCA-on-FPS-sampled-points as the project's interim "3D encoder" and document
   the limitation. `OpenShape` skipped for MVP (sparse-conv backbones are painful without CUDA).
6. **Image encoder:** DINOv2-ViT-S via `torch.hub.load(...)` (frozen, already available in torch's model
   zoo, no additional dependencies).
7. **Normalization & absolute position:** Pretrained 3D encoders (Uni3D/OpenShape) expect object-centered,
   unit-scale clouds, which destroy absolute world position essential for action retrieval (spec 8.3).
   This is why the MVP includes B5 (embedding + explicit pose features) and the adapter's spatial-feature
   ablation (spec 12.2) — they are not optional extras, they are mandatory to restore task-relevant
   position information once foundation embeddings are used.
8. **Target identity protocol:** Default to `xyzrgb` (symmetric across image and 3D; no privileged
   segmentation). Ablate `target_mask` later if time permits.
9. **Action-compatibility frame:** world/robot-base frame as primary; target-relative as the required
   ablation (spec 11.2). Provisional weights (to be set on train/val only, never test, per spec 11.5):
   goal/displacement α=0.5, trajectory-shape β=0.4, rotation γ=0.05, gripper η=0.05.
10. **Downstream baseline:** Nearest-trajectory transfer (spec 15.2, option 1) before Flow Matching.
    Transforms a retrieved trajectory from its source frame into the query target frame; no training
    needed, cheapest possible baseline.
11. **Demo count:** 15-20 episodes for the MVP ReachTarget set.
12. **Flow Matching:** Stretch goal only; implement only after all retrieval experiments (Phases 1-6)
    and at least one simpler downstream baseline are stable and reproducible.

## Phased Build Order (Spec Section 18)

- **Phase 0 (now):** Environment feasibility, repo skeleton, Hydra configuration, smoke-test script.
  **Status:** scaffolding in progress.
- **Phase 1 (pending):** Dataset generation and validation. Depends on Phase-0 smoke test passing
  (simulator confirmed working or fallback to saved-demo path decided).
- **Phase 2 (pending):** Retrieval MVP (random, pose descriptor, image, 3D baselines; exact cosine
  retrieval; caching).
- **Phase 3 (pending):** Action-compatibility evaluation and metrics.
- **Phase 4-6 (pending):** Robustness, adaptation, downstream evaluation.
- **Phase 7 (pending):** Flow Matching stretch goal.

**Current status: Phase 1 dataset/export scaffold underway.** Phase 0 acceptance has been satisfied
through the saved-demo fallback path in `scripts/smoke_env.py`, so the next step is exporting the
Phase 1 ReachTarget dataset and validating the manifest/split layout.

## Environment Setup (For Users and Future Sessions)

### WSL2 / CoppeliaSim / PyRep / RLBench Installation

**The user is responsible for setting up WSL2/CoppeliaSim/PyRep/RLBench.** Claude Code will NOT run WSL2
commands or attempt to install these. If you (a future Claude session) are asked to work on Phase 1+ and
the user hasn't confirmed Phase 0 yet, ask them to run `scripts/smoke_env.py` first and report the result.

For the user's reference (detailed instructions in the implementation plan document):
1. Inside WSL2 Ubuntu 24.04, install system dependencies: `sudo apt-get update && sudo apt-get install -y
   build-essential xvfb libgl1-mesa-dev libgl1-mesa-glx libxkbcommon-x11-0 libxcb-xinerama0 libglib2.0-0
   qtbase5-dev python3.12-venv python3-pip`.
2. Download and extract CoppeliaSim 4.1 EDU, set `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, and
   `QT_QPA_PLATFORM_PLUGIN_PATH` environment variables.
3. Use `xvfb-run` with the **normal** CoppeliaSim binary (not `libcoppeliaSimHeadless.so`, which lacks
   vision-sensor support). Example: `xvfb-run -a -s "-screen 0 1024x768x24" python
   scripts/smoke_env.py`.
4. Clone PyRep and RLBench as git submodules (exact commits pinned in `third_party/README.md` once
   determined); install as editable packages.
5. Run `scripts/smoke_env.py` and report success/failure to the next Claude session or session planning
   notes.

### Native Windows Venv

A Python venv is recommended for development/editing convenience:
```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
```

This venv need not have RLBench/PyRep installed (those are only needed inside WSL2). The Windows venv
can run static analysis, config validation, linting, and tests that don't require the simulator.

### Intended Data/Output Symlink

After Phase-0 is confirmed working, the user should set up fast I/O by symlinking WSL2-native storage
into the repo:
```bash
# Inside WSL2:
mkdir -p ~/action_retrieval_data/{data,outputs}

# On Windows (in repo root):
# Use `mklink /D data \\wsl$\Ubuntu\home\<user>\action_retrieval_data\data` (admin prompt)
# or Hydra config override: python scripts/run_retrieval.py hydra.runtime.output_dir=~/action_retrieval_data/outputs
```

## Git Workflow & Commits

Each meaningful code addition is committed and pushed to `origin/main` with a clean, descriptive message
(e.g., "Add CLAUDE.md and repo skeleton", "Add Hydra configuration composition", "Add Phase-0 environment
report script"). This keeps the project's history auditable and reproducible from git alone, per spec 17.4.

## Key Files & Entry Points

- `3d_retrieval_experiment_infrastructure_spec.md` — authoritative specification (read-only reference).
- `CLAUDE.md` (this file) — project context and setup instructions for future sessions.
- `configs/config.yaml` — root Hydra composition; all experiments configured here.
- `scripts/smoke_env.py` — Phase-0 acceptance script (run inside WSL2 after CoppeliaSim setup).
- `src/action_retrieval/utils/env_report.py` — environment/reproducibility metadata collection.
- `src/action_retrieval/encoders/base.py` — SceneEncoder Protocol and registry (core abstraction).
- `src/action_retrieval/data/schema.py` — EpisodeRecord/ExperienceRecord dataclasses (once Phase 1 begins).

## Future Claude Session Instructions

If you're a Claude session reading this after Phase 0:

1. **Do not run WSL2 / CoppeliaSim / RLBench commands.** The user has set these up; you maintain the code.
2. **Check Phase status.** Read the "Phased Build Order" section above. Ask the user for confirmation
   (e.g., smoke test exit status) before advancing to the next phase.
3. **Commit + push after each logical code change,** with a clean message.
4. **Reference the spec** (`3d_retrieval_experiment_infrastructure_spec.md`), not this document, for
   design decisions. This file is a guide; the spec is the source of truth.
5. **Update this file** if new constraints or decisions emerge (e.g., "Uni3D doesn't fit; using PCA
   fallback").
