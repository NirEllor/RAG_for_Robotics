# Third-Party Dependencies

This directory tracks external projects that are pinned to specific commits for reproducibility.

## PyRep

[PyRep](https://github.com/stepjam/PyRep) — Python API for CoppeliaSim simulator.

- **Installation:** `git submodule add <url>` (to be determined by user after testing compatibility with
  CoppeliaSim 4.1 EDU and Ubuntu 24.04).
- **Pinned commit:** (to be documented after Phase 0).
- **Install command:** `pip install -e third_party/PyRep`

## RLBench

[RLBench](https://github.com/stepjam/RLBench) — Robot manipulation tasks and environment.

- **Installation:** `git submodule add <url>` (exact fork/branch TBD).
- **Pinned commit:** (to be documented after Phase 0).
- **Install command:** `pip install -e third_party/RLBench`

## CoppeliaSim

[CoppeliaSim](https://coppeliarobotics.com) — Robotics simulator (EDU version 4.1).

- **Download:** Manual download from CoppeliaSim website (not a git dependency).
- **Setup:** Extract and set `COPPELIASIM_ROOT`, `LD_LIBRARY_PATH`, `QT_QPA_PLATFORM_PLUGIN_PATH`
  environment variables (WSL2 Ubuntu only; see CLAUDE.md for details).
- **Version:** 4.1 EDU (pinned per RLBench compatibility).

## Attribution

All third-party code remains under its original license. We include only references and setup instructions
here, not copies of upstream repositories. Consult each project's repository for license details.
