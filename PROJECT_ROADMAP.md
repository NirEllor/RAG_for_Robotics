# Project Roadmap

This roadmap is the live execution plan for the next 8 days. It is aligned with the research
direction approved by Dr. Benaim:

> Does 3D geometric similarity between scenes, encoded via a pre-trained 3D representation,
> provide a meaningful retrieval signal for robotic manipulation planning, and under what
> scene variations does this signal remain reliable?

Current date: September 1, 2026
Target deadline: September 6, 2026

## Research Status in One Sentence

We now have a working retrieval-and-evaluation pipeline on both the 8-task subset and the full 19-task dataset; real Uni3D and Pointcept PTv3 backends have passed smoke validation, while the remaining open questions are robustness and downstream planning value.

## Where We Are Now

We are currently here:

1. the CLUSTER has finished building the 8-task RLBench subset dataset;
2. task-level relevance annotations for subset-8 have been generated;
3. subset-8 retrieval evaluation has completed successfully;
4. `reach_target` was already exported and validated as the smoke benchmark;
5. a real pretrained Uni3D checkpoint has been loaded and passed a real-backend smoke test;
6. a real Pointcept PTv3 checkpoint has been loaded and passed a real-backend forward smoke test;
7. the full 19-task dataset has been exported and the unified evaluation sweep is running or ready to run;
8. the current retrieval evidence is strong for color-aware baselines, while the remaining research work is robustness and downstream planning evaluation.

## Live Roadmap

| Status | Step | Goal | Next concrete work | Done when |
| --- | --- | --- | --- | --- |
| DONE | 1 | Finish the 8-task dataset build | Let the SLURM build complete, validate the manifest, and keep the subset dataset root stable | `v2_multitask_subset8` exists, validates, and contains the 8 intended tasks |
| DONE | 2 | Build evaluation labels for subset-8 | Generate task-level relevance annotations from the subset manifest | A JSON annotations file exists for all subset-8 episodes |
| DONE | 3 | Run clean retrieval evaluation on subset-8 | Evaluate `no retrieval`, `random`, `image`, `geometry`, and color-aware baselines on the same episodes/candidate sets | Results are reproducible and report Top-1 / Top-K side by side |
| DONE | 4 | Make retrieval color-aware | Improve color sensitivity so episodes differing mainly by object color separate more reliably | Color-changing tasks are distinguishable and color-aware baselines are evaluated |
| DONE | 5 | Make Uni3D real | Load and smoke-test the official pretrained Uni3D checkpoint on the cluster | `uni3d` uses the official backend without proxy fallback |
| DONE | 6 | Make PTv3 real | Align Pointcept and the clean PyTorch/CUDA environment, then smoke-test the official PTv3 checkpoint | `ptv3` uses the official backend and completes forward |
| IN PROGRESS | 7 | Complete full evaluation | Evaluate all 19 tasks with all baselines and both learned 3D backbones | Full-dataset CSV, JSON, and Markdown reports are complete |
| NEXT | 8 | Robustness tests | Evaluate viewpoint change, partial occlusion, and geometry variation | We can state under which scene changes the signal remains reliable |
| NEXT | 9 | Minimal downstream planning baseline | Add nearest-trajectory transfer from retrieved demos | Retrieval is connected to planning, not only ranking |
| LAST | 10 | Freeze final outputs | Lock reports, figures, configs, commands, and final comparison tables | Final results are ready to present and reproduce |

## Recommended Near-Term Task Subset

The 8-task subset is the right near-term benchmark because it covers diverse manipulation patterns
without blowing up debugging time:

1. `reach_target`
2. `open_drawer`
3. `slide_block_to_color_target`
4. `close_jar`
5. `stack_blocks`
6. `place_shape_in_shape_sorter`
7. `light_bulb_in`
8. `insert_onto_square_peg`

Keep the remaining 11 RLBench tasks in the full config as future scaling candidates. They matter
for eventual breadth, but they should not block the next 8 days.

## Deadline Plan

| Date | Focus | Output |
| --- | --- | --- |
| Aug 29-30 | Finish dataset subset-8 build and validation | Stable subset dataset root and manifest |
| Aug 31 | Build evaluation labels and sanity-check them | Subset-8 annotations JSON |
| Sep 1-2 | Run clean subset-8 retrieval evaluation | Tables for `random`, image, geometry, and color-aware baselines |
| Sep 1 | Complete full 19-task evaluation | Unified comparison across all methods |
| Sep 2-3 | Robustness and task-level analysis | Variation analysis and per-task breakdowns |
| Sep 4-5 | Minimal downstream planning baseline | Retrieved-trajectory evidence |
| Sep 6 | Freeze results and write the final summary | Presentation-ready outputs and reproducible commands |

## Immediate Priority

The active order is:

1. finish and verify the full 19-task evaluation,
2. run the variation/robustness checks,
3. add the minimal downstream planning baseline,
4. freeze the final results by September 6, 2026.

## What Is Already Effectively Done

| Area | Status | Notes |
| --- | --- | --- |
| Dataset export | Done | `v2_multitask_subset8` is built and validated. |
| Evaluation labels | Done | Subset-8 annotations JSON exists. |
| Retrieval evaluation | Done | Baselines and proxy 3D backbones have been evaluated on subset-8. |
| Color-aware retrieval | Done | `global_color` and `rgb_histogram` are implemented and empirically strong. |
| Uni3D/PTv3 integration | Done | Uni3D and Pointcept PTv3 real backends passed cluster smoke validation. |
| Full 19-task dataset | Done | `v2_multitask_full` contains all 19 configured tasks and 1,804 episodes. |
| Full unified evaluation | In progress | The all-methods sweep is running or awaiting final output verification. |
