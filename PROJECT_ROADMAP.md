# Project Roadmap

This roadmap is the live execution plan for the next 8 days. It is aligned with the research
direction approved by Dr. Benaim:

> Does 3D geometric similarity between scenes, encoded via a pre-trained 3D representation,
> provide a meaningful retrieval signal for robotic manipulation planning, and under what
> scene variations does this signal remain reliable?

Current date: August 29, 2026
Target deadline: September 6, 2026

## Research Status in One Sentence

We now have a working retrieval-and-evaluation pipeline on the 8-task subset; the strongest current baselines are color-aware, while the remaining open question is whether a real learned 3D backbone can beat those baselines under scene variation.

## Where We Are Now

We are currently here:

1. the CLUSTER has finished building the 8-task RLBench subset dataset;
2. task-level relevance annotations for subset-8 have been generated;
3. subset-8 retrieval evaluation has completed successfully;
4. `reach_target` was already exported and validated as the smoke benchmark;
5. `Uni3D` is now wired to try a real pretrained checkpoint first, with a safe proxy fallback if the cluster setup is incomplete;
6. `Point Transformer V3` now has the same real-backend adapter path, again with proxy fallback;
7. the current retrieval evidence is strong for color-aware baselines, but we still need the real learned 3D backbone results and robustness checks to satisfy the research question.

## Live Roadmap

| Status | Step | Goal | Next concrete work | Done when |
| --- | --- | --- | --- | --- |
| DONE | 1 | Finish the 8-task dataset build | Let the SLURM build complete, validate the manifest, and keep the subset dataset root stable | `v2_multitask_subset8` exists, validates, and contains the 8 intended tasks |
| DONE | 2 | Build evaluation labels for subset-8 | Generate task-level relevance annotations from the subset manifest | A JSON annotations file exists for all subset-8 episodes |
| DONE | 3 | Run clean retrieval evaluation on subset-8 | Evaluate `no retrieval`, `random`, `image`, `geometry`, and color-aware baselines on the same episodes/candidate sets | Results are reproducible and report Top-1 / Top-K side by side |
| DONE | 4 | Make retrieval color-aware | Improve color sensitivity so episodes differing mainly by object color separate more reliably | Color-changing tasks are distinguishable and color-aware baselines are evaluated |
| IN PROGRESS | 5 | Make Uni3D real | Load an actual pretrained Uni3D checkpoint on the cluster and validate that the retrieval API uses it | `uni3d` runs on the official backbone instead of the handcrafted proxy |
| IN PROGRESS | 6 | Make PTv3 real | Load an actual pretrained Point Transformer V3 checkpoint on the cluster and validate that the retrieval API uses it | `ptv3` runs on the official backbone instead of the handcrafted proxy |
| NEXT | 7 | Re-run evaluation with learned 3D backbones | Compare learned 3D retrieval against image and hand-crafted baselines | We have fair, quantitative evidence for 3D geometric retrieval |
| NEXT | 8 | Robustness tests | Evaluate viewpoint change, partial occlusion, and geometry variation | We can state under which scene changes the signal remains reliable |
| NEXT | 9 | Minimal downstream planning baseline | Add nearest-trajectory transfer from retrieved demos | Retrieval is connected to planning, not only ranking |
| LAST | 10 | Freeze final outputs | Lock reports, figures, configs, commands, and the final comparison tables | Final results are ready to present and reproduce |

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
| Sep 3-4 | Integrate `Uni3D` / `PTv3` | Cluster-ready learned 3D encoder path |
| Sep 5 | Re-run evaluation and robustness checks | Learned 3D comparison and variation analysis |
| Sep 6 | Freeze results and write the final summary | Presentation-ready outputs and reproducible commands |

## Immediate Priority

The active order is:

1. finish the real Uni3D hookup and validate it on the cluster,
2. finish the real PTv3 hookup and validate it on the cluster,
3. repeat the evaluation with learned 3D embeddings,
4. run the variation/robustness checks,
5. add the minimal downstream planning baseline,
6. freeze the final results by September 6, 2026.

## What Is Already Effectively Done

| Area | Status | Notes |
| --- | --- | --- |
| Dataset export | Done | `v2_multitask_subset8` is built and validated. |
| Evaluation labels | Done | Subset-8 annotations JSON exists. |
| Retrieval evaluation | Done | Baselines and proxy 3D backbones have been evaluated on subset-8. |
| Color-aware retrieval | Done | `global_color` and `rgb_histogram` are implemented and empirically strong. |
| Uni3D/PTv3 integration | In progress | Both encoders now have real-backend adapters, each with safe proxy fallback. |
