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
5. `Uni3D` / `Point Transformer V3` are wired into the MVP as cluster-ready proxies;
6. the current retrieval evidence is strong for color-aware baselines, but we still need a learned 3D backbone result and robustness checks to satisfy the research question.

## Live Roadmap

| Status | Step | Goal | Next concrete work | Done when |
| --- | --- | --- | --- | --- |
| DONE | 1 | Finish the 8-task dataset build | Let the SLURM build complete, validate the manifest, and keep the subset dataset root stable | `v2_multitask_subset8` exists, validates, and contains the 8 intended tasks |
| DONE | 2 | Build evaluation labels for subset-8 | Generate task-level relevance annotations from the subset manifest | A JSON annotations file exists for all subset-8 episodes |
| DONE | 3 | Run clean retrieval evaluation on subset-8 | Evaluate `no retrieval`, `random`, `image`, `geometry`, and color-aware baselines on the same episodes/candidate sets | Results are reproducible and report Top-1 / Top-K side by side |
| NEXT | 4 | Make retrieval color-aware | Improve color sensitivity so episodes differing mainly by object color separate more reliably | Color-changing tasks become distinguishable in retrieval |
| NEXT | 5 | Make the learned 3D backbones real | Replace the proxy `uni3d` / `ptv3` encoders with real checkpoints if available on the cluster | `uni3d` and `ptv3` reflect an actual learned 3D representation rather than a handcrafted proxy |
| NEXT | 6 | Re-run evaluation with learned 3D backbones | Compare learned 3D retrieval against image and hand-crafted baselines | We have fair, quantitative evidence for 3D geometric retrieval |
| LATER IN WEEK | 7 | Robustness tests | Evaluate viewpoint change, partial occlusion, and geometry variation | We can state under which scene changes the signal remains reliable |
| LATER IN WEEK | 8 | Minimal downstream planning baseline | Add nearest-trajectory transfer from retrieved demos | Retrieval is connected to planning, not only ranking |
| LAST | 9 | Freeze final outputs | Lock reports, figures, configs, commands, and the final comparison tables | Final results are ready to present and reproduce |

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

1. make the learned 3D backbones real or, if the cluster cannot support that in time, clearly document the proxy limitation,
2. repeat the evaluation with learned 3D embeddings,
3. run the variation/robustness checks,
4. add the minimal downstream planning baseline,
5. freeze the final results by September 6, 2026.
