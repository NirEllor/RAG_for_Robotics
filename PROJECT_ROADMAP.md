# Project Roadmap

This roadmap is the live execution plan for the next 8 days. It is aligned with the research
direction approved by Dr. Benaim:

> Does 3D geometric similarity between scenes, encoded via a pre-trained 3D representation,
> provide a meaningful retrieval signal for robotic manipulation planning, and under what
> scene variations does this signal remain reliable?

Current date: August 29, 2026
Target deadline: September 6, 2026

## Where We Are Now

We are currently here:

1. the CLUSTER is building the 8-task RLBench subset dataset;
2. `reach_target` has already been exported and validated;
3. retrieval evaluation works on `reach_target`;
4. we still need a clean subset-8 evaluation protocol;
5. `Uni3D` / `Point Transformer V3` must be made truly usable in the MVP, not left as placeholders.

## Live Roadmap

| Status | Step | Goal | Next concrete work | Done when |
| --- | --- | --- | --- | --- |
| NOW | 1 | Finish the 8-task dataset build | Let the SLURM build complete, validate the manifest, and keep the subset dataset root stable | `v2_multitask_subset8` exists, validates, and contains the 8 intended tasks |
| NEXT | 2 | Build evaluation labels for subset-8 | Generate task-level relevance annotations from the subset manifest | A JSON annotations file exists for all subset-8 episodes |
| NEXT | 3 | Run clean retrieval evaluation on subset-8 | Evaluate `no retrieval`, `random`, `image`, `geometry`, and color-aware baselines on the same episodes/candidate sets | Results are reproducible and report Top-1 / Top-K side by side |
| NEXT | 4 | Make retrieval color-aware | Improve color sensitivity so episodes differing mainly by object color separate more reliably | Color-changing tasks become distinguishable in retrieval |
| NEXT | 5 | Integrate `Uni3D` and `PTv3` | Replace placeholder backbones with real cluster-ready learned 3D encoders | `uni3d` and `ptv3` run end-to-end through the same retrieval API |
| NEXT | 6 | Re-run evaluation with learned 3D backbones | Compare learned 3D retrieval against image and hand-crafted baselines | We have fair, quantitative evidence for 3D geometric retrieval |
| LATER IN WEEK | 7 | Robustness tests | Evaluate viewpoint change, partial occlusion, and geometry variation | We can state under which scene changes the signal remains reliable |
| LATER IN WEEK | 8 | Minimal downstream planning baseline | Add nearest-trajectory transfer from retrieved demos | Retrieval is connected to planning, not only ranking |
| LAST | 9 | Freeze final outputs | Lock reports, figures, configs, and README reproduction commands | Final results are ready to present and reproduce |

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

1. finish the subset-8 dataset build on the CLUSTER,
2. validate that dataset root,
3. generate subset-8 annotations,
4. run clean retrieval evaluation,
5. wire in `Uni3D` and `PTv3`,
6. repeat the evaluation with learned 3D embeddings,
7. run the variation/robustness checks,
8. freeze the final results by September 6, 2026.
