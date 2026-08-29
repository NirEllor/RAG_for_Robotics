# Project Roadmap

| Step | Goal | Current status | Next concrete work | Done when |
| --- | --- | --- | --- | --- |
| 1 | Finish dataset ingestion | A raw RLBench mirror is being extracted and the multi-task export path is wired up | Complete the remaining raw-task extraction, then rebuild the multi-task dataset | The dataset root contains the intended task set and the validator passes |
| 2 | Make retrieval color-aware | Retrieval MVP exists with `random`, `pose_descriptor`, `rgb_histogram`, and `geometry_only` | Improve the baseline so global color and object-specific color differences are captured more explicitly | Retrieval can separate episodes that differ mainly by color when that matters |
| 3 | Formalize evaluation | Top-1 / Top-K evaluation already exists | Report methods side by side with a stable protocol, relevance labels, and visual inspection | Results are reproducible and comparable across methods and cutoffs |
| 4 | Add learned 3D backbones | Hand-crafted encoders are the current baseline | Integrate `Uni3D` first, then `Point Transformer V3` through the same encoder API and cluster job helpers | Learned 3D embeddings run end-to-end through retrieval and evaluation |
| 5 | Scale to more tasks | Only a subset of tasks is currently available locally | Add more RLBench tasks and more episodes, including live generation only if the simulator path becomes stable | The dataset spans several tasks and a larger episode count |
| 6 | Add downstream planning | Not implemented yet | Implement a nearest-trajectory transfer baseline from retrieved episodes | Retrieval improves a simple downstream planning control versus no retrieval |
| 7 | Robustness and adaptation | Not implemented yet | Test viewpoint changes, partial occlusion, and lightweight adaptation | Retrieval remains useful under perturbations |
| 8 | Stretch goals | Not implemented yet | Explore flow matching or more advanced planners | Only after the earlier steps are stable and reproducible |

## Recommended Near-Term Task Subset

For the current time budget, focus the main experiments on a smaller subset of tasks that gives
good diversity without creating too much debugging overhead. Recommended subset:

1. `reach_target` - saved-demo anchor task and smoke-test baseline.
2. `open_drawer` - simple articulated object interaction.
3. `slide_block_to_color_target` - important for color-aware retrieval.
4. `close_jar` - precision grasping and closure geometry.
5. `insert_onto_square_peg` - tight pose/alignment task.
6. `stack_blocks` - classic stacking geometry.
7. `place_shape_in_shape_sorter` - shape/category discrimination.
8. `light_bulb_in` - fine manipulation with a compact target region.

Keep the remaining RLBench tasks in the full config as future expansion candidates. They are
useful later for scaling and robustness, but they should not block the near-term MVP.

## Immediate Priority

Right now the most important order is:

1. finish ingesting the RLBench raw mirror,
2. rebuild the multi-task dataset,
3. rebuild the dataset first on the recommended subset above,
4. run the `reach_target` evaluation cleanly,
5. integrate and validate `Uni3D/PTv3` on the cluster as required MVP backbones,
6. then improve the retrieval representation to handle color more explicitly,
7. later expand back to the full 19-task set once the MVP is stable.
