# Project Roadmap

| Step | Goal | Current status | Next concrete work | Done when |
| --- | --- | --- | --- | --- |
| 1 | Finish dataset ingestion | A raw RLBench mirror is being extracted and the multi-task export path is wired up | Complete the remaining raw-task extraction, then rebuild the multi-task dataset | The dataset root contains the intended task set and the validator passes |
| 2 | Make retrieval color-aware | Retrieval MVP exists with `random`, `pose_descriptor`, `rgb_histogram`, and `geometry_only` | Improve the baseline so global color and object-specific color differences are captured more explicitly | Retrieval can separate episodes that differ mainly by color when that matters |
| 3 | Formalize evaluation | Top-1 / Top-K evaluation already exists | Report methods side by side with a stable protocol, relevance labels, and visual inspection | Results are reproducible and comparable across methods and cutoffs |
| 4 | Add learned 3D backbones | Hand-crafted encoders are the current baseline | Integrate `Uni3D` first, then `Point Transformer V3` through the same encoder API | Learned 3D embeddings run end-to-end through retrieval and evaluation |
| 5 | Scale to more tasks | Only a subset of tasks is currently available locally | Add more RLBench tasks and more episodes, including live generation only if the simulator path becomes stable | The dataset spans several tasks and a larger episode count |
| 6 | Add downstream planning | Not implemented yet | Implement a nearest-trajectory transfer baseline from retrieved episodes | Retrieval improves a simple downstream planning control versus no retrieval |
| 7 | Robustness and adaptation | Not implemented yet | Test viewpoint changes, partial occlusion, and lightweight adaptation | Retrieval remains useful under perturbations |
| 8 | Stretch goals | Not implemented yet | Explore flow matching or more advanced planners | Only after the earlier steps are stable and reproducible |

## Immediate Priority

Right now the most important order is:

1. finish ingesting the RLBench raw mirror,
2. rebuild the multi-task dataset,
3. run evaluation again with the existing baselines,
4. then improve the retrieval representation to handle color more explicitly.

