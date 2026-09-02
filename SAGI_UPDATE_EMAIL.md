# Email Draft To Sagi

**Subject:** Research project update: 3D retrieval for robotic manipulation

Hi Sagi,

I wanted to update you on the progress of the project and ask if we could meet
briefly to discuss the final direction and presentation.

The project investigates whether similarity between 3D scene observations,
encoded by pretrained 3D representations, provides a useful retrieval signal
for robotic manipulation. The implementation now includes:

- an RLBench experience database with 19 tasks and 1,804 exported episodes;
- a focused 8-task subset for rapid experiments;
- retrieval baselines based on random selection, pose descriptors, RGB
  histograms, global color, and geometry-only features;
- real pretrained Uni3D and Pointcept Point Transformer V3 backends;
- Top-1/Top-K retrieval evaluation with task-level relevance annotations;
- controlled robustness tests for viewpoint changes, partial occlusion, and
  geometry noise;
- an offline trajectory-transfer compatibility evaluation.

The main current result is that color-aware and pose-based baselines are very
strong on this RLBench subset, Uni3D is competitive, and PTv3 is informative
but weaker in this setup. This is a useful result rather than a failed
experiment because it indicates that the retrieval signal is sensitive to the
task distribution and the available scene variation.

There are also important limitations that I want to report explicitly. The
current downstream experiment is an offline trajectory-transfer proxy, not a
full learned planner and not a simulator planning-success benchmark. The
backbones are used pretrained and frozen; a small action-aware projection-head
pilot is being prepared, but full backbone fine-tuning is outside the remaining
time budget. I am also preparing a small RLBench trajectory-replay pilot on a
few simple tasks, but I do not want to present it as general planning success
unless it executes reliably and has a proper baseline.

I would appreciate your feedback on whether this scope and the negative result
are appropriate for the final report, and I would be glad to meet to review the
results and the remaining presentation choices.

Best,
Nir
