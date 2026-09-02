# Related Work Sources

This file records the primary sources to cite in the final report. The report
should paraphrase the contributions and use the bibliography entries below;
quoted text should remain short.

## Simulation and Demonstration Benchmarks

**RLBench: The Robot Learning Benchmark & Learning Environment**

James et al. introduce RLBench as a benchmark and learning environment with
many procedurally varied manipulation tasks and demonstrations generated in a
simulator. It is relevant because our experience database and task-level
evaluation are built from RLBench episodes. Unlike RLBench policy-learning
benchmarks, our main measurement is retrieval quality over stored scene and
trajectory records.

Citation: James et al., 2019, arXiv:1909.12271.
https://arxiv.org/abs/1909.12271

**What Matters in Learning from Offline Human Demonstrations for Robot
Manipulation (robomimic)**

Mandlekar et al. study offline demonstration learning across simulated and
real manipulation tasks, emphasizing dataset quality, algorithm choices, and
reproducible evaluation. This motivates treating the trajectory database and
the train/test protocol as first-class experimental objects. Our work differs
by evaluating scene-embedding retrieval before training a control policy.

Citation: Mandlekar et al., 2021, arXiv:2108.03298.
https://arxiv.org/abs/2108.03298

**MimicGen: A Data Generation System for Scalable Robot Learning using Human
Demonstrations**

MimicGen demonstrates how demonstrations can be adapted to new contexts to
scale robot-learning datasets. It is relevant to the future extension of our
pipeline: retrieved trajectories could be transformed or adapted before being
passed to a downstream planner. In the current project, trajectories are
retrieved and evaluated offline rather than adapted into new executable plans.

Citation: Mandlekar et al., 2023, arXiv:2310.17596.
https://arxiv.org/abs/2310.17596

## Retrieval and Demonstration-Based Planning

**Multi-Agent Behavior Retrieval: Retrieval-Augmented Policy Training for
Cooperative Push Manipulation by Mobile Robots**

Kuroki et al. propose a skill database and a learned skill representation for
retrieving behavior demonstrations and augmenting policy training. This is the
closest conceptual precedent for using memory retrieval to support robot
control. Our project isolates an earlier question: whether a pretrained 3D
scene representation is a useful retrieval key. We do not claim to reproduce
their policy-training or real-robot success-rate setting.

Citation: Kuroki et al., 2023, arXiv:2312.02008.
https://arxiv.org/abs/2312.02008

**Distilling Motion Planner Augmented Policies into Visual Control Policies for
Robot Manipulation**

Liu et al. combine motion-planning supervision with policy learning and study
visual control in obstructed manipulation settings. This work clarifies the
distinction between a retrieval module and a downstream action-generating
policy. Our simulator pilot does not implement this type of learned policy;
it only validates the simulator integration and performs a clearly labeled
trajectory replay diagnostic.

Citation: Liu et al., CoRL 2022, PMLR 164:641-650.
https://proceedings.mlr.press/v164/liu22b.html

## Pretrained 3D Representations

**Uni3D: Exploring Unified 3D Representation at Scale**

Zhou et al. present Uni3D as a 3D foundation model that aligns point-cloud
features with image-text aligned representations, using large-scale pretraining
and a 2D-initialized vision transformer. The paper reports broad 3D transfer
and retrieval applications. We use an official pretrained Uni3D checkpoint as
one retrieval encoder, but test it on robotic scene episodes rather than
claiming that its original benchmarks directly predict manipulation success.

Citation: Zhou et al., 2023, arXiv:2310.06773.
https://arxiv.org/abs/2310.06773

**Point Transformer V3: Simpler, Faster, Stronger**

Wu et al. develop PTv3 around serialized point-cloud processing and scaling,
with an emphasis on efficiency and strong performance across 3D tasks. We use
the Pointcept v1.5.2 runtime and a compatible pretrained checkpoint as a second
3D retrieval encoder. The comparison is useful because it tests whether a
modern point-cloud backbone alone guarantees a useful task-retrieval signal;
our results show that representation quality is task- and dataset-dependent.

Citation: Wu et al., CVPR 2024, pp. 4840-4851.
https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html

## Positioning of This Project

The project sits at the intersection of these areas but makes a narrower,
testable contribution. It compares hand-designed geometric and appearance
baselines with real pretrained 3D encoders for nearest-neighbor retrieval of
robot manipulation experiences. It also tests robustness to query viewpoint,
occlusion, and geometry noise. The downstream and simulator experiments are
reported as proxies/pilots, not as a learned planning-success benchmark.
