# Final Report Outline

This outline is the writing contract for the final submission. Numeric claims in
the report must be copied from the locked CSV/JSON artifacts listed below.

## 1. Introduction and Related Work

- Motivation: retrieval-augmented robotic manipulation using 3D scene embeddings.
- Research question: whether pretrained 3D geometric similarity is meaningful for
  retrieving manipulation experiences and how it behaves under scene changes.
- Hypothesis: geometric retrieval should be more robust than appearance retrieval.
- Scope: RLBench scenes, trajectories, and task-level outcomes; retrieval is the
  primary evaluated component.
- Related work: RLBench, retrieval-augmented planning, Uni3D, Point Transformer
  V3, and image/color retrieval baselines.
- Proposal correspondence: learned downstream planning was proposed, but the
  implemented downstream experiment is an offline trajectory-transfer proxy.

## 2. Method

- Dataset representation: exported observation and trajectory records with a
  manifest, task labels, and fixed train/test splits.
- Experience database: each record contains a scene observation, trajectory
  data, task identity, and outcome metadata.
- Encoders and baselines: random, pose descriptor, RGB histogram, global color,
  geometry-only, Uni3D, and Pointcept PTv3.
- Similarity and ranking: embedding generation followed by nearest-neighbor
  retrieval for k in {1, 2, 3}.
- Real backends: official Uni3D checkpoint and Pointcept v1.5.2 PTv3 checkpoint.
- Action-aware ablation: frozen Uni3D backbone with a small trajectory-state
  projection head; train only on the train split.
- Reproducibility: Cluster environment, checkpoint hashes, configs, and SLURM
  commands.

## 3. Evaluation and Results

### 3.1 Main Retrieval Evaluation

Use `retrieval_full/v2_multitask_full` for the primary 19-task result and
`retrieval_all/v2_multitask_subset8` for the focused subset analysis.

Report precision, recall, MRR, MAP, NDCG, Top-1 accuracy, and hit rate for each
method and k. Include `full_top1_by_method.png`, `full_hit_rate_by_k.png`, and
`subset8_top1_by_method.png`.

### 3.2 Robustness

Use `robustness/v2_multitask_subset8`. Explain that perturbations are applied to
query observations while candidates remain clean. Include
`robustness_hit_rate_at_1.png` and discuss viewpoint, occlusion, and geometry
noise separately.

### 3.3 Held-Out Projection-Head Ablation

Use only the held-out comparison in `action_head/uni3d_subset8_heldout_final`.
The protocol is test queries against train candidates, with 422 train episodes,
141 test episodes, and zero overlap. Include `heldout_un3d_action_head.png`.

The earlier all-episode action-head value is an in-sample diagnostic and must
not be presented as the primary generalization result.

### 3.4 Downstream and Simulator Pilot

- Offline downstream transfer is reported from `downstream/v2_multitask_full`.
- The simulator pilot is reported as an integration/replay diagnostic only.
- The pilot used derived joint-position replay and is not evidence of a learned
  planner or a ground-truth planning-success benchmark.

## 4. Discussion and Conclusion

- Main finding: 3D retrieval provides signal above random, but color/appearance
  baselines are stronger in the evaluated RLBench configuration.
- Uni3D is stronger than PTv3 in the subset result, but neither dominates the
  color baselines.
- The held-out action-aware head does not improve over the original Uni3D
  representation in the current experiment.
- Limitations: task-group relevance labels, offline downstream proxy, limited
  simulator action representation, and no learned action-generating planner.
- Future work: explicit action collection, learned trajectory/action decoder,
  cross-task relevance definitions, and broader variation-controlled evaluation.

## Evidence Locations

| Evidence | Location |
| --- | --- |
| Full retrieval | `RAG_for_Robotics_outputs/evaluation/retrieval_full/v2_multitask_full` |
| Subset retrieval | `RAG_for_Robotics_outputs/evaluation/retrieval_all/v2_multitask_subset8` |
| Robustness | `RAG_for_Robotics_outputs/evaluation/robustness/v2_multitask_subset8` |
| Held-out Uni3D | `RAG_for_Robotics_outputs/evaluation/retrieval_heldout/uni3d_subset8` |
| Held-out action head | `RAG_for_Robotics_outputs/evaluation/action_head/uni3d_subset8_heldout_final` |
| Offline downstream | `RAG_for_Robotics_outputs/evaluation/downstream/v2_multitask_full` |
| Planning pilot | `RAG_for_Robotics_outputs/evaluation/planning_pilot` |
| Reproducibility | `RAG_for_Robotics_outputs/evaluation/reproducibility` |
| Figures | `RAG_for_Robotics_outputs/evaluation/report_figures` |
