# Project Dictionary

This file is the plain-language reference for the project. It explains the
main terms, files, data objects, and the math notation used across the repo.

If you want the function-level view of the code, see
[`PROJECT_API_REFERENCE.md`](PROJECT_API_REFERENCE.md).

## 1. Project in One Sentence

`RAG_for_Robotics` is a robotics retrieval project: given a robot scene or
episode, we try to retrieve past demonstrations that are geometrically similar,
color-aware when needed, and eventually useful for downstream manipulation
planning.

## 2. Current Status

- Dataset export and validation are implemented.
- Retrieval baselines already exist.
- Retrieval evaluation already exists.
- The project is now focused on:
  - scaling the dataset,
  - making retrieval more color-aware,
  - adding more RLBench tasks,
  - then integrating learned 3D encoders such as Uni3D and Point
    Transformer V3.

## 3. Core Libraries

| Term | Meaning | Why it matters |
| --- | --- | --- |
| RLBench | Robotics benchmark with manipulation tasks and demonstrations. | It is the source of episodes and tasks. |
| PyRep | Python API for CoppeliaSim. | It is the bridge between Python and the simulator. |
| CoppeliaSim | The simulator itself. | RLBench and PyRep depend on it for live rendering and execution. |
| Hydra | Configuration system. | It controls dataset, retrieval, and evaluation settings. |
| OmegaConf | YAML-backed config object library. | It is what Hydra uses under the hood. |
| NumPy | Array library. | It stores observations, trajectories, and point clouds. |
| Pandas | Table library. | It stores the dataset manifest and evaluation summaries. |
| PyArrow | Columnar storage backend. | It lets the manifest be saved as Parquet. |

## 4. Important Files

| File | Meaning |
| --- | --- |
| [`scripts/smoke_env.py`](scripts/smoke_env.py) | Phase 0 environment check. |
| [`scripts/build_reach_target_dataset.py`](scripts/build_reach_target_dataset.py) | Phase 1 single-task exporter. |
| [`scripts/build_multitask_dataset.py`](scripts/build_multitask_dataset.py) | Multi-task dataset exporter. |
| [`scripts/fetch_rlbench_mirror.py`](scripts/fetch_rlbench_mirror.py) | Downloads and extracts the RLBench raw mirror. |
| [`scripts/run_retrieval_mvp.py`](scripts/run_retrieval_mvp.py) | Runs the retrieval MVP on the exported dataset. |
| [`scripts/evaluate_retrieval.py`](scripts/evaluate_retrieval.py) | Runs Top-1 / Top-K retrieval evaluation. |
| [`src/action_retrieval/data/exporter.py`](src/action_retrieval/data/exporter.py) | Converts demonstrations into the project dataset layout. |
| [`src/action_retrieval/data/validator.py`](src/action_retrieval/data/validator.py) | Checks dataset integrity. |
| [`src/action_retrieval/retrieval/pipeline.py`](src/action_retrieval/retrieval/pipeline.py) | Loads episodes, builds embeddings, and ranks neighbors. |
| [`src/action_retrieval/retrieval/encoders.py`](src/action_retrieval/retrieval/encoders.py) | Current MVP encoders. |
| [`src/action_retrieval/evaluation/retrieval_eval.py`](src/action_retrieval/evaluation/retrieval_eval.py) | Evaluation logic and metric aggregation. |
| [`src/action_retrieval/simulation/saved_importer.py`](src/action_retrieval/simulation/saved_importer.py) | Loads saved RLBench demos. |
| [`src/action_retrieval/simulation/raw_rlbench_importer.py`](src/action_retrieval/simulation/raw_rlbench_importer.py) | Loads raw RLBench mirror data. |
| [`configs/dataset/rlbench_multitask.yaml`](configs/dataset/rlbench_multitask.yaml) | Multi-task dataset config. |
| [`configs/evaluation/rlbench_reach_target_hand_labels.json`](configs/evaluation/rlbench_reach_target_hand_labels.json) | Relevance labels for the current evaluation set. |

## 5. Dataset Terms

### Episode

An episode is one complete robot run for one task instance.
It starts at reset and ends when the episode terminates.

In this project, an episode is the main unit we store, embed, and retrieve.

### Saved demo

A saved demo is RLBench's on-disk representation of an episode.
It usually contains the raw simulator observations, low-dimensional state,
and RGB/depth/point-cloud data for each timestep.

### Exported episode

An exported episode is the project-specific version of a demo.
It is written into the dataset root with files like `observation.npz`,
`trajectory.npz`, and `metadata.json`, plus one manifest row.

### Manifest

The manifest is the table of all exported episodes.
It stores episode-level metadata such as:

- `episode_id`
- `task_name`
- `variation_id`
- `split`
- `success`
- `source_kind`
- file paths
- checksums and provenance

### `observation.npz`

This file stores the episode's observation-side arrays.
Typical contents include:

- `front_rgb`
- `front_depth`
- `front_point_cloud_world`
- camera intrinsics/extrinsics

Think of it as: what the robot saw.

### `trajectory.npz`

This file stores the episode's trajectory-side arrays.
Typical contents include:

- `joint_positions`
- `joint_velocities`
- `gripper_open`
- `gripper_pose`
- action sequences if available

Think of it as: what the robot did over time.

### `metadata.json`

Per-episode JSON metadata file.
It ties together the arrays, checksums, coordinate frame, and source
provenance.

### Split

The train/val/test partition.
The repo splits by episode, not by frame, so frames from the same episode never
leak across splits.

## 6. Observation Terms

### `front_rgb[0]`

The first RGB frame in the episode from the front camera.

### `front_rgb[-1]`

The last RGB frame in the episode from the front camera.

### `num_observations`

The number of timesteps or frames in the episode.
If `num_observations` is 70, that usually means the episode has 70 observation
steps from start to finish.

### Why the first and last frames matter

The current encoders deliberately use the first and last observation because
they often capture:

- the initial scene layout,
- the final arrangement after the manipulation.

That makes them useful for a cheap retrieval baseline.

## 7. Geometry and Coordinate Terms

| Term | Meaning |
| --- | --- |
| world frame | Global simulator coordinate system. |
| camera frame | Coordinates expressed relative to a camera. |
| robot base frame | Coordinates expressed relative to the robot base. |
| intrinsics | Camera parameters such as focal length and principal point. |
| extrinsics | Camera pose in the world, usually a 4x4 transform matrix. |
| point cloud | A set of 3D points representing geometry in space. |
| target crop | A local point-cloud crop around the object or target region. |
| xyz_only | Keep only 3D coordinates, not color channels, in a point cloud. |

## 8. Retrieval Terms

### Retrieval pipeline

The retrieval pipeline:

1. loads exported episodes,
2. turns each episode into an embedding,
3. compares query embeddings against the database,
4. returns the top-k nearest neighbors,
5. evaluates whether the retrieved neighbors are relevant.

### Current retrieval baselines

The current MVP already includes:

- `random`
- `pose_descriptor`
- `rgb_histogram`
- `geometry_only`

These are all episode-level retrieval baselines.

### What is being embedded?

At the moment, the unit of retrieval is the whole episode.
So yes, the retrieval system embeds episodes, not single frames.

### Exact cosine retrieval

If embeddings are normalized, cosine similarity is:

```text
sim(z_query, z_i) = (z_query dot z_i) / (||z_query|| * ||z_i||)
```

The top candidate is the one with the highest similarity score.

## 9. Evaluation Terms

### Top-1

The single best retrieved neighbor.
If it is relevant, the retrieval is counted as correct at rank 1.

### Top-K

The best K retrieved neighbors.
This checks whether a relevant episode appears anywhere in the shortlist.

### Relevance annotations

The JSON annotation file says which episodes should count as relevant to each
query episode.
For example, in the current ReachTarget evaluation set, episode pairs are
hand-labeled as relevant because they match the same underlying task pattern.

### Common metrics

- `recall@k`
- `precision@k`
- `MRR`
- `MAP@k`
- `NDCG@k`
- `top1_accuracy`
- `hit_rate@k`

## 10. Math Notation

| Symbol | Meaning |
| --- | --- |
| `x_i` | The i-th sample or observation. |
| `y_i` | The i-th label or target. |
| `z_i` | The embedding of the i-th sample. |
| `f_theta(x)` | A model with parameters `theta` that maps `x` to an output. |
| `L` | A loss function. |
| `K` | The number of top retrieved results kept. |
| `sim(a, b)` | Similarity between vectors `a` and `b`. |
| `||z||` | Vector norm, usually L2 norm. |
| `argmax` | Choose the index with the largest value. |

### Common retrieval objective

```text
i* = argmax_i sim(z_query, z_i)
```

That means: choose the database item most similar to the query.

### What future learned encoders mean

Later, `z_i` may come from:

- Uni3D,
- Point Transformer V3,
- or another 3D backbone.

The retrieval API should stay the same even if the encoder changes.

## 11. Reading Order

If you want to understand the repo in order:

1. Read `CLAUDE.md` for project status and constraints.
2. Read this file for the terms.
3. Read `PROJECT_ROADMAP.md` for the next steps.
4. Read `README.md` for the short summary.
5. Read `scripts/smoke_env.py` and the dataset scripts for the implementation.
