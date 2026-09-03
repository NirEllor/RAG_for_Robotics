# Project Glossary

This file is a plain-language dictionary for the repo. It explains the main
libraries, files, dataset objects, and the math notation used in the spec.

## 1. Big Picture

- `RAG_for_Robotics` is a research project about retrieving robot demonstrations
  that are both geometrically similar and action-compatible.
- The completed benchmark is the full 19-task RLBench export, with `ReachTarget`
  retained as the simulator pilot task.
- The current workflow is:
  1. load or export demonstrations,
  2. store them as a versioned dataset,
  3. build embeddings,
  4. retrieve similar episodes,
  5. evaluate whether the retrieved demo is useful for the query.

## 2. Core Libraries

| Term | Meaning | Why it matters here |
| --- | --- | --- |
| `RLBench` | A robotics benchmark library with manipulation tasks and saved demos. | It provides tasks like `ReachTarget` and the demo format we are using. |
| `PyRep` | Python API that controls CoppeliaSim. | It is the bridge between Python and the simulator. |
| `CoppeliaSim` | The robotics simulator itself. | RLBench and PyRep depend on it for live simulation and rendering. |
| `Hydra` | Configuration management library. | It composes dataset, encoder, retrieval, and experiment settings. |
| `OmegaConf` | YAML/config object library used by Hydra. | It loads and reads the config files. |
| `NumPy` | Array library. | It stores observations, trajectories, and point clouds. |
| `Pandas` | Tabular data library. | It stores the manifest of episodes. |
| `PyArrow` | Columnar data backend used by Parquet. | It lets the manifest be saved as `manifest.parquet`. |

## 3. Important Files

| File | Meaning |
| --- | --- |
| [`scripts/smoke_env.py`](scripts/smoke_env.py) | Phase 0 check: confirms the environment works and loads or generates one episode. |
| [`scripts/build_reach_target_dataset.py`](scripts/build_reach_target_dataset.py) | Phase 1 dataset exporter. It writes the first versioned dataset on disk. |
| [`src/action_retrieval/data/exporter.py`](src/action_retrieval/data/exporter.py) | Converts saved demos into the project dataset layout. |
| [`src/action_retrieval/data/validator.py`](src/action_retrieval/data/validator.py) | Checks that the dataset files exist, match the manifest, and are internally consistent. |
| [`src/action_retrieval/data/schema.py`](src/action_retrieval/data/schema.py) | Defines the manifest records and dataset metadata objects. |
| [`src/action_retrieval/data/transforms.py`](src/action_retrieval/data/transforms.py) | Coordinate-frame helpers such as camera-to-world and world-to-camera transforms. |
| [`src/action_retrieval/simulation/saved_importer.py`](src/action_retrieval/simulation/saved_importer.py) | Loads saved RLBench demonstrations when live simulation is unstable. |
| [`configs/dataset/rlbench_reach_target.yaml`](configs/dataset/rlbench_reach_target.yaml) | Dataset settings for the MVP ReachTarget export. |
| [`configs/retrieval/exact_cosine.yaml`](configs/retrieval/exact_cosine.yaml) | Retrieval config for normalized cosine similarity. |

## 4. Dataset Terms

### Episode

An episode is one complete robot demonstration for one task instance.
For `ReachTarget`, it means one short run where the arm moves toward the target.

### Manifest

The manifest is the table of all exported episodes.
It stores episode-level metadata such as:
- `episode_id`
- `task_name`
- `variation_id`
- `seed`
- `split`
- `success`
- file paths
- checksums

### `observation.npz`

This file stores the episode’s observation-side arrays.
Typical contents include:
- `front_rgb`
- `front_depth`
- `front_point_cloud_world`
- camera intrinsics/extrinsics

Think of it as: “what the robot saw.”

### `trajectory.npz`

This file stores the episode’s trajectory-side arrays.
Typical contents include:
- `joint_positions`
- `joint_velocities`
- `gripper_open`
- `gripper_pose`
- action sequences if available

Think of it as: “what the robot did over time.”

### `metadata.json`

Per-episode JSON metadata file.
It ties together the arrays, checksums, coordinate frame, and dataset provenance.

### Split

The train/val/test partition.
The spec requires splitting by **episode**, not by frame, so frames from the
same episode never leak across splits.

## 5. Coordinate and Geometry Terms

| Term | Meaning |
| --- | --- |
| `world frame` | The global simulator coordinate system. |
| `camera frame` | Coordinates expressed relative to a camera. |
| `robot base frame` | Coordinates expressed relative to the robot base. |
| `intrinsics` | Camera parameters describing focal length and principal point. |
| `extrinsics` | Camera pose in the world, usually a 4x4 transform matrix. |
| `point cloud` | A set of 3D points representing geometry in space. |
| `target crop` | A local point-cloud crop around the object or target region. |
| `xyz_only` | Keep only 3D coordinates, not color channels, in a point cloud. |

### Useful transforms

- `world_to_camera(points, extrinsics)`: convert world coordinates into the camera frame.
- `camera_to_world(points, extrinsics)`: convert camera coordinates back into the world frame.

## 6. Retrieval Terms

### Retrieval pipeline

The retrieval pipeline is the system that will eventually:
- embed episodes,
- compare a query episode to a database,
- return the most similar candidate episodes,
- evaluate whether those candidates are also action-compatible.

### Current status

The retrieval pipeline is implemented and evaluated. It includes dataset export,
episode embedding, leave-one-out ranking, task-group relevance evaluation,
real Uni3D and PTv3 backends, robustness tests, and a held-out projection-head
pilot. The downstream and simulator results are explicitly proxies or pilots,
not a learned action-generating planner.

### Exact cosine retrieval

If embeddings are normalized, cosine retrieval ranks candidates by:

```text
sim(z_query, z_i) = (z_query · z_i) / (||z_query|| * ||z_i||)
```

The top candidate is the one with the highest similarity score.

## 7. Math Notation

| Symbol | Meaning |
| --- | --- |
| `x_i` | The i-th observation or input sample. |
| `y_i` | The i-th label or metadata target. |
| `z_i` | The embedding of the i-th sample. |
| `f_theta(x)` | A model with parameters `theta` that maps input `x` to an output. |
| `L` | A loss function. |
| `K` or `Top-K` | The number of best retrieval results kept. |
| `sim(a, b)` | Similarity function between vectors `a` and `b`. |
| `||z||` | Vector norm, usually L2 norm. |
| `argmax` | Choose the index with the largest value. |

### Common retrieval objective

For a query embedding `z_query` and database embeddings `z_i`, retrieval often
means:

```text
i* = argmax_i sim(z_query, z_i)
```

That means “pick the database item with the highest similarity to the query.”

### Common learning objective

The implemented projection-head objective is trajectory-signature regression.
The project also discusses possible objectives like:
- contrastive loss,
- ranking loss,
- regression loss for pose or trajectory prediction,
- action-compatibility scores.

The exact implemented formula is documented in Appendix B of `FINAL_REPORT.md`.

## 8. RLBench Data Layout

RLBench saved demos usually look like this:

```text
reach_target/
└── variation0/
    └── episodes/
        ├── episode0/
        │   ├── low_dim_obs.pkl
        │   ├── front_rgb/
        │   ├── front_depth/
        │   ├── front_mask/
        │   └── ...
        ├── episode1/
        └── ...
```

That is the source material.
The Phase 1 exporter converts that source format into the project’s own
manifest-plus-NPZ layout.

## 9. Phase Vocabulary

| Phase | Meaning |
| --- | --- |
| Phase 0 | Environment feasibility and smoke testing. |
| Phase 1 | Dataset generation and validation. |
| Phase 2 | Retrieval MVP. |
| Phase 3 | Action-compatibility evaluation. |
| Phase 4-6 | Robustness, adaptation, and downstream evaluation. |
| Phase 7 | Flow matching stretch goal. |

## 10. Short Reading Guide

If you are trying to understand the repo in order:

1. Read `CLAUDE.md` for project status and constraints.
2. Read `README.md` for the short summary.
3. Read `3d_retrieval_experiment_infrastructure_spec.md` for the authoritative design.
4. Read `scripts/smoke_env.py` to see how the environment is checked.
5. Read `scripts/build_reach_target_dataset.py` to see how Phase 1 data is exported.
6. Read `src/action_retrieval/data/exporter.py` and `validator.py` for the dataset logic.
