# Project API Reference

This document is a practical reference for the APIs that already exist in the
repository.

The goal is to answer three questions quickly:

1. What does this API do?
2. What does it take in and return?
3. Where does it sit in the overall pipeline?

## 1. End-to-End Flow

| Stage | Main API / Script | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| Environment check | `scripts/smoke_env.py` | Local or cluster runtime | Smoke-test logs and optional episode artifacts | Confirms that RLBench, PyRep, and CoppeliaSim can be used in the current environment. |
| Raw data fetch | `scripts/fetch_rlbench_mirror.py` | Hugging Face RLBench mirror spec | Extracted raw RLBench mirror on disk | Downloads and unpacks the source demos. |
| Dataset export | `scripts/build_reach_target_dataset.py` | ReachTarget saved demos | `manifest.parquet`, `dataset_metadata.json`, `splits/*.json`, per-episode NPZ/JSON files | Builds the first project dataset. |
| Multitask export | `scripts/build_multitask_dataset.py` | Task config + raw/saved RLBench sources | Versioned multitask dataset root | Builds the 8-task subset or the full task set. |
| Retrieval embedding | `scripts/run_retrieval_mvp.py` | Exported dataset root + encoder name | Retrieval results CSV/JSON | Embeds episodes and ranks nearest neighbors. |
| Retrieval evaluation | `scripts/evaluate_retrieval.py` | Dataset root + relevance annotations + method list | Summary CSV/JSON/Markdown | Compares retrieval methods with Top-1 / Top-K metrics. |

## 2. Data APIs

| API | Location | Signature / Surface | Returns | Notes |
| --- | --- | --- | --- | --- |
| `load_manifest` | [`src/action_retrieval/retrieval/dataset.py`](src/action_retrieval/retrieval/dataset.py) | `load_manifest(dataset_root)` | `pandas.DataFrame` | Loads the exported episode manifest from `manifest.parquet`. |
| `load_exported_episode` | [`src/action_retrieval/retrieval/dataset.py`](src/action_retrieval/retrieval/dataset.py) | `load_exported_episode(dataset_root, row)` | `ExportedEpisode` | Loads one exported episode from manifest row + NPZ/JSON files. |
| `iter_exported_episodes` | [`src/action_retrieval/retrieval/dataset.py`](src/action_retrieval/retrieval/dataset.py) | `iter_exported_episodes(dataset_root)` | iterator of `ExportedEpisode` | Streams episodes lazily from a dataset root. |
| `load_exported_episodes` | [`src/action_retrieval/retrieval/dataset.py`](src/action_retrieval/retrieval/dataset.py) | `load_exported_episodes(dataset_root)` | `list[ExportedEpisode]` | Eager load of the whole exported dataset. |
| `validate_dataset_root` | [`src/action_retrieval/data/validator.py`](src/action_retrieval/data/validator.py) | `validate_dataset_root(dataset_root)` | `ValidationResult` | Checks file existence, hashes, and consistency. |
| `export_rlbench_dataset_from_specs` | [`src/action_retrieval/data/exporter.py`](src/action_retrieval/data/exporter.py) | `export_rlbench_dataset_from_specs(...)` | `DatasetBuildResult` | General exporter for batches of RLBench demos. |
| `export_reach_target_dataset_from_demos` | [`src/action_retrieval/data/exporter.py`](src/action_retrieval/data/exporter.py) | `export_reach_target_dataset_from_demos(...)` | `DatasetBuildResult` | Special-case exporter for ReachTarget demos. |
| `export_reach_target_dataset` | [`src/action_retrieval/data/exporter.py`](src/action_retrieval/data/exporter.py) | `export_reach_target_dataset(...)` | `DatasetBuildResult` | End-user ReachTarget dataset builder. |

### Data objects

| Object | Location | Meaning |
| --- | --- | --- |
| `EpisodeRecord` | [`src/action_retrieval/data/schema.py`](src/action_retrieval/data/schema.py) | Manifest row for one episode. |
| `DatasetMetadata` | [`src/action_retrieval/data/schema.py`](src/action_retrieval/data/schema.py) | Dataset-level metadata such as task list, source provenance, and build time. |
| `DatasetBuildResult` | [`src/action_retrieval/data/schema.py`](src/action_retrieval/data/schema.py) | Summary returned after a dataset export. |
| `ExportedEpisode` | [`src/action_retrieval/retrieval/dataset.py`](src/action_retrieval/retrieval/dataset.py) | Loaded exported episode with observation, trajectory, and metadata dictionaries. |

## 3. Retrieval APIs

| API | Location | Signature / Surface | Returns | Purpose |
| --- | --- | --- | --- | --- |
| `build_encoder` | [`src/action_retrieval/retrieval/pipeline.py`](src/action_retrieval/retrieval/pipeline.py) | `build_encoder(encoder_name, output_dim=..., seed=...)` | Encoder instance | Maps a string like `pose_descriptor`, `geometry_only`, `uni3d`, or `ptv3` to a concrete encoder. |
| `embed_episodes` | [`src/action_retrieval/retrieval/pipeline.py`](src/action_retrieval/retrieval/pipeline.py) | `embed_episodes(episodes, encoder_name=..., ...)` | `list[EpisodeEmbedding]` | Converts exported episodes into embeddings. |
| `run_leave_one_out_retrieval` | [`src/action_retrieval/retrieval/pipeline.py`](src/action_retrieval/retrieval/pipeline.py) | `run_leave_one_out_retrieval(dataset_root, encoder_name=..., k=...)` | `RetrievalRunResult` | Runs the full retrieval MVP on a dataset root. |
| `top_k_cosine` | [`src/action_retrieval/retrieval/ranking.py`](src/action_retrieval/retrieval/ranking.py) | `top_k_cosine(query, candidates, k=..., exclude_query_episode=True)` | `list[RetrievalMatch]` | Returns the nearest neighbors by cosine similarity. |
| `cosine_similarity` | [`src/action_retrieval/retrieval/ranking.py`](src/action_retrieval/retrieval/ranking.py) | `cosine_similarity(query, candidate)` | `float` | Computes pairwise similarity between embeddings. |

### Encoder surfaces

| Encoder | Name | What it uses | Status |
| --- | --- | --- | --- |
| `PoseDescriptorEncoder` | `pose_descriptor` | First/last RGB, point cloud stats, joint states, gripper state | Baseline MVP encoder |
| `RGBHistogramEncoder` | `rgb_histogram` | Global RGB statistics only | Baseline appearance encoder |
| `GlobalColorEncoder` | `global_color` | Global + temporal color summaries | Baseline appearance encoder |
| `GeometryOnlyEncoder` | `geometry_only` | 3D geometry + robot kinematics, no RGB | Baseline geometry encoder |
| `RandomEpisodeEncoder` | `random` | Deterministic seeded random projection | Sanity-check baseline |
| `Uni3DEncoder` | `uni3d` | Official Uni3D checkpoint when configured, proxy fallback otherwise | Real checkpoint loads when `UNI3D_REPO_ROOT` and `UNI3D_CHECKPOINT` are set |
| `PointTransformerV3Encoder` | `ptv3` | Official PTv3 checkpoint when configured, proxy fallback otherwise | Real checkpoint loads when `PTV3_REPO_ROOT` and `PTV3_CHECKPOINT` are set |

## 4. Evaluation APIs

| API | Location | Signature / Surface | Returns | Purpose |
| --- | --- | --- | --- | --- |
| `load_relevance_annotations` | [`src/action_retrieval/evaluation/retrieval_eval.py`](src/action_retrieval/evaluation/retrieval_eval.py) | `load_relevance_annotations(path)` | `dict[str, set[str]]` | Loads the query -> relevant-episode mapping from JSON. |
| `evaluate_retrieval_run` | [`src/action_retrieval/evaluation/retrieval_eval.py`](src/action_retrieval/evaluation/retrieval_eval.py) | `evaluate_retrieval_run(run, relevance_annotations, method=..., k=...)` | `RetrievalEvaluationRun` | Scores one retrieval run at a single cutoff. |
| `evaluate_retrieval_methods` | [`src/action_retrieval/evaluation/retrieval_eval.py`](src/action_retrieval/evaluation/retrieval_eval.py) | `evaluate_retrieval_methods(dataset_root, relevance_annotations, methods, ks, ...)` | `list[RetrievalEvaluationRun]` | Runs multiple methods and multiple K cutoffs. |
| `runs_to_dataframe` | [`src/action_retrieval/evaluation/retrieval_eval.py`](src/action_retrieval/evaluation/retrieval_eval.py) | `runs_to_dataframe(runs)` | `pandas.DataFrame` | Converts aggregate metrics to a table. |
| `per_query_to_dataframe` | [`src/action_retrieval/evaluation/retrieval_eval.py`](src/action_retrieval/evaluation/retrieval_eval.py) | `per_query_to_dataframe(runs)` | `pandas.DataFrame` | Converts per-query metrics to a table. |
| `precision_at_k` / `recall_at_k` / `mean_reciprocal_rank` / `average_precision_at_k` / `normalized_discounted_cumulative_gain` | [`src/action_retrieval/evaluation/metrics.py`](src/action_retrieval/evaluation/metrics.py) | Metric helpers | `float` | Standard ranking metrics used by the evaluation script. |

## 5. CLI Entry Points

| Script | What it does | Typical use |
| --- | --- | --- |
| [`scripts/run_retrieval_mvp.py`](scripts/run_retrieval_mvp.py) | Runs retrieval on one dataset and one encoder. | `python3 scripts/run_retrieval_mvp.py --encoder pose_descriptor --k 1` |
| [`scripts/evaluate_retrieval.py`](scripts/evaluate_retrieval.py) | Runs retrieval evaluation across many methods and K cutoffs. | `python3 scripts/evaluate_retrieval.py --dataset-root ... --methods random pose_descriptor uni3d ptv3` |
| [`scripts/build_multitask_dataset.py`](scripts/build_multitask_dataset.py) | Builds the multitask dataset root from RLBench sources. | `python3 scripts/build_multitask_dataset.py --config ...` |
| [`scripts/build_task_relevance_annotations.py`](scripts/build_task_relevance_annotations.py) | Builds the relevance label JSON for evaluation. | `python3 scripts/build_task_relevance_annotations.py ...` |

## 6. SLURM Helpers

| Script | What it launches | Notes |
| --- | --- | --- |
| [`slurm/run_evaluate.sh`](slurm/run_evaluate.sh) | SLURM job for retrieval evaluation | Defaults now include `uni3d` and `ptv3`. |
| [`slurm/run_retrieval.sh`](slurm/run_retrieval.sh) | SLURM job for the retrieval MVP | Good for quick neighbor inspection. |
| [`slurm/run_build_multitask_dataset.sh`](slurm/run_build_multitask_dataset.sh) | SLURM job for multitask dataset export | Used to build the subset-8 and full-19 datasets. |
| [`slurm/run_future_3d_encoder.sh`](slurm/run_future_3d_encoder.sh) | GPU-ready launcher for `uni3d` / `ptv3` | Now usable with the proxy encoders and ready for real checkpoints later. |

## 7. What Returns What

| Return type | Meaning |
| --- | --- |
| `RetrievalRunResult` | Embeddings plus ranked neighbors for each query episode. |
| `RetrievalEvaluationRun` | One scored method at one cutoff `k`. |
| `RetrievalAggregateMetrics` | Mean scores across all queries. |
| `RetrievalQueryMetrics` | Per-query metrics and retrieved episode ids. |
| `ValidationResult` | Dataset validation status, errors, and warnings. |
| `DatasetBuildResult` | Export summary, manifest path, split path, and counts. |

## 8. The Main Mental Model

The project currently has two practical APIs:

1. a **dataset API** that turns RLBench demonstrations into a versioned dataset;
2. a **retrieval API** that turns exported episodes into embeddings and nearest-neighbor results.

Then the **evaluation API** sits on top and compares methods with the same labels.

That is the core loop:

```text
RLBench demo -> exported episode -> episode embedding -> ranked neighbors -> evaluation metrics
```

## 9. Notes on the 3D Backends

`Uni3D` and `PTV3` are already wired into the retrieval and evaluation APIs.
At the moment they are deterministic point-cloud proxies that preserve the
expected interface. That means:

- the pipeline already accepts the names `uni3d` and `ptv3`,
- the cluster scripts already know how to launch them,
- the code can later be swapped to a real learned checkpoint with minimal API
  changes.
