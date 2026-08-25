# Retrieval Evaluation Protocol

This document explains how we evaluate retrieval methods in this project.

## 1. Goal

We want to know whether a query episode retrieves other episodes that are:

1. geometrically similar,
2. color-aware when color matters,
3. useful for downstream robotics planning.

The evaluation is intentionally lightweight first, then grows in difficulty.

## 2. What Is Evaluated

The current evaluation runs on exported episodes from the dataset root.
For each query episode, we rank all other episodes as retrieval candidates.

The current methods already supported by the code are:

- `random`
- `pose_descriptor`
- `rgb_histogram`
- `geometry_only`

Later, the same evaluation interface should also support learned 3D encoders
such as:

- Uni3D
- Point Transformer V3

## 3. Relevance Definition

Relevance is defined by a label file:

- [`configs/evaluation/rlbench_reach_target_hand_labels.json`](configs/evaluation/rlbench_reach_target_hand_labels.json)

That file maps `query_episode_id -> [relevant_episode_ids]`.

In other words, evaluation is not guessing relevance from similarity alone.
It uses explicit human-curated labels.

## 4. Ranking Setup

For each query:

1. encode the episode,
2. compare it against all other episodes,
3. sort candidates by cosine similarity,
4. remove the query episode itself from the candidate list,
5. compute metrics at one or more cutoffs.

The command that runs this is:

```bash
python scripts/evaluate_retrieval.py
```

## 5. Metrics

### Top-1 accuracy

Was the first retrieved episode relevant?

### Precision@K

Of the top K retrieved episodes, how many are relevant?

### Recall@K

Of all relevant episodes for the query, how many were found in the top K?

### MRR

Mean Reciprocal Rank. Rewards early retrieval of the first relevant item.

### MAP@K

Mean Average Precision truncated at K.

### NDCG@K

Normalized Discounted Cumulative Gain. Rewards relevant items appearing earlier
in the ranking.

### Hit rate@K

Did at least one relevant episode appear in the top K?

### Median first relevant rank

Across queries, what is the median position of the first relevant candidate?

## 6. How To Compare Methods

The recommended comparison order is:

1. `random`
2. `pose_descriptor`
3. `rgb_histogram`
4. `geometry_only`
5. learned 3D encoders

This order tells you whether the signal comes from:

- chance,
- explicit low-dimensional state,
- color,
- geometry,
- or a learned representation.

## 7. How To Read Top-1 vs Top-K

### Top-1

Use this when you care about the single best retrieved episode.
It is the strictest test.

### Top-K

Use this when multiple neighbors are acceptable.
It is more forgiving and often more realistic for robotics retrieval.

If `top1_accuracy` is weak but `recall@k` is strong, the method is finding
useful neighbors but not ranking them perfectly.

## 8. Color-Aware Retrieval

If the task is sensitive to object color, a geometry-only encoder is not enough.
Then we should compare against methods that include appearance:

- `rgb_histogram`
- future global color features
- future object-centric color features

For the project's current tasks, color can matter because episodes may differ
only by object color while sharing the same geometry.

## 9. Current Evaluation Command

```bash
python scripts/evaluate_retrieval.py \
  --dataset-root data/processed/v2_multitask \
  --methods random pose_descriptor rgb_histogram geometry_only \
  --ks 1 2 3
```

Outputs are written to:

- `outputs/evaluation/retrieval_mvp/summary_metrics.csv`
- `outputs/evaluation/retrieval_mvp/per_query_metrics.csv`
- `outputs/evaluation/retrieval_mvp/evaluation.json`

## 10. What A Good Result Looks Like

A useful retrieval method should ideally show:

- better than random top-1 accuracy,
- high recall at small K,
- stable results across query episodes,
- and qualitative neighbors that look physically or semantically similar.

If the method is color-aware, it should also preserve object-color distinctions
that matter for the task.

## 11. What Comes Next

After the current MVP:

1. add more tasks and more episodes,
2. evaluate on a broader task mix,
3. integrate Uni3D or Point Transformer V3,
4. test robustness to viewpoint and occlusion,
5. then move toward downstream planning.

