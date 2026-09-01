# Retrieval Evaluation Guide

This document describes the 19-task RLBench benchmark, the retrieval methods, and
the metrics reported by the evaluation scripts.

## Benchmark Tasks

| # | Task | Meaning |
| ---: | --- | --- |
| 1 | `reach_target` | Move the gripper to a target position. |
| 2 | `open_drawer` | Open a drawer. |
| 3 | `slide_block_to_color_target` | Slide a block to the matching colored target. |
| 4 | `sweep_to_dustpan_of_size` | Sweep an object into a dustpan of the appropriate size. |
| 5 | `meat_off_grill` | Remove a piece of meat from a grill. |
| 6 | `turn_tap` | Turn a tap. |
| 7 | `put_item_in_drawer` | Place an item inside a drawer. |
| 8 | `close_jar` | Close a jar. |
| 9 | `reach_and_drag` | Reach an object and drag it. |
| 10 | `stack_blocks` | Stack blocks on top of one another. |
| 11 | `light_bulb_in` | Insert a light bulb into its socket or holder. |
| 12 | `put_money_in_safe` | Put money into a safe. |
| 13 | `place_wine_at_rack_location` | Place a wine bottle at the correct rack location. |
| 14 | `put_groceries_in_cupboard` | Put groceries into a cupboard. |
| 15 | `place_shape_in_shape_sorter` | Insert a shape through its matching sorter opening. |
| 16 | `push_buttons` | Push buttons. |
| 17 | `insert_onto_square_peg` | Insert an object onto a square peg. |
| 18 | `stack_cups` | Stack cups. |
| 19 | `place_cups` | Place cups at the required locations. |

The full dataset contains 4 `reach_target` episodes and 100 episodes for each of
the other 18 tasks, for 1,804 episodes total.

## Retrieval Methods

| Method | Category | What it represents |
| --- | --- | --- |
| `random` | Sanity baseline | A deterministic random embedding per episode. It establishes a weak reference point. |
| `pose_descriptor` | Hand-crafted mixed baseline | RGB, point-cloud statistics, joint positions and velocities, gripper pose, and gripper state. |
| `rgb_histogram` | Image baseline | RGB summaries and histograms from the observation images; it ignores point clouds and robot state. |
| `global_color` | Color baseline | Global RGB statistics, histograms, and first-to-last-frame color changes. |
| `geometry_only` | Hand-crafted 3D baseline | Point-cloud statistics plus robot kinematics; it ignores RGB. |
| `uni3d` | Pretrained 3D foundation model | A real pretrained Uni3D checkpoint produces the point-cloud embedding. |
| `ptv3` | Pretrained 3D model | A real Point Transformer V3 checkpoint, loaded through Pointcept, produces the embedding. |

The `uni3d` and `ptv3` entries are real learned backends only when the evaluation
log does not report a proxy fallback and the real-backend smoke test succeeds.

## Cutoff `k`

| Value | Meaning |
| ---: | --- |
| `k=1` | Evaluate only the top-ranked retrieved episode. |
| `k=2` | Evaluate the top two retrieved episodes. |
| `k=3` | Evaluate the top three retrieved episodes. |

Increasing `k` allows more candidates to be returned, but precision can decrease
because additional candidates may be less relevant.

## Report Columns

| Column | Meaning |
| --- | --- |
| `method` | Retrieval method used for the row. |
| `k` | Number of top-ranked results considered. |
| `num_queries` | Number of episodes evaluated as queries. |
| `recall_at_k` | Fraction of all relevant candidates retrieved within the first `k` results. |
| `precision_at_k` | Fraction of the first `k` retrieved candidates that are relevant. |
| `mrr` | Mean Reciprocal Rank of the first relevant result. Rank 1 gives 1.0, rank 2 gives 0.5, and so on. |
| `map_at_k` | Mean Average Precision through rank `k`, considering both relevance and ordering. |
| `ndcg_at_k` | Rank-sensitive relevance score that gives more credit to relevant results appearing earlier. |
| `top1_accuracy` | Fraction of queries whose first result is relevant. |
| `mean_top1_score` | Mean similarity score of the first retrieved result. |
| `mean_topk_score` | Mean similarity score across the first `k` retrieved results. |
| `median_first_relevant_rank` | Median rank of the first relevant result; lower is better. |
| `hit_rate_at_k` | Fraction of queries with at least one relevant result among the first `k`. |

## Interpretation Notes

- `random` is a sanity check, not a competitive method.
- `top1_accuracy`, `mrr`, and `hit_rate_at_k` are the easiest metrics to explain in a presentation.
- `precision_at_k` measures the quality of returned candidates, while `recall_at_k` measures coverage of the relevant set.
- If explicit annotations are absent, the evaluator infers relevance from `task_name`; episodes from the same task are treated as relevant to one another.
- Therefore, the current benchmark measures task-level retrieval. It does not yet prove that a retrieved trajectory is physically executable in a new scene.
