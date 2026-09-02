#!/usr/bin/env python
"""Evaluate retrieval baselines with Top-1 / Top-K metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.evaluation.retrieval_eval import (
    evaluate_retrieval_methods,
    load_relevance_annotations,
    per_query_to_dataframe,
    runs_to_dataframe,
)
from action_retrieval.retrieval.dataset import load_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_reach_target",
        help="Exported dataset root.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT / "configs" / "evaluation" / "rlbench_reach_target_hand_labels.json",
        help="Relevance annotation file (JSON mapping query episode_id -> list of relevant episode_ids).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "random",
            "pose_descriptor",
            "rgb_histogram",
            "global_color",
            "geometry_only",
            "uni3d",
            "ptv3",
        ],
        help="Retrieval methods/encoders to evaluate.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Top-K cutoffs to report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "retrieval_mvp",
        help="Directory where evaluation outputs are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for stochastic methods such as random retrieval.",
    )
    parser.add_argument("--query-split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--candidate-split", choices=["all", "train", "val", "test"], default="all")
    return parser.parse_args()


def _format_markdown_table(df: pd.DataFrame, *, columns: list[str]) -> str:
    rows = []
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows.append(header)
    rows.append(separator)
    for _, row in df[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _infer_task_relevance_annotations(dataset_root: Path) -> dict[str, set[str]]:
    manifest = load_manifest(dataset_root)
    if "episode_id" not in manifest.columns or "task_name" not in manifest.columns:
        raise ValueError(
            "Cannot infer relevance annotations because the manifest is missing "
            "'episode_id' or 'task_name' columns."
        )

    annotations: dict[str, set[str]] = {}
    grouped = manifest.groupby("task_name", dropna=False)
    for _, group in grouped:
        episode_ids = [str(episode_id) for episode_id in group["episode_id"].tolist()]
        for query_episode_id in episode_ids:
            annotations[query_episode_id] = {
                candidate_episode_id
                for candidate_episode_id in episode_ids
                if candidate_episode_id != query_episode_id
            }
    return annotations


def main() -> int:
    args = _parse_args()
    if args.annotations.exists():
        annotations = load_relevance_annotations(args.annotations)
        annotations_source = args.annotations
    else:
        print(
            f"WARNING: Missing annotations file {args.annotations}; "
            "inferring relevance from dataset task_name groups."
        )
        annotations = _infer_task_relevance_annotations(args.dataset_root)
        annotations_source = None
    runs = evaluate_retrieval_methods(
        args.dataset_root,
        annotations,
        methods=args.methods,
        ks=args.ks,
        seed=args.seed,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = runs_to_dataframe(runs)
    per_query_df = per_query_to_dataframe(runs)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    per_query_df.to_csv(output_dir / "per_query_metrics.csv", index=False)

    report_columns = [
        "method",
        "k",
        "num_queries",
        "top1_accuracy",
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "map_at_k",
        "ndcg_at_k",
        "hit_rate_at_k",
    ]
    report_md = [
        "# Retrieval Evaluation Summary",
        "",
        f"- Dataset root: `{args.dataset_root}`",
        f"- Annotations: `{annotations_source or 'inferred from manifest task_name groups'}`",
        f"- Methods: {', '.join(args.methods)}",
        f"- K values: {', '.join(str(k) for k in args.ks)}",
        f"- Query split: `{args.query_split}`",
        f"- Candidate split: `{args.candidate_split}`",
        "",
        "## Aggregate metrics",
        "",
        _format_markdown_table(summary_df.sort_values(['method', 'k']), columns=report_columns),
    ]
    if not per_query_df.empty:
        best_rows = (
            per_query_df.sort_values(["method", "k", "top1_score"], ascending=[True, True, False])
            .groupby(["method", "k"], as_index=False)
            .head(1)
        )
        report_md.extend(
            [
                "",
                "## Best-scoring query examples",
                "",
                _format_markdown_table(
                    best_rows,
                    columns=[
                        "method",
                        "k",
                        "query_episode_id",
                        "topk_episode_ids",
                        "top1_score",
                        "first_relevant_rank",
                    ],
                ),
            ]
        )
    (output_dir / "summary_report.md").write_text("\n".join(report_md), encoding="utf-8")

    payload = {
        "dataset_root": str(args.dataset_root),
        "annotations": str(annotations_source or "inferred from manifest task_name groups"),
        "methods": args.methods,
        "ks": args.ks,
        "summary": summary_df.to_dict(orient="records"),
        "per_query": per_query_df.to_dict(orient="records"),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Retrieval Evaluation")
    print("=" * 80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Annotations: {annotations_source or 'inferred from manifest task_name groups'}")
    print(f"Output dir: {output_dir}")
    print()
    print(summary_df.sort_values(["method", "k"]).to_string(index=False))
    print()
    print(f"Markdown report written to: {output_dir / 'summary_report.md'}")
    print("Per-query outputs written to:")
    print(f"  {output_dir / 'per_query_metrics.csv'}")
    print(f"  {output_dir / 'summary_metrics.csv'}")
    print(f"  {output_dir / 'evaluation.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
