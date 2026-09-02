#!/usr/bin/env python3
"""Evaluate saved projection-head embeddings and compare them with Uni3D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.evaluation.retrieval_eval import (
    RetrievalRunResult,
    evaluate_retrieval_run,
    load_relevance_annotations,
    per_query_to_dataframe,
    runs_to_dataframe,
)
from action_retrieval.retrieval.dataset import load_manifest
from action_retrieval.retrieval.encoders import EpisodeEmbedding
from action_retrieval.retrieval.ranking import top_k_cosine


def _args() -> argparse.Namespace:
    """Implement the _args operation used by this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--projected-embeddings", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--query-split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--candidate-split", default="all", choices=["all", "train", "val", "test"])
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point."""
    args = _args()
    manifest = load_manifest(args.dataset_root)
    manifest_by_id = {str(row.episode_id): row for row in manifest.itertuples()}
    payload = np.load(args.projected_embeddings, allow_pickle=False)
    ids = [str(value) for value in payload["episode_id"].tolist()]
    vectors = np.asarray(payload["embedding"], dtype=np.float32)
    if vectors.ndim != 2 or len(ids) != vectors.shape[0]:
        raise ValueError("Projected embedding file must contain aligned 2D embedding and episode_id arrays.")

    embeddings: list[EpisodeEmbedding] = []
    for episode_id, vector in zip(ids, vectors):
        row = manifest_by_id.get(episode_id)
        if row is None:
            raise KeyError(f"Projected embedding episode is absent from manifest: {episode_id}")
        embeddings.append(
            EpisodeEmbedding(
                episode_id=episode_id,
                task_name=str(row.task_name),
                split=str(row.split),
                vector=vector,
                encoder_name="uni3d_action_head",
            )
        )

    annotations = load_relevance_annotations(args.annotations)
    candidates = [
        item for item in embeddings
        if args.candidate_split == "all" or item.split == args.candidate_split
    ]
    queries = [
        item for item in embeddings
        if args.query_split == "all" or item.split == args.query_split
    ]
    candidate_ids = {item.episode_id for item in candidates}
    filtered_annotations = {
        query_id: relevant.intersection(candidate_ids)
        for query_id, relevant in annotations.items()
    }
    full_matches = {
        query.episode_id: top_k_cosine(query, candidates, len(candidates))
        for query in queries
    }
    run = RetrievalRunResult(embeddings=candidates, matches=full_matches)
    evaluated = [
        evaluate_retrieval_run(run, filtered_annotations, method="uni3d_action_head", k=k)
        for k in sorted(set(args.ks))
    ]
    summary = runs_to_dataframe(evaluated)
    detail = per_query_to_dataframe(evaluated)
    # Top-1 accuracy is invariant to the reported K. Derive it from the
    # audited K=1 precision column so stale aggregate fields cannot leak into
    # the comparison table.
    top1_rows = detail[detail["k"] == 1]
    if not top1_rows.empty:
        top1_accuracy = float(top1_rows["precision_at_k"].mean())
        summary["top1_accuracy"] = top1_accuracy
    baseline = pd.DataFrame()
    if args.baseline_summary is not None and args.baseline_summary.exists():
        baseline = pd.read_csv(args.baseline_summary)
        baseline = baseline[baseline["method"].astype(str).str.lower() == "uni3d"].copy()
        baseline["comparison_method"] = "uni3d_original"
    summary["comparison_method"] = "uni3d_action_head"
    comparison = pd.concat([baseline, summary], ignore_index=True, sort=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    detail.to_csv(args.output_dir / "per_query_metrics.csv", index=False)
    comparison.to_csv(args.output_dir / "comparison_with_uni3d.csv", index=False)
    report = [
        "# Action-Aware Projection Evaluation",
        "",
        "The projection head is evaluated from saved embeddings without recomputing the backbone.",
        "The Uni3D baseline is copied from the supplied locked summary CSV.",
        "",
        "## Comparison",
        "",
        "```text",
        comparison.to_string(index=False),
        "```",
    ]
    (args.output_dir / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "projected_embeddings": str(args.projected_embeddings),
                "annotations": str(args.annotations),
                "baseline_summary": str(args.baseline_summary) if args.baseline_summary else None,
                "method": "uni3d_action_head",
                "query_split": args.query_split,
                "candidate_split": args.candidate_split,
                "num_queries": len(queries),
                "num_candidates": len(candidates),
                "backbone_recomputed": False,
                "summary": summary.to_dict(orient="records"),
                "comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Projection evaluation written to: {args.output_dir}")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
