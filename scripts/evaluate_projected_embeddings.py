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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--projected-embeddings", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def main() -> int:
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
    full_matches = {
        query.episode_id: top_k_cosine(query, embeddings, len(embeddings) - 1)
        for query in embeddings
    }
    run = RetrievalRunResult(embeddings=embeddings, matches=full_matches)
    evaluated = [
        evaluate_retrieval_run(run, annotations, method="uni3d_action_head", k=k)
        for k in sorted(set(args.ks))
    ]
    summary = runs_to_dataframe(evaluated)
    detail = per_query_to_dataframe(evaluated)
    # Keep the aggregate aligned with the auditable per-query ground truth.
    # This also catches stale summaries produced by an older evaluator build.
    for k in sorted(set(args.ks)):
        mask = detail["k"] == k
        if mask.any():
            recomputed = float(detail.loc[mask, "top1_correct"].mean())
            summary.loc[summary["k"] == k, "top1_accuracy"] = recomputed
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
                "baseline_summary": str(args.baseline_summary),
                "method": "uni3d_action_head",
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
