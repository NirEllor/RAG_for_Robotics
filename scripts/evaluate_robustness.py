#!/usr/bin/env python3
"""Evaluate retrieval under controlled point-cloud and image perturbations."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.evaluation.retrieval_eval import (  # noqa: E402
    evaluate_retrieval_run,
    load_relevance_annotations,
)
from action_retrieval.retrieval.dataset import (  # noqa: E402
    ExportedEpisode,
    load_manifest,
    load_exported_episode,
)
from action_retrieval.retrieval.pipeline import build_encoder  # noqa: E402
from action_retrieval.retrieval.ranking import top_k_cosine  # noqa: E402
from action_retrieval.retrieval.encoders import EpisodeEmbedding  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=[
        "pose_descriptor", "rgb_histogram", "global_color", "geometry_only", "uni3d", "ptv3"
    ])
    parser.add_argument("--conditions", nargs="+", default=["viewpoint", "occlusion", "geometry_noise"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-queries", type=int, default=0, help="0 means all queries")
    return parser.parse_args()


def _infer_annotations(dataset_root: Path) -> dict[str, set[str]]:
    manifest = load_manifest(dataset_root)
    annotations: dict[str, set[str]] = {}
    for _, group in manifest.groupby("task_name", dropna=False):
        ids = [str(value) for value in group["episode_id"].tolist()]
        for query_id in ids:
            annotations[query_id] = set(ids) - {query_id}
    return annotations


def _point_cloud_keys(observation: dict[str, np.ndarray]) -> list[str]:
    return [key for key in ("front_point_cloud_world", "front_point_cloud_camera") if key in observation]


def _perturb_episode(episode: ExportedEpisode, condition: str, rng: np.random.Generator) -> ExportedEpisode:
    observation = {key: np.array(value, copy=True) for key, value in episode.observation.items()}
    point_keys = _point_cloud_keys(observation)

    if condition == "viewpoint":
        angle = float(rng.uniform(-np.pi, np.pi))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        for key in point_keys:
            points = observation[key]
            observation[key] = np.einsum("...c,dc->...d", points, rotation).astype(points.dtype, copy=False)
    elif condition == "geometry_noise":
        for key in point_keys:
            points = observation[key]
            noise = rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
            observation[key] = (points.astype(np.float32) + noise).astype(points.dtype, copy=False)
    elif condition == "occlusion":
        for key in point_keys:
            points = observation[key]
            mask = rng.random(points.shape[:-1]) < 0.35
            points[mask] = np.nan
        if "front_rgb" in observation:
            rgb = observation["front_rgb"]
            mask = rng.random(rgb.shape[:-1]) < 0.35
            rgb[mask] = 0
    else:
        raise ValueError(f"Unsupported robustness condition: {condition}")

    return replace(episode, observation=observation)


def main() -> int:
    args = _args()
    ks = sorted({k for k in args.ks if k > 0})
    if not ks:
        raise ValueError("At least one positive k is required")

    manifest = load_manifest(args.dataset_root)
    rows = list(manifest.iterrows())
    if args.max_queries > 0:
        rows = rows[: args.max_queries]
    annotations = load_relevance_annotations(args.annotations) if args.annotations and args.annotations.exists() else _infer_annotations(args.dataset_root)
    rng = np.random.default_rng(args.seed)
    aggregate_rows: list[dict[str, object]] = []
    json_runs: list[dict[str, object]] = []

    for method in args.methods:
        print(f"[baseline] embedding clean candidates: {method}", flush=True)
        encoder = build_encoder(method, seed=args.seed)
        candidates = [
            EpisodeEmbedding(
                episode_id=episode.episode_id,
                task_name=episode.task_name,
                split=episode.split,
                vector=np.asarray(encoder.encode(episode), dtype=np.float32),
                encoder_name=getattr(encoder, "name", method),
            )
            for _, row in rows
            for episode in [load_exported_episode(args.dataset_root, row)]
        ]

        for condition in args.conditions:
            print(f"[condition] {condition}", flush=True)
            query_embeddings = [
                EpisodeEmbedding(
                    episode_id=episode.episode_id,
                    task_name=episode.task_name,
                    split=episode.split,
                    vector=np.asarray(encoder.encode(episode), dtype=np.float32),
                    encoder_name=getattr(encoder, "name", method),
                )
                for _, row in rows
                for episode in [_perturb_episode(load_exported_episode(args.dataset_root, row), condition, rng)]
            ]
            matches = {
                query.episode_id: top_k_cosine(query, candidates, max(ks), exclude_query_episode=True)
                for query in query_embeddings
            }
            from action_retrieval.retrieval.pipeline import RetrievalRunResult

            retrieval_run = RetrievalRunResult(embeddings=candidates, matches=matches)
            for k in ks:
                aggregate = evaluate_retrieval_run(
                    retrieval_run, annotations, method=method, k=k
                ).aggregate.to_dict()
                aggregate["condition"] = condition
                aggregate_rows.append(aggregate)
                json_runs.append(aggregate)
            del query_embeddings, matches, retrieval_run
            gc.collect()

        del candidates, encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    summary = pd.DataFrame(aggregate_rows)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    payload = {
        "dataset_root": str(args.dataset_root),
        "annotations": str(args.annotations) if args.annotations and args.annotations.exists() else "inferred from task_name",
        "methods": args.methods,
        "conditions": args.conditions,
        "ks": ks,
        "num_queries": len(rows),
        "runs": json_runs,
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = [
        "# Robustness Evaluation",
        "",
        f"- Dataset: `{args.dataset_root}`",
        f"- Queries: `{len(rows)}`",
        f"- Annotations: `{payload['annotations']}`",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "Perturbations are applied to query observations only; the candidate database remains clean.",
    ]
    (args.output_dir / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Robustness outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
