"""Retrieval evaluation over exported episode datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from action_retrieval.evaluation.metrics import (
    RetrievalAggregateMetrics,
    RetrievalQueryMetrics,
    average_precision_at_k,
    mean_reciprocal_rank,
    normalized_discounted_cumulative_gain,
    precision_at_k,
    recall_at_k,
)
from action_retrieval.retrieval.dataset import load_exported_episodes
from action_retrieval.retrieval.encoders import EpisodeEmbedding
from action_retrieval.retrieval.pipeline import RetrievalRunResult, embed_episodes
from action_retrieval.retrieval.ranking import RetrievalMatch, top_k_cosine


@dataclass(frozen=True)
class RetrievalEvaluationRun:
    method: str
    k: int
    aggregate: RetrievalAggregateMetrics
    per_query: list[RetrievalQueryMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "k": self.k,
            "aggregate": self.aggregate.to_dict(),
            "per_query": [item.to_dict() for item in self.per_query],
        }


def load_relevance_annotations(path: Path | str) -> dict[str, set[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing relevance annotations: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(query_id): {str(candidate) for candidate in candidates} for query_id, candidates in payload.items()}


def _infer_relevance_flags(retrieved_ids: Iterable[str], relevant_ids: set[str]) -> list[int]:
    return [1 if candidate in relevant_ids else 0 for candidate in retrieved_ids]


def evaluate_retrieval_run(
    run: RetrievalRunResult,
    relevance_annotations: dict[str, set[str]],
    *,
    method: str,
    k: int,
) -> RetrievalEvaluationRun:
    per_query: list[RetrievalQueryMetrics] = []
    for query_id, matches in run.matches.items():
        relevant_ids = relevance_annotations.get(query_id, set())
        retrieved_episode_ids = [match.candidate_episode_id for match in matches[:k]]
        retrieved_scores = [float(match.score) for match in matches[:k]]
        relevant_flags = _infer_relevance_flags(retrieved_episode_ids, relevant_ids)
        first_relevant_rank = next(
            (index for index, flag in enumerate(relevant_flags, start=1) if flag),
            None,
        )

        per_query.append(
            RetrievalQueryMetrics(
                query_episode_id=query_id,
                method=method,
                k=k,
                relevant_count=len(relevant_ids),
                retrieved_count=len(retrieved_episode_ids),
                first_relevant_rank=first_relevant_rank,
                hit_at_k=bool(relevant_flags and any(relevant_flags)),
                precision_at_k=precision_at_k(relevant_flags, k),
                recall_at_k=recall_at_k(relevant_flags, len(relevant_ids), k),
                reciprocal_rank=mean_reciprocal_rank(relevant_flags),
                average_precision_at_k=average_precision_at_k(relevant_flags, len(relevant_ids), k),
                ndcg_at_k=normalized_discounted_cumulative_gain(relevant_flags, len(relevant_ids), k),
                top1_correct=bool(relevant_flags[:1] and relevant_flags[0]),
                top1_score=retrieved_scores[0] if retrieved_scores else None,
                topk_mean_score=float(np.mean(retrieved_scores)) if retrieved_scores else 0.0,
                topk_scores=retrieved_scores,
                topk_episode_ids=retrieved_episode_ids,
            )
        )

    aggregate = RetrievalAggregateMetrics(
        method=method,
        k=k,
        num_queries=len(per_query),
        recall_at_k=float(np.mean([item.recall_at_k for item in per_query])) if per_query else 0.0,
        precision_at_k=float(np.mean([item.precision_at_k for item in per_query])) if per_query else 0.0,
        mrr=float(np.mean([item.reciprocal_rank for item in per_query])) if per_query else 0.0,
        map_at_k=float(np.mean([item.average_precision_at_k for item in per_query])) if per_query else 0.0,
        ndcg_at_k=float(np.mean([item.ndcg_at_k for item in per_query])) if per_query else 0.0,
        top1_accuracy=float(np.mean([float(item.top1_correct) for item in per_query])) if per_query else 0.0,
        mean_top1_score=float(np.mean([item.top1_score for item in per_query if item.top1_score is not None]))
        if any(item.top1_score is not None for item in per_query)
        else 0.0,
        mean_topk_score=float(np.mean([item.topk_mean_score for item in per_query])) if per_query else 0.0,
        median_first_relevant_rank=float(np.median([item.first_relevant_rank for item in per_query if item.first_relevant_rank is not None]))
        if any(item.first_relevant_rank is not None for item in per_query)
        else None,
        hit_rate_at_k=float(np.mean([float(item.hit_at_k) for item in per_query])) if per_query else 0.0,
    )

    return RetrievalEvaluationRun(method=method, k=k, aggregate=aggregate, per_query=per_query)


def evaluate_retrieval_methods(
    dataset_root: Path | str,
    relevance_annotations: dict[str, set[str]],
    methods: Iterable[str],
    ks: Iterable[int],
    *,
    output_dim: int = 512,
    seed: int = 42,
) -> list[RetrievalEvaluationRun]:
    dataset_root = Path(dataset_root)
    ks = sorted({int(k) for k in ks if int(k) > 0})
    if not ks:
        raise ValueError("At least one positive K value is required.")
    episodes = load_exported_episodes(dataset_root)
    embeddings_cache: dict[str, list[EpisodeEmbedding]] = {}

    runs: list[RetrievalEvaluationRun] = []
    for method in methods:
        embeddings = embeddings_cache.get(method)
        if embeddings is None:
            embeddings = embed_episodes(
                episodes,
                encoder_name=method,
                output_dim=output_dim,
                seed=seed,
            )
            embeddings_cache[method] = embeddings

        full_k = max(0, len(embeddings) - 1)
        matches: dict[str, list[RetrievalMatch]] = {}
        for query in embeddings:
            matches[query.episode_id] = top_k_cosine(
                query,
                embeddings,
                k=full_k,
                exclude_query_episode=True,
            )
        run = RetrievalRunResult(embeddings=embeddings, matches=matches)
        for k in ks:
            runs.append(
                evaluate_retrieval_run(
                    run,
                    relevance_annotations,
                    method=method,
                    k=k,
                )
            )
    return runs


def runs_to_dataframe(runs: Iterable[RetrievalEvaluationRun]) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for run in runs:
        row = run.aggregate.to_dict()
        row.update(
            {
                "method": run.method,
                "k": run.k,
                "num_queries": run.aggregate.num_queries,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def per_query_to_dataframe(runs: Iterable[RetrievalEvaluationRun]) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for run in runs:
        for item in run.per_query:
            row = item.to_dict()
            row.update({"method": run.method, "k": run.k})
            rows.append(row)
    return pd.DataFrame(rows)
