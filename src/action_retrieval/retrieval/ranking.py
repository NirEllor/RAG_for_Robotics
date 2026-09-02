"""Retrieval ranking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from action_retrieval.retrieval.encoders import EpisodeEmbedding


@dataclass(frozen=True)
class RetrievalMatch:
    query_episode_id: str
    candidate_episode_id: str
    score: float
    rank: int


def cosine_similarity(query: np.ndarray, candidate: np.ndarray) -> float:
    """Implement the cosine_similarity operation used by this module."""
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(query) * np.linalg.norm(candidate))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(query, candidate) / denominator)


def top_k_cosine(
    query: EpisodeEmbedding,
    candidates: Iterable[EpisodeEmbedding],
    k: int,
    exclude_query_episode: bool = True,
) -> list[RetrievalMatch]:
    """Implement the top_k_cosine operation used by this module."""
    scored: list[RetrievalMatch] = []
    for candidate in candidates:
        if exclude_query_episode and candidate.episode_id == query.episode_id:
            continue
        score = cosine_similarity(query.vector, candidate.vector)
        scored.append(
            RetrievalMatch(
                query_episode_id=query.episode_id,
                candidate_episode_id=candidate.episode_id,
                score=score,
                rank=0,
            )
        )

    scored.sort(key=lambda item: item.score, reverse=True)
    ranked: list[RetrievalMatch] = []
    for index, item in enumerate(scored[:k], start=1):
        ranked.append(
            RetrievalMatch(
                query_episode_id=item.query_episode_id,
                candidate_episode_id=item.candidate_episode_id,
                score=item.score,
                rank=index,
            )
        )
    return ranked
