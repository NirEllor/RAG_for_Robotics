"""Retrieval evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalQueryMetrics:
    query_episode_id: str
    method: str
    k: int
    relevant_count: int
    retrieved_count: int
    first_relevant_rank: int | None
    hit_at_k: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision_at_k: float
    ndcg_at_k: float
    top1_correct: bool
    top1_score: float | None
    topk_mean_score: float
    topk_scores: list[float] = field(default_factory=list)
    topk_episode_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["topk_scores"] = list(self.topk_scores)
        data["topk_episode_ids"] = list(self.topk_episode_ids)
        return data


@dataclass(frozen=True)
class RetrievalAggregateMetrics:
    method: str
    k: int
    num_queries: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    map_at_k: float
    ndcg_at_k: float
    top1_accuracy: float
    mean_top1_score: float
    mean_topk_score: float
    median_first_relevant_rank: float | None
    hit_rate_at_k: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def precision_at_k(relevant_flags: Sequence[int | bool], k: int) -> float:
    """Implement the precision_at_k operation used by this module."""
    if k <= 0:
        return 0.0
    retrieved = list(relevant_flags)[:k]
    if not retrieved:
        return 0.0
    return float(np.mean(np.asarray(retrieved, dtype=np.float32)))


def recall_at_k(relevant_flags: Sequence[int | bool], total_relevant: int, k: int) -> float:
    """Implement the recall_at_k operation used by this module."""
    if total_relevant <= 0:
        return 0.0
    retrieved = list(relevant_flags)[:k]
    return float(np.sum(np.asarray(retrieved, dtype=np.float32)) / float(total_relevant))


def mean_reciprocal_rank(relevant_flags: Sequence[int | bool]) -> float:
    """Implement the mean_reciprocal_rank operation used by this module."""
    for index, flag in enumerate(relevant_flags, start=1):
        if bool(flag):
            return 1.0 / float(index)
    return 0.0


def average_precision_at_k(
    relevant_flags: Sequence[int | bool],
    total_relevant: int,
    k: int,
) -> float:
    """Implement the average_precision_at_k operation used by this module."""
    flags = np.asarray(list(relevant_flags)[:k], dtype=np.float32)
    if flags.size == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for index, flag in enumerate(flags, start=1):
        if flag:
            hits += 1
            precision_sum += hits / float(index)
    if hits == 0 or total_relevant <= 0:
        return 0.0
    return precision_sum / float(min(total_relevant, k))


def discounted_cumulative_gain(relevant_flags: Sequence[int | bool], k: int) -> float:
    """Implement the discounted_cumulative_gain operation used by this module."""
    flags = np.asarray(list(relevant_flags)[:k], dtype=np.float32)
    if flags.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, flags.size + 2, dtype=np.float32))
    gains = (2.0**flags - 1.0) / discounts
    return float(np.sum(gains))


def normalized_discounted_cumulative_gain(
    relevant_flags: Sequence[int | bool],
    total_relevant: int,
    k: int,
) -> float:
    """Implement the normalized_discounted_cumulative_gain operation used by this module."""
    flags = list(relevant_flags)[:k]
    if not flags:
        return 0.0

    dcg = discounted_cumulative_gain(flags, k)
    ideal_flags = [1] * min(max(total_relevant, 0), k)
    ideal_dcg = discounted_cumulative_gain(ideal_flags, k)
    if ideal_dcg == 0.0:
        return 0.0
    return float(dcg / ideal_dcg)
