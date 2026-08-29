"""MVP retrieval pipeline for exported episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from action_retrieval.retrieval.dataset import ExportedEpisode, iter_exported_episodes
from action_retrieval.retrieval.encoders import (
    EpisodeEmbedding,
    GeometryOnlyEncoder,
    GlobalColorEncoder,
    PoseDescriptorEncoder,
    PointTransformerV3Encoder,
    RGBHistogramEncoder,
    RandomEpisodeEncoder,
    Uni3DEncoder,
)
from action_retrieval.retrieval.ranking import RetrievalMatch, top_k_cosine


@dataclass(frozen=True)
class RetrievalRunResult:
    embeddings: list[EpisodeEmbedding]
    matches: dict[str, list[RetrievalMatch]]


def build_encoder(encoder_name: str, *, output_dim: int = 512, seed: int = 42):
    encoder_name = encoder_name.lower()
    if encoder_name == "pose_descriptor":
        return PoseDescriptorEncoder()
    if encoder_name == "rgb_histogram":
        return RGBHistogramEncoder()
    if encoder_name == "global_color":
        return GlobalColorEncoder()
    if encoder_name == "geometry_only":
        return GeometryOnlyEncoder()
    if encoder_name == "random":
        return RandomEpisodeEncoder(output_dim=output_dim, seed=seed)
    if encoder_name == "uni3d":
        return Uni3DEncoder()
    if encoder_name in {"ptv3", "point_transformer_v3"}:
        return PointTransformerV3Encoder()
    raise ValueError(f"Unsupported encoder: {encoder_name}")


def embed_episodes(
    episodes: Iterable[ExportedEpisode],
    encoder_name: str = "pose_descriptor",
    *,
    output_dim: int = 512,
    seed: int = 42,
) -> list[EpisodeEmbedding]:
    encoder = build_encoder(encoder_name, output_dim=output_dim, seed=seed)
    embeddings: list[EpisodeEmbedding] = []
    for episode in episodes:
        embeddings.append(
            EpisodeEmbedding(
                episode_id=episode.episode_id,
                task_name=episode.task_name,
                split=episode.split,
                vector=np.asarray(encoder.encode(episode), dtype=np.float32),
                encoder_name=getattr(encoder, "name", encoder_name),
            )
        )
    return embeddings


def run_leave_one_out_retrieval(
    dataset_root: Path | str,
    encoder_name: str = "pose_descriptor",
    k: int = 1,
    *,
    output_dim: int = 512,
    seed: int = 42,
    exclude_query_episode: bool = True,
) -> RetrievalRunResult:
    embeddings = embed_episodes(
        iter_exported_episodes(dataset_root),
        encoder_name=encoder_name,
        output_dim=output_dim,
        seed=seed,
    )
    matches: dict[str, list[RetrievalMatch]] = {}
    for query in embeddings:
        matches[query.episode_id] = top_k_cosine(
            query,
            embeddings,
            k=k,
            exclude_query_episode=exclude_query_episode,
        )
    return RetrievalRunResult(embeddings=embeddings, matches=matches)
