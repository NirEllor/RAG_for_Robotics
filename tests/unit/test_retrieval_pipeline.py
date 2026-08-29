from __future__ import annotations

from pathlib import Path

import numpy as np

from action_retrieval.retrieval.dataset import ExportedEpisode
from action_retrieval.retrieval.encoders import (
    GeometryOnlyEncoder,
    PoseDescriptorEncoder,
    RGBHistogramEncoder,
)
from action_retrieval.retrieval.pipeline import build_encoder, embed_episodes
from action_retrieval.retrieval.ranking import top_k_cosine


def _make_episode(episode_id: str, offset: float) -> ExportedEpisode:
    front_rgb = np.full((2, 4, 4, 3), 128 + int(offset), dtype=np.uint8)
    front_point_cloud = np.full((2, 4, 4, 3), offset, dtype=np.float32)
    joint_positions = np.full((2, 7), offset, dtype=np.float32)
    joint_velocities = np.full((2, 7), offset + 0.5, dtype=np.float32)
    gripper_pose = np.full((2, 7), offset + 1.0, dtype=np.float32)
    gripper_open = np.full((2,), offset + 2.0, dtype=np.float32)

    return ExportedEpisode(
        episode_id=episode_id,
        task_name="reach_target",
        split="train",
        observation_path=Path("/tmp/observation.npz"),
        trajectory_path=Path("/tmp/trajectory.npz"),
        metadata_path=Path("/tmp/metadata.json"),
        observation={
            "front_rgb": front_rgb,
            "front_point_cloud_world": front_point_cloud,
        },
        trajectory={
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "gripper_pose": gripper_pose,
            "gripper_open": gripper_open,
        },
        metadata={"episode": {"seed": 42}},
    )


def _make_color_episode(episode_id: str, rgb_value: tuple[int, int, int]) -> ExportedEpisode:
    front_rgb = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    front_rgb[:] = np.array(rgb_value, dtype=np.uint8)
    front_point_cloud = np.zeros((2, 8, 8, 3), dtype=np.float32)
    joint_positions = np.zeros((2, 7), dtype=np.float32)
    joint_velocities = np.zeros((2, 7), dtype=np.float32)
    gripper_pose = np.zeros((2, 7), dtype=np.float32)
    gripper_open = np.zeros((2,), dtype=np.float32)

    return ExportedEpisode(
        episode_id=episode_id,
        task_name="reach_target",
        split="train",
        observation_path=Path("/tmp/observation.npz"),
        trajectory_path=Path("/tmp/trajectory.npz"),
        metadata_path=Path("/tmp/metadata.json"),
        observation={
            "front_rgb": front_rgb,
            "front_point_cloud_world": front_point_cloud,
        },
        trajectory={
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "gripper_pose": gripper_pose,
            "gripper_open": gripper_open,
        },
        metadata={"episode": {"seed": 42}},
    )


def test_pose_descriptor_encoder_produces_unit_vector():
    episode = _make_episode("episode0", 0.0)
    vector = PoseDescriptorEncoder().encode(episode)

    assert vector.ndim == 1
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_top_k_cosine_excludes_query_episode():
    episodes = [_make_episode("episode0", 0.0), _make_episode("episode1", 1.0)]
    embeddings = embed_episodes(episodes, encoder_name="pose_descriptor")

    matches = top_k_cosine(embeddings[0], embeddings, k=1, exclude_query_episode=True)

    assert len(matches) == 1
    assert matches[0].candidate_episode_id == "episode1"


def test_color_sensitive_descriptor_prefers_matching_color():
    red_query = _make_color_episode("episode0", (255, 0, 0))
    green_candidate = _make_color_episode("episode1", (0, 255, 0))
    red_candidate = _make_color_episode("episode2", (255, 0, 0))

    embeddings = embed_episodes(
        [red_query, green_candidate, red_candidate],
        encoder_name="pose_descriptor",
    )

    matches = top_k_cosine(embeddings[0], embeddings, k=2, exclude_query_episode=True)

    assert matches[0].candidate_episode_id == "episode2"
    assert matches[0].score > matches[1].score


def test_rgb_histogram_encoder_prefers_matching_color():
    red_query = _make_color_episode("episode0", (255, 0, 0))
    green_candidate = _make_color_episode("episode1", (0, 255, 0))
    red_candidate = _make_color_episode("episode2", (255, 0, 0))

    embeddings = embed_episodes(
        [red_query, green_candidate, red_candidate],
        encoder_name="rgb_histogram",
    )

    matches = top_k_cosine(embeddings[0], embeddings, k=2, exclude_query_episode=True)

    assert matches[0].candidate_episode_id == "episode2"
    assert matches[0].score > matches[1].score


def test_global_color_encoder_prefers_matching_color():
    red_query = _make_color_episode("episode0", (255, 0, 0))
    green_candidate = _make_color_episode("episode1", (0, 255, 0))
    red_candidate = _make_color_episode("episode2", (255, 0, 0))

    embeddings = embed_episodes(
        [red_query, green_candidate, red_candidate],
        encoder_name="global_color",
    )

    matches = top_k_cosine(embeddings[0], embeddings, k=2, exclude_query_episode=True)

    assert matches[0].candidate_episode_id == "episode2"
    assert matches[0].score > matches[1].score


def test_geometry_only_encoder_prefers_matching_geometry():
    query = _make_episode("episode0", 0.0)
    same_geometry = _make_episode("episode1", 0.0)
    different_geometry = _make_episode("episode2", 2.0)

    embeddings = embed_episodes(
        [query, same_geometry, different_geometry],
        encoder_name="geometry_only",
    )

    matches = top_k_cosine(embeddings[0], embeddings, k=2, exclude_query_episode=True)

    assert matches[0].candidate_episode_id == "episode1"
    assert matches[0].score > matches[1].score


def test_geometry_only_encoder_produces_unit_vector():
    episode = _make_episode("episode0", 0.0)
    vector = GeometryOnlyEncoder().encode(episode)

    assert vector.ndim == 1
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_build_encoder_supports_uni3d_and_ptv3_backends():
    episode = _make_episode("episode0", 0.0)

    uni3d = build_encoder("uni3d")
    ptv3 = build_encoder("ptv3")

    uni3d_vector = uni3d.encode(episode)
    ptv3_vector = ptv3.encode(episode)

    assert uni3d_vector.ndim == 1
    assert ptv3_vector.ndim == 1
    assert np.isclose(np.linalg.norm(uni3d_vector), 1.0)
    assert np.isclose(np.linalg.norm(ptv3_vector), 1.0)
    assert uni3d_vector.shape == ptv3_vector.shape
    assert not np.allclose(uni3d_vector, ptv3_vector)
