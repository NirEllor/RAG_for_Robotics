"""Episode-level encoders used by the MVP retrieval pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from action_retrieval.retrieval.dataset import ExportedEpisode


def _as_float32(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _safe_stats(array: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = _as_float32(array)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
    return flat.mean(axis=0), flat.std(axis=0)


def _color_histogram(rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    flat = _as_float32(rgb).reshape(-1, 3)
    flat = np.clip(flat, 0.0, 1.0)
    hist_parts: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(flat[:, channel], bins=bins, range=(0.0, 1.0), density=True)
        hist_parts.append(hist.astype(np.float32))
    return np.concatenate(hist_parts, axis=0)


def _saturated_pixels(rgb: np.ndarray, saturation_threshold: float = 0.12) -> np.ndarray:
    flat = _as_float32(rgb).reshape(-1, 3)
    flat = np.clip(flat, 0.0, 1.0)
    chroma = flat.max(axis=1) - flat.min(axis=1)
    saturated = flat[chroma >= saturation_threshold]
    if saturated.size == 0:
        return flat
    return saturated


def _center_crop(rgb: np.ndarray, crop_fraction: float = 0.5) -> np.ndarray:
    image = _as_float32(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        return image
    height, width = image.shape[:2]
    crop_h = max(1, int(round(height * crop_fraction)))
    crop_w = max(1, int(round(width * crop_fraction)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    return image[top : top + crop_h, left : left + crop_w]


def _rgb_summary_features(rgb: np.ndarray) -> list[np.ndarray]:
    """Return compact global color statistics for one RGB frame."""

    mean = rgb.mean(axis=(0, 1))
    std = rgb.std(axis=(0, 1))
    hist = _color_histogram(rgb)
    return [mean, std, hist]


def _rgb_local_features(rgb: np.ndarray) -> list[np.ndarray]:
    """Return localized color statistics that emphasize salient objects."""

    crop = _center_crop(rgb, crop_fraction=0.5)
    saturated = _saturated_pixels(rgb)
    crop_sat = _saturated_pixels(crop)
    return [
        crop.mean(axis=(0, 1)),
        crop.std(axis=(0, 1)),
        _color_histogram(crop),
        saturated.mean(axis=0),
        saturated.std(axis=0),
        _color_histogram(saturated),
        crop_sat.mean(axis=0),
        crop_sat.std(axis=0),
        _color_histogram(crop_sat),
    ]


def _pad_or_trim(array: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32).reshape(-1)
    if arr.size >= length:
        return arr[:length]
    padded = np.zeros((length,), dtype=np.float32)
    padded[: arr.size] = arr
    return padded


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _primary_point_cloud_points(episode: ExportedEpisode) -> np.ndarray | None:
    observation = episode.observation
    for key in ("front_point_cloud_world", "front_point_cloud_camera"):
        point_cloud = observation.get(key)
        if point_cloud is None:
            continue
        points = _as_float32(point_cloud)
        if points.ndim == 4:
            points = points[0]
        points = points.reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.size > 0:
            return points
    return None


def _normalize_point_cloud(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    scale = float(np.linalg.norm(centered, axis=1).max(initial=1.0))
    if scale <= 0.0:
        scale = 1.0
    normalized = centered / scale
    return centroid.astype(np.float32), normalized.astype(np.float32), scale


def _ordered_point_indices(normalized_points: np.ndarray, *, mode: str) -> np.ndarray:
    if normalized_points.size == 0:
        return np.empty((0,), dtype=np.int64)

    if mode == "radial":
        radial = np.linalg.norm(normalized_points, axis=1)
        return np.lexsort(
            (
                normalized_points[:, 2],
                normalized_points[:, 1],
                normalized_points[:, 0],
                radial,
            )
        )

    if mode == "voxel":
        voxel_grid = 16
        clipped = np.clip((normalized_points + 1.0) * 0.5, 0.0, 1.0)
        quantized = np.floor(clipped * (voxel_grid - 1)).astype(np.int32)
        morton_like = (
            quantized[:, 0] * (voxel_grid**2)
            + quantized[:, 1] * voxel_grid
            + quantized[:, 2]
        )
        return np.lexsort(
            (
                normalized_points[:, 2],
                normalized_points[:, 1],
                normalized_points[:, 0],
                morton_like,
            )
        )

    raise ValueError(f"Unsupported point order mode: {mode}")


def _sample_ordered_points(points: np.ndarray, *, sample_count: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros((sample_count, 3), dtype=np.float32)

    if points.shape[0] >= sample_count:
        sample_indices = np.linspace(0, points.shape[0] - 1, sample_count, dtype=np.int64)
        return points[sample_indices]

    repeats = np.resize(points, (sample_count, 3))
    return repeats.astype(np.float32)


def _build_point_cloud_backbone_embedding(
    episode: ExportedEpisode,
    *,
    variant: str,
    sample_count: int = 128,
) -> np.ndarray:
    points = _primary_point_cloud_points(episode)
    if points is None:
        return np.zeros((sample_count * 6,), dtype=np.float32)

    centroid, normalized_points, _ = _normalize_point_cloud(points)
    order_mode = "radial" if variant == "uni3d" else "voxel"
    ordered_indices = _ordered_point_indices(normalized_points, mode=order_mode)
    ordered_points = points[ordered_indices]
    ordered_normalized = normalized_points[ordered_indices]

    sampled_raw = _sample_ordered_points(ordered_points, sample_count=sample_count)
    sampled_normalized = _sample_ordered_points(ordered_normalized, sample_count=sample_count)

    if variant == "uni3d":
        # Mimic an object-centric foundation backbone: preserve absolute world
        # position in the first half and normalized geometry in the second half.
        features = np.concatenate(
            [
                sampled_raw.reshape(-1),
                sampled_normalized.reshape(-1),
            ],
            axis=0,
        )
    elif variant == "ptv3":
        # Mimic a voxel-aware backbone: keep normalized geometry and a coarse
        # quantized version that emphasizes local spatial layout.
        voxel_grid = 16
        clipped = np.clip((sampled_normalized + 1.0) * 0.5, 0.0, 1.0)
        quantized = np.floor(clipped * (voxel_grid - 1)).astype(np.float32)
        voxelized = quantized / float(voxel_grid - 1)
        voxelized = voxelized * 2.0 - 1.0
        features = np.concatenate(
            [
                sampled_normalized.reshape(-1),
                voxelized.reshape(-1),
            ],
            axis=0,
        )
    else:
        raise ValueError(f"Unsupported foundation variant: {variant}")

    # Inject a tiny amount of context so the vector is not purely shape-only.
    features = np.concatenate(
        [
            features,
            centroid.astype(np.float32),
            np.array([float(points.shape[0])], dtype=np.float32),
        ],
        axis=0,
    )
    return _normalize(features)


@dataclass(frozen=True)
class EpisodeEmbedding:
    episode_id: str
    task_name: str
    split: str
    vector: np.ndarray
    encoder_name: str


class PoseDescriptorEncoder:
    """Hand-crafted episode descriptor for the Phase 2 MVP.

    This is not a learned model. It summarizes the episode using initial/final
    low-dimensional state plus simple visual/geometry statistics.
    """

    name = "pose_descriptor"

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        observation = episode.observation
        trajectory = episode.trajectory

        front_rgb = observation.get("front_rgb")
        point_cloud = observation.get("front_point_cloud_world")
        joint_positions = trajectory.get("joint_positions")
        joint_velocities = trajectory.get("joint_velocities")
        gripper_pose = trajectory.get("gripper_pose")
        gripper_open = trajectory.get("gripper_open")

        features: list[np.ndarray] = []

        if front_rgb is not None:
            rgb_first = _as_float32(front_rgb[0]) / 255.0
            rgb_last = _as_float32(front_rgb[-1]) / 255.0
            features.extend(
                [
                    *_rgb_summary_features(rgb_first),
                    *_rgb_local_features(rgb_first),
                    *_rgb_summary_features(rgb_last),
                    *_rgb_local_features(rgb_last),
                ]
            )
        else:
            features.extend(
                [
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                ]
            )

        if point_cloud is not None:
            pc_first = _as_float32(point_cloud[0]).reshape(-1, 3)
            pc_first = pc_first[np.isfinite(pc_first).all(axis=1)]
            if pc_first.size > 0:
                features.append(pc_first.mean(axis=0))
                features.append(pc_first.std(axis=0))
                features.append(pc_first.min(axis=0))
                features.append(pc_first.max(axis=0))
            else:
                features.extend([np.zeros((3,), dtype=np.float32) for _ in range(4)])
        else:
            features.extend([np.zeros((3,), dtype=np.float32) for _ in range(4)])

        if joint_positions is not None:
            features.append(_pad_or_trim(joint_positions[0], 14))
            features.append(_pad_or_trim(joint_positions[-1], 14))
        else:
            features.extend([np.zeros((14,), dtype=np.float32) for _ in range(2)])

        if joint_velocities is not None:
            features.append(_pad_or_trim(joint_velocities[0], 14))
            features.append(_pad_or_trim(joint_velocities[-1], 14))
        else:
            features.extend([np.zeros((14,), dtype=np.float32) for _ in range(2)])

        if gripper_pose is not None:
            features.append(_pad_or_trim(gripper_pose[0], 7))
            features.append(_pad_or_trim(gripper_pose[-1], 7))
        else:
            features.extend([np.zeros((7,), dtype=np.float32) for _ in range(2)])

        if gripper_open is not None:
            features.append(np.array([float(gripper_open[0])], dtype=np.float32))
            features.append(np.array([float(gripper_open[-1])], dtype=np.float32))
        else:
            features.extend([np.zeros((1,), dtype=np.float32) for _ in range(2)])

        episode_seed = episode.metadata.get("episode", {}).get("seed", 0)
        features.append(np.array([float(episode_seed)], dtype=np.float32))
        features.append(np.array([float(len(joint_positions) if joint_positions is not None else 0)], dtype=np.float32))

        return _normalize(np.concatenate(features, axis=0))


class RGBHistogramEncoder:
    """Appearance-only baseline based on RGB statistics.

    This encoder intentionally ignores point clouds and robot state so we can
    compare a pure visual baseline against the geometry-heavy pose descriptor.
    """

    name = "rgb_histogram"

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        front_rgb = episode.observation.get("front_rgb")
        features: list[np.ndarray] = []

        if front_rgb is not None:
            rgb_first = _as_float32(front_rgb[0]) / 255.0
            rgb_last = _as_float32(front_rgb[-1]) / 255.0
            features.extend(
                [
                    *_rgb_summary_features(rgb_first),
                    *_rgb_local_features(rgb_first),
                    *_rgb_summary_features(rgb_last),
                    *_rgb_local_features(rgb_last),
                ]
            )
        else:
            features.extend(
                [
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                ]
            )

        return _normalize(np.concatenate(features, axis=0))


class GlobalColorEncoder:
    """Color-aware baseline that emphasizes global scene color cues.

    This encoder keeps the retrieval signal intentionally simple: it focuses on
    global RGB statistics plus a small temporal summary between the first and
    last frames. That makes it a useful baseline for episodes where color is a
    differentiating factor.
    """

    name = "global_color"

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        front_rgb = episode.observation.get("front_rgb")
        features: list[np.ndarray] = []

        if front_rgb is not None:
            rgb_first = _as_float32(front_rgb[0]) / 255.0
            rgb_last = _as_float32(front_rgb[-1]) / 255.0

            first_mean, first_std = rgb_first.mean(axis=(0, 1)), rgb_first.std(axis=(0, 1))
            last_mean, last_std = rgb_last.mean(axis=(0, 1)), rgb_last.std(axis=(0, 1))
            mean_delta = last_mean - first_mean
            std_delta = last_std - first_std

            features.extend(
                [
                    first_mean,
                    first_std,
                    last_mean,
                    last_std,
                    mean_delta,
                    std_delta,
                    _color_histogram(rgb_first),
                    _color_histogram(rgb_last),
                ]
            )
        else:
            features.extend(
                [
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((3,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                    np.zeros((24,), dtype=np.float32),
                ]
            )

        return _normalize(np.concatenate(features, axis=0))


class GeometryOnlyEncoder:
    """Geometry-only baseline built from 3D state and robot kinematics.

    This encoder ignores RGB entirely so we can compare a purely geometric
    retrieval signal against the mixed pose descriptor and the appearance-only
    RGB histogram baseline.
    """

    name = "geometry_only"

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        observation = episode.observation
        trajectory = episode.trajectory

        point_cloud = observation.get("front_point_cloud_world")
        joint_positions = trajectory.get("joint_positions")
        joint_velocities = trajectory.get("joint_velocities")
        gripper_pose = trajectory.get("gripper_pose")
        gripper_open = trajectory.get("gripper_open")

        features: list[np.ndarray] = []

        if point_cloud is not None:
            pc_first = _as_float32(point_cloud[0]).reshape(-1, 3)
            pc_first = pc_first[np.isfinite(pc_first).all(axis=1)]
            if pc_first.size > 0:
                features.append(pc_first.mean(axis=0))
                features.append(pc_first.std(axis=0))
                features.append(pc_first.min(axis=0))
                features.append(pc_first.max(axis=0))
            else:
                features.extend([np.zeros((3,), dtype=np.float32) for _ in range(4)])
        else:
            features.extend([np.zeros((3,), dtype=np.float32) for _ in range(4)])

        if joint_positions is not None:
            features.append(_pad_or_trim(joint_positions[0], 14))
            features.append(_pad_or_trim(joint_positions[-1], 14))
        else:
            features.extend([np.zeros((14,), dtype=np.float32) for _ in range(2)])

        if joint_velocities is not None:
            features.append(_pad_or_trim(joint_velocities[0], 14))
            features.append(_pad_or_trim(joint_velocities[-1], 14))
        else:
            features.extend([np.zeros((14,), dtype=np.float32) for _ in range(2)])

        if gripper_pose is not None:
            features.append(_pad_or_trim(gripper_pose[0], 7))
            features.append(_pad_or_trim(gripper_pose[-1], 7))
        else:
            features.extend([np.zeros((7,), dtype=np.float32) for _ in range(2)])

        if gripper_open is not None:
            features.append(np.array([float(gripper_open[0])], dtype=np.float32))
            features.append(np.array([float(gripper_open[-1])], dtype=np.float32))
        else:
            features.extend([np.zeros((1,), dtype=np.float32) for _ in range(2)])

        episode_seed = episode.metadata.get("episode", {}).get("seed", 0)
        features.append(np.array([float(episode_seed)], dtype=np.float32))
        features.append(np.array([float(len(joint_positions) if joint_positions is not None else 0)], dtype=np.float32))

        return _normalize(np.concatenate(features, axis=0))


class RandomEpisodeEncoder:
    """Deterministic random baseline for retrieval sanity checks."""

    name = "random"

    def __init__(self, output_dim: int = 512, seed: int = 42):
        self.output_dim = output_dim
        self.seed = seed

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        payload = f"{self.seed}:{episode.task_name}:{episode.episode_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        local_seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(local_seed)
        vector = rng.normal(size=(self.output_dim,)).astype(np.float32)
        return _normalize(vector)


class Uni3DEncoder:
    """CPU-friendly Uni3D-style 3D foundation encoder proxy.

    The repository does not vendor a real Uni3D checkpoint. This backend is a
    deterministic, cluster-ready point-cloud encoder that matches the intended
    interface and can later be swapped for a real Uni3D loader without changing
    downstream retrieval code.
    """

    name = "uni3d"

    def __init__(self, *, sample_count: int = 128, output_dim: int = 768, **_: Any):
        self.sample_count = sample_count
        self.output_dim = output_dim

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        vector = _build_point_cloud_backbone_embedding(
            episode,
            variant="uni3d",
            sample_count=self.sample_count,
        )
        if vector.size != self.output_dim:
            vector = np.resize(vector, (self.output_dim,)).astype(np.float32)
        return _normalize(vector)


class PointTransformerV3Encoder:
    """CPU-friendly Point Transformer V3-style 3D foundation encoder proxy.

    The point-cloud tokenization mirrors a learned point transformer interface:
    a deterministic 3D backbone that can later be replaced with an actual PTv3
    checkpoint while keeping the retrieval API unchanged.
    """

    name = "ptv3"

    def __init__(self, *, sample_count: int = 128, output_dim: int = 768, **_: Any):
        self.sample_count = sample_count
        self.output_dim = output_dim

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        vector = _build_point_cloud_backbone_embedding(
            episode,
            variant="ptv3",
            sample_count=self.sample_count,
        )
        if vector.size != self.output_dim:
            vector = np.resize(vector, (self.output_dim,)).astype(np.float32)
        return _normalize(vector)
