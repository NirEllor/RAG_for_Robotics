"""Episode-level encoders used by the MVP retrieval pipeline."""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

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


def _primary_xyzrgb_points(episode: ExportedEpisode) -> np.ndarray | None:
    observation = episode.observation
    point_cloud = None
    for key in ("front_point_cloud_world", "front_point_cloud_camera"):
        if observation.get(key) is not None:
            point_cloud = observation.get(key)
            break
    if point_cloud is None:
        return None

    points = _as_float32(point_cloud)
    if points.ndim == 4:
        points = points[0]
    points = points.reshape(-1, 3)

    front_rgb = observation.get("front_rgb")
    if front_rgb is None:
        colors = np.zeros_like(points, dtype=np.float32)
    else:
        rgb = _as_float32(front_rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        colors = rgb.reshape(-1, 3) / 255.0

    if colors.shape[0] != points.shape[0]:
        sample_count = min(points.shape[0], colors.shape[0])
        points = points[:sample_count]
        colors = colors[:sample_count]

    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[finite_mask]
    colors = colors[finite_mask]
    if points.size == 0:
        return None

    return np.concatenate([points, colors], axis=1).astype(np.float32)


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


def _sample_xyzrgb_points(xyzrgb: np.ndarray, sample_count: int) -> np.ndarray:
    if xyzrgb.size == 0:
        return np.zeros((sample_count, 6), dtype=np.float32)
    if xyzrgb.shape[0] >= sample_count:
        indices = np.linspace(0, xyzrgb.shape[0] - 1, sample_count, dtype=np.int64)
        return xyzrgb[indices].astype(np.float32)
    repeats = np.resize(xyzrgb, (sample_count, xyzrgb.shape[1]))
    return repeats.astype(np.float32)


def _coerce_optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return Path(text).expanduser()


def _extract_checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("module", "state_dict", "model", "ema"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint
    raise TypeError(
        "Expected a checkpoint dictionary containing a state dict under one of the keys "
        "'module', 'state_dict', 'model', or 'ema'."
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


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
    """Uni3D-style 3D foundation encoder with real-backend fallback support.

    If ``UNI3D_REPO_ROOT`` and ``UNI3D_CHECKPOINT`` are set, this class tries to
    load the official Uni3D implementation from the cloned repository and run a
    real pretrained checkpoint. If anything is missing, it falls back to the
    deterministic proxy encoder so the retrieval pipeline keeps working.
    """

    name = "uni3d"

    def __init__(
        self,
        *,
        sample_count: int = 10000,
        output_dim: int = 1024,
        repo_root: str | os.PathLike[str] | None = None,
        checkpoint: str | os.PathLike[str] | None = None,
        pc_model: str | None = None,
        pretrained_pc: str | os.PathLike[str] | None = None,
        pc_feat_dim: int | None = None,
        embed_dim: int | None = None,
        group_size: int | None = None,
        num_group: int | None = None,
        pc_encoder_dim: int | None = None,
        drop_path_rate: float | None = None,
        patch_dropout: float | None = None,
        device: str | None = None,
        use_real: bool | None = None,
        **_: Any,
    ):
        self.sample_count = sample_count
        self.output_dim = output_dim
        self.repo_root = _coerce_optional_path(repo_root or os.getenv("UNI3D_REPO_ROOT"))
        self.checkpoint = _coerce_optional_path(checkpoint or os.getenv("UNI3D_CHECKPOINT"))
        self.pc_model = pc_model or os.getenv(
            "UNI3D_PC_MODEL",
            "eva_giant_patch14_560.m30m_ft_in22k_in1k",
        )
        self.pretrained_pc = str(
            pretrained_pc or os.getenv("UNI3D_PRETRAINED_PC") or ""
        ).strip()
        self.pc_feat_dim = pc_feat_dim if pc_feat_dim is not None else _env_int("UNI3D_PC_FEAT_DIM", 1408)
        self.embed_dim = embed_dim if embed_dim is not None else _env_int("UNI3D_EMBED_DIM", output_dim)
        self.group_size = group_size if group_size is not None else _env_int("UNI3D_GROUP_SIZE", 64)
        self.num_group = num_group if num_group is not None else _env_int("UNI3D_NUM_GROUP", 512)
        self.pc_encoder_dim = (
            pc_encoder_dim if pc_encoder_dim is not None else _env_int("UNI3D_PC_ENCODER_DIM", 512)
        )
        self.drop_path_rate = (
            drop_path_rate if drop_path_rate is not None else _env_float("UNI3D_DROP_PATH_RATE", 0.0)
        )
        self.patch_dropout = (
            patch_dropout if patch_dropout is not None else _env_float("UNI3D_PATCH_DROPOUT", 0.0)
        )
        self.device = torch.device(device or os.getenv("UNI3D_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
        env_use_real = os.getenv("UNI3D_USE_REAL")
        if use_real is not None:
            self.use_real = use_real
        elif env_use_real is not None and env_use_real.strip():
            self.use_real = env_use_real.strip().lower() in {"1", "true", "yes", "on"}
        else:
            self.use_real = self.repo_root is not None and self.checkpoint is not None
        self._real_model: Any | None = None
        self._real_backend_attempted = False
        self._real_backend_failed = False

    def _build_real_backend(self) -> Any:
        if self.repo_root is None or self.checkpoint is None:
            raise FileNotFoundError(
                "Set both UNI3D_REPO_ROOT and UNI3D_CHECKPOINT to enable the real Uni3D backend."
            )
        if not self.repo_root.exists():
            raise FileNotFoundError(f"UNI3D repo root does not exist: {self.repo_root}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"UNI3D checkpoint does not exist: {self.checkpoint}")

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        try:
            from models.uni3d import create_uni3d  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on cluster install
            raise ImportError(
                f"Could not import the official Uni3D model code from {self.repo_root}"
            ) from exc

        args = SimpleNamespace(
            pc_model=self.pc_model,
            pretrained_pc=self.pretrained_pc,
            drop_path_rate=self.drop_path_rate,
            pc_feat_dim=self.pc_feat_dim,
            embed_dim=self.embed_dim,
            group_size=self.group_size,
            num_group=self.num_group,
            pc_encoder_dim=self.pc_encoder_dim,
            patch_dropout=self.patch_dropout,
        )

        model = create_uni3d(args)
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        load_result = model.load_state_dict(state_dict, strict=False)
        missing_keys = getattr(load_result, "missing_keys", [])
        unexpected_keys = getattr(load_result, "unexpected_keys", [])
        if missing_keys or unexpected_keys:
            warnings.warn(
                "Loaded Uni3D checkpoint with a non-empty key mismatch. "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}. "
                "The real backend is active, but you should verify the checkpoint matches the code."
            )
        model.eval()
        model.to(self.device)
        return model

    def _get_real_backend(self) -> Any | None:
        if self._real_backend_failed:
            return None
        if self._real_model is not None:
            return self._real_model
        if self._real_backend_attempted:
            return None
        self._real_backend_attempted = True
        try:
            self._real_model = self._build_real_backend()
        except Exception as exc:
            self._real_backend_failed = True
            warnings.warn(
                "Uni3D real backend is unavailable, so the proxy encoder will be used instead. "
                f"Reason: {exc}"
            )
            self._real_model = None
        return self._real_model

    def encode(self, episode: ExportedEpisode) -> np.ndarray:
        if self.use_real:
            model = self._get_real_backend()
            if model is not None:
                xyzrgb = _primary_xyzrgb_points(episode)
                if xyzrgb is not None:
                    sampled = _sample_xyzrgb_points(xyzrgb, self.sample_count)
                    pc = torch.from_numpy(sampled).unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        embedding = model.encode_pc(pc)
                    if isinstance(embedding, (tuple, list)):
                        embedding = embedding[0]
                    if embedding.ndim > 1:
                        embedding = embedding.squeeze(0)
                    vector = embedding.detach().float().cpu().numpy().reshape(-1)
                    if vector.size > 0:
                        return _normalize(vector)

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
