"""Episode-level encoders used by the MVP retrieval pipeline."""

from __future__ import annotations

import hashlib
import inspect
import importlib.util
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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
                return _strip_state_dict_prefix(nested)
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return _strip_state_dict_prefix(checkpoint)
    raise TypeError(
        "Expected a checkpoint dictionary containing a state dict under one of the keys "
        "'module', 'state_dict', 'model', or 'ema'."
    )


def _strip_state_dict_prefix(state_dict: dict[str, Any], prefix: str = "module.") -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(isinstance(key, str) and key.startswith(prefix) for key in state_dict.keys()):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _remap_state_dict_prefixes(
    state_dict: dict[str, Any],
    prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Strip any leading prefixes from each key, in order, when present."""

    if not state_dict or not prefixes:
        return state_dict

    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            remapped[key] = value
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        remapped[new_key] = value
    return remapped


def _state_dict_alignment_summary(
    model_state_dict: dict[str, Any],
    checkpoint_state_dict: dict[str, Any],
) -> dict[str, Any]:
    model_keys = {key for key in model_state_dict.keys() if isinstance(key, str)}
    checkpoint_keys = {key for key in checkpoint_state_dict.keys() if isinstance(key, str)}
    shared_keys = sorted(model_keys & checkpoint_keys)
    shape_matches = []
    shape_mismatches = []
    for key in shared_keys:
        model_tensor = model_state_dict[key]
        checkpoint_tensor = checkpoint_state_dict[key]
        model_shape = tuple(getattr(model_tensor, "shape", ()))
        checkpoint_shape = tuple(getattr(checkpoint_tensor, "shape", ()))
        if model_shape == checkpoint_shape:
            shape_matches.append(key)
        else:
            shape_mismatches.append((key, model_shape, checkpoint_shape))
    return {
        "model_key_count": len(model_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "shared_key_count": len(shared_keys),
        "shape_match_count": len(shape_matches),
        "shape_mismatch_count": len(shape_mismatches),
        "shape_mismatches": shape_mismatches,
    }


def _best_state_dict_remap(
    model_state_dict: dict[str, Any],
    checkpoint_state_dict: dict[str, Any],
    *,
    candidate_prefix_sets: tuple[tuple[str, ...], ...] = (
        (),
        ("module.",),
        ("backbone.",),
        ("module.", "backbone."),
    ),
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    best_name = "raw"
    best_state_dict = checkpoint_state_dict
    best_summary = _state_dict_alignment_summary(model_state_dict, checkpoint_state_dict)
    best_score = (
        best_summary["shape_match_count"],
        best_summary["shared_key_count"],
    )

    for prefixes in candidate_prefix_sets:
        if not prefixes:
            continue
        candidate_name = "+".join(prefixes)
        candidate_state_dict = _remap_state_dict_prefixes(checkpoint_state_dict, prefixes)
        summary = _state_dict_alignment_summary(model_state_dict, candidate_state_dict)
        score = (
            summary["shape_match_count"],
            summary["shared_key_count"],
        )
        if score > best_score:
            best_name = candidate_name
            best_state_dict = candidate_state_dict
            best_summary = summary
            best_score = score

    return best_name, best_state_dict, best_summary


def _load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _git_command(repo_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    stdout = completed.stdout.strip()
    return stdout or None


def _repo_release_summary(repo_root: Path) -> dict[str, str | bool | None]:
    git_dir = repo_root / ".git"
    has_git = git_dir.exists()
    tag = _git_command(repo_root, "describe", "--tags", "--exact-match", "HEAD") if has_git else None
    commit = _git_command(repo_root, "rev-parse", "--short", "HEAD") if has_git else None
    dirty = _git_command(repo_root, "status", "--porcelain") if has_git else None
    return {
        "has_git": has_git,
        "tag": tag,
        "commit": commit,
        "dirty": bool(dirty),
    }


def _detect_ptv3_repo_layout(repo_root: Path) -> str:
    has_pointcept_pkg = (repo_root / "pointcept").is_dir()
    has_standalone_model = (repo_root / "model.py").is_file()
    if has_pointcept_pkg and not has_standalone_model:
        return "pointcept"
    if has_standalone_model and not has_pointcept_pkg:
        return "standalone"
    if has_pointcept_pkg and has_standalone_model:
        return "pointcept"
    return "unknown"


def _validate_ptv3_release_alignment(
    repo_root: Path,
    *,
    expected_tag: str | None,
    expected_commit: str | None,
    strict: bool,
) -> dict[str, str | bool | None]:
    summary = _repo_release_summary(repo_root)
    expected_tag = (expected_tag or "").strip() or None
    expected_commit = (expected_commit or "").strip().lower() or None

    observed_tag = summary["tag"]
    observed_commit = summary["commit"]

    tag_match = expected_tag is None or observed_tag == expected_tag
    commit_match = expected_commit is None or (
        observed_commit is not None and observed_commit.lower().startswith(expected_commit)
    )

    if not summary["has_git"]:
        message = (
            "PTv3 repo root is not a git checkout, so release alignment to "
            f"{expected_tag or expected_commit or 'the expected release'} cannot be verified."
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
        return summary

    if tag_match and commit_match:
        return summary

    message = (
        "PTv3 release alignment check failed. "
        f"Expected tag={expected_tag!r} or commit prefix={expected_commit!r}, "
        f"observed tag={observed_tag!r}, commit={observed_commit!r}, dirty={summary['dirty']}."
    )
    if strict:
        raise RuntimeError(message)
    warnings.warn(message)
    return summary


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 1:
        return tuple([int(parts[0])] * len(default))
    if len(parts) != len(default):
        raise ValueError(
            f"{name} must contain either 1 value or {len(default)} comma-separated values."
        )
    return tuple(int(part) for part in parts)


def _ensure_package_module(package_name: str, package_path: Path) -> ModuleType:
    module = sys.modules.get(package_name)
    if module is not None:
        return module  # type: ignore[return-value]
    module = ModuleType(package_name)
    module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = module
    return module


def _load_module_from_path(module_name: str, module_path: Path, package_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(module_path),
        submodule_search_locations=[str(package_path)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _offset2batch(offset: torch.Tensor) -> torch.Tensor:
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.inference_mode()
def _batch2offset(batch: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(batch.bincount(), dim=0).long()


def _infer_ptv3_num_classes(state_dict: dict[str, Any], default: int = 20) -> int:
    for key in ("seg_head.weight", "module.seg_head.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return default


def _pointcept_ptv3_backbone_config(*, in_channels: int) -> dict[str, Any]:
    """Construct the Pointcept v1.5.2 PTv3 backbone config used for loading weights."""

    return dict(
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    )


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
                f"Could not import the official Uni3D model code from {self.repo_root}: "
                f"{exc.__class__.__name__}: {exc}"
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
        checkpoint = _load_checkpoint(self.checkpoint)
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        remap_name, remapped_state_dict, summary = _best_state_dict_remap(model.state_dict(), state_dict)
        if remap_name != "raw":
            warnings.warn(
                "Uni3D checkpoint state dict remapped using "
                f"{remap_name}; shared={summary['shared_key_count']}, "
                f"shape_matches={summary['shape_match_count']}."
            )
        load_result = model.load_state_dict(remapped_state_dict, strict=False)
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
    """Point Transformer V3-style 3D foundation encoder with real-backend fallback.

    If ``PTV3_REPO_ROOT`` and ``PTV3_CHECKPOINT`` are set, this encoder loads
    the official PTv3 code from the cloned repository and runs a pretrained
    checkpoint. Otherwise it falls back to the deterministic proxy path that
    keeps the retrieval MVP usable.
    """

    name = "ptv3"

    def __init__(
        self,
        *,
        sample_count: int = 10000,
        output_dim: int = 768,
        repo_root: str | os.PathLike[str] | None = None,
        checkpoint: str | os.PathLike[str] | None = None,
        in_channels: int | None = None,
        grid_size: float | None = None,
        enable_flash: bool | None = None,
        enable_rpe: bool | None = None,
        cls_mode: bool | None = None,
        order: str | None = None,
        stride: str | None = None,
        enc_depths: str | None = None,
        enc_channels: str | None = None,
        enc_num_head: str | None = None,
        enc_patch_size: str | None = None,
        dec_depths: str | None = None,
        dec_channels: str | None = None,
        dec_num_head: str | None = None,
        dec_patch_size: str | None = None,
        drop_path: float | None = None,
        upcast_attention: bool | None = None,
        upcast_softmax: bool | None = None,
        device: str | None = None,
        use_real: bool | None = None,
        **_: Any,
    ):
        self.sample_count = sample_count
        self.output_dim = output_dim
        self.repo_root = _coerce_optional_path(repo_root or os.getenv("PTV3_REPO_ROOT"))
        self.checkpoint = _coerce_optional_path(checkpoint or os.getenv("PTV3_CHECKPOINT"))
        self.in_channels = in_channels if in_channels is not None else _env_int("PTV3_IN_CHANNELS", 6)
        self.grid_size = grid_size if grid_size is not None else float(os.getenv("PTV3_GRID_SIZE", "0.01"))
        self.enable_flash = enable_flash if enable_flash is not None else _env_bool("PTV3_ENABLE_FLASH", False)
        self.enable_rpe = enable_rpe if enable_rpe is not None else _env_bool("PTV3_ENABLE_RPE", False)
        self.cls_mode = cls_mode if cls_mode is not None else _env_bool("PTV3_CLS_MODE", False)
        self.order = tuple((order or os.getenv("PTV3_ORDER", "z,z-trans,hilbert,hilbert-trans")).split(","))
        self.stride = _env_int_tuple("PTV3_STRIDE", (2, 2, 2, 2)) if stride is None else tuple(int(v) for v in stride.split(","))
        self.enc_depths = (
            _env_int_tuple("PTV3_ENC_DEPTHS", (2, 2, 2, 6, 2))
            if enc_depths is None
            else tuple(int(v) for v in enc_depths.split(","))
        )
        self.enc_channels = (
            _env_int_tuple("PTV3_ENC_CHANNELS", (32, 64, 128, 256, 512))
            if enc_channels is None
            else tuple(int(v) for v in enc_channels.split(","))
        )
        self.enc_num_head = (
            _env_int_tuple("PTV3_ENC_NUM_HEAD", (2, 4, 8, 16, 32))
            if enc_num_head is None
            else tuple(int(v) for v in enc_num_head.split(","))
        )
        self.enc_patch_size = (
            _env_int_tuple("PTV3_ENC_PATCH_SIZE", (128, 128, 128, 128, 128))
            if enc_patch_size is None
            else tuple(int(v) for v in enc_patch_size.split(","))
        )
        self.dec_depths = (
            _env_int_tuple("PTV3_DEC_DEPTHS", (2, 2, 2, 2))
            if dec_depths is None
            else tuple(int(v) for v in dec_depths.split(","))
        )
        self.dec_channels = (
            _env_int_tuple("PTV3_DEC_CHANNELS", (64, 64, 128, 256))
            if dec_channels is None
            else tuple(int(v) for v in dec_channels.split(","))
        )
        self.dec_num_head = (
            _env_int_tuple("PTV3_DEC_NUM_HEAD", (4, 4, 8, 16))
            if dec_num_head is None
            else tuple(int(v) for v in dec_num_head.split(","))
        )
        self.dec_patch_size = (
            _env_int_tuple("PTV3_DEC_PATCH_SIZE", (128, 128, 128, 128))
            if dec_patch_size is None
            else tuple(int(v) for v in dec_patch_size.split(","))
        )
        self.drop_path = drop_path if drop_path is not None else _env_float("PTV3_DROP_PATH", 0.3)
        self.upcast_attention = (
            upcast_attention if upcast_attention is not None else _env_bool("PTV3_UPCAST_ATTENTION", False)
        )
        self.upcast_softmax = (
            upcast_softmax if upcast_softmax is not None else _env_bool("PTV3_UPCAST_SOFTMAX", False)
        )
        self.device = torch.device(device or os.getenv("PTV3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.allow_key_mismatch = _env_bool("PTV3_ALLOW_KEY_MISMATCH", False)
        self.expected_release_tag = (os.getenv("PTV3_EXPECTED_RELEASE_TAG") or "v1.5.2").strip()
        self.expected_release_commit = (os.getenv("PTV3_EXPECTED_RELEASE_COMMIT") or "ad653ee").strip()
        self.strict_release = _env_bool("PTV3_STRICT_RELEASE", False)
        self.repo_layout = (os.getenv("PTV3_REPO_LAYOUT") or "auto").strip().lower()
        env_use_real = os.getenv("PTV3_USE_REAL")
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
                "Set both PTV3_REPO_ROOT and PTV3_CHECKPOINT to enable the real PTv3 backend."
            )
        if not self.repo_root.exists():
            raise FileNotFoundError(f"PTv3 repo root does not exist: {self.repo_root}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"PTv3 checkpoint does not exist: {self.checkpoint}")

        _validate_ptv3_release_alignment(
            self.repo_root,
            expected_tag=self.expected_release_tag,
            expected_commit=self.expected_release_commit,
            strict=self.strict_release,
        )

        checkpoint = _load_checkpoint(self.checkpoint)
        state_dict = _extract_checkpoint_state_dict(checkpoint)
        layout = self.repo_layout
        if layout == "auto":
            layout = _detect_ptv3_repo_layout(self.repo_root)

        if layout == "pointcept":
            pointcept_pkg_root = self.repo_root / "pointcept"
            pointcept_models_root = pointcept_pkg_root / "models"
            pointcept_ptv3_root = pointcept_models_root / "point_transformer_v3"
            pointcept_utils_root = pointcept_models_root / "utils"
            try:
                _ensure_package_module("pointcept", pointcept_pkg_root)
                _ensure_package_module("pointcept.models", pointcept_models_root)
                _ensure_package_module("pointcept.models.point_transformer_v3", pointcept_ptv3_root)
                pointcept_utils_module = _ensure_package_module("pointcept.models.utils", pointcept_utils_root)
                pointcept_utils_utils_module = ModuleType("pointcept.models.utils.utils")
                pointcept_utils_utils_module.offset2batch = _offset2batch  # type: ignore[attr-defined]
                pointcept_utils_utils_module.batch2offset = _batch2offset  # type: ignore[attr-defined]
                pointcept_utils_utils_module.__all__ = ["offset2batch", "batch2offset"]  # type: ignore[attr-defined]
                sys.modules["pointcept.models.utils.utils"] = pointcept_utils_utils_module
                pointcept_utils_module.offset2batch = _offset2batch  # type: ignore[attr-defined]
                pointcept_utils_module.batch2offset = _batch2offset  # type: ignore[attr-defined]
                pointcept_utils_module.__all__ = ["offset2batch", "batch2offset"]  # type: ignore[attr-defined]

                modules_module_path = pointcept_models_root / "modules.py"
                if modules_module_path.exists():
                    _load_module_from_path(
                        "pointcept.models.modules",
                        modules_module_path,
                        pointcept_models_root,
                    )
                serialization_module_path = pointcept_utils_root / "serialization.py"
                if serialization_module_path.exists():
                    _load_module_from_path(
                        "pointcept.models.utils.serialization",
                        serialization_module_path,
                        pointcept_utils_root,
                    )
                ptv3_serialization_module_path = pointcept_ptv3_root / "serialization.py"
                if ptv3_serialization_module_path.exists():
                    _load_module_from_path(
                        "pointcept.models.point_transformer_v3.serialization",
                        ptv3_serialization_module_path,
                        pointcept_ptv3_root,
                    )
                ptv3_module_path = pointcept_ptv3_root / "point_transformer_v3m1_base.py"
                if ptv3_module_path.exists():
                    ptv3_module = _load_module_from_path(
                        "pointcept.models.point_transformer_v3.point_transformer_v3m1_base",
                        ptv3_module_path,
                        pointcept_ptv3_root,
                    )
                else:
                    raise FileNotFoundError(
                        f"Could not find Pointcept PTv3 backbone module at: {ptv3_module_path}"
                    )
                structure_module = _load_module_from_path(
                    "pointcept.models.utils.structure",
                    pointcept_utils_root / "structure.py",
                    pointcept_utils_root,
                )
                Point = getattr(structure_module, "Point")
                PointTransformerV3 = getattr(ptv3_module, "PointTransformerV3")
                pointcept_utils_module.Point = Point  # type: ignore[attr-defined]
            except Exception as exc:
                raise ImportError(
                    "Could not import Pointcept PTv3 runtime from the cloned repo. "
                    "Make sure PTV3_REPO_ROOT points to a Pointcept checkout that matches v1.5.2. "
                    f"Cause: {exc.__class__.__name__}: {exc}"
                ) from exc

            backbone_cfg = _pointcept_ptv3_backbone_config(
                in_channels=self.in_channels,
            )
            backbone_cfg["in_channels"] = self.in_channels
            backbone_cfg["enable_flash"] = self.enable_flash
            backbone_cfg["enable_rpe"] = self.enable_rpe
            backbone_cfg["upcast_attention"] = self.upcast_attention
            backbone_cfg["upcast_softmax"] = self.upcast_softmax
            accepted_backbone_keys = {
                name
                for name in inspect.signature(PointTransformerV3.__init__).parameters
                if name != "self"
            }
            ignored_backbone_keys = sorted(key for key in backbone_cfg if key not in accepted_backbone_keys)
            if ignored_backbone_keys:
                warnings.warn(
                    "Ignoring unsupported PTv3 backbone config keys: "
                    f"{ignored_backbone_keys}"
                )
            backbone_cfg = {key: value for key, value in backbone_cfg.items() if key in accepted_backbone_keys}
            backbone = PointTransformerV3(**backbone_cfg)

            remap_name, remapped_state_dict, summary = _best_state_dict_remap(backbone.state_dict(), state_dict)
            if remap_name != "raw":
                warnings.warn(
                    "PTv3 checkpoint state dict remapped using "
                    f"{remap_name}; shared={summary['shared_key_count']}, "
                    f"shape_matches={summary['shape_match_count']}."
                )
            load_result = backbone.load_state_dict(remapped_state_dict, strict=False)
            missing_keys = list(getattr(load_result, "missing_keys", []))
            unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
            unexpected_keys = [key for key in unexpected_keys if not key.startswith("seg_head.")]
            if missing_keys or unexpected_keys:
                message = (
                    "Loaded PTv3 checkpoint with a non-empty key mismatch. "
                    f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}."
                )
                if self.allow_key_mismatch:
                    warnings.warn(
                        message
                        + " The real backend is active, but you should verify the checkpoint matches the code."
                    )
                else:
                    raise RuntimeError(
                        message
                        + " Falling back to the proxy encoder by default. "
                        "Set PTV3_ALLOW_KEY_MISMATCH=1 if you want to try the real backend anyway."
                    )

            return backbone.eval().to(self.device)

        package_name = "_ptv3_runtime"
        _ensure_package_module(package_name, self.repo_root)

        model_module_path = self.repo_root / "model.py"
        if not model_module_path.exists():
            raise FileNotFoundError(f"Could not find PTv3 model.py at: {model_module_path}")

        module_name = f"{package_name}.model"
        module = _load_module_from_path(module_name, model_module_path, self.repo_root)
        try:
            PointTransformerV3 = getattr(module, "PointTransformerV3")
        except AttributeError as exc:
            raise ImportError(
                f"PTv3 model module at {model_module_path} does not expose PointTransformerV3: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

        model = PointTransformerV3(
            in_channels=self.in_channels,
            order=self.order,
            stride=self.stride,
            enc_depths=self.enc_depths,
            enc_channels=self.enc_channels,
            enc_num_head=self.enc_num_head,
            enc_patch_size=self.enc_patch_size,
            dec_depths=self.dec_depths,
            dec_channels=self.dec_channels,
            dec_num_head=self.dec_num_head,
            dec_patch_size=self.dec_patch_size,
            drop_path=self.drop_path,
            enable_rpe=self.enable_rpe,
            enable_flash=self.enable_flash,
            upcast_attention=self.upcast_attention,
            upcast_softmax=self.upcast_softmax,
            cls_mode=self.cls_mode,
        )

        remap_name, remapped_state_dict, summary = _best_state_dict_remap(model.state_dict(), state_dict)
        if remap_name != "raw":
            warnings.warn(
                "PTv3 checkpoint state dict remapped using "
                f"{remap_name}; shared={summary['shared_key_count']}, "
                f"shape_matches={summary['shape_match_count']}."
            )
        load_result = model.load_state_dict(remapped_state_dict, strict=False)
        missing_keys = getattr(load_result, "missing_keys", [])
        unexpected_keys = getattr(load_result, "unexpected_keys", [])
        if missing_keys or unexpected_keys:
            message = (
                "Loaded PTv3 checkpoint with a non-empty key mismatch. "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}."
            )
            if self.allow_key_mismatch:
                warnings.warn(
                    message
                    + " The real backend is active, but you should verify the checkpoint matches the code."
                )
            else:
                raise RuntimeError(
                    message
                    + " Falling back to the proxy encoder by default. "
                    "Set PTV3_ALLOW_KEY_MISMATCH=1 if you want to try the real backend anyway."
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
                "PTv3 real backend is unavailable, so the proxy encoder will be used instead. "
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
                    coord = torch.from_numpy(sampled[:, :3].astype(np.float32)).to(self.device)
                    feat = torch.from_numpy(sampled.astype(np.float32)).to(self.device)
                    batch = torch.zeros((sampled.shape[0],), dtype=torch.long, device=self.device)
                    data_dict = {
                        "coord": coord,
                        "feat": feat,
                        "batch": batch,
                        "grid_size": self.grid_size,
                    }
                    with torch.inference_mode():
                        output = model(data_dict)

                    if isinstance(output, dict):
                        feat_tensor = output.get("feat")
                    else:
                        feat_tensor = getattr(output, "feat", output)

                    if isinstance(feat_tensor, torch.Tensor):
                        if feat_tensor.ndim == 2:
                            vector = feat_tensor.mean(dim=0)
                        else:
                            vector = feat_tensor.reshape(-1)
                        vector_np = vector.detach().float().cpu().numpy().reshape(-1)
                        if vector_np.size > 0:
                            return _normalize(vector_np)

        vector = _build_point_cloud_backbone_embedding(
            episode,
            variant="ptv3",
            sample_count=self.sample_count,
        )
        if vector.size != self.output_dim:
            vector = np.resize(vector, (self.output_dim,)).astype(np.float32)
        return _normalize(vector)
