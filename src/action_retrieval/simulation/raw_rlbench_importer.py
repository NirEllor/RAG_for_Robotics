"""Utilities for loading raw RLBench demonstration directories.

This adapter targets PerAct-style or mirrored RLBench datasets where each
task lives under a split directory such as:

  <root>/train/<task>/all_variations/episodes/episode<N>
  <root>/val/<task>/all_variations/episodes/episode<N>

It loads low-dim observations plus front-camera RGB/depth and derives front
point clouds so the existing exporter can consume the result.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from rlbench.backend.const import DEPTH_SCALE, EPISODE_FOLDER, EPISODES_FOLDER, LOW_DIM_PICKLE


def _project_root() -> Path:
    """Resolve the relevant project path."""
    return Path(__file__).resolve().parents[3]


def candidate_raw_dataset_roots() -> list[Path]:
    """Implement the candidate_raw_dataset_roots operation used by this module."""
    roots: list[Path] = []
    roots.append(_project_root() / "data" / "rlbench" / "raw")
    roots.append(_project_root() / "data" / "rlbench_raw")
    roots.append(_project_root() / "data" / "processed" / "rlbench_raw")
    return roots


def _episode_dir_sort_key(path: Path) -> tuple[int, str]:
    """Implement the _episode_dir_sort_key operation used by this module."""
    name = path.name
    if name.startswith("episode"):
        suffix = name[len("episode") :]
        if suffix.isdigit():
            return int(suffix), name
    return 10**9, name


def _episode_dirs_from_root(root: Path) -> list[Path]:
    """Implement the _episode_dirs_from_root operation used by this module."""
    if not root.exists():
        return []
    if (root / LOW_DIM_PICKLE).exists():
        return [root]
    if (root / EPISODES_FOLDER).exists():
        return [
            entry
            for entry in sorted((root / EPISODES_FOLDER).iterdir(), key=_episode_dir_sort_key)
            if entry.is_dir() and (entry / LOW_DIM_PICKLE).exists()
        ]
    return [
        entry
        for entry in sorted(root.iterdir(), key=_episode_dir_sort_key)
        if entry.is_dir() and (entry / LOW_DIM_PICKLE).exists()
    ]


def resolve_raw_task_root(
    task_name: str,
    search_roots: Optional[Iterable[Path]] = None,
    split_name: str | None = "train",
) -> Optional[Path]:
    """Resolve the directory that contains a raw RLBench task."""

    search_candidates = list(search_roots) if search_roots is not None else candidate_raw_dataset_roots()
    for base in search_candidates:
        base = Path(base)
        if not base.exists():
            continue
        candidate_paths = []
        if split_name:
            candidate_paths.extend(
                [
                    base / split_name / task_name / "all_variations",
                    base / split_name / task_name / "variation0",
                    base / split_name / task_name,
                ]
            )
        candidate_paths.extend(
            [
                base / task_name / "all_variations",
                base / task_name / "variation0",
                base / task_name,
            ]
        )
        for candidate in candidate_paths:
            if candidate.exists():
                if (candidate / EPISODES_FOLDER).exists():
                    return candidate
                if (candidate / "episodes").exists():
                    return candidate
                if any((candidate / sub).exists() for sub in ("front_rgb", "front", "low_dim_obs.pkl")):
                    return candidate
    return None


def _find_camera_folder(episode_dir: Path, camera_name: str, kind: str) -> Path | None:
    """Implement the _find_camera_folder operation used by this module."""
    candidates = [
        episode_dir / f"{camera_name}_{kind}",
        episode_dir / camera_name / kind,
        episode_dir / camera_name / f"{kind}s",
        episode_dir / camera_name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _load_image_series(folder: Path) -> list[Path]:
    """Load the requested data or model artifact."""
    return [entry for entry in sorted(folder.iterdir()) if entry.is_file() and entry.suffix.lower() == ".png"]


def _load_observations(
    episode_dir: Path,
    image_paths: bool,
) -> list[SimpleNamespace]:
    """Load the requested data or model artifact."""
    from pyrep.objects import VisionSensor
    from rlbench.backend.utils import image_to_float_array

    with (episode_dir / LOW_DIM_PICKLE).open("rb") as handle:
        observations = pickle.load(handle)

    if not isinstance(observations, list):
        observations = list(observations)

    front_rgb_folder = _find_camera_folder(episode_dir, "front", "rgb")
    front_depth_folder = _find_camera_folder(episode_dir, "front", "depth")
    if front_rgb_folder is None:
        front_rgb_folder = episode_dir / "front_rgb"
    if front_depth_folder is None:
        front_depth_folder = episode_dir / "front_depth"

    front_rgb_files = _load_image_series(front_rgb_folder) if front_rgb_folder.exists() else []
    front_depth_files = _load_image_series(front_depth_folder) if front_depth_folder.exists() else []

    num_steps = len(observations)
    if front_rgb_files and len(front_rgb_files) < num_steps:
        num_steps = len(front_rgb_files)
    if front_depth_files and len(front_depth_files) < num_steps:
        num_steps = len(front_depth_files)

    for index in range(num_steps):
        obs = observations[index]
        if not hasattr(obs, "__dict__"):
            obs = SimpleNamespace(**dict(obs))
            observations[index] = obs

        rgb_path = front_rgb_files[index] if front_rgb_files else None
        depth_path = front_depth_files[index] if front_depth_files else None
        if rgb_path is not None:
            obs.front_rgb = str(rgb_path) if image_paths else np.array(Image.open(rgb_path))
        if depth_path is not None:
            obs.front_depth = str(depth_path) if image_paths else image_to_float_array(
                Image.open(depth_path),
                DEPTH_SCALE,
            )
        if depth_path is not None and not image_paths:
            intrinsics = np.asarray((obs.misc or {}).get("front_camera_intrinsics"), dtype=np.float32)
            extrinsics = np.asarray((obs.misc or {}).get("front_camera_extrinsics"), dtype=np.float32)
            depth = np.asarray(obs.front_depth, dtype=np.float32)
            obs.front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
                depth,
                extrinsics,
                intrinsics,
            )

    return observations[:num_steps]


def load_raw_rlbench_task_demo_batch(
    task_name: str,
    amount: int = 1,
    image_paths: bool = False,
    split_name: str | None = "train",
    from_episode_number: int = 0,
    search_roots: Optional[Iterable[Path]] = None,
) -> tuple[list[list[SimpleNamespace]], Path, list[Path]]:
    """Load raw RLBench demos from an extracted dataset tree."""

    raw_task_root = resolve_raw_task_root(task_name, search_roots=search_roots, split_name=split_name)
    if raw_task_root is None:
        raise FileNotFoundError(
            f"Could not resolve a raw RLBench task root for {task_name}. "
            "Expected a PerAct-style split directory or extracted RLBench mirror."
        )

    episodes_root = raw_task_root / EPISODES_FOLDER
    if not episodes_root.exists():
        episodes_root = raw_task_root

    episode_dirs = _episode_dirs_from_root(episodes_root)
    if from_episode_number >= len(episode_dirs):
        raise RuntimeError(
            f"No raw episodes available from_episode_number={from_episode_number} under {episodes_root}"
        )

    available_count = len(episode_dirs) - from_episode_number
    if amount > available_count:
        print(
            f"WARNING: Requested {amount} raw demos but only {available_count} are available "
            f"from episode {from_episode_number}; exporting {available_count}."
        )
        amount = available_count

    selected_dirs = episode_dirs[from_episode_number : from_episode_number + amount]
    demos = [_load_observations(episode_dir, image_paths=image_paths) for episode_dir in selected_dirs]
    return demos, raw_task_root, selected_dirs
