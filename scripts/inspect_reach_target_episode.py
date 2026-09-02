#!/usr/bin/env python
"""Visually inspect one exported ReachTarget episode.

This script helps validate a single episode after Phase 1 export by saving:
- the first and last RGB frames,
- a subsampled 3D point-cloud scatter plot,
- a simple trajectory plot from the low-dimensional state.

It is intentionally lightweight so it can run in the WSL2 environment without
launching CoppeliaSim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "v1_reach_target",
        help="Path to the exported ReachTarget dataset root.",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Episode id to inspect, e.g. episode0. Defaults to the first manifest row.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "inspection",
        help="Where to save visual inspection artifacts.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figures interactively if a display is available.",
    )
    return parser.parse_args()


def _load_manifest(dataset_root: Path) -> pd.DataFrame:
    """Load the requested data or model artifact."""
    manifest_path = dataset_root / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return pd.read_parquet(manifest_path)


def _load_episode(dataset_root: Path, row: pd.Series) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """Load the requested data or model artifact."""
    observation_path = dataset_root / str(row["observation_path"])
    trajectory_path = dataset_root / str(row["trajectory_path"])
    metadata_path = dataset_root / str(row["metadata_path"])

    with np.load(observation_path) as observation_npz:
        observations = {key: observation_npz[key] for key in observation_npz.files}
    with np.load(trajectory_path) as trajectory_npz:
        trajectory = {key: trajectory_npz[key] for key in trajectory_npz.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return observations, trajectory, metadata


def _plot_rgb_frames(observations: dict[str, np.ndarray], output_dir: Path, episode_id: str) -> None:
    """Implement the _plot_rgb_frames operation used by this module."""
    rgb = observations.get("front_rgb")
    if rgb is None:
        return

    first = rgb[0]
    last = rgb[-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(first)
    axes[0].set_title("front_rgb[0]")
    axes[0].axis("off")
    axes[1].imshow(last)
    axes[1].set_title("front_rgb[-1]")
    axes[1].axis("off")
    fig.suptitle(f"Episode {episode_id}: RGB frames")
    fig.tight_layout()
    fig.savefig(output_dir / f"{episode_id}_rgb_frames.png", dpi=160)
    plt.close(fig)


def _plot_point_cloud(observations: dict[str, np.ndarray], output_dir: Path, episode_id: str) -> None:
    """Implement the _plot_point_cloud operation used by this module."""
    point_cloud = observations.get("front_point_cloud_world")
    if point_cloud is None:
        return

    points = np.asarray(point_cloud[0], dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return

    if len(points) > 4000:
        rng = np.random.default_rng(42)
        points = points[rng.choice(len(points), size=4000, replace=False)]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.7)
    ax.set_title(f"Episode {episode_id}: front point cloud (world frame)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(output_dir / f"{episode_id}_point_cloud.png", dpi=180)
    plt.close(fig)


def _plot_trajectory(trajectory: dict[str, np.ndarray], output_dir: Path, episode_id: str) -> None:
    """Implement the _plot_trajectory operation used by this module."""
    fig, ax = plt.subplots(figsize=(8, 4))

    if "joint_positions" in trajectory:
        joint_positions = np.asarray(trajectory["joint_positions"], dtype=np.float32)
        ax.plot(joint_positions)
        ax.set_title(f"Episode {episode_id}: joint positions over time")
        ax.set_xlabel("t")
        ax.set_ylabel("joint position")
    elif "gripper_pose" in trajectory:
        gripper_pose = np.asarray(trajectory["gripper_pose"], dtype=np.float32)
        if gripper_pose.ndim >= 2 and gripper_pose.shape[-1] >= 3:
            ax.plot(gripper_pose[:, :3])
            ax.set_title(f"Episode {episode_id}: gripper pose translation")
            ax.set_xlabel("t")
            ax.set_ylabel("position")
    else:
        ax.text(0.5, 0.5, "No trajectory field found", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / f"{episode_id}_trajectory.png", dpi=180)
    plt.close(fig)


def main() -> int:
    """Run the command-line entry point."""
    args = _parse_args()
    dataset_root = args.dataset_root
    manifest = _load_manifest(dataset_root)

    if args.episode_id is None:
        row = manifest.iloc[0]
    else:
        matches = manifest[manifest["episode_id"].astype(str) == args.episode_id]
        if matches.empty:
            raise ValueError(f"Episode id not found in manifest: {args.episode_id}")
        row = matches.iloc[0]

    observations, trajectory, metadata = _load_episode(dataset_root, row)

    output_dir = args.output_dir / str(row["episode_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_rgb_frames(observations, output_dir, str(row["episode_id"]))
    _plot_point_cloud(observations, output_dir, str(row["episode_id"]))
    _plot_trajectory(trajectory, output_dir, str(row["episode_id"]))

    summary_path = output_dir / f"{row['episode_id']}_summary.json"
    summary = {
        "dataset_root": str(dataset_root),
        "episode_id": str(row["episode_id"]),
        "task_name": str(row["task_name"]),
        "split": str(row["split"]),
        "num_observations": int(row["num_observations"]),
        "observation_keys": list(observations.keys()),
        "trajectory_keys": list(trajectory.keys()),
        "metadata_keys": list(metadata.keys()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved visual inspection artifacts to: {output_dir}")
    print(f"Summary: {summary_path}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
