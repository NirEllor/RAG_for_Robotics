#!/usr/bin/env python
"""Compare two exported ReachTarget episodes side by side."""

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
        help="Exported ReachTarget dataset root.",
    )
    parser.add_argument("episode_a", type=str, help="First episode id, e.g. episode0")
    parser.add_argument("episode_b", type=str, help="Second episode id, e.g. episode1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "comparison",
        help="Where to save the comparison figures.",
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


def _load_row(manifest: pd.DataFrame, episode_id: str) -> pd.Series:
    """Load the requested data or model artifact."""
    matches = manifest[manifest["episode_id"].astype(str) == episode_id]
    if matches.empty:
        raise ValueError(f"Episode id not found in manifest: {episode_id}")
    return matches.iloc[0]


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


def _plot_rgb(ax, rgb: np.ndarray, title: str) -> None:
    """Implement the _plot_rgb operation used by this module."""
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")


def _plot_point_cloud(ax, point_cloud: np.ndarray, title: str) -> None:
    """Implement the _plot_point_cloud operation used by this module."""
    points = np.asarray(point_cloud[0], dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > 2500:
        rng = np.random.default_rng(42)
        points = points[rng.choice(len(points), size=2500, replace=False)]
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _plot_trajectory(ax, trajectory: dict[str, np.ndarray], title: str) -> None:
    """Implement the _plot_trajectory operation used by this module."""
    if "joint_positions" in trajectory:
        ax.plot(np.asarray(trajectory["joint_positions"], dtype=np.float32))
        ax.set_xlabel("t")
        ax.set_ylabel("joint position")
    elif "gripper_pose" in trajectory:
        gripper_pose = np.asarray(trajectory["gripper_pose"], dtype=np.float32)
        ax.plot(gripper_pose[:, :3])
        ax.set_xlabel("t")
        ax.set_ylabel("position")
    else:
        ax.text(0.5, 0.5, "No trajectory field found", ha="center", va="center")
    ax.set_title(title)


def main() -> int:
    """Run the command-line entry point."""
    args = _parse_args()
    dataset_root = args.dataset_root
    manifest = _load_manifest(dataset_root)
    row_a = _load_row(manifest, args.episode_a)
    row_b = _load_row(manifest, args.episode_b)
    obs_a, traj_a, meta_a = _load_episode(dataset_root, row_a)
    obs_b, traj_b, meta_b = _load_episode(dataset_root, row_b)

    output_dir = args.output_dir / f"{args.episode_a}_vs_{args.episode_b}"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _plot_rgb(axes[0, 0], obs_a["front_rgb"][0], f"{args.episode_a}: front_rgb[0]")
    _plot_rgb(axes[0, 1], obs_b["front_rgb"][0], f"{args.episode_b}: front_rgb[0]")
    _plot_rgb(axes[1, 0], obs_a["front_rgb"][-1], f"{args.episode_a}: front_rgb[-1]")
    _plot_rgb(axes[1, 1], obs_b["front_rgb"][-1], f"{args.episode_b}: front_rgb[-1]")
    fig.suptitle("RGB comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "rgb_comparison.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    _plot_point_cloud(ax1, obs_a["front_point_cloud_world"], f"{args.episode_a}: point cloud")
    _plot_point_cloud(ax2, obs_b["front_point_cloud_world"], f"{args.episode_b}: point cloud")

    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    _plot_trajectory(ax3, traj_a, f"{args.episode_a}: trajectory")
    _plot_trajectory(ax4, traj_b, f"{args.episode_b}: trajectory")
    fig.suptitle("Point cloud and trajectory comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "geometry_trajectory_comparison.png", dpi=170)
    plt.close(fig)

    summary = {
        "dataset_root": str(dataset_root),
        "episode_a": args.episode_a,
        "episode_b": args.episode_b,
        "rows": {
            args.episode_a: {
                "task_name": str(row_a["task_name"]),
                "split": str(row_a["split"]),
                "num_observations": int(row_a["num_observations"]),
            },
            args.episode_b: {
                "task_name": str(row_b["task_name"]),
                "split": str(row_b["split"]),
                "num_observations": int(row_b["num_observations"]),
            },
        },
        "metadata_keys": {
            args.episode_a: list(meta_a.keys()),
            args.episode_b: list(meta_b.keys()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved comparison artifacts to: {output_dir}")
    print(f"RGB: {output_dir / 'rgb_comparison.png'}")
    print(f"Geometry/trajectory: {output_dir / 'geometry_trajectory_comparison.png'}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
