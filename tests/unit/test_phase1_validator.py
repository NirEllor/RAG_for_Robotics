from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from action_retrieval.data.validator import validate_dataset_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_episode(root: Path, episode_id: str, split: str, seed: int) -> dict[str, object]:
    episode_dir = root / "episodes" / "reach_target" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    observation_path = episode_dir / "observation.npz"
    trajectory_path = episode_dir / "trajectory.npz"
    metadata_path = episode_dir / "metadata.json"

    np.savez_compressed(
        observation_path,
        front_rgb=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        front_point_cloud_world=np.zeros((2, 4, 4, 3), dtype=np.float32),
    )
    np.savez_compressed(
        trajectory_path,
        joint_positions=np.zeros((2, 7), dtype=np.float32),
        gripper_open=np.zeros((2,), dtype=np.float32),
    )
    metadata_payload = {
        "episode": {
            "dataset_version": "v1_reach_target",
            "task_name": "reach_target",
            "episode_id": episode_id,
            "variation_id": 0,
            "seed": seed,
            "split": split,
            "success": True,
            "num_observations": 2,
            "snapshot_policy": "initial",
            "coordinate_frame": "world",
            "source_kind": "saved_demo",
            "source_root": "/tmp/source",
            "language_descriptions": ["reach the target"],
            "camera_names": ["front"],
            "observation_path": f"episodes/reach_target/{episode_id}/observation.npz",
            "trajectory_path": f"episodes/reach_target/{episode_id}/trajectory.npz",
            "metadata_path": f"episodes/reach_target/{episode_id}/metadata.json",
            "observation_sha256": _sha256(observation_path),
            "trajectory_sha256": _sha256(trajectory_path),
            "metadata_sha256": "",
            "observation_shapes": {"front_rgb": [2, 4, 4, 3], "front_point_cloud_world": [2, 4, 4, 3]},
            "trajectory_shapes": {"joint_positions": [2, 7], "gripper_open": [2]},
            "observation_dtypes": {"front_rgb": "uint8", "front_point_cloud_world": "float32"},
            "trajectory_dtypes": {"joint_positions": "float32", "gripper_open": "float32"},
            "action_sequence_available": False,
            "generated_at_utc": "2026-08-20T00:00:00+00:00",
        }
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    return {
        "dataset_version": "v1_reach_target",
        "task_name": "reach_target",
        "episode_id": episode_id,
        "variation_id": 0,
        "seed": seed,
        "split": split,
        "success": True,
        "num_observations": 2,
        "snapshot_policy": "initial",
        "coordinate_frame": "world",
        "source_kind": "saved_demo",
        "source_root": "/tmp/source",
        "language_descriptions": json.dumps(["reach the target"]),
        "camera_names": json.dumps(["front"]),
        "observation_path": f"episodes/reach_target/{episode_id}/observation.npz",
        "trajectory_path": f"episodes/reach_target/{episode_id}/trajectory.npz",
        "metadata_path": f"episodes/reach_target/{episode_id}/metadata.json",
        "observation_sha256": _sha256(observation_path),
        "trajectory_sha256": _sha256(trajectory_path),
        "metadata_sha256": _sha256(metadata_path),
        "observation_shapes": json.dumps({"front_rgb": [2, 4, 4, 3]}),
        "trajectory_shapes": json.dumps({"joint_positions": [2, 7]}),
        "observation_dtypes": json.dumps({"front_rgb": "uint8"}),
        "trajectory_dtypes": json.dumps({"joint_positions": "float32"}),
        "action_sequence_available": False,
        "generated_at_utc": "2026-08-20T00:00:00+00:00",
    }


def test_validate_dataset_root_accepts_synthetic_dataset(tmp_path: Path):
    root = tmp_path / "v1_reach_target"
    root.mkdir()

    rows = [
        _write_episode(root, "episode0", "train", 42),
        _write_episode(root, "episode1", "val", 43),
    ]

    pd.DataFrame(rows).to_parquet(root / "manifest.parquet", index=False)
    (root / "splits").mkdir()
    (root / "splits" / "split_seed_42.json").write_text(
        json.dumps(
            {
                "dataset_version": "v1_reach_target",
                "task_name": "reach_target",
                "split_seed": 42,
                "splits": {
                    "train": ["episode0"],
                    "val": ["episode1"],
                    "test": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "dataset_version": "v1_reach_target",
                "task_name": "reach_target",
                "split_seed": 42,
                "num_requested_episodes": 2,
                "num_exported_episodes": 2,
                "source_kind": "saved_demo",
                "source_root": "/tmp/source",
                "source_description": "synthetic",
                "coordinate_frame": "world",
                "camera_names": ["front"],
                "image_size": [128, 128],
                "snapshot_policy": "initial",
                "point_cloud_enabled": True,
                "target_crop_enabled": True,
                "target_crop_size": 256,
                "target_crop_margin": 0.1,
                "xyz_only": True,
                "rlbench_version": None,
                "coppeliasim_version": None,
                "generated_at_utc": "2026-08-20T00:00:00+00:00",
                "generator": "scripts/build_reach_target_dataset.py",
                "extra": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = validate_dataset_root(root)

    assert result.ok
    assert result.errors == ()
