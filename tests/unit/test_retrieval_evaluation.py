from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from action_retrieval.evaluation.retrieval_eval import (
    evaluate_retrieval_methods,
    load_relevance_annotations,
)
from action_retrieval.evaluation.metrics import (
    average_precision_at_k,
    normalized_discounted_cumulative_gain,
)


def test_load_relevance_annotations(tmp_path: Path):
    annotations_path = tmp_path / "labels.json"
    annotations_path.write_text('{"episode0": ["episode1"]}', encoding="utf-8")

    annotations = load_relevance_annotations(annotations_path)

    assert annotations["episode0"] == {"episode1"}


def _write_exported_episode(dataset_root: Path, episode_id: str, task_name: str) -> None:
    observation_dir = dataset_root / "observation"
    trajectory_dir = dataset_root / "trajectory"
    metadata_dir = dataset_root / "metadata"
    observation_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        observation_dir / f"{episode_id}.npz",
        front_rgb=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        front_point_cloud_world=np.zeros((2, 4, 4, 3), dtype=np.float32),
    )
    np.savez(
        trajectory_dir / f"{episode_id}.npz",
        joint_positions=np.zeros((2, 7), dtype=np.float32),
        joint_velocities=np.zeros((2, 7), dtype=np.float32),
        gripper_pose=np.zeros((2, 7), dtype=np.float32),
        gripper_open=np.zeros((2,), dtype=np.float32),
    )
    (metadata_dir / f"{episode_id}.json").write_text(
        json.dumps({"episode": {"seed": 42}, "task_name": task_name}),
        encoding="utf-8",
    )


def _write_minimal_dataset(dataset_root: Path) -> None:
    rows = []
    for episode_id, task_name, split in [
        ("episode0", "reach_target", "train"),
        ("episode1", "reach_target", "train"),
        ("episode2", "reach_target", "train"),
        ("episode3", "reach_target", "train"),
    ]:
        _write_exported_episode(dataset_root, episode_id, task_name)
        rows.append(
            {
                "episode_id": episode_id,
                "task_name": task_name,
                "split": split,
                "observation_path": f"observation/{episode_id}.npz",
                "trajectory_path": f"trajectory/{episode_id}.npz",
                "metadata_path": f"metadata/{episode_id}.json",
            }
        )

    pd.DataFrame(rows).to_parquet(dataset_root / "manifest.parquet", index=False)


def test_evaluate_retrieval_methods_returns_multiple_cutoffs(tmp_path: Path):
    dataset_root = tmp_path / "v1_reach_target"
    annotations = {
        "episode0": {"episode1"},
        "episode1": {"episode0"},
        "episode2": {"episode3"},
        "episode3": {"episode2"},
    }

    dataset_root.mkdir(parents=True, exist_ok=True)
    _write_minimal_dataset(dataset_root)

    runs = evaluate_retrieval_methods(
        dataset_root,
        annotations,
        methods=["pose_descriptor"],
        ks=[1, 2],
    )

    assert {run.k for run in runs} == {1, 2}
    assert {run.method for run in runs} == {"pose_descriptor"}


def test_average_precision_and_ndcg_are_normalized():
    assert average_precision_at_k([1, 0, 1], total_relevant=3, k=3) == 0.5555555555555555
    assert normalized_discounted_cumulative_gain([1, 0], total_relevant=1, k=2) == 1.0
