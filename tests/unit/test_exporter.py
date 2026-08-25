from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from action_retrieval.data.exporter import RLBenchDemoSpec, export_rlbench_dataset_from_specs
from action_retrieval.data.validator import validate_dataset_root


class _FakeObservation:
    def __init__(self, *, color: int, offset: float, variation_index: int):
        self.front_rgb = np.full((4, 4, 3), color, dtype=np.uint8)
        self.front_depth = np.full((4, 4), offset, dtype=np.float32)
        self.front_point_cloud = np.full((4, 4, 3), offset, dtype=np.float32)
        self.joint_positions = np.full((7,), offset, dtype=np.float32)
        self.joint_velocities = np.full((7,), offset + 0.5, dtype=np.float32)
        self.joint_forces = np.full((7,), offset + 1.0, dtype=np.float32)
        self.gripper_open = np.array([1.0], dtype=np.float32)
        self.gripper_pose = np.full((7,), offset + 2.0, dtype=np.float32)
        self.gripper_matrix = np.eye(4, dtype=np.float32)
        self.gripper_joint_positions = np.full((2,), offset + 3.0, dtype=np.float32)
        self.gripper_touch_forces = np.full((2,), offset + 4.0, dtype=np.float32)
        self.task_low_dim_state = np.full((3,), offset + 5.0, dtype=np.float32)
        self.misc = {
            "variation_index": variation_index,
            "front_camera_intrinsics": np.eye(3, dtype=np.float32),
            "front_camera_extrinsics": np.eye(4, dtype=np.float32),
            "joint_position_action": np.full((7,), offset + 6.0, dtype=np.float32),
        }


def _fake_demo(color: int, offset: float, variation_index: int):
    return [_FakeObservation(color=color, offset=offset, variation_index=variation_index) for _ in range(2)]


def test_export_rlbench_dataset_from_specs_supports_multiple_tasks(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    for task_name, description in {
        "reach_target": "reach the target",
        "push_buttons": "push the button",
    }.items():
        variation_dir = tasks_root / task_name / "variation0"
        variation_dir.mkdir(parents=True, exist_ok=True)
        with (variation_dir / "variation_descriptions.pkl").open("wb") as handle:
            pickle.dump([description], handle)

    dataset_root = tmp_path / "exported_dataset"
    result = export_rlbench_dataset_from_specs(
        dataset_root=dataset_root,
        specs=[
            RLBenchDemoSpec(
                task_name="reach_target",
                demo=_fake_demo(color=128, offset=0.0, variation_index=0),
                source_root=str(tasks_root),
                source_episode_directory=tasks_root / "reach_target" / "variation0" / "episodes" / "episode0",
                variation_id=0,
            ),
            RLBenchDemoSpec(
                task_name="push_buttons",
                demo=_fake_demo(color=64, offset=1.0, variation_index=0),
                source_root=str(tasks_root),
                source_episode_directory=tasks_root / "push_buttons" / "variation0" / "episodes" / "episode0",
                variation_id=0,
            ),
        ],
        split_seed=7,
        overwrite=True,
        dataset_task_name="mixed",
        generator="scripts/build_multitask_dataset.py",
    )

    assert Path(result.dataset_root) == dataset_root
    assert (dataset_root / "episodes" / "reach_target" / "episode0" / "observation.npz").exists()
    assert (dataset_root / "episodes" / "push_buttons" / "episode1" / "observation.npz").exists()
    assert validate_dataset_root(dataset_root).ok

    manifest = pd.read_parquet(dataset_root / "manifest.parquet")
    assert set(manifest["task_name"]) == {"reach_target", "push_buttons"}
    assert set(manifest["episode_id"]) == {"episode0", "episode1"}
