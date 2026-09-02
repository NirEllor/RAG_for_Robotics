"""Live RLBench demo collection for ReachTarget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReachTargetDemoSource:
    source_kind: str
    source_root: str
    source_episode_directory: str | None = None


def build_reach_target_observation_config():
    """Implement the build_reach_target_observation_config operation used by this module."""
    from rlbench.observation_config import CameraConfig, ObservationConfig

    front_camera = CameraConfig(rgb=True, depth=True, point_cloud=True, mask=False, image_size=(128, 128))
    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.set_all_low_dim(True)
    obs_config.front_camera = front_camera
    obs_config.wrist_camera.rgb = False
    obs_config.wrist_camera.depth = False
    obs_config.wrist_camera.point_cloud = False
    return obs_config


def collect_live_reach_target_demos(
    amount: int,
    *,
    variation_number: int = 0,
    image_paths: bool = False,
):
    """Collect live ReachTarget demos from a running RLBench/CoppeliaSim scene."""

    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.tasks import ReachTarget

    obs_config = build_reach_target_observation_config()
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        dataset_root="",
        obs_config=obs_config,
        headless=True,
    )
    try:
        task = env.get_task(ReachTarget)
        task.set_variation(variation_number)
        demos = task.get_demos(
            amount,
            live_demos=True,
            image_paths=image_paths,
            random_selection=False,
        )
        sources = [
            ReachTargetDemoSource(
                source_kind="live_demo",
                source_root="live://ReachTarget",
                source_episode_directory=None,
            )
            for _ in demos
        ]
        return demos, sources
    finally:
        env.shutdown()
