"""Utilities for loading saved RLBench demonstrations.

This repository's Phase 0 fallback uses pre-generated RLBench demos when live
CoppeliaSim episode generation is unstable on the current machine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def _project_root() -> Path:
    """Resolve the relevant project path."""
    return Path(__file__).resolve().parents[3]


def candidate_saved_demo_roots() -> list[Path]:
    """Return likely saved-demo dataset roots in priority order."""
    roots: list[Path] = []

    env_root = os.environ.get("RLBENCH_DATASET_ROOT")
    if env_root:
        roots.append(Path(env_root))

    roots.append(_project_root() / "data" / "rlbench_saved")
    roots.append(_project_root() / "data" / "rlbench_cache")
    roots.append(_project_root() / "third_party" / "RLBench" / "tests" / "unit" / "assets" / "tasks")
    return roots


def _has_task_variation(root: Path, task_name: str, variation_number: int = 0) -> bool:
    """Implement the _has_task_variation operation used by this module."""
    return (root / task_name / f"variation{variation_number}" / "variation_descriptions.pkl").exists()


def resolve_saved_task_demo_root(
    task_name: str,
    search_roots: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    """Find a usable saved-demo root for an RLBench task.

    The root can either be the dataset root itself or the tasks/ directory.
    """
    roots = list(search_roots) if search_roots is not None else candidate_saved_demo_roots()
    for root in roots:
        if not root.exists():
            continue
        if _has_task_variation(root, task_name):
            return root
        if _has_task_variation(root / "tasks", task_name):
            return root / "tasks"
        if (root / "variation0" / "variation_descriptions.pkl").exists():
            return root

    # Explicit final fallback to the bundled RLBench test assets.
    bundled_tasks_root = (
        _project_root() / "third_party" / "RLBench" / "tests" / "unit" / "assets" / "tasks"
    )
    if _has_task_variation(bundled_tasks_root, task_name):
        return bundled_tasks_root
    return None


def resolve_saved_demo_root(search_roots: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """Backward-compatible wrapper for ReachTarget saved-demo discovery."""
    return resolve_saved_task_demo_root("reach_target", search_roots=search_roots)


def load_saved_reach_target_demo(
    amount: int = 1,
    image_paths: bool = False,
    variation_number: int = 0,
    from_episode_number: int = 0,
    search_roots: Optional[Iterable[Path]] = None,
):
    """Load saved ReachTarget demos from disk using RLBench's stored-demo API."""
    demos, _, _ = load_saved_task_demo_batch(
        task_name="reach_target",
        amount=amount,
        image_paths=image_paths,
        variation_number=variation_number,
        from_episode_number=from_episode_number,
        search_roots=search_roots,
    )
    return demos


def load_saved_reach_target_demo_batch(
    amount: int = 1,
    image_paths: bool = False,
    variation_number: int = 0,
    from_episode_number: int = 0,
    search_roots: Optional[Iterable[Path]] = None,
):
    """Load saved ReachTarget demos and return the resolved source metadata."""
    return load_saved_task_demo_batch(
        task_name="reach_target",
        amount=amount,
        image_paths=image_paths,
        variation_number=variation_number,
        from_episode_number=from_episode_number,
        search_roots=search_roots,
    )


def load_saved_task_demo_batch(
    task_name: str,
    amount: int = 1,
    image_paths: bool = False,
    variation_number: int = 0,
    from_episode_number: int = 0,
    search_roots: Optional[Iterable[Path]] = None,
):
    """Load saved demos for a specific RLBench task and return source metadata."""
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig

    dataset_root = resolve_saved_task_demo_root(task_name, search_roots)
    if dataset_root is None:
        raise FileNotFoundError(
            "Could not find a saved RLBench demo root. "
            "Expected either RLBENCH_DATASET_ROOT or the bundled test assets."
        )

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.front_camera.rgb = True
    obs_config.wrist_camera.rgb = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(),
        gripper_action_mode=Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        dataset_root=str(dataset_root),
        obs_config=obs_config,
        headless=True,
    )
    episodes_root = dataset_root / task_name / f"variation{variation_number}" / "episodes"
    if not episodes_root.exists():
        alternate_root = dataset_root / f"variation{variation_number}" / "episodes"
        if alternate_root.exists():
            episodes_root = alternate_root
    source_episode_dirs = sorted(p for p in episodes_root.iterdir() if p.is_dir())
    available_count = max(0, len(source_episode_dirs) - from_episode_number)
    if available_count <= 0:
        raise RuntimeError(
            f"No saved episodes available from_episode_number={from_episode_number} "
            f"under {episodes_root}"
        )
    if amount > available_count:
        print(
            f"WARNING: Requested {amount} saved demos but only {available_count} "
            f"are available from episode {from_episode_number}; exporting {available_count}."
        )
        amount = available_count
    try:
        demos = env.get_demos(
            task_name,
            amount,
            variation_number=variation_number,
            image_paths=image_paths,
            random_selection=False,
            from_episode_number=from_episode_number,
        )
        selected_source_episode_dirs = source_episode_dirs[
            from_episode_number : from_episode_number + len(demos)
        ]
        return demos, dataset_root, selected_source_episode_dirs
    finally:
        env.shutdown()
