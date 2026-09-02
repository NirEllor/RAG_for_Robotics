#!/usr/bin/env python3
"""Run a deliberately small RLBench trajectory-replay planning pilot.

Without ``--execute`` this only validates the dataset, RLBench imports, and
action availability. With ``--execute`` it replays stored actions in a live
headless environment. This is an execution pilot, not a learned planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASK_CLASSES = {
    "reach_target": "ReachTarget",
    "open_drawer": "OpenDrawer",
    "close_jar": "CloseJar",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_CLASSES), default="reach_target")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually launch CoppeliaSim/RLBench.")
    parser.add_argument(
        "--allow-derived-actions",
        action="store_true",
        help="Allow joint-position states to be replayed as position targets; not ground-truth actions.",
    )
    return parser.parse_args()


def _trajectory_actions(path: Path, *, allow_derived: bool = False) -> tuple[np.ndarray | None, str]:
    with np.load(path) as payload:
        if "joint_position_action" in payload.files and payload["joint_position_action"].size:
            return np.asarray(payload["joint_position_action"], dtype=np.float32), "explicit_action"
        if allow_derived and "joint_positions" in payload.files and payload["joint_positions"].size:
            positions = np.asarray(payload["joint_positions"], dtype=np.float32)
            if "gripper_open" in payload.files:
                gripper = np.asarray(payload["gripper_open"], dtype=np.float32).reshape(len(positions), -1)
                return np.concatenate([positions, gripper[:, :1]], axis=1), "derived_from_joint_positions"
            return positions, "derived_from_joint_positions"
    return None, "missing"


def main() -> int:
    args = _args()
    manifest = pd.read_parquet(args.dataset_root / "manifest.parquet")
    rows = manifest[manifest["task_name"].astype(str) == args.task].head(args.episodes)
    if rows.empty:
        raise RuntimeError(f"No dataset episodes found for task {args.task!r}.")
    records = []
    for _, row in rows.iterrows():
        path = args.dataset_root / str(row["trajectory_path"])
        actions, source = _trajectory_actions(path, allow_derived=True)
        records.append({"episode_id": str(row["episode_id"]), "action_steps": int(len(actions)) if actions is not None else 0, "action_source": source})
    if not any(item["action_steps"] for item in records):
        raise RuntimeError("Selected episodes contain neither explicit actions nor joint-position trajectories.")

    result = {
        "task": args.task,
        "episodes_requested": args.episodes,
        "episodes_checked": len(records),
        "execute": args.execute,
        "allow_derived_actions": args.allow_derived_actions,
        "records": records,
        "evaluation_type": "rlbench_trajectory_replay_pilot",
        "warning": "Replay is not a learned planner. Derived joint-position replay is not ground-truth action replay.",
    }
    if args.execute:
        try:
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import JointPosition
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.environment import Environment
            task_module = __import__("rlbench.tasks", fromlist=[TASK_CLASSES[args.task]])
            task_class = getattr(task_module, TASK_CLASSES[args.task])
        except Exception as exc:  # pragma: no cover - cluster-only dependency
            raise RuntimeError(f"RLBench/CoppeliaSim imports failed: {type(exc).__name__}: {exc}") from exc

        env = Environment(
            action_mode=MoveArmThenGripper(arm_action_mode=JointPosition(), gripper_action_mode=Discrete()),
            dataset_root="",
            obs_config=None,
            headless=True,
        )
        successes = 0
        try:
            task = env.get_task(task_class)
            task.set_variation(args.variation)
            for record, (_, row) in zip(records, rows.iterrows()):
                actions, source = _trajectory_actions(
                    args.dataset_root / str(row["trajectory_path"]),
                    allow_derived=args.allow_derived_actions,
                )
                if source != "explicit_action" and not args.allow_derived_actions:
                    raise RuntimeError(
                        "Refusing simulator execution without explicit joint_position_action arrays. "
                        "Use --allow-derived-actions only for the clearly labeled replay pilot."
                    )
                task.reset()
                reward = 0.0
                for action in actions if actions is not None else []:
                    _, reward, terminate = task.step(action)
                    if terminate:
                        break
                record["final_reward"] = float(reward)
                record["success"] = bool(reward > 0)
                successes += int(record["success"])
            result["successes"] = successes
            result["success_rate"] = successes / len(records)
        finally:
            env.shutdown()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "summary_report.md").write_text(
        "# RLBench Planning Pilot\n\n"
        f"- Task: `{args.task}`\n- Executed: `{args.execute}`\n"
        f"- Episodes checked: `{len(records)}`\n\n"
        "This is a small trajectory-replay execution pilot. Derived joint-position replay is "
        "not a general planning-policy benchmark.\n",
        encoding="utf-8",
    )
    print(f"Planning pilot outputs written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
