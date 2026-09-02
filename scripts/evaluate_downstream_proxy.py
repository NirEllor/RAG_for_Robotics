#!/usr/bin/env python3
"""Evaluate a lightweight retrieved-trajectory transfer proxy.

This is not simulator execution. It measures whether the retrieved episode
provides a task-compatible trajectory and whether its action arrays are present
and structurally usable for a downstream planner.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _parse_ids(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else []


def main() -> int:
    args = _args()
    manifest = pd.read_parquet(args.dataset_root / "manifest.parquet")
    episode_to_task = dict(zip(manifest["episode_id"].astype(str), manifest["task_name"].astype(str)))
    trajectory_paths = dict(zip(manifest["episode_id"].astype(str), manifest["trajectory_path"].astype(str)))
    per_query_path = args.retrieval_dir / "per_query_metrics.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(f"Missing retrieval per-query output: {per_query_path}")

    per_query = pd.read_csv(per_query_path)
    rows: list[dict[str, object]] = []
    for _, item in per_query.iterrows():
        query_id = str(item["query_episode_id"])
        candidate_ids = _parse_ids(item.get("topk_episode_ids"))
        top1_id = candidate_ids[0] if candidate_ids else None
        query_task = episode_to_task.get(query_id)
        candidate_task = episode_to_task.get(top1_id) if top1_id else None
        query_path = args.dataset_root / str(trajectory_paths[query_id]) if query_id in trajectory_paths else None
        candidate_path = args.dataset_root / str(trajectory_paths[top1_id]) if top1_id in trajectory_paths else None

        action_available = False
        shape_compatible = False
        query_steps = 0
        candidate_steps = 0
        if query_path and candidate_path and query_path.exists() and candidate_path.exists():
            with np.load(query_path) as query_npz, np.load(candidate_path) as candidate_npz:
                query_keys = set(query_npz.files)
                candidate_keys = set(candidate_npz.files)
                action_available = "joint_position_action" in candidate_keys or "gripper_pose" in candidate_keys
                shared = query_keys & candidate_keys
                shape_compatible = bool(shared) and all(query_npz[key].ndim == candidate_npz[key].ndim for key in shared)
                if shared:
                    first_key = sorted(shared)[0]
                    query_steps = int(query_npz[first_key].shape[0]) if query_npz[first_key].ndim else 1
                    candidate_steps = int(candidate_npz[first_key].shape[0]) if candidate_npz[first_key].ndim else 1

        rows.append(
            {
                "method": str(item["method"]),
                "k": int(item["k"]),
                "query_episode_id": query_id,
                "query_task": query_task,
                "retrieved_episode_id": top1_id,
                "retrieved_task": candidate_task,
                "task_match": bool(query_task and query_task == candidate_task),
                "trajectory_action_available": action_available,
                "trajectory_shape_compatible": shape_compatible,
                "query_steps": query_steps,
                "retrieved_steps": candidate_steps,
            }
        )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["method", "k"], as_index=False)
        .agg(
            num_queries=("query_episode_id", "size"),
            task_transfer_rate=("task_match", "mean"),
            action_available_rate=("trajectory_action_available", "mean"),
            shape_compatible_rate=("trajectory_shape_compatible", "mean"),
            mean_retrieved_steps=("retrieved_steps", "mean"),
        )
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "per_query_downstream_proxy.csv", index=False)
    summary.to_csv(output_dir / "summary_downstream_proxy.csv", index=False)

    report = [
        "# Downstream Trajectory-Transfer Proxy",
        "",
        f"- Dataset: `{args.dataset_root}`",
        f"- Retrieval outputs: `{args.retrieval_dir}`",
        "",
        "This is an offline transfer proxy, not simulator execution or planning success.",
        "`task_transfer_rate` measures whether the top retrieved trajectory comes from the query task.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
    ]
    (output_dir / "summary_report.md").write_text("\n".join(report), encoding="utf-8")
    (output_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "retrieval_dir": str(args.retrieval_dir),
                "evaluation_type": "offline_trajectory_transfer_proxy",
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Downstream proxy outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
