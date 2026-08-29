#!/usr/bin/env python
"""Build task-level relevance annotations from an exported RLBench dataset.

For a given dataset root, this script creates a JSON file that maps each
episode_id to the list of other episode_ids with the same task_name.
That is a practical evaluation proxy for the current subset benchmarks.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Exported dataset root containing manifest.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON file that will be written.",
    )
    parser.add_argument(
        "--task-name",
        nargs="*",
        default=None,
        help="Optional task-name filter. If omitted, all tasks in the dataset are used.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.dataset_root / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_parquet(manifest_path)
    if args.task_name:
        selected_tasks = {str(task_name) for task_name in args.task_name}
        manifest = manifest[manifest["task_name"].astype(str).isin(selected_tasks)].copy()
        if manifest.empty:
            raise ValueError(f"No rows matched the requested task filter: {sorted(selected_tasks)}")

    task_to_episode_ids: dict[str, list[str]] = defaultdict(list)
    for _, row in manifest.iterrows():
        task_to_episode_ids[str(row["task_name"])].append(str(row["episode_id"]))

    annotations: dict[str, list[str]] = {}
    for _, row in manifest.iterrows():
        task_name = str(row["task_name"])
        episode_id = str(row["episode_id"])
        relevant = [candidate for candidate in task_to_episode_ids[task_name] if candidate != episode_id]
        annotations[episode_id] = relevant

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(annotations, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Task Relevance Annotations")
    print("=" * 80)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {args.output}")
    print(f"Tasks: {sorted(task_to_episode_ids)}")
    print(f"Episode count: {len(annotations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
