"""Helpers for loading exported episode datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExportedEpisode:
    episode_id: str
    task_name: str
    split: str
    observation_path: Path
    trajectory_path: Path
    metadata_path: Path
    observation: dict[str, np.ndarray]
    trajectory: dict[str, np.ndarray]
    metadata: dict[str, Any]


def load_manifest(dataset_root: Path | str) -> pd.DataFrame:
    """Implement the load_manifest operation used by this module."""
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return pd.read_parquet(manifest_path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load the requested data or model artifact."""
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _load_json(path: Path) -> dict[str, Any]:
    """Load the requested data or model artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_exported_episode(dataset_root: Path | str, row: pd.Series | dict[str, Any]) -> ExportedEpisode:
    """Implement the load_exported_episode operation used by this module."""
    dataset_root = Path(dataset_root)
    observation_path = dataset_root / str(row["observation_path"])
    trajectory_path = dataset_root / str(row["trajectory_path"])
    metadata_path = dataset_root / str(row["metadata_path"])
    return ExportedEpisode(
        episode_id=str(row["episode_id"]),
        task_name=str(row["task_name"]),
        split=str(row["split"]),
        observation_path=observation_path,
        trajectory_path=trajectory_path,
        metadata_path=metadata_path,
        observation=_load_npz(observation_path),
        trajectory=_load_npz(trajectory_path),
        metadata=_load_json(metadata_path),
    )


def iter_exported_episodes(dataset_root: Path | str):
    """Implement the iter_exported_episodes operation used by this module."""
    dataset_root = Path(dataset_root)
    manifest = load_manifest(dataset_root)

    for _, row in manifest.iterrows():
        yield load_exported_episode(dataset_root, row)


def load_exported_episodes(dataset_root: Path | str) -> list[ExportedEpisode]:
    """Implement the load_exported_episodes operation used by this module."""
    return list(iter_exported_episodes(dataset_root))
