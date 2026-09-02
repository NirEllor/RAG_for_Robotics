"""Dataset validation helpers for Phase 1 exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


def _sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for the requested file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path):
    """Load the requested data or model artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_file(errors: list[str], path: Path, label: str) -> bool:
    """Implement the _require_file operation used by this module."""
    if not path.exists():
        errors.append(f"Missing {label}: {path}")
        return False
    return True


def validate_dataset_root(dataset_root: Path | str) -> ValidationResult:
    """Implement the validate_dataset_root operation used by this module."""
    dataset_root = Path(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []

    metadata_path = dataset_root / "dataset_metadata.json"
    manifest_path = dataset_root / "manifest.parquet"
    split_dir = dataset_root / "splits"

    if not dataset_root.exists():
        return ValidationResult(False, (f"Dataset root does not exist: {dataset_root}",), ())

    _require_file(errors, metadata_path, "dataset metadata")
    _require_file(errors, manifest_path, "manifest parquet")
    if not split_dir.exists():
        errors.append(f"Missing splits directory: {split_dir}")

    if errors:
        return ValidationResult(False, tuple(errors), tuple(warnings))

    metadata = _load_json(metadata_path)
    manifest = pd.read_parquet(manifest_path)
    required_columns = {
        "dataset_version",
        "task_name",
        "episode_id",
        "variation_id",
        "seed",
        "split",
        "success",
        "num_observations",
        "snapshot_policy",
        "coordinate_frame",
        "source_kind",
        "source_root",
        "observation_path",
        "trajectory_path",
        "metadata_path",
        "observation_sha256",
        "trajectory_sha256",
        "metadata_sha256",
    }
    missing_columns = sorted(required_columns - set(manifest.columns))
    if missing_columns:
        errors.append(f"Manifest missing required columns: {missing_columns}")

    if manifest.empty:
        errors.append("Manifest is empty")

    if not errors:
        manifest_episode_ids = set(manifest["episode_id"].astype(str))
        if len(manifest_episode_ids) != len(manifest):
            errors.append("Manifest episode_id values are not unique")

        split_files = sorted(split_dir.glob("split_seed_*.json"))
        if not split_files:
            errors.append(f"No split manifest found under {split_dir}")
        else:
            split_payload = _load_json(split_files[0])
            split_map = split_payload.get("splits", {})
            split_episode_ids: set[str] = set()
            for split_name, ids in split_map.items():
                ids = [str(episode_id) for episode_id in ids]
                overlap = split_episode_ids.intersection(ids)
                if overlap:
                    errors.append(
                        f"Split leakage detected in {split_name}: duplicated ids {sorted(overlap)}"
                    )
                split_episode_ids.update(ids)
                for episode_id in ids:
                    if episode_id not in manifest_episode_ids:
                        errors.append(f"Split file references missing episode_id {episode_id}")

            missing_from_split = manifest_episode_ids - split_episode_ids
            if missing_from_split:
                errors.append(
                    f"Some manifest episode_ids are missing from the split file: "
                    f"{sorted(missing_from_split)}"
                )

    for _, row in manifest.iterrows():
        episode_dir = dataset_root / "episodes" / str(row["task_name"]) / str(row["episode_id"])
        observation_path = dataset_root / str(row["observation_path"])
        trajectory_path = dataset_root / str(row["trajectory_path"])
        episode_metadata_path = dataset_root / str(row["metadata_path"])

        _require_file(errors, observation_path, "observation npz")
        _require_file(errors, trajectory_path, "trajectory npz")
        _require_file(errors, episode_metadata_path, "episode metadata")
        if not episode_dir.exists():
            errors.append(f"Missing episode directory: {episode_dir}")

        if observation_path.exists():
            if _sha256_file(observation_path) != str(row["observation_sha256"]):
                errors.append(f"Observation checksum mismatch for {row['episode_id']}")
            with np.load(observation_path) as arrays:
                if "front_rgb" not in arrays:
                    errors.append(f"front_rgb missing from {observation_path}")
                else:
                    if arrays["front_rgb"].ndim != 4:
                        errors.append(
                            f"front_rgb has unexpected shape {arrays['front_rgb'].shape}"
                        )
                    if not np.isfinite(arrays["front_rgb"].astype(np.float32)).all():
                        errors.append(f"front_rgb contains non-finite values in {observation_path}")
                if "front_point_cloud_world" in arrays:
                    if not np.isfinite(arrays["front_point_cloud_world"].astype(np.float32)).all():
                        errors.append(
                            f"front_point_cloud_world contains non-finite values in {observation_path}"
                        )

        if trajectory_path.exists():
            if _sha256_file(trajectory_path) != str(row["trajectory_sha256"]):
                errors.append(f"Trajectory checksum mismatch for {row['episode_id']}")

        if episode_metadata_path.exists():
            if _sha256_file(episode_metadata_path) != str(row["metadata_sha256"]):
                errors.append(f"Episode metadata checksum mismatch for {row['episode_id']}")
            episode_metadata = _load_json(episode_metadata_path)
            if episode_metadata.get("episode", {}).get("episode_id") != str(row["episode_id"]):
                errors.append(
                    f"Episode metadata episode_id mismatch for {row['episode_id']}"
                )
            if episode_metadata.get("episode", {}).get("dataset_version") != str(
                row["dataset_version"]
            ):
                errors.append(
                    f"Episode metadata dataset_version mismatch for {row['episode_id']}"
                )

    return ValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))
