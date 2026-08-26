"""Dataset schema objects for Phase 1 exports.

The Phase 1 pipeline keeps a small, explicit manifest for each episode and
stores dense arrays in per-episode NPZ files. The dataclasses here are used by
the exporter and validator to keep the manifest format stable and human-readable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EpisodeRecord:
    """Scalar metadata for a single exported episode."""

    dataset_version: str
    task_name: str
    episode_id: str
    variation_id: int
    seed: int
    split: str
    success: bool
    num_observations: int
    snapshot_policy: str
    coordinate_frame: str
    source_kind: str
    source_root: str
    language_descriptions: Sequence[str] = field(default_factory=tuple)
    camera_names: Sequence[str] = field(default_factory=tuple)
    observation_path: str = ""
    trajectory_path: str = ""
    metadata_path: str = ""
    observation_sha256: str = ""
    trajectory_sha256: str = ""
    metadata_sha256: str = ""
    observation_shapes: Mapping[str, Sequence[int]] = field(default_factory=dict)
    trajectory_shapes: Mapping[str, Sequence[int]] = field(default_factory=dict)
    observation_dtypes: Mapping[str, str] = field(default_factory=dict)
    trajectory_dtypes: Mapping[str, str] = field(default_factory=dict)
    action_sequence_available: bool = False
    generated_at_utc: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["language_descriptions"] = list(self.language_descriptions)
        data["camera_names"] = list(self.camera_names)
        data["observation_shapes"] = {
            key: list(value) for key, value in self.observation_shapes.items()
        }
        data["trajectory_shapes"] = {
            key: list(value) for key, value in self.trajectory_shapes.items()
        }
        return data


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset-level metadata and fingerprint inputs."""

    dataset_version: str
    task_name: str
    split_seed: int
    num_requested_episodes: int
    num_exported_episodes: int
    source_kind: str
    source_root: str
    source_description: str
    coordinate_frame: str
    camera_names: Sequence[str] = field(default_factory=tuple)
    image_size: Sequence[int] = field(default_factory=tuple)
    snapshot_policy: str = "initial"
    point_cloud_enabled: bool = True
    target_crop_enabled: bool = False
    target_crop_size: int | None = None
    target_crop_margin: float | None = None
    xyz_only: bool = True
    rlbench_version: str | None = None
    coppeliasim_version: str | None = None
    generated_at_utc: str = field(default_factory=_utc_now)
    generator: str = "scripts/build_reach_target_dataset.py"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["camera_names"] = list(self.camera_names)
        data["image_size"] = list(self.image_size)
        data["extra"] = dict(self.extra)
        return data


@dataclass(frozen=True)
class DatasetBuildResult:
    """Convenience bundle returned by the exporter."""

    dataset_root: str
    metadata_path: str
    manifest_path: str
    split_path: str
    num_exported_episodes: int
    exported_episode_ids: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "metadata_path": self.metadata_path,
            "manifest_path": self.manifest_path,
            "split_path": self.split_path,
            "num_exported_episodes": self.num_exported_episodes,
            "exported_episode_ids": list(self.exported_episode_ids),
        }
