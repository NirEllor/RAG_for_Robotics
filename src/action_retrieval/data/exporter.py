"""Dataset export helpers for RLBench demonstrations."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from action_retrieval.data.schema import DatasetBuildResult, DatasetMetadata, EpisodeRecord
from action_retrieval.data.transforms import camera_to_world, summarize_array, world_to_camera
from action_retrieval.simulation.saved_importer import (
    load_saved_task_demo_batch,
    resolve_saved_task_demo_root,
)
from action_retrieval.utils.env_report import collect_environment_info


@dataclass(frozen=True)
class RLBenchDemoSpec:
    """One exported RLBench demo and its provenance."""

    task_name: str
    demo: Any
    source_kind: str = "saved_demo"
    source_root: str = ""
    source_episode_directory: Path | str | None = None
    variation_id: int | None = None
    language_descriptions: Sequence[str] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.write("\n")


def _stack_field(observations: Sequence[Any], field: str) -> np.ndarray | None:
    values = []
    for obs in observations:
        value = getattr(obs, field, None)
        if value is None:
            return None
        values.append(np.asarray(value))
    return np.stack(values, axis=0)


def _stack_optional_misc(observations: Sequence[Any], key: str) -> np.ndarray | None:
    values = []
    for obs in observations:
        misc = getattr(obs, "misc", {}) or {}
        if key not in misc:
            return None
        values.append(np.asarray(misc[key]))
    return np.stack(values, axis=0)


def _extract_language_descriptions_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace")]
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        return _extract_language_descriptions_from_value(value.tolist())
    if isinstance(value, dict):
        for candidate_key in ("descriptions", "description", "language_descriptions"):
            if candidate_key in value:
                return _extract_language_descriptions_from_value(value[candidate_key])
        return [json.dumps(value, sort_keys=True)]
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_extract_language_descriptions_from_value(item))
        return flattened
    return [str(value)]


def _extract_language_descriptions(task_name: str, variation_id: int, source_root: str = "") -> list[str]:
    search_roots = [Path(source_root)] if source_root else None
    task_root = resolve_saved_task_demo_root(task_name, search_roots)
    if task_root is None:
        return []

    variation_candidates = [
        task_root / task_name / f"variation{variation_id}" / "variation_descriptions.pkl",
        task_root / f"variation{variation_id}" / "variation_descriptions.pkl",
    ]
    variation_path = next((path for path in variation_candidates if path.exists()), None)
    if variation_path is None:
        return []

    with variation_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        for key in (variation_id, str(variation_id), "descriptions", "variation_descriptions"):
            if key in payload:
                payload = payload[key]
                break

    if isinstance(payload, (str, bytes)):
        return [payload.decode() if isinstance(payload, bytes) else payload]
    if isinstance(payload, np.ndarray):
        payload = payload.tolist()
    if isinstance(payload, (list, tuple)):
        flattened: list[str] = []
        for item in payload:
            flattened.extend(_extract_language_descriptions_from_value(item))
        return flattened
    return _extract_language_descriptions_from_value(payload)


def _build_split(episode_ids: Sequence[str], seed: int) -> dict[str, list[str]]:
    ids = list(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)

    if n <= 1:
        return {"train": ids, "val": [], "test": []}
    if n == 2:
        return {"train": [ids[0]], "val": [ids[1]], "test": []}

    n_test = max(1, n // 5)
    n_val = max(1, n // 5)
    n_train = n - n_val - n_test

    while n_train < 1:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
        n_train = n - n_val - n_test

    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def _normalize_episode_directories(values: Sequence[Path | str | None] | None) -> list[Path | None]:
    if values is None:
        return []
    normalized: list[Path | None] = []
    for value in values:
        normalized.append(None if value is None else Path(value))
    return normalized


def _load_existing_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    return pd.read_parquet(manifest_path).to_dict(orient="records")


def _load_existing_split(split_path: Path) -> dict[str, list[str]]:
    if not split_path.exists():
        return {"train": [], "val": [], "test": []}
    with split_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    splits = payload.get("splits", payload)
    return {
        "train": list(splits.get("train", [])),
        "val": list(splits.get("val", [])),
        "test": list(splits.get("test", [])),
    }


def _next_episode_index(existing_rows: Sequence[dict[str, Any]]) -> int:
    indices = []
    for row in existing_rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id.startswith("episode") and episode_id[len("episode") :].isdigit():
            indices.append(int(episode_id[len("episode") :]))
    return max(indices, default=-1) + 1


def export_rlbench_dataset_from_specs(
    dataset_root: Path,
    specs: Sequence[RLBenchDemoSpec],
    split_seed: int,
    overwrite: bool = False,
    resume: bool = False,
    snapshot_policy: str = "initial",
    target_crop_enabled: bool = True,
    target_crop_size: int | None = 256,
    target_crop_margin: float | None = 0.1,
    xyz_only: bool = True,
    *,
    generator: str = "scripts/build_reach_target_dataset.py",
    dataset_task_name: str | None = None,
) -> DatasetBuildResult:
    """Export RLBench demos into the Phase 1 on-disk layout."""

    dataset_root = Path(dataset_root)
    if dataset_root.exists():
        if resume:
            pass
        elif overwrite:
            import shutil

            shutil.rmtree(dataset_root)
        else:
            raise FileExistsError(f"Dataset root already exists: {dataset_root}")
    dataset_root.mkdir(parents=True, exist_ok=True)

    spec_list = list(specs)
    if not spec_list:
        raise ValueError("At least one demo spec is required to export a dataset.")

    env_info = collect_environment_info(_project_root())
    manifest_path = dataset_root / "manifest.parquet"
    split_path = dataset_root / "splits" / f"split_seed_{split_seed}.json"
    existing_rows = _load_existing_manifest_rows(manifest_path) if resume else []
    existing_split = _load_existing_split(split_path) if resume else {"train": [], "val": [], "test": []}
    next_episode_index = _next_episode_index(existing_rows)

    episode_records: list[EpisodeRecord] = []
    pending_metadata: list[tuple[Path, dict[str, Any], str]] = []
    episode_ids: list[str] = [str(row["episode_id"]) for row in existing_rows]

    for offset, spec in enumerate(spec_list):
        episode_id = f"episode{next_episode_index + offset}"
        episode_ids.append(episode_id)
        task_name = spec.task_name
        episode_dir = dataset_root / "episodes" / task_name / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        observations = list(spec.demo)
        if not observations:
            raise ValueError(f"Demo {episode_id} for task {task_name} is empty.")
        first_obs = observations[0]
        variation_id = (
            int(spec.variation_id)
            if spec.variation_id is not None
            else int((first_obs.misc or {}).get("variation_index", 0))
        )

        source_root = str(spec.source_root or "")
        source_episode_directory = (
            Path(spec.source_episode_directory)
            if spec.source_episode_directory is not None
            else None
        )
        language_descriptions = list(spec.language_descriptions) or _extract_language_descriptions(
            task_name,
            variation_id,
            source_root=source_root,
        )

        front_rgb = _stack_field(observations, "front_rgb")
        front_depth = _stack_field(observations, "front_depth")
        front_point_cloud = _stack_field(observations, "front_point_cloud")
        front_camera_intrinsics = np.asarray(
            (first_obs.misc or {}).get("front_camera_intrinsics"), dtype=np.float32
        )
        front_camera_extrinsics = np.asarray(
            (first_obs.misc or {}).get("front_camera_extrinsics"), dtype=np.float32
        )
        front_point_cloud_camera = (
            world_to_camera(front_point_cloud, front_camera_extrinsics)
            if front_point_cloud is not None
            else None
        )
        front_point_cloud_world = (
            camera_to_world(front_point_cloud_camera, front_camera_extrinsics)
            if front_point_cloud_camera is not None
            else None
        )

        observation_arrays: dict[str, np.ndarray] = {
            "front_rgb": front_rgb,
            "front_depth": front_depth,
            "front_point_cloud_world": front_point_cloud_world,
            "front_point_cloud_camera": front_point_cloud_camera,
            "front_camera_intrinsics": front_camera_intrinsics,
            "front_camera_extrinsics": front_camera_extrinsics,
        }
        observation_arrays = {
            key: value for key, value in observation_arrays.items() if value is not None
        }

        trajectory_arrays: dict[str, np.ndarray] = {
            "joint_positions": _stack_field(observations, "joint_positions"),
            "joint_velocities": _stack_field(observations, "joint_velocities"),
            "joint_forces": _stack_field(observations, "joint_forces"),
            "gripper_open": _stack_field(observations, "gripper_open"),
            "gripper_pose": _stack_field(observations, "gripper_pose"),
            "gripper_matrix": _stack_field(observations, "gripper_matrix"),
            "gripper_joint_positions": _stack_field(observations, "gripper_joint_positions"),
            "gripper_touch_forces": _stack_field(observations, "gripper_touch_forces"),
            "task_low_dim_state": _stack_field(observations, "task_low_dim_state"),
            "joint_position_action": _stack_optional_misc(observations, "joint_position_action"),
        }
        trajectory_arrays = {
            key: value for key, value in trajectory_arrays.items() if value is not None
        }

        observation_path = episode_dir / "observation.npz"
        trajectory_path = episode_dir / "trajectory.npz"
        metadata_path = episode_dir / "metadata.json"

        np.savez_compressed(observation_path, **observation_arrays)
        np.savez_compressed(trajectory_path, **trajectory_arrays)

        record = EpisodeRecord(
            dataset_version=dataset_root.name,
            task_name=task_name,
            episode_id=episode_id,
            variation_id=variation_id,
            seed=split_seed + next_episode_index + offset,
            split="",
            success=True,
            num_observations=len(observations),
            snapshot_policy=snapshot_policy,
            coordinate_frame="world",
            source_kind=spec.source_kind,
            source_root=source_root,
            language_descriptions=tuple(language_descriptions),
            camera_names=("front",),
            observation_path=str(observation_path.relative_to(dataset_root)),
            trajectory_path=str(trajectory_path.relative_to(dataset_root)),
            metadata_path=str(metadata_path.relative_to(dataset_root)),
            observation_sha256=_sha256_file(observation_path),
            trajectory_sha256=_sha256_file(trajectory_path),
            metadata_sha256="",
            observation_shapes={key: summarize_array(value)["shape"] for key, value in observation_arrays.items()},
            trajectory_shapes={key: summarize_array(value)["shape"] for key, value in trajectory_arrays.items()},
            observation_dtypes={key: summarize_array(value)["dtype"] for key, value in observation_arrays.items()},
            trajectory_dtypes={key: summarize_array(value)["dtype"] for key, value in trajectory_arrays.items()},
            action_sequence_available="joint_position_action" in trajectory_arrays,
        )

        metadata_payload = {
            "episode": record.to_dict(),
            "task_name": task_name,
            "source_episode_directory": str(source_episode_directory)
            if source_episode_directory is not None
            else None,
            "camera_metadata": {
                "front_camera_intrinsics": front_camera_intrinsics,
                "front_camera_extrinsics": front_camera_extrinsics,
            },
            "environment": env_info.to_dict(),
            "language_descriptions": list(language_descriptions),
            "coordinate_frames": {
                "point_cloud": "world",
                "camera": "camera",
            },
            "geometry": {
                "front_point_cloud_world_shape": list(front_point_cloud_world.shape)
                if front_point_cloud_world is not None
                else None,
                "front_point_cloud_camera_shape": list(front_point_cloud_camera.shape)
                if front_point_cloud_camera is not None
                else None,
            },
            "snapshot_policy": snapshot_policy,
            "target_crop": {
                "enabled": target_crop_enabled,
                "size": target_crop_size,
                "margin": target_crop_margin,
            },
            "xyz_only": xyz_only,
            "generated_at_utc": _utc_now(),
        }
        pending_metadata.append((metadata_path, metadata_payload, episode_id))
        episode_records.append(record)

    split = _build_split([record.episode_id for record in episode_records], split_seed)
    if existing_rows:
        split = {
            "train": list(existing_split.get("train", [])) + list(split.get("train", [])),
            "val": list(existing_split.get("val", [])) + list(split.get("val", [])),
            "test": list(existing_split.get("test", [])) + list(split.get("test", [])),
        }
    final_records: list[EpisodeRecord] = []
    for episode_record in episode_records:
        split_name = next(
            (name for name, split_ids in split.items() if episode_record.episode_id in split_ids),
            "",
        )
        final_records.append(replace(episode_record, split=split_name))

    metadata_lookup = {episode_id: (path, payload) for path, payload, episode_id in pending_metadata}
    finalized_records: list[EpisodeRecord] = []
    for record in final_records:
        metadata_path, metadata_payload = metadata_lookup[record.episode_id]
        metadata_payload["episode"] = record.to_dict()
        _write_json(metadata_path, metadata_payload)
        finalized_record = replace(record, metadata_sha256=_sha256_file(metadata_path))
        finalized_records.append(finalized_record)

    manifest_rows = existing_rows + [record.to_dict() for record in finalized_records]
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(manifest_path, index=False)

    unique_task_names = sorted({row["task_name"] for row in manifest_rows})
    unique_source_kinds = sorted({row["source_kind"] for row in manifest_rows})
    unique_source_roots = sorted({row["source_root"] for row in manifest_rows})
    dataset_task_name_value = (
        dataset_task_name
        if dataset_task_name is not None
        else unique_task_names[0]
        if len(unique_task_names) == 1
        else "mixed"
    )
    dataset_source_kind = unique_source_kinds[0] if len(unique_source_kinds) == 1 else "mixed"
    dataset_source_root = unique_source_roots[0] if len(unique_source_roots) == 1 else "mixed"
    source_description = (
        "RLBench demos collected for " + ", ".join(unique_task_names)
        if len(unique_task_names) == 1
        else "RLBench demos collected for multiple tasks"
    )

    metadata = DatasetMetadata(
        dataset_version=dataset_root.name,
        task_name=dataset_task_name_value,
        split_seed=split_seed,
        num_requested_episodes=len(spec_list),
        num_exported_episodes=len(episode_records),
        source_kind=dataset_source_kind,
        source_root=dataset_source_root,
        source_description=source_description,
        coordinate_frame="world",
        camera_names=("front",),
        image_size=(128, 128),
        snapshot_policy=snapshot_policy,
        point_cloud_enabled=True,
        target_crop_enabled=target_crop_enabled,
        target_crop_size=target_crop_size,
        target_crop_margin=target_crop_margin,
        xyz_only=xyz_only,
        rlbench_version=None,
        coppeliasim_version=None,
        generated_at_utc=_utc_now(),
        generator=generator,
        extra={
            "env_info": env_info.to_dict(),
            "task_names": unique_task_names,
            "source_kinds": unique_source_kinds,
            "source_roots": unique_source_roots,
        },
    )

    metadata_path = dataset_root / "dataset_metadata.json"
    _write_json(metadata_path, metadata.to_dict())

    _write_json(
        split_path,
        {
            "dataset_version": dataset_root.name,
            "task_name": dataset_task_name_value,
            "split_seed": split_seed,
            "splits": split,
        },
    )

    return DatasetBuildResult(
        dataset_root=str(dataset_root),
        metadata_path=str(metadata_path),
        manifest_path=str(manifest_path),
        split_path=str(split_path),
        num_exported_episodes=len(finalized_records),
        exported_episode_ids=tuple(record.episode_id for record in finalized_records),
    )


def export_reach_target_dataset_from_demos(
    dataset_root: Path,
    demos: Sequence[Any],
    split_seed: int,
    overwrite: bool = False,
    snapshot_policy: str = "initial",
    target_crop_enabled: bool = True,
    target_crop_size: int | None = 256,
    target_crop_margin: float | None = 0.1,
    xyz_only: bool = True,
    *,
    source_kind: str = "saved_demo",
    source_root: str = "",
    source_episode_directories: Sequence[Path | str | None] | None = None,
    episode_source_kinds: Sequence[str] | None = None,
    episode_source_roots: Sequence[str] | None = None,
) -> DatasetBuildResult:
    """Export a small ReachTarget dataset into the Phase 1 on-disk layout."""

    normalized_episode_dirs = _normalize_episode_directories(source_episode_directories)
    episode_source_kinds = list(episode_source_kinds or [])
    episode_source_roots = list(episode_source_roots or [])

    specs = []
    for index, demo in enumerate(demos):
        specs.append(
            RLBenchDemoSpec(
                task_name="reach_target",
                demo=demo,
                source_kind=episode_source_kinds[index] if index < len(episode_source_kinds) else source_kind,
                source_root=episode_source_roots[index] if index < len(episode_source_roots) else source_root,
                source_episode_directory=(
                    normalized_episode_dirs[index] if index < len(normalized_episode_dirs) else None
                ),
            )
        )

    return export_rlbench_dataset_from_specs(
        dataset_root,
        specs,
        split_seed,
        overwrite=overwrite,
        snapshot_policy=snapshot_policy,
        target_crop_enabled=target_crop_enabled,
        target_crop_size=target_crop_size,
        target_crop_margin=target_crop_margin,
        xyz_only=xyz_only,
        generator="scripts/build_reach_target_dataset.py",
        dataset_task_name="reach_target",
    )


def export_reach_target_dataset(
    dataset_root: Path,
    num_episodes: int,
    split_seed: int,
    source_root: Path | None = None,
    overwrite: bool = False,
    snapshot_policy: str = "initial",
    target_crop_enabled: bool = True,
    target_crop_size: int | None = 256,
    target_crop_margin: float | None = 0.1,
    xyz_only: bool = True,
) -> DatasetBuildResult:
    """Export a small ReachTarget dataset into the Phase 1 on-disk layout."""

    demos, resolved_source_root, source_episode_directories = load_saved_task_demo_batch(
        task_name="reach_target",
        amount=num_episodes,
        image_paths=False,
        variation_number=0,
        from_episode_number=0,
        search_roots=[source_root] if source_root is not None else None,
    )
    return export_reach_target_dataset_from_demos(
        dataset_root,
        demos,
        split_seed,
        overwrite=overwrite,
        snapshot_policy=snapshot_policy,
        target_crop_enabled=target_crop_enabled,
        target_crop_size=target_crop_size,
        target_crop_margin=target_crop_margin,
        xyz_only=xyz_only,
        source_kind="saved_demo",
        source_root=str(resolved_source_root),
        source_episode_directories=source_episode_directories,
    )
