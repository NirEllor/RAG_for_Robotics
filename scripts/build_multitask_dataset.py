#!/usr/bin/env python
"""Build a multi-task RLBench dataset from saved demonstrations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.data.exporter import RLBenchDemoSpec, export_rlbench_dataset_from_specs
from action_retrieval.simulation.raw_rlbench_importer import load_raw_rlbench_task_demo_batch
from action_retrieval.simulation.saved_importer import load_saved_task_demo_batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset" / "rlbench_multitask.yaml",
        help="Dataset config YAML to use as defaults.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset output root.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override the split seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing dataset root.",
    )
    return parser.parse_args()


def _task_cfg_value(task_cfg, key: str, default=None):
    if hasattr(task_cfg, key):
        value = getattr(task_cfg, key)
        return default if value is None else value
    if isinstance(task_cfg, dict):
        return task_cfg.get(key, default)
    return default


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)

    dataset_root = args.dataset_root or PROJECT_ROOT / cfg.data_root
    split_seed = args.split_seed if args.split_seed is not None else int(cfg.split_seed)
    dataset_task_name = str(getattr(cfg, "dataset_task_name", "mixed"))
    generator = "scripts/build_multitask_dataset.py"
    default_source_root = getattr(cfg, "source_root", None)
    default_source_split = str(getattr(cfg, "source_split", "train"))
    default_source_layout = str(getattr(cfg, "source_layout", "rlbench_saved"))

    task_specs = list(getattr(cfg, "tasks", []))
    if not task_specs:
        raise ValueError("Config must define at least one task under `tasks`.")

    _log("=" * 80)
    _log("Phase 2: Multitask dataset export")
    _log("=" * 80)
    _log(f"Config: {args.config}")
    _log(f"Dataset root: {dataset_root}")
    _log(f"Split seed: {split_seed}")
    _log(f"Tasks configured: {len(task_specs)}")

    specs: list[RLBenchDemoSpec] = []
    summary_rows: list[tuple[str, int, int]] = []

    for task_index, task_cfg in enumerate(task_specs, start=1):
        task_name = str(_task_cfg_value(task_cfg, "task_name"))
        requested = int(_task_cfg_value(task_cfg, "num_episodes", 0))
        variation_id = int(_task_cfg_value(task_cfg, "variation_id", 0))
        from_episode_number = int(_task_cfg_value(task_cfg, "from_episode_number", 0))
        source_root_value = _task_cfg_value(task_cfg, "source_root", default_source_root)
        source_roots = [Path(source_root_value)] if source_root_value else None
        source_kind = str(_task_cfg_value(task_cfg, "source_kind", "saved_demo"))
        required = bool(_task_cfg_value(task_cfg, "required", False))
        source_layout = str(_task_cfg_value(task_cfg, "source_layout", default_source_layout))
        source_split = str(_task_cfg_value(task_cfg, "source_split", default_source_split))

        _log(
            f"[task {task_index}/{len(task_specs)}] {task_name} "
            f"(layout={source_layout}, kind={source_kind}, requested={requested}, "
            f"variation={variation_id}, from_episode={from_episode_number}, required={required})"
        )
        if source_roots:
            _log(f"  source_roots: {[str(root) for root in source_roots]}")
        else:
            _log("  source_roots: <default search roots>")

        try:
            if source_layout == "peract_raw":
                demos, resolved_source_root, source_episode_dirs = load_raw_rlbench_task_demo_batch(
                    task_name=task_name,
                    amount=requested,
                    image_paths=False,
                    split_name=source_split,
                    from_episode_number=from_episode_number,
                    search_roots=source_roots,
                )
            else:
                demos, resolved_source_root, source_episode_dirs = load_saved_task_demo_batch(
                    task_name=task_name,
                    amount=requested,
                    image_paths=False,
                    variation_number=variation_id,
                    from_episode_number=from_episode_number,
                    search_roots=source_roots,
                )
            _log(
                f"  resolved_source_root: {resolved_source_root}"
            )
            _log(
                f"  loaded_demos: {len(demos)} "
                f"(episodes={len(source_episode_dirs)})"
            )
        except Exception as exc:
            if required:
                raise
            _log(f"WARNING: Skipping task {task_name}: {exc}")
            summary_rows.append((task_name, requested, 0))
            continue

        for index, demo in enumerate(demos):
            specs.append(
                RLBenchDemoSpec(
                    task_name=task_name,
                    demo=demo,
                    source_kind=source_kind,
                    source_root=str(resolved_source_root),
                    source_episode_directory=(
                        source_episode_dirs[index] if index < len(source_episode_dirs) else None
                    ),
                    variation_id=variation_id,
                )
            )
        summary_rows.append((task_name, requested, len(demos)))
        _log(f"  queued_for_export: {len(demos)}")

    _log(f"Collected specs: {len(specs)}")

    if not specs:
        raise RuntimeError("No demos were collected from the configured tasks.")

    _log("Starting export to disk...")
    result = export_rlbench_dataset_from_specs(
        dataset_root=dataset_root,
        specs=specs,
        split_seed=split_seed,
        overwrite=args.overwrite,
        snapshot_policy=str(cfg.snapshot_policy),
        target_crop_enabled=bool(cfg.point_cloud.target_crop),
        target_crop_size=int(cfg.point_cloud.crop_size),
        target_crop_margin=float(cfg.point_cloud.crop_margin),
        xyz_only=bool(cfg.point_cloud.xyz_only),
        generator=generator,
        dataset_task_name=dataset_task_name,
    )

    _log("=" * 80)
    _log("Phase 2: Multitask dataset export complete")
    _log("=" * 80)
    _log(f"Dataset root: {result.dataset_root}")
    _log(f"Manifest: {result.manifest_path}")
    _log(f"Metadata: {result.metadata_path}")
    _log(f"Splits: {result.split_path}")
    _log(f"Episodes exported: {result.num_exported_episodes}")
    for task_name, requested, exported in summary_rows:
        _log(f"  {task_name}: requested={requested}, exported={exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
