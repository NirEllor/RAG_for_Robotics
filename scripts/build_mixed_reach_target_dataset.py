#!/usr/bin/env python
"""Build a larger ReachTarget dataset from saved and live RLBench demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.data.exporter import export_reach_target_dataset_from_demos
from action_retrieval.simulation.reach_target_generation import collect_live_reach_target_demos
from action_retrieval.simulation.saved_importer import load_saved_reach_target_demo_batch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset" / "rlbench_reach_target.yaml",
        help="Dataset config YAML to use as defaults.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override the dataset output root.",
    )
    parser.add_argument(
        "--saved-count",
        type=int,
        default=None,
        help="Number of saved demos to export.",
    )
    parser.add_argument(
        "--live-count",
        type=int,
        default=0,
        help="Number of live demos to generate and export.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Override the saved-demo source root used for the saved batch.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override the split seed.",
    )
    parser.add_argument(
        "--variation-id",
        type=int,
        default=None,
        help="ReachTarget variation id for live generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing dataset root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = OmegaConf.load(args.config)

    dataset_root = args.dataset_root or PROJECT_ROOT / cfg.data_root
    split_seed = args.split_seed if args.split_seed is not None else int(cfg.split_seed)
    saved_count = args.saved_count if args.saved_count is not None else int(cfg.num_episodes)
    live_count = int(args.live_count)
    variation_id = args.variation_id if args.variation_id is not None else int(cfg.variation_id)

    if saved_count <= 0 and live_count <= 0:
        raise ValueError("At least one of --saved-count or --live-count must be positive.")

    demos = []
    episode_sources = []

    if saved_count > 0:
        saved_demos, saved_root, saved_episode_dirs = load_saved_reach_target_demo_batch(
            amount=saved_count,
            image_paths=False,
            variation_number=variation_id,
            from_episode_number=0,
            search_roots=[args.source_root] if args.source_root is not None else None,
        )
        demos.extend(saved_demos)
        for episode_dir in saved_episode_dirs:
            episode_sources.append(
                {
                    "source_kind": "saved_demo",
                    "source_root": str(saved_root),
                    "source_episode_directory": str(episode_dir),
                }
            )

    if live_count > 0:
        live_demos, live_sources = collect_live_reach_target_demos(
            live_count,
            variation_number=variation_id,
            image_paths=False,
        )
        demos.extend(live_demos)
        episode_sources.extend([source.__dict__ for source in live_sources])

    result = export_reach_target_dataset_from_demos(
        dataset_root=dataset_root,
        demos=demos,
        split_seed=split_seed,
        overwrite=args.overwrite,
        snapshot_policy=str(cfg.snapshot_policy),
        target_crop_enabled=bool(cfg.point_cloud.target_crop),
        target_crop_size=int(cfg.point_cloud.crop_size),
        target_crop_margin=float(cfg.point_cloud.crop_margin),
        xyz_only=bool(cfg.point_cloud.xyz_only),
        source_kind="mixed" if saved_count > 0 and live_count > 0 else "saved_demo" if saved_count > 0 else "live_demo",
        source_root="mixed" if saved_count > 0 and live_count > 0 else "saved" if saved_count > 0 else "live",
        source_episode_directories=[item.get("source_episode_directory") for item in episode_sources],
        episode_source_kinds=[item["source_kind"] for item in episode_sources],
        episode_source_roots=[item["source_root"] for item in episode_sources],
    )

    print("=" * 80)
    print("Phase 1+: Mixed ReachTarget dataset build")
    print("=" * 80)
    print(f"Dataset root: {result.dataset_root}")
    print(f"Saved demos: {saved_count}")
    print(f"Live demos: {live_count}")
    print(f"Episodes exported: {result.num_exported_episodes}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Splits: {result.split_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
