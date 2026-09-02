#!/usr/bin/env python
"""Build the Phase 1 ReachTarget dataset from saved RLBench demos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.data.exporter import export_reach_target_dataset
from action_retrieval.data.validator import validate_dataset_root


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dataset" / "rlbench_reach_target.yaml",
        help="Dataset config YAML to use.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the dataset output root.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Override the saved-demo source root.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override the requested episode count.",
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


def main() -> int:
    """Run the command-line entry point."""
    args = _parse_args()
    cfg = OmegaConf.load(args.config)

    output_root = args.output_root or PROJECT_ROOT / cfg.data_root
    num_episodes = args.num_episodes if args.num_episodes is not None else int(cfg.num_episodes)
    split_seed = args.split_seed if args.split_seed is not None else int(cfg.split_seed)

    result = export_reach_target_dataset(
        dataset_root=output_root,
        num_episodes=num_episodes,
        split_seed=split_seed,
        source_root=args.source_root,
        overwrite=args.overwrite,
        snapshot_policy=str(cfg.snapshot_policy),
        target_crop_enabled=bool(cfg.point_cloud.target_crop),
        target_crop_size=int(cfg.point_cloud.crop_size),
        target_crop_margin=float(cfg.point_cloud.crop_margin),
        xyz_only=bool(cfg.point_cloud.xyz_only),
    )

    validation = validate_dataset_root(output_root)
    print("=" * 80)
    print("Phase 1: ReachTarget dataset export")
    print("=" * 80)
    print(f"Dataset root: {result.dataset_root}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Splits: {result.split_path}")
    print(f"Episodes exported: {result.num_exported_episodes}")
    if validation.ok:
        print("Validation: PASSED")
        return 0

    print("Validation: FAILED")
    for error in validation.errors:
        print(f"  - {error}")
    for warning in validation.warnings:
        print(f"  - WARNING: {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
