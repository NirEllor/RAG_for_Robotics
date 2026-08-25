#!/usr/bin/env python
"""Audit available RLBench saved-demo sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.simulation.saved_importer import candidate_saved_demo_roots
from action_retrieval.simulation.source_inventory import (
    audit_saved_demo_source,
    audit_many_saved_demo_sources,
    discover_saved_demo_roots,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Audit one explicit RLBench saved-demo root instead of the default candidates.",
    )
    parser.add_argument(
        "--search-dir",
        type=Path,
        action="append",
        default=[],
        help="Recursively search this directory for RLBench saved-demo roots. Repeatable.",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=15,
        help="Target episode count used in the summary message.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the audit summary as JSON.",
    )
    return parser.parse_args()


def _print_inventory(prefix: str, inventory) -> None:
    print(f"{prefix}source_root: {inventory.source_root}")
    print(f"{prefix}tasks_root: {inventory.tasks_root}")
    print(f"{prefix}tasks: {inventory.task_count}")
    print(f"{prefix}variations: {inventory.variation_count}")
    print(f"{prefix}episodes: {inventory.episode_count}")
    for task in inventory.tasks:
        print(f"{prefix}  - {task.task_name}: {task.episode_count} episodes across {task.variation_count} variations")
        for variation in task.variations:
            print(f"{prefix}    * {variation.variation_name}: {variation.episode_count} episodes")


def main() -> int:
    args = _parse_args()

    if args.source_root is not None:
        inventories = [audit_saved_demo_source(args.source_root)]
    else:
        candidate_roots = candidate_saved_demo_roots()
        candidate_roots.extend(discover_saved_demo_roots(args.search_dir))
        deduped_roots: list[Path] = []
        seen: set[Path] = set()
        for root in candidate_roots:
            resolved = Path(root).resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped_roots.append(root)
        inventories = audit_many_saved_demo_sources(deduped_roots)

    print("=" * 80)
    print("RLBench Saved-Demo Audit")
    print("=" * 80)

    if not inventories:
        print("No usable saved-demo source roots were found.")
        return 1

    for index, inventory in enumerate(inventories, start=1):
        print(f"[{index}]")
        _print_inventory("  ", inventory)
        print()

    best = max(inventories, key=lambda item: item.episode_count)
    enough_for_target = best.episode_count >= args.min_episodes
    print(f"Best source has {best.episode_count} episodes.")
    if enough_for_target:
        print(f"That is enough for the current target of {args.min_episodes} episodes.")
    else:
        print(
            f"That is not enough for the current target of {args.min_episodes} episodes; "
            "we should add more sources or move to live generation."
        )

    if args.json_output is not None:
        payload = {
            "min_episodes": args.min_episodes,
            "enough_for_target": enough_for_target,
            "best_source": best.to_dict(),
            "inventories": [inventory.to_dict() for inventory in inventories],
            "candidate_roots": [str(root) for root in candidate_saved_demo_roots()],
            "search_dirs": [str(path) for path in args.search_dir],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written to: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
