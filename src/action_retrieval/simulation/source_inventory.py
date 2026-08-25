"""Filesystem inventory helpers for RLBench saved-demo sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class VariationInventory:
    variation_name: str
    episode_count: int
    episode_dirs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TaskInventory:
    task_name: str
    variation_count: int
    episode_count: int
    variations: tuple[VariationInventory, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SourceInventory:
    source_root: str
    tasks_root: str
    task_count: int
    variation_count: int
    episode_count: int
    tasks: tuple[TaskInventory, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_root": self.source_root,
            "tasks_root": self.tasks_root,
            "task_count": self.task_count,
            "variation_count": self.variation_count,
            "episode_count": self.episode_count,
            "tasks": [
                {
                    "task_name": task.task_name,
                    "variation_count": task.variation_count,
                    "episode_count": task.episode_count,
                    "variations": [
                        {
                            "variation_name": variation.variation_name,
                            "episode_count": variation.episode_count,
                            "episode_dirs": list(variation.episode_dirs),
                        }
                        for variation in task.variations
                    ],
                }
                for task in self.tasks
            ],
        }


def resolve_tasks_root(source_root: Path) -> Path | None:
    """Resolve a RLBench tasks root from either a dataset root or a tasks dir."""

    source_root = Path(source_root)
    if (source_root / "reach_target" / "variation0" / "variation_descriptions.pkl").exists():
        return source_root
    if (source_root / "tasks" / "reach_target" / "variation0" / "variation_descriptions.pkl").exists():
        return source_root / "tasks"
    if (source_root / "variation0" / "variation_descriptions.pkl").exists():
        return source_root
    return None


def discover_saved_demo_roots(search_dirs: Iterable[Path]) -> list[Path]:
    """Recursively discover candidate RLBench saved-demo roots under directories."""

    discovered: list[Path] = []
    seen: set[Path] = set()
    for search_dir in search_dirs:
        search_dir = Path(search_dir)
        if not search_dir.exists() or not search_dir.is_dir():
            continue
        for variation_file in search_dir.rglob("variation_descriptions.pkl"):
            if len(variation_file.parents) < 3:
                continue
            candidate_root = variation_file.parents[2]
            resolved = resolve_tasks_root(candidate_root)
            if resolved is None:
                continue
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(resolved)
    return discovered


def _count_episode_dirs(variation_root: Path) -> tuple[int, tuple[str, ...]]:
    episodes_root = variation_root / "episodes"
    if not episodes_root.exists():
        return 0, tuple()
    episode_dirs = [
        entry.name
        for entry in sorted(episodes_root.iterdir())
        if entry.is_dir() and (entry / "low_dim_obs.pkl").exists()
    ]
    return len(episode_dirs), tuple(episode_dirs)


def audit_saved_demo_source(source_root: Path) -> SourceInventory:
    """Scan a saved-demo source and summarize task/variation coverage."""

    source_root = Path(source_root)
    tasks_root = resolve_tasks_root(source_root)
    if tasks_root is None:
        raise FileNotFoundError(
            f"Could not resolve a RLBench tasks root from {source_root}. "
            "Expected reach_target/variation0/variation_descriptions.pkl."
        )

    tasks: list[TaskInventory] = []
    task_dirs = [entry for entry in sorted(tasks_root.iterdir()) if entry.is_dir()]
    total_variations = 0
    total_episodes = 0

    for task_dir in task_dirs:
        variation_dirs = [
            entry
            for entry in sorted(task_dir.iterdir())
            if entry.is_dir() and entry.name.startswith("variation")
        ]
        variations: list[VariationInventory] = []
        task_episode_count = 0
        for variation_dir in variation_dirs:
            episode_count, episode_dirs = _count_episode_dirs(variation_dir)
            variations.append(
                VariationInventory(
                    variation_name=variation_dir.name,
                    episode_count=episode_count,
                    episode_dirs=episode_dirs,
                )
            )
            task_episode_count += episode_count
        if variations:
            total_variations += len(variations)
            total_episodes += task_episode_count
            tasks.append(
                TaskInventory(
                    task_name=task_dir.name,
                    variation_count=len(variations),
                    episode_count=task_episode_count,
                    variations=tuple(variations),
                )
            )

    return SourceInventory(
        source_root=str(source_root),
        tasks_root=str(tasks_root),
        task_count=len(tasks),
        variation_count=total_variations,
        episode_count=total_episodes,
        tasks=tuple(tasks),
    )


def audit_many_saved_demo_sources(source_roots: Iterable[Path]) -> list[SourceInventory]:
    """Audit multiple candidate source roots, skipping missing ones."""

    inventories: list[SourceInventory] = []
    for root in source_roots:
        root = Path(root)
        if not root.exists():
            continue
        try:
            inventories.append(audit_saved_demo_source(root))
        except FileNotFoundError:
            continue
    return inventories
