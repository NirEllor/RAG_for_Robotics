from __future__ import annotations

from pathlib import Path

from action_retrieval.simulation.source_inventory import (
    audit_saved_demo_source,
    discover_saved_demo_roots,
    resolve_tasks_root,
)


def _make_episode(root: Path, task_name: str, variation_name: str, episode_name: str) -> None:
    episode_dir = root / task_name / variation_name / "episodes" / episode_name
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "low_dim_obs.pkl").write_bytes(b"demo")


def test_resolve_tasks_root_from_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "saved_dataset"
    variation_file = dataset_root / "tasks" / "reach_target" / "variation0" / "variation_descriptions.pkl"
    variation_file.parent.mkdir(parents=True, exist_ok=True)
    variation_file.write_text("x", encoding="utf-8")

    assert resolve_tasks_root(dataset_root) == dataset_root / "tasks"


def test_audit_saved_demo_source_counts_tasks_variations_and_episodes(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    _make_episode(tasks_root, "reach_target", "variation0", "episode0")
    _make_episode(tasks_root, "reach_target", "variation0", "episode1")
    _make_episode(tasks_root, "push_button", "variation0", "episode0")
    _make_episode(tasks_root, "push_button", "variation1", "episode0")
    _make_episode(tasks_root, "push_button", "variation1", "episode1")

    (tasks_root / "reach_target" / "variation0" / "variation_descriptions.pkl").write_bytes(b"x")
    (tasks_root / "push_button" / "variation0" / "variation_descriptions.pkl").write_bytes(b"x")
    (tasks_root / "push_button" / "variation1" / "variation_descriptions.pkl").write_bytes(b"x")

    inventory = audit_saved_demo_source(tasks_root)

    assert inventory.task_count == 2
    assert inventory.variation_count == 3
    assert inventory.episode_count == 5
    assert {task.task_name for task in inventory.tasks} == {"reach_target", "push_button"}


def test_discover_saved_demo_roots_finds_nested_tasks_root(tmp_path: Path):
    search_dir = tmp_path / "downloads"
    variation_file = search_dir / "rlbench" / "tasks" / "reach_target" / "variation0" / "variation_descriptions.pkl"
    variation_file.parent.mkdir(parents=True, exist_ok=True)
    variation_file.write_text("x", encoding="utf-8")

    roots = discover_saved_demo_roots([search_dir])

    assert roots == [search_dir / "rlbench" / "tasks"]
