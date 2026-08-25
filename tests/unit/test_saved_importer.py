"""Unit tests for saved RLBench demo discovery."""

from pathlib import Path

from action_retrieval.simulation.saved_importer import (
    resolve_saved_demo_root,
    resolve_saved_task_demo_root,
)


def test_resolve_saved_demo_root_from_tasks_dir(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    (tasks_root / "reach_target" / "variation0").mkdir(parents=True)

    resolved = resolve_saved_demo_root([tasks_root])

    assert resolved == tasks_root


def test_resolve_saved_demo_root_from_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "saved_dataset"
    (dataset_root / "tasks" / "reach_target" / "variation0").mkdir(parents=True)

    resolved = resolve_saved_demo_root([dataset_root])

    assert resolved == dataset_root / "tasks"


def test_resolve_saved_demo_root_missing(tmp_path: Path):
    resolved = resolve_saved_demo_root([tmp_path / "empty"])

    assert resolved is None


def test_resolve_saved_task_demo_root_from_tasks_dir(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    (tasks_root / "push_buttons" / "variation0").mkdir(parents=True)

    resolved = resolve_saved_task_demo_root("push_buttons", [tasks_root])

    assert resolved == tasks_root
