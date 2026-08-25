from __future__ import annotations

from pathlib import Path

from action_retrieval.simulation.raw_rlbench_importer import resolve_raw_task_root


def test_resolve_raw_task_root_from_peract_train_layout(tmp_path: Path):
    root = tmp_path / "rlbench_raw"
    (root / "train" / "push_buttons" / "all_variations" / "episodes").mkdir(parents=True)

    resolved = resolve_raw_task_root("push_buttons", [root], split_name="train")

    assert resolved == root / "train" / "push_buttons" / "all_variations"


def test_resolve_raw_task_root_from_variation_layout(tmp_path: Path):
    root = tmp_path / "rlbench_raw"
    (root / "train" / "reach_target" / "variation0" / "episodes").mkdir(parents=True)

    resolved = resolve_raw_task_root("reach_target", [root], split_name="train")

    assert resolved == root / "train" / "reach_target" / "variation0"
