from __future__ import annotations

from pathlib import Path

from action_retrieval.evaluation.retrieval_eval import (
    evaluate_retrieval_methods,
    load_relevance_annotations,
)
from action_retrieval.evaluation.metrics import (
    average_precision_at_k,
    normalized_discounted_cumulative_gain,
)


def test_load_relevance_annotations(tmp_path: Path):
    annotations_path = tmp_path / "labels.json"
    annotations_path.write_text('{"episode0": ["episode1"]}', encoding="utf-8")

    annotations = load_relevance_annotations(annotations_path)

    assert annotations["episode0"] == {"episode1"}


def test_evaluate_retrieval_methods_returns_multiple_cutoffs():
    dataset_root = Path("data/processed/v1_reach_target")
    annotations = {
        "episode0": {"episode1"},
        "episode1": {"episode0"},
        "episode2": {"episode3"},
        "episode3": {"episode2"},
    }

    runs = evaluate_retrieval_methods(
        dataset_root,
        annotations,
        methods=["pose_descriptor"],
        ks=[1, 2],
    )

    assert {run.k for run in runs} == {1, 2}
    assert {run.method for run in runs} == {"pose_descriptor"}


def test_average_precision_and_ndcg_are_normalized():
    assert average_precision_at_k([1, 0, 1], total_relevant=3, k=3) == 0.5555555555555555
    assert normalized_discounted_cumulative_gain([1, 0], total_relevant=1, k=2) == 1.0
