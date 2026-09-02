#!/usr/bin/env python3
"""Create report figures from locked evaluation CSV files.

The script is CPU-only and intentionally reads exported result files rather
than recomputing embeddings. Missing optional result directories are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_ROOT = Path("RAG_for_Robotics_outputs/evaluation")
DEFAULT_OUT = Path("report_figures")


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] missing: {path}")
        return None
    frame = pd.read_csv(path)
    print(f"[read] {path} ({len(frame)} rows)")
    return frame


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_retrieval(frame: pd.DataFrame, output: Path, title: str, name: str) -> None:
    data = frame[frame["k"] == 1].copy()
    data = data.sort_values("top1_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(data["method"], data["top1_accuracy"], color="#176b87")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Top-1 accuracy")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for index, value in enumerate(data["top1_accuracy"]):
        ax.text(value + 0.012, index, f"{value:.3f}", va="center", fontsize=9)
    _save(fig, output, name)


def plot_retrieval_by_k(frame: pd.DataFrame, output: Path, title: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method, group in frame.groupby("method", sort=True):
        group = group.sort_values("k")
        ax.plot(group["k"], group["hit_rate_at_k"], marker="o", label=method)
    ax.set_xticks(sorted(frame["k"].unique()))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("k")
    ax.set_ylabel("Hit rate@k")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, output, name)


def plot_robustness(frame: pd.DataFrame, output: Path) -> None:
    data = frame[frame["k"] == 1].copy()
    pivot = data.pivot(index="method", columns="condition", values="hit_rate_at_k")
    conditions = [column for column in ("clean", "viewpoint", "occlusion", "geometry_noise") if column in pivot]
    if "clean" not in pivot:
        clean = data.groupby("method")["hit_rate_at_k"].max().rename("clean")
        pivot = pivot.join(clean, how="left")
        conditions = ["clean"] + [column for column in conditions if column != "clean"]
    plot_data = pivot[conditions]
    ax = plot_data.plot(kind="bar", figsize=(10, 5.5), color=["#263238", "#e07a5f", "#3d8b8b", "#f2cc8f"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Hit rate@1")
    ax.set_title("Retrieval robustness under query perturbations")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Condition", fontsize=8)
    _save(ax.get_figure(), output, "robustness_hit_rate_at_1")


def plot_heldout(action: pd.DataFrame, baseline: pd.DataFrame, output: Path) -> None:
    rows = []
    for label, frame in (("Uni3D original", baseline), ("Uni3D action head", action)):
        row = frame[frame["k"] == 1].iloc[0]
        rows.append({"method": label, "top1_accuracy": row["top1_accuracy"]})
    data = pd.DataFrame(rows).sort_values("top1_accuracy")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(data["method"], data["top1_accuracy"], color=["#7a8b99", "#c85c5c"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Held-out test to train comparison")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(data["top1_accuracy"]):
        ax.text(index, value + 0.018, f"{value:.3f}", ha="center")
    _save(fig, output, "heldout_un3d_action_head")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = _read(args.evaluation_root / "retrieval_full/v2_multitask_full/summary_metrics.csv")
    subset = _read(args.evaluation_root / "retrieval_all/v2_multitask_subset8/summary_metrics.csv")
    robustness = _read(args.evaluation_root / "robustness/v2_multitask_subset8/summary_metrics.csv")
    action = _read(args.evaluation_root / "action_head/uni3d_subset8_heldout_final/summary_metrics.csv")
    baseline = _read(args.evaluation_root / "retrieval_heldout/uni3d_subset8/summary_metrics.csv")

    if full is not None:
        plot_retrieval(full, args.output_dir, "Full 19-task retrieval evaluation", "full_top1_by_method")
        plot_retrieval_by_k(full, args.output_dir, "Full 19-task hit rate across k", "full_hit_rate_by_k")
    if subset is not None:
        plot_retrieval(subset, args.output_dir, "Subset-8 retrieval evaluation", "subset8_top1_by_method")
    if robustness is not None:
        plot_robustness(robustness, args.output_dir)
    if action is not None and baseline is not None:
        plot_heldout(action, baseline, args.output_dir)

    print(f"Figures written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
