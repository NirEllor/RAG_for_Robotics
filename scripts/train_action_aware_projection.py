#!/usr/bin/env python3
"""Train a small trajectory-aware projection head on frozen episode embeddings.

The backbone is never updated. The target is a compact signature of the stored
action sequence, so this is an action-aware representation pilot rather than
full Uni3D/PTv3 fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from action_retrieval.retrieval.dataset import load_exported_episodes
from action_retrieval.retrieval.pipeline import embed_episodes


def _args() -> argparse.Namespace:
    """Implement the _args operation used by this module."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--encoder", default="uni3d")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _trajectory_signature(trajectory: dict[str, np.ndarray]) -> tuple[np.ndarray, str] | None:
    """Implement the _trajectory_signature operation used by this module."""
    for key in ("joint_position_action", "gripper_pose", "joint_positions"):
        value = trajectory.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.size == 0:
            continue
        flat = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(-1, 1)
        features = np.concatenate(
            [flat.mean(0), flat.std(0), flat.min(0), flat.max(0), flat[0], flat[-1]]
        )
        result = np.zeros(64, dtype=np.float32)
        result[: min(result.size, features.size)] = features[: result.size]
        source = "explicit_action" if key == "joint_position_action" else f"trajectory_state:{key}"
        return result, source
    return None


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int, target_dim: int):
        """Initialize this component."""
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.predict = nn.Linear(projection_dim, target_dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass through the component."""
        projected = nn.functional.normalize(self.project(value), dim=-1)
        return projected, self.predict(projected)


def main() -> int:
    """Run the command-line entry point."""
    args = _args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    episodes = load_exported_episodes(args.dataset_root)
    embeddings = embed_episodes(episodes, encoder_name=args.encoder, seed=args.seed)
    vectors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_sources: list[str] = []
    metadata: list[dict[str, str]] = []
    for episode, embedding in zip(episodes, embeddings):
        signature = _trajectory_signature(episode.trajectory)
        if signature is None:
            continue
        target, target_source = signature
        vectors.append(np.asarray(embedding.vector, dtype=np.float32))
        targets.append(target)
        target_sources.append(target_source)
        metadata.append({"episode_id": episode.episode_id, "task_name": episode.task_name, "split": episode.split})
    if len(vectors) < 2:
        raise RuntimeError("Fewer than two episodes contain a usable action trajectory.")

    x = np.stack(vectors)
    y = np.stack(targets)
    train_mask = np.asarray([item["split"] == "train" for item in metadata])
    if train_mask.sum() < 2:
        train_mask[:] = True
    x_mean, x_std = x[train_mask].mean(0), x[train_mask].std(0)
    y_mean, y_std = y[train_mask].mean(0), y[train_mask].std(0)
    x_std[x_std < 1e-6] = 1.0
    y_std[y_std < 1e-6] = 1.0
    x_tensor = torch.from_numpy((x - x_mean) / x_std).to(device)
    y_tensor = torch.from_numpy((y - y_mean) / y_std).to(device)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask)).to(device)

    model = ProjectionHead(x.shape[1], args.hidden_dim, args.projection_dim, y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    model.train()
    losses: list[float] = []
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        projected, predicted = model(x_tensor[train_indices])
        loss = nn.functional.mse_loss(predicted, y_tensor[train_indices])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"[epoch {epoch + 1}/{args.epochs}] loss={losses[-1]:.6f}", flush=True)

    model.eval()
    with torch.inference_mode():
        projected, _ = model(x_tensor)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": args.encoder,
            "model": model.state_dict(),
            "input_mean": x_mean,
            "input_std": x_std,
            "target_mean": y_mean,
            "target_std": y_std,
            "projection_dim": args.projection_dim,
        },
        output_dir / "projection_head.pt",
    )
    np.savez_compressed(
        output_dir / "projected_embeddings.npz",
        episode_id=np.asarray([item["episode_id"] for item in metadata]),
        embedding=projected.cpu().numpy().astype(np.float32),
    )
    payload = {
        "encoder": args.encoder,
        "backbone_frozen": True,
        "objective": "trajectory_signature_regression",
        "target_sources": sorted(set(target_sources)),
        "num_episodes": len(metadata),
        "num_train_episodes": int(train_mask.sum()),
        "projection_dim": args.projection_dim,
        "epochs": args.epochs,
        "final_train_loss": losses[-1],
        "device": str(device),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "summary_report.md").write_text(
        "# Action-Aware Projection Head\n\n"
        "This pilot freezes the pretrained 3D encoder and trains only a small projection head. "
        "The supervision target is a compact signature of the stored trajectory. "
        "Explicit actions are preferred; when unavailable, joint-position state trajectories are used as a clearly labeled proxy. "
        "It is not full backbone fine-tuning and does not establish simulator planning success.\n\n"
        f"- Encoder: `{args.encoder}`\n- Episodes: `{len(metadata)}`\n"
        f"- Train episodes: `{int(train_mask.sum())}`\n- Final train loss: `{losses[-1]:.6f}`\n",
        encoding="utf-8",
    )
    print(f"Projection outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
