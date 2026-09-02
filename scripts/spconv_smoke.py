#!/usr/bin/env python3
"""Minimal spconv smoke test for PTv3 debugging.

This intentionally exercises a small sparse convolution forward pass so we can
separate a generic spconv/CUDA runtime problem from a PTv3-specific issue.
"""

from __future__ import annotations

import argparse
import faulthandler
import os

import torch


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="Torch device to use.")
    parser.add_argument("--num-points", type=int, default=8, help="Number of sparse points.")
    parser.add_argument("--in-channels", type=int, default=4, help="Feature channels.")
    parser.add_argument("--out-channels", type=int, default=8, help="Output channels.")
    parser.add_argument("--spatial-size", type=int, default=8, help="Cubic sparse grid size.")
    return parser.parse_args()


def _make_sparse_indices(num_points: int, spatial_size: int, device: torch.device) -> torch.Tensor:
    """Implement the _make_sparse_indices operation used by this module."""
    coords = []
    for idx in range(num_points):
        x = idx % spatial_size
        y = (idx // spatial_size) % spatial_size
        z = (idx // (spatial_size * spatial_size)) % spatial_size
        coords.append([0, z, y, x])
    return torch.tensor(coords, dtype=torch.int32, device=device)


def main() -> int:
    """Run the command-line entry point."""
    faulthandler.enable()
    args = _parse_args()
    device = torch.device(args.device)

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"device: {device}")
    print(f"pid: {os.getpid()}")

    if device.type != "cuda":
        raise SystemExit("CUDA device required for this smoke test.")

    import spconv.pytorch as spconv

    torch.cuda.set_device(0)
    torch.manual_seed(7)

    indices = _make_sparse_indices(args.num_points, args.spatial_size, device)
    features = torch.randn((args.num_points, args.in_channels), device=device)
    print(f"[spconv] indices shape: {tuple(indices.shape)}")
    print(f"[spconv] features shape: {tuple(features.shape)}")

    sparse = spconv.SparseConvTensor(features, indices, [args.spatial_size] * 3, 1)
    conv = spconv.SubMConv3d(
        args.in_channels,
        args.out_channels,
        kernel_size=3,
        padding=1,
        bias=False,
    ).to(device)
    conv.eval()

    print("[spconv] running forward smoke test...", flush=True)
    with torch.inference_mode():
        output = conv(sparse)

    print(f"[spconv] output features shape: {tuple(output.features.shape)}")
    print("[spconv] smoke test complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
