"""Coordinate-frame helpers for stored RLBench observations."""

from __future__ import annotations

import numpy as np


def _as_float32(array) -> np.ndarray:
    """Implement the _as_float32 operation used by this module."""
    return np.asarray(array, dtype=np.float32)


def world_to_camera(points, extrinsics) -> np.ndarray:
    """Convert world-frame points to camera-frame points.

    RLBench stores camera poses as 4x4 matrices in world coordinates. The
    inverse transform maps world coordinates into the camera frame.
    """

    pts = _as_float32(points)
    matrix = _as_float32(extrinsics)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected points with last dimension 3, got {pts.shape}")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    flat = pts.reshape(-1, 3)
    camera = (rotation.T @ (flat - translation).T).T
    return camera.reshape(pts.shape)


def camera_to_world(points, extrinsics) -> np.ndarray:
    """Convert camera-frame points to world-frame points."""

    pts = _as_float32(points)
    matrix = _as_float32(extrinsics)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected points with last dimension 3, got {pts.shape}")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    flat = pts.reshape(-1, 3)
    world = (rotation @ flat.T).T + translation
    return world.reshape(pts.shape)


def summarize_array(array) -> dict[str, object]:
    """Return compact shape/dtype summary for manifest bookkeeping."""

    arr = np.asarray(array)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}
