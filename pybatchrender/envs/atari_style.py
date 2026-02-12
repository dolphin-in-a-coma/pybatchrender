"""Shared visual style helpers for Atari-inspired environments.

These helpers mirror PacMan's top-down and flat 2D presentation.
"""
from __future__ import annotations

import torch


def use_2d(cfg) -> bool:
    return bool(getattr(cfg, "use_2d_objects", True))


def topdown_camera_pose(arena_extent: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return (position, look_at, fov_y) for a stable top-down camera."""
    cam_z = max(3.0, 2.2 * float(arena_extent))
    pos = torch.tensor([0.0, -0.5, cam_z], dtype=torch.float32)
    look_at = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    fov_y = 38.0
    return pos, look_at, fov_y


def light_kwargs(flat_2d: bool) -> dict:
    if flat_2d:
        return {
            "ambient": (0.92, 0.92, 0.92),
            "dir_dir": (0.0, 0.0, -1.0),
            "dir_col": (0.25, 0.25, 0.25),
            "strength": 0.35,
        }
    return {"ambient": (0.35, 0.35, 0.35)}
