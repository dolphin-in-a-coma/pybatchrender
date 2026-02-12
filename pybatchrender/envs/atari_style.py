"""Shared visual style helpers for Atari-inspired environments.

These helpers mirror PacMan's top-down and flat 2D presentation.
"""
from __future__ import annotations

from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent.parent
_ASSET_ROOT = _ROOT / "assets" / "sprites"
_PLANE_MODEL = _ROOT / "models" / "plane.egg"


def sprite_path(name: str) -> str:
    return str((_ASSET_ROOT / name).resolve())


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


def plane_model_path() -> str:
    return str(_PLANE_MODEL.resolve())


def add_sprite_node(renderer, *, texture: str, instances_per_scene: int, scale_xy: tuple[float, float], z: float = 0.02):
    return renderer.add_node(
        plane_model_path(),
        texture=sprite_path(texture),
        model_scale=(float(scale_xy[0]), float(scale_xy[1]), float(z)),
        model_scale_units="absolute",
        model_hpr=(0.0, -90.0, 0.0),
        instances_per_scene=int(instances_per_scene),
        shared_across_scenes=False,
        sprite_transparency=True,
        depth_write=False,
        depth_test=True,
        bin_name="transparent",
        bin_sort=10,
    )
