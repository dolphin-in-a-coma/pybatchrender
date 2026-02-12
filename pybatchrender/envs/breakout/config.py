"""Breakout environment configuration."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class BreakoutConfig(PBRConfig):
    action_n: int | None = 3  # left, stay, right
    action_type: str = "discrete"
    max_steps: int = 1000
    auto_reset: bool = True

    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (96, 96)
    offscreen: bool = True
    report_fps: bool = False

    seed: int = 0
    render: bool = True

    brick_rows: int = 4
    brick_cols: int = 8
    paddle_speed: float = 0.10
    paddle_width: float = 0.35
    ball_speed: float = 0.045

    worker_index: int = 0
    num_workers: int = 1
