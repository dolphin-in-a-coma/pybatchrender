"""Asteroids config."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class AsteroidsConfig(PBRConfig):
    action_n: int | None = 5  # stay, left, right, up, down
    action_type: str = "discrete"
    max_steps: int = 900
    auto_reset: bool = True

    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (84, 84)
    offscreen: bool = True
    report_fps: bool = False

    seed: int = 0
    render: bool = True

    # Visual style
    use_2d_objects: bool = True

    num_asteroids: int = 6
    ship_speed: float = 0.08
    asteroid_speed: float = 0.03

    worker_index: int = 0
    num_workers: int = 1
