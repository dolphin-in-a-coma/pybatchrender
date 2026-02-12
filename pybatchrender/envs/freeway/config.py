"""Freeway config."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class FreewayConfig(PBRConfig):
    action_n: int | None = 3  # stay, up, down
    action_type: str = "discrete"
    max_steps: int = 600
    auto_reset: bool = True

    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (96, 96)
    offscreen: bool = True
    report_fps: bool = False

    seed: int = 0
    render: bool = True

    lanes: int = 6
    cars_per_lane: int = 2
    chicken_step: float = 0.10
    car_speed: float = 0.05

    worker_index: int = 0
    num_workers: int = 1
