"""Space Invaders config."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class SpaceInvadersConfig(PBRConfig):
    action_n: int | None = 4  # left, stay, right, fire
    action_type: str = "discrete"
    max_steps: int = 1200
    auto_reset: bool = True

    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (96, 96)
    offscreen: bool = True
    report_fps: bool = False

    seed: int = 0
    render: bool = True

    rows: int = 3
    cols: int = 6
    player_speed: float = 0.09
    bullet_speed: float = 0.12
    invader_speed: float = 0.025
    invader_drop: float = 0.05

    worker_index: int = 0
    num_workers: int = 1
