"""PingPong environment configuration."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class PingPongConfig(PBRConfig):
    direct_obs_dim: int | None = 6
    action_n: int | None = 3  # stay, up, down
    action_type: str = "discrete"
    max_steps: int = 800
    auto_reset: bool = True

    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (84, 84)
    offscreen: bool = True
    report_fps: bool = False

    seed: int = 0
    render: bool = True

    # Visual style
    use_2d_objects: bool = True

    paddle_speed: float = 0.08
    paddle_height: float = 0.30
    ball_speed: float = 0.04
    opponent_speed: float = 0.06

    worker_index: int = 0
    num_workers: int = 1
