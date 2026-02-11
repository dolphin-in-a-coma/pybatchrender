"""Pac-Man-like 2D environment configuration."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class PacManConfig(PBRConfig):
    # TorchRL env defaults
    direct_obs_dim: int | None = 8
    action_n: int | None = 5  # stay, up, down, left, right
    action_type: str = "discrete"
    max_steps: int = 400
    auto_reset: bool = True

    # Rendering defaults
    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (96, 96)
    offscreen: bool = True
    report_fps: bool = False

    seed: int = 0
    render: bool = True

    # Gameplay
    powered_steps: int = 24
    step_penalty: float = -0.01
    reward_pellet: float = 1.0
    reward_cherry: float = 3.0
    reward_power_pill: float = 5.0
    reward_eat_ghost: float = 8.0
    reward_win: float = 20.0
    reward_lose: float = -10.0

    # Parallel env controls
    worker_index: int = 0
    num_workers: int = 1
