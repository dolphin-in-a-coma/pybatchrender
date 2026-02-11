"""Pac-Man-like 2D environment configuration."""
from dataclasses import dataclass, field

from ...config import PBRConfig


@dataclass
class PacManConfig(PBRConfig):
    # TorchRL env defaults (obs dim is inferred at runtime from map/items)
    direct_obs_dim: int | None = None
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

    # Map / layout configurability
    map_width: int = 28
    map_height: int = 31
    wall_matrix: list[list[int]] | None = None
    border_matrix: list[list[int]] | None = None
    pellet_matrix: list[list[int]] | None = None
    power_pellet_matrix: list[list[int]] | None = None
    cherry_matrix: list[list[int]] | None = None

    # Fallback counts when matrices are not provided
    default_pellets: int = 144
    default_power_pellets: int = 4
    default_cherries: int = 2

    # Actor settings
    num_ghosts: int = 4
    pacman_start: tuple[float, float] | None = None
    ghost_starts: list[tuple[float, float]] = field(default_factory=list)

    # Allow positions between cells
    pacman_step_size: float = 0.35
    ghost_step_size: float = 0.30
    actor_radius: float = 0.22
    collect_radius: float = 0.36
    collision_radius: float = 0.30

    # Gameplay rewards
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
