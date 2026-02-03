"""Template environment configuration."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class MyEnvConfig(PBRConfig):
    """
    Configuration for MyEnv environment.
    
    Extend this with your environment-specific parameters.
    All PBRConfig parameters (num_scenes, tile_resolution, etc.) are inherited.
    
    Example parameters to add:
        - Physics parameters (gravity, mass, friction, etc.)
        - Episode parameters (max_steps, reward_scale, etc.)
        - Initial state ranges
    """
    # TorchRL env defaults
    direct_obs_dim: int | None = 4      # Dimension of observation vector
    action_n: int | None = 2            # Number of actions (discrete) or action dim (continuous)
    action_type: str = "discrete"       # "discrete" or "continuous"
    max_steps: int = 500                # Maximum episode length
    auto_reset: bool = True             # Auto-reset on termination

    # Rendering defaults (inherited from PBRConfig, override as needed)
    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (64, 64)
    offscreen: bool = True
    report_fps: bool = False

    # === ADD YOUR ENVIRONMENT-SPECIFIC PARAMETERS BELOW ===
    
    # Example physics parameters
    # gravity: float = 9.8
    # mass: float = 1.0
    
    # Example initial state ranges
    # init_position_range: tuple[float, float] = (-1.0, 1.0)
    # init_velocity_range: tuple[float, float] = (-0.5, 0.5)
    
    seed: int = 0
    render: bool = True
    
    # Parallel env controls
    worker_index: int = 0
    num_workers: int = 1
