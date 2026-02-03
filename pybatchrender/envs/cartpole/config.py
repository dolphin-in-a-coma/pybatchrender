"""CartPole environment configuration."""
from dataclasses import dataclass

from ...config import PBRConfig


@dataclass
class CartPoleConfig(PBRConfig):
    """
    Configuration for the CartPole environment.
    
    Extends PBRConfig with CartPole-specific physics and initialization parameters.
    
    Physics Parameters:
        gravity: Gravitational acceleration (m/s²)
        masscart: Mass of the cart (kg)
        masspole: Mass of the pole (kg)
        length: Half-length of the pole (m)
        force_mag: Magnitude of force applied to cart (N)
        tau: Time step between updates (s)
        theta_threshold_deg: Angle at which episode terminates (degrees)
        x_threshold: Position at which episode terminates (m)
        
    Initialization Parameters:
        init_x_range: Range for initial cart position
        init_theta_range_deg: Range for initial pole angle (degrees)
        init_x_dot_range: Range for initial cart velocity
        init_theta_dot_range_deg: Range for initial angular velocity (degrees/s)
    """
    # TorchRL env defaults
    direct_obs_dim: int | None = 4
    action_n: int | None = 2
    action_type: str = "discrete"
    max_steps: int = 500
    auto_reset: bool = True

    # Rendering defaults
    num_channels: int = 3
    tile_resolution: tuple[int, int] | None = (64, 64)
    offscreen: bool = True
    report_fps: bool = False

    # CartPole physics
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5  # half-pole length
    force_mag: float = 10.0
    tau: float = 0.02  # seconds between updates
    theta_threshold_deg: float = 90.0
    x_threshold: float = 2.4

    seed: int = 0
    render: bool = True

    # Initial state ranges
    init_x_range: tuple[float, float] = (-2.0, 2.0)
    # Theta ranges are specified in DEGREES in config; converted to radians in env init
    init_theta_range_deg: tuple[float, float] = (-30.0, 30.0)
    init_x_dot_range: tuple[float, float] = (-1.0, 1.0)
    # Theta-dot range specified in DEGREES/SEC in config; converted to radians/sec in env init
    init_theta_dot_range_deg: tuple[float, float] = (-15.0, 15.0)

    # Saving controls
    save_every_steps: int = 50  # < 0 disables; 0 saves every step; >0 saves every N steps
    save_examples_num: int = 16
    save_out_dir: str | None = None

    # Parallel env controls
    worker_index: int = 0
    num_workers: int = 1
