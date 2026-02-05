# -*- coding: utf-8 -*-
"""Configuration for the Steering RL environment."""

from dataclasses import dataclass, field
from typing import List

from pybatchrender import PBRConfig


@dataclass
class SteeringConfig(PBRConfig):
    """Configuration for the steering task environment.

    This extends PBRConfig with steering-specific parameters for:
    - Scene layout (lane, obstacles, player)
    - Physics (speed, steering, collisions)
    - RL training (rewards, termination)
    """

    # --- TorchRL env defaults ---
    direct_obs_dim: int = 6  # player_x, player_y, velocity, grace_active, next_obs_x, next_obs_y
    action_n: int = 1  # continuous steering
    action_type: str = "continuous"  # "discrete" or "continuous"
    action_low: float = -327.68  # min steering input (matches wheel range / 1000)
    action_high: float = 327.68  # max steering input
    max_steps: int = 10000
    auto_reset: bool = True

    # --- Rendering defaults ---
    num_channels: int = 3
    tile_resolution: tuple[int, int] = (64, 64)
    offscreen: bool = True
    report_fps: bool = False

    # --- Camera ---
    fov_y_deg: float = 40.0
    z_near: float = 0.5
    z_far: float = 1000.0
    camera_eye_offset: tuple[float, float, float] = (0.0, -16.3, 4.0)
    camera_forward_vector: tuple[float, float, float] = (0.0, 1.0, 0.0)

    # --- Lighting ---
    ambient_light: tuple[float, float, float] = (0.2, 0.2, 0.2)
    directional_light_dir: tuple[float, float, float] = (0.22, 0.44, 0.88)
    directional_light_color: tuple[float, float, float] = (0.8, 0.8, 0.8)

    # --- Colors ---
    red_obstacle_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    gold_obstacle_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)
    player_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    edge_color: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 1.0)
    background_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    crash_player_color: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 1.0)

    # --- Models ---
    player_model: str = "models/box"
    obstacle_sphere_model: str = "models/smiley"
    obstacle_cone_model: str = "models/cone.egg"
    border_model: str = "models/cylinder/scene.gltf"

    # --- Scene layout ---
    lane_width: float = 25.0
    obstacle_dimensions: tuple[float, float, float] = (2.0, 2.0, 2.0)
    player_dimensions: tuple[float, float, float] = (2.0, 2.0, 2.0)
    rail_dimensions: tuple[float, float, float] = (0.2, 1012.0, 0.2)
    rail_offset: tuple[float, float, float] = (0.0, -12.0, 0.5)
    track_length: float = 1000.0

    player_pivot_relative_point: tuple[float, float, float] = (0.5, 0.5, 0.5)
    obstacle_pivot_relative_point: tuple[float, float, float] = (0.5, 0.5, 0.5)
    rail_pivot_relative_point: tuple[float, float, float] = (0.5, 0.5, 0.0)

    # --- Obstacle generation ---
    obstacle_seed: int | None = None
    prob_gold: float = 0.2
    prob_cone: float = 0.5
    distance_to_first_obstacle: float = 200.0
    number_of_obstacles: int = 2000
    obstacle_y_spacing: float = 12.0
    obstacle_x_position: List[int] = field(
        default_factory=lambda: list(range(-11, 12, 2))
    )

    # --- Physics ---
    speed_initial: float = 96.0
    minimal_speed: float = 18.0
    speed_increment: float = 0.072
    speed_decrement: float = 6.12
    grace_period: float = 100.0 / 60.0  # ~1.67 seconds
    steering_speed: float = 20.0  # For keyboard/discrete control

    # --- RL training ---
    tau: float = 1.0 / 60.0  # Time step (seconds)
    wheel_sensitivity: float = 800.0  # Steering sensitivity (higher = less sensitive)
    collision_penalty: float = -1.0  # Sparse reward: only penalty on collision
    terminate_on_collision: bool = False
    terminate_on_track_end: bool = True

    # --- Saving/debug ---
    seed: int = 0
    render: bool = True
    save_every_steps: int = -1  # <0 disables, 0 saves every step, >0 saves every N steps
    save_examples_num: int = 16
    save_out_dir: str | None = None

    # --- Parallel env controls ---
    worker_index: int = 0
    num_workers: int = 1
