# -*- coding: utf-8 -*-
"""Renderer for the Steering RL environment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pybatchrender import PBRRenderer

if TYPE_CHECKING:
    from .config import SteeringConfig

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class SteeringRenderer(PBRRenderer):
    """Pybatchrender-based renderer for the steering task.

    This renderer manages:
    - Player vehicle visualization
    - Obstacle spheres and cones
    - Lane border rails
    - Camera following the player

    Note: setup_environment() is called after build_obstacles() to ensure
    all nodes are created before lighting is configured.
    """

    def __init__(self, cfg: SteeringConfig | dict | None = None) -> None:
        super().__init__(cfg)

        num_scenes = int(self.cfg.num_scenes)
        self._num_scenes = num_scenes
        self._setup_called = False

        # Colors
        self._color_player = torch.tensor(self.cfg.player_color, dtype=torch.float32)
        self._color_red = torch.tensor(self.cfg.red_obstacle_color, dtype=torch.float32)
        self._color_gold = torch.tensor(self.cfg.gold_obstacle_color, dtype=torch.float32)
        self._color_border = torch.tensor(self.cfg.edge_color, dtype=torch.float32)
        self._color_crash = torch.tensor(
            getattr(self.cfg, "crash_player_color", self.cfg.player_color),
            dtype=torch.float32,
        )

        # Set background color
        bg = self.cfg.background_color
        self.set_background_color(float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3]))

        # Scene nodes
        self.sphere_node = None
        self.cone_node = None
        self.player_node = None
        self.left_border = None
        self.right_border = None

        # Positions buffers
        self._player_pos = torch.zeros((num_scenes, 1, 3), dtype=torch.float32)
        self._player_color_buf = self._color_player.unsqueeze(0).unsqueeze(0).expand(num_scenes, 1, -1).clone()
        self._left_border_pos = torch.zeros((num_scenes, 1, 3), dtype=torch.float32)
        self._right_border_pos = torch.zeros((num_scenes, 1, 3), dtype=torch.float32)

        # Camera offset
        self._camera_eye_offset = torch.tensor(
            self.cfg.camera_eye_offset, dtype=torch.float32
        ).reshape(1, 3)

        # Build static scene elements first
        self._build_player()
        self._build_borders()

        # Camera and lighting (but don't call setup_environment yet)
        self.add_camera(
            z_far=float(self.cfg.z_far),
            z_near=float(self.cfg.z_near),
            fov_y_deg=float(self.cfg.fov_y_deg),
        )
        self.add_light(
            ambient=self.cfg.ambient_light,
            dir_dir=self.cfg.directional_light_dir,
        )

        # NOTE: setup_environment() will be called after build_obstacles()

    def _resolve_model_path(self, model: str) -> str:
        """Resolve model path to absolute or Panda3D path.

        For models starting with "models/", check if they exist in the
        cogcarsim/models directory. If so, return absolute path.
        Otherwise return as-is for Panda3D built-in models.
        """
        if model.startswith("models/"):
            # Check if it's a custom model in cogcarsim/models
            rel_path = model[7:]  # Strip "models/" prefix
            custom_model = MODELS_DIR / rel_path
            if custom_model.exists():
                return str(custom_model.resolve())
            # Otherwise it's a Panda3D built-in model
            return model
        model_path = Path(model)
        if model_path.is_absolute():
            return str(model_path)
        return str((BASE_DIR / model_path).resolve())

    def _build_player(self) -> None:
        """Build the player node."""
        model_path = self._resolve_model_path(self.cfg.player_model)
        self.player_node = self.add_node(
            model_path,
            instances_per_scene=1,
            model_scale=self.cfg.player_dimensions,
            model_hpr=self.cfg.player_model_hpr,
            model_scale_units="absolute",
            model_pivot_relative_point=self.cfg.player_pivot_relative_point,
            shared_across_scenes=False,
        )
        self.player_node.set_positions(self._player_pos)
        self.player_node.set_colors(self._player_color_buf)

    def _build_borders(self) -> None:
        """Build the lane border rails."""
        border_model = self._resolve_model_path(self.cfg.border_model)
        border_scale = self.cfg.rail_dimensions

        self.left_border = self.add_node(
            border_model,
            instances_per_scene=1,
            model_scale=border_scale,
            model_hpr=self.cfg.border_model_hpr,
            model_scale_units="absolute",
            shared_across_scenes=False,
        )
        self.right_border = self.add_node(
            border_model,
            instances_per_scene=1,
            model_scale=border_scale,
            model_hpr=self.cfg.border_model_hpr,
            model_scale_units="absolute",
            shared_across_scenes=False,
        )

        lane_half_width = float(self.cfg.lane_width) * 0.5
        z_offset = float(self.cfg.rail_offset[2])

        self._left_border_pos[:, 0, 0] = -lane_half_width
        self._left_border_pos[:, 0, 2] = z_offset
        self._right_border_pos[:, 0, 0] = lane_half_width
        self._right_border_pos[:, 0, 2] = z_offset

        border_color = self._color_border.unsqueeze(0).unsqueeze(0).expand(
            self._num_scenes, 1, -1
        ).clone()

        self.left_border.set_positions(self._left_border_pos)
        self.right_border.set_positions(self._right_border_pos)
        self.left_border.set_colors(border_color)
        self.right_border.set_colors(border_color)

    def build_obstacles(self, obstacles: torch.Tensor) -> None:
        """Build obstacle nodes from obstacle tensor.

        Args:
            obstacles: Tensor of shape (batch, num_obstacles, 4) where each obstacle
                       has [x, y, gold_flag, cone_flag].

        This method also calls setup_environment() on first invocation.
        """
        # Destroy existing obstacle nodes if any
        for attr in ("sphere_node", "cone_node"):
            node = getattr(self, attr, None)
            if node is not None:
                try:
                    node.np.removeNode()
                except Exception:
                    pass
            setattr(self, attr, None)

        if obstacles is None or obstacles.numel() == 0:
            if not self._setup_called:
                self.setup_environment()
                self._setup_called = True
            return

        B = obstacles.shape[0]
        N = obstacles.shape[1]
        obs_cpu = obstacles.detach().cpu()

        # Per-scene positions: (B, N, 3)
        positions = torch.zeros(B, N, 3, dtype=torch.float32)
        positions[:, :, 0] = obs_cpu[:, :, 0]  # x
        positions[:, :, 1] = obs_cpu[:, :, 1]  # y

        # Per-scene colors: (B, N, 4) based on gold flag
        gold_mask = obs_cpu[:, :, 2] >= 0.5
        colors = self._color_red.unsqueeze(0).unsqueeze(0).expand(B, N, -1).clone()
        colors[gold_mask] = self._color_gold

        sphere_model = self._resolve_model_path(self.cfg.obstacle_sphere_model)
        self.sphere_node = self.add_node(
            sphere_model,
            instances_per_scene=N,
            model_scale=self.cfg.obstacle_dimensions,
            model_hpr=self.cfg.obstacle_model_hpr,
            model_scale_units="absolute",
            model_pivot_relative_point=self.cfg.obstacle_pivot_relative_point,
            shared_across_scenes=False,
        )
        self.sphere_node.set_positions(positions)
        self.sphere_node.set_colors(colors)

        # Call setup_environment on first obstacle build
        if not self._setup_called:
            self.setup_environment()
            self._setup_called = True

    def _step(self, state_batch: torch.Tensor | None = None) -> None:
        """Update renderer from environment state.

        Args:
            state_batch: Tensor of shape (batch, obs_dim) containing:
                - state[:, 0]: player_x
                - state[:, 1]: player_y
                - state[:, 2]: forward_speed
                - state[:, 3]: grace_active (0 or 1)
                Additional obs dimensions are ignored for rendering.
        """
        if state_batch is None:
            return

        state = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        B = int(state.shape[0])
        N = self._num_scenes

        # Pad/truncate to match num_scenes
        if B < N:
            pad = state[-1:].repeat(N - B, 1)
            state = torch.cat([state, pad], dim=0)
        elif B > N:
            state = state[:N]

        player_x = state[:, 0:1]
        player_y = state[:, 1:2]
        grace_active = state[:, 3:4] if state.shape[1] > 3 else torch.zeros(N, 1)

        # Update player position
        self._player_pos[:, 0, 0:1] = player_x
        self._player_pos[:, 0, 1:2] = player_y
        self.player_node.set_positions(self._player_pos)

        # Update player color based on grace period
        normal_color = self._color_player.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)
        crash_color = self._color_crash.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)
        grace_mask = grace_active.unsqueeze(-1) > 0.5
        self._player_color_buf = torch.where(grace_mask, crash_color, normal_color)
        self.player_node.set_colors(self._player_color_buf)

        # Update borders to follow player Y position
        self._left_border_pos[:, 0, 1:2] = player_y
        self._right_border_pos[:, 0, 1:2] = player_y
        self.left_border.set_positions(self._left_border_pos, lazy=True)
        self.right_border.set_positions(self._right_border_pos, lazy=True)

        # Update camera to follow player
        if self._pbr_cam is not None:
            cam_pos = self._player_pos[:, 0, :].clone()  # (num_scenes, 3)
            cam_pos += self._camera_eye_offset
            self._pbr_cam.set_positions(cam_pos)
