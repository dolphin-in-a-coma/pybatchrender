"""Renderer for Breakout environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class BreakoutRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        self.rows = int(getattr(self.cfg, "brick_rows", 4))
        self.cols = int(getattr(self.cfg, "brick_cols", 8))
        self.n_bricks = self.rows * self.cols
        n = int(self.cfg.num_scenes)

        self.paddle = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.35, 0.08, 0.06), instances_per_scene=1, shared_across_scenes=False)
        self.ball = self.add_node("models/smiley", model_scale=(0.08, 0.08, 0.08), model_scale_units="absolute", instances_per_scene=1, shared_across_scenes=False)
        self.bricks = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.22, 0.10, 0.05), instances_per_scene=self.n_bricks, shared_across_scenes=False)

        self.paddle_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.ball_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.brick_pos = torch.zeros((n, self.n_bricks, 3), dtype=torch.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                i = r * self.cols + c
                self.brick_pos[:, i, 0] = -0.9 + c * (1.8 / max(1, self.cols - 1))
                self.brick_pos[:, i, 1] = 0.35 + r * 0.18
        self.bricks.set_positions(self.brick_pos)

        self.paddle.set_colors(torch.tensor([[[0.9, 0.9, 0.2, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.ball.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self._brick_colors = torch.zeros((n, self.n_bricks, 4), dtype=torch.float32)
        self._brick_colors[:, :, 3] = 1.0
        for i in range(self.n_bricks):
            self._brick_colors[:, i, :3] = torch.tensor([0.2 + (i % self.cols) / max(1, self.cols), 0.4, 1.0 - (i % self.cols) / max(1, self.cols)])
        self.bricks.set_colors(self._brick_colors)

        self.add_camera(fov_y_deg=45.0)
        self._pbr_cam.set_positions(torch.tensor([0.0, -3.0, 0.0], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.add_light(ambient=(0.35, 0.35, 0.35))
        self.setup_environment()

    def _step(self, state_batch=None):
        if state_batch is None:
            return
        st = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        self.paddle_pos[:, :, 0] = st[:, 0:1]
        self.paddle_pos[:, :, 1] = -0.9
        self.ball_pos[:, :, 0] = st[:, 1:2]
        self.ball_pos[:, :, 1] = st[:, 2:3]
        self.paddle.set_positions(self.paddle_pos)
        self.ball.set_positions(self.ball_pos)

        alive = st[:, 5:] > 0.5
        pos = self.brick_pos.clone()
        pos[:, :, 2] = torch.where(alive, pos[:, :, 2], torch.full_like(pos[:, :, 2], -50.0))
        self.bricks.set_positions(pos)
