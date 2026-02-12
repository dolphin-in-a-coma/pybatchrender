"""Renderer for PingPong environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class PingPongRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        n = int(self.cfg.num_scenes)

        self.left = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.06, 0.30, 0.06), instances_per_scene=1, shared_across_scenes=False)
        self.right = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.06, 0.30, 0.06), instances_per_scene=1, shared_across_scenes=False)
        self.ball = self.add_node("models/smiley", model_scale=(0.08, 0.08, 0.08), model_scale_units="absolute", instances_per_scene=1, shared_across_scenes=False)

        self.left_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.right_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.ball_pos = torch.zeros((n, 1, 3), dtype=torch.float32)

        self.left_pos[:, :, 0] = -1.0
        self.right_pos[:, :, 0] = 1.0
        self.left.set_colors(torch.tensor([[[0.4, 0.9, 1.0, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.right.set_colors(torch.tensor([[[1.0, 0.6, 0.7, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.ball.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))

        self.add_camera(fov_y_deg=45.0)
        self._pbr_cam.set_positions(torch.tensor([0.0, -3.0, 0.0], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.add_light(ambient=(0.35, 0.35, 0.35))
        self.setup_environment()

    def _step(self, state_batch=None):
        if state_batch is None:
            return
        st = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        self.left_pos[:, :, 1] = st[:, 0:1]
        self.right_pos[:, :, 1] = st[:, 1:2]
        self.ball_pos[:, :, 0] = st[:, 2:3]
        self.ball_pos[:, :, 1] = st[:, 3:4]
        self.left.set_positions(self.left_pos)
        self.right.set_positions(self.right_pos)
        self.ball.set_positions(self.ball_pos)
