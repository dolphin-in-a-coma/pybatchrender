"""Renderer for Space Invaders inspired environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class SpaceInvadersRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        self.rows = int(getattr(self.cfg, "rows", 3))
        self.cols = int(getattr(self.cfg, "cols", 6))
        self.n_inv = self.rows * self.cols
        n = int(self.cfg.num_scenes)

        self.player = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.20, 0.08, 0.06), instances_per_scene=1, shared_across_scenes=False)
        self.bullet = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.02, 0.06, 0.03), instances_per_scene=1, shared_across_scenes=False)
        self.invaders = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.12, 0.08, 0.05), instances_per_scene=self.n_inv, shared_across_scenes=False)

        self.player_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.bullet_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.inv_base = torch.zeros((n, self.n_inv, 3), dtype=torch.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                i = r * self.cols + c
                self.inv_base[:, i, 0] = -0.7 + c * (1.4 / max(1, self.cols - 1))
                self.inv_base[:, i, 1] = 0.75 - r * 0.18

        self.player.set_colors(torch.tensor([[[0.3, 1.0, 0.3, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.bullet.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.invaders.set_colors(torch.tensor([[[1.0, 0.2, 0.2, 1.0]]], dtype=torch.float32).repeat(n, self.n_inv, 1))

        self.add_camera(fov_y_deg=45.0)
        self._pbr_cam.set_positions(torch.tensor([0.0, -3.1, 0.0], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.add_light(ambient=(0.35, 0.35, 0.35))
        self.setup_environment()

    def _step(self, state_batch=None):
        if state_batch is None:
            return
        st = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        self.player_pos[:, :, 0] = st[:, 0:1]
        self.player_pos[:, :, 1] = -0.92
        self.player.set_positions(self.player_pos)

        bullet_active = st[:, 1:2] > 0.5
        self.bullet_pos[:, :, 0] = st[:, 2:3]
        self.bullet_pos[:, :, 1] = st[:, 3:4]
        self.bullet_pos[:, :, 2] = torch.where(bullet_active, torch.zeros_like(self.bullet_pos[:, :, 2]), torch.full_like(self.bullet_pos[:, :, 2], -50.0))
        self.bullet.set_positions(self.bullet_pos)

        pos = self.inv_base.clone()
        pos[:, :, 0] += st[:, 4:5]
        pos[:, :, 1] += st[:, 5:6]
        alive = st[:, 6:] > 0.5
        pos[:, :, 2] = torch.where(alive, pos[:, :, 2], torch.full_like(pos[:, :, 2], -50.0))
        self.invaders.set_positions(pos)
