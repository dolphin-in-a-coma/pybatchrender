"""Renderer for Asteroids inspired environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class AsteroidsRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        self.n_ast = int(getattr(self.cfg, "num_asteroids", 6))
        n = int(self.cfg.num_scenes)

        self.ship = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.10, 0.10, 0.06), instances_per_scene=1, shared_across_scenes=False)
        self.asts = self.add_node("models/smiley", model_scale=(0.10, 0.10, 0.08), model_scale_units="absolute", instances_per_scene=self.n_ast, shared_across_scenes=False)

        self.ship_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.ast_pos = torch.zeros((n, self.n_ast, 3), dtype=torch.float32)
        self.ship.set_colors(torch.tensor([[[0.6, 1.0, 0.9, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.asts.set_colors(torch.tensor([[[0.8, 0.8, 0.8, 1.0]]], dtype=torch.float32).repeat(n, self.n_ast, 1))

        self.add_camera(fov_y_deg=45.0)
        self._pbr_cam.set_positions(torch.tensor([0.0, -3.2, 0.0], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.add_light(ambient=(0.35, 0.35, 0.35))
        self.setup_environment()

    def _step(self, state_batch=None):
        if state_batch is None:
            return
        st = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        self.ship_pos[:, :, 0] = st[:, 0:1]
        self.ship_pos[:, :, 1] = st[:, 1:2]
        self.ast_pos[:, :, 0:2] = st[:, 2:2 + 2 * self.n_ast].reshape(-1, self.n_ast, 2)
        self.ship.set_positions(self.ship_pos)
        self.asts.set_positions(self.ast_pos)
