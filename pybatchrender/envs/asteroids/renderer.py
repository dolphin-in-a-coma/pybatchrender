"""Renderer for Asteroids inspired environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer
from ..atari_style import add_sprite_node, light_kwargs, topdown_camera_pose, use_2d


class AsteroidsRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        self.n_ast = int(getattr(self.cfg, "num_asteroids", 6))
        n = int(self.cfg.num_scenes)
        flat_2d = use_2d(self.cfg)

        if flat_2d:
            self.setBackgroundColor(0.0, 0.0, 0.0, 1.0)
            self.ship = add_sprite_node(self, texture="asteroid_ship.png", instances_per_scene=1, scale_xy=(0.10, 0.10))
            self.asts = add_sprite_node(self, texture="asteroid_rock.png", instances_per_scene=self.n_ast, scale_xy=(0.10, 0.10))
        else:
            self.ship = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.10, 0.10, 0.06), instances_per_scene=1, shared_across_scenes=False)
            self.asts = self.add_node("models/smiley", model_scale=(0.10, 0.10, 0.08), model_scale_units="absolute", instances_per_scene=self.n_ast, shared_across_scenes=False)

        self.ship_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.ast_pos = torch.zeros((n, self.n_ast, 3), dtype=torch.float32)
        self.ship_pos[:, :, 2] = 0.05
        self.ast_pos[:, :, 2] = 0.05
        self.ship.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.asts.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(n, self.n_ast, 1))

        cam_pos, cam_look, cam_fov = topdown_camera_pose(arena_extent=1.8)
        self.add_camera(fov_y_deg=cam_fov)
        self._pbr_cam.set_positions(cam_pos)
        self._pbr_cam.look_at(cam_look)
        self.add_light(**light_kwargs(flat_2d))
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
