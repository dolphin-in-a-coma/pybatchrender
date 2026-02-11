"""Renderer for Pac-Man-like 2D environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer
from .layout import build_spec


class PacManRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)

        self.spec = build_spec(self.cfg)
        self.layout = self.spec["layout"]
        self.pellet_cells = self.spec["pellet_cells"]
        self.power_cells = self.spec["power_cells"]
        self.cherry_cells = self.spec["cherry_cells"]

        self.setBackgroundColor(0.03, 0.08, 0.9, 1.0)

        self.cell = 1.0
        self.base_z = 0.15

        self.wall_cells = sorted(list(self.layout.walls), key=lambda p: (p[1], p[0]))
        self.track_cells = self.layout.walkable
        self.num_ghosts = int(self.cfg.num_ghosts)

        num_scenes = int(self.cfg.num_scenes)

        self.walls = self.add_node(
            "models/box",
            model_pivot_relative_point=(0.5, 0.5, 0.5),
            model_scale=(0.96, 0.96, 0.24),
            instances_per_scene=max(1, len(self.wall_cells)),
            shared_across_scenes=True,
        )
        self.tracks = self.add_node(
            "models/box",
            model_pivot_relative_point=(0.5, 0.5, 0.5),
            model_scale=(0.96, 0.96, 0.04),
            instances_per_scene=max(1, len(self.track_cells)),
            shared_across_scenes=True,
        )

        self.pacman = self.add_node(
            "models/smiley",
            model_scale=(0.56, 0.56, 0.11),
            model_scale_units="absolute",
            instances_per_scene=1,
            shared_across_scenes=False,
        )
        self.ghosts = self.add_node(
            "models/smiley",
            model_scale=(0.52, 0.52, 0.1),
            model_scale_units="absolute",
            instances_per_scene=max(1, self.num_ghosts),
            shared_across_scenes=False,
        )

        self.pellets = self.add_node(
            "models/smiley",
            model_scale=(0.14, 0.14, 0.05),
            model_scale_units="absolute",
            instances_per_scene=max(1, len(self.pellet_cells)),
            shared_across_scenes=False,
        )
        self.power_pellets = self.add_node(
            "models/smiley",
            model_scale=(0.30, 0.30, 0.07),
            model_scale_units="absolute",
            instances_per_scene=max(1, len(self.power_cells)),
            shared_across_scenes=False,
        )
        self.cherries = self.add_node(
            "models/smiley",
            model_scale=(0.24, 0.24, 0.07),
            model_scale_units="absolute",
            instances_per_scene=max(1, len(self.cherry_cells)),
            shared_across_scenes=False,
        )

        self._set_static_geometry()
        self._set_colors(num_scenes)

        self.add_camera(fov_y_deg=38.0)
        cam_z = max(12.0, 0.85 * max(self.layout.width, self.layout.height))
        self._pbr_cam.set_positions(torch.tensor([0.0, -0.5, cam_z], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.add_light(ambient=(0.32, 0.32, 0.35), dir_dir=(0.1, -0.2, -1.0), dir_col=(1.0, 1.0, 1.0), strength=1.0)

        self.setup_environment()

    def _set_static_geometry(self):
        wall_pos = torch.zeros((1, max(1, len(self.wall_cells)), 3), dtype=torch.float32)
        for i, (x, y) in enumerate(self.wall_cells):
            wall_pos[0, i] = torch.tensor([*self._to_world_xy(float(x), float(y)), 0.0])
        self.walls.set_positions(wall_pos)
        self.walls.set_colors(torch.tensor([[[0.05, 0.15, 1.0, 1.0]]], dtype=torch.float32).repeat(1, max(1, len(self.wall_cells)), 1))

        track_pos = torch.zeros((1, max(1, len(self.track_cells)), 3), dtype=torch.float32)
        for i, (x, y) in enumerate(self.track_cells):
            track_pos[0, i] = torch.tensor([*self._to_world_xy(float(x), float(y)), -0.02])
        self.tracks.set_positions(track_pos)
        self.tracks.set_colors(torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32).repeat(1, max(1, len(self.track_cells)), 1))

    def _set_colors(self, num_scenes: int):
        self.pacman.set_colors(torch.tensor([[[1.0, 0.9, 0.05, 1.0]]], dtype=torch.float32).repeat(num_scenes, 1, 1))

        base_cols = torch.tensor(
            [[1.0, 0.25, 0.25, 1.0], [1.0, 0.5, 0.9, 1.0], [0.2, 1.0, 1.0, 1.0], [1.0, 0.7, 0.05, 1.0]],
            dtype=torch.float32,
        )
        ghost_cols = torch.zeros((num_scenes, max(1, self.num_ghosts), 4), dtype=torch.float32)
        for g in range(max(1, self.num_ghosts)):
            ghost_cols[:, g, :] = base_cols[g % base_cols.shape[0]]
        self.ghosts.set_colors(ghost_cols)

        self.pellets.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(num_scenes, max(1, len(self.pellet_cells)), 1))
        self.power_pellets.set_colors(torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32).repeat(num_scenes, max(1, len(self.power_cells)), 1))
        self.cherries.set_colors(torch.tensor([[[1.0, 0.1, 0.1, 1.0]]], dtype=torch.float32).repeat(num_scenes, max(1, len(self.cherry_cells)), 1))

    def _to_world_xy(self, x: float, y: float) -> tuple[float, float]:
        cx = x - (self.layout.width - 1) / 2.0
        cy = (self.layout.height - 1) / 2.0 - y
        return cx * self.cell, cy * self.cell

    def _xy_list_to_world(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.dim() == 2:
            xy = xy.unsqueeze(1)
        B, N, _ = xy.shape
        out = torch.zeros((B, N, 3), dtype=torch.float32)
        for b in range(B):
            for n in range(N):
                x, y = float(xy[b, n, 0]), float(xy[b, n, 1])
                wx, wy = self._to_world_xy(x, y)
                out[b, n] = torch.tensor([wx, wy, self.base_z], dtype=torch.float32)
        return out

    def _hide_by_mask(self, pos: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        if pos.shape[1] == 0:
            return pos
        hidden = pos.clone()
        hidden[:, :, 2] = torch.where(alive_mask, hidden[:, :, 2], torch.full_like(hidden[:, :, 2], -50.0))
        return hidden

    def _step(self, state_batch=None):
        if state_batch is None:
            return

        pac_xy = torch.as_tensor(state_batch["pac_xy"], dtype=torch.float32).detach().cpu()
        ghosts_xy = torch.as_tensor(state_batch["ghosts_xy"], dtype=torch.float32).detach().cpu()

        B = int(self.cfg.num_scenes)
        self.pacman.set_positions(self._xy_list_to_world(pac_xy).reshape(B, 1, 3))
        self.ghosts.set_positions(self._xy_list_to_world(ghosts_xy))

        if len(self.pellet_cells) > 0:
            pellet_xy = torch.as_tensor(state_batch["pellet_xy"], dtype=torch.float32).detach().cpu()
            pellet_alive = torch.as_tensor(state_batch["pellet_alive"], dtype=torch.bool).detach().cpu()
            pellet_pos = self._xy_list_to_world(pellet_xy.unsqueeze(0).repeat(B, 1, 1))
            self.pellets.set_positions(self._hide_by_mask(pellet_pos, pellet_alive))

        if len(self.power_cells) > 0:
            power_xy = torch.as_tensor(state_batch["power_xy"], dtype=torch.float32).detach().cpu()
            power_alive = torch.as_tensor(state_batch["power_alive"], dtype=torch.bool).detach().cpu()
            power_pos = self._xy_list_to_world(power_xy.unsqueeze(0).repeat(B, 1, 1))
            self.power_pellets.set_positions(self._hide_by_mask(power_pos, power_alive))

        if len(self.cherry_cells) > 0:
            cherry_xy = torch.as_tensor(state_batch["cherry_xy"], dtype=torch.float32).detach().cpu()
            cherry_alive = torch.as_tensor(state_batch["cherry_alive"], dtype=torch.bool).detach().cpu()
            cherry_pos = self._xy_list_to_world(cherry_xy.unsqueeze(0).repeat(B, 1, 1))
            self.cherries.set_positions(self._hide_by_mask(cherry_pos, cherry_alive))
