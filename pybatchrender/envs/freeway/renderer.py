"""Renderer for Freeway inspired environment."""
from __future__ import annotations

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class FreewayRenderer(PBRRenderer):
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        self.lanes = int(getattr(self.cfg, "lanes", 6))
        self.cars_per_lane = int(getattr(self.cfg, "cars_per_lane", 2))
        self.n_cars = self.lanes * self.cars_per_lane
        n = int(self.cfg.num_scenes)

        self.chicken = self.add_node("models/smiley", model_scale=(0.10, 0.10, 0.08), model_scale_units="absolute", instances_per_scene=1, shared_across_scenes=False)
        self.cars = self.add_node("models/box", model_pivot_relative_point=(0.5, 0.5, 0.5), model_scale=(0.16, 0.08, 0.06), instances_per_scene=self.n_cars, shared_across_scenes=False)

        self.chicken_pos = torch.zeros((n, 1, 3), dtype=torch.float32)
        self.cars_pos = torch.zeros((n, self.n_cars, 3), dtype=torch.float32)

        lane_h = 2.0 / max(1, self.lanes)
        for lane in range(self.lanes):
            y = -1.0 + (lane + 0.5) * lane_h
            sl = slice(lane * self.cars_per_lane, (lane + 1) * self.cars_per_lane)
            self.cars_pos[:, sl, 1] = y

        self.chicken.set_colors(torch.tensor([[[1.0, 0.95, 0.25, 1.0]]], dtype=torch.float32).repeat(n, 1, 1))
        self.cars.set_colors(torch.tensor([[[1.0, 0.3, 0.3, 1.0]]], dtype=torch.float32).repeat(n, self.n_cars, 1))

        self.add_camera(fov_y_deg=45.0)
        self._pbr_cam.set_positions(torch.tensor([0.0, -3.0, 0.0], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        self.add_light(ambient=(0.35, 0.35, 0.35))
        self.setup_environment()

    def _step(self, state_batch=None):
        if state_batch is None:
            return
        st = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        self.chicken_pos[:, :, 0] = 0.0
        self.chicken_pos[:, :, 1] = st[:, 0:1]

        car_x = st[:, 1:1 + self.n_cars]
        self.cars_pos[:, :, 0] = car_x
        self.chicken.set_positions(self.chicken_pos)
        self.cars.set_positions(self.cars_pos)
