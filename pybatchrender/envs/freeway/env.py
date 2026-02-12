"""Freeway inspired TorchRL environment."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import FreewayConfig
from .renderer import FreewayRenderer


class FreewayEnv(PBREnv):
    def __init__(self, renderer: FreewayRenderer, cfg: FreewayConfig | None = None, **cfg_overrides):
        super().__init__(renderer=renderer, cfg=cfg, device=torch.device(cfg.device), batch_size=torch.Size([cfg.num_scenes]))
        self.cfg: FreewayConfig = cfg
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.render = bool(cfg.render)

        self.lanes = int(cfg.lanes)
        self.cars_per_lane = int(cfg.cars_per_lane)
        self.n_cars = self.lanes * self.cars_per_lane
        self.chicken_step = float(cfg.chicken_step)
        self.car_speed = float(cfg.car_speed)

        self.obs_dim = 1 + 2 * self.n_cars  # chicken_y + car x positions + direction signs
        self.set_default_specs(direct_obs_dim=self.obs_dim, actions=3, with_pixels=self.render, pixels_only=False, discrete_actions=True)
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _sample(self, bs: torch.Size):
        B = int(bs[0])
        chicken_y = torch.full((B, 1), -1.0, dtype=torch.float32, device=self.device)
        car_x = torch.empty((B, self.n_cars), dtype=torch.float32, device=self.device).uniform_(-1.0, 1.0, generator=self.rng)
        sign = torch.ones((B, self.n_cars), dtype=torch.float32, device=self.device)
        for lane in range(self.lanes):
            s = 1.0 if lane % 2 == 0 else -1.0
            sign[:, lane * self.cars_per_lane:(lane + 1) * self.cars_per_lane] = s
        return torch.cat([chicken_y, car_x, sign], dim=-1)

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        obs = self._sample(bs)
        done = torch.zeros((*bs, 1), dtype=torch.bool, device=self.device)
        step = torch.zeros(bs, dtype=torch.long, device=self.device)
        out = {"observation": obs, "done": done, "step_count": step}
        if self.render:
            out["pixels"] = self.render_pixels(obs)
        return TensorDict(out, batch_size=self.batch_size)

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.get("observation", None)
        if obs is None:
            td0 = self._reset()
            obs, step_count = td0["observation"], td0["step_count"]
        else:
            step_count = tensordict.get("step_count", torch.zeros_like(obs[..., 0], dtype=torch.long, device=self.device))

        action = tensordict["action"].to(self.device)
        chicken_y = obs[:, 0]
        car_x = obs[:, 1:1 + self.n_cars]
        sign = obs[:, 1 + self.n_cars:]

        chicken_y[action == 1] += self.chicken_step
        chicken_y[action == 2] -= self.chicken_step
        chicken_y = chicken_y.clamp(-1.0, 1.0)

        car_x = car_x + sign * self.car_speed
        wrap_hi = car_x > 1.05
        wrap_lo = car_x < -1.05
        car_x[wrap_hi] = -1.05
        car_x[wrap_lo] = 1.05

        # Collision by lane proximity
        lane_h = 2.0 / max(1, self.lanes)
        collide = torch.zeros((obs.shape[0],), dtype=torch.bool, device=self.device)
        for lane in range(self.lanes):
            lane_center = -1.0 + (lane + 0.5) * lane_h
            in_lane = (chicken_y - lane_center).abs() < (lane_h * 0.4)
            cars = car_x[:, lane * self.cars_per_lane:(lane + 1) * self.cars_per_lane]
            car_hit = (cars.abs() < 0.10).any(dim=1)
            collide |= in_lane & car_hit

        goal = chicken_y >= 0.95
        timeout = step_count >= (self.max_steps - 1)
        done = (collide | goal | timeout).unsqueeze(-1)

        reward = torch.full((obs.shape[0], 1), -0.001, dtype=torch.float32, device=self.device)
        reward[goal] += 5.0
        reward[collide] -= 1.0

        next_raw = torch.cat([chicken_y.unsqueeze(-1), car_x, sign], dim=-1)

        if self.auto_reset:
            reset_obs = self._sample(next_raw.shape[:-1])
            next_obs = torch.where(done, reset_obs, next_raw)
            next_step = torch.where(done.squeeze(-1), torch.zeros_like(step_count), step_count + 1)
        else:
            next_obs = next_raw
            next_step = step_count + 1

        out = {"observation": next_obs, "reward": reward, "done": done, "step_count": next_step}
        if self.render:
            out["pixels"] = self.render_pixels(next_obs)
        return TensorDict(out, batch_size=self.batch_size)
