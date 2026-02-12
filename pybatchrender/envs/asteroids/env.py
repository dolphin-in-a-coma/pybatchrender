"""Asteroids inspired TorchRL environment."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import AsteroidsConfig
from .renderer import AsteroidsRenderer


class AsteroidsEnv(PBREnv):
    def __init__(self, renderer: AsteroidsRenderer, cfg: AsteroidsConfig | None = None, **cfg_overrides):
        super().__init__(renderer=renderer, cfg=cfg, device=torch.device(cfg.device), batch_size=torch.Size([cfg.num_scenes]))
        self.cfg: AsteroidsConfig = cfg
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.render = bool(cfg.render)
        self.n = int(cfg.num_asteroids)
        self.ship_speed = float(cfg.ship_speed)
        self.ast_speed = float(cfg.asteroid_speed)

        self.obs_dim = 2 + 4 * self.n  # ship_xy + asteroid(x,y,vx,vy)
        self.set_default_specs(direct_obs_dim=self.obs_dim, actions=5, with_pixels=self.render, pixels_only=False, discrete_actions=True)
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _sample(self, bs: torch.Size) -> torch.Tensor:
        B = int(bs[0])
        ship = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        ast_xy = torch.empty((B, self.n, 2), dtype=torch.float32, device=self.device).uniform_(-1.0, 1.0, generator=self.rng)
        ast_v = torch.empty((B, self.n, 2), dtype=torch.float32, device=self.device).uniform_(-self.ast_speed, self.ast_speed, generator=self.rng)
        return torch.cat([ship, ast_xy.reshape(B, -1), ast_v.reshape(B, -1)], dim=-1)

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

        ship = obs[:, :2].clone()
        ast_xy = obs[:, 2:2 + 2 * self.n].reshape(-1, self.n, 2).clone()
        ast_v = obs[:, 2 + 2 * self.n:].reshape(-1, self.n, 2).clone()

        ship[action == 1, 0] -= self.ship_speed
        ship[action == 2, 0] += self.ship_speed
        ship[action == 3, 1] += self.ship_speed
        ship[action == 4, 1] -= self.ship_speed
        ship = ship.clamp(-1.0, 1.0)

        ast_xy = ast_xy + ast_v
        out = (ast_xy.abs() > 1.0)
        ast_v = torch.where(out, -ast_v, ast_v)
        ast_xy = ast_xy.clamp(-1.0, 1.0)

        d = ast_xy - ship.unsqueeze(1)
        collide = (d.pow(2).sum(dim=-1).sqrt() < 0.14).any(dim=1)
        timeout = step_count >= (self.max_steps - 1)
        done = (collide | timeout).unsqueeze(-1)

        reward = torch.full((obs.shape[0], 1), 0.01, dtype=torch.float32, device=self.device)
        reward[collide] = -2.0

        next_raw = torch.cat([ship, ast_xy.reshape(obs.shape[0], -1), ast_v.reshape(obs.shape[0], -1)], dim=-1)

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
