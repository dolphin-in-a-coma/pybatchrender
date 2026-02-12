"""Space Invaders inspired TorchRL environment."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import SpaceInvadersConfig
from .renderer import SpaceInvadersRenderer


class SpaceInvadersEnv(PBREnv):
    def __init__(self, renderer: SpaceInvadersRenderer, cfg: SpaceInvadersConfig | None = None, **cfg_overrides):
        super().__init__(renderer=renderer, cfg=cfg, device=torch.device(cfg.device), batch_size=torch.Size([cfg.num_scenes]))
        self.cfg: SpaceInvadersConfig = cfg
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.render = bool(cfg.render)

        self.rows = int(cfg.rows)
        self.cols = int(cfg.cols)
        self.n_inv = self.rows * self.cols
        self.obs_dim = 6 + self.n_inv  # player_x, bullet_active,x,y, invader_dx, invader_yoff + mask

        self.player_speed = float(cfg.player_speed)
        self.bullet_speed = float(cfg.bullet_speed)
        self.invader_speed = float(cfg.invader_speed)
        self.invader_drop = float(cfg.invader_drop)

        self.set_default_specs(direct_obs_dim=self.obs_dim, actions=4, with_pixels=self.render, pixels_only=False, discrete_actions=True)
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _sample(self, bs: torch.Size):
        B = int(bs[0])
        player_x = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        bullet_active = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        bullet_x = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        bullet_y = torch.full((B, 1), -0.9, dtype=torch.float32, device=self.device)
        inv_dx = torch.full((B, 1), self.invader_speed, dtype=torch.float32, device=self.device)
        inv_yoff = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        inv_alive = torch.ones((B, self.n_inv), dtype=torch.float32, device=self.device)
        return torch.cat([player_x, bullet_active, bullet_x, bullet_y, inv_dx, inv_yoff, inv_alive], dim=-1)

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

        player_x = obs[:, 0]
        bullet_active = obs[:, 1]
        bullet_x = obs[:, 2]
        bullet_y = obs[:, 3]
        inv_dx = obs[:, 4]
        inv_yoff = obs[:, 5]
        inv_alive = obs[:, 6:].clone()

        player_x[action == 0] -= self.player_speed
        player_x[action == 2] += self.player_speed
        player_x = player_x.clamp(-1.0, 1.0)

        fire = (action == 3) & (bullet_active < 0.5)
        bullet_active[fire] = 1.0
        bullet_x[fire] = player_x[fire]
        bullet_y[fire] = -0.85

        bullet_y = bullet_y + bullet_active * self.bullet_speed
        bullet_active[bullet_y > 1.0] = 0.0

        inv_center = inv_dx * step_count.to(torch.float32)
        edge = inv_center.abs() > 0.65
        if edge.any():
            inv_dx[edge] = -inv_dx[edge]
            inv_yoff[edge] -= self.invader_drop

        # bullet hits nearest alive invader in simple grid projection
        hit = torch.zeros((obs.shape[0],), dtype=torch.bool, device=self.device)
        if (bullet_active > 0.5).any():
            col = (((bullet_x + 1.0) * 0.5) * self.cols).long().clamp(0, self.cols - 1)
            row = (((0.8 - (bullet_y - inv_yoff)) / 0.2)).long().clamp(0, self.rows - 1)
            idx = row * self.cols + col
            alive = inv_alive[torch.arange(obs.shape[0], device=self.device), idx] > 0.5
            hit = alive & (bullet_active > 0.5) & (bullet_y > 0.2)
            inv_alive[torch.arange(obs.shape[0], device=self.device)[hit], idx[hit]] = 0.0
            bullet_active[hit] = 0.0

        invader_reach = inv_yoff < -0.75
        cleared = inv_alive.sum(dim=1) == 0
        timeout = step_count >= (self.max_steps - 1)
        done = (invader_reach | cleared | timeout).unsqueeze(-1)

        reward = torch.full((obs.shape[0], 1), -0.001, dtype=torch.float32, device=self.device)
        reward[hit] += 1.0
        reward[cleared] += 8.0
        reward[invader_reach] -= 4.0

        next_raw = torch.cat([
            player_x.unsqueeze(-1), bullet_active.unsqueeze(-1), bullet_x.unsqueeze(-1), bullet_y.unsqueeze(-1),
            inv_dx.unsqueeze(-1), inv_yoff.unsqueeze(-1), inv_alive
        ], dim=-1)

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
