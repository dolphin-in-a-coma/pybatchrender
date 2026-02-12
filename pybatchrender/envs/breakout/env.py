"""Breakout TorchRL environment."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import BreakoutConfig
from .renderer import BreakoutRenderer


class BreakoutEnv(PBREnv):
    def __init__(self, renderer: BreakoutRenderer, cfg: BreakoutConfig | None = None, **cfg_overrides):
        super().__init__(renderer=renderer, cfg=cfg, device=torch.device(cfg.device), batch_size=torch.Size([cfg.num_scenes]))
        self.cfg: BreakoutConfig = cfg
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.render = bool(cfg.render)

        self.rows = int(cfg.brick_rows)
        self.cols = int(cfg.brick_cols)
        self.n_bricks = self.rows * self.cols
        self.paddle_speed = float(cfg.paddle_speed)
        self.paddle_half_w = float(cfg.paddle_width) * 0.5
        self.ball_speed = float(cfg.ball_speed)

        self.obs_dim = 5 + self.n_bricks
        self.set_default_specs(direct_obs_dim=self.obs_dim, actions=3, with_pixels=self.render, pixels_only=False, discrete_actions=True)
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _sample(self, batch_shape: torch.Size) -> torch.Tensor:
        B = int(batch_shape[0])
        paddle_x = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        ball = torch.tensor([0.0, -0.5], dtype=torch.float32, device=self.device).repeat(B, 1)
        vx = torch.sign(torch.randn((B,), generator=self.rng, device=self.device)) * self.ball_speed
        vy = torch.full((B,), self.ball_speed, dtype=torch.float32, device=self.device)
        vel = torch.stack([vx, vy], dim=-1)
        bricks = torch.ones((B, self.n_bricks), dtype=torch.float32, device=self.device)
        return torch.cat([paddle_x, ball, vel, bricks], dim=-1)

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        obs = self._sample(bs)
        done = torch.zeros((*bs, 1), dtype=torch.bool, device=self.device)
        step = torch.zeros(bs, dtype=torch.long, device=self.device)
        fields = {"observation": obs, "done": done, "step_count": step}
        if self.render:
            fields["pixels"] = self.render_pixels(obs)
        return TensorDict(fields, batch_size=self.batch_size)

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.get("observation", None)
        if obs is None:
            td0 = self._reset()
            obs, step_count = td0["observation"], td0["step_count"]
        else:
            step_count = tensordict.get("step_count", torch.zeros_like(obs[..., 0], dtype=torch.long, device=self.device))

        action = tensordict["action"].to(self.device)
        paddle_x = obs[:, 0]
        bx, by, vx, vy = obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4]
        bricks = obs[:, 5:].clone()

        move = torch.zeros_like(paddle_x)
        move[action == 0] = -self.paddle_speed
        move[action == 2] = self.paddle_speed
        paddle_x = torch.clamp(paddle_x + move, -1.0 + self.paddle_half_w, 1.0 - self.paddle_half_w)

        bx = bx + vx
        by = by + vy

        side = (bx < -1.0) | (bx > 1.0)
        top = by > 1.0
        vx = torch.where(side, -vx, vx)
        vy = torch.where(top, -vy, vy)
        bx = torch.clamp(bx, -1.0, 1.0)
        by = torch.clamp(by, -1.0, 1.0)

        paddle_hit = (by <= -0.85) & (vy < 0) & ((bx - paddle_x).abs() <= self.paddle_half_w)
        vy = torch.where(paddle_hit, vy.abs(), vy)

        # Brick collision via nearest cell in top half
        grid_x = ((bx + 1.0) * 0.5 * self.cols).long().clamp(0, self.cols - 1)
        grid_y = (((1.0 - by) * 0.5) * self.rows).long().clamp(0, self.rows - 1)
        in_brick_zone = by > 0.15
        idx = grid_y * self.cols + grid_x
        hit = in_brick_zone & (bricks[torch.arange(bricks.shape[0], device=self.device), idx] > 0.5)
        if hit.any():
            bricks[torch.arange(bricks.shape[0], device=self.device)[hit], idx[hit]] = 0.0
            vy = torch.where(hit, -vy, vy)

        fell = by < -1.0
        cleared = (bricks.sum(dim=1) == 0)
        timeout = step_count >= (self.max_steps - 1)
        done = (fell | cleared | timeout).unsqueeze(-1)

        reward = torch.full((obs.shape[0], 1), -0.001, dtype=torch.float32, device=self.device)
        reward[hit] += 1.0
        reward[cleared] += 5.0
        reward[fell] -= 1.0

        next_obs_raw = torch.cat([paddle_x.unsqueeze(-1), bx.unsqueeze(-1), by.unsqueeze(-1), vx.unsqueeze(-1), vy.unsqueeze(-1), bricks], dim=-1)

        if self.auto_reset:
            reset_obs = self._sample(next_obs_raw.shape[:-1])
            next_obs = torch.where(done, reset_obs, next_obs_raw)
            next_step = torch.where(done.squeeze(-1), torch.zeros_like(step_count), step_count + 1)
        else:
            next_obs = next_obs_raw
            next_step = step_count + 1

        fields = {"observation": next_obs, "reward": reward, "done": done, "step_count": next_step}
        if self.render:
            fields["pixels"] = self.render_pixels(next_obs)
        return TensorDict(fields, batch_size=self.batch_size)
