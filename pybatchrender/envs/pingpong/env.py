"""PingPong TorchRL environment."""
from __future__ import annotations

import math
import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import PingPongConfig
from .renderer import PingPongRenderer


class PingPongEnv(PBREnv):
    """Atari Pong inspired environment with simple deterministic opponent."""

    def __init__(self, renderer: PingPongRenderer, cfg: PingPongConfig | None = None, **cfg_overrides):
        super().__init__(renderer=renderer, cfg=cfg, device=torch.device(cfg.device), batch_size=torch.Size([cfg.num_scenes]))
        self.cfg: PingPongConfig = cfg
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.render = bool(cfg.render)

        self.paddle_speed = float(cfg.paddle_speed)
        self.paddle_half_h = float(cfg.paddle_height) * 0.5
        self.ball_speed = float(cfg.ball_speed)
        self.opp_speed = float(cfg.opponent_speed)

        self.set_default_specs(direct_obs_dim=6, actions=3, with_pixels=self.render, pixels_only=False, discrete_actions=True)
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _sample_state(self, batch_shape: torch.Size):
        B = int(batch_shape[0])
        paddle = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        opp = torch.zeros((B, 1), dtype=torch.float32, device=self.device)
        ball_xy = torch.empty((B, 2), dtype=torch.float32, device=self.device).uniform_(-0.2, 0.2, generator=self.rng)
        angle = torch.empty((B,), dtype=torch.float32, device=self.device).uniform_(-0.8, 0.8, generator=self.rng)
        vx = torch.sign(torch.randn((B,), generator=self.rng, device=self.device)) * self.ball_speed
        vy = torch.sin(angle) * self.ball_speed
        ball_v = torch.stack([vx, vy], dim=-1)
        return torch.cat([paddle, opp, ball_xy, ball_v], dim=-1)

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        obs = self._sample_state(bs)
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
            obs = td0["observation"]
            step_count = td0["step_count"]
        else:
            step_count = tensordict.get("step_count", torch.zeros_like(obs[..., 0], dtype=torch.long, device=self.device))

        action = tensordict["action"].to(self.device)
        paddle_y, opp_y, bx, by, vx, vy = obs.unbind(-1)

        dy = torch.zeros_like(paddle_y)
        dy[action == 1] = self.paddle_speed
        dy[action == 2] = -self.paddle_speed
        paddle_y = torch.clamp(paddle_y + dy, -1.0 + self.paddle_half_h, 1.0 - self.paddle_half_h)

        target = by
        opp_move = torch.clamp(target - opp_y, -self.opp_speed, self.opp_speed)
        opp_y = torch.clamp(opp_y + opp_move, -1.0 + self.paddle_half_h, 1.0 - self.paddle_half_h)

        bx = bx + vx
        by = by + vy

        wall_hit = (by > 1.0) | (by < -1.0)
        vy = torch.where(wall_hit, -vy, vy)
        by = torch.clamp(by, -1.0, 1.0)

        left_contact = (bx <= -0.9) & ((by - paddle_y).abs() <= self.paddle_half_h)
        right_contact = (bx >= 0.9) & ((by - opp_y).abs() <= self.paddle_half_h)
        vx = torch.where(left_contact | right_contact, -vx, vx)

        scored_left = bx < -1.0
        scored_right = bx > 1.0
        done = (scored_left | scored_right | (step_count >= self.max_steps - 1)).unsqueeze(-1)

        reward = torch.zeros((obs.shape[0], 1), dtype=torch.float32, device=self.device)
        reward[scored_right] = 1.0
        reward[scored_left] = -1.0

        next_obs_raw = torch.stack([paddle_y, opp_y, bx, by, vx, vy], dim=-1)

        if self.auto_reset:
            reset_obs = self._sample_state(next_obs_raw.shape[:-1])
            next_obs = torch.where(done, reset_obs, next_obs_raw)
            next_step = torch.where(done.squeeze(-1), torch.zeros_like(step_count), step_count + 1)
        else:
            next_obs = next_obs_raw
            next_step = step_count + 1

        fields = {"observation": next_obs, "reward": reward, "done": done, "step_count": next_step}
        if self.render:
            fields["pixels"] = self.render_pixels(next_obs)
        return TensorDict(fields, batch_size=self.batch_size)
