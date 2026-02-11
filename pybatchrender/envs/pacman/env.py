"""Pac-Man-like TorchRL environment."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import PacManConfig
from .renderer import PacManRenderer


class PacManEnv(PBREnv):
    """
    Minimal 2D Pac-Man-like environment.

    Actions:
      0 stay, 1 up, 2 down, 3 left, 4 right

    Observation (8 dims):
      [pac_x, pac_y, ghost_x, ghost_y, powered_frac, pellets_frac, cherry_alive, power_frac]
      where coordinates are normalized into [-1, 1].
    """

    def __init__(self, renderer: PacManRenderer, cfg: PacManConfig | dict | None = None, **cfg_overrides):
        super().__init__(
            renderer=renderer,
            cfg=cfg,
            device=torch.device(cfg.device),
            batch_size=torch.Size([cfg.num_scenes]),
        )

        self.cfg: PacManConfig = cfg  # typing hint
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.powered_steps = int(cfg.powered_steps)
        self.render = bool(cfg.render)

        self.set_default_specs(
            direct_obs_dim=8,
            actions=5,
            with_pixels=self.render,
            pixels_only=False,
            discrete_actions=True,
        )

        # Map must match renderer map
        self.map_rows = renderer.map_rows
        self.H = len(self.map_rows)
        self.W = len(self.map_rows[0])

        # Grid->world conversion helper from renderer
        self._to_world_xy = renderer._to_world_xy

        self.walkable = torch.zeros((self.H, self.W), dtype=torch.bool)
        for y, row in enumerate(self.map_rows):
            for x, c in enumerate(row):
                self.walkable[y, x] = c != "#"

        # Collectible coordinates (same ordering as renderer.collectible_coords)
        self.collect_xy: list[tuple[int, int]] = []
        for y, row in enumerate(self.map_rows):
            for x, c in enumerate(row):
                if c != "#":
                    self.collect_xy.append((x, y))

        self.n_collect = len(self.collect_xy)
        self.collect_index = {xy: i for i, xy in enumerate(self.collect_xy)}

        # Fixed special items in-grid
        self.pac_start = (1, 1)
        self.ghost_start = (self.W - 2, self.H - 2)
        self.cherry_cell = (self.W // 2, self.H // 2)
        self.power_cells = [(1, self.H - 2), (self.W - 2, 1), (self.W - 2, self.H - 2), (1, 1)]

        # Runtime state tensors
        B = int(cfg.num_scenes)
        self.pac_xy = torch.zeros((B, 2), dtype=torch.long, device=self.device)
        self.ghost_xy = torch.zeros((B, 2), dtype=torch.long, device=self.device)
        self.power_timer = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.step_count = torch.zeros((B,), dtype=torch.long, device=self.device)

        self.pellet_alive = torch.zeros((B, self.n_collect), dtype=torch.bool, device=self.device)
        self.cherry_alive = torch.zeros((B, self.n_collect), dtype=torch.bool, device=self.device)
        self.power_alive = torch.zeros((B, self.n_collect), dtype=torch.bool, device=self.device)

        self._last_render_payload = None
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        # grid [0,W-1],[0,H-1] -> [-1,1]
        x = (xy[:, 0].to(torch.float32) / max(1.0, (self.W - 1))) * 2.0 - 1.0
        y = (xy[:, 1].to(torch.float32) / max(1.0, (self.H - 1))) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    def _obs(self) -> torch.Tensor:
        pac = self._normalize_xy(self.pac_xy)
        ghost = self._normalize_xy(self.ghost_xy)
        powered_frac = (self.power_timer.to(torch.float32) / max(1, self.powered_steps)).unsqueeze(-1)
        pellets_frac = self.pellet_alive.to(torch.float32).mean(dim=1, keepdim=True)
        cherry_alive = self.cherry_alive.to(torch.float32).amax(dim=1, keepdim=True)
        power_frac = self.power_alive.to(torch.float32).mean(dim=1, keepdim=True)
        return torch.cat([pac, ghost, powered_frac, pellets_frac, cherry_alive, power_frac], dim=-1)

    def _prepare_items(self):
        B = self.batch_size[0]
        self.pellet_alive[:] = True
        self.cherry_alive[:] = False
        self.power_alive[:] = False

        # Remove regular pellets where special entities start
        for cell in [self.pac_start, self.ghost_start, self.cherry_cell, *self.power_cells]:
            idx = self.collect_index[cell]
            self.pellet_alive[:, idx] = False

        # Cherry at one cell
        self.cherry_alive[:, self.collect_index[self.cherry_cell]] = True

        # Power pills at selected cells
        for cell in self.power_cells:
            self.power_alive[:, self.collect_index[cell]] = True

    def _build_render_payload(self):
        def grid_to_world(xy: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((xy.shape[0], 2), dtype=torch.float32, device=xy.device)
            for i in range(xy.shape[0]):
                wx, wy = self._to_world_xy(int(xy[i, 0]), int(xy[i, 1]))
                out[i, 0] = wx
                out[i, 1] = wy
            return out

        self._last_render_payload = {
            "pac_xy": grid_to_world(self.pac_xy).cpu(),
            "ghost_xy": grid_to_world(self.ghost_xy).cpu(),
            "pellet_alive": self.pellet_alive.cpu(),
            "cherry_alive": self.cherry_alive.cpu(),
            "power_alive": self.power_alive.cpu(),
            "powered": (self.power_timer > 0).cpu(),
        }

    def render_pixels(self, obs: torch.Tensor | None = None) -> torch.Tensor:
        if self._last_render_payload is None:
            self._build_render_payload()
        return self._renderer.step(self._last_render_payload)

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        B = self.batch_size[0]
        self.pac_xy[:] = torch.tensor(self.pac_start, device=self.device)
        self.ghost_xy[:] = torch.tensor(self.ghost_start, device=self.device)
        self.power_timer.zero_()
        self.step_count.zero_()
        self._prepare_items()

        obs = self._obs()
        done = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        td_fields = {
            "observation": obs,
            "step_count": self.step_count.clone(),
            "done": done,
        }
        if self.render:
            self._build_render_payload()
            td_fields["pixels"] = self.render_pixels(obs)
        return TensorDict(td_fields, batch_size=self.batch_size)

    def _apply_action(self, action: torch.Tensor):
        # 0 stay, 1 up, 2 down, 3 left, 4 right
        delta = torch.zeros((action.shape[0], 2), dtype=torch.long, device=self.device)
        delta[action == 1] = torch.tensor([0, -1], device=self.device)
        delta[action == 2] = torch.tensor([0, 1], device=self.device)
        delta[action == 3] = torch.tensor([-1, 0], device=self.device)
        delta[action == 4] = torch.tensor([1, 0], device=self.device)

        cand = self.pac_xy + delta
        cand[:, 0] = cand[:, 0].clamp(0, self.W - 1)
        cand[:, 1] = cand[:, 1].clamp(0, self.H - 1)

        ok = self.walkable[cand[:, 1].cpu(), cand[:, 0].cpu()].to(self.device)
        self.pac_xy = torch.where(ok.unsqueeze(-1), cand, self.pac_xy)

    def _move_ghost(self):
        # Random valid one-step move
        moves = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]], dtype=torch.long, device=self.device)
        B = self.batch_size[0]
        for i in range(B):
            candidates = self.ghost_xy[i].unsqueeze(0) + moves
            candidates[:, 0] = candidates[:, 0].clamp(0, self.W - 1)
            candidates[:, 1] = candidates[:, 1].clamp(0, self.H - 1)
            valid = self.walkable[candidates[:, 1].cpu(), candidates[:, 0].cpu()]
            valid_idx = torch.where(valid)[0]
            pick = valid_idx[torch.randint(0, len(valid_idx), (1,), generator=self.rng)][0]
            self.ghost_xy[i] = candidates[pick]

    def _collect_items(self) -> torch.Tensor:
        B = self.batch_size[0]
        reward = torch.full((B, 1), float(self.cfg.step_penalty), dtype=torch.float32, device=self.device)

        for i in range(B):
            cell = (int(self.pac_xy[i, 0]), int(self.pac_xy[i, 1]))
            idx = self.collect_index.get(cell)
            if idx is None:
                continue
            if self.pellet_alive[i, idx]:
                self.pellet_alive[i, idx] = False
                reward[i, 0] += float(self.cfg.reward_pellet)
            if self.cherry_alive[i, idx]:
                self.cherry_alive[i, idx] = False
                reward[i, 0] += float(self.cfg.reward_cherry)
            if self.power_alive[i, idx]:
                self.power_alive[i, idx] = False
                self.power_timer[i] = self.powered_steps
                reward[i, 0] += float(self.cfg.reward_power_pill)

        return reward

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].to(self.device)
        self._apply_action(action)
        self._move_ghost()

        reward = self._collect_items()

        # Collision logic
        collide = (self.pac_xy == self.ghost_xy).all(dim=1)
        powered = self.power_timer > 0
        eat_ghost = collide & powered
        lose = collide & (~powered)

        if eat_ghost.any():
            self.ghost_xy[eat_ghost] = torch.tensor(self.ghost_start, device=self.device)
            reward[eat_ghost] += float(self.cfg.reward_eat_ghost)

        done = lose.unsqueeze(-1)
        reward[lose] += float(self.cfg.reward_lose)

        # Win condition: all collectible groups consumed
        no_items_left = (~self.pellet_alive & ~self.cherry_alive & ~self.power_alive).all(dim=1)
        newly_won = no_items_left & (~done.squeeze(-1))
        reward[newly_won] += float(self.cfg.reward_win)
        done = done | newly_won.unsqueeze(-1)

        # Max step termination
        done = done | (self.step_count >= (self.max_steps - 1)).unsqueeze(-1)

        # Update timers
        self.power_timer = torch.clamp(self.power_timer - 1, min=0)

        # Auto reset selected scenes
        if self.auto_reset and done.any():
            idx = done.squeeze(-1)
            self.pac_xy[idx] = torch.tensor(self.pac_start, device=self.device)
            self.ghost_xy[idx] = torch.tensor(self.ghost_start, device=self.device)
            self.power_timer[idx] = 0
            self.step_count[idx] = 0
            # reset collectibles for completed scenes only
            pellets, cherries, powers = self.pellet_alive[idx], self.cherry_alive[idx], self.power_alive[idx]
            pellets[:] = True
            cherries[:] = False
            powers[:] = False
            for cell in [self.pac_start, self.ghost_start, self.cherry_cell, *self.power_cells]:
                ii = self.collect_index[cell]
                pellets[:, ii] = False
            cherries[:, self.collect_index[self.cherry_cell]] = True
            for cell in self.power_cells:
                powers[:, self.collect_index[cell]] = True
            self.pellet_alive[idx], self.cherry_alive[idx], self.power_alive[idx] = pellets, cherries, powers
        else:
            self.step_count += 1

        obs = self._obs()
        self.step_count += (~done.squeeze(-1)).to(torch.long)

        out_fields = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "step_count": self.step_count.clone(),
        }

        if self.render:
            self._build_render_payload()
            out_fields["pixels"] = self.render_pixels(obs)

        return TensorDict(out_fields, batch_size=self.batch_size)
