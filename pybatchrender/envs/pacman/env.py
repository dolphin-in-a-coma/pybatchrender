"""Pac-Man-like TorchRL environment with configurable map/item matrices."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import PacManConfig
from .layout import build_spec
from .renderer import PacManRenderer


class PacManEnv(PBREnv):
    """
    Observation dimension is dynamic:
      2 * (1 + num_ghosts) + num_pellets + num_power_pellets + num_cherries

    Actor coordinates are continuous (float), so Pac-Man/ghosts can be between cells.
    """

    def __init__(self, renderer: PacManRenderer, cfg: PacManConfig | dict | None = None, **cfg_overrides):
        super().__init__(
            renderer=renderer,
            cfg=cfg,
            device=torch.device(cfg.device),
            batch_size=torch.Size([cfg.num_scenes]),
        )

        self.cfg: PacManConfig = cfg
        self.spec = build_spec(self.cfg)
        self.layout = self.spec["layout"]
        self.W, self.H = self.layout.width, self.layout.height

        self.pac_start = self.spec["pac_start"]
        self.ghost_starts = self.spec["ghost_starts"]
        self.pellet_cells = self.spec["pellet_cells"]
        self.power_cells = self.spec["power_cells"]
        self.cherry_cells = self.spec["cherry_cells"]

        self.num_ghosts = len(self.ghost_starts)
        self.n_pellets = len(self.pellet_cells)
        self.n_power = len(self.power_cells)
        self.n_cherries = len(self.cherry_cells)

        self.obs_dim = 2 * (1 + self.num_ghosts) + self.n_pellets + self.n_power + self.n_cherries

        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.powered_steps = int(cfg.powered_steps)
        self.render = bool(cfg.render)

        self.bind_to_cells = bool(getattr(cfg, "bind_actor_positions_to_cells", False))
        self.pacman_step_size = 1.0 if self.bind_to_cells else float(cfg.pacman_step_size)
        self.ghost_step_size = 1.0 if self.bind_to_cells else float(cfg.ghost_step_size)
        self.actor_radius = 0.01 if self.bind_to_cells else float(cfg.actor_radius)
        self.collect_radius = 0.51 if self.bind_to_cells else float(cfg.collect_radius)
        self.collision_radius = 0.51 if self.bind_to_cells else float(cfg.collision_radius)

        self.set_default_specs(
            direct_obs_dim=self.obs_dim,
            actions=5,
            with_pixels=self.render,
            pixels_only=False,
            discrete_actions=True,
        )

        self.walkable = torch.zeros((self.H, self.W), dtype=torch.bool, device=self.device)
        for y in range(self.H):
            for x in range(self.W):
                self.walkable[y, x] = self.layout.is_walkable(x, y)

        self._walkable_offsets = torch.tensor(
            [
                (-self.actor_radius, -self.actor_radius),
                (-self.actor_radius, self.actor_radius),
                (self.actor_radius, -self.actor_radius),
                (self.actor_radius, self.actor_radius),
                (0.0, 0.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._ghost_moves = torch.tensor(
            [[0.0, -self.ghost_step_size], [0.0, self.ghost_step_size], [-self.ghost_step_size, 0.0], [self.ghost_step_size, 0.0], [0.0, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )

        self.pellet_xy = torch.tensor(self.pellet_cells, dtype=torch.float32, device=self.device) if self.n_pellets else torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        self.power_xy = torch.tensor(self.power_cells, dtype=torch.float32, device=self.device) if self.n_power else torch.zeros((0, 2), dtype=torch.float32, device=self.device)
        self.cherry_xy = torch.tensor(self.cherry_cells, dtype=torch.float32, device=self.device) if self.n_cherries else torch.zeros((0, 2), dtype=torch.float32, device=self.device)

        B = int(cfg.num_scenes)
        self.pac_xy = torch.zeros((B, 2), dtype=torch.float32, device=self.device)
        self.ghosts_xy = torch.zeros((B, self.num_ghosts, 2), dtype=torch.float32, device=self.device)

        self.pellet_alive = torch.zeros((B, self.n_pellets), dtype=torch.bool, device=self.device)
        self.power_alive = torch.zeros((B, self.n_power), dtype=torch.bool, device=self.device)
        self.cherry_alive = torch.zeros((B, self.n_cherries), dtype=torch.bool, device=self.device)

        self.power_timer = torch.zeros((B,), dtype=torch.long, device=self.device)
        self.step_count = torch.zeros((B,), dtype=torch.long, device=self.device)

        self._last_render_payload = None
        self.set_seed(int(cfg.seed))

    def _set_seed(self, seed: int) -> None:
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def _normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        x = (xy[..., 0] / max(1.0, (self.W - 1))) * 2.0 - 1.0
        y = (xy[..., 1] / max(1.0, (self.H - 1))) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    def _obs(self) -> torch.Tensor:
        pac = self._normalize_xy(self.pac_xy)
        ghosts = self._normalize_xy(self.ghosts_xy).reshape(self.batch_size[0], -1)
        items = torch.cat(
            [
                self.pellet_alive.to(torch.float32),
                self.power_alive.to(torch.float32),
                self.cherry_alive.to(torch.float32),
            ],
            dim=1,
        )
        return torch.cat([pac, ghosts, items], dim=1)

    def _snap_to_cells(self, pos: torch.Tensor) -> torch.Tensor:
        out = pos.clone()
        out[:, 0] = torch.clamp(torch.round(out[:, 0]), 0, self.W - 1)
        out[:, 1] = torch.clamp(torch.round(out[:, 1]), 0, self.H - 1)
        return out

    def _is_pos_walkable(self, pos: torch.Tensor) -> torch.Tensor:
        """Approximate circular actor collision vs wall cells (vectorized)."""
        samples = pos.unsqueeze(1) + self._walkable_offsets.unsqueeze(0)  # [N, 5, 2]
        sx = samples[..., 0]
        sy = samples[..., 1]

        in_bounds = (sx >= 0.0) & (sx <= (self.W - 1)) & (sy >= 0.0) & (sy <= (self.H - 1))

        cx = torch.round(sx).long().clamp(0, self.W - 1)
        cy = torch.round(sy).long().clamp(0, self.H - 1)
        walkable = self.walkable[cy, cx]

        return (in_bounds & walkable).all(dim=1)

    def _init_scene(self, mask: torch.Tensor | None = None):
        if mask is None:
            mask = torch.ones((self.batch_size[0],), dtype=torch.bool, device=self.device)
        if mask.sum() == 0:
            return

        self.pac_xy[mask] = torch.tensor(self.pac_start, dtype=torch.float32, device=self.device)
        for gi, g in enumerate(self.ghost_starts):
            self.ghosts_xy[mask, gi] = torch.tensor(g, dtype=torch.float32, device=self.device)

        if self.n_pellets:
            self.pellet_alive[mask] = True
        if self.n_power:
            self.power_alive[mask] = True
        if self.n_cherries:
            self.cherry_alive[mask] = True
        self.power_timer[mask] = 0
        self.step_count[mask] = 0

    def _build_render_payload(self):
        self._last_render_payload = {
            "pac_xy": self.pac_xy.detach().cpu(),
            "ghosts_xy": self.ghosts_xy.detach().cpu(),
            "pellet_xy": self.pellet_xy.detach().cpu(),
            "pellet_alive": self.pellet_alive.detach().cpu(),
            "power_xy": self.power_xy.detach().cpu(),
            "power_alive": self.power_alive.detach().cpu(),
            "cherry_xy": self.cherry_xy.detach().cpu(),
            "cherry_alive": self.cherry_alive.detach().cpu(),
        }

    def render_pixels(self, obs: torch.Tensor | None = None) -> torch.Tensor:
        if self._last_render_payload is None:
            self._build_render_payload()
        return self._renderer.step(self._last_render_payload)

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        self._init_scene(None)
        obs = self._obs()
        done = torch.zeros((self.batch_size[0], 1), dtype=torch.bool, device=self.device)
        fields = {
            "observation": obs,
            "step_count": self.step_count.clone(),
            "done": done,
        }
        if self.render:
            self._build_render_payload()
            fields["pixels"] = self.render_pixels(obs)
        return TensorDict(fields, batch_size=self.batch_size)

    def _apply_action(self, action: torch.Tensor):
        delta = torch.zeros((action.shape[0], 2), dtype=torch.float32, device=self.device)
        s = self.pacman_step_size
        delta[action == 1] = torch.tensor([0.0, -s], device=self.device)
        delta[action == 2] = torch.tensor([0.0, s], device=self.device)
        delta[action == 3] = torch.tensor([-s, 0.0], device=self.device)
        delta[action == 4] = torch.tensor([s, 0.0], device=self.device)

        cand = self.pac_xy + delta
        if self.bind_to_cells:
            cand = self._snap_to_cells(cand)
        ok = self._is_pos_walkable(cand)
        self.pac_xy = torch.where(ok.unsqueeze(-1), cand, self.pac_xy)
        if self.bind_to_cells:
            self.pac_xy = self._snap_to_cells(self.pac_xy)

    def _move_ghosts(self):
        B = self.batch_size[0]
        BG = B * self.num_ghosts

        cur = self.ghosts_xy.reshape(BG, 2)
        candidates = cur.unsqueeze(1) + self._ghost_moves.unsqueeze(0)  # [BG, 5, 2]
        if self.bind_to_cells:
            candidates = self._snap_to_cells(candidates.reshape(-1, 2)).reshape(BG, 5, 2)

        valid = self._is_pos_walkable(candidates.reshape(-1, 2)).reshape(BG, 5)
        has_valid = valid.any(dim=1)

        rand_scores = torch.rand((BG, 5), generator=self.rng, device=self.device)
        rand_scores = rand_scores.masked_fill(~valid, -1.0)
        choice = rand_scores.argmax(dim=1)

        chosen = candidates[torch.arange(BG, device=self.device), choice]
        next_pos = torch.where(has_valid.unsqueeze(-1), chosen, cur)

        self.ghosts_xy = next_pos.reshape(B, self.num_ghosts, 2)

    def _collect_group(self, actor_xy: torch.Tensor, item_xy: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        if item_xy.shape[0] == 0:
            return torch.zeros((actor_xy.shape[0], 0), dtype=torch.bool, device=self.device)
        d = actor_xy.unsqueeze(1) - item_xy.unsqueeze(0)
        dist = torch.sqrt((d * d).sum(dim=-1))
        hit = dist <= self.collect_radius
        return hit & alive

    def _collect(self) -> torch.Tensor:
        B = self.batch_size[0]
        reward = torch.full((B, 1), float(self.cfg.step_penalty), dtype=torch.float32, device=self.device)

        if self.n_pellets:
            hit = self._collect_group(self.pac_xy, self.pellet_xy, self.pellet_alive)
            got = hit.any(dim=1)
            self.pellet_alive[hit] = False
            reward[got] += float(self.cfg.reward_pellet)

        if self.n_power:
            hit = self._collect_group(self.pac_xy, self.power_xy, self.power_alive)
            got = hit.any(dim=1)
            self.power_alive[hit] = False
            self.power_timer[got] = self.powered_steps
            reward[got] += float(self.cfg.reward_power_pill)

        if self.n_cherries:
            hit = self._collect_group(self.pac_xy, self.cherry_xy, self.cherry_alive)
            got = hit.any(dim=1)
            self.cherry_alive[hit] = False
            reward[got] += float(self.cfg.reward_cherry)

        return reward

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].to(self.device)
        self._apply_action(action)
        self._move_ghosts()

        reward = self._collect()

        d = self.ghosts_xy - self.pac_xy.unsqueeze(1)
        ghost_dist = torch.sqrt((d * d).sum(dim=-1))
        collide = (ghost_dist <= self.collision_radius).any(dim=1)

        powered = self.power_timer > 0
        eat_ghost = collide & powered
        lose = collide & (~powered)

        if eat_ghost.any():
            idx = torch.where(eat_ghost)[0]
            for g, start in enumerate(self.ghost_starts):
                self.ghosts_xy[idx, g] = torch.tensor(start, dtype=torch.float32, device=self.device)
            reward[eat_ghost] += float(self.cfg.reward_eat_ghost)

        done = lose.unsqueeze(-1)
        reward[lose] += float(self.cfg.reward_lose)

        no_pellets = (~self.pellet_alive).all(dim=1) if self.n_pellets else torch.ones((self.batch_size[0],), dtype=torch.bool, device=self.device)
        no_power = (~self.power_alive).all(dim=1) if self.n_power else torch.ones((self.batch_size[0],), dtype=torch.bool, device=self.device)
        no_cherries = (~self.cherry_alive).all(dim=1) if self.n_cherries else torch.ones((self.batch_size[0],), dtype=torch.bool, device=self.device)
        all_items_eaten = no_pellets & no_power & no_cherries
        win = all_items_eaten & (~done.squeeze(-1))
        reward[win] += float(self.cfg.reward_win)
        done = done | win.unsqueeze(-1)

        done = done | (self.step_count >= (self.max_steps - 1)).unsqueeze(-1)

        self.power_timer = torch.clamp(self.power_timer - 1, min=0)
        self.step_count += 1

        if self.auto_reset and done.any():
            self._init_scene(done.squeeze(-1))

        obs = self._obs()

        fields = {
            "observation": obs,
            "reward": reward,
            "done": done,
            "step_count": self.step_count.clone(),
        }

        if self.render:
            self._build_render_payload()
            fields["pixels"] = self.render_pixels(obs)

        return TensorDict(fields, batch_size=self.batch_size)
