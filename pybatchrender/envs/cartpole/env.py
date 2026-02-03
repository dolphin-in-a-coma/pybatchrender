"""CartPole TorchRL environment."""
import math

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import CartPoleConfig
from .renderer import CartPoleRenderer


class CartPoleEnv(PBREnv):
    """
    CartPole environment compatible with TorchRL.
    
    A pole is attached by an un-actuated joint to a cart, which moves along a
    frictionless track. The pendulum starts upright, and the goal is to prevent
    it from falling over by increasing and reducing the cart's velocity.
    
    Observation Space:
        - x: Cart position
        - x_dot: Cart velocity
        - theta: Pole angle (radians, 0 = upright)
        - theta_dot: Pole angular velocity
        
    Action Space:
        - 0: Push cart to the left
        - 1: Push cart to the right
        
    Reward:
        +1 for every step taken (survival reward)
        
    Termination:
        - Pole angle exceeds theta_threshold
        - Cart position exceeds x_threshold
        - Episode length exceeds max_steps
    """
    
    def __init__(
        self,
        renderer: CartPoleRenderer,
        cfg: CartPoleConfig | dict | None = None,
        **cfg_overrides,
    ):
        super().__init__(
            renderer=renderer,
            cfg=cfg,
            device=torch.device(cfg.device),
            batch_size=torch.Size([cfg.num_scenes]),
        )

        # Physics parameters
        self.gravity = float(cfg.gravity if cfg is not None else 9.8)
        self.masscart = float(cfg.masscart if cfg is not None else 1.0)
        self.masspole = float(cfg.masspole if cfg is not None else 0.1)
        self.total_mass = self.masscart + self.masspole
        self.length = float(cfg.length if cfg is not None else 0.5)
        self.polemass_length = self.masspole * self.length
        self.force_mag = float(cfg.force_mag if cfg is not None else 10.0)
        self.tau = float(cfg.tau if cfg is not None else 0.02)
        self.theta_threshold = float(
            (cfg.theta_threshold_deg if cfg is not None else 12.0) * 2 * math.pi / 360.0
        )
        self.x_threshold = float(cfg.x_threshold if cfg is not None else 2.4)
        self.max_steps = int(cfg.max_steps if cfg is not None else 500)
        self.auto_reset = bool(cfg.auto_reset if cfg is not None else True)

        self.seed = int(cfg.seed if cfg is not None else 0)
        self.render = bool(cfg.render if cfg is not None else True)

        # Convert degree-based config ranges to radians
        th0_lo, th0_hi = float(cfg.init_theta_range_deg[0]), float(cfg.init_theta_range_deg[1])
        thd_lo, thd_hi = float(cfg.init_theta_dot_range_deg[0]), float(cfg.init_theta_dot_range_deg[1])
        self._init_theta_range_rad = (
            math.radians(min(th0_lo, th0_hi)),
            math.radians(max(th0_lo, th0_hi)),
        )
        self._init_theta_dot_range_rad = (
            math.radians(min(thd_lo, thd_hi)),
            math.radians(max(thd_lo, thd_hi)),
        )

        # Set up TorchRL specs
        self.set_default_specs(
            direct_obs_dim=4,
            actions=2,
            with_pixels=self.render,
            pixels_only=False,
            discrete_actions=True,
        )

        if self.seed is not None:
            self.set_seed(self.seed)

    def _sample_initial_state(self, batch_shape: torch.Size) -> torch.Tensor:
        """Sample random initial states."""
        x = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(
            *self.cfg.init_x_range
        )
        x_dot = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(
            *self.cfg.init_x_dot_range
        )
        theta = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(
            *self._init_theta_range_rad
        )
        theta_dot = torch.empty(*batch_shape, 1, dtype=torch.float32, device=self.device).uniform_(
            *self._init_theta_dot_range_rad
        )
        return torch.cat([x, x_dot, theta, theta_dot], dim=-1)

    def _dynamics(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute next state using CartPole dynamics."""
        x, x_dot, theta, theta_dot = obs.unbind(-1)
        force = torch.where(action == 1, self.force_mag, -self.force_mag).to(torch.float32)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot.pow(2) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta.pow(2) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        return torch.stack([x, x_dot, theta, theta_dot], dim=-1)

    def _termination(self, obs: torch.Tensor, step_count: torch.Tensor) -> torch.Tensor:
        """Check termination conditions."""
        x, x_dot, theta, theta_dot = obs.unbind(-1)
        done = (
            (x.abs() > self.x_threshold)
            | (theta.abs() > self.theta_threshold)
            | (step_count >= (self.max_steps - 1))
        ).unsqueeze(-1)
        return done

    def _reward(self, obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute reward (survival bonus)."""
        return torch.ones_like(done, dtype=torch.float32, device=self.device)

    def _set_seed(self, seed: int) -> None:
        """Set random seed."""
        torch.manual_seed(int(seed))

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Reset the environment."""
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        state = self._sample_initial_state(bs)
        step_count = torch.zeros(*bs, dtype=torch.long, device=self.device)
        done = torch.zeros(*bs, 1, dtype=torch.bool, device=self.device)
        
        td_fields = {
            "observation": state,
            "step_count": step_count,
            "done": done,
        }
        
        if self.render:
            try:
                pixels = self.render_pixels(state)
                td_fields["pixels"] = pixels
            except Exception:
                pass
                
        return TensorDict(td_fields, batch_size=self.batch_size)

    @torch.no_grad()
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute one environment step."""
        obs = tensordict.get("observation", None)

        if obs is None:
            td0 = self._reset()
            obs = td0["observation"]
            step_count = td0["step_count"]
        else:
            step_count = tensordict.get(
                "step_count",
                torch.zeros_like(obs[..., 0], dtype=torch.long, device=self.device),
            )

        action = tensordict["action"].to(self.device)
        next_obs_raw = self._dynamics(obs, action)
        done = self._termination(next_obs_raw, step_count)
        reward = self._reward(next_obs_raw, done)

        pixels = None
        if self.render:
            try:
                pixels = self.render_pixels(next_obs_raw)
            except Exception:
                pixels = None

        if self.auto_reset:
            reset_state = self._sample_initial_state(next_obs_raw.shape[:-1])
            next_obs = torch.where(done, reset_state, next_obs_raw)
            next_step_count = torch.where(
                done.squeeze(-1), torch.zeros_like(step_count), step_count + 1
            )
        else:
            next_obs = next_obs_raw
            next_step_count = step_count + 1

        out_fields = {
            "observation": next_obs,
            "reward": reward,
            "done": done,
            "step_count": next_step_count,
        }

        if pixels is not None:
            out_fields["pixels"] = pixels

        return TensorDict(out_fields, batch_size=self.batch_size)
