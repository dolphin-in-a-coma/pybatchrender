"""Template TorchRL environment."""
import math

import torch
from tensordict import TensorDict

from ...env import PBREnv
from .config import MyEnvConfig
from .renderer import MyEnvRenderer


class MyEnv(PBREnv):
    """
    Template environment compatible with TorchRL.
    
    This class implements the core environment logic:
    - State initialization and reset
    - Physics/dynamics simulation
    - Reward computation
    - Termination conditions
    
    Observation Space:
        Describe your observation space here.
        Example: [x, x_dot, theta, theta_dot]
        
    Action Space:
        Describe your action space here.
        Example: Discrete(2) - left/right force
        
    Reward:
        Describe your reward function here.
        Example: +1 for each timestep survived
        
    Termination:
        Describe termination conditions here.
        Example: Episode ends when angle > threshold
    """
    
    def __init__(
        self,
        renderer: MyEnvRenderer,
        cfg: MyEnvConfig | dict | None = None,
        **cfg_overrides,
    ):
        super().__init__(
            renderer=renderer,
            cfg=cfg,
            device=torch.device(cfg.device),
            batch_size=torch.Size([cfg.num_scenes]),
        )

        # === STORE PHYSICS/ENV PARAMETERS FROM CONFIG ===
        # self.gravity = float(cfg.gravity if cfg is not None else 9.8)
        # self.mass = float(cfg.mass if cfg is not None else 1.0)
        
        self.max_steps = int(cfg.max_steps if cfg is not None else 500)
        self.auto_reset = bool(cfg.auto_reset if cfg is not None else True)
        self.seed = int(cfg.seed if cfg is not None else 0)
        self.render = bool(cfg.render if cfg is not None else True)

        # === SET UP TORCHRL SPECS ===
        # Adjust these based on your environment
        self.set_default_specs(
            direct_obs_dim=4,           # Observation dimension
            actions=2,                   # Number of discrete actions or continuous action dim
            with_pixels=self.render,     # Include pixel observations
            pixels_only=False,           # If True, only pixel observations (no state)
            discrete_actions=True,       # True for discrete, False for continuous
        )

        if self.seed is not None:
            self.set_seed(self.seed)

    def _sample_initial_state(self, batch_shape: torch.Size) -> torch.Tensor:
        """
        Sample random initial states for reset.
        
        Args:
            batch_shape: Shape of the batch (usually self.batch_size)
            
        Returns:
            Tensor of shape [*batch_shape, obs_dim] with initial states
        """
        # TODO: Implement your initial state sampling
        # Example:
        # x = torch.empty(*batch_shape, 1, device=self.device).uniform_(-1, 1)
        # x_dot = torch.empty(*batch_shape, 1, device=self.device).uniform_(-0.5, 0.5)
        # return torch.cat([x, x_dot, ...], dim=-1)
        
        return torch.zeros(*batch_shape, 4, dtype=torch.float32, device=self.device)

    def _dynamics(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute next state given current state and action.
        
        Args:
            obs: Current observation tensor [B, obs_dim]
            action: Action tensor [B] (discrete) or [B, action_dim] (continuous)
            
        Returns:
            Next observation tensor [B, obs_dim]
        """
        # TODO: Implement your dynamics/physics
        # Example for simple integrator:
        # x, x_dot = obs[..., 0], obs[..., 1]
        # force = torch.where(action == 1, 1.0, -1.0)
        # x_dot_new = x_dot + force * self.dt
        # x_new = x + x_dot_new * self.dt
        # return torch.stack([x_new, x_dot_new], dim=-1)
        
        return obs  # Placeholder: no dynamics

    def _termination(self, obs: torch.Tensor, step_count: torch.Tensor) -> torch.Tensor:
        """
        Check termination conditions.
        
        Args:
            obs: Current observation tensor [B, obs_dim]
            step_count: Current step count tensor [B]
            
        Returns:
            Boolean tensor [B, 1] indicating termination
        """
        # TODO: Implement your termination conditions
        # Example:
        # x = obs[..., 0]
        # done = (x.abs() > self.x_threshold) | (step_count >= self.max_steps - 1)
        # return done.unsqueeze(-1)
        
        done = step_count >= (self.max_steps - 1)
        return done.unsqueeze(-1)

    def _reward(self, obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """
        Compute reward.
        
        Args:
            obs: Current observation tensor [B, obs_dim]
            done: Termination tensor [B, 1]
            
        Returns:
            Reward tensor [B, 1]
        """
        # TODO: Implement your reward function
        # Example: survival reward
        return torch.ones_like(done, dtype=torch.float32, device=self.device)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(int(seed))

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        """
        Reset the environment.
        
        Args:
            tensordict: Optional tensordict (unused, for TorchRL compatibility)
            
        Returns:
            TensorDict with initial observation, step_count, done, and optionally pixels
        """
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
        """
        Execute one environment step.
        
        Args:
            tensordict: TensorDict containing "observation" and "action"
            
        Returns:
            TensorDict with next observation, reward, done, step_count, and optionally pixels
        """
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
