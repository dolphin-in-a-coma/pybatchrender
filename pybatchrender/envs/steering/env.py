# -*- coding: utf-8 -*-
"""Steering task RL environment following pybatchrender pattern."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from pybatchrender import PBREnv

if TYPE_CHECKING:
    from .config import SteeringConfig
    from .renderer import SteeringRenderer


class SteeringEnv(PBREnv):
    """RL environment for the steering/driving task.

    The agent controls a vehicle that must navigate through obstacles on a track.
    Hitting obstacles reduces speed and incurs a penalty. Successfully avoiding
    obstacles increases speed and earns rewards.

    State observations include:
    - player_x: Lateral position (-lane_width/2 to lane_width/2)
    - player_y: Forward position along track
    - forward_speed: Current forward velocity
    - grace_active: Whether in collision grace period (0 or 1)
    - next_obstacle_x: X position of next obstacle
    - next_obstacle_y: Y position of next obstacle

    Actions (discrete mode):
    - 0: Steer left
    - 1: No steering
    - 2: Steer right

    Actions (continuous mode):
    - Single float in [-1, 1] for steering direction
    """

    def __init__(
        self,
        renderer: SteeringRenderer,
        cfg: SteeringConfig | dict | None = None,
        **cfg_overrides,
    ) -> None:
        super().__init__(
            renderer=renderer,
            cfg=cfg,
            device=torch.device(cfg.device),
            batch_size=torch.Size([cfg.num_scenes]),
        )

        # Physics parameters from config
        self.speed_initial = float(cfg.speed_initial)
        self.minimal_speed = float(cfg.minimal_speed)
        self.speed_increment = float(cfg.speed_increment)
        self.speed_decrement = float(cfg.speed_decrement)
        self.grace_period = float(cfg.grace_period)
        self.steering_speed = float(cfg.steering_speed)
        self.tau = float(cfg.tau)

        # Scene layout
        self.lane_width = float(cfg.lane_width)
        self.player_half_width = float(cfg.player_dimensions[0]) * 0.5
        self.obstacle_half_width = float(cfg.obstacle_dimensions[0]) * 0.5
        self.collision_threshold = self.player_half_width + self.obstacle_half_width
        self.rail_half_width = float(cfg.rail_dimensions[0]) * 0.5

        # X movement limits
        limit = self.lane_width * 0.5 - self.player_half_width - self.rail_half_width
        self.x_min = -limit
        self.x_max = limit

        # Obstacle generation parameters
        self.obstacle_seed = cfg.obstacle_seed
        self.prob_gold = float(cfg.prob_gold)
        self.prob_cone = float(cfg.prob_cone)
        self.distance_to_first_obstacle = float(cfg.distance_to_first_obstacle)
        self.number_of_obstacles = int(cfg.number_of_obstacles)
        self.obstacle_y_spacing = float(cfg.obstacle_y_spacing)
        self.obstacle_x_positions = list(cfg.obstacle_x_position)

        # RL parameters
        self.max_steps = int(cfg.max_steps)
        self.auto_reset = bool(cfg.auto_reset)
        self.wheel_sensitivity = float(cfg.wheel_sensitivity)
        self.collision_penalty = float(cfg.collision_penalty)
        self.terminate_on_collision = bool(cfg.terminate_on_collision)
        self.terminate_on_track_end = bool(cfg.terminate_on_track_end)

        # Action bounds for continuous control
        self.action_low = float(cfg.action_low)
        self.action_high = float(cfg.action_high)

        self.seed_val = int(cfg.seed) if cfg.seed is not None else 0
        self.render_enabled = bool(cfg.render)

        # State tensors (will be initialized in reset)
        self._obstacles: torch.Tensor | None = None
        self._player_x: torch.Tensor | None = None
        self._player_y: torch.Tensor | None = None
        self._forward_speed: torch.Tensor | None = None
        self._grace_remaining: torch.Tensor | None = None
        self._next_obstacle_idx: torch.Tensor | None = None
        self._prev_player_x: torch.Tensor | None = None
        self._prev_player_y: torch.Tensor | None = None

        # Configure specs
        discrete = cfg.action_type == "discrete"
        obs_dim = int(cfg.direct_obs_dim)
        action_n = int(cfg.action_n)

        self._setup_specs(obs_dim, action_n, discrete)

        if self.seed_val is not None:
            self.set_seed(self.seed_val)

    def _setup_specs(self, obs_dim: int, action_n: int, discrete: bool) -> None:
        """Configure observation and action specs."""
        from torchrl.data.tensor_specs import Composite, Unbounded, Categorical, Bounded

        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        fields: dict[str, Unbounded] = {}

        # Pixel observation
        if self.render_enabled:
            C = int(self.cfg.num_channels)
            W = int(self.cfg.tile_resolution[0])
            H = int(self.cfg.tile_resolution[1])
            fields["pixels"] = Unbounded(
                shape=bs + torch.Size([C, H, W]), dtype=torch.uint8, device=self.device
            )

        # Direct observation
        fields["observation"] = Unbounded(
            shape=bs + torch.Size([obs_dim]), dtype=torch.float32, device=self.device
        )
        self.observation_spec = Composite(**fields, shape=bs)

        # Action spec
        if discrete:
            self.action_spec = Categorical(
                n=action_n, shape=bs, dtype=torch.long, device=self.device
            )
        else:
            # Continuous action with bounds matching wheel input range
            self.action_spec = Bounded(
                low=self.action_low,
                high=self.action_high,
                shape=bs + torch.Size([action_n]),
                dtype=torch.float32,
                device=self.device,
            )

        # Reward / Done
        self.reward_spec = Unbounded(
            shape=bs + torch.Size([1]), dtype=torch.float32, device=self.device
        )
        self.done_spec = Unbounded(
            shape=bs + torch.Size([1]), dtype=torch.bool, device=self.device
        )

    def _generate_obstacles(self) -> torch.Tensor:
        """Generate obstacle tensor for all batch elements.

        Returns:
            Tensor of shape (batch, num_obstacles, 4) where each obstacle is
            [x, y, gold_flag, cone_flag].
        """
        B = int(self.cfg.num_scenes)
        N = self.number_of_obstacles

        obstacles = torch.zeros(B, N, 4, dtype=torch.float32, device=self.device)

        # X positions (random from allowed positions)
        x_choices = torch.tensor(self.obstacle_x_positions, dtype=torch.float32, device=self.device)
        x_idx = torch.randint(0, len(x_choices), (B, N), device=self.device)
        obstacles[:, :, 0] = x_choices[x_idx]

        # Y positions (evenly spaced)
        y_offsets = torch.arange(N, dtype=torch.float32, device=self.device)
        obstacles[:, :, 1] = y_offsets * self.obstacle_y_spacing + self.distance_to_first_obstacle

        # Gold flags
        gold_rand = torch.rand(B, N, device=self.device)
        obstacles[:, :, 2] = (gold_rand < self.prob_gold).float()

        # Cone flags
        cone_rand = torch.rand(B, N, device=self.device)
        obstacles[:, :, 3] = (cone_rand < self.prob_cone).float()

        return obstacles

    def _get_observation(self) -> torch.Tensor:
        """Build observation tensor from current state.

        Returns:
            Tensor of shape (batch, obs_dim) with:
            [player_x, player_y, forward_speed, grace_active, next_obs_x, next_obs_y]
        """
        B = int(self.cfg.num_scenes)

        # Get next obstacle info
        next_obs_x = torch.zeros(B, device=self.device)
        next_obs_y = torch.zeros(B, device=self.device)

        if self._obstacles is not None:
            max_obs = self._obstacles.shape[1]
            batch_idx = torch.arange(B, device=self.device)
            valid_mask = self._next_obstacle_idx < max_obs
            valid_idx = torch.clamp(self._next_obstacle_idx, max=max_obs - 1)

            next_obs_data = self._obstacles[batch_idx, valid_idx]
            next_obs_x = torch.where(valid_mask, next_obs_data[:, 0], torch.zeros_like(next_obs_x))
            next_obs_y = torch.where(valid_mask, next_obs_data[:, 1], self._player_y + 1000.0)

        grace_active = (self._grace_remaining > 0).float()

        obs = torch.stack(
            [
                self._player_x,
                self._player_y,
                self._forward_speed,
                grace_active,
                next_obs_x,
                next_obs_y,
            ],
            dim=-1,
        )
        return obs

    def _check_collisions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check for obstacle collisions.

        Returns:
            collisions: Number of collisions per batch element
            passages: Number of safe passages per batch element
        """
        B = int(self.cfg.num_scenes)
        collisions = torch.zeros(B, dtype=torch.int32, device=self.device)
        passages = torch.zeros(B, dtype=torch.int32, device=self.device)

        if self._obstacles is None or self._obstacles.numel() == 0:
            return collisions, passages

        max_obstacles = self._obstacles.shape[1]
        batch_idx = torch.arange(B, dtype=torch.long, device=self.device)

        # Process obstacles that player has passed
        while True:
            active_mask = self._next_obstacle_idx < max_obstacles
            if not active_mask.any():
                break

            current_idx = self._next_obstacle_idx[active_mask]
            active_batch = batch_idx[active_mask]

            next_obs = self._obstacles[active_batch, current_idx]
            obs_y = next_obs[:, 1]
            player_y_active = self._player_y[active_mask]

            passed_mask = obs_y <= player_y_active
            if not passed_mask.any():
                break

            passed_batch = active_batch[passed_mask]
            passed_obs = next_obs[passed_mask]

            self._next_obstacle_idx[passed_batch] += 1
            passages[passed_batch] += 1

            # Interpolate player X at obstacle Y for accurate collision detection
            prev_y = self._prev_player_y[passed_batch]
            curr_y = self._player_y[passed_batch]
            denom = curr_y - prev_y
            denom = torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom)
            t = (passed_obs[:, 1] - prev_y) / denom

            prev_x = self._prev_player_x[passed_batch]
            curr_x = self._player_x[passed_batch]
            player_x_at_obs = prev_x + (curr_x - prev_x) * t

            x_distance = (player_x_at_obs - passed_obs[:, 0]).abs()
            hit = (x_distance <= self.collision_threshold).int()
            collisions[passed_batch] += hit

        return collisions, passages

    def _dynamics(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply physics dynamics for one timestep.

        Args:
            action: Action tensor, continuous (batch, 1) in range [action_low, action_high]
                    representing wheel/steering input

        Returns:
            crashes: Boolean tensor of collisions this step
            reached_end: Whether track end was reached
        """
        # Store previous position for collision interpolation
        self._prev_player_x = self._player_x.clone()
        self._prev_player_y = self._player_y.clone()

        # Decrement grace period
        self._grace_remaining = torch.clamp(self._grace_remaining - self.tau, min=0.0)

        # Forward movement
        self._player_y = self._player_y + self._forward_speed * self.tau

        # Lateral movement (steering) - matches game formula:
        # lateral_delta = action * forward_speed / wheel_sensitivity * dt
        if action.dim() == 1:
            steering_input = action.float()
        else:
            steering_input = action.squeeze(-1)

        lateral_delta = steering_input * self._forward_speed / self.wheel_sensitivity * self.tau
        new_x = torch.clamp(self._player_x + lateral_delta, self.x_min, self.x_max)
        self._player_x = new_x

        # Check collisions
        collisions, passages = self._check_collisions()

        # Apply collision effects (only if not in grace period)
        crashes = (collisions > 0) & (self._grace_remaining <= 0)
        self._grace_remaining = torch.where(
            crashes, torch.full_like(self._grace_remaining, self.grace_period), self._grace_remaining
        )

        # Speed changes
        self._forward_speed = self._forward_speed + passages.float() * self.speed_increment
        self._forward_speed = self._forward_speed - crashes.float() * self.speed_decrement
        self._forward_speed = torch.clamp(self._forward_speed, min=self.minimal_speed)

        # Check if reached track end
        reached_end = self._next_obstacle_idx >= self.number_of_obstacles

        return crashes, reached_end

    def _termination(self, crashes: torch.Tensor, reached_end: torch.Tensor, step_count: torch.Tensor) -> torch.Tensor:
        """Check termination conditions.

        Returns:
            done: Boolean tensor indicating which environments are done
        """
        done = torch.zeros_like(step_count, dtype=torch.bool)

        if self.terminate_on_collision:
            done = done | (crashes > 0)
        if self.terminate_on_track_end:
            done = done | reached_end
        done = done | (step_count >= self.max_steps - 1)

        return done.unsqueeze(-1)

    def _reward(self, crashes: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute sparse rewards: only penalty on collision.

        Returns:
            reward: Reward tensor of shape (batch, 1)
        """
        # Sparse reward: 0 normally, collision_penalty (-1) on crash
        reward = crashes.float() * self.collision_penalty
        return reward.unsqueeze(-1)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        random.seed(int(seed))

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Reset the environment."""
        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        B = int(bs[0]) if len(bs) > 0 else 1

        # Generate new obstacles
        self._obstacles = self._generate_obstacles()

        # Build obstacle visuals in renderer
        self._renderer.build_obstacles(self._obstacles)

        # Reset state
        self._player_x = torch.zeros(B, device=self.device)
        self._player_y = torch.zeros(B, device=self.device)
        self._prev_player_x = torch.zeros(B, device=self.device)
        self._prev_player_y = torch.zeros(B, device=self.device)
        self._forward_speed = torch.full((B,), self.speed_initial, device=self.device)
        self._grace_remaining = torch.zeros(B, device=self.device)
        self._next_obstacle_idx = torch.zeros(B, dtype=torch.long, device=self.device)

        step_count = torch.zeros(*bs, dtype=torch.long, device=self.device)
        done = torch.zeros(*bs, 1, dtype=torch.bool, device=self.device)
        observation = self._get_observation()

        td_fields = {
            "observation": observation,
            "step_count": step_count,
            "done": done,
        }

        if self.render_enabled:
            try:
                pixels = self.render_pixels(observation)
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

        # Apply dynamics
        crashes, reached_end = self._dynamics(action)

        # Get next observation
        next_obs = self._get_observation()

        # Check termination
        done = self._termination(crashes, reached_end, step_count)

        # Compute reward
        reward = self._reward(crashes, done)

        # Render if enabled
        pixels = None
        if self.render_enabled:
            try:
                pixels = self.render_pixels(next_obs)
            except Exception:
                pass

        # Auto-reset if needed
        if self.auto_reset:
            reset_mask = done.squeeze(-1)
            if reset_mask.any():
                # For simplicity, do full reset if any env is done
                # TODO: Implement partial reset for efficiency
                reset_obs = self._get_observation()  # After potential reset
                next_obs = torch.where(done, reset_obs, next_obs)
                next_step_count = torch.where(reset_mask, torch.zeros_like(step_count), step_count + 1)
            else:
                next_step_count = step_count + 1
        else:
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


# Demo/test when run as module: python -m cogcarsim.envs.steering.env
if __name__ == "__main__":
    import time
    from cogcarsim.envs.steering.config import SteeringConfig
    from cogcarsim.envs.steering.renderer import SteeringRenderer

    print("=== CogCarSim Steering RL Environment Demo ===\n")

    num_scenes = 64
    tile_resolution = (64, 64)

    config = SteeringConfig(
        num_scenes=num_scenes,
        tile_resolution=tile_resolution,
        number_of_obstacles=100,
        distance_to_first_obstacle=50.0,
        render=True,
        offscreen=True,
    )

    print(f"Creating environment with {num_scenes} parallel scenes...")
    renderer = SteeringRenderer(config)
    env = SteeringEnv(renderer=renderer, cfg=config)

    print(f"Batch size: {env.batch_size}")
    print(f"Action space: continuous [{config.action_low}, {config.action_high}]")
    print(f"Wheel sensitivity: {config.wheel_sensitivity}")
    print(f"Reward: sparse (only {config.collision_penalty} on collision)")

    td = env.reset()
    print(f"\nObservation shape: {td['observation'].shape}")
    print(f"Observation: [player_x, player_y, speed, grace, next_obs_x, next_obs_y]")
    print(f"Initial obs[0]: {[f'{v:.1f}' for v in td['observation'][0].tolist()]}")

    if "pixels" in td.keys():
        print(f"Pixels shape: {td['pixels'].shape}")

    total_reward = 0.0
    crash_count = 0
    tm_start = time.time()
    steps = 500
    total_frames = steps * num_scenes

    print(f"\nRunning {steps} steps with random actions...")
    for t in range(steps):
        action = env.action_spec.rand()
        td["action"] = action
        td = env.step(td)

        step_reward = td["next", "reward"].sum().item()
        total_reward += step_reward
        if step_reward < 0:
            crash_count += 1

        td = td["next"]

        if t % 100 == 0:
            obs = td['observation'][0]
            print(f"  Step {t:3d}: y={obs[1]:6.1f}, speed={obs[2]:5.1f}, crashes={crash_count}")

    tm_end = time.time()
    fps = total_frames / (tm_end - tm_start)

    print(f"\n=== Results ===")
    print(f"Total reward: {total_reward:.1f} (sparse: {int(-total_reward)} collisions penalized)")
    print(f"Crash events (outside grace): {crash_count}")
    print(f"Final position Y: {td['observation'][0, 1]:.1f}")
    print(f"Final speed: {td['observation'][0, 2]:.1f}")
    print(f"Time: {tm_end - tm_start:.2f}s")
    print(f"FPS: {fps:.0f} (frames/sec across all scenes)")
