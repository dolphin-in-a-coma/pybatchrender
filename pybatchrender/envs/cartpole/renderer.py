"""CartPole environment renderer."""
import math

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class CartPoleRenderer(PBRRenderer):
    """
    Renderer for the CartPole environment.
    
    Creates a 3D scene with a rail, cart, and pole. The cart moves along the rail
    and the pole rotates on top of the cart.
    """
    
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        instances_per_scene = 1
        
        # Geometry sizes
        self.rail_size = (6.0, 0.05, 0.05)
        self.cart_size = (1.2, 0.8, 0.5)
        self.pole_size = (0.1, 0.1, 2.0)
        
        # Colors
        self.rail_pos_color = (0.2, 0.2, 0.2, 1.0)
        self.cart_pos_color_range = ((0.6, 0.8, 1.0, 1.0), (1.0, 0.6, 0.8, 1.0))
        self.pole_pos_color = (1.0, 0.7, 0.2, 1.0)

        # Create nodes
        self.rail = self.add_node(
            "models/box",
            model_pivot_relative_point=(0.5, 0.5, 0.5),
            model_scale=self.rail_size,
            instances_per_scene=instances_per_scene,
            shared_across_scenes=True,
        )
        self.cart = self.add_node(
            "models/box",
            model_pivot_relative_point=(0.5, 0.5, 0.5),
            model_scale=self.cart_size,
            instances_per_scene=instances_per_scene,
            shared_across_scenes=False,
        )
        self.pole = self.add_node(
            "models/box",
            model_pivot_relative_point=(0.5, 0.5, 0.05),
            model_scale=self.pole_size,
            instances_per_scene=instances_per_scene,
            shared_across_scenes=False,
        )

        # Initialize position buffers
        num_scenes = int(self.cfg.num_scenes)
        self.rail_pos = torch.zeros((1, instances_per_scene, 3), dtype=torch.float32)
        self.cart_pos = torch.zeros((num_scenes, instances_per_scene, 3), dtype=torch.float32)
        self.pole_pos = self.cart_pos.clone() + torch.tensor(
            [0, (self.cart_size[1] + self.pole_size[1]) * 0.5, 0], dtype=torch.float32
        )
        self.rail.set_positions(self.rail_pos)
        self.cart.set_positions(self.cart_pos)
        self.pole.set_positions(self.pole_pos)

        # Initialize color buffers
        self.rail_base_color = torch.ones((1, instances_per_scene, 4), dtype=torch.float32) * torch.tensor(
            self.rail_pos_color, dtype=torch.float32
        )
        start_col = torch.tensor(self.cart_pos_color_range[0], dtype=torch.float32)
        end_col = torch.tensor(self.cart_pos_color_range[1], dtype=torch.float32)
        t = torch.linspace(0.0, 1.0, steps=num_scenes, dtype=torch.float32)
        self.cart_base_color = start_col.unsqueeze(0) + (end_col - start_col).unsqueeze(0) * t.unsqueeze(1)
        self.pole_base_color = torch.ones((num_scenes, instances_per_scene, 4), dtype=torch.float32) * torch.tensor(
            self.pole_pos_color, dtype=torch.float32
        )
        self.rail.set_colors(self.rail_base_color)
        self.cart.set_colors(self.cart_base_color)
        self.pole.set_colors(self.pole_base_color)

        # Initialize rotation buffer
        self.pole_hpr = torch.zeros((num_scenes, instances_per_scene, 3), dtype=torch.float32)
        self.pole_hpr[:, :, 1] = math.pi * 0.5
        self.pole.set_hprs(self.pole_hpr)

        # Buffers mapped from state
        self.cart_x_pos = self.cart_pos[:, :, 0:1].clone()
        self.pole_theta = torch.zeros_like(self.cart_x_pos)

        # Camera and lighting
        self.add_camera()
        self._pbr_cam.set_positions(torch.tensor([5, 5, 2], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0, 0, 0], dtype=torch.float32))
        self.add_light()
        
        self.setup_environment()

    def _step(self, state_batch: torch.Tensor | None = None):
        """
        Update the visual state from simulation state.
        
        Args:
            state_batch: Tensor of shape [B, 4] where:
                - state[:, 0] = cart x position
                - state[:, 1] = cart x velocity (unused for rendering)
                - state[:, 2] = pole angle (theta)
                - state[:, 3] = pole angular velocity (unused for rendering)
        """
        if state_batch is None:
            return
            
        state_t = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
        B = int(state_t.shape[0])
        N = int(self.cfg.num_scenes)
        
        # Truncate/expand to num_scenes
        if B != N:
            if B < N:
                pad = state_t[-1:].repeat(N - B, 1)
                state_t = torch.cat([state_t, pad], dim=0)
            else:
                state_t = state_t[:N]
        
        # Extract x and theta
        x = state_t[:, 0:1]
        theta = state_t[:, 2:3]
        
        # Update cart position
        self.cart_x_pos[:, :, 0] = x
        self.cart_pos[:, :, 0:1] = self.cart_x_pos
        self.cart.set_positions(self.cart_pos)
        
        # Update pole position and rotation
        self.pole_pos[:, :, 0:1] = self.cart_x_pos
        self.pole.set_positions(self.pole_pos, lazy=True)
        self.pole_theta[:, :, 0] = theta
        self.pole_hpr[:, :, 1:2] = self.pole_theta
        self.pole.set_hprs(self.pole_hpr)
