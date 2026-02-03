"""Template environment renderer."""
import math

import torch

from ...config import PBRConfig
from ...renderer.renderer import PBRRenderer


class MyEnvRenderer(PBRRenderer):
    """
    Renderer for MyEnv environment.
    
    This class is responsible for:
    1. Creating the 3D scene (nodes, camera, lights)
    2. Updating visual state from simulation state via _step()
    
    The renderer runs independently from the physics simulation,
    receiving state tensors and updating the scene accordingly.
    """
    
    def __init__(self, cfg: PBRConfig | dict | None = None):
        super().__init__(cfg)
        
        num_scenes = int(self.cfg.num_scenes)
        instances_per_scene = 1  # Adjust based on your environment
        
        # === CREATE YOUR SCENE NODES ===
        # Example: a simple box
        # self.box = self.add_node(
        #     "models/box",
        #     model_pivot_relative_point=(0.5, 0.5, 0.5),
        #     model_scale=(1.0, 1.0, 1.0),
        #     instances_per_scene=instances_per_scene,
        #     shared_across_scenes=False,  # True if same across all scenes
        # )
        
        # === INITIALIZE POSITION/COLOR BUFFERS ===
        # self.positions = torch.zeros((num_scenes, instances_per_scene, 3), dtype=torch.float32)
        # self.colors = torch.ones((num_scenes, instances_per_scene, 4), dtype=torch.float32)
        # self.box.set_positions(self.positions)
        # self.box.set_colors(self.colors)
        
        # === SET UP CAMERA ===
        self.add_camera()
        self._pbr_cam.set_positions(torch.tensor([5, 5, 2], dtype=torch.float32))
        self._pbr_cam.look_at(torch.tensor([0, 0, 0], dtype=torch.float32))
        
        # === SET UP LIGHTING ===
        self.add_light()
        
        # REQUIRED: Call setup_environment() at the end
        self.setup_environment()

    def _step(self, state_batch: torch.Tensor | None = None):
        """
        Update the visual state from simulation state.
        
        This method is called by the renderer's step() function to sync
        the 3D scene with the current simulation state.
        
        Args:
            state_batch: Tensor of shape [B, obs_dim] containing the
                current observation/state for each environment instance.
                
        Example implementation:
            if state_batch is None:
                return
            
            state = torch.as_tensor(state_batch, dtype=torch.float32).detach().cpu()
            
            # Extract relevant state components
            x = state[:, 0:1]  # position x
            y = state[:, 1:2]  # position y
            
            # Update node positions
            self.positions[:, :, 0] = x
            self.positions[:, :, 1] = y
            self.box.set_positions(self.positions)
        """
        if state_batch is None:
            return
        
        # TODO: Implement your state-to-visual mapping here
        pass
