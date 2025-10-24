from abc import ABC, abstractmethod
import math
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import Composite, Unbounded, Categorical

# Optional renderer / config (kept optional to avoid hard deps when not rendering)
from .config import P3DConfig
from .renderer.renderer import P3DRenderer

class P3DEnv(EnvBase, ABC):

    def __init__(
        self,
        renderer: P3DRenderer,
        cfg: P3DConfig | None = None,
        device: str | torch.device = "cpu",
        batch_size: torch.Size = torch.Size([128]),
        **kwargs,
    ) -> None:
        super().__init__(device=torch.device(device), batch_size=batch_size, **kwargs)
        self._renderer = renderer
        self.cfg = cfg if cfg is not None else renderer.cfg
        
    # Minimal helpers to configure specs
    def set_default_specs(
        self,
        *,
        direct_obs_dim: int | None = None,
        actions: int | None = None,
        with_pixels: bool = False,
        pixels_only: bool = False,
        discrete_actions: bool = True,
    ) -> None:
        assert with_pixels or not pixels_only, "Cannot set both with_pixels and pixels_only to True."

        bs = self.batch_size if self.batch_size != torch.Size([]) else torch.Size([1])
        fields: dict[str, Unbounded] = {}

        # Pixel observation
        if with_pixels:
            C = int(self.cfg.num_channels)
            W = int(self.cfg.tile_resolution[0])
            H = int(self.cfg.tile_resolution[1])
            fields["pixels"] = Unbounded(
                shape=bs + torch.Size([C, H, W]), dtype=torch.uint8, device=self.device
            )

        # Direct observation
        if not pixels_only:
            if direct_obs_dim is None:
                raise ValueError("direct_obs_dim must be provided when pixels_only=False.")
            fields["observation"] = Unbounded(
                shape=bs + torch.Size([int(direct_obs_dim)]), dtype=torch.float32, device=self.device
            )
        self.observation_spec = Composite(**fields, shape=bs)

        # Actions
        if actions is None:
            raise ValueError("actions must be provided.")
        if discrete_actions:
            self.action_spec = Categorical(
                n=int(actions), shape=bs, dtype=torch.long, device=self.device
            )
        else:
            self.action_spec = Unbounded(
                shape=bs + torch.Size([int(actions)]), dtype=torch.float32, device=self.device
            )
        # Reward / Done
        self.reward_spec = Unbounded(
            shape=bs + torch.Size([1]), dtype=torch.float32, device=self.device
        )
        self.done_spec = Unbounded(
            shape=bs + torch.Size([1]), dtype=torch.bool, device=self.device
        )

    @abstractmethod
    def _step(self, tensordict: TensorDict) -> TensorDict:
        pass

    @abstractmethod
    def _reset(self) -> TensorDict:
        pass

    # @abstractmethod
    # def _render(self) -> torch.Tensor:
    #     pass

    # # ---- Rendering helpers ----
    # def _num_instances(self) -> int:
    #     if self.batch_size == torch.Size([]):
    #         return 1
    #     n = 1
    #     for d in self.batch_size:
    #         n *= int(d)
    #     return max(1, n)

    # # Base class assumes renderer is constructed and passed in.
    # def sync_renderer_from_observation(self, obs: torch.Tensor) -> None:
    #     """
    #     Optional hook to be overridden by subclasses to reflect the current observation
    #     into the renderer's scene graph (e.g., set node transforms/colors per instance).
    #     Default is a no-op.
    #     """
    #     return None

    # def render_pixels(self, sync_from_obs: bool = True) -> torch.Tensor:
    #     if self._renderer is None:
    #         raise RuntimeError("Renderer is not initialized. Construct env with a renderer and pass it to P3DEnv.")
    #     # if sync_from_obs:
    #     #     try:
    #     #         obs = getattr(self, "_last_obs", None)
    #     #         if obs is not None:
    #     #             self.sync_renderer_from_observation(obs)
    #     #     except Exception:
    #     #         pass
    #     img = self._renderer.step()
    #     return img

    # # Utility to cache last observation
    # def __setattr__(self, key, value):
    #     super().__setattr__(key, value)
    #     if key == "_last_td" and isinstance(value, TensorDict):
    #         try:
    #             self._last_obs = value.get("observation", None)
    #         except Exception:
    #             self._last_obs = None


