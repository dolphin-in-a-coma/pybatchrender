from abc import ABC, abstractmethod
import math
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.data.tensor_specs import Composite, Unbounded, Categorical

# Optional renderer / config (kept optional to avoid hard deps when not rendering)
try:
    from .config import PBRConfig
    from .renderer.renderer import PBRRenderer
except ImportError:
    # Fallback when run as a script (no package parent)
    from config import PBRConfig
    from renderer.renderer import PBRRenderer

class PBREnv(EnvBase, ABC):

    def __init__(
        self,
        renderer: PBRRenderer,
        cfg: PBRConfig | None = None,
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
    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        pass

    def render_pixels(self, obs: torch.Tensor | None = None) -> torch.Tensor:
        if self._renderer is None:
            raise RuntimeError("Renderer is not initialized. Construct env with a renderer and pass it to PBREnv.")
        return self._renderer.step(obs)

    def save_batch_examples(
        self,
        *,
        pixels: torch.Tensor | None = None,
        obs: torch.Tensor | None = None,
        indices: list[int] | None = None,
        num: int = 16,
        scale: int = 1,
        out_dir: str | None = './outputs',
        filename_prefix: str = "batch_example",
    ) -> str:
        """
        Save a single image grid composed of N batch elements.

        Args:
            pixels: Pre-rendered pixels [B, C, H, W] (or None to render from obs)
            obs: Observation tensor to render (if pixels is None)
            indices: Specific batch indices to include (or None for first `num`)
            num: Number of examples if indices not provided
            scale: Scale factor for output (2 = 2x larger)
            out_dir: Output directory for the image
            filename_prefix: Prefix for the output filename

        Returns:
            The absolute path of the saved image.
        """
        import time
        import random
        from pathlib import Path

        if self.cfg.num_workers > 1 and self.cfg.worker_index != 0:
            return None

        # Acquire pixel batch [B, C, H, W]
        if pixels is None:
            if obs is None:
                pixels = self.render_pixels(None)
            else:
                pixels = self.render_pixels(obs)

        if pixels is None:
            raise RuntimeError("No pixels available to save.")

        # Create grid using shared helper
        canvas = self._make_grid_frame(pixels, indices, num, scale)

        # Output path and filename
        ts = time.strftime("%Y%m%d_%H%M%S")
        rand_code = f"{random.randint(0, 999999):06d}"
        ext = "png"
        out_path = Path(out_dir) if out_dir is not None else Path(".")
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"{filename_prefix}_{ts}_{rand_code}.{ext}"
        fpath = (out_path / fname).resolve()

        # Save via PIL if available, fallback to imageio or matplotlib
        try:
            from PIL import Image
            Image.fromarray(canvas).save(str(fpath))
        except Exception:
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(fpath), canvas)
            except Exception:
                try:
                    import matplotlib.pyplot as plt
                    plt.imsave(str(fpath), canvas)
                except Exception as e:
                    raise RuntimeError(f"Failed to save image: {e}")

        return str(fpath)

    def _make_grid_frame(
        self,
        pixels: torch.Tensor,
        indices: list[int] | None = None,
        num: int = 16,
        scale: int = 1,
    ) -> "np.ndarray":
        """
        Create a grid image from batch pixels.
        
        Args:
            pixels: Tensor of shape [B, C, H, W]
            indices: Specific batch indices to include (or None for first `num`)
            num: Number of examples if indices not provided
            scale: Scale factor for output (2 = 2x larger)
            
        Returns:
            numpy array of shape [grid_H, grid_W, C] as uint8
        """
        import numpy as np
        
        if pixels.dim() == 5:
            pixels = torch.flatten(pixels, start_dim=0, end_dim=1)
        
        B, C, H, W = pixels.shape
        k = min(max(1, int(num)), int(B))
        
        # Select indices
        if indices is not None and len(indices) > 0:
            sel = [int(i) for i in indices if 0 <= int(i) < B][:k]
        else:
            sel = list(range(k))
        
        imgs = pixels[sel].detach().cpu()
        if imgs.dtype != torch.uint8:
            imgs = imgs.clamp(0, 255).to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).numpy()  # [k, H, W, C]
        
        # Build grid
        rows = int(math.ceil(math.sqrt(k)))
        cols = int(math.ceil(k / rows))
        canvas = np.zeros((rows * H, cols * W, C), dtype=np.uint8)
        
        for n in range(len(sel)):
            r, c = n // cols, n % cols
            canvas[r*H:(r+1)*H, c*W:(c+1)*W] = imgs[n]
        
        # Scale up if requested
        if scale > 1:
            from PIL import Image
            img = Image.fromarray(canvas)
            new_size = (canvas.shape[1] * scale, canvas.shape[0] * scale)
            img = img.resize(new_size, Image.NEAREST)
            canvas = np.array(img)
        
        return canvas

    def save_batch_gif(
        self,
        *,
        num_steps: int = 100,
        frame_interval: int = 2,
        indices: list[int] | None = None,
        num: int = 16,
        scale: int = 2,
        duration_ms: int = 100,
        out_dir: str | None = "./outputs",
        filename_prefix: str = "batch_animation",
        return_bytes: bool = False,
    ) -> str | tuple[str, bytes]:
        """
        Run the environment and save an animated GIF of the batch.
        
        Args:
            num_steps: Number of environment steps to run
            frame_interval: Capture a frame every N steps
            indices: Specific batch indices to include (or None for first `num`)
            num: Number of examples if indices not provided
            scale: Scale factor for output (2 = 2x larger pixels)
            duration_ms: Milliseconds per frame in GIF
            out_dir: Output directory for the GIF
            filename_prefix: Prefix for the output filename
            return_bytes: If True, also return GIF bytes for notebook display
            
        Returns:
            Path to saved GIF, or (path, bytes) if return_bytes=True
            
        Example:
            # Save GIF and display in notebook
            path, gif_bytes = env.save_batch_gif(num_steps=100, return_bytes=True)
            from IPython.display import display, Image
            display(Image(data=gif_bytes, format='gif'))
        """
        import io
        import time
        import random
        from pathlib import Path
        
        if self.cfg.num_workers > 1 and self.cfg.worker_index != 0:
            return None
        
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("PIL is required for GIF creation. Install with: pip install Pillow")
        
        frames = []
        
        # Reset environment
        td = self._reset()
        
        # Capture initial frame
        pixels = td.get("pixels", None)
        if pixels is None:
            pixels = self.render_pixels(td.get("observation", None))
        if pixels is not None:
            frames.append(self._make_grid_frame(pixels, indices, num, scale))
        
        # Run steps and capture frames
        for step in range(1, num_steps + 1):
            td["action"] = self.action_spec.rand()
            td = self.step(td)
            td = td["next"]
            
            if step % frame_interval == 0:
                pixels = td.get("pixels", None)
                if pixels is None:
                    pixels = self.render_pixels(td.get("observation", None))
                if pixels is not None:
                    frames.append(self._make_grid_frame(pixels, indices, num, scale))
        
        if not frames:
            raise RuntimeError("No frames captured. Ensure rendering is enabled.")
        
        # Convert to PIL images
        pil_frames = [Image.fromarray(f) for f in frames]
        
        # Save GIF
        ts = time.strftime("%Y%m%d_%H%M%S")
        rand_code = f"{random.randint(0, 999999):06d}"
        out_path = Path(out_dir) if out_dir is not None else Path(".")
        out_path.mkdir(parents=True, exist_ok=True)
        fname = f"{filename_prefix}_{ts}_{rand_code}.gif"
        fpath = (out_path / fname).resolve()
        
        # Save to file
        pil_frames[0].save(
            str(fpath),
            format='GIF',
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        
        if return_bytes:
            # Also create bytes for notebook display
            buffer = io.BytesIO()
            pil_frames[0].save(
                buffer,
                format='GIF',
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
            )
            buffer.seek(0)
            return str(fpath), buffer.getvalue()
        
        return str(fpath)

    @classmethod
    def make_parallel_env(
        cls,
        *,
        config: "PBRConfig",
        renderer_cls: type["PBRRenderer"],
        num_workers: int,
        mp_start_method: str = "spawn",
        shared_memory: bool = True,
    ) -> ParallelEnv:
        """
        Generic factory to create a TorchRL ParallelEnv for any PBREnv subclass.
        
        Each worker process creates its own renderer instance, avoiding the
        Panda3D ShowBase singleton limitation.
        
        Args:
            config: Configuration object for the environment
            renderer_cls: The PBRRenderer subclass to instantiate per worker
            num_workers: Number of parallel worker processes
            mp_start_method: Multiprocessing start method. MUST be "spawn" for
                Panda3D/OpenGL compatibility. "fork" and "forkserver" are NOT
                supported and will cause crashes.
            shared_memory: Whether to use shared memory for tensor data
            
        Returns:
            TorchRL ParallelEnv instance
            
        Note:
            Only "spawn" is supported because OpenGL contexts cannot be forked.
        """
        import multiprocessing as mp

        try:
            mp.set_start_method(mp_start_method, force=True)
        except RuntimeError:
            # start method already set in this process
            pass

        create_env_kwargs = [
            {
                "env_cls": cls,
                "renderer_cls": renderer_cls,
                "config": config,
                "worker_index": i,
                "num_workers": num_workers,
            }
            for i in range(int(num_workers))
        ]

        return ParallelEnv(
            int(num_workers),
            create_env_fn=cls._make_env_worker,
            shared_memory=shared_memory,
            mp_start_method=mp_start_method,
            create_env_kwargs=create_env_kwargs,
        )

    @staticmethod
    def _make_env_worker(
        env_cls: type["PBREnv"],
        renderer_cls: type["PBRRenderer"],
        config: "PBRConfig",
        worker_index: int,
        num_workers: int,
    ) -> "PBREnv":
        """
        Spawn-safe worker factory to build a PBREnv subclass instance with its renderer.
        """
        cfg = PBRConfig.from_config(config, worker_index=worker_index, num_workers=num_workers)
        cfg.worker_index = worker_index
        try:
            if getattr(cfg, "seed", None) is not None:
                cfg.seed = int(cfg.seed) + int(worker_index)
        except Exception:
            # If seed is not present or not an int, ignore
            pass

        renderer = renderer_cls(cfg)
        env = env_cls(renderer=renderer, cfg=cfg)
        return env

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
    #         raise RuntimeError("Renderer is not initialized. Construct env with a renderer and pass it to PBREnv.")
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


