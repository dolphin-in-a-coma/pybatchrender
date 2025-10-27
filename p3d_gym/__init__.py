from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("p3d-gym")
except Exception:
    __version__ = "0.1.0"

from .config import P3DConfig  # noqa: F401

# TorchRL is an optional dependency; guard the import so top-level package import works
try:
    from .env import P3DEnv  # noqa: F401
except Exception:
    P3DEnv = None  # type: ignore

try:
    from .renderer.renderer import P3DRenderer  # noqa: F401
except Exception:
    P3DRenderer = None  # type: ignore

# Expose renderer helpers (optional: depend on Panda3D/OpenGL availability)
try:
    from .renderer.node import P3DNode  # noqa: F401
except Exception:
    P3DNode = None  # type: ignore

try:
    from .renderer.camera import P3DCam  # noqa: F401
except Exception:
    P3DCam = None  # type: ignore

try:
    from .renderer.light import P3DLight  # noqa: F401
except Exception:
    P3DLight = None  # type: ignore

try:
    from .renderer.shader_context import P3DShaderContext  # noqa: F401
except Exception:
    P3DShaderContext = None  # type: ignore

try:
    from .renderer.frame_grabber import GPUFrameGrabber, CPUFrameGrabber, GPU_AVAILABLE  # noqa: F401
    # Convenience alias: prefer GPU if available
    FrameGrabber = GPUFrameGrabber if GPU_AVAILABLE else CPUFrameGrabber  # type: ignore
except Exception:
    GPUFrameGrabber = None  # type: ignore
    CPUFrameGrabber = None  # type: ignore
    GPU_AVAILABLE = False  # type: ignore
    FrameGrabber = None  # type: ignore

__all__ = [
    "P3DConfig",
    "P3DEnv",
    "P3DRenderer",
    # Renderer helpers
    "P3DNode",
    "P3DCam",
    "P3DLight",
    "P3DShaderContext",
    "GPUFrameGrabber",
    "CPUFrameGrabber",
    "FrameGrabber",
    "__version__",
]
