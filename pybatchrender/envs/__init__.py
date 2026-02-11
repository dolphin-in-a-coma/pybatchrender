"""
Environment registry for pybatchrender.

Provides a Gym-style factory pattern for creating environments.

IMPORTANT - Single Renderer Per Process:
    Panda3D's ShowBase is a singleton. Only ONE renderer/environment can exist
    per Python process. Attempting to create a second environment will raise:
        Exception: Attempt to spawn multiple ShowBase instances!
    
    For multiple environments, use make_parallel() which spawns separate
    worker processes (requires mp_start_method="spawn").

Usage:
    import pybatchrender as pbr
    
    # List available environments
    print(pbr.envs.list_envs())
    
    # Create a single environment (one per process!)
    env = pbr.envs.make("CartPole-v0", num_scenes=1024)
    
    # For multiple environments, use parallel workers:
    env = pbr.envs.make_parallel("CartPole-v0", num_workers=4, num_scenes=1024)
    
    # Get raw classes for inspection or custom parallel setup:
    EnvCls, RendererCls, ConfigCls = pbr.envs.get_env_classes("CartPole-v0")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Type, Dict, Tuple, Any, Callable

if TYPE_CHECKING:
    from ..config import PBRConfig
    from ..renderer.renderer import PBRRenderer
    from ..env import PBREnv

# Global registry: name -> (EnvClass, RendererClass, ConfigClass)
_ENV_REGISTRY: Dict[str, Tuple[Type["PBREnv"], Type["PBRRenderer"], Type["PBRConfig"]]] = {}


def register(
    name: str,
    env_cls: Type["PBREnv"],
    renderer_cls: Type["PBRRenderer"],
    config_cls: Type["PBRConfig"],
) -> None:
    """
    Register an environment by name.
    
    Args:
        name: Unique identifier for the environment (e.g., "CartPole-v0")
        env_cls: The PBREnv subclass
        renderer_cls: The PBRRenderer subclass
        config_cls: The PBRConfig subclass
    """
    if name in _ENV_REGISTRY:
        import warnings
        warnings.warn(f"Environment '{name}' is already registered. Overwriting.")
    _ENV_REGISTRY[name] = (env_cls, renderer_cls, config_cls)


def unregister(name: str) -> None:
    """Remove an environment from the registry."""
    if name in _ENV_REGISTRY:
        del _ENV_REGISTRY[name]


def list_envs() -> list[str]:
    """List all registered environment names."""
    return sorted(_ENV_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if an environment is registered."""
    return name in _ENV_REGISTRY


def get_env_classes(name: str) -> Tuple[Type["PBREnv"], Type["PBRRenderer"], Type["PBRConfig"]]:
    """
    Get the (EnvClass, RendererClass, ConfigClass) tuple for an environment.
    
    Useful for advanced use cases like creating parallel environments.
    
    Args:
        name: The registered environment name
        
    Returns:
        Tuple of (EnvClass, RendererClass, ConfigClass)
        
    Example:
        EnvCls, RendererCls, ConfigCls = get_env_classes("CartPole-v0")
        config = ConfigCls(num_scenes=2048)
        env = EnvCls.make_parallel_env(config=config, renderer_cls=RendererCls, num_workers=4)
    """
    if name not in _ENV_REGISTRY:
        raise ValueError(f"Unknown environment: '{name}'. Available: {list_envs()}")
    return _ENV_REGISTRY[name]


def make(name: str, **config_overrides) -> "PBREnv":
    """
    Create an environment by name (Gym-style factory).
    
    Args:
        name: The registered environment name
        **config_overrides: Override any config parameters
        
    Returns:
        An initialized PBREnv instance
        
    Example:
        env = make("CartPole-v0", num_scenes=1024, tile_resolution=(64, 64))
        td = env.reset()
        td["action"] = env.action_spec.rand()
        td = env.step(td)
    """
    if name not in _ENV_REGISTRY:
        raise ValueError(f"Unknown environment: '{name}'. Available: {list_envs()}")
    
    env_cls, renderer_cls, config_cls = _ENV_REGISTRY[name]
    cfg = config_cls(**config_overrides)
    renderer = renderer_cls(cfg)
    return env_cls(renderer=renderer, cfg=cfg)


def make_parallel(
    name: str,
    num_workers: int,
    shared_memory: bool = True,
    **config_overrides,
) -> "ParallelEnv":
    """
    Create a parallel environment by name.
    
    Each worker spawns its own renderer in a separate process, avoiding
    the ShowBase singleton limitation. Uses "spawn" multiprocessing method
    (required for Panda3D/OpenGL compatibility).
    
    Args:
        name: The registered environment name
        num_workers: Number of parallel workers
        shared_memory: Whether to use shared memory for tensors
        **config_overrides: Override any config parameters
        
    Returns:
        A TorchRL ParallelEnv instance
        
    Example:
        env = make_parallel("CartPole-v0", num_workers=4, num_scenes=1024)
    """
    if name not in _ENV_REGISTRY:
        raise ValueError(f"Unknown environment: '{name}'. Available: {list_envs()}")
    
    env_cls, renderer_cls, config_cls = _ENV_REGISTRY[name]
    cfg = config_cls(**config_overrides)
    
    return env_cls.make_parallel_env(
        config=cfg,
        renderer_cls=renderer_cls,
        num_workers=num_workers,
        mp_start_method="spawn",
        shared_memory=shared_memory,
    )


def _register_builtins() -> None:
    """Auto-register built-in environments."""
    # CartPole
    try:
        from .cartpole import CartPoleEnv, CartPoleRenderer, CartPoleConfig
        register("CartPole-v0", CartPoleEnv, CartPoleRenderer, CartPoleConfig)
    except ImportError:
        pass

    # Pac-Man-like 2D environment
    try:
        from .pacman import PacManEnv, PacManRenderer, PacManConfig
        register("PacMan-v0", PacManEnv, PacManRenderer, PacManConfig)
    except ImportError:
        pass


def _register_from_entry_points() -> None:
    """
    Discover and register environments from installed packages.
    
    Third-party packages can register environments by adding an entry point
    in their pyproject.toml:
    
        [project.entry-points."pybatchrender.envs"]
        my_env = "my_package.envs:register_envs"
    
    The entry point should be a callable that calls pybatchrender.envs.register().
    """
    try:
        from importlib.metadata import entry_points
        
        # Python 3.10+ returns a SelectableGroups object
        try:
            eps = entry_points(group="pybatchrender.envs")
        except TypeError:
            # Python 3.9 fallback
            eps = entry_points().get("pybatchrender.envs", [])
        
        for ep in eps:
            try:
                register_fn = ep.load()
                if callable(register_fn):
                    register_fn()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load environment entry point '{ep.name}': {e}")
    except ImportError:
        pass


# Auto-register on import
_register_builtins()
_register_from_entry_points()


__all__ = [
    "register",
    "unregister",
    "list_envs",
    "is_registered",
    "get_env_classes",
    "make",
    "make_parallel",
]
