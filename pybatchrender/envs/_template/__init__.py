"""
Template environment for pybatchrender.

Copy this directory to create a new environment. Then:

1. Rename the directory to your environment name (e.g., `pendulum/`)
2. Update all class names (MyEnv -> PendulumEnv, etc.)
3. Implement your environment logic
4. Register it in `pybatchrender/envs/__init__.py`:

    from .pendulum import PendulumEnv, PendulumRenderer, PendulumConfig
    register("Pendulum-v0", PendulumEnv, PendulumRenderer, PendulumConfig)
"""
from .config import MyEnvConfig
from .renderer import MyEnvRenderer
from .env import MyEnv

__all__ = ["MyEnvConfig", "MyEnvRenderer", "MyEnv"]
