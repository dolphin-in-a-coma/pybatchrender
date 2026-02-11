"""Pac-Man-like environment for pybatchrender."""
from .config import PacManConfig
from .renderer import PacManRenderer
from .env import PacManEnv

__all__ = ["PacManConfig", "PacManRenderer", "PacManEnv"]
