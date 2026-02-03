"""
CartPole environment for pybatchrender.

A classic control environment where a pole is attached to a cart moving on a rail.
The goal is to balance the pole by applying forces to the cart.
"""
from .config import CartPoleConfig
from .renderer import CartPoleRenderer
from .env import CartPoleEnv

__all__ = ["CartPoleConfig", "CartPoleRenderer", "CartPoleEnv"]
