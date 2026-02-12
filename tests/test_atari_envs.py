import unittest

import torch

from pybatchrender.envs.pingpong import PingPongConfig, PingPongEnv
from pybatchrender.envs.breakout import BreakoutConfig, BreakoutEnv
from pybatchrender.envs.spaceinvaders import SpaceInvadersConfig, SpaceInvadersEnv
from pybatchrender.envs.asteroids import AsteroidsConfig, AsteroidsEnv
from pybatchrender.envs.freeway import FreewayConfig, FreewayEnv


class DummyRenderer:
    def __init__(self, cfg):
        self.cfg = cfg

    def step(self, obs=None):
        w, h = self.cfg.tile_resolution
        return torch.zeros((self.cfg.num_scenes, self.cfg.num_channels, h, w), dtype=torch.uint8)


class AtariEnvSmokeTests(unittest.TestCase):
    def _check_env(self, cfg_cls, env_cls):
        cfg = cfg_cls(num_scenes=4, offscreen=True, render=False)
        env = env_cls(renderer=DummyRenderer(cfg), cfg=cfg)
        td = env.reset()
        self.assertIn("observation", td.keys())
        self.assertEqual(td["observation"].shape[0], cfg.num_scenes)

        td["action"] = env.action_spec.rand()
        td1 = env.step(td)["next"]
        self.assertIn("reward", td1.keys())
        self.assertEqual(td1["reward"].shape, torch.Size([cfg.num_scenes, 1]))
        self.assertEqual(td1["done"].shape, torch.Size([cfg.num_scenes, 1]))

    def test_pingpong(self):
        self._check_env(PingPongConfig, PingPongEnv)

    def test_breakout(self):
        self._check_env(BreakoutConfig, BreakoutEnv)

    def test_spaceinvaders(self):
        self._check_env(SpaceInvadersConfig, SpaceInvadersEnv)

    def test_asteroids(self):
        self._check_env(AsteroidsConfig, AsteroidsEnv)

    def test_freeway(self):
        self._check_env(FreewayConfig, FreewayEnv)


if __name__ == "__main__":
    unittest.main()
