import unittest

from pybatchrender.envs.atari_style import light_kwargs, topdown_camera_pose
from pybatchrender.envs.asteroids import AsteroidsConfig
from pybatchrender.envs.breakout import BreakoutConfig
from pybatchrender.envs.freeway import FreewayConfig
from pybatchrender.envs.pingpong import PingPongConfig
from pybatchrender.envs.spaceinvaders import SpaceInvadersConfig


class AtariRenderStyleTests(unittest.TestCase):
    def test_2d_style_enabled_by_default(self):
        cfgs = [
            PingPongConfig(render=False),
            BreakoutConfig(render=False),
            SpaceInvadersConfig(render=False),
            AsteroidsConfig(render=False),
            FreewayConfig(render=False),
        ]
        for cfg in cfgs:
            self.assertTrue(cfg.use_2d_objects)

    def test_topdown_camera_pose(self):
        pos, look, fov = topdown_camera_pose(arena_extent=1.8)
        self.assertGreater(float(pos[2]), 3.0 - 1e-6)
        self.assertAlmostEqual(float(look[2]), 0.0, places=6)
        self.assertLess(abs(float(pos[1])), float(pos[2]))
        self.assertEqual(fov, 38.0)

    def test_flat_light_preset(self):
        lk = light_kwargs(True)
        self.assertGreater(lk["ambient"][0], 0.8)
        self.assertLessEqual(lk["strength"], 0.5)


if __name__ == "__main__":
    unittest.main()
