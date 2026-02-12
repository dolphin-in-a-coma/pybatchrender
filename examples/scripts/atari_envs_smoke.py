"""Quick smoke run for Atari-inspired environments.

Runs each env in a separate subprocess because Panda3D ShowBase is singleton per process.
"""
import argparse
import subprocess
import sys

import pybatchrender as pbr


def run_one(name: str):
    env = pbr.envs.make(name, num_scenes=32, render=False, offscreen=True)
    td = env.reset()
    for _ in range(8):
        td["action"] = env.action_spec.rand()
        td = env.step(td)["next"]
    print(f"{name}: ok, obs={tuple(td['observation'].shape)}, reward_mean={float(td['reward'].mean()):.3f}")
    renderer = getattr(env, "_renderer", None)
    if renderer is not None:
        try:
            renderer.destroy()
        except Exception:
            pass


def run_all():
    for env_name in ["PingPong-v0", "Breakout-v0", "SpaceInvaders-v0", "Asteroids-v0", "Freeway-v0"]:
        cmd = [sys.executable, __file__, "--env", env_name]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    args = parser.parse_args()

    if args.env:
        run_one(args.env)
    else:
        run_all()
