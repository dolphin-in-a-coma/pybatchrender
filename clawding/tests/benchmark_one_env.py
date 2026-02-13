#!/usr/bin/env python3
"""Benchmark a single env and output JSON.

This module exists to run in a fresh process per env, to avoid Panda3D ShowBase
re-entrancy issues.

Usage (internal):
  python -m clawding.tests.benchmark_one_env --env CartPole-v0 --num-scenes 1024 --steps 100

Outputs:
- human-readable header
- last line is a JSON dict suitable for Result(**payload)
"""

from __future__ import annotations

import argparse
import json
import time

import torch

import pybatchrender as pbr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--num-scenes", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tile-resolution", type=int, nargs=2, default=(64, 64))
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--onscreen", action="store_true")
    return ap.parse_args()


def main() -> None:
    a = parse_args()

    kwargs = dict(
        num_scenes=a.num_scenes,
        tile_resolution=tuple(a.tile_resolution),
        render=not a.no_render,
        offscreen=not a.onscreen,
    )
    if a.device is not None:
        kwargs["device"] = a.device

    env = pbr.envs.make(a.env, **kwargs)

    td = env.reset()
    for _ in range(a.warmup_steps):
        td["action"] = env.action_spec.rand()
        td = env.step(td)["next"]

    t0 = time.perf_counter()
    td = env.reset()
    for _ in range(a.steps):
        td["action"] = env.action_spec.rand()
        td = env.step(td)["next"]

    if torch.cuda.is_available() and (a.device == "cuda" or (a.device is None and getattr(env, "device", None) == "cuda")):
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    t1 = time.perf_counter()

    wall = t1 - t0
    fps = (a.steps * a.num_scenes) / max(wall, 1e-9)

    device = a.device or getattr(env, "device", "unknown")

    payload = dict(
        env=a.env,
        num_scenes=a.num_scenes,
        steps=a.steps,
        device=str(device),
        render=not a.no_render,
        offscreen=not a.onscreen,
        wall_s=wall,
        fps=fps,
    )

    # Last line: machine-readable
    print(json.dumps(payload))

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
