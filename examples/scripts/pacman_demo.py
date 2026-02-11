#!/usr/bin/env python3
"""Pac-Man-like environment demo for pybatchrender."""
from __future__ import annotations

import argparse

import pybatchrender as pbr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PacMan-v0 demo")
    parser.add_argument("--num-scenes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-num", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="./outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = pbr.envs.make(
        "PacMan-v0",
        num_scenes=args.num_scenes,
        tile_resolution=(96, 96),
        offscreen=True,
        render=True,
    )

    td = env.reset()
    print("observation:", td["observation"].shape)
    print("pixels:", td["pixels"].shape if "pixels" in td.keys() else None)

    for step in range(args.steps):
        td["action"] = env.action_spec.rand()
        td = env.step(td)
        if args.save_every >= 0 and (step % max(1, args.save_every) == 0):
            px = td.get(("next", "pixels"), None)
            if px is not None:
                path = env.save_batch_examples(
                    pixels=px,
                    num=args.save_num,
                    out_dir=args.save_dir,
                    filename_prefix=f"pacman_step{step}",
                )
                print("saved:", path)
        td = td["next"]


if __name__ == "__main__":
    main()
