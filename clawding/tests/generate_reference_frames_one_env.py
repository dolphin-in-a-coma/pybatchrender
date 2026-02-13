#!/usr/bin/env python3
"""Generate reference frames for a single env+resolution.

Used by generate_reference_frames.py to isolate Panda3D ShowBase per process.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import pybatchrender as pbr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=int, required=True)
    ap.add_argument("--num-scenes", type=int, default=16)
    ap.add_argument("--save-num", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    r = int(a.resolution)
    tile_res = (r, r)

    kwargs = dict(
        num_scenes=a.num_scenes,
        tile_resolution=tile_res,
        render=True,
        offscreen=True,
        seed=a.seed,
    )
    if a.device is not None:
        kwargs["device"] = a.device

    # Make action sampling deterministic
    import random
    import numpy as np
    import torch

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    env = pbr.envs.make(a.env, **kwargs)

    td = env.reset()
    for step in range(a.steps):
        td["action"] = env.action_spec.rand()
        td = env.step(td)

        pixels = td.get(("next", "pixels"), None)
        if pixels is None:
            raise SystemExit(f"No pixels for env={a.env}; render must be enabled")

        if step % a.save_every == 0:
            canvas = env._make_grid_frame(
                pixels,
                indices=list(range(min(a.save_num, a.num_scenes))),
                num=a.save_num,
                scale=1,
            )
            out_dir = a.out / a.env / f"res{r}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{a.env}_res{r}x{r}_step{step}.png"
            Image.fromarray(canvas).save(out_path)
            print("wrote", out_path)

        td = td["next"]

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
