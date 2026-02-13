#!/usr/bin/env python3
"""Generate deterministic reference frame grids for compare_frames.

This script saves a few PNG grids per env/res/step with stable filenames.

Recommended usage: run on a specific target machine (e.g. Puhti V100) and commit
results as baseline for regression tests on the same platform.

Example:
  python clawding/tests/generate_reference_frames.py \
    --out clawding/tests/ref_frames \
    --device cuda \
    --num-scenes 128 \
    --save-num 16 \
    --steps 2 \
    --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import pybatchrender as pbr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--envs", type=str, default=None, help="Comma-separated env list (default: all)")
    ap.add_argument("--resolutions", type=str, default="64,128,256,512", help="Comma-separated square tile resolutions")

    ap.add_argument("--num-scenes", type=int, default=128)
    ap.add_argument("--save-num", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    envs = list(pbr.envs.list_envs())
    if a.envs:
        allow = [e.strip() for e in a.envs.split(",") if e.strip()]
        envs = allow

    resolutions = [int(x) for x in a.resolutions.split(",") if x.strip()]

    for env_name in envs:
        for r in resolutions:
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

            env = pbr.envs.make(env_name, **kwargs)

            td = env.reset()
            for step in range(a.steps):
                td["action"] = env.action_spec.rand()
                td = env.step(td)

                pixels = td.get(("next", "pixels"), None)
                if pixels is None:
                    raise SystemExit(f"No pixels for env={env_name}; render must be enabled")

                if step % a.save_every == 0:
                    canvas = env._make_grid_frame(
                        pixels,
                        indices=list(range(min(a.save_num, a.num_scenes))),
                        num=a.save_num,
                        scale=1,
                    )
                    out_dir = a.out / env_name / f"res{r}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{env_name}_res{r}x{r}_step{step}.png"
                    Image.fromarray(canvas).save(out_path)
                    print("wrote", out_path)

                td = td["next"]

            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
