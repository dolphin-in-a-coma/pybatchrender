#!/usr/bin/env python3
"""Compare generated frames against pre-generated reference frames.

This is meant as a *regression test* to catch rendering changes.

Workflow:
1) Generate a small deterministic rollout for an env and save a few PNGs.
2) Compare against a reference directory with matching filenames.

By default we compare exact pixel equality; you can loosen with --max-mean-abs.

Example:
  python clawding/tests/compare_frames.py \
    --env CartPole-v0 \
    --num-scenes 16 --steps 2 --save-every 1 --save-num 4 \
    --out ./tmp_frames \
    --ref ./ref_frames/cartpole
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import pybatchrender as pbr


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--env", type=str, default="CartPole-v0")
    ap.add_argument("--num-scenes", type=int, default=16)
    ap.add_argument("--tile-resolution", type=int, nargs=2, default=(64, 64))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])

    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--save-num", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True, help="Output dir for generated frames")
    ap.add_argument("--ref", type=Path, required=True, help="Reference dir with expected PNGs")

    ap.add_argument("--max-mean-abs", type=float, default=0.0, help="Allowed mean abs pixel error (0 = exact)")
    return ap.parse_args()


def load_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.int16)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        num_scenes=args.num_scenes,
        tile_resolution=tuple(args.tile_resolution),
        render=True,
        offscreen=True,
        save_every_steps=args.save_every,
        save_examples_num=args.save_num,
        save_out_dir=str(args.out),
        seed=args.seed,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    env = pbr.envs.make(args.env, **kwargs)

    td = env.reset()
    for step in range(args.steps):
        td["action"] = env.action_spec.rand()
        td = env.step(td)
        td = td["next"]

    # Compare all pngs created
    gen = sorted(args.out.glob("*.png"))
    if not gen:
        raise SystemExit(f"No PNGs generated in {args.out}")

    failures = 0
    for g in gen:
        r = args.ref / g.name
        if not r.exists():
            print(f"MISSING REF: {r}")
            failures += 1
            continue

        a = load_png(g)
        b = load_png(r)
        if a.shape != b.shape:
            print(f"SHAPE MISMATCH: {g.name} {a.shape} vs {b.shape}")
            failures += 1
            continue

        diff = np.abs(a - b)
        mean_abs = float(diff.mean())
        max_abs = int(diff.max())
        ok = mean_abs <= args.max_mean_abs
        status = "OK" if ok else "DIFF"
        print(f"{status}: {g.name} mean_abs={mean_abs:.4f} max_abs={max_abs}")
        if not ok:
            failures += 1

    try:
        env.close()
    except Exception:
        pass

    if failures:
        raise SystemExit(f"FAIL: {failures} mismatches")

    print("PASS")


if __name__ == "__main__":
    main()
