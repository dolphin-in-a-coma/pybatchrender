#!/usr/bin/env python3
"""Benchmark all registered pybatchrender envs on the current machine.

Goals:
- Measure step throughput (FPS = num_scenes * steps / wall_time)
- Produce a concise report (text + optional JSON)

Notes:
- This script is meant to be run inside a configured environment (e.g. Puhti pytorch/2.7 module).
- For cluster usage, keep `--save-*` disabled to avoid I/O overhead.

Example:
  python clawding/tests/benchmark_all_envs.py --num-scenes 1024 --steps 200 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import pybatchrender as pbr


@dataclass
class Result:
    env: str
    num_scenes: int
    steps: int
    device: str
    render: bool
    offscreen: bool
    wall_s: float
    fps: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--num-scenes", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tile-resolution", type=int, nargs=2, default=(64, 64))
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Override device; default uses env default")
    ap.add_argument("--no-render", action="store_true", help="Disable pixel rendering")
    ap.add_argument("--onscreen", action="store_true", help="Use onscreen mode (window). Default: offscreen")

    ap.add_argument("--include", type=str, default=None, help="Comma-separated allowlist of env names")
    ap.add_argument("--exclude", type=str, default=None, help="Comma-separated blocklist of env names")

    ap.add_argument("--warmup-steps", type=int, default=5)

    ap.add_argument("--json-out", type=Path, default=None, help="Write results to JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    envs = list(pbr.envs.list_envs())
    if args.include:
        allow = {e.strip() for e in args.include.split(",") if e.strip()}
        envs = [e for e in envs if e in allow]
    if args.exclude:
        block = {e.strip() for e in args.exclude.split(",") if e.strip()}
        envs = [e for e in envs if e not in block]

    print("=== system ===")
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    print("=== benchmark ===")
    print("envs:", envs)

    results: list[Result] = []

    for env_name in envs:
        # Create env
        kwargs = dict(
            num_scenes=args.num_scenes,
            tile_resolution=tuple(args.tile_resolution),
            render=not args.no_render,
            offscreen=not args.onscreen,
        )
        if args.device is not None:
            kwargs["device"] = args.device

        env = pbr.envs.make(env_name, **kwargs)

        # Warmup
        td = env.reset()
        for _ in range(args.warmup_steps):
            td["action"] = env.action_spec.rand()
            td = env.step(td)["next"]

        # Timed
        t0 = time.perf_counter()
        td = env.reset()
        for _ in range(args.steps):
            td["action"] = env.action_spec.rand()
            td = env.step(td)["next"]
        # Sync if on CUDA to include device work
        if torch.cuda.is_available() and (args.device == "cuda" or (args.device is None and getattr(env, "device", None) == "cuda")):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        t1 = time.perf_counter()

        wall = t1 - t0
        fps = (args.steps * args.num_scenes) / max(wall, 1e-9)

        device = args.device or getattr(env, "device", "unknown")
        r = Result(
            env=env_name,
            num_scenes=args.num_scenes,
            steps=args.steps,
            device=str(device),
            render=not args.no_render,
            offscreen=not args.onscreen,
            wall_s=wall,
            fps=fps,
        )
        results.append(r)

        print(f"{env_name:16s} | wall={wall:7.3f}s | fps={fps:12.0f} | device={device} | render={not args.no_render}")

        try:
            env.close()
        except Exception:
            pass

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "results": [asdict(x) for x in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print("wrote:", args.json_out)


if __name__ == "__main__":
    main()
