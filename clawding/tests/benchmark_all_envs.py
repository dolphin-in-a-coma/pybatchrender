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

    # Panda3D / ShowBase is not reliably re-entrant in a single process:
    # creating multiple envs sequentially can hit "Attempt to spawn multiple ShowBase instances".
    # We therefore benchmark each env in an isolated subprocess.

    import subprocess
    import sys

    for env_name in envs:
        cmd = [
            sys.executable,
            "-m",
            "clawding.tests.benchmark_one_env",
            "--env",
            env_name,
            "--num-scenes",
            str(args.num_scenes),
            "--steps",
            str(args.steps),
            "--tile-resolution",
            str(args.tile_resolution[0]),
            str(args.tile_resolution[1]),
            "--warmup-steps",
            str(args.warmup_steps),
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        if args.no_render:
            cmd += ["--no-render"]
        if args.onscreen:
            cmd += ["--onscreen"]

        out = subprocess.check_output(cmd, text=True)
        # The helper prints a single JSON line.
        line = out.strip().splitlines()[-1]
        payload = json.loads(line)
        r = Result(**payload)
        results.append(r)

        print(f"{r.env:16s} | wall={r.wall_s:7.3f}s | fps={r.fps:12.0f} | device={r.device} | render={r.render}")

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
