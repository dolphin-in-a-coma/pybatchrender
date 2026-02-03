#!/usr/bin/env python3
"""
CartPole Benchmark Example

Demonstrates running the CartPole environment with benchmarking and image saving.
Uses the pybatchrender.envs registry API for plug-and-play environment creation.

Usage:
    # Single process (default)
    python cartpole_benchmark.py
    
    # Parallel workers
    python cartpole_benchmark.py --parallel --num-workers 4
    
    # With image saving
    python cartpole_benchmark.py --save-every 50 --save-dir ./outputs
    
    # High-throughput benchmark
    python cartpole_benchmark.py --num-scenes 4096 --steps 1000
"""
from __future__ import annotations

import argparse
import time

import torch

import pybatchrender as pbr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CartPole environment benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-scenes", type=int, default=1024,
        help="Number of parallel scenes per renderer.",
    )
    parser.add_argument(
        "--tile-resolution", type=int, nargs=2, metavar=("W", "H"), default=(64, 64),
        help="Resolution per scene tile.",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Number of environment steps.",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Use parallel workers (spawns separate processes).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel workers (requires --parallel).",
    )
    parser.add_argument(
        "--save-every", type=int, default=-1,
        help="Save image grid every N steps (-1 to disable).",
    )
    parser.add_argument(
        "--save-num", type=int, default=16,
        help="Number of examples to include in saved grids.",
    )
    parser.add_argument(
        "--save-dir", type=str, default='./output',
        help="Output directory for saved images.",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable pixel rendering (state-only mode).",
    )
    return parser.parse_args()


def run_single(args: argparse.Namespace) -> None:
    """Run environment in single-process mode."""
    print("=" * 60)
    print("CartPole Benchmark - Single Process")
    print("=" * 60)
    
    # Create environment using registry
    env = pbr.envs.make(
        "CartPole-v0",
        num_scenes=args.num_scenes,
        tile_resolution=tuple(args.tile_resolution),
        render=not args.no_render,
        save_every_steps=args.save_every,
        save_examples_num=args.save_num,
        save_out_dir=args.save_dir,
    )
    
    print(f"Environment: CartPole-v0")
    print(f"Num scenes: {args.num_scenes}")
    print(f"Tile resolution: {args.tile_resolution}")
    print(f"Rendering: {'enabled' if not args.no_render else 'disabled'}")
    print()
    
    # Reset and run
    td = env.reset()
    print(f"Observation shape: {td['observation'].shape}")
    if "pixels" in td.keys():
        print(f"Pixels shape: {td['pixels'].shape}")
    print()
    
    total_reward = 0.0
    start_time = time.time()
    
    for step in range(args.steps):
        td["action"] = env.action_spec.rand()
        td = env.step(td)
        total_reward += td["next", "reward"].sum().item()
        
        # Periodic saving
        if args.save_every > 0 and step % args.save_every == 0:
            try:
                pixels = td.get(("next", "pixels"), None)
                if pixels is not None:
                    path = env.save_batch_examples(
                        pixels=pixels,
                        indices=list(range(min(args.save_num, args.num_scenes))),
                        num=args.save_num,
                        out_dir=args.save_dir,
                        filename_prefix=f"cartpole_step{step}",
                    )
                    print(f"  Saved: {path}")
            except Exception as e:
                print(f"  Save failed: {e}")
        
        td = td["next"]
    
    elapsed = time.time() - start_time
    total_frames = args.steps * args.num_scenes
    fps = total_frames / max(elapsed, 1e-6)
    
    print()
    print(f"Results:")
    print(f"  Steps: {args.steps}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total reward: {total_reward:,.0f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {fps:,.0f}")


def run_parallel(args: argparse.Namespace) -> None:
    """Run environment with parallel workers."""
    print("=" * 60)
    print("CartPole Benchmark - Parallel Workers")
    print("=" * 60)
    
    num_workers = args.num_workers
    
    # Create parallel environment using registry
    env = pbr.envs.make_parallel(
        "CartPole-v0",
        num_workers=num_workers,
        num_scenes=args.num_scenes,
        tile_resolution=tuple(args.tile_resolution),
        render=not args.no_render,
    )
    
    print(f"Environment: CartPole-v0")
    print(f"Num workers: {num_workers}")
    print(f"Num scenes per worker: {args.num_scenes}")
    print(f"Total parallel scenes: {num_workers * args.num_scenes}")
    print(f"Tile resolution: {args.tile_resolution}")
    print(f"Rendering: {'enabled' if not args.no_render else 'disabled'}")
    print()
    
    # Reset and run
    td = env.reset()
    print(f"Batch size: {env.batch_size}")
    print(f"Observation shape: {td['observation'].shape}")
    if "pixels" in td.keys():
        print(f"Pixels shape: {td['pixels'].shape}")
    print()
    
    total_reward = 0.0
    start_time = time.time()
    
    for step in range(args.steps):
        td = td.clone()
        td["action"] = env.action_spec.rand()
        td = env.step(td)
        total_reward += td["next", "reward"].sum().item()
        td = td["next"]
    
    elapsed = time.time() - start_time
    total_frames = args.steps * num_workers * args.num_scenes
    fps = total_frames / max(elapsed, 1e-6)
    
    print()
    print(f"Results:")
    print(f"  Steps: {args.steps}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total reward: {total_reward:,.0f}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  FPS: {fps:,.0f}")
    
    env.close()


def main() -> None:
    # Show available environments
    print(f"Available environments: {pbr.envs.list_envs()}")
    print()
    
    args = parse_args()
    
    if args.parallel:
        run_parallel(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
