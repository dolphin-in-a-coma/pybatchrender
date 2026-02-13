# Test scripts (clawding)

These scripts are meant for quick performance + correctness checks.

## 1) Benchmark all envs

```bash
python clawding/tests/benchmark_all_envs.py --num-scenes 1024 --steps 200 --device cuda
```

Implementation note: each env is benchmarked in a fresh subprocess to avoid Panda3D `ShowBase` re-entrancy issues.

Add `--json-out results.json` for machine-readable results.

## 2) Frame regression test

Generate frames + compare to reference PNGs.

```bash
python clawding/tests/compare_frames.py \
  --env CartPole-v0 \
  --num-scenes 16 --steps 2 --save-every 1 --save-num 4 \
  --out ./tmp_frames \
  --ref ./ref_frames/cartpole \
  --max-mean-abs 0
```

## 3) Submit SLURM benchmark (Puhti default)

This generates an sbatch script under `/scratch/<project>/<user>/` and submits it.

```bash
python clawding/tests/submit_slurm_benchmark.py --cluster puhti --num-scenes 4096 --steps 50
```

Assumes you already prepared a scratch-first layout per `clawding/ENV_SETUP.md`.

## Suggested additional tests

- Determinism test (same seed -> identical first frame checksum)
- Memory usage tracking (peak GPU memory per env)
- Render correctness invariants (no NaNs in pixels, pixel range, etc.)
