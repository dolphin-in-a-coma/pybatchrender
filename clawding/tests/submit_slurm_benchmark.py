#!/usr/bin/env python3
"""Submit a SLURM benchmark job (Puhti default) and print job id.

This is a convenience script for running performance checks on clusters.
It writes a temporary sbatch script that:
- initializes CSC module env
- uses pytorch/2.7
- activates a scratch-first venv (system-site-packages)
- runs CartPole benchmark for a specified number of scenes

Assumptions (Puhti defaults):
- repo is already cloned on login node in $BASE/src/pybatchrender
- venv exists at $BASE/venvs/pybatchrender-pt27

Example:
  python clawding/tests/submit_slurm_benchmark.py --cluster puhti --num-scenes 4096 --steps 50
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cluster", choices=["puhti", "mahti"], default="puhti")
    ap.add_argument("--account", default="project_2013820")
    ap.add_argument("--base", default=None, help="Scratch base dir (defaults to cluster-specific)")

    ap.add_argument("--num-scenes", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=50)

    # Puhti GPU defaults
    ap.add_argument("--partition", default=None)
    ap.add_argument("--gres", default=None)
    ap.add_argument("--time", default="00:30:00")
    ap.add_argument("--cpus-per-task", type=int, default=16)
    ap.add_argument("--mem", default="64G")
    return ap.parse_args()


def run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True)
    return out.strip()


def main() -> None:
    a = parse_args()

    user = run(["bash", "-lc", "echo $USER"])  # ok on CSC

    if a.cluster == "puhti":
        base = a.base or f"/scratch/{a.account}/{user}"
        partition = a.partition or "gpu"
        gres = a.gres or "gpu:v100:1"
    else:
        base = a.base or f"/scratch/{a.account}/{user}"
        partition = a.partition or "gpusmall"
        gres = a.gres or "gpu:a100:1"

    basep = Path(base)
    sbatch_path = basep / "sbatch_pbr_bench.sh"
    out_pattern = str(basep / "pbr-bench-%j.out")

    script = f"""#!/bin/bash
#SBATCH --job-name=pbr-bench
#SBATCH --account={a.account}
#SBATCH --partition={partition}
#SBATCH --time={a.time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={a.cpus_per_task}
#SBATCH --mem={a.mem}
#SBATCH --gres={gres}
#SBATCH --output={out_pattern}

set -eo pipefail

source /appl/profile/zz-csc-env.sh
module purge
module load pytorch/2.7

BASE={base}
source $BASE/venvs/pybatchrender-pt27/bin/activate
cd $BASE/src/pybatchrender

python examples/scripts/cartpole_benchmark.py \
  --num-scenes {a.num_scenes} \
  --steps {a.steps} \
  --save-every 0 \
  --save-num 0 \
  --save-dir $BASE/outputs
"""

    # Ensure base exists
    basep.mkdir(parents=True, exist_ok=True)
    sbatch_path.write_text(script)

    # Submit
    submit_out = run(["sbatch", str(sbatch_path)])
    print(submit_out)


if __name__ == "__main__":
    main()
