#!/usr/bin/env bash
set -euo pipefail

# Submit GPU correctness + perf tests on Puhti.
# Run this on a Puhti login node.

ACCOUNT=${ACCOUNT:-project_2013820}
USER_NAME=${USER_NAME:-$USER}
BASE=${BASE:-/scratch/${ACCOUNT}/${USER_NAME}}

SRC=${SRC:-$BASE/src/pybatchrender}
VENV=${VENV:-$BASE/venvs/pybatchrender-pt27}

STAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=${RESULTS_DIR:-$BASE/test_results/$STAMP}

mkdir -p "$RESULTS_DIR"

# --- Ensure venv + renderer deps exist on the login node ---
# We use CSC module torch (pytorch/2.7) and avoid pip-installing torch.
# But we DO need Panda3D (+ optional CUDA interop deps) inside the venv.

ensure_venv() {
  # CSC env init script is not `set -u` clean (references e.g. SLURM_JOB_ID).
  # Temporarily disable nounset while sourcing.
  set +u
  source /appl/profile/zz-csc-env.sh
  set -u
  module purge
  module load pytorch/2.7

  if [[ ! -d "$SRC/.git" ]]; then
    echo "ERROR: expected repo at $SRC (clone first)" >&2
    return 2
  fi

  if [[ ! -d "$VENV" ]]; then
    echo "Creating venv: $VENV"
    python -m venv --system-site-packages "$VENV"
  fi

  # Activate and test imports.
  # Note: `python` should be from module env; venv contains site-packages for extras.
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"

  python - <<'PY' || NEED_RENDER_DEPS=1
import panda3d
print('panda3d ok:', panda3d.__version__)
PY

  python - <<'PY' || NEED_RL_DEPS=1
import tensordict, torchrl
print('tensordict ok:', tensordict.__version__)
print('torchrl ok:', torchrl.__version__)
PY

  pip install -U pip

  if [[ "${NEED_RENDER_DEPS:-0}" == "1" ]]; then
    echo "Installing renderer deps into venv (panda3d + optional cuda extras)..."
    pip install 'panda3d>=1.10.13'

    # Optional CUDA interop deps (can be heavy; keep enabled because tests request CUDA path)
    pip install 'cuda-python>=12,<12.9' 'cupy-cuda12x>=12' 'PyOpenGL>=3.1'
  fi

  if [[ "${NEED_RL_DEPS:-0}" == "1" ]]; then
    echo "Installing RL deps into venv (tensordict + torchrl; torch comes from module)..."
    pip install 'tensordict>=0.5' 'torchrl>=0.5'
  fi

  echo "Installing pybatchrender editable (no deps)..."
  cd "$SRC"
  pip install --no-deps -e .
}

ensure_venv


cat > "$RESULTS_DIR/sbatch_bench_all_envs.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=pbr-bench-all
#SBATCH --account=$ACCOUNT
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=$RESULTS_DIR/bench-all-%j.out

set -eo pipefail

source /appl/profile/zz-csc-env.sh
module purge
module load pytorch/2.7

source $VENV/bin/activate
cd $SRC

python clawding/tests/benchmark_all_envs.py \
  --num-scenes 1024 \
  --steps 100 \
  --device cuda \
  --json-out $RESULTS_DIR/bench_all_envs.json
EOF

cat > "$RESULTS_DIR/sbatch_compare_frames.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=pbr-cmp
#SBATCH --account=$ACCOUNT
#SBATCH --partition=gpu
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=$RESULTS_DIR/compare-%j.out

set -eo pipefail

source /appl/profile/zz-csc-env.sh
module purge
module load pytorch/2.7

source $VENV/bin/activate
cd $SRC

# Use committed reference PNGs as baseline.
# If you want a fresh baseline, run generate_reference_frames.py separately.
REF_DIR=clawding/tests/ref_frames

TMP_DIR=$RESULTS_DIR/tmp_compare
rm -rf "\$TMP_DIR"
mkdir -p "\$TMP_DIR"

ENVS=(Asteroids-v0 Breakout-v0 CartPole-v0 Freeway-v0 PacMan-v0 PingPong-v0 SpaceInvaders-v0 Steering-v0)
RES=(64 128 256 512)

for ENV in "\${ENVS[@]}"; do
  for R in "\${RES[@]}"; do
    python clawding/tests/compare_frames.py \
      --env \$ENV \
      --num-scenes 16 \
      --steps 2 \
      --save-every 1 \
      --save-num 4 \
      --tile-resolution \$R \$R \
      --device cuda \
      --seed 0 \
      --out \$TMP_DIR/\$ENV/res\$R \
      --ref \$REF_DIR/\$ENV/res\$R \
      --max-mean-abs 0
  done
done

echo ALL_PASS
EOF

BENCH_JOB=$(sbatch "$RESULTS_DIR/sbatch_bench_all_envs.sh" | awk '{print $4}')
CMP_JOB=$(sbatch "$RESULTS_DIR/sbatch_compare_frames.sh" | awk '{print $4}')

echo "Submitted:"
echo "- bench_all_envs: $BENCH_JOB"
echo "- compare_frames: $CMP_JOB"
echo "Results dir: $RESULTS_DIR"
