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
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

ENVS=(Asteroids-v0 Breakout-v0 CartPole-v0 Freeway-v0 PacMan-v0 PingPong-v0 SpaceInvaders-v0 Steering-v0)
RES=(64 128 256 512)

for ENV in "${ENVS[@]}"; do
  for R in "${RES[@]}"; do
    python clawding/tests/compare_frames.py \
      --env $ENV \
      --num-scenes 16 \
      --steps 2 \
      --save-every 1 \
      --save-num 4 \
      --tile-resolution $R $R \
      --device cuda \
      --seed 0 \
      --out $TMP_DIR/$ENV/res$R \
      --ref $REF_DIR/$ENV/res$R \
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
