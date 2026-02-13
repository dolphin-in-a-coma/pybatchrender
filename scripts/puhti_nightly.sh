#!/usr/bin/env bash
set -euo pipefail

# Nightly Puhti GPU test runner.
# Run this on a Puhti login node (e.g. via crontab).
#
# Responsibilities:
# - update repo (optional)
# - submit Puhti GPU tests (benchmark + compare_frames)
# - optionally wait for completion and summarize
#
# Outputs are written under $BASE/test_results/<timestamp>/

ACCOUNT=${ACCOUNT:-project_2013820}
USER_NAME=${USER_NAME:-$USER}
BASE=${BASE:-/scratch/${ACCOUNT}/${USER_NAME}}

REPO_DIR=${REPO_DIR:-$BASE/src/pybatchrender}
BRANCH=${BRANCH:-clawding}

WAIT=${WAIT:-1}               # 1 = wait for jobs to finish and write summary
POLL_SECS=${POLL_SECS:-30}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR=$BASE/test_results/$STAMP
mkdir -p "$RUN_DIR"

LOG=$RUN_DIR/runner.log
SUMMARY=$RUN_DIR/summary.txt

{
  echo "[$(date -Is)] start puhti_nightly"
  echo "ACCOUNT=$ACCOUNT"
  echo "BASE=$BASE"
  echo "REPO_DIR=$REPO_DIR"
  echo "BRANCH=$BRANCH"

  source /appl/profile/zz-csc-env.sh
  module purge

  if [[ -d "$REPO_DIR/.git" ]]; then
    cd "$REPO_DIR"
    git fetch --all
    git checkout "$BRANCH"
    git pull --ff-only
  else
    echo "ERROR: repo not found at $REPO_DIR. Clone first (see clawding/ENV_SETUP.md)." >&2
    exit 2
  fi

  echo "Submitting GPU tests..."
  # submit_puhti_gpu_tests.sh creates its own timestamp dir. We override RESULTS_DIR to use RUN_DIR.
  RESULTS_DIR="$RUN_DIR" ACCOUNT="$ACCOUNT" USER_NAME="$USER_NAME" BASE="$BASE" bash ./scripts/submit_puhti_gpu_tests.sh | tee "$RUN_DIR/submit.log"

  BENCH_JOB=$(grep -E '^\- bench_all_envs:' "$RUN_DIR/submit.log" | awk '{print $3}')
  CMP_JOB=$(grep -E '^\- compare_frames:' "$RUN_DIR/submit.log" | awk '{print $3}')
  echo "BENCH_JOB=$BENCH_JOB"
  echo "CMP_JOB=$CMP_JOB"

  if [[ "$WAIT" != "1" ]]; then
    echo "WAIT=0; not waiting for jobs."
    exit 0
  fi

  echo "Waiting for jobs to complete..."
  for J in "$BENCH_JOB" "$CMP_JOB"; do
    while squeue -j "$J" -h 2>/dev/null | grep -q .; do
      echo "[$(date -Is)] job $J still running/pending"
      sleep "$POLL_SECS"
    done
    echo "[$(date -Is)] job $J left queue"
  done

  echo "Summarizing..."
  {
    echo "Puhti nightly run: $STAMP"
    echo "bench_job: $BENCH_JOB"
    echo "compare_job: $CMP_JOB"
    echo "run_dir: $RUN_DIR"
    echo

    CMP_OUT=$(ls -1 "$RUN_DIR"/compare-*.out 2>/dev/null | tail -n 1 || true)
    BENCH_JSON="$RUN_DIR/bench_all_envs.json"

    if [[ -n "$CMP_OUT" ]]; then
      if grep -q "ALL_PASS" "$CMP_OUT"; then
        echo "compare_frames: PASS"
      else
        echo "compare_frames: FAIL (see $CMP_OUT)"
        tail -n 80 "$CMP_OUT" || true
      fi
    else
      echo "compare_frames: missing output file"
    fi

    if [[ -f "$BENCH_JSON" ]]; then
      echo
      echo "bench_all_envs.json: present"
      python - <<'PY' "$BENCH_JSON" || true
import json, sys
p=sys.argv[1]
obj=json.load(open(p))
# print top fps env
res=obj.get('results',[])
res=sorted(res, key=lambda x: x.get('fps',0), reverse=True)
if res:
    print('top_fps:', res[0]['env'], int(res[0]['fps']))
PY
    else
      echo "bench_all_envs.json: missing"
    fi
  } | tee "$SUMMARY"

  echo "[$(date -Is)] done"
} 2>&1 | tee "$LOG"
