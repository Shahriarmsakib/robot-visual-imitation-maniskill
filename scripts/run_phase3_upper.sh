#!/usr/bin/env bash
# Phase 3 upper-bound arm: ACT on all DEMO_COUNT demos with their true actions, on one NVIDIA GPU.
# Same ACT settings, step budget, seed and evaluation scenes as the arms of scripts/run_phase3_pair.sh,
# so it shows how much success the schedule allows when every demo is labelled.
#
# Usage (from the repo root): OUT=/kaggle/working bash scripts/run_phase3_upper.sh
# Smoke test: DEMO_COUNT=10 ITERS=100 EVAL_FREQ=50 EVAL_EPS=10
set -euo pipefail

SEED="${SEED:-0}"
DEMO_COUNT="${DEMO_COUNT:-1000}"
ITERS="${ITERS:-12000}"
EVAL_FREQ="${EVAL_FREQ:-4000}"
EVAL_EPS="${EVAL_EPS:-100}"
MAX_HOURS="${MAX_HOURS:-5.0}"
GPU="${GPU:-0}"
OUT="${OUT:-results/phase3_upper}"
SKIP_SETUP="${SKIP_SETUP:-0}"

mkdir -p "$OUT"
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ "$SKIP_SETUP" != "1" ]; then
  stamp "setup"
  bash scripts/setup_linux_gpu.sh
fi

stamp "demos: download and convert $DEMO_COUNT to rgb + pd_ee_delta_pos"
CONV=data/action_free/conversion
H5="$CONV/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
if [ ! -f "$H5" ]; then
  [ -f data/demos/PickCube-v1/motionplanning/trajectory.h5 ] || python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos
  mkdir -p "$CONV"
  cp data/demos/PickCube-v1/motionplanning/trajectory.h5 data/demos/PickCube-v1/motionplanning/trajectory.json "$CONV/"
  python -m mani_skill.trajectory.replay_trajectory --traj-path "$CONV/trajectory.h5" --use-first-env-state \
    -c pd_ee_delta_pos -o rgb --save-traj -b physx_cpu --count "$DEMO_COUNT" 2>&1 | grep -E "Replayed|Error" || true
fi
NE=$(python -c "import json; print(len(json.load(open('${H5%.h5}.json'))['episodes']))")

stamp "training: upper bound ($NE episodes, true actions) on GPU $GPU, $ITERS steps"
set +e
CUDA_VISIBLE_DEVICES="$GPU" python scripts/train.py --config configs/baseline_act_rgb_100d.json \
  --demo-path "$H5" --num-demos "$NE" --seed "$SEED" --exp-name "phase3_upper_N${NE}_seed${SEED}" \
  --total-iters "$ITERS" --eval-freq "$EVAL_FREQ" --save-freq "$EVAL_FREQ" --num-eval-episodes "$EVAL_EPS" \
  --max-hours "$MAX_HOURS" --runs-dir "$OUT/runs" > "$OUT/train_upper.log" 2>&1
RC=$?
set -e
stamp "training finished: exit $RC (0 = done, 3 = stopped by time budget)"

python scripts/collect_results.py "$OUT/runs" --out "$OUT/metrics" | tail -8 || true
python scripts/plot_results.py "$OUT/runs" --out "$OUT/figures" | tail -3 || true
grep -E "^\[eval\]|^Run |Error" "$OUT/train_upper.log" | tail -8 || true
stamp "done"
