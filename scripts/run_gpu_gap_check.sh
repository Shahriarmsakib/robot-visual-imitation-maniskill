#!/usr/bin/env bash
# Why does the 1,000-demo upper-bound policy score 43/100 on the GPU simulator but 95/100 on the CPU simulator?
# Three checks on one NVIDIA GPU, no training (~30-40 min including setup):
#   1. Stage breakdown of CKPT on physx_cuda, the same 10-env layout and scene seed as the training-time evaluation
#      (should reproduce 43/100), with one video per failure category.
#   2. The same on physx_cpu on Linux (Mac gave 95/100 for these settings): separates simulator from platform.
#   3. Open-loop replay of the recorded demos' pd_ee_delta_pos actions on both simulators: do CPU-recorded
#      demos still succeed under GPU physics?
#
# Usage (from the repo root): CKPT=/path/to/final.pt OUT=/kaggle/working bash scripts/run_gpu_gap_check.sh
set -euo pipefail

CKPT="${CKPT:?set CKPT to the upper-bound final.pt}"
OUT="${OUT:-results/gpu_gap_check}"
EPISODES="${EPISODES:-100}"
DEMO_COUNT="${DEMO_COUNT:-100}"
SKIP_SETUP="${SKIP_SETUP:-0}"

mkdir -p "$OUT"
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ "$SKIP_SETUP" != "1" ]; then
  stamp "setup"
  bash scripts/setup_linux_gpu.sh
fi
sha256sum "$CKPT" | tee "$OUT/checkpoint_sha256.txt"

for BACKEND in physx_cuda physx_cpu; do
  stamp "failure analysis on $BACKEND ($EPISODES episodes, scene seed = the run's eval_seed)"
  python scripts/failure_analysis.py "$CKPT" --label "upper_N1000_${BACKEND}" --sim-backend "$BACKEND" \
    --num-episodes "$EPISODES" --num-envs 10 --device cuda --videos \
    --out "$OUT/failure_analysis_${BACKEND}" 2>&1 | grep -vE "^\s*$" | tail -12 | tee "$OUT/failure_analysis_${BACKEND}.txt" || true
  python scripts/summarize_failure_analysis.py --dir "$OUT/failure_analysis_${BACKEND}" \
    --primary "upper_N1000_${BACKEND}" 2>&1 | tail -3 || true
done

stamp "demo replay: first $DEMO_COUNT demos, recorded actions, on both simulators"
CONV=data/action_free/conversion
H5="$CONV/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
if [ ! -f "$H5" ]; then
  [ -f data/demos/PickCube-v1/motionplanning/trajectory.h5 ] || python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos
  mkdir -p "$CONV"
  cp data/demos/PickCube-v1/motionplanning/trajectory.h5 data/demos/PickCube-v1/motionplanning/trajectory.json "$CONV/"
  python -m mani_skill.trajectory.replay_trajectory --traj-path "$CONV/trajectory.h5" --use-first-env-state \
    -c pd_ee_delta_pos -o rgb --save-traj -b physx_cpu --count "$DEMO_COUNT" 2>&1 | grep -E "Replayed|Error" || true
fi
for BACKEND in physx_cpu physx_cuda; do
  R="$OUT/replay_tmp_$BACKEND"; mkdir -p "$R"
  cp "$H5" "$R/demos.h5"; cp "${H5%.h5}.json" "$R/demos.json"
  NE=1; [ "$BACKEND" = "physx_cuda" ] && NE=10
  # Success is counted at each demo's last step, as in the original conversion (no target control mode: plain action replay).
  python -m mani_skill.trajectory.replay_trajectory --traj-path "$R/demos.h5" --use-first-env-state \
    -o none --save-traj -b "$BACKEND" -n "$NE" --count "$DEMO_COUNT" 2>&1 \
    | grep -E "Replayed|Error|rror:" | tee "$OUT/demo_replay_${BACKEND}.txt" || true
  rm -rf "$R"
done

stamp "summary"
for f in "$OUT"/failure_analysis_*.txt "$OUT"/demo_replay_*.txt; do echo "== $f"; cat "$f"; done
stamp "done"
