#!/usr/bin/env bash
# Download the official PickCube-v1 motion-planning demos and convert the first
# NUM_DEMOS to RGB observations with pd_ee_delta_pos actions (official ACT recipe).
# Usage: scripts/prepare_data.sh [NUM_DEMOS]   (default 100)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUM_DEMOS="${1:-100}"
DEMO_DIR="$ROOT/data/demos/PickCube-v1/motionplanning"
OUT="$DEMO_DIR/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
cd "$ROOT"

if [ ! -f "$DEMO_DIR/trajectory.h5" ]; then
  python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos
fi

if [ -f "$OUT" ]; then
  echo "Already converted: $OUT (delete it to regenerate)"
  exit 0
fi

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$DEMO_DIR/trajectory.h5" \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj -b physx_cpu --count "$NUM_DEMOS"

echo "Wrote $OUT"
