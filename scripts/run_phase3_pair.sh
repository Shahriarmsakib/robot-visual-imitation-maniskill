#!/usr/bin/env bash
# Phase 3 policy comparison at one labelled budget N, on a Linux machine with 2 NVIDIA GPUs (Kaggle T4 x2):
#   arm A: ACT on the N labelled demos only (true actions)
#   arm B: ACT on the N labelled demos + the action-free demos pseudo-labelled by an IDM trained on those N
# Both arms: same ACT settings, step budget, seed and evaluation scenes; trained in parallel (GPU 0 and GPU 1).
#
# Usage (from the repo root): OUT=/kaggle/working bash scripts/run_phase3_pair.sh
# Sizes can be reduced for a smoke test, e.g. DEMO_COUNT=130 MAX_FREE=30 IDM_STEPS=200 ITERS=100 EVAL_FREQ=50 EVAL_EPS=10
set -euo pipefail

N="${N:-25}"
SEED="${SEED:-0}"
DEMO_COUNT="${DEMO_COUNT:-1000}"      # demos to convert: 0-99 labelled pool, 100+ action-free
MAX_FREE="${MAX_FREE:-}"             # optional cap on the action-free pool
IDM_STEPS="${IDM_STEPS:-3000}"
ITERS="${ITERS:-15000}"
EVAL_FREQ="${EVAL_FREQ:-5000}"
EVAL_EPS="${EVAL_EPS:-100}"
MAX_HOURS="${MAX_HOURS:-5.0}"
OUT="${OUT:-results/phase3_run}"
SKIP_SETUP="${SKIP_SETUP:-0}"

mkdir -p "$OUT"
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [ "$SKIP_SETUP" != "1" ]; then
  stamp "setup"
  bash scripts/setup_linux_gpu.sh
fi

stamp "demos: download and convert $DEMO_COUNT to rgb + pd_ee_delta_pos"
CONV=data/action_free/conversion
if [ ! -f "$CONV/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5" ]; then
  [ -f data/demos/PickCube-v1/motionplanning/trajectory.h5 ] || python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos
  mkdir -p "$CONV"
  cp data/demos/PickCube-v1/motionplanning/trajectory.h5 data/demos/PickCube-v1/motionplanning/trajectory.json "$CONV/"
  python -m mani_skill.trajectory.replay_trajectory --traj-path "$CONV/trajectory.h5" --use-first-env-state \
    -c pd_ee_delta_pos -o rgb --save-traj -b physx_cpu --count "$DEMO_COUNT" 2>&1 | grep -E "Replayed|Error" || true
fi

stamp "IDM: train on N=$N labelled demos (seed $SEED), pseudo-label the action-free pool"
python scripts/idm_sweep.py --ns "$N" --seeds "$SEED" --offsets 0 1 2 --steps "$IDM_STEPS" --out "$OUT/idm" \
  --pseudo-dir data/action_free/pseudo_labels ${MAX_FREE:+--max-action-free "$MAX_FREE"}

stamp "datasets"
SPLIT="$OUT/idm/splits/N${N}_seed${SEED}.json"
python scripts/build_pseudo_dataset.py --split "$SPLIT" --out data/action_free/sets/A_N${N}_seed${SEED}.h5
python scripts/build_pseudo_dataset.py --split "$SPLIT" --pseudo "data/action_free/pseudo_labels/idm_N${N}_seed${SEED}.npz" \
  --out data/action_free/sets/B_N${N}_seed${SEED}.h5
NB=$(python -c "import json; print(len(json.load(open('data/action_free/sets/B_N${N}_seed${SEED}.json'))['episodes']))")

train() {  # $1 arm, $2 gpu, $3 dataset, $4 number of episodes
  CUDA_VISIBLE_DEVICES="$2" python scripts/train.py --config configs/baseline_act_rgb_100d.json \
    --demo-path "$3" --num-demos "$4" --seed "$SEED" --exp-name "phase3_${1}_N${N}_seed${SEED}" \
    --total-iters "$ITERS" --eval-freq "$EVAL_FREQ" --save-freq "$EVAL_FREQ" --num-eval-episodes "$EVAL_EPS" \
    --max-hours "$MAX_HOURS" --runs-dir "$OUT/runs" > "$OUT/train_${1}.log" 2>&1
}
stamp "training: arm A ($N episodes) on GPU 0, arm B ($NB episodes) on GPU 1, $ITERS steps each"
train A 0 "data/action_free/sets/A_N${N}_seed${SEED}.h5" "$N" & PA=$!
train B 1 "data/action_free/sets/B_N${N}_seed${SEED}.h5" "$NB" & PB=$!
set +e
wait $PA; RA=$?
wait $PB; RB=$?
set -e
stamp "training finished: arm A exit $RA, arm B exit $RB (0 = done, 3 = stopped by time budget)"

python scripts/collect_results.py "$OUT/runs" --out "$OUT/metrics" | tail -8 || true
python scripts/plot_results.py "$OUT/runs" --out "$OUT/figures" | tail -3 || true
for arm in A B; do echo "--- arm $arm"; grep -E "^\[eval\]|^Run |Error" "$OUT/train_${arm}.log" | tail -8 || true; done
stamp "done"
