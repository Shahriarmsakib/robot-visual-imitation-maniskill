#!/usr/bin/env bash
# Phase 3 label-noise sweep: two conditions per session, one per GPU (Kaggle T4 x2).
# Each condition = arm B built from the N labelled demos (true actions) + the action-free demos whose TRUE actions
# are corrupted in a known way (scripts/make_noisy_labels.py). Same ACT settings, steps, seed and evaluation
# scenes as the pair and upper-bound runs, so success can be read against the injected label quality.
#
# Usage (from the repo root):
#   OUT=/kaggle/working TAG1=xy025 ARGS1="--xy-r2 0.25" TAG2=zg_idm ARGS2="--z-r2 0.81 --grip-flip 0.06" \
#     bash scripts/run_phase3_noise.sh
# Smoke test: DEMO_COUNT=130 MAX_FREE=30 ITERS=100 EVAL_FREQ=50 EVAL_EPS=10
set -euo pipefail

N="${N:-100}"
SEED="${SEED:-0}"
NOISE_SEED="${NOISE_SEED:-0}"
DEMO_COUNT="${DEMO_COUNT:-1000}"
MAX_FREE="${MAX_FREE:-}"
ITERS="${ITERS:-12000}"
EVAL_FREQ="${EVAL_FREQ:-4000}"
EVAL_EPS="${EVAL_EPS:-100}"
MAX_HOURS="${MAX_HOURS:-6.0}"
OUT="${OUT:-results/phase3_noise}"
SKIP_SETUP="${SKIP_SETUP:-0}"
TAG1="${TAG1:?set TAG1}"; ARGS1="${ARGS1:?set ARGS1}"
TAG2="${TAG2:-}"; ARGS2="${ARGS2:-}"

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

SPLIT="$OUT/splits/N${N}_seed${SEED}.json"
python -c "from src.action_free.data import save_split; print('labelled ids:', len(save_split('$SPLIT', $N, $SEED)))"

build() {  # $1 tag, $2 corruption args -> prints the number of episodes
  local labels="data/action_free/pseudo_labels/noise_${1}.npz" set="data/action_free/sets/noise_${1}_N${N}_seed${SEED}.h5"
  # shellcheck disable=SC2086
  python scripts/make_noisy_labels.py --out "$labels" --seed "$NOISE_SEED" ${MAX_FREE:+--max-action-free "$MAX_FREE"} $2 >&2
  mkdir -p "$OUT/labels"; cp "${labels%.npz}.json" "$OUT/labels/noise_${1}.json"
  python scripts/build_pseudo_dataset.py --split "$SPLIT" --pseudo "$labels" --out "$set" >&2
  python -c "import json; print(len(json.load(open('${set%.h5}.json'))['episodes']))"
}
train() {  # $1 tag, $2 gpu, $3 episodes
  CUDA_VISIBLE_DEVICES="$2" python scripts/train.py --config configs/baseline_act_rgb_100d.json \
    --demo-path "data/action_free/sets/noise_${1}_N${N}_seed${SEED}.h5" --num-demos "$3" --seed "$SEED" \
    --exp-name "phase3_noise_${1}_N${N}_seed${SEED}" --total-iters "$ITERS" --eval-freq "$EVAL_FREQ" \
    --save-freq "$EVAL_FREQ" --num-eval-episodes "$EVAL_EPS" --max-hours "$MAX_HOURS" --runs-dir "$OUT/runs" \
    > "$OUT/train_${1}.log" 2>&1
}

stamp "labels and datasets: $TAG1 [$ARGS1]${TAG2:+, $TAG2 [$ARGS2]}"
NE1=$(build "$TAG1" "$ARGS1")
[ -n "$TAG2" ] && NE2=$(build "$TAG2" "$ARGS2")

# The loader resizes frames on access (keeps two 1,000-demo runs within the machine's RAM); stop here if that
# ever differs from the old up-front resize on this machine's torch build.
stamp "check: resize-on-access matches the old loader"
python scripts/check_lazy_resize.py --demo-path "data/action_free/sets/noise_${TAG1}_N${N}_seed${SEED}.h5" \
  --num-demos 50 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = "0" ] || { stamp "resize check failed, not training"; exit 1; }

stamp "training: $TAG1 ($NE1 episodes) on GPU 0${TAG2:+, $TAG2 ($NE2 episodes) on GPU 1}, $ITERS steps each"
set +e
train "$TAG1" 0 "$NE1" & P1=$!
[ -n "$TAG2" ] && { train "$TAG2" 1 "$NE2" & P2=$!; }
wait $P1; R1=$?
R2=0; [ -n "$TAG2" ] && { wait $P2; R2=$?; }
set -e
stamp "training finished: $TAG1 exit $R1${TAG2:+, $TAG2 exit $R2} (0 = done, 3 = stopped by time budget)"

python scripts/collect_results.py "$OUT/runs" --out "$OUT/metrics" | tail -8 || true
python scripts/plot_results.py "$OUT/runs" --out "$OUT/figures" | tail -3 || true
for t in $TAG1 $TAG2; do echo "--- $t"; grep -E "^\[eval\]|^Run |Error" "$OUT/train_${t}.log" | tail -8 || true; done
stamp "done"
