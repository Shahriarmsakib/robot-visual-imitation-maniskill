#!/usr/bin/env bash
# Phase 3 label-noise sweep: two conditions per session, one per GPU (Kaggle T4 x2).
# Each condition = arm B built from the N labelled demos (true actions) + the action-free demos whose TRUE actions
# are corrupted in a known way (scripts/make_noisy_labels.py). Same ACT settings, steps, seed and evaluation
# scenes as the pair and upper-bound runs, so success can be read against the injected label quality.
#
# Usage (from the repo root):
#   OUT=/kaggle/working TAG1=xy025 ARGS1="--xy-r2 0.25" TAG2=zg_idm ARGS2="--z-r2 0.81 --grip-flip 0.06" \
#     bash scripts/run_phase3_noise.sh
# Learned labellers instead of injected noise: ARGS1="labeller:pos_cnn" ARGS2="labeller:fusion" (scripts/labeller_smoke.py)
# An IDM trained on the N labelled demos with this SEED, as for arm B (scripts/idm_sweep.py options after "idm:"),
#   e.g. Morin et al.'s recipe: ARGS1="idm:--arch impala --batch 128 --lr 2e-4 --schedule warmup_linear --loss smoothl1 --res 64 --offsets 0 1 --steps 60000"
# All 1,000 demos with true actions (the upper bound, same data file and command as scripts/run_phase3_upper.sh): ARGS1="true"
# SEED sets the policy's training seed and the labellers' seed; with N = 100 the labelled split is demos 0-99 for every seed.
# Smoke test: DEMO_COUNT=130 MAX_FREE=30 ITERS=100 EVAL_FREQ=50 EVAL_EPS=10 (LABELLER_STEPS=50 for labellers)
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
LABELLER_STEPS="${LABELLER_STEPS:-3000}"
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

build() {  # $1 tag, $2 corruption args or "labeller:<name>" -> prints the number of episodes
  local labels="data/action_free/pseudo_labels/noise_${1}.npz" set="data/action_free/sets/noise_${1}_N${N}_seed${SEED}.h5"
  if [ "$2" = "true" ]; then
    # Upper bound: train on the converted file itself, exactly as scripts/run_phase3_upper.sh does.
    mkdir -p "$(dirname "$set")"; ln -sf "$PWD/$CONV/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5" "$set"
    ln -sf "$PWD/$CONV/trajectory.rgb.pd_ee_delta_pos.physx_cpu.json" "${set%.h5}.json"
    python -c "import json; print(len(json.load(open('${set%.h5}.json'))['episodes']))"
    return
  fi
  if [[ "$2" == idm:* ]]; then
    # shellcheck disable=SC2086
    python scripts/idm_sweep.py --ns "$N" --seeds "$SEED" --out "$OUT/idm_${1}" --pseudo-dir data/action_free/pseudo_labels \
      --name "noise_${1}" ${MAX_FREE:+--max-action-free "$MAX_FREE"} ${2#idm:} >&2
    labels="data/action_free/pseudo_labels/noise_${1}_N${N}_seed${SEED}.npz"
    mkdir -p "$OUT/labels"; cp "$labels" "$OUT/labels/noise_${1}.npz"; cp "$OUT/idm_${1}/metrics.csv" "$OUT/labels/noise_${1}_idm_metrics.csv"
    python scripts/build_pseudo_dataset.py --split "$SPLIT" --pseudo "$labels" --out "$set" >&2
    python -c "import json; print(len(json.load(open('${set%.h5}.json'))['episodes']))"
    return
  fi
  if [[ "$2" == labeller:* ]]; then
    # Learned labellers (scripts/labeller_smoke.py): trained once on labelled demos 0-79, calibrated on 80-99;
    # writes idm, idm_rescale, pos_oracle, pos_cnn and fusion labels for every action-free demo.
    local dir=data/action_free/pseudo_labels/labeller_seed${SEED}
    [ -f "$dir/${2#labeller:}.npz" ] || python scripts/labeller_smoke.py --out "$dir" --steps "$LABELLER_STEPS" --seed "$SEED" \
      ${MAX_FREE:+--max-free "$MAX_FREE"} >&2
    labels="$dir/${2#labeller:}.npz"
    mkdir -p "$OUT/labels"; cp "$dir/calibration.json" "$OUT/labels/labeller_calibration.json"
    cp "$labels" "$OUT/labels/noise_${1}.npz"  # keep the exact labels the policy trains on (a few MB)
  else
    # shellcheck disable=SC2086
    python scripts/make_noisy_labels.py --out "$labels" --seed "$NOISE_SEED" ${MAX_FREE:+--max-action-free "$MAX_FREE"} $2 >&2
  fi
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

# SAPIEN downloads the PhysX GPU library on first use and unzips it over any copy already there. Two runs starting
# together can both download it and one can overwrite the file while the other loads it (a candidate for session 3's
# bus error), so load it once here first.
if command -v nvidia-smi >/dev/null; then
  stamp "PhysX GPU library: load once before the parallel runs"
  python -c "import sapien; sapien.physx.enable_gpu(); print('PhysX GPU library ready')"
fi
export PYTHONUNBUFFERED=1  # a crashed run's log keeps its last lines

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
