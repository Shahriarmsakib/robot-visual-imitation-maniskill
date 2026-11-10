# Phase 3: label-noise sweep (plan)

## Question

How good must the action labels of the action-free demos be for them to help, and which errors matter? So far the
policy trained on IDM pseudo-labels (arm B) fails completely at N = 25 and N = 100 (0/100, cube reached in 9–13
episodes), while the same 1,000 demos with true labels succeed (upper bound, 95/100 on `physx_cpu`). The pipeline
check (`scripts/make_oracle_labels.py`) showed the arm-B data is exactly the upper-bound data when labels are true,
so the gap is entirely label quality. This sweep measures policy success against *known* label errors.

## Design

- Data: the N = 100 labelled demos (true actions) + the 900 action-free demos with their **true actions corrupted**
  by `scripts/make_noisy_labels.py`. Everything else is fixed: ACT RGB, official settings, 12k steps, seed 0,
  the same 100 evaluation scenes. Runner: `scripts/run_phase3_noise.sh`, two conditions per Kaggle session (T4 x2).
- Corruptions: i.i.d. Gaussian noise per dimension, sized by target R² against the true actions; a constant bias;
  random gripper flips. Reference points from the IDM at N = 100 (Kaggle): x R² 0.23, y 0.29, z 0.81, gripper 94%.
  Mean bias of the Mac IDM labels at N = 100: x −0.021, z −0.014 (true ≈ −0.001).
- Measures: training-time success on `physx_cuda`, then the CPU stage breakdown (`scripts/failure_analysis.py`,
  100 episodes, scene seed 10000) for paired comparison with arms A, B and the upper bound.

## Sessions (about 4 GPU-hours each; quota 30 h/week)

| Session | GPU 0 | GPU 1 | What it answers |
|---|---|---|---|
| 1 | `xy025`: x, y at R² 0.25 (IDM level), z and gripper true | `zg_idm`: z R² 0.81 + 6% gripper flips (IDM level), x, y true | Which dimensions break arm B |
| 2 | `xy050`: x, y at R² 0.5 | `xy075`: x, y at R² 0.75 | Threshold on horizontal quality |
| 3 | `xy090`: x, y at R² 0.9 | `bias`: IDM-sized mean bias only | Threshold continued; does bias alone hurt |
| 4 | `all_idm`: all dims at IDM level, i.i.d. | (repeat of the most informative condition, seed 1) | Does the IDM's error *structure* matter beyond its size |

The order after session 1 depends on its result: if `xy025` collapses like arm B and `zg_idm` does not, the sweep
continues along horizontal quality (sessions 2–3); otherwise it moves to the other dimensions. Seeds 1 and 2 for
the key conditions follow once the threshold region is known.

## Caveats fixed in advance

- i.i.d. noise is not the IDM's error: IDM errors are correlated over time and with the scene. Session 4 compares
  the two at matched size.
- One seed per condition at first; claims about thresholds wait for seeds 1–2.
- Results are read on the CPU simulator, which matches the demo recordings (see `results/action_free/upper_N1000/`).

## Session log

- Session 1 (2026-11-10, code fee7bf4): `zg_idm` complete (CPU 90/100). `xy025` was killed out of memory, because two
  1,000-demo runs in parallel held ~23 GB of pre-resized frames. The loader now resizes on access (same pixels,
  `scripts/check_lazy_resize.py`), and `xy025` is rerun first in the next session. Results:
  `results/action_free/noise_sweep/README.md`.
