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

| Session | GPU 0 | GPU 1 | What it answers | Status |
|---|---|---|---|---|
| 1 | `xy025`: x, y at R² 0.25 (IDM level), z and gripper true | `zg_idm`: z R² 0.81 + 6% gripper flips (IDM level), x, y true | Which dimensions break arm B | `zg_idm` 90/100 CPU; `xy025` killed (out of memory) |
| 2 | `xy025` (rerun) | `xy050`: x, y at R² 0.5 | Does horizontal noise of the IDM's size break it | both 93/100 CPU: no |
| 3 | `xy_shrink04`: x, y scaled by 0.4, no noise | `xy025_ar09`: x, y at R² 0.25, noise AR(1) with ρ = 0.9 | Which of the IDM's two error properties breaks it | `xy_shrink04` 1/100 CPU: shrinkage alone breaks it; `xy025_ar09` crashed before training |
| 4 | `xy025_ar09` (rerun) | `xy_shrink07`: x, y scaled by 0.7, no noise (step R² 0.91) | Does correlated noise alone also break it; how much shrinkage is too much | `xy025_ar09` 52/100 CPU, `xy_shrink07` 25/100: both hurt alone |
| 5 | `pos_cnn`: gripper position estimated from each image by a CNN, actions from position differences | `fusion`: Kalman/RTS smoother of CNN positions and IDM motion | Do labels with correct scale and cancelling errors (chunk R² 0.98–0.99 offline, `results/action_free/labeller_smoke/`) give a working policy; which matters more, scale (`pos_cnn`) or step accuracy (`fusion`) | `pos_cnn` 50/100, `fusion` 57/100 CPU: rescues arm B (0), below upper (95); label scores did not predict the gap |
| 6 | `fusion_truegrip`: `fusion` x, y, z + the true gripper command (diagnostic) | `fusion_gripw`: `fusion` x, y, z + gripper from the finger opening estimated in each image | How much of the 57 → 95 gap do the IDM's gripper labels cause (wrong in 22.8% of steps holding at the goal); does the image-only finger-width fix (1.1%) recover it | `fusion_truegrip` 60/100, `fusion_gripw` 53/100 CPU: drops gone, success unchanged; gap is goal precision |
| 7 | `posrec_truegrip`: recorded gripper positions -> actions (same map), true gripper | `xyz_cnnlevel`: true actions + independent noise on x, y, z at the CNN labels' step R² (0.42 / 0.66 / 0.79), true gripper | Goal precision: is position-to-action lossless (expect ≈ 95)? Does random noise on all three axes at the learned labels' level break precision, or is it the learned errors' structure? | done: CPU 93 / 88 |
| 8 | `fusion_errshuf`: `fusion_truegrip` with its x, y, z residual errors moved to *another* episode (neighbour in length); the shrink (slope × true action) is kept; true gripper. Mac check (`results/action_free/labeller_smoke/structure_errshuf.json`): step R², slope, autocorrelation and chunk R² match the source within 0.01 | `fusion_truegrip_s8`: `fusion_truegrip` rerun from the same labeller build (matched control) | Do the image-based errors hurt because they depend on the scene? Same error size and pattern over time, link to the scene broken. Near 88 -> scene dependence is the harmful property; near 60 -> the pattern over time is | done: CPU 72 vs 74 (p = 0.83): not scene dependence; control 74 vs session 6's 60 (p = 0.007): large run-to-run spread |
| 9 | `xyz_shrink_ar045`: true actions × slope 0.947 / 0.983 / 0.963 + AR(1) noise (ρ = 0.45) to the session-8 `fusion` per-step R²; true gripper. Offline: step R² 0.88 / 0.93 / 0.95, slope 0.95 / 0.98 / 0.96, error autocorrelation 0.46 / 0.45 / 0.47 (`fusion_truegrip_s8`: 0.46 / 0.40 / 0.54), chunk R² 0.98 / 0.99 / 0.99 | `xyz_cnnlevel_r2`: `xyz_cnnlevel` rerun with identical labels (training-only repeat) | Do the image labels' mild shrink and persistence explain the 74 vs 88 gap (near 74: yes; near 88: no)? How much does GPU training alone move success (vs 60 / 74 for labeller rebuilds)? | done: CPU 76 (vs `fusion_truegrip_s8` 74, p = 0.83; vs `xyz_cnnlevel` 88, 2 vs 14, p = 0.004): mild shrink + persistence reproduce the gap. Repeat bit-identical to session 7 (88): training deterministic, spread came from the labeller rebuild |
| 10 | `upper`, seed 1: all 1,000 demos with true actions (same data file and command as the upper bound) | `fusion_truegrip`, seed 1: labeller rebuilt with seed 1 | Seed repeats (see below) | done: CPU upper 97 (seed 0: 95, p = 0.73); `fusion_truegrip` 95 (vs session 8 build 74, 23 vs 2, p = 2e-5; vs upper seed 1, 1 vs 3, p = 0.63) |
| 11 | `xyz_cnnlevel`, seed 1 (same labels, noise seed 0) | `xyz_shrink_ar045`, seed 1 (same labels) | Seed repeats; after session 10, also: do the seed-0 gaps (88 vs 76) survive another seed? | done: CPU 97 and 96 with the same labels (seed 0: 88 and 76; 11 vs 2, p = 0.02; 22 vs 2, p = 4e-5): the seed-0 gaps were a seed effect |
| 12 | `xy025_ar09`, seed 1 (same labels as session 4) | `xy_shrink07`, seed 1 (same labels as session 4) | Do the strong-error effects (52 and 25 at seed 0, against 93 for random noise) survive another seed? | done: CPU 48 and 23 (seed 0: 52 and 25; p = 0.61 and 0.63), against 97 for random noise and true labels at seed 1: both reproduce |
| 13 | arm A (100 labelled demos only), seed 1 | arm B (+ 900 IDM pseudo-labelled), seed 1; `scripts/run_phase3_pair.sh` with N = 100, as in `results/action_free/policy_N100/` | Does the IDM collapse (B 0, A 3 at seed 0) survive another seed? `xy025` seed 1 follows in a later session | done: CPU B 0, A 13 (13 vs 0, p = 2e-4), upper seed 1 97: the collapse reproduces (`results/action_free/policy_N100/`) |
| 14 | `xy025`, seed 1 (same labels as session 2) | `xy_shrink04`, seed 1 (same labels as session 3) | Does random noise of the IDM's size stay harmless (93), and the ×0.4 shrink fatal (1), at another seed? Completes seed 1 for every core condition | done: CPU 95 and 1 (seed 0: 93 and 1; p = 0.75 and 1): every core condition now has two seeds |
| 15 | `upper`, seed 2 | `xy025`, seed 2 | Third seed of the core conditions (see below); main account | prepared |
| 16 | `xy025_ar09`, seed 2 | `xy_shrink07`, seed 2 | Third seed; second account, in parallel with 15 | prepared |
| 17 | `xy_shrink04`, seed 2 | `fusion_truegrip`, seed 2 (labeller rebuilt with seed 2) | Third seed; main account, after 15 | prepared |
| 18 | arm A, seed 2 | arm B (IDM rebuilt with seed 2); `scripts/run_phase3_pair.sh`, N = 100 | Third seed; main account, after 17 | prepared |

**Third seed (sessions 15–18).** After session 14 every core condition had two seeds. Sessions 15–18 add seed 2 for
all of them, plus the image-based labeller (`fusion_truegrip`), with the same labels as before for the synthetic
conditions. Settings are those of the earlier sessions with `SEED=2` (kernel settings in each session's Kaggle
script; runners unchanged since session 14). Paired tests: `scripts/paired_tests.py`.

**Re-plan after session 11.** Sessions 10–11 showed that mild label errors make results seed-dependent, and that
the seed-0 differences of sessions 7–9 were a seed effect. Sessions 12–13 therefore repeat the large, strong-error
effects that the main claims rest on, instead of a second seed of conditions that sit at the ceiling at seed 1.

**Seed repeats (sessions 10–13, second Kaggle account).** Session 9 showed that training is deterministic at a
fixed seed, so a repeat only measures variation if the seed changes. Sessions 10–13 rerun the four conditions behind
the main claims with training seeds 1 and 2: the upper bound (95), CNN-sized random noise (88), mild shrink +
persistence (76) and the image-based labeller (60 / 74). With N = 100 the labelled split is demos 0–99 for every
seed, so only the training randomness changes. Synthetic labels keep noise seed 0, so their repeats measure policy
training alone; the labeller is rebuilt with the new seed (`scripts/labeller_smoke.py --seed`), so its repeats
measure the whole pipeline. Each session takes about 4–4.5 GPU-hours, about 17 hours for the four. Read on the CPU
simulator as before; the result for each condition is reported as the seed mean and range, with paired tests per seed.

Sessions 1–2 changed the plan. Independent noise of the IDM's size is harmless (`results/action_free/noise_sweep/`),
but the IDM's errors are structured: horizontal labels are shrunk (slope 0.27–0.47) and the errors are strongly
autocorrelated (0.78–0.95), as measured by `scripts/label_error_structure.py`. Session 3 tests each property alone,
with no other error. Checked offline on the 900 demos: `--xy-scale 0.4` gives slope 0.40 and 30-step
displacement R² 0.64; `--xy-r2 0.25 --ar1 0.9` gives slope 1.0, autocorrelation 0.90 and 30-step displacement R²
0.19–0.34 (the IDM: 0.21–0.68). The original sessions 3–4 (`xy075`, `xy090`, `bias`) are dropped; with R² 0.25
already harmless they would not be informative.

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
- Session 2 (2026-11-10, code f585784): resize check PASS on Kaggle (torch 2.11); `xy025` and `xy050` both 93/100
  on CPU, GPU final 29 and 33/100.
- Session 3 (2026-11-10, code d0797df): resize check PASS. `xy_shrink04` complete: GPU final 2/100, CPU 1/100
  (reached cube 24/100), close to arm B with IDM labels (0/100). `xy025_ar09` died with a bus error before its
  first evaluation. The likely but unconfirmed cause is both runs unpacking SAPIEN's PhysX GPU library at once. The
  runner now loads that library once before the parallel start and runs Python unbuffered.
- Session 4 (2026-11-13, code 73a79d3): PhysX library pre-loaded, both runs completed. `xy025_ar09` (correlated
  noise, R² 0.25): CPU 52/100 vs 93 for independent noise of the same size (paired 42 vs 1). `xy_shrink07` (×0.7,
  step R² 0.91): CPU 25/100, reaches and grasps the cube but rarely the goal. GPU final 27 and 20/100.
- Session 5 (2026-11-15, code 71d2895): labellers built in the kernel; both runs completed. CPU success at end
  `pos_cnn` 50/100, `fusion` 57/100 (arm B 0, upper 95, `xy025` 93); reached target 66 / 79. GPU final 13 / 17.
  The chunk / slope label scores predicted `pos_cnn` ≥ `xy025`; it was not.
- Session 6 (2026-11-16, code c4a69fc): labels npz kept in the output. True gripper labels remove the drops
  (20 -> 0) but success at end 60 vs 57 (p = 0.74); finger-width gripper 53/100 (more missed grasps). Failures
  shift to stopping short and low (median 2.3 cm sideways, 2.6 cm low). GPU final 21 / 23.
- Session 7 (2026-11-16, code f5c967a): both runs completed. CPU success at end `posrec_truegrip` 93/100 (upper 95,
  paired 2 vs 0) and `xyz_cnnlevel` 88/100 (upper paired 9 vs 2, p = 0.07), both far above `fusion_truegrip` 60
  (35 vs 2 and 33 vs 5). The position-to-action map is lossless and CNN-sized random noise is nearly harmless, so
  the gap is the structure of the image-based position errors. GPU final 38 / 35.
- Session 8 (2026-11-17, code 54755eb): both runs completed. CPU success at end `fusion_errshuf` 72/100,
  `fusion_truegrip_s8` 74/100 (paired 10 vs 12, p = 0.83): moving errors between scenes changes nothing. The control
  is 14 above session 6's build of the same labeller (19 vs 5, p = 0.007; x step R² 0.88 vs 0.80), so learned
  labellers need repeats. Still below `xyz_cnnlevel` 88 (p = 0.007). GPU final 27 / 28.
- Session 9 (2026-11-18, code 5c69a43): both runs completed. CPU success at end `xyz_shrink_ar045` 76/100, level
  with `fusion_truegrip_s8` 74 (12 vs 10, p = 0.83) and below `xyz_cnnlevel` 88 (2 vs 14, p = 0.004); losses at
  reaching and holding the goal, as for the image labels. `xyz_cnnlevel_r2` gave weights identical to session 7's
  run (max difference 0.0) and the same 88/100, so training is deterministic given labels and seed. GPU final 38 / 35.
- Session 10 (2026-11-19, code 96cd208, second Kaggle account): both runs completed. CPU success at end: upper
  bound seed 1 97/100 (seed 0 95, 3 vs 5, p = 0.73); `fusion_truegrip` with labeller and policy seed 1 95/100, level
  with the upper bound (1 vs 3, p = 0.63) and far above the seed-0 builds (60, 74). Label statistics are close to the
  session-8 build's, and the seed changed labeller and policy together, so the cause is not separated. The seed-0
  conclusions of sessions 7–9 wait on sessions 11–13. GPU final 33 / 43.
- Session 11 (2026-11-19, code 3c516f4, second Kaggle account): both runs completed with the same labels as at seed 0.
  CPU success at end `xyz_cnnlevel` 97/100 (seed 0: 88) and `xyz_shrink_ar045` 96/100 (seed 0: 76); at seed 1 all
  four conditions score 95–97. The seed-0 differences of sessions 7–10 came from the training seed; the conclusions
  of sessions 7–9 are withdrawn. The strong-error results of sessions 1–4 still have one seed each. GPU final 38 / 37.
- Session 12 (2026-11-19, code 91d680f, second Kaggle account): both runs completed with the same labels as session 4.
  CPU success at end `xy025_ar09` 48/100 (seed 0: 52; 19 vs 15, p = 0.61) and `xy_shrink07` 23/100 (seed 0: 25; 3 vs 1,
  p = 0.63), against 97 for random noise and true labels at seed 1: both strong-error effects reproduce. GPU final
  20 / 18.
- Session 13 (2026-11-19, code 3a91e28, second Kaggle account): N = 100 pair with IDM and policy seed 1. IDM R² x 0.12,
  y 0.51, z 0.80, gripper 93.6%. CPU success at end arm B 0/100 (reached cube 24), arm A 13/100 (seed 0: 3; 11 vs 1,
  p = 0.006); A vs B 13 vs 0, p = 2e-4. The IDM collapse reproduces. GPU final 8 / 0.
- Session 14 (2026-11-20, code 074fd83, second Kaggle account): both runs completed with the same labels as sessions
  2–3. CPU success at end `xy025` 95/100 (seed 0: 93; 4 vs 6, p = 0.75) and `xy_shrink04` 1/100 (seed 0: 1). Every core
  condition now has two seeds; none of the strong-error conditions moved by more than 4 episodes. GPU final 37 / 4.
