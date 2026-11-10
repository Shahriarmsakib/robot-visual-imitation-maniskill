# Phase 3: label-noise sweep, session 1

**Short answer: vertical and gripper labels as noisy as the IDM's are almost harmless.** Arm B with the IDM's
level of error on z (R² 0.81) and the gripper (6% of steps flipped), but true x and y labels, succeeds in **90/100**
episodes on the CPU simulator, against 95/100 for the upper bound with all labels true (paired difference 6 vs 1
episodes, p = 0.125) and 0/100 for arm B with the real IDM pseudo-labels. So the z and gripper errors do not explain
arm B's collapse; by elimination the horizontal (x, y) labels are the prime suspect. The direct test of that,
condition `xy025`, did not run: Kaggle killed it for running out of memory (see below). One seed, i.i.d. noise.

Plan and design: [`experiments/phase3_noise_sweep.md`](../../../experiments/phase3_noise_sweep.md).

## Setup (Kaggle T4 x2, 2026-11-10, code `fee7bf4`)

Both conditions use the arm-B pipeline at N = 100: the 100 labelled demos with true actions plus the 900
action-free demos whose **true** actions are corrupted in a known way by `scripts/make_noisy_labels.py`
(noise seed 0). Same ACT RGB settings, 12,000 steps, seed 0, and evaluation scenes as the N = 100 pair and the
upper bound. Runner: `scripts/run_phase3_noise.sh`; kernel: `kaggle/phase3_noise_kernel.py`.

| Condition | Corruption | Achieved label quality on the 900 demos (`labels/`) | Ran? |
|---|---|---|---|
| `zg_idm` (GPU 1) | z: Gaussian noise to R² 0.81; gripper flipped on 6% of steps; x, y true | x R² 1.0, y 1.0, z 0.810, gripper 94.0% correct | yes, 12,000 steps |
| `xy025` (GPU 0) | x, y: Gaussian noise to R² 0.25; z, gripper true | x R² 0.249, y 0.249, z 1.0, gripper 100% | **no**: killed during training (exit 137) |
| Reference: IDM at N = 100 (`../policy_N100/`) | learned labels | x 0.23, y 0.29, z 0.81, gripper 94.3% | |

## Results

**Training-time success** (`physx_cuda`, out of 100; success at end / success once):

| Step | 0 | 4,000 | 8,000 | 12,000 (final) |
|---|---|---|---|---|
| `zg_idm` | 0 / 0 | 19 / 62 | 30 / 84 | **31 / 82** (95% CI 23–41% at end) |
| Upper bound, all true labels (`../upper_N1000/`) | 0 / 0 | 19 / 74 | 33 / 84 | 43 / 87 (CI 34–53%) |

On the GPU simulator `zg_idm` ends 12 episodes below the upper bound. These are separate runs on GPU-simulator
scenes that are not paired with the CPU analysis, and the CIs overlap. The GPU simulator also understates every
policy trained on these CPU-recorded demos (`../upper_N1000/gpu_gap_check/`).

**Stage breakdown on the CPU simulator** (`physx_cpu`, scene seed 10000, the same 100 scenes as arms A, B and the
upper bound; `failure_analysis/`, comparison in `compare_N100/`):

| | Reached cube | Grasped | Stable lift | Reached target | Success once | Success at end |
|---|---|---|---|---|---|---|
| `zg_idm` (noisy z + gripper) | 100 | 100 | 95 | 93 | 93 | **90** |
| Upper bound (all true) | 100 | 100 | 96 | 97 | 97 | 95 |
| Arm A, 100 labelled only | 97 | 96 | 90 | 11 | 4 | 3 |
| Arm B, + IDM pseudo-labels | 9 | 12 | 7 | 0 | 0 | 0 |

![stage breakdown](compare_N100/figures/success_breakdown.png)

**Paired exact McNemar tests on the same 100 scenes** (episodes where only the first / only the second succeeds;
`paired_tests.json`):

| Comparison | Reached target | Success at end |
|---|---|---|
| `zg_idm` vs arm A | 82 vs 0, p = 4e-25 | 87 vs 0, p = 1e-26 |
| `zg_idm` vs arm B | 93 vs 0, p = 2e-28 | 90 vs 0, p = 2e-27 |
| Upper bound vs `zg_idm` | 5 vs 1, p = 0.22 | 6 vs 1, p = 0.125 |

Failure modes of `zg_idm` (CPU): 90 success at end, 4 grasped and held but never reached the target, 3 reached
the target but did not hold success, 3 lost the grasp before the target.

## What this shows

1. **z and gripper errors at the IDM's level cost little.** The 900 extra demos with these labels lift the policy
   from 3/100 (arm A) to 90/100, close to the 95/100 with perfect labels. The 6-point gap to the upper bound is
   not significant with one seed and 100 scenes.
2. **So arm B's collapse must come from somewhere else.** Arm B's z and gripper labels are about as good as the
   ones here, yet it reaches the cube in only 9/100 episodes. The remaining differences are the x and y labels
   (R² 0.23 / 0.29) and the *structure* of the IDM's errors (correlated over time and with the scene, plus a
   mean bias), which this i.i.d. noise does not reproduce.
3. **What it does not show yet.** Whether i.i.d. x, y noise at R² 0.25 alone is enough to break the policy;
   that is condition `xy025`. If it is, horizontal label quality is the bottleneck; if `xy025` also does well,
   the IDM's error structure (not its size) is what hurts, which session 4 (`all_idm` vs IDM labels) tests.

## Caveats

- One seed for training and one noise seed. 100 evaluation scenes.
- i.i.d. Gaussian noise is a model of label error, not the IDM's actual errors.
- Results are read on the CPU simulator, which matches how the demos were recorded.

## Why `xy025` did not run: out of memory

`train.py` keeps every training frame in RAM, resized to 224×224 (`src/data/demo_dataset.py`): about
78,000 × 3 × 224 × 224 bytes ≈ 11.7 GB per 1,000-demo run. Two such runs in parallel need over 23 GB plus
the simulators and data workers, which exceeds the Kaggle machine's memory. `xy025` finished its step-0 evaluation
and was killed about 6 minutes into training (`runs/xy025/train_xy025.log` ends abruptly; the kernel log reports
`Killed`, exit 137). The N = 100 pair fitted because arm A has only 100 demos. `zg_idm` ran alone after that and
is unaffected (its own log, `DONE` marker and 4 evaluations are complete).

**Fix (for the rerun):** the loader now keeps frames at their recorded 128×128 and resizes each sample to
224×224 when it is read, which is about 3.8 GB per 1,000-demo run instead of 11.7 GB. It is the same resize on
the same frames: `scripts/check_lazy_resize.py` found 0 differing pixels over all 7,720 samples of 100 demos
(Mac, torch 2.6), and the runner repeats this check on Kaggle before training. Loading a batch of 256 this way
took about 20 ms on the Mac, against about 1.1 s per training step on the T4.

## Files

- `labels/`: corruption settings and achieved label quality of each condition (from `make_noisy_labels.py`).
- `runs/zg_idm/`, `runs/xy025/`: config, sessions, metrics and training log copied from the Kaggle output
  (checkpoints not committed).
- `failure_analysis/`: CPU stage analysis of the `zg_idm` final checkpoint (`scripts/failure_analysis.py`,
  100 episodes, 10 envs, scene seed 10000).
- `compare_N100/`: combined summary and figures with arms A, B (N = 100) and the upper bound.
- `paired_tests.json`: counts and exact McNemar tests (each entry: only-first, only-second, p).
