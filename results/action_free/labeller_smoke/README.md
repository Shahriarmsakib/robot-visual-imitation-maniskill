# Labeller smoke test: label structure only, no policy training (2026-11-15)

**Policy results (session 5, 2026-11-15): position-based labels rescue arm B but do not reach the upper bound, and
the label scores below did not predict that gap.** Trained on `pos_cnn` labels the policy succeeds in **50/100** CPU
episodes and on `fusion` labels in **57/100**, against 0/100 with IDM labels and 95/100 with true labels
(`../noise_sweep/README.md`, session 5). By the label scores, `pos_cnn` (chunk R² 0.98–0.99, slope 0.97–0.98)
should have done at least as well as random x, y noise (chunk R² 0.94–0.95, 93/100); it did not.

**Label measurements: labels built from estimated gripper positions avoid both flaws that break arm B, and rescaling the
IDM does not.** A small CNN that estimates the gripper position in each image, with actions taken as differences
of positions, gives labels with the right scale (slope 0.97–0.98) and errors that cancel instead of persisting
(lag-1 autocorrelation −0.07 / −0.33). Over a 30-step chunk its displacement R² is 0.98–0.99, better than the
random-noise labels that trained a 93/100 policy (0.94–0.95). These are label measurements only; whether they
translate into policy success has to be tested with training runs. One training seed per labeller.

Script: `scripts/labeller_smoke.py`. Scores: `scripts/label_error_structure.py` (`structure.json`); calibration
values in `calibration.json`.

## Protocol

- **Train:** the IDM and the position CNN train on labelled demos 0–79 (3,000 steps, same IDM recipe as the study).
- **Calibrate:** rescaling factors and noise levels come from held-out labelled demos 80–99.
- **Score:** labels of the 900 action-free demos (100–999) against their true actions, which no labeller sees.
- The gripper channel is the IDM's for every labeller; only x, y, z differ.

| Labeller | How it labels x, y, z |
|---|---|
| `idm` | Image IDM on (o_t, o_t+1), as in the study |
| `idm_rescale` | `idm` divided by its held-out slope per axis (bias correction, as in climate "variance scaling") |
| `pos_oracle` | Recorded gripper positions (`obs/extra/tcp_pose`), differenced; a linear map from displacement windows (t−1 … t+2) to the action, fit on demos 0–99 |
| `pos_cnn` | Gripper position estimated from each 128×128 image by a CNN (same trunk as the IDM), then the same map |
| `fusion` | Kalman/RTS smoother per axis: IDM-implied motion as the process model, CNN positions as measurements, noise variances from demos 80–99; then the same map |

## Results (x / y / z)

| Labeller | Step R² | Slope (label on true) | Error autocorrelation (x / y) | 30-step displacement R² |
|---|---|---|---|---|
| `idm` | 0.17 / 0.23 / 0.73 | 0.35 / 0.32 / 0.70 | 0.88 / 0.90 | 0.30 / 0.29 / 0.82 |
| `idm_rescale` | −6.17 / −2.02 / 0.63 | 1.45 / 1.00 / 1.01 | 0.73 / 0.75 | −6.06 / −2.01 / 0.82 |
| `pos_cnn` | 0.42 / 0.66 / 0.79 | 0.97 / 0.98 / 0.98 | −0.07 / −0.33 | **0.98 / 0.99 / 0.99** |
| `fusion` | 0.90 / 0.95 / 0.96 | 0.92 / 0.95 / 0.96 | 0.64 / 0.57 | **0.99 / 0.99 / 0.99** |
| `pos_oracle` (recorded positions) | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | (error near zero) | 1.00 / 1.00 / 1.00 |
| Reference: `noise_xy025` (93/100 policy) | 0.25 / 0.25 / 1.00 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 | 0.95 / 0.94 / 1.00 |

Calibration on held-out demos 80–99: IDM slope 0.24 / 0.32 / 0.69; position-CNN error SD 6.0 / 3.5 / 5.4 mm;
IDM-implied motion error SD 2.6 / 2.3 / 2.7 mm per step.

## Reading

- **The action is a near-exact function of gripper motion.** With recorded positions the linear map recovers the
  true actions with R² 0.998–1.0. Labelling therefore reduces to estimating the gripper position from images.
- **Position-based labels have the right structure.** Per-frame position errors of a few millimetres become step
  errors that are noisy (step R² 0.42–0.79) but correctly scaled and anti-correlated, so they cancel over a chunk.
  This is the kind of error the noise sweep found harmless.
- **Fusion trades a little scale for much less noise.** Combining the IDM's motion with CNN positions raises step
  R² to 0.90–0.96 with chunk R² 0.99, but leaves a slight shrink (0.92–0.96) and more persistent errors (0.57–0.64).
  `pos_cnn` vs `fusion` is a direct test of whether scale or step accuracy matters more for the policy.
- **Rescaling the IDM fails.** Dividing by the held-out slope restores the average scale but multiplies the noise
  (step R² −6.2 on x). The slope estimated on 20 held-out demos also overshoots on x (1.45). Shrinkage cannot be
  undone after the fact here, because the IDM's predictions carry too little signal on x and y.

## Why position-based labels stop at 50–57/100: diagnosis (2026-11-16)

Mac-only checks on the session-5 policies (CPU episodes) and on the Mac-built labels (same method as Kaggle; the
Kaggle label files were not kept).

**1. The policies let go of the cube at the goal.** The failure analysis flags 14 (`pos_cnn`) and 20 (`fusion`)
episodes as "released or dropped after reaching the target", against none for the upper bound and `xy025`. Nearly
all failed episodes that grasped the cube end with it below the goal (47 / 47 and 41 / 42).

**2. The gripper labels say "open" while holding at the goal.** Both labellers take their gripper channel from the
IDM. Phases on the 900 demos from the recorded state: *approach* (not grasped), *carry* (grasped), *hold* (grasped,
cube within 2.5 cm of the goal).

| Gripper labels | Approach wrong | Carry wrong | Hold wrong |
|---|---|---|---|
| IDM (used by `pos_cnn`, `fusion`) | 4.7% | 7.6% | **22.8%** (1,258 of 5,506 steps, all "open" for a closed gripper) |
| `zg_idm` (random 6% flips; 90/100 policy) | 6.2% | 5.9% | 5.6% |
| Finger width from images (CNN), see 4 | 11.8% | 2.8% | **1.9%** |

In the last 10 steps of each episode the IDM's gripper label is wrong 16.9% of the time. A plausible reason:
while the arm holds still, consecutive frames barely change, so the IDM cannot see the command. Random flips at
the same overall rate (`zg_idm`) were nearly harmless, so again it is the *structure* (errors concentrated in one
phase) that matters.

**3. The x, y, z labels are not the main problem.** Mean label error (bias) is below 0.01 in every phase, and
gripper position explains at most 5% of the error variance (quadratic fit), so the errors do not depend on where
the gripper is. During *hold* the error SD (`pos_cnn` 0.064 / 0.046 / 0.086, `fusion` 0.032 / 0.022 / 0.044) is
similar to `xy025`'s (0.055 / 0.057) on x, y, but z is noisy too, which `xy025` never was.

**4. A position-style fix for the gripper.** `scripts/gripper_label_smoke.py` estimates the finger opening
(sum of the two finger joints) from each image with the same CNN trunk (trained on demos 0–79; held-out error SD
9.1 mm over a 37–80 mm range) and maps the opening over t … t+3 to the command with a logistic fit on demos 0–99.
With recorded openings the map is exact (100%). With estimated openings, *hold* errors fall from 22.8% to 1.9% and
*carry* errors from 7.6% to 2.8%, but *approach* errors rise from 4.7% to 11.8% (closing too early or late around
the grasp). Policy effect untested.

Built into `scripts/labeller_smoke.py` as `fusion_gripw` (same models, 3,000 steps, Mac): gripper labels wrong
in 7.3% of *approach*, 1.9% of *carry* and **1.1% of *hold*** steps (IDM 4.7 / 7.6 / 22.8%); overall accuracy 95.2%
vs 92.8%. Held-out finger-opening error and the map's fit are in the labeller's `calibration.json`.

**Session 6 result (2026-11-16):** with the true gripper (`fusion_truegrip`) the drops disappear but success at end
is 60/100 vs 57 (p = 0.74); with the finger-width gripper (`fusion_gripw`) 53/100. The episodes that dropped the
cube now stop short and low, so the gripper was not the main cause of the gap; the remaining failure is goal
precision (`../noise_sweep/README.md`, session 6).

## Caveats

- The labels used for the session-5 policies were built on Kaggle (GPU) and differ somewhat from the Mac labels
  scored here: achieved step R² for `pos_cnn` x / y / z 0.21 / 0.61 / 0.77 on Kaggle vs 0.42 / 0.66 / 0.79 on the Mac;
  `fusion` 0.89 / 0.95 / 0.96 vs 0.90 / 0.95 / 0.96 (`../noise_sweep/labels/`). Their slope and chunk scores were
  not measured (the Kaggle npz files were not kept).
- The chunk and slope scores did not predict the policy result for these labellers (see the short answer).
- The setting matters. In this pipeline the action-free demos keep their recorded robot state, so `pos_oracle`
  is available in "robot logs without commanded actions". For video-only data (e.g. human or internet video),
  only image-based labellers (`idm`, `pos_cnn`, `fusion`) apply.
- The position CNN needs gripper positions on the labelled demos as training targets; in simulation they come
  from the recorded state, on a real robot from forward kinematics.
- One training seed per labeller; one task.
