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
