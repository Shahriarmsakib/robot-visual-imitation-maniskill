# Phase 3: is arm B's IDM undertrained? (plan and prediction, written before the policy runs)

## Why

Arm B (policy on IDM pseudo-labels) gave 0/100 at two seeds. Its IDM was trained for 3,000 steps. While adding the
IDM recipe of Morin et al. (2026, arXiv 2602.02762, Appendix D.2: IMPALA CNN, Adam 2e-4 with warm-up and linear decay,
batch 128, 60k steps) we found that training length alone changes the labels a lot. Label quality on the 900
action-free demos (N = 100 labelled demos, seed 0, Mac, same data file, `scripts/label_error_structure.py`):

| IDM | Steps | Step R² x / y / z | Slope pred on true x / y / z | Error lag-1 autocorr x / y / z | Gripper acc | Mac time |
|---|---|---|---|---|---|---|
| Ours (arm B) | 3k | 0.10 / 0.44 / 0.76 | 0.35 / 0.56 / 0.72 | 0.91 / 0.87 / 0.90 | 94.5% | ~2 min |
| Ours, longer | 60k | 0.70 / 0.93 / 0.94 | 0.73 / 0.92 / 0.93 | 0.77 / 0.69 / 0.75 | 95.6% | 37 min |
| Morin et al. recipe (64×64 input, 2 frames, SmoothL1 for xyz) | 60k | 0.76 / 0.95 / 0.95 | 0.75 / 0.94 / 0.94 | 0.83 / 0.71 / 0.80 | 99.0% | ~3.5 h |

(3k row: `results/action_free/idm/` seed 0; the 60k rows: `results/action_free/idm_small60k/`, `idm_morin/`.
Kaggle copies of the 3k IDM, trained on Kaggle's conversion of the same demos: x 0.23 / 0.12, y 0.29 / 0.51,
z 0.81 / 0.80 at seeds 0 / 1.)
So most of the gain comes from training longer; the IMPALA network adds a little on xyz and clearly on the gripper.
Arm B's 0/100 is therefore a result for an undertrained IDM, not for IDM pseudo-labels in general.

## Design

One Kaggle session per training seed (0, 1, 2), runner `scripts/run_phase3_noise.sh` in its `idm:` mode: the IDM is
trained inside the session on the N = 100 labelled demos with the session's seed, labels the 900 action-free demos,
and the policy (ACT RGB, 12k steps, same settings as every other run) trains on them.

- GPU 0, `idm60k`: our IDM, 3 frames (offsets 0 1 2, as arm B), 60k steps, other settings as arm B.
- GPU 1, `idm_morin`: Morin et al.'s recipe, `--arch impala --batch 128 --lr 2e-4 --schedule warmup_linear --loss smoothl1 --res 64 --offsets 0 1 --steps 60000`.
- Read on the CPU failure analysis (100 episodes, scene seed 10000), paired against the upper bound, arm B and
  `xy_shrink07` of the same seed.

## Prediction (written 2026-11-26, before any of these policies were trained)

Reference points from the noise sweep (CPU success at end, seeds 0 / 1): upper bound 95 / 97; random x, y noise of
the 3k IDM's size 93 / 95; x and y shrunk ×0.7 (`xy_shrink07`) 25 / 23; ×0.4 1 / 1; time-correlated noise 52 / 48;
3k IDM (arm B) 0 / 0.

The 60k labels still shrink x (slope ≈ 0.73, close to `xy_shrink07`) but y and z only a little (≈ 0.93), and their
errors persist over time (lag-1 autocorrelation 0.7–0.8). If label-error structure is what decides success, we expect:

1. Both new label sets lift the policy well above arm B and above `xy_shrink07`: success above 25 at every seed.
2. Both stay below the upper bound: success under 90 at a majority of seeds, because shrinkage and persistence remain.
3. `idm_morin` at least as good as `idm60k` (slightly less shrinkage, better gripper), but the difference may be within
   the seed spread (up to ~22 episodes for mildly noisy labels).

If both reach the upper bound (≥ 93 at every seed), the remaining x shrinkage of ≈ 0.73 on one axis does not matter
and prediction 2 fails. If either stays near 0, label structure as we measure it does not explain the failure.
Labels made on Kaggle differ a little from the Mac labels above; each session records its own label quality.

## Outcome (2026-11-26)

Results: [`results/action_free/idm_budget/README.md`](../results/action_free/idm_budget/README.md). Success 27–30/100
(vs 0 for arm B, 95–97 for true labels), level with `xy_shrink07` (25, 23). Prediction 1 held only in part (above 25,
but not above `xy_shrink07`), 2 held, 3 within the spread. The seed-1 `idm60k` IDM collapsed to a constant output.
Measured on a T4: `idm60k` IDM 21 min, `idm_morin` IDM 2.2–2.3 h, whole session 6.45 h.

## Sessions

| Session | Account | Seed | GPU 0 | GPU 1 | Status |
|---|---|---|---|---|---|
| 19 | main | 0 | `idm60k` | `idm_morin` | done 2026-11-26: CPU 30 / 27 |
| 20 | second | 1 | `idm60k` | `idm_morin` | done 2026-11-26: CPU 0 (IDM collapsed) / 29 |
| 21 | main | 2 | `idm60k_wl`: as `idm60k` but the learning rate warms up and decays linearly to 0 (the seed-1 constant-rate IDM collapsed) | `idm_morin` | prepared 2026-11-26 |
| 22 | main | 1 | `idm60k_wl` (replaces the collapsed run) | to decide (e.g. the shrinkage fix) | planned |
| — | — | 0 | `idm60k_wl`, so all three seeds use the same IDM recipe | — | planned |

Session 15 (seed 2 of the upper bound and `xy025`, cancelled by Kaggle on 2026-11-24) is rerun in parallel on the second
account as `phase3-seeds-s15r`.

Each session: setup ~20 min, both IDMs trained one after the other (time on a T4 not measured yet; Mac: 37 min and
~3.5 h), then the two policies in parallel (~3.5 h). Estimate 6–7 GPU-hours; MAX_HOURS caps each policy at 6 h.
