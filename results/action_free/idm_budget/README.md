# IDM training budget and Morin et al.'s IDM recipe (sessions 19–20)

Plan and prediction (written before these runs): [`experiments/phase3_idm_budget.md`](../../../experiments/phase3_idm_budget.md).
Data: N = 100 labelled demos (0–99, true actions) + 900 action-free demos labelled by an IDM trained inside the
Kaggle session on the 100 labelled demos with the session's seed. Policy: ACT RGB, 12k steps, same settings as every
other run. Code 861ebbb. Main measure: CPU stage analysis (`scripts/failure_analysis.py`, `physx_cpu`, 100 episodes,
10 envs, scene seed 10000), paired exact McNemar tests (`scripts/paired_tests.py`, `../noise_sweep/paired_tests.json`).

## Conditions

- `idm60k`: our small IDM (arm B's network, 3 frames), trained 60k steps instead of 3k.
- `idm_morin`: Morin et al. (2026)'s Procgen IDM recipe adapted to continuous actions (IMPALA CNN, 64×64 input,
  2 frames, Adam 2e-4 with warm-up and linear decay, batch 128, 60k steps, SmoothL1 for xyz).

## Label quality on the 900 action-free demos (Kaggle, `label_error_structure.json`)

| Run | Step R² x / y / z | Slope x / y / z | Error lag-1 autocorr x / y / z | 30-step displacement R² x / y / z | Gripper acc | IDM time on T4 |
|---|---|---|---|---|---|---|
| seed 0 `idm60k` | 0.68 / 0.92 / 0.94 | 0.73 / 0.90 / 0.97 | 0.80 / 0.73 / 0.79 | 0.81 / 0.96 / 0.97 | 95.9% | 21 min |
| seed 0 `idm_morin` | 0.73 / 0.95 / 0.95 | 0.73 / 0.92 / 0.94 | 0.84 / 0.75 / 0.81 | 0.83 / 0.97 / 0.97 | 99.2% | 2.2 h |
| seed 1 `idm60k` | **0.00 / 0.00 / 0.00** | 0 / 0 / 0 | — | — | 54.6% (= majority) | 21 min |
| seed 1 `idm_morin` | 0.76 / 0.95 / 0.95 | 0.74 / 0.93 / 0.94 | 0.82 / 0.71 / 0.77 | 0.84 / 0.97 / 0.97 | 98.9% | 2.3 h |
| reference: arm B IDM (3k steps, Kaggle) seed 0 / 1 | x 0.23 / 0.12, y 0.29 / 0.51, z 0.81 / 0.80 | | | | 94.3% / 93.6% | |

**The seed-1 `idm60k` IDM collapsed**: it predicts a constant (final training loss 0.70, R² 0, gripper accuracy equal
to the majority class). Our small IDM at a constant learning rate of 1e-3 is not stable over 60k steps at every
seed. That run is a failed IDM, not a test of well-trained labels; its policy (0/100, never approaches the cube) is
reported but not used for conclusions. A rerun needs a stable schedule (e.g. the learning-rate decay of `idm_morin`).

## Policy results (CPU, 100 episodes, scene seed 10000)

| Run | Reached cube | Grasped | Stable lift | Reached goal | Success at end | Main failure |
|---|---|---|---|---|---|---|
| seed 0 `idm60k` | 97 | 97 | 86 | 41 | **30** | grasped, not delivered 37; grasp lost 19 |
| seed 0 `idm_morin` | 96 | 99 | 87 | 29 | **27** | grasped, not delivered 61 |
| seed 1 `idm_morin` | 95 | 96 | 83 | 33 | **29** | grasped, not delivered 53 |
| seed 1 `idm60k` (collapsed IDM) | 0 | 0 | 0 | 0 | 0 | never approaches |
| upper bound (true labels) seed 0 / 1 | 100 / 100 | 100 / 100 | 96 / 97 | 97 / 98 | 95 / 97 | |
| `xy_shrink07` (x, y × 0.7) seed 0 / 1 | 87 / 75 | 83 / 75 | 70 / 56 | 27 / 27 | 25 / 23 | |
| arm B (3k IDM) seed 0 / 1 | 9 / 24 | 12 / 27 | 7 / 9 | 0 / 1 | 0 / 0 | never approaches |
| arm A (100 labelled only) seed 0 / 1 | 97 / 98 | 96 / 97 | 90 / 95 | 11 / 16 | 3 / 13 | grasped, not delivered |

GPU (`physx_cuda`) training evaluation, final: seed 0 `idm60k` 3, `idm_morin` 13; seed 1 `idm_morin` 15, `idm60k` 0
(upper bound seed 0 on GPU: 43).

Paired tests (success at end unless stated):

- Upper bound vs new labels: 95 vs 30 (65/0, p = 5e-20), 95 vs 27 (69/1, p = 1e-19); seed 1: 97 vs 29 (68/0, p = 7e-21).
- New labels vs `xy_shrink07` of the same seed: 30 vs 25 (p = 0.44), 27 vs 25 (p = 0.86); seed 1: 29 vs 23 (p = 0.34). Not different.
- `idm_morin` vs `idm60k` (seed 0): 27 vs 30 (10/13, p = 0.68). Not different. `idm_morin` seed 0 vs seed 1: 27 vs 29 (p = 0.77).
- New labels vs arm B: success 27–30 vs 0 (p ≤ 4e-9); reaching the cube 95–97 vs 9–24 (p ≤ 9e-22).
- New labels vs arm A: same reaching and grasping (p ≥ 0.38), but reaching the goal 29–41 vs 11–16 (p ≤ 0.006).

## Reading

1. **Training the IDM properly removes arm B's collapse.** With 60k-step labels the policy reaches and grasps the cube
   as often as with labelled demos only (95–99 of 100), against 9–27 for arm B, and succeeds in 27–30 episodes against 0.
2. **It stays far below true labels** (27–30 vs 95–97) although the labels look accurate by per-step R²
   (0.68–0.76 on x, 0.92–0.95 on y and z). The policy grasps but mostly fails to bring the cube to the goal.
3. **Success matches the x/y × 0.7 shrinkage condition** (25 and 23) at both seeds (p ≥ 0.34). The new labels' remaining
   shrinkage on x (slope ≈ 0.73) is close to that condition's 0.7, while their y slope (0.90–0.93) is milder. This fits
   the noise sweep's finding that shrinkage, not error size, limits the policy, though it is not a direct test of it.
4. **Morin et al.'s recipe is not better for the policy here** (27 vs 30 at seed 0), despite better gripper labels
   (99% vs 96%); both share the x shrinkage.

## Against the prediction

| Prediction (written before the runs) | Outcome |
|---|---|
| 1. Success above 25 at every seed, above `xy_shrink07` | Above 25: yes (27, 29, 30; the collapsed seed-1 `idm60k` aside). Above `xy_shrink07`: **no**, level with it (p ≥ 0.34) |
| 2. Below 90 at most seeds | Yes, at every seed (27–30) |
| 3. `idm_morin` at least as good as `idm60k`, maybe within seed spread | Within spread (27 vs 30, p = 0.68); not better |

The policy did worse than predicted: the remaining shrinkage seems to cost as much as shrinking both x and y by 0.7.
Two seeds for `idm_morin`, one valid seed for `idm60k`.

## Files

- `runs/seed{0,1}_{idm60k,idm_morin}/`: config, metrics, sessions, training log and the IDM's label metrics from the
  Kaggle output (checkpoints and label files not committed).
- `failure_analysis/metrics/`: per-episode CPU stage analysis and run summaries.
- `label_error_structure.json`: structure of the Kaggle labels (`scripts/label_error_structure.py`).
