# Experiment status

Only runs with real logs appear as complete. Every result row needs a run ID, seed, trial count and checkpoint reference.

| Milestone | Status | Evidence |
|---|---|---|
| M0 audit | Done (2026-11-03) | `docs/m0_audit.md`, `experiments/manifests/env_darwin.json` |
| M1 baseline code + smoke tests | Done (2026-11-03): Mac tests and Kaggle T4 smoke run | `docs/m1_training.md`, `pytest --run-slow` (14 passed), `kaggle_gpu_smoke_seed1` below |
| M2 logging, plotting, rollout export | Done (2026-11-03); figures are "pending" templates until a run completes | `results/README.md`, `pytest --run-slow` (19 passed) |
| Phase A failure analysis | Done (2026-11-06) | `results/phase_a_failure_analysis/failure_analysis_report.md`: 97% grasp, 25% reach target, 16% final (200 eps, `physx_cpu`); step 20k beats 30k on paired scenes (45 vs 32, p = 0.002) |
| Phase 3 step 1: IDM pseudo-label quality (action-free demos) | Done (2026-11-08) | `results/action_free/idm/README.md`: labels worse than none at N ≤ 10, useful from N ≈ 25 (z R² 0.63–0.81, gripper 88–95%); plan in `experiments/phase3_action_free_plan.md` |
| Phase 3 step 2: policy at N = 25, labelled-only (A) vs + IDM pseudo-labels (B) | Done (2026-11-08, Kaggle, 1 seed) | `results/action_free/policy_N25/README.md`: both 0/100; stage breakdown A grasp 74 vs B 14 of 100 (paired p < 1e-12) |
| Phase 3 step 3: upper bound, ACT on all 1,000 true-labelled demos (same 12k schedule) | Done (2026-11-09, Kaggle, 1 seed) | `results/action_free/upper_N1000/README.md`: final 43/100 success at end on `physx_cuda` (95% CI 34–53%; once 87/100); 95/100 on the arm A/B CPU scenes (90/100 on a second scene set). Paired vs arm A: reached target 97 vs 1. 12k steps suffice; N = 25 is label-limited. Large GPU/CPU gap, cause open |
| M3 experiments | Not started | |
| M4 publication | Not started | |

## Runs

| Run ID | Phase | Demos | Seed | Steps | Eval trials | success_once | success_at_end | Checkpoint | Status |
|---|---|---|---|---|---|---|---|---|---|
| colab_pilot (prior) | A | 100 | 1 | 6,700+ of 30,000 | 100 at iter 5,000 | 2% | 2% | not inventoried | Interrupted; pilot only, not reused |
| kaggle_gpu_smoke_seed1 | smoke | 100 | 1 | 2,000 of 2,000 | 100 at steps 0/1k/2k | 0 / 0 / 5 of 100 | 0 / 0 / 2 of 100 | Kaggle output (not kept) | Pipeline check only, not a baseline: shortened schedule (LR drop at step 1,333) |
| act_pickcube_rgb_baseline_100d_seed1 | A | 100 | 1 | 30,000 of 30,000 | 100 per eval, 7 evals | 29/100 (final) | **18/100 (final; 95% CI 12–27%)** | final.pt (Kaggle output, not committed) | Complete, 2026-11-04. Mac `physx_cpu` re-eval: 16/100 at end. See `docs/phase_a_baseline.md` |
| phase3_upper_N1000_seed0 | 3 (upper bound) | 1,000 | 0 | 12,000 of 12,000 | 100 per eval, 4 evals | 87/100 (final) | **43/100 (final; 95% CI 34–53%)** | final.pt (Kaggle output, not committed) | Complete, 2026-11-09, code 05aace1. Mac `physx_cpu`: 95/100 (scene seed 10000), 90/100 (seed 20000). See `results/action_free/upper_N1000/README.md` |

## Measured compute

| Where | Setup | Training speed (batch 256) | Eval, 100 trials | Full 30k-step run |
|---|---|---|---|---|
| Kaggle T4 (1 of 2 GPUs) | Python 3.13.16, torch 2.11.0+cu130, `physx_cuda` | 1.089 s/step (smoke), 0.973 s/step (baseline) | ~87 s | baseline took one session (< 11 h) |
| Mac M5 (MPS) | Python 3.11.18, torch 2.6.0, `physx_cpu` | 1.59 s/step | 35 s (10 workers) | ~13-14 h estimated |
