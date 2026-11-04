# Experiment status

Only runs with real logs appear as complete. Every result row needs a run ID, seed, trial count and checkpoint reference.

| Milestone | Status | Evidence |
|---|---|---|
| M0 audit | Done (2026-11-03) | `docs/m0_audit.md`, `experiments/manifests/env_darwin.json` |
| M1 baseline code + smoke tests | Done (2026-11-03): Mac tests and Kaggle T4 smoke run | `docs/m1_training.md`, `pytest --run-slow` (14 passed), `kaggle_gpu_smoke_seed1` below |
| M2 logging, plotting, rollout export | Done (2026-11-03); figures are "pending" templates until a run completes | `results/README.md`, `pytest --run-slow` (19 passed) |
| M3 experiments | Not started | |
| M4 publication | Not started | |

## Runs

| Run ID | Phase | Demos | Seed | Steps | Eval trials | success_once | success_at_end | Checkpoint | Status |
|---|---|---|---|---|---|---|---|---|---|
| colab_pilot (prior) | A | 100 | 1 | 6,700+ of 30,000 | 100 at iter 5,000 | 2% | 2% | not inventoried | Interrupted; pilot only, not reused |
| kaggle_gpu_smoke_seed1 | smoke | 100 | 1 | 2,000 of 2,000 | 100 at steps 0/1k/2k | 0 / 0 / 5 of 100 | 0 / 0 / 2 of 100 | Kaggle output (not kept) | Pipeline check only, not a baseline: shortened schedule (LR drop at step 1,333) |
| act_pickcube_rgb_baseline_100d_seed1 | A | 100 | 1 | running | | | | | Launched on Kaggle T4, 2026-11-03 |

## Measured compute

| Where | Setup | Training speed (batch 256) | Eval, 100 trials | Full 30k-step run |
|---|---|---|---|---|
| Kaggle T4 (1 of 2 GPUs) | Python 3.13.16, torch 2.11.0+cu130, `physx_cuda`, code 65a2085 | 1.089 s/step (median) | ~88 s | ~9.5 h incl. evals and setup |
| Mac M5 (MPS) | Python 3.11.18, torch 2.6.0, `physx_cpu` | 1.59 s/step | not measured at 100 trials | ~13-14 h estimated |
