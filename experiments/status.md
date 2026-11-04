# Experiment status

Only runs with real logs appear as complete. Every result row needs a run ID, seed, trial count and checkpoint reference.

| Milestone | Status | Evidence |
|---|---|---|
| M0 audit | Done (2026-11-03) | `docs/m0_audit.md`, `experiments/manifests/env_darwin.json` |
| M1 baseline code + smoke tests | Done on Mac (2026-11-03); Kaggle GPU smoke test pending | `docs/m1_training.md`, `pytest --run-slow` (14 passed) |
| M2 logging, plotting, rollout export | Not started (structured logging done in M1) | |
| M3 experiments | Not started | |
| M4 publication | Not started | |

## Runs

| Run ID | Phase | Demos | Seed | Steps | Eval trials | success_once | success_at_end | Checkpoint | Status |
|---|---|---|---|---|---|---|---|---|---|
| colab_pilot (prior) | A | 100 | 1 | 6,700+ of 30,000 | 100 at iter 5,000 | 2% | 2% | not inventoried | Interrupted; pilot only, not reused |
| act_pickcube_rgb_baseline_100d_seed1 | A | 100 | 1 | 0 of 30,000 | | | | | Not started |
