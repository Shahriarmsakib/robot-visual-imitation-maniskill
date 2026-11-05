# Phase B plan: data efficiency (agreed 2026-11-05, not started)

**Question:** how does final success change with 10, 25, 50 and 100 expert demonstrations, across 3 seeds?

## Protocol
- 12 runs: demo budgets 10/25/50/100 × seeds 1–3. `act_pickcube_rgb_baseline_100d_seed1` (Phase A) is the 100-demo seed-1 run, so 11 are new.
- Identical to Phase A (`configs/baseline_act_rgb_100d.json`): same model and hyperparameters, 30,000 steps, 100-trial evals every 5,000 steps, eval seed 10000, `physx_cuda` on Kaggle. Only `num_demos` and `seed` change.
- Names: `act_pickcube_rgb_baseline_{N}d_seed{S}`.
- **Demo subsets: nested and fixed.** Budget N uses the first N episodes (traj_0 … traj_{N-1}), the same for every seed; the exact keys are logged in each run's `sessions.jsonl`. Seeds vary training randomness only.
- **Fixed 30,000 steps for every budget** (equal compute; about 10,600 / 4,100 / 2,000 / 1,000 passes over the data for 10 / 25 / 50 / 100 demos). Headline = final model; best checkpoint reported as secondary.
- All Phase B training and evaluation on Kaggle (one simulator backend, one software stack).

| Demos | Samples | Batches per epoch (256) |
|---|---|---|
| 10 | 726 | 2 |
| 25 | 1,852 | 7 |
| 50 | 3,760 | 14 |
| 100 | 7,720 | 30 |

## Schedule (two runs per Kaggle session, one per T4)

| Session | GPU 0 | GPU 1 | Milestone |
|---|---|---|---|
| 1 | 10d seed 1 | 25d seed 1 | |
| 2 | 50d seed 1 | 100d seed 2 | one-seed curve complete |
| 3 | 10d seed 2 | 25d seed 2 | |
| 4 | 50d seed 2 | 100d seed 3 | two seeds at every budget |
| 5 | 10d seed 3 | 25d seed 3 | |
| 6 | 50d seed 3 | free | three seeds: final figure |

Estimated ~63 GPU hours, about 2–2.5 weeks of Kaggle's ~30 h/week.

## To build before session 1
1. Notebook: launch a list of runs in parallel, one per GPU (`CUDA_VISIBLE_DEVICES`), separate logs, both resumable.
2. `collect_results.py`: warn when runs used datasets with different SHA-256 (Kaggle and Mac conversions differ, see `docs/phase_a_baseline.md`).
3. Two-GPU smoke test on Kaggle (1,000 steps each, ~20 min). If each run is slower than ~1.25 s/step, two 30k runs will not fit in one 11 h session and will resume in the next.

## Risks
- CPU contention between two runs on Kaggle's 4 vCPUs (measured by the smoke test).
- 10 demos may give near-0% success: valid, but only bounds the rate.
- Kaggle rebuilds the dataset each session; the SHA check confirms consistency.
