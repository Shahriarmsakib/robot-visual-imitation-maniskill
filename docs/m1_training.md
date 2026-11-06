# M1: ACT trainer, evaluation and smoke tests

Date: 2026-11-03. Code: `src/`, `scripts/train.py`, `configs/`, `tests/`, `notebooks/kaggle_train.ipynb`.

## Relation to the official baseline

The upstream ACT baseline (ManiSkill `v3.0.1`) is vendored unmodified in `third_party/maniskill_act/`. The model code (`act/detr/`) is imported as-is. The trainer and evaluator in `src/` are derived from `train_rgbd.py` and `act/evaluate.py`; each file's header lists its changes.

**Unchanged:** the model, the loss (L1 + 10 × KL), AdamW with a separate backbone learning rate, the 10× LR drop at 2/3 of training (stepped every batch), EMA with power 0.75 (the EMA model is what gets evaluated), batch size, chunk size 30, temporal ensembling, and the data padding at the end of episodes.

**Changed**, with reasons:

| Change | Why |
|---|---|
| Device chosen automatically (cuda > mps > cpu) | Upstream calls `.cuda()` and crashes without NVIDIA |
| Full checkpoints in `latest.pt` (model, EMA, optimizer, scheduler, RNG, step) | Upstream saves weights only, so an interrupted run can't continue |
| Exact resume, including data order | Kaggle stops sessions at 12 h |
| Evaluation at step 0 (untrained), every `eval_freq`, and after the final update | Upstream evaluated after 1 update and never evaluated the final model |
| Seeded evaluation scenes (`eval_seed`) | Upstream evaluation was unseeded, so checkpoints were tested on different scenes. **Correction (2026-11-06):** until `enhanced_determinism=True` was added to `make_eval_envs`, only the first scene of each parallel env was fixed; later scenes came from automatic resets driven by global RNG state. Phase A's training-time evaluations therefore share their first 10 scenes and draw the other 90 at random (still valid random samples, but not identical sets across checkpoints) |
| Metrics per trial with counts and 95% Wilson intervals | Avoids the batch-vs-trial confusion noted in the brief |
| Training RNG saved and restored around evaluation | In-process evaluation otherwise changes the training randomness |
| Same-step autoreset for CPU vector envs | Keeps the upstream evaluator correct on gymnasium ≥ 1.0 |
| JSON-lines logs; TensorBoard/W&B removed | Structured, plottable logs with no account needed |

"Step N" means N optimizer updates have been applied. Upstream's "iteration 0" evaluation came after one update; here step 0 is before any training.

## Outputs of a run (`<runs_dir>/<exp_name>/`)

| File | Contents |
|---|---|
| `config.json` | Full config at the start of the run |
| `sessions.jsonl` | One row per session: device, sim backend, git commit, package versions, dataset SHA-256 and episode keys |
| `metrics.jsonl` | `train` rows (loss, l1, kl, lr, sec/step) and `eval` rows (trial counts, rates, CIs) |
| `eval/step_*.npz` | Per-trial evaluation results |
| `checkpoints/latest.pt` | Full, resumable state (about 318 MB) |
| `checkpoints/{step_*,best_eval_*,final}.pt` | EMA policy weights only (about 64 MB) |
| `DONE` | Written only when every step finished |

## Verification

| Check | Result |
|---|---|
| Unit tests (`pytest`) | 12 passed: dataset shapes and padding, model loss and action shapes, metrics, sampler, RNG, atomic save |
| End-to-end tests (`pytest --run-slow`) | 2 passed: a run stopped at step 13 and resumed matches an uninterrupted 20-step run exactly (every loss, bit-identical final weights, on CPU); resuming with a changed learning rate is refused |
| Mac smoke run (`configs/smoke_mac.json`, MPS + `physx_cpu`, 2 parallel eval workers) | Completes; evaluations at steps 0, 10, 20 with 2 trials each (0/2 success, expected after 20 tiny updates) |
| Mac training speed, batch 256, 100 demos | 1.59 s/step on MPS (Apple M5), so about 13 h for 30k steps |
| Kaggle notebook | **Not run yet.** Generated and syntax-checked only |

Exact resume was verified on CPU. On CUDA, cuDNN kernels can differ slightly from run to run even with `torch_deterministic`; a resumed GPU run gets the same data order and state but may not be bit-identical.

## macOS issue found during M1

CPU evaluation workers on macOS failed with `IndexError: unordered_map::at: key not found` when the launching script imported ManiSkill at module level: the workers inherited an initialized SAPIEN/Metal renderer. `scripts/train.py` therefore imports the trainer inside `if __name__ == "__main__":`. New entry-point scripts must follow the same pattern.

## Commands

```bash
source scripts/env_macos.sh                                      # macOS, every new terminal
python -m pytest -q                                              # fast tests
python -m pytest -q --run-slow                                   # + end-to-end resume test (~1.5 min)
python scripts/train.py --config configs/smoke_mac.json          # ~30 s smoke run
python scripts/train.py --config configs/baseline_act_rgb_100d.json --seed 1 \
  --exp-name act_pickcube_rgb_baseline_100d_seed1                # full baseline (GPU recommended)
```

Rerunning the same command resumes from `latest.pt`. `--max-hours H` stops cleanly with exit code 3.
