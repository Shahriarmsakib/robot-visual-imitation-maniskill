# Vendored: ManiSkill ACT baseline

| | |
|---|---|
| Source | https://github.com/haosulab/ManiSkill/tree/v3.0.1/examples/baselines/act |
| Tag / commit | `v3.0.1` / `a4a4f9272ad64b1564035874b605ceb687b63ed8` |
| License | Apache-2.0 (`LICENSE`, copied from the ManiSkill repository root) |
| Original method | Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023 (see `README.md`) |

Files here are **unmodified** copies. `act/` (model, upstream evaluation and env helpers) is imported by this project; `train_rgbd.py` is kept only as the reference the project's trainer was derived from.

The project's own trainer and evaluator (`src/`, `scripts/train.py`) are modified derivatives of `train_rgbd.py` and `act/evaluate.py`. Each of those files states its changes in its header.
