# Object-Centric Visual Imitation Learning in ManiSkill

Does focusing a robot policy's vision on the relevant objects make imitation learning work with fewer demonstrations and hold up better when the scene looks different? This project tests that on ManiSkill 3's `PickCube-v1` task, using the official ACT (Action Chunking with Transformers) RGB baseline as the reference.

> **Status: implementation in progress (M0 and M1 done). No training results yet.** See [experiments/status.md](experiments/status.md).

## Plan

| Phase | Question |
|---|---|
| A. Baseline | Reproduce ACT on PickCube-v1 from RGB images (100 demos, 30k iterations) |
| B. Data efficiency | How does success change with 10, 25, 50 and 100 demos? |
| C. Object-centric vision | Does an object-focused view (crop or mask around cube and gripper) help? |
| D. Visual robustness | How much does success drop under camera, lighting and texture changes? |

All experiments are in simulation. Nothing here claims transfer to a real robot.

## Setup (macOS, Apple Silicon)

```bash
scripts/setup_macos.sh            # Python 3.11 venv + pinned packages + local MoltenVK
source scripts/env_macos.sh       # every new terminal
scripts/prepare_data.sh 100       # download official demos, convert 100 to RGB
python scripts/check_env.py --render-test
```

On macOS the simulator runs on CPU only and training uses Apple's GPU (MPS). Setup details and known macOS issues are in [docs/m0_audit.md](docs/m0_audit.md).

## Training

```bash
python -m pytest -q --run-slow                                   # tests, incl. exact-resume check
python scripts/train.py --config configs/smoke_mac.json          # 30-second end-to-end smoke run
python scripts/train.py --config configs/baseline_act_rgb_100d.json --seed 1 \
  --exp-name act_pickcube_rgb_baseline_100d_seed1                # Phase A baseline
```

Rerunning a command resumes from its last checkpoint. Full runs are meant for a CUDA GPU: [notebooks/kaggle_train.ipynb](notebooks/kaggle_train.ipynb) runs the same scripts on Kaggle and resumes across 12-hour sessions. What changed relative to the official ACT trainer, and how it was verified, is in [docs/m1_training.md](docs/m1_training.md).

## Layout

```
configs/      experiment configs
scripts/      setup, data, train, evaluate, plot
src/          data loading, models, evaluation, visualization
tests/        dataset, model, metric and smoke tests
results/      small metrics files, figures, short media
docs/         methodology, audit, limitations
experiments/  run manifests and status tracker
third_party/  unmodified upstream ACT baseline (Apache-2.0)
```

Large files (demo `.h5` files, checkpoints) are never committed.

## Acknowledgements

Built on [ManiSkill](https://github.com/haosulab/ManiSkill) (tag `v3.0.1`) and its ACT baseline (Apache-2.0, vendored in `third_party/maniskill_act/`). ACT is from Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023.
