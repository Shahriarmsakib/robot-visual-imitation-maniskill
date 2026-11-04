# Object-Centric Visual Imitation Learning in ManiSkill

Does focusing a robot policy's vision on the relevant objects make imitation learning work with fewer demonstrations and hold up better when the scene looks different? This project tests that on ManiSkill 3's `PickCube-v1` task, using the official ACT (Action Chunking with Transformers) RGB baseline as the reference.

> **Status: implementation in progress (M0 of M4 done). No training results yet.** See [experiments/status.md](experiments/status.md).

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

On macOS the simulator runs on CPU only, so full training runs on a CUDA GPU (Kaggle). Setup details and known macOS issues are in [docs/m0_audit.md](docs/m0_audit.md).

## Layout

```
configs/      experiment configs
scripts/      setup, data, train, evaluate, plot
src/          data loading, models, evaluation, visualization
tests/        dataset, model, metric and smoke tests
results/      small metrics files, figures, short media
docs/         methodology, audit, limitations
experiments/  run manifests and status tracker
```

Large files (demo `.h5` files, checkpoints) are never committed.

## Acknowledgements

Built on [ManiSkill](https://github.com/haosulab/ManiSkill) (tag `v3.0.1`) and its ACT baseline.
