# Object-Centric Visual Imitation Learning in ManiSkill

Does focusing a robot policy's vision on the relevant objects make imitation learning work with fewer demonstrations and hold up better when the scene looks different? This project tests that on ManiSkill 3's `PickCube-v1` task, using the official ACT (Action Chunking with Transformers) RGB baseline as the reference.

> **Status: Phase A baseline and failure analysis done; Phase 3 (learning from action-free demos) step 1 done.** See [experiments/status.md](experiments/status.md).

## Plan

| Phase | Question |
|---|---|
| A. Baseline | Reproduce ACT on PickCube-v1 from RGB images (100 demos, 30k iterations) |
| B. Data efficiency | How does success change with 10, 25, 50 and 100 demos? |
| C. Object-centric vision | Does an object-focused view (crop or mask around cube and gripper) help? |
| D. Visual robustness | How much does success drop under camera, lighting and texture changes? |
| 3. Action-free demos | When only N demos keep their actions, do IDM pseudo-labels or latent actions learned from the rest help? ([plan](experiments/phase3_action_free_plan.md), [literature](docs/literature_action_free_video.md)) |

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

## Results

**Phase A baseline (100 demos, seed 1, 30k steps, Kaggle T4):** the final policy succeeds in **18/100** trials (95% CI 12–27%) and reaches the goal at some point in 29/100. A Mac CPU-simulator re-evaluation gives 16/100. One seed only; details and limits in [docs/phase_a_baseline.md](docs/phase_a_baseline.md).

| Success during training | Example rollouts (success and failure) |
|---|---|
| ![success vs step](results/figures/fig2b_success_vs_step.png) | ![success](results/media/act_pickcube_rgb_baseline_100d_seed1/ep01_success.gif) ![failure](results/media/act_pickcube_rgb_baseline_100d_seed1/ep00_failure.gif) |

**Why it fails** ([failure analysis](results/phase_a_failure_analysis/failure_analysis_report.md), 200 episodes per checkpoint, no retraining): the policy grasps the cube in 97% of episodes but brings it into the 2.5 cm goal tolerance in only 25%. In 68% it carries the cube to about 4 cm short of the goal (usually too low) and stops there. The bottleneck is goal-reaching precision, not grasping.

**Learning from action-free demos, step 1** ([IDM label quality](results/action_free/idm/README.md), 1,000 demos, 15 runs): an inverse dynamics model trained on N labelled demos labels the other 900 demos' hidden actions. With 5–10 labelled demos its movement labels are worse than predicting no motion; from about 25 they become useful (vertical motion R² 0.63–0.81, gripper 88–95% correct), while motion along the camera's viewing axis stays hard to recover.

Tables, figures and rollout GIFs are regenerated from the raw logs in `results/runs/` by `scripts/collect_results.py`, `scripts/plot_results.py` and `scripts/export_rollouts.py`; see [results/README.md](results/README.md).

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
