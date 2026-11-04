# M0 audit: environment, data and prior work

Date: 2026-11-03. Machine-readable record: `experiments/manifests/env_darwin.json` (regenerate with `python scripts/check_env.py --render-test`).

## Local environment (macOS)

| Item | Value |
|---|---|
| Hardware | Apple M5, 24 GB RAM, no NVIDIA GPU |
| Python | 3.11.18 (installed with `uv`, project-local `.venv/`) |
| Packages | mani_skill 3.0.1, sapien 3.0.3, torch 2.6.0, torchvision 0.21.0, gymnasium 1.4.0, pin 4.1.0 |
| Devices | PyTorch MPS available; CUDA not available |
| Simulator | `physx_cpu` only (`physx_cuda` needs NVIDIA) |
| Rendering | Works through MoltenVK v1.4.2 (Vulkan on Metal), project-local in `.tools/` |
| Speed | 100 PickCube-v1 steps with RGB cameras: about 0.25 s (1 env) |

Two macOS-specific fixes were needed; both are scripted in `scripts/setup_macos.sh`:

1. **No Vulkan driver.** SAPIEN always creates a renderer, so every environment failed with `vk::createInstanceUnique: ErrorIncompatibleDriver`. Fixed by pointing `VK_ICD_FILENAMES` at MoltenVK (`scripts/env_macos.sh`).
2. **No Pinocchio.** SAPIEN's macOS wheel does not bundle Pinocchio, which the `pd_ee_delta_pos` controller uses for inverse kinematics, giving `TypeError: 'NoneType' object is not callable`. Fixed by installing `pin==4.1.0`.

`mplib` (motion planning) is not installed; it is not needed because the official demos are downloaded rather than regenerated.

## Dataset

| Item | Value |
|---|---|
| Source | Official ManiSkill PickCube-v1 motion-planning demos (1,000 episodes, all successful, `pd_joint_pos`) |
| Converted | First 100 episodes, replayed on `physx_cpu` with `--use-first-env-state` to RGB + `pd_ee_delta_pos` |
| Replay success | 100/100 |
| File | `data/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5` (40 MB, not committed) |
| SHA-256 | `d0f2a05140652bcebce8aebe5aad1c79b3ebe63929398669a8fefff27ef8c2cc` |
| Samples | 7,720 (actions `(T, 4)`, RGB `(T+1, 128, 128, 3)` uint8) |

This matches the earlier Colab dataset in sample count, shapes and replay success. Byte-level identity with the Colab file was not checked because that file is not available here.

## Upstream code

ACT baseline: `haosulab/ManiSkill` tag `v3.0.1`, commit `a4a4f9272ad64b1564035874b605ceb687b63ed8`, path `examples/baselines/act` (`train_rgbd.py`, `act/evaluate.py`, `act/make_env.py`, `act/utils.py`, `act/detr/`). It is not part of the pip package; M1 brings it in with attribution.

## Prior Colab work

Per the project brief: a 30,000-iteration baseline on a Colab T4 reached at least iteration 6,700 before the GPU quota ran out, with 2% success (`success_once` and `success_at_end`, 100 episodes) at iteration 5,000. Its checkpoints were **not** inventoried here (Google Drive is not accessible from this machine), and by decision the baseline will be **retrained from scratch** with a fixed seed. The earlier run is cited only as a pilot.

## Implications for the plan

- The Mac can run the full data pipeline, CPU-simulator evaluation and short training smoke tests (on MPS).
- Full training uses Kaggle (free T4/P100, about 30 GPU hours a week, 12-hour sessions), so training must checkpoint and resume reliably.
- Evaluation on the Mac uses `physx_cpu`; Kaggle can use `physx_cuda`. Results from the two backends should be labeled with the backend used.
