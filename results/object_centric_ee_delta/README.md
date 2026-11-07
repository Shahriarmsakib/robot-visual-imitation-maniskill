# Phase 2A: ACT-compatible object-centric dataset: validation report

**Result: DATASET VALIDATION PASSED.** All 5 demonstrations converted and replayed successfully; all 7 checks pass. Built on macOS (`physx_cpu`), 2026-11-06. No model was trained. Machine-readable: `summary.json`; figure: `ee_delta_frames.png`.

## How it was built

1. **Conversion (official tool).** The first 5 official PickCube-v1 motion-planning demos (`traj_0`–`traj_4`, the same ones as the smoke test) were copied to a work folder and converted with:
   `python -m mani_skill.trajectory.replay_trajectory --use-first-env-state -c pd_ee_delta_pos -o rgbd --save-traj -b physx_cpu --count 5`.
   The 8-D joint targets were not reinterpreted by hand. The source file (SHA-256 `03ca6054…`) is never written, and its hash is checked before and after.
2. **Replay and extraction.** Each converted demo is replayed from its recorded initial state with the converted 4-D actions in a `pd_ee_delta_pos` environment. RGB, depth, cube pose, TCP pose, goal, joint state, camera parameters and success are read from the simulator at every step.

## Results

| Episode | Source | Converted T | Original T | Actions | Conversion | Replay success |
|---|---|---|---|---|---|---|
| ep_000 | traj_0 | 74 | 74 | (74, 4) | ok | yes |
| ep_001 | traj_1 | 74 | 74 | (74, 4) | ok | yes |
| ep_002 | traj_2 | 50 | 50 | (50, 4) | ok | yes |
| ep_003 | traj_3 | 86 | 86 | (86, 4) | ok | yes |
| ep_004 | traj_4 | 76 | 76 | (76, 4) | ok | yes |

| Check | Result | Evidence |
|---|---|---|
| 1. Five converted demos replayed | Pass | 5/5 converted, none missing |
| 2. Four action components | Pass | every action is 4-D: [dx, dy, dz, gripper] |
| 3. Synchronized | Pass | all per-step arrays have T+1 entries, actions and success T; each replayed state equals the converted file's state at the same index (joints 2e-6 rad, cube 0.00 mm); a one-step shift would give ≥ 0.021 rad / ≥ 4.7 mm |
| 4. All demos succeed | Pass | 5/5 at the last step, in the replay and the converted file; per-step success agrees on 100% of steps |
| 5. Valid for `pd_ee_delta_pos` | Pass | all 360 actions inside the controller's action space [-1, 1]⁴; xyz components -0.266 to 0.399; gripper exactly -1 or +1 |
| 6. Poses come from the converted replay | Pass | equal to the converted file's states (TCP and cube 0.00 mm). Against the original joint-control replay, the TCP path agrees within 0.25 mm (the converter follows the expert's path) but the arm posture differs by 0.06–0.31 rad in the joints, so the two replays are clearly distinct |
| 7. No missing values, shifts or bad depth | Pass | no NaN/inf; depth int16 mm, every pixel > 0, range 185–9,436 mm; the ground-truth cube projects onto red pixels at t = 0 in 5/5, and depth there agrees within 0.8 cm |

## Dataset

`data/object_centric/pickcube_5demos_objectcentric_ee_delta.h5`: 2.9 MB, not committed. Regenerate with:

```bash
python scripts/build_objectcentric_ee_delta.py --num-demos 5
```

Per episode (`episodes/ep_000` … `ep_004`; shapes shown for ep_000, T = 74):

| Dataset | Shape | Notes |
|---|---|---|
| `obs/rgb` | (75, 128, 128, 3) uint8 | base camera |
| `obs/depth` | (75, 128, 128) int16 | millimetres |
| `objects/cube_pose` | (75, 7) | simulator ground truth, xyz (m) + wxyz quaternion |
| `objects/cube_pose_recorded` | (75, 7) | from the converted file, for cross-checking |
| `objects/goal_pos` | (75, 3) | goal xyz (m) |
| `robot/tcp_pose` | (75, 7) | end-effector pose |
| `robot/qpos`, `robot/qvel` | (75, 9) | joints + 2 finger joints (debugging) |
| `robot/is_grasped` | (75,) | |
| `camera/intrinsic_cv`, `camera/extrinsic_cv` | (3, 3), (3, 4) | fixed camera, OpenCV convention |
| `actions` | (74, 4) float32 | [dx, dy, dz, gripper], `pd_ee_delta_pos`, normalised to [-1, 1] |
| `success` | (74,) | after each action |
| `obs_step`, `action_step` | (75,), (74,) | explicit step indices |

Alignment: index t of every per-step array is the state before `actions[t]`; index T is the final state; `success[t]` is evaluated after `actions[t]`.

## Notes

- The converter keeps each demo's length and reproduces the expert's end-effector path within 0.25 mm, so these are the same demonstrations expressed in ACT's action space.
- Built on macOS. A Linux (Colab/Kaggle) build may differ by sub-millimetre amounts and rare pixels (see `results/object_centric_smoke/README.md`); use one platform per experiment.
- Tests: `python -m pytest -q tests/test_objectcentric_dataset.py` (dataset integrity; `--run-slow` also rebuilds a 2-demo copy end to end).
- Colab: sections 7–8 of `notebooks/object_centric_smoke_test_colab.ipynb` (not yet run on Colab).
