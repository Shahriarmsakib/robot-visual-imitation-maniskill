# Object-centric data extraction: smoke test

**Result: SMOKE TEST PASSED** (all 7 required checks plus 1 supporting check). Run on macOS (`physx_cpu`, MoltenVK rendering), 2026-11-06. Machine-readable details: `validation.json`; figure: `smoke_frames.png`.

**What it does.** Replays the first 5 official ManiSkill PickCube-v1 motion-planning demonstrations (`traj_0`–`traj_4`, seeds 0–4) from their recorded initial simulator state with the expert's original `pd_joint_pos` actions, in `obs_mode="rgbd"`. At every step it records RGB, depth, joint positions and velocities, end-effector (TCP) pose, cube pose and goal position from the simulator, plus camera intrinsics and extrinsics. The source file (`trajectory.h5`, SHA-256 `03ca6054…`) is only read.

| Check | Result | Evidence |
|---|---|---|
| 1. Five demos processed | Pass | 5 episodes, T = 74, 74, 50, 86, 76 actions |
| 2. RGB extracted | Pass | uint8, (T+1, 128, 128, 3) |
| 3. Depth valid | Pass | int16 millimetres, (T+1, 128, 128), every pixel > 0, range 184–9,436 mm |
| 4. Cube and end-effector positions | Pass | (T+1, 7) each (xyz + wxyz quaternion); replayed cube matches the recorded simulator state within 0.34 mm |
| 5. Expert actions | Pass | original `pd_joint_pos` actions, (T, 8), copied exactly |
| 6. Observation/action alignment | Pass | T+1 observations, T actions, T success flags. Each replayed state matches the recorded state at the same index (joints 2e-5 rad, cube 0.34 mm); a one-step shift gives ≥ 0.021 rad and ≥ 4.7 mm, so a shift would be detected. Per-step success agrees on 99.7% of steps: in `traj_0` the replay first succeeds at step 70 instead of 69, because of the 0.34 mm replay drift |
| 7. Demos successful | Pass | 5/5 successful at the last step, in the replay and the recording |
| 8. Ground truth projects onto the cube (supporting) | Pass | at t = 0 the cube's 3-D position projects onto red pixels in 5/5 episodes; depth there agrees with the geometry within 0.8 cm |

**Dataset.** `data/object_centric/pickcube_5demos_rgbd.h5` (2.9 MB, not committed; regenerate in about 5 seconds). One group per episode (`episodes/ep_000` … `ep_004`):

```
obs/rgb (T+1,128,128,3) uint8      obs/depth (T+1,128,128) int16 mm
robot/qpos (T+1,9)  robot/qvel (T+1,9)  robot/tcp_pose (T+1,7)  robot/is_grasped (T+1,)
objects/cube_pose (T+1,7)  objects/cube_pose_recorded (T+1,7)  objects/goal_pos (T+1,3)
camera/intrinsic_cv (3,3)  camera/extrinsic_cv (3,4)
actions (T,8)  success (T,)
```

Alignment rule (also stored in the file): index t of every per-step array is the state *before* `actions[t]`; index T is the final state; `success[t]` is evaluated *after* `actions[t]`.

**Notes for the next stage.**
- Actions are the expert's original joint-position targets. The Phase A ACT baseline uses end-effector deltas (`pd_ee_delta_pos`); converting is one option of ManiSkill's replay tool if the object-centric policy should share the baseline's action space.
- The camera is fixed, so the camera parameters are stored once per episode; with them, any 3-D position can be projected into the image, e.g. to build cube-centred crops.

**Reproduce.**
- Mac: `source scripts/env_macos.sh && python scripts/object_centric_smoke_test.py`.
- Colab: `notebooks/object_centric_smoke_test_colab.ipynb` on a T4 GPU runtime. Not yet run on Colab.
