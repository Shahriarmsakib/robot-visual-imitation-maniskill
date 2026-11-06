"""Extract aligned RGB-D, object and robot data from ManiSkill PickCube demonstrations (smoke test).

Each demonstration is replayed from its recorded initial simulator state with the expert's original
actions (pd_joint_pos), reading observations and simulator ground truth at every step. The source
demonstration file is only read.

Alignment convention (stored in the HDF5 attributes):
- per-step arrays under obs/, objects/ and robot/ have T+1 entries: index t is the state *before*
  action t; index T is the final state after the last action.
- actions[t] (T entries) is the expert action applied at state t.
- success[t] (T entries) is the environment's success flag *after* action t.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

CAMERA = "base_camera"


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def load_source_episodes(h5_path, num):
    """Read the first `num` episodes (actions, recorded states, success, seeds) from the original demo file."""
    meta = json.loads(Path(str(h5_path)[:-2] + "json").read_text())
    episodes = []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))[:num]
        for key, info in zip(keys, meta["episodes"]):
            assert info["episode_id"] == int(key.split("_")[-1])
            g = f[key]
            states = {kind: {name: g["env_states"][kind][name][:] for name in g["env_states"][kind]}
                      for kind in g["env_states"]}
            episodes.append({"key": key, "seed": info["episode_seed"], "reset_kwargs": info["reset_kwargs"],
                             "control_mode": info["control_mode"], "actions": g["actions"][:],
                             "success": g["success"][:], "env_states": states})
    return meta, episodes


def _first_state(states):
    return {kind: {name: arr[0] for name, arr in group.items()} for kind, group in states.items()}


def replay_episode(env, ep):
    """Replay one demonstration and record observations and simulator ground truth at every step."""
    from mani_skill.utils import common

    u = env.unwrapped
    env.reset(**ep["reset_kwargs"])
    u.set_state_dict(common.to_tensor(common.batch(_first_state(ep["env_states"]))))
    obs = u.get_obs()

    rec = {k: [] for k in ("rgb", "depth", "qpos", "qvel", "tcp_pose", "is_grasped", "goal_pos", "cube_pose",
                           "intrinsic_cv", "extrinsic_cv")}

    def record(obs):
        cam = obs["sensor_data"][CAMERA]
        par = obs["sensor_param"][CAMERA]
        rec["rgb"].append(_np(cam["rgb"])[0])
        rec["depth"].append(_np(cam["depth"])[0, ..., 0])
        rec["qpos"].append(_np(obs["agent"]["qpos"])[0])
        rec["qvel"].append(_np(obs["agent"]["qvel"])[0])
        rec["tcp_pose"].append(_np(u.agent.tcp.pose.raw_pose)[0])      # simulator ground truth
        rec["is_grasped"].append(bool(_np(obs["extra"]["is_grasped"])[0]))
        rec["goal_pos"].append(_np(obs["extra"]["goal_pos"])[0])
        rec["cube_pose"].append(_np(u.cube.pose.raw_pose)[0])         # simulator ground truth
        rec["intrinsic_cv"].append(_np(par["intrinsic_cv"])[0])
        rec["extrinsic_cv"].append(_np(par["extrinsic_cv"])[0])

    record(obs)
    success = []
    for action in ep["actions"]:
        obs, _, _, _, info = env.step(action)
        record(obs)
        success.append(bool(_np(info["success"]).reshape(-1)[0]))

    out = {k: np.stack(v) if k != "is_grasped" else np.array(v) for k, v in rec.items()}
    out.update(actions=ep["actions"].astype(np.float32), success=np.array(success),
               success_recorded=ep["success"].astype(bool),
               cube_pose_recorded=ep["env_states"]["actors"]["cube"][:, :7],
               # Panda articulation state: root pose (13) | qpos (9) | qvel (9)
               qpos_recorded=ep["env_states"]["articulations"]["panda"][:, 13:22],
               key=ep["key"], seed=ep["seed"], control_mode=ep["control_mode"])
    return out


def project(points, intrinsic, extrinsic):
    """World points (N, 3) -> pixel (u, v) and camera-frame depth (m), OpenCV convention."""
    cam = (extrinsic[:, :3] @ points.T + extrinsic[:, 3:4]).T
    uvw = (intrinsic @ cam.T).T
    return uvw[:, :2] / uvw[:, 2:3], cam[:, 2]


def validate(episodes, expected=5, cube_tol=0.005, depth_tol=0.03):
    """Return a list of (check, passed, detail). Every check uses measured data only."""
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    add("1. five demonstrations processed", len(episodes) == expected, f"{len(episodes)} episodes")

    rgb_ok = all(e["rgb"].dtype == np.uint8 and e["rgb"].ndim == 4 and e["rgb"].shape[-1] == 3
                 and e["rgb"].std() > 1 for e in episodes)
    add("2. RGB images extracted", rgb_ok, "uint8 (T+1, H, W, 3), non-constant")

    dmin = min(int(e["depth"].min()) for e in episodes)
    dmax = max(int(e["depth"].max()) for e in episodes)
    depth_ok = all(e["depth"].shape == e["rgb"].shape[:3] and (e["depth"] > 0).all() for e in episodes) \
        and dmin >= 50 and dmax <= 20000
    add("3. depth valid", depth_ok, f"int16 mm, all pixels > 0, range {dmin}-{dmax} mm")

    pos_ok = all(np.isfinite(e["cube_pose"]).all() and np.isfinite(e["tcp_pose"]).all()
                 and e["cube_pose"].shape == (len(e["rgb"]), 7) and e["tcp_pose"].shape == (len(e["rgb"]), 7)
                 for e in episodes)
    gt_err = max(float(np.abs(e["cube_pose"][:, :3] - e["cube_pose_recorded"][:, :3]).max()) for e in episodes)
    add("4. cube and end-effector poses", pos_ok and gt_err < cube_tol,
        f"(T+1, 7) each; replayed cube vs recorded state max error {1000 * gt_err:.2f} mm (tol {1000 * cube_tol:.0f} mm)")

    act_ok = all(e["actions"].ndim == 2 and np.isfinite(e["actions"]).all() for e in episodes)
    add("5. expert actions extracted", act_ok,
        f"original {episodes[0]['control_mode']} actions, shape (T, {episodes[0]['actions'].shape[1]})")

    # Alignment. Replaying action t from state t must reproduce the recorded state t+1 at the same index;
    # pairing with the neighbouring index must be clearly worse, so a one-step shift would be detected.
    lengths_ok = all(len(e["rgb"]) == len(e["actions"]) + 1 == len(e["success"]) + 1 for e in episodes)
    q_al = max(float(np.abs(e["qpos"] - e["qpos_recorded"]).max()) for e in episodes)
    q_sh = min(float(np.abs(e["qpos"][1:] - e["qpos_recorded"][:-1]).max()) for e in episodes)
    c_al = max(float(np.abs(e["cube_pose"][:, :3] - e["cube_pose_recorded"][:, :3]).max()) for e in episodes)
    c_sh = min(float(np.abs(e["cube_pose"][1:, :3] - e["cube_pose_recorded"][:-1, :3]).max()) for e in episodes)
    final_match = all(e["success"][-1] == e["success_recorded"][-1] for e in episodes)
    first = [(int(np.argmax(e["success"])), int(np.argmax(e["success_recorded"]))) for e in episodes]
    timing_ok = all(abs(a - b) <= 1 for a, b in first)
    step_agree = np.mean(np.concatenate([e["success"] == e["success_recorded"] for e in episodes]))
    align_ok = lengths_ok and q_al * 10 < q_sh and c_al * 10 < c_sh and final_match and timing_ok
    add("6. observation/action alignment", align_ok,
        f"obs T+1, actions T, success T; same-index error vs recorded states: joints {q_al:.1e} rad, cube "
        f"{1000 * c_al:.2f} mm; shifted by one step: joints >= {q_sh:.3f} rad, cube >= {1000 * c_sh:.1f} mm; "
        f"per-step success agrees with recording on {100 * step_agree:.1f}% of steps "
        f"(first-success step replay/recorded: {first})")

    succ = [bool(e["success"][-1]) for e in episodes]
    add("7. demonstrations successful", all(succ) and all(bool(e["success_recorded"][-1]) for e in episodes),
        f"{sum(succ)}/{len(succ)} successful at the last step (replay and recording)")

    # Supporting check for object-centric use: ground-truth cube projects onto the cube in the images.
    hits, depth_errs = 0, []
    for e in episodes:
        (u, v), z = [a[0] for a in project(e["cube_pose"][:1, :3], e["intrinsic_cv"][0], e["extrinsic_cv"][0])]
        ui, vi = int(round(u)), int(round(v))
        h, w = e["rgb"].shape[1:3]
        if 0 <= ui < w and 0 <= vi < h:
            r, g, b = e["rgb"][0, vi, ui].astype(int)
            hits += int(r > 120 and r > g + 50 and r > b + 50)
            depth_errs.append(abs(e["depth"][0, vi, ui] / 1000.0 - (z - 0.02)))  # visible face ~ half-size closer
    proj_ok = hits == len(episodes) and max(depth_errs, default=1) < depth_tol
    add("8. cube projects onto red pixels at t=0 (supporting)", proj_ok,
        f"{hits}/{len(episodes)} on red pixels; depth at projected pixel vs geometry max error "
        f"{100 * max(depth_errs, default=float('nan')):.1f} cm")
    return checks


def save_hdf5(path, episodes, source_path, source_sha256, env_info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = {"compression": "gzip", "compression_opts": 4, "chunks": True}
    with h5py.File(path, "w") as f:
        f.attrs.update({
            "description": "ManiSkill PickCube-v1 expert demos replayed with RGB-D and simulator ground truth",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_file": str(source_path), "source_sha256": source_sha256,
            "alignment": "obs/objects/robot arrays have T+1 entries (index t = state before action t, index T = final "
                         "state); actions[t] is applied at state t; success[t] is evaluated after action t",
            "units": "positions m, quaternions wxyz, depth int16 mm, joint angles rad, joint velocities rad/s",
            **{k: str(v) for k, v in env_info.items()},
        })
        for i, e in enumerate(episodes):
            g = f.create_group(f"episodes/ep_{i:03d}")
            g.attrs.update({"source_episode": e["key"], "episode_seed": int(e["seed"]), "T": len(e["actions"]),
                            "success": bool(e["success"][-1]), "control_mode": e["control_mode"]})
            g.create_dataset("obs/rgb", data=e["rgb"], **img)
            g.create_dataset("obs/depth", data=e["depth"], **img)
            g.create_dataset("robot/qpos", data=e["qpos"])
            g.create_dataset("robot/qvel", data=e["qvel"])
            g.create_dataset("robot/tcp_pose", data=e["tcp_pose"])
            g.create_dataset("robot/is_grasped", data=e["is_grasped"])
            g.create_dataset("objects/cube_pose", data=e["cube_pose"])
            g.create_dataset("objects/cube_pose_recorded", data=e["cube_pose_recorded"])
            g.create_dataset("objects/goal_pos", data=e["goal_pos"])
            g.create_dataset("camera/intrinsic_cv", data=e["intrinsic_cv"][0])
            g.create_dataset("camera/extrinsic_cv", data=e["extrinsic_cv"][0])
            g.create_dataset("actions", data=e["actions"])
            g.create_dataset("success", data=e["success"])
    return path


def shape_table(episodes):
    rows = []
    for e in episodes:
        rows.append({k: list(e[k].shape) for k in ("rgb", "depth", "qpos", "qvel", "tcp_pose", "cube_pose",
                                                      "goal_pos", "actions", "success")})
    return rows
