"""Offline failure diagnostics for PickCube-v1 rollouts.

`DiagnosticInfoWrapper` copies simulator state into the step *info* dict only; observations
are untouched, so the policy receives exactly the inputs it was trained on. Episode flags and
the mutually exclusive failure category are then computed from the per-step records.

Signals and their sources (ManiSkill 3.0.1, PickCube-v1, Panda):
- success        = is_obj_placed AND is_robot_static          (env.evaluate)
- is_obj_placed  = ||goal - cube|| <= goal_thresh (0.025 m)    (env.evaluate)
- is_robot_static= max |arm joint velocity| <= 0.2 rad/s        (env.evaluate)
- is_grasped     = both fingers in contact, force >= 0.5 N, contact angle <= 85 deg (agent.is_grasping)
- cube / goal / TCP positions and gripper opening: read from the simulator by this wrapper.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

# Per-step fields added to info (all scalars per env).
DIAG_KEYS = ["obj_x", "obj_y", "obj_z", "goal_x", "goal_y", "goal_z", "tcp_to_obj", "obj_to_goal", "gripper_width"]
# Per-step fields already produced by PickCube's evaluate().
ENV_KEYS = ["success", "is_obj_placed", "is_robot_static", "is_grasped"]
STEP_KEYS = ENV_KEYS + DIAG_KEYS


class DiagnosticInfoWrapper(gym.Wrapper):
    """Adds diag_<key> entries to info at reset and every step. Does not change observations."""

    def _diag(self):
        env = self.env.unwrapped
        with torch.no_grad():
            obj = env.cube.pose.p
            goal = env.goal_site.pose.p
            tcp = env.agent.tcp_pose.p
            fingers = env.agent.robot.get_qpos()[..., -2:]
            values = {
                "obj_x": obj[..., 0], "obj_y": obj[..., 1], "obj_z": obj[..., 2],
                "goal_x": goal[..., 0], "goal_y": goal[..., 1], "goal_z": goal[..., 2],
                "tcp_to_obj": torch.linalg.norm(obj - tcp, dim=-1),
                "obj_to_goal": torch.linalg.norm(goal - obj, dim=-1),
                "gripper_width": fingers.sum(dim=-1),
            }
        return {f"diag_{k}": v.clone() for k, v in values.items()}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.update(self._diag())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info.update(self._diag())
        return obs, reward, terminated, truncated, info


@dataclass(frozen=True)
class Thresholds:
    """Diagnostic thresholds (metres / steps). Fixed before looking at results; reported with them."""
    approach_dist: float = 0.03      # TCP within 3 cm of the cube centre (cube half-size is 2 cm)
    lift_height: float = 0.01        # cube centre >= 1 cm above its resting height
    rest_height: float = 0.02        # PickCube cube half-size: resting centre height
    stable_lift_steps: int = 5       # grasped AND lifted for >= 5 consecutive steps
    release_steps: int = 3           # not grasped for >= 3 consecutive steps counts as losing the grasp


CATEGORIES = [
    ("success", "Success at end"),
    ("reached_target_lost_success", "Reached target, success not held at end"),
    ("grasp_lost_before_target", "Grasped, lost grasp before reaching target"),
    ("grasped_not_delivered", "Grasped and held, never reached target"),
    ("approached_no_grasp", "Reached cube, never grasped"),
    ("no_approach", "Never reached the cube"),
    ("other", "Other / ambiguous"),
]
CATEGORY_LABEL = dict(CATEGORIES)
CATEGORY_SHORT = {
    "success": "SUCCESS", "reached_target_lost_success": "REACHED TARGET, LOST",
    "grasp_lost_before_target": "LOST GRASP", "grasped_not_delivered": "HELD, MISSED TARGET",
    "approached_no_grasp": "NO GRASP", "no_approach": "NO APPROACH", "other": "OTHER",
}


def _first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) + 1 if len(idx) else None  # 1-based step at which the condition first held


def _longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def _goal_errors(s, t, prefix):
    """Horizontal distance and signed height error (cube minus goal; negative = cube below goal) at step index t."""
    return {
        f"{prefix}_xy_err": float(np.hypot(s["obj_x"][t] - s["goal_x"][t], s["obj_y"][t] - s["goal_y"][t])),
        f"{prefix}_z_err": float(s["obj_z"][t] - s["goal_z"][t]),
    }


def episode_flags(steps, th=Thresholds()):
    """Event flags for one episode. `steps` maps STEP_KEYS to arrays over steps 1..T (post-action state)."""
    s = {k: np.asarray(v) for k, v in steps.items()}
    grasped = s["is_grasped"].astype(bool)
    placed = s["is_obj_placed"].astype(bool)
    success = s["success"].astype(bool)
    lifted = s["obj_z"] >= th.rest_height + th.lift_height
    grasp_lift = grasped & lifted

    stable_lift = _longest_run(grasp_lift) >= th.stable_lift_steps
    t_lift = _first(grasp_lift)
    # Lost grasp after lifting: >= release_steps consecutive ungrasped steps after the first lift.
    lost_grasp_after_lift = False
    if t_lift is not None:
        lost_grasp_after_lift = _longest_run(~grasped[t_lift:]) >= th.release_steps

    t_reach = _first(placed)
    flags = {
        "T": int(len(grasped)),
        "approached": bool(s["tcp_to_obj"].min() <= th.approach_dist),
        "min_tcp_to_obj": float(s["tcp_to_obj"].min()),
        "ever_grasped": bool(grasped.any()),
        "stable_lift": bool(stable_lift),
        "max_lift": float(s["obj_z"].max() - th.rest_height),
        "reached_target": bool(placed.any()),
        "min_obj_to_goal": float(s["obj_to_goal"].min()),
        "success_once": bool(success.any()),
        "final_success": bool(success[-1]),
        "grasped_at_end": bool(grasped[-1]),
        "placed_at_end": bool(placed[-1]),
        "static_at_end": bool(s["is_robot_static"][-1]),
        "final_obj_to_goal": float(s["obj_to_goal"][-1]),
        "lost_grasp_after_lift": bool(lost_grasp_after_lift),
        "reached_without_grasp": bool(placed.any() and not grasped[: t_reach].any()) if t_reach else False,
        "t_grasp": _first(grasped), "t_lift": t_lift, "t_reach": t_reach, "t_success": _first(success),
        "steps_placed": int(placed.sum()), "steps_success": int(success.sum()),
        # Where the cube missed the goal: horizontal vs vertical error, at the closest approach and at the end.
        **_goal_errors(s, int(np.argmin(s["obj_to_goal"])), "closest"),
        **_goal_errors(s, -1, "final"),
        "goal_z": float(s["goal_z"][0]),
        "goal_height_above_rest": float(s["goal_z"][0] - th.rest_height),
        "cube_goal_xy_dist0": float(np.hypot(s["goal_x"][0] - s["obj_x"][0], s["goal_y"][0] - s["obj_y"][0])),
    }
    flags["category"] = classify(flags)
    flags["lost_success_reason"] = lost_success_reason(flags) if flags["category"] == "reached_target_lost_success" else None
    return flags


def classify(f):
    """Mutually exclusive category, checked in priority order (later stages win)."""
    if f["final_success"]:
        return "success"
    if f["reached_target"]:
        # Reaching the goal without ever grasping (e.g. cube spawned or pushed into a low goal) is ambiguous.
        return "other" if f["reached_without_grasp"] else "reached_target_lost_success"
    if f["ever_grasped"]:
        return "grasped_not_delivered" if f["grasped_at_end"] else "grasp_lost_before_target"
    if f["approached"]:
        return "approached_no_grasp"
    return "no_approach"


def lost_success_reason(f):
    """Why an episode that reached the target was not successful at the end."""
    if f["placed_at_end"]:
        return "in_target_robot_moving"          # cube still within 2.5 cm, arm not static
    if f["grasped_at_end"]:
        return "carried_out_of_target"           # still holding the cube, but outside the goal
    return "released_or_dropped_after_target"    # cube left the goal without being held
