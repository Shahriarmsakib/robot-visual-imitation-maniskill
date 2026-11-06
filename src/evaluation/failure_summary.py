"""Aggregate per-episode failure-analysis rows into summary statistics (no plotting)."""
from collections import Counter
from math import comb

import numpy as np

from src.evaluation.diagnostics import CATEGORIES
from src.utils import wilson_interval

# (field, label, definition) in pipeline order
STAGES = [
    ("approached", "Reached cube", "TCP came within 3 cm of the cube centre at some step"),
    ("ever_grasped", "Grasped", "ManiSkill is_grasped was true at some step"),
    ("stable_lift", "Stable lift", "grasped and cube >= 1 cm above rest for >= 5 consecutive steps"),
    ("reached_target", "Reached target", "cube within 2.5 cm of the goal (is_obj_placed) at some step"),
    ("success_once", "Success once", "success (in target AND arm static) at some step"),
    ("final_success", "Success at end", "success at the last step (step 100)"),
]
GOAL_HEIGHT_BINS = [0.0, 0.075, 0.15, 0.225, 0.31]


def _bool(row, key):
    v = row[key]
    return v if isinstance(v, bool) else str(v) == "True"


def _float(row, key):
    v = row[key]
    return None if v in (None, "", "None") else float(v)


def _rate(count, n):
    lo, hi = wilson_interval(count, n)
    return {"count": count, "n": n, "rate": count / n if n else None, "ci95": [lo, hi]}


def summarize(rows, timeseries=None):
    n = len(rows)
    out = {"n_episodes": n}
    out["stages"] = {k: {"label": lbl, "definition": d, **_rate(sum(_bool(r, k) for r in rows), n)}
                     for k, lbl, d in STAGES}
    cats = Counter(r["category"] for r in rows)
    out["categories"] = {k: {"label": lbl, **_rate(cats.get(k, 0), n)} for k, lbl in CATEGORIES}
    lost = [r for r in rows if r["category"] == "reached_target_lost_success"]
    out["lost_success_reasons"] = dict(Counter(r["lost_success_reason"] for r in lost))
    dropped = [r for r in rows if r["category"] == "grasp_lost_before_target"]
    out["lost_grasp_after_stable_lift"] = sum(_bool(r, "lost_grasp_after_lift") for r in dropped)

    # Conditional rates along the pipeline.
    grasped = [r for r in rows if _bool(r, "ever_grasped")]
    reached = [r for r in rows if _bool(r, "reached_target")]
    out["conditional"] = {
        "reached_target_given_grasped": _rate(sum(_bool(r, "reached_target") for r in grasped), len(grasped)),
        "final_success_given_reached_target": _rate(sum(_bool(r, "final_success") for r in reached), len(reached)),
    }

    held = [r for r in rows if r["category"] == "grasped_not_delivered"]
    if held:
        def med(key, scale=100.0):
            v = np.array([_float(r, key) for r in held]) * scale
            return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}
        z = np.array([_float(r, "closest_z_err") for r in held])
        xy = np.array([_float(r, "closest_xy_err") for r in held])
        mind = np.array([_float(r, "min_obj_to_goal") for r in held])
        out["held_missed_target"] = {
            "n": len(held),
            "units": "cm",
            "min_obj_to_goal": med("min_obj_to_goal"),
            "closest_xy_err": med("closest_xy_err"),
            "closest_z_err_signed": med("closest_z_err"),
            "final_xy_err": med("final_xy_err"),
            "final_z_err_signed": med("final_z_err"),
            "cube_below_goal_at_closest": int((z < 0).sum()),
            "z_err_larger_than_xy_err_at_closest": int((np.abs(z) > xy).sum()),
            "closest_within_5cm": int((mind < 0.05).sum()),
            "arm_static_at_end": sum(_bool(r, "static_at_end") for r in held),
        }
        if timeseries is not None:
            idx = [i for i, r in enumerate(rows) if r["category"] == "grasped_not_delivered"]
            d = timeseries["obj_to_goal"][idx]
            out["held_missed_target"]["obj_to_goal_median_cm_at_step"] = {
                str(s): float(100 * np.median(d[:, s - 1])) for s in (25, 50, 75, 100)}
            out["held_missed_target"]["change_last_20_steps_cm_median"] = float(100 * np.median(d[:, -1] - d[:, -21]))

    out["by_goal_height"] = []
    for lo, hi in zip(GOAL_HEIGHT_BINS[:-1], GOAL_HEIGHT_BINS[1:]):
        sub = [r for r in rows if lo <= _float(r, "goal_height_above_rest") < hi]
        out["by_goal_height"].append({
            "goal_height_cm": [100 * lo, 100 * hi],
            "final_success": _rate(sum(_bool(r, "final_success") for r in sub), len(sub)),
            "reached_target": _rate(sum(_bool(r, "reached_target") for r in sub), len(sub)),
            "stable_lift": _rate(sum(_bool(r, "stable_lift") for r in sub), len(sub)),
        })

    def median_step(key):
        v = [int(float(r[key])) for r in rows if r[key] not in (None, "", "None")]
        return {"median": float(np.median(v)), "n": len(v)} if v else None
    out["timing_steps"] = {k: median_step(k) for k in ("t_grasp", "t_lift", "t_reach", "t_success")}
    return out


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test for paired booleans (same scenes, two policies)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n)
    return {"both": int((a & b).sum()), "only_first": only_a, "only_second": only_b,
            "neither": int((~a & ~b).sum()), "p_value": p}
