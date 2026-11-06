import numpy as np

from src.evaluation.diagnostics import STEP_KEYS, classify, episode_flags
from src.evaluation.failure_summary import mcnemar_exact, summarize

T = 20


def _episode(grasp=(), lift=(), placed=(), success=(), static=(), tcp_to_obj=0.01, goal_z=0.15):
    s = {k: np.zeros(T) for k in STEP_KEYS}
    s["tcp_to_obj"][:] = tcp_to_obj
    s["obj_z"][:] = 0.02
    s["goal_z"][:] = goal_z
    s["obj_to_goal"][:] = 0.2
    for key, steps in (("is_grasped", grasp), ("is_obj_placed", placed), ("success", success),
                       ("is_robot_static", static)):
        for t in steps:
            s[key][t] = 1.0
    for t in lift:
        s["obj_z"][t] = 0.06
    for t in placed:
        s["obj_to_goal"][t] = 0.01
    return s


def test_success():
    f = episode_flags(_episode(grasp=range(5, 20), lift=range(8, 20), placed=range(15, 20),
                               success=range(15, 20), static=range(15, 20)))
    assert f["category"] == "success" and f["stable_lift"] and f["t_grasp"] == 6 and f["t_reach"] == 16


def test_reached_target_then_carried_out():
    f = episode_flags(_episode(grasp=range(5, 20), lift=range(8, 20), placed=range(12, 15), success=[13]))
    assert f["category"] == "reached_target_lost_success"
    assert f["lost_success_reason"] == "carried_out_of_target"
    assert f["success_once"] and not f["final_success"]


def test_held_never_reached_vs_dropped():
    held = episode_flags(_episode(grasp=range(5, 20), lift=range(8, 20)))
    assert held["category"] == "grasped_not_delivered" and held["grasped_at_end"]
    dropped = episode_flags(_episode(grasp=range(5, 12), lift=range(8, 12)))
    assert dropped["category"] == "grasp_lost_before_target" and dropped["lost_grasp_after_lift"]


def test_approach_categories():
    assert episode_flags(_episode(tcp_to_obj=0.02))["category"] == "approached_no_grasp"
    assert episode_flags(_episode(tcp_to_obj=0.10))["category"] == "no_approach"


def test_target_without_grasp_is_ambiguous():
    f = episode_flags(_episode(placed=range(0, 3)))
    assert f["reached_without_grasp"] and f["category"] == "other"


def test_short_lift_is_not_stable():
    f = episode_flags(_episode(grasp=range(5, 20), lift=range(8, 11)))  # 3 lifted steps < 5
    assert not f["stable_lift"]


def test_summary_and_mcnemar():
    rows = [episode_flags(_episode(grasp=range(5, 20), lift=range(8, 20))) for _ in range(3)]
    rows.append(episode_flags(_episode(tcp_to_obj=0.10)))
    s = summarize(rows)
    assert s["stages"]["ever_grasped"]["count"] == 3 and s["categories"]["no_approach"]["count"] == 1
    assert s["held_missed_target"]["n"] == 3
    m = mcnemar_exact([True] * 10 + [False] * 10, [True] * 2 + [False] * 18)
    assert m["only_first"] == 8 and m["only_second"] == 0 and m["p_value"] < 0.01
    assert classify({"final_success": True}) == "success"
