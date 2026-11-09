"""Phase A failure analysis: per-episode simulator diagnostics, failure categories and example videos.

Usage:
    python scripts/failure_analysis.py CHECKPOINT --label step30000 [--videos] [--num-episodes 200]

Runs the standard evaluation loop (same scene seed and parallel-env layout as training-time
evaluation, with fully reproducible scenes) with an info-only diagnostic wrapper; the policy's inputs are unchanged. Writes, under
--out (default results/phase_a_failure_analysis):
    metrics/episodes_<label>.csv     one row per episode: event flags + category
    metrics/timeseries_<label>.npz   per-step signals, shape (episodes, steps)
    metrics/run_<label>.json         checkpoint, seeds, thresholds, env settings, consistency checks
    videos/*.mp4, metrics/examples_<label>.json   (with --videos) first episode of each category
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Video name -> how its episode is chosen: the lowest episode ID matching the rule (no hand-picking).
VIDEO_RULES = {
    "successful_rollout": lambda f: f["category"] == "success",
    "failed_grasp": lambda f: f["category"] == "approached_no_grasp",
    "dropped_cube": lambda f: f["category"] == "grasp_lost_before_target" and f["lost_grasp_after_lift"],
    "lost_success": lambda f: f["category"] == "reached_target_lost_success",
}


def main():
    import dataclasses

    import gymnasium as gym
    import numpy as np
    from mani_skill.utils import common

    from src.evaluation.diagnostics import (CATEGORY_LABEL, CATEGORY_SHORT, DIAG_KEYS, STEP_KEYS, DiagnosticInfoWrapper,
                                            Thresholds, episode_flags)
    from src.evaluation.envs import make_eval_envs
    from src.evaluation.evaluate import evaluate
    from src.evaluation.policy import load_policy
    from src.utils import select_device, sha256_file
    from src.visualization.failure_media import annotate_frame, write_mp4

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--label", required=True, help="name for this checkpoint in outputs, e.g. step30000")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="scene seed (default: the run's eval_seed)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sim-backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"],
                        help="physx_cuda needs an NVIDIA GPU (e.g. Kaggle); macOS only has physx_cpu")
    parser.add_argument("--out", type=Path, default=Path("results/phase_a_failure_analysis"))
    parser.add_argument("--videos", action="store_true", help="render and save one MP4 per video category")
    parser.add_argument("--video-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    sim_backend = args.sim_backend  # recorded in the outputs
    device = select_device(args.device)
    agent, cfg, norm_stats, ckpt_step = load_policy(args.checkpoint, device)
    seed = cfg.eval_seed if args.seed is None else args.seed
    th = Thresholds()
    T = cfg.max_episode_steps
    n_envs = args.num_envs
    envs = make_eval_envs(cfg.env_id, n_envs, sim_backend, cfg.control_mode, T, include_depth=cfg.include_depth,
                          extra_wrappers=[DiagnosticInfoWrapper])

    state = {"batch": 0, "buf": {k: np.zeros((T, n_envs)) for k in STEP_KEYS}, "frames": [[] for _ in range(n_envs)]}
    episodes, series = [], {k: [] for k in STEP_KEYS}
    examples = {}  # video name -> {"episode": id, "frames": [...]}

    def need_frames():
        return args.videos and len(examples) < len(VIDEO_RULES)

    def frame_hook(envs, ts):
        if ts == 0:
            state["frames"] = [[] for _ in range(n_envs)]
        if need_frames():
            if sim_backend == "physx_cpu":
                frames = [np.asarray(common.to_numpy(fr)) for fr in envs.call("render")]
                frames = [f.reshape(-1, *f.shape[-3:])[0] for f in frames]
            else:  # one batched GPU env: render() returns (num_envs, H, W, 3)
                frames = np.asarray(common.to_numpy(envs.render()))
            for i in range(n_envs):
                state["frames"][i].append(frames[i])

    def info_hook(info, step, truncated):
        src = info["final_info"] if truncated.any() else info  # last step: the finished episode's own info
        for k in STEP_KEYS:
            key = k if k in ("success", "is_obj_placed", "is_robot_static", "is_grasped") else f"diag_{k}"
            state["buf"][k][step - 1] = np.asarray(common.to_numpy(src[key]), dtype=np.float64).reshape(-1)
        if not truncated.any():
            return
        assert step == T, f"episode ended at step {step}, expected {T}"
        for i in range(n_envs):
            ep_id = state["batch"] * n_envs + i
            steps = {k: state["buf"][k][:, i].copy() for k in STEP_KEYS}
            flags = {"episode": ep_id, "batch": state["batch"], "env_index": i, **episode_flags(steps, th)}
            episodes.append(flags)
            for k in STEP_KEYS:
                series[k].append(steps[k])
            for name, rule in VIDEO_RULES.items():
                if name not in examples and rule(flags) and len(state["frames"][i]) == T:
                    examples[name] = {"episode": ep_id, "flags": flags, "frames": state["frames"][i], "steps": steps}
        state["batch"] += 1
        state["buf"] = {k: np.zeros((T, n_envs)) for k in STEP_KEYS}

    t0 = time.time()
    result = evaluate(agent, envs, args.num_episodes, cfg.num_queries, cfg.temporal_agg, T, device, sim_backend,
                      norm_stats=norm_stats, seed=seed, frame_hook=frame_hook, info_hook=info_hook)
    envs.close()
    elapsed = time.time() - t0

    # Consistency: our per-step reconstruction must agree with ManiSkill's own episode metrics.
    env_at_end = result["per_episode"]["success_at_end"] > 0.5
    env_once = result["per_episode"]["success_once"] > 0.5
    ours_at_end = np.array([e["final_success"] for e in episodes])
    ours_once = np.array([e["success_once"] for e in episodes])
    checks = {
        "n_episodes": len(episodes),
        "final_success_matches_env_metric": bool((env_at_end == ours_at_end).all()),
        "success_once_matches_env_metric": bool((env_once == ours_once).all()),
    }
    if not (checks["final_success_matches_env_metric"] and checks["success_once_matches_env_metric"]):
        sys.exit(f"Inconsistent with environment metrics: {checks}")

    out = args.out
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    with open(out / "metrics" / f"episodes_{args.label}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["checkpoint", *episodes[0].keys()])
        writer.writeheader()
        for e in episodes:
            writer.writerow({"checkpoint": args.label, **e})
    np.savez_compressed(out / "metrics" / f"timeseries_{args.label}.npz",
                        episode=np.array([e["episode"] for e in episodes]),
                        **{k: np.stack(v) for k, v in series.items()})
    run_info = {
        "label": args.label, "checkpoint": str(args.checkpoint), "checkpoint_step": ckpt_step,
        "checkpoint_sha256": sha256_file(args.checkpoint), "run": cfg.exp_name,
        "env_id": cfg.env_id, "control_mode": cfg.control_mode, "obs": "rgb (+ standard 29-d state)",
        "max_episode_steps": T, "num_queries": cfg.num_queries, "temporal_agg": cfg.temporal_agg,
        "sim_backend": sim_backend, "device": str(device), "scene_seed": seed, "num_envs": n_envs,
        "episode_id_rule": "episode = batch * num_envs + env_index; env i of the first batch is seeded scene_seed + i",
        "thresholds": dataclasses.asdict(th), "diag_keys": DIAG_KEYS, "checks": checks,
        "seconds": round(elapsed, 1),
    }
    (out / "metrics" / f"run_{args.label}.json").write_text(json.dumps(run_info, indent=2) + "\n")

    if args.videos:
        (out / "videos").mkdir(parents=True, exist_ok=True)
        chosen = {}
        for name in VIDEO_RULES:
            ex = examples.get(name)
            if ex is None:
                chosen[name] = None
                print(f"[videos] no episode matched '{name}': not generated")
                continue
            f = ex["flags"]
            header = f"ep {f['episode']}: {CATEGORY_SHORT[f['category']]}"
            frames = [annotate_frame(img, args.video_size, t, header, f["category"],
                                     {k: ex["steps"][k][t - 1] for k in STEP_KEYS} if t > 0 else None)
                      for t, img in enumerate(ex["frames"])]
            write_mp4(frames, out / "videos" / f"{name}.mp4", args.fps)
            chosen[name] = {"episode": f["episode"], "category": f["category"],
                            "rule": "lowest episode ID matching the category rule", "file": f"videos/{name}.mp4"}
            np.save(out / "metrics" / f"_frames_{args.label}_{name}.npy",
                    np.stack([np.asarray(fr) for fr in frames]))  # for the montage figure; not committed
        (out / "metrics" / f"examples_{args.label}.json").write_text(json.dumps(chosen, indent=2) + "\n")

    cats = {}
    for e in episodes:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    n = len(episodes)
    print(f"{args.label} ({cfg.exp_name} @ step {ckpt_step}, {sim_backend}, seed {seed}): {n} episodes in {elapsed:.0f}s")
    print(f"  final success {int(ours_at_end.sum())}/{n}, success once {int(ours_once.sum())}/{n}, "
          f"reached target {sum(e['reached_target'] for e in episodes)}/{n}, ever grasped "
          f"{sum(e['ever_grasped'] for e in episodes)}/{n}, stable lift {sum(e['stable_lift'] for e in episodes)}/{n}")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>4}  {CATEGORY_LABEL[k]}")
    print(f"  checks: {checks}")


if __name__ == "__main__":
    main()
