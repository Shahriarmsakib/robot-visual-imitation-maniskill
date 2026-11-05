"""Record labelled rollout GIFs and a success/failure montage from a checkpoint (Fig. 5).

Usage:
    python scripts/export_rollouts.py CHECKPOINT [--num-episodes 8] [--out results/media/<run>]

Episodes use the run's evaluation seed, so they are the first scenes of the logged evaluations
(on the same simulator backend). Outcomes come from the environment's success metrics.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import gymnasium as gym
    import numpy as np
    from mani_skill.utils import common

    from src.evaluation.envs import make_eval_envs
    from src.evaluation.evaluate import evaluate
    from src.evaluation.policy import load_policy
    from src.utils import select_device, select_sim_backend
    from src.visualization.rollouts import label_frame, montage, outcome_text, write_gif

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None, help="scene seed (default: the run's eval_seed)")
    parser.add_argument("--out", type=Path, default=None, help="default: results/media/<run>")
    parser.add_argument("--size", type=int, default=256, help="GIF frame size in pixels")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2, help="keep every Nth simulator step in GIFs")
    parser.add_argument("--montage-rows", type=int, default=4)
    parser.add_argument("--max-gifs-per-outcome", type=int, default=2,
                        help="GIFs kept per outcome (success/failure); every episode still appears in rollouts.json")
    args = parser.parse_args()

    device = select_device(args.device)
    sim_backend = select_sim_backend(args.sim_backend)
    agent, cfg, norm_stats, step = load_policy(args.checkpoint, device)
    seed = cfg.eval_seed if args.seed is None else args.seed
    out = args.out or Path("results/media") / cfg.exp_name

    envs = make_eval_envs(cfg.env_id, 1, sim_backend, cfg.control_mode, cfg.max_episode_steps,
                          include_depth=cfg.include_depth)
    episodes = []

    def grab(envs, ts):
        if isinstance(envs, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
            frame = np.asarray(common.to_numpy(envs.call("render")[0]))
        else:  # ManiSkill GPU vector env
            frame = np.asarray(common.to_numpy(envs.render()))
        frame = frame.reshape(-1, *frame.shape[-3:])[0]
        if ts == 0:
            episodes.append([])
        if ts % args.stride == 0:
            episodes[-1].append((ts, frame))

    result = evaluate(agent, envs, args.num_episodes, cfg.num_queries, cfg.temporal_agg, cfg.max_episode_steps,
                      device, sim_backend, norm_stats=norm_stats, seed=seed, frame_hook=grab)
    envs.close()

    at_end = result["per_episode"]["success_at_end"] > 0.5
    once = result["per_episode"]["success_once"] > 0.5
    labelled = []
    gifs_written = {True: 0, False: 0}
    for i, raw in enumerate(episodes):
        text = outcome_text(i, bool(at_end[i]), bool(once[i]))
        frames = [label_frame(f, args.size, t, text, bool(at_end[i])) for t, f in raw]
        name = None
        if gifs_written[bool(at_end[i])] < args.max_gifs_per_outcome:
            name = f"ep{i:02d}_{'success' if at_end[i] else 'failure'}.gif"
            write_gif(frames, out / name, args.fps)
            gifs_written[bool(at_end[i])] += 1
        labelled.append({"episode": i, "success_at_end": bool(at_end[i]), "success_once": bool(once[i]),
                         "gif": name, "frames": frames})

    # Montage: successes first, then failures, so both outcomes appear when available.
    half = args.montage_rows // 2
    succ = [e for e in labelled if e["success_at_end"]]
    fail = [e for e in labelled if not e["success_at_end"]]
    pick = succ[:half] + fail[:args.montage_rows - min(half, len(succ))]
    pick += [e for e in labelled if e not in pick][:args.montage_rows - len(pick)]
    title = (f"{cfg.exp_name} @ step {step}: {int(at_end.sum())}/{len(at_end)} success at end "
             f"({sim_backend}, scene seed {seed})")
    montage(sorted(pick, key=lambda e: e["episode"]), out / "rollout_montage.png", title=title)

    manifest = {"checkpoint": str(args.checkpoint), "checkpoint_step": step, "run": cfg.exp_name,
                "sim_backend": sim_backend, "scene_seed": seed, "size": args.size, "fps": args.fps,
                "stride": args.stride, "summary": result["summary"],
                "episodes": [{k: v for k, v in e.items() if k != "frames"} for e in labelled]}
    (out / "rollouts.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(title)
    print(f"wrote {sum(gifs_written.values())} GIFs ({len(labelled)} episodes) + rollout_montage.png + rollouts.json to {out}/")


if __name__ == "__main__":
    main()
