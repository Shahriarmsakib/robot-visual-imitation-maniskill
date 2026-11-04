"""Evaluate a saved checkpoint on seeded PickCube-v1 scenes (same protocol as during training).

Usage:
    python scripts/evaluate.py CHECKPOINT [--num-episodes 100] [--num-envs 10] [--sim-backend auto]

Writes <run>/evaluations/<checkpoint>_<backend>_seed<seed>.json and prints success counts.
Useful for re-testing a downloaded Kaggle checkpoint locally, or on another backend.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    # Heavy imports stay inside main(): CPU eval workers re-import this file (see docs/m1_training.md).
    from src.evaluation.envs import make_eval_envs
    from src.evaluation.evaluate import evaluate
    from src.evaluation.policy import load_policy
    from src.utils import select_device, select_sim_backend

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None, help="evaluation scene seed (default: the run's eval_seed)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    sim_backend = select_sim_backend(args.sim_backend)
    agent, cfg, norm_stats, step = load_policy(args.checkpoint, device)
    seed = cfg.eval_seed if args.seed is None else args.seed
    envs = make_eval_envs(cfg.env_id, args.num_envs, sim_backend, cfg.control_mode, cfg.max_episode_steps,
                          include_depth=cfg.include_depth)
    t0 = time.time()
    result = evaluate(agent, envs, args.num_episodes, cfg.num_queries, cfg.temporal_agg, cfg.max_episode_steps,
                      device, sim_backend, norm_stats=norm_stats, seed=seed)
    envs.close()
    s = result["summary"]
    record = {"checkpoint": str(args.checkpoint), "checkpoint_step": step, "run": cfg.exp_name,
              "sim_backend": sim_backend, "device": str(device), "eval_seed": seed,
              "seconds": round(time.time() - t0, 1), **s}
    out = args.out or args.checkpoint.parent.parent / "evaluations" / \
        f"{args.checkpoint.stem}_{sim_backend}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{cfg.exp_name} @ step {step} on {sim_backend}: success_once {s['success_once']['count']}/{s['n_trials']}, "
          f"success_at_end {s['success_at_end']['count']}/{s['n_trials']}  -> {out}")


if __name__ == "__main__":
    main()
