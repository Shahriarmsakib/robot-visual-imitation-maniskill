"""Policy evaluation with temporal ensembling.

Derived from `evaluate` in ManiSkill v3.0.1 examples/baselines/act/act/evaluate.py
(Apache-2.0). Changes: evaluation episodes are seeded so every checkpoint is tested on
the same initial scenes; per-episode metrics are flattened across parallel envs (one
entry per trial, not per batch); gymnasium >= 1.0 info dicts are supported; results
include success counts and 95% Wilson intervals.
"""
from collections import defaultdict

import numpy as np
import torch

from mani_skill.utils import common
from src.utils import wilson_interval


def _episode_metrics(info):
    """Yield {metric: per-env array} for episodes that just ended, across gymnasium/ManiSkill info formats."""
    final_info = info["final_info"]
    if isinstance(final_info, dict):  # ManiSkill GPU vector env, or gymnasium >= 1.0 CPU vector env
        mask = info.get("_final_info")
        mask = None if mask is None else np.asarray(common.to_numpy(mask), dtype=bool)
        out = {}
        for k, v in final_info["episode"].items():
            if k.startswith("_"):  # gymnasium >= 1.0 adds "_<key>" validity masks
                continue
            v = np.asarray(common.to_numpy(v), dtype=np.float64).reshape(-1)
            out[k] = v[mask] if mask is not None and mask.shape == v.shape else v
        return out
    out = defaultdict(list)  # gymnasium 0.29 CPU vector env: list of per-env dicts
    for fi in final_info:
        if fi is not None:
            for k, v in fi["episode"].items():
                out[k].append(float(common.to_numpy(v)))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


@torch.no_grad()
def evaluate(agent, envs, n_episodes, num_queries, temporal_agg, max_timesteps, device, sim_backend,
             norm_stats=None, seed=None, frame_hook=None):
    """Roll out `agent` until at least `n_episodes` trials finish.

    Returns {"per_episode": {metric: array(n_trials)}, "summary": {...}}; n_trials is a
    multiple of the number of parallel envs. `frame_hook(envs, ts)`, if given, is called
    before every env step (e.g. to render frames for videos).
    """
    delta_control = not norm_stats
    if not delta_control:
        state_mean, state_std = norm_stats["state_mean"].to(device), norm_stats["state_std"].to(device)
        action_mean, action_std = norm_stats["action_mean"].to(device), norm_stats["action_std"].to(device)

    action_dim = envs.single_action_space.shape[-1]
    num_envs = envs.num_envs
    query_frequency = 1 if temporal_agg else num_queries

    def fresh_action_table():
        return torch.zeros([num_envs, max_timesteps, max_timesteps + num_queries, action_dim], device=device)

    agent.eval()
    metrics = defaultdict(list)
    obs, _ = envs.reset(seed=seed)
    ts, eps_count = 0, 0
    all_time_actions = fresh_action_table() if temporal_agg else None
    actions_to_take = None
    exp_weights = None
    while eps_count < n_episodes:
        obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
        if not delta_control:
            obs["state"] = (obs["state"] - state_mean) / state_std

        if ts % query_frequency == 0:
            action_seq = agent.get_action(obs)  # (num_envs, num_queries, act_dim)

        if temporal_agg:
            all_time_actions[:, ts, ts : ts + num_queries] = action_seq
            populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
            populated[max(0, ts + 1 - num_queries) : ts + 1] = True
            candidates = all_time_actions[:, populated, ts]  # (num_envs, num_populated, act_dim)
            if ts < num_queries:
                w = torch.exp(-0.01 * torch.arange(candidates.shape[1], device=device))
                exp_weights = (w / w.sum()).tile(num_envs, 1).unsqueeze(-1)
            raw_action = (candidates * exp_weights).sum(dim=1)
        else:
            if ts % query_frequency == 0:
                actions_to_take = action_seq
            raw_action = actions_to_take[:, ts % query_frequency]

        action = raw_action if delta_control else raw_action * action_std + action_mean
        if sim_backend == "physx_cpu":
            action = action.cpu().numpy()

        if frame_hook is not None:
            frame_hook(envs, ts)
        obs, _, _, truncated, info = envs.step(action)
        ts += 1

        truncated = np.asarray(common.to_numpy(truncated))
        if truncated.any():
            assert truncated.all(), "all eval envs must truncate together (ignore_terminations=True)"
            for k, v in _episode_metrics(info).items():
                metrics[k].append(v)
            eps_count += num_envs
            ts = 0
            if temporal_agg:
                all_time_actions = fresh_action_table()
    agent.train()

    per_episode = {k: np.concatenate(v) for k, v in metrics.items()}
    return {"per_episode": per_episode, "summary": summarize(per_episode)}


def summarize(per_episode):
    summary = {}
    n = len(next(iter(per_episode.values()))) if per_episode else 0
    summary["n_trials"] = n
    for k, v in per_episode.items():
        if k.startswith("success") or k.startswith("fail"):
            hits = int(np.sum(v > 0.5))
            lo, hi = wilson_interval(hits, len(v))
            summary[k] = {"count": hits, "n": len(v), "rate": hits / len(v), "ci95": [lo, hi]}
        else:
            summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
    return summary
