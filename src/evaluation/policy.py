"""Load a trained ACT policy from any checkpoint written by the trainer."""
from types import SimpleNamespace

import torch

from src.models.act_agent import ACTAgent


def load_policy(ckpt_path, device):
    """Return (EMA agent in eval mode, config namespace, norm_stats, checkpoint step).

    Works with both full (latest.pt) and weights-only (final.pt, best_*.pt, step_*.pt) checkpoints.
    The state/action dimensions are read from the weights, so no dataset is needed.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ckpt["config"])
    weights = ckpt["ema_agent"]
    state_dim = weights["model.input_proj_robot_state.weight"].shape[1]
    act_dim = weights["model.action_head.weight"].shape[0]
    agent = ACTAgent(state_dim, act_dim, cfg)
    agent.load_state_dict(weights)
    agent.to(device).eval()
    return agent, cfg, ckpt.get("norm_stats"), ckpt["step"]
