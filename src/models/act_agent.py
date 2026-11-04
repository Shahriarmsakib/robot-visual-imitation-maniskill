"""ACT policy (CVAE encoder + ResNet backbone + transformer decoder).

Derived from `Agent` and `kl_divergence` in ManiSkill v3.0.1
examples/baselines/act/train_rgbd.py (Apache-2.0). Changes: dimensions are
passed in directly instead of read from an environment, and the depth flag
comes from the config instead of a global.
"""
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import src  # noqa: F401  puts the vendored `act` package on sys.path
from act.detr.backbone import build_backbone
from act.detr.detr_vae import DETRVAE, build_encoder
from act.detr.transformer import build_transformer


class ACTAgent(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, cfg):
        """cfg needs the upstream ACT fields: backbone, lr_backbone, masks, dilation, include_depth,
        position_embedding, hidden_dim, dropout, nheads, dim_feedforward, enc_layers, dec_layers,
        pre_norm, num_queries, kl_weight."""
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.kl_weight = cfg.kl_weight
        self.include_depth = cfg.include_depth
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = DETRVAE(
            [build_backbone(cfg)],
            build_transformer(cfg),
            build_encoder(cfg),
            state_dim=state_dim,
            action_dim=act_dim,
            num_queries=cfg.num_queries,
        )

    def _prepare(self, obs):
        obs = dict(obs)
        obs["rgb"] = self.normalize(obs["rgb"].float() / 255.0)
        if self.include_depth:
            obs["depth"] = obs["depth"].float()
        return obs

    def compute_loss(self, obs, action_seq):
        a_hat, (mu, logvar) = self.model(self._prepare(obs), action_seq)
        total_kld = kl_divergence(mu, logvar)
        l1 = F.l1_loss(action_seq, a_hat, reduction="none").mean()
        return {"l1": l1, "kl": total_kld[0], "loss": l1 + total_kld[0] * self.kl_weight}

    def get_action(self, obs):
        a_hat, _ = self.model(self._prepare(obs))  # no actions given: sample from the prior
        return a_hat


def kl_divergence(mu, logvar):
    if mu.ndim == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.ndim == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0, True)
