"""Inverse dynamics model (IDM): predicts the action a_t from a short frame window around the transition o_t -> o_{t+1}.

Used for VPT-style pseudo-labelling: train on a few labelled demos, then label action-free demos.
The default window (o_t, o_{t+1}) can be widened, e.g. (o_{t-1}, o_t, o_{t+1}, o_{t+2}) as in Morin et al. (2026),
because the controller's response to a_t continues over the following frames.
Action layout (pd_ee_delta_pos): [dx, dy, dz, gripper], xyz in [-1, 1], gripper exactly -1 or +1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.action_free.data import window_indices

PAIR = (0, 1)


class IDM(nn.Module):
    """Small CNN on a stack of k 128x128 RGB frames (3k channels) -> 3 xyz values + 1 gripper logit."""

    def __init__(self, n_frames=2, width=32):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv2d(i, o, k, 2, k // 2), nn.GroupNorm(8, o), nn.ReLU())

        self.conv = nn.Sequential(block(3 * n_frames, width, 5), block(width, 2 * width, 3),
                                  block(2 * width, 4 * width, 3), block(4 * width, 4 * width, 3))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * width * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, windows):
        """windows: uint8 or float (B, k, H, W, 3). Returns (xyz (B, 3), gripper logit (B,))."""
        b, k, h, w, c = windows.shape
        x = windows.permute(0, 1, 4, 2, 3).reshape(b, k * c, h, w).float() / 255.0
        out = self.head(self.conv(x))
        return out[:, :3], out[:, 3]

    @torch.no_grad()
    def predict(self, frames, offsets=PAIR, batch=256):
        """Pseudo-label one episode (frames (T+1, H, W, 3)) -> actions (T, 4): xyz clipped to [-1, 1], gripper +1/-1."""
        self.eval()
        dev = next(self.parameters()).device
        idx = window_indices(len(frames) - 1, offsets)
        outs = []
        for i in range(0, len(idx), batch):
            xyz, g = self(torch.as_tensor(frames[idx[i:i + batch]], device=dev))
            outs.append(torch.cat([xyz.clamp(-1, 1), torch.where(g > 0, 1.0, -1.0)[:, None]], 1).cpu())
        return torch.cat(outs).numpy() if outs else np.zeros((0, 4), np.float32)


class _Pool:
    """All labelled frames in one array plus, for every transition, the global indices of its frame window."""

    def __init__(self, episodes, offsets):
        frames, windows, acts, start = [], [], [], 0
        for fr, ac in episodes:
            frames.append(fr)
            windows.append(window_indices(len(ac), offsets) + start)
            acts.append(ac)
            start += len(fr)
        self.frames = torch.as_tensor(np.concatenate(frames))
        self.windows = torch.as_tensor(np.concatenate(windows))
        self.actions = np.concatenate(acts)


def train_idm(episodes, device, steps=3000, batch=64, lr=1e-3, seed=0, offsets=PAIR, log_every=0):
    """Train an IDM on labelled episodes [(frames (T+1,H,W,3), actions (T,4)), ...]. Same step budget for every N."""
    torch.manual_seed(seed)
    pool = _Pool(episodes, offsets)
    xyz_t = torch.as_tensor(pool.actions[:, :3])
    grip_t = torch.as_tensor((pool.actions[:, 3] > 0).astype(np.float32))
    model = IDM(n_frames=len(offsets)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        idx = torch.randint(len(pool.actions), (batch,), generator=gen)
        xyz, g = model(pool.frames[pool.windows[idx]].to(device))
        loss = F.mse_loss(xyz, xyz_t[idx].to(device)) + F.binary_cross_entropy_with_logits(g, grip_t[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}: loss {loss.item():.4f}")
    model.eval()
    return model, {"train_transitions": int(len(pool.actions)), "final_loss": float(loss.item()),
                   "offsets": list(offsets)}


def label_quality(pred, true):
    """Per-dimension pseudo-label accuracy against ground-truth actions."""
    err = pred[:, :3] - true[:, :3]
    var = true[:, :3].var(0)
    out = {f"{d}_mae": float(np.abs(err[:, i]).mean()) for i, d in enumerate("xyz")}
    out.update({f"{d}_r2": float(1 - (err[:, i] ** 2).mean() / var[i]) for i, d in enumerate("xyz")})
    out["xyz_mae"] = float(np.abs(err).mean())
    out["xyz_zero_baseline_mae"] = float(np.abs(true[:, :3]).mean())
    out["gripper_acc"] = float(((pred[:, 3] > 0) == (true[:, 3] > 0)).mean())
    out["gripper_majority_baseline_acc"] = float(max((true[:, 3] > 0).mean(), (true[:, 3] <= 0).mean()))
    return out
