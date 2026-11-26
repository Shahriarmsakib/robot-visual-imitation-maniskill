"""Inverse dynamics model (IDM): predicts the action a_t from a short frame window around the transition o_t -> o_{t+1}.

Used for VPT-style pseudo-labelling: train on a few labelled demos, then label action-free demos.
The default window (o_t, o_{t+1}) can be widened, e.g. (o_{t-1}, o_t, o_{t+1}, o_{t+2}) as in Morin et al. (2026),
because the controller's response to a_t continues over the following frames.
Action layout (pd_ee_delta_pos): [dx, dy, dz, gripper], xyz in [-1, 1], gripper exactly -1 or +1.

Two architectures: `IDM` (our small CNN, the default) and `ImpalaIDM`, the IDM recipe of Morin et al. (2026,
arXiv 2602.02762, Appendix D.2; code github.com/sachaMorin/lapo-plus, MIT): an IMPALA CNN (channels 16/32/32 x scale 4,
two residual blocks per stage, 256 features) on the stacked frames, then a fully connected action decoder (128, 128),
trained with Adam at 2e-4 on a piecewise linear schedule (1,000 warm-up steps from 1% of the rate, then linear decay to
0) with batch 128. Their Procgen actions are discrete; for our continuous xyz we use the SmoothL1 loss (beta 1) of their
LIBERO linear-head variant (Appendix D.3), and a binary cross-entropy for the gripper as in our IDM.
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


class _ImpalaResidual(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv0 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.conv1(F.relu(self.conv0(F.relu(x))))


class _ImpalaStage(nn.Module):
    """conv 3x3 -> max-pool 3x3 stride 2 -> two residual blocks (IMPALA ConvSequence)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.res0, self.res1 = _ImpalaResidual(cout), _ImpalaResidual(cout)

    def forward(self, x):
        return self.res1(self.res0(F.max_pool2d(self.conv(x), 3, 2, 1)))


class ImpalaIDM(IDM):
    """Morin et al.'s IDM: IMPALA CNN on k stacked RGB frames -> 256 features -> latent -> decoder -> 3 xyz + 1 gripper logit."""

    def __init__(self, n_frames=2, size=128, scale=4, channels=(16, 32, 32), features=256, latent=128,
                 decoder=(128, 128)):
        nn.Module.__init__(self)
        self.size = size
        stages, cin, s = [], 3 * n_frames, size
        for c in channels:
            stages.append(_ImpalaStage(cin, scale * c))
            cin, s = scale * c, (s + 1) // 2
        self.conv = nn.Sequential(*stages, nn.Flatten(), nn.ReLU())
        self.fc = nn.Linear(cin * s * s, features)
        self.latent = nn.Linear(features, latent)  # their IDM's latent-action head; the decoder reads it unquantised
        layers, d = [], latent
        for h in decoder:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.decoder = nn.Sequential(*layers, nn.Linear(d, 4))

    def forward(self, windows):
        b, k, h, w, c = windows.shape
        x = windows.permute(0, 1, 4, 2, 3).reshape(b, k * c, h, w).float() / 255.0 - 0.5
        if h != self.size:  # e.g. 128 -> 64 (their Procgen resolution): area averaging
            x = F.interpolate(x, size=(self.size, self.size), mode="area")
        out = self.decoder(self.latent(F.relu(self.fc(self.conv(x)))))
        return out[:, :3].contiguous(), out[:, 3]  # contiguous: MPS smooth_l1_loss rejects the strided slice


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


ARCHS = ("small", "impala")


def train_idm(episodes, device, steps=3000, batch=64, lr=1e-3, seed=0, offsets=PAIR, log_every=0, arch="small",
              schedule="constant", loss="mse", res=None):
    """Train an IDM on labelled episodes [(frames (T+1,H,W,3), actions (T,4)), ...]. Same step budget for every N.

    Defaults reproduce our original IDM. Morin et al.'s recipe: arch="impala", batch=128, lr=2e-4,
    schedule="warmup_linear", loss="smoothl1" (see the module docstring). res: input size of the IMPALA IDM
    (default: the frames' own size, 128; 64 = their Procgen resolution).
    """
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    torch.manual_seed(seed)
    pool = _Pool(episodes, offsets)
    xyz_t = torch.as_tensor(pool.actions[:, :3])
    grip_t = torch.as_tensor((pool.actions[:, 3] > 0).astype(np.float32))
    if arch == "impala":
        model = ImpalaIDM(n_frames=len(offsets), size=res or pool.frames.shape[1]).to(device)
    else:
        model = IDM(n_frames=len(offsets)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if schedule == "warmup_linear":  # Morin et al.: 0.01*lr -> lr over 1,000 steps, then linear to 0 at the last step
        warm = min(1000, max(1, steps // 10))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.01 + 0.99 * s / warm if s < warm else max(0.0, (steps + 1 - s) / (steps + 1 - warm)))
    elif schedule == "constant":
        sched = None
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    xyz_loss = F.smooth_l1_loss if loss == "smoothl1" else F.mse_loss
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        idx = torch.randint(len(pool.actions), (batch,), generator=gen)
        xyz, g = model(pool.frames[pool.windows[idx]].to(device))
        loss_v = xyz_loss(xyz, xyz_t[idx].to(device)) + F.binary_cross_entropy_with_logits(g, grip_t[idx].to(device))
        opt.zero_grad()
        loss_v.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"    step {step + 1}: loss {loss_v.item():.4f}", flush=True)
    model.eval()
    return model, {"train_transitions": int(len(pool.actions)), "final_loss": float(loss_v.item()),
                   "offsets": list(offsets), "arch": arch, "batch": batch, "lr": lr, "schedule": schedule,
                   "loss": loss, "res": int(res or pool.frames.shape[1]) if arch == "impala" else None}


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
