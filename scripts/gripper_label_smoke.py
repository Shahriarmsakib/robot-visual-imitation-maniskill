"""Prototype: gripper labels from estimated finger opening instead of the IDM.

Finger opening w_t = sum of the two finger joints (qpos[:, 7:9]). A logistic map from (w_t .. w_t+3) to the true
gripper command is fit on labelled demos 0-99 with recorded widths; a CNN (same trunk as the position CNN) estimates
w_t from each image, trained on demos 0-79. Accuracy by phase on the 900 action-free demos, vs the IDM's gripper.

Usage: python scripts/gripper_label_smoke.py <dir with idm.npz from scripts/labeller_smoke.py>
Writes <dir>/grip_width_cnn.npz (gripper channel only).
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from labeller_smoke import PosCNN  # noqa: E402

H5 = REPO / "data/action_free/conversion/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
OUT = Path(sys.argv[1])
LEADS = (0, 1, 2, 3)


def feats(w):
    T = len(w) - 1
    cols = [w[np.minimum(np.arange(T) + k, T)] for k in LEADS]
    cols += [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    return np.stack(cols + [np.ones(T)], 1)


def fit_logreg(X, y, steps=3000, lr=0.5):
    Xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    mu, sd = Xt[:, :-1].mean(0), Xt[:, :-1].std(0) + 1e-8
    Xn = torch.cat([(Xt[:, :-1] - mu) / sd, Xt[:, -1:]], 1)
    w = torch.zeros(X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=lr, max_iter=steps)

    def closure():
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(Xn @ w, yt); loss.backward(); return loss
    opt.step(closure)
    return lambda Z: ((torch.cat([(torch.as_tensor(Z, dtype=torch.float32)[:, :-1] - mu) / sd,
                                  torch.as_tensor(Z, dtype=torch.float32)[:, -1:]], 1) @ w) > 0).numpy()


f = h5py.File(H5, "r")
ep = lambda i: (f[f"traj_{i}/obs/sensor_data/base_camera/rgb"][:], f[f"traj_{i}/actions"][:],
                f[f"traj_{i}/obs/agent/qpos"][:, 7:9].sum(1))
lab = {i: ep(i) for i in range(100)}
X = np.concatenate([feats(w) for _, _, w in lab.values()]); y = np.concatenate([a[:, 3] > 0 for _, a, _ in lab.values()])
clf = fit_logreg(X, y)
print(f"oracle widths -> command, accuracy on labelled 0-99: {(clf(X) == y).mean():.4f}", flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)
fr = np.concatenate([lab[i][0] for i in range(80)]); wt = np.concatenate([lab[i][2] for i in range(80)])
mu, sd = wt.mean(), wt.std()
m = PosCNN().to(dev); m.head[-1] = torch.nn.Linear(256, 1).to(dev)
opt = torch.optim.Adam(m.parameters(), lr=1e-3); gen = torch.Generator().manual_seed(0)
Xf, Yf = torch.as_tensor(fr), torch.as_tensor(((wt - mu) / sd).astype(np.float32))[:, None]
for s in range(3000):
    idx = torch.randint(len(Xf), (64,), generator=gen)
    loss = F.mse_loss(m(Xf[idx].to(dev)), Yf[idx].to(dev)); opt.zero_grad(); loss.backward(); opt.step()
m.eval()
pred = lambda fr_: np.concatenate([m(torch.as_tensor(fr_[i:i + 512], device=dev)).detach().cpu().numpy()[:, 0]
                                   for i in range(0, len(fr_), 512)]) * sd + mu
cal_err = np.concatenate([pred(lab[i][0]) - lab[i][2] for i in range(80, 100)])
print(f"width CNN held-out (80-99) error SD {cal_err.std() * 1000:.2f} mm, width range {wt.min()*1000:.1f}-{wt.max()*1000:.1f} mm", flush=True)

idm = np.load(OUT / "idm.npz")
agg = {}
out = {}
for i in range(100, 1000):
    g = f[f"traj_{i}"]; fr_, a, w = ep(i); T = len(a)
    grasp = g["obs/extra/is_grasped"][:T]; near = np.linalg.norm(g["env_states/actors/cube"][:T, :3] - g["obs/extra/goal_pos"][:T], axis=1) < 0.025
    ph = np.where(near & grasp, "hold", np.where(grasp, "carry", "approach"))
    true = a[:, 3] > 0
    cnn = clf(feats(pred(fr_)))
    out[f"traj_{i}"] = np.where(cnn, 1.0, -1.0).astype(np.float32)
    for name, lb in [("idm", idm[f"traj_{i}"][:, 3] > 0), ("width_cnn", cnn), ("width_oracle", clf(feats(w)))]:
        for p in ("approach", "carry", "hold"):
            k = ph == p; e = agg.setdefault((name, p), [0, 0]); e[0] += k.sum(); e[1] += (lb[k] != true[k]).sum()
np.savez(OUT / "grip_width_cnn.npz", **out)
for name in ("idm", "width_cnn", "width_oracle"):
    tot = sum(agg[(name, p)][0] for p in ("approach", "carry", "hold")); wr = sum(agg[(name, p)][1] for p in ("approach", "carry", "hold"))
    print(f"{name:12s} overall wrong {wr / tot * 100:5.2f}% | " + " | ".join(
        f"{p} {agg[(name, p)][1] / agg[(name, p)][0] * 100:5.1f}%" for p in ("approach", "carry", "hold")))
