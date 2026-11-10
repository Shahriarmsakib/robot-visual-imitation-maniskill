"""Structure of action-label errors on the 900 action-free demos: IDM pseudo-labels vs controlled noise.

Usage:
    python scripts/label_error_structure.py --out results/action_free/noise_sweep/label_error_structure.json \
        idm_N100_seed0=data/action_free/pseudo_labels/idm_N100_seed0.npz noise_xy025=data/action_free/pseudo_labels/noise_xy025.npz

Per x/y/z dimension, against the true actions:
  step_r2                        R^2 of single-step labels (what the IDM study reports)
  bias                           mean error
  slope_pred_on_true             least-squares slope of label on true action (1 = unbiased scale, <1 = shrunk to zero)
  err_lag1_autocorr              correlation of the error at step t and t+1 within an episode (0 = independent)
  chunk30_displacement_r2        R^2 of the summed action over 30-step windows (one ACT action chunk)
  episode_offset_share_of_err_var share of the error variance that is a constant offset per episode
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.action_free.data import ACTION_FREE_POOL, DEFAULT_H5  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("labels", nargs="+", help="name=path.npz")
ap.add_argument("--out", type=Path)
args = ap.parse_args()
files = dict(a.split("=", 1) for a in args.labels)
with h5py.File(DEFAULT_H5, "r") as f:
    true = {f"traj_{i}": f[f"traj_{i}"]["actions"][:].astype(np.float64) for i in ACTION_FREE_POOL}
H = 30  # ACT chunk length used in training
def r2(p, t): return 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
out = {}
for name, path in files.items():
    lab = np.load(path); keys = list(true)
    T = np.concatenate([true[k][:, :3] for k in keys]); P = np.concatenate([lab[k][:, :3].astype(np.float64) for k in keys])
    E = P - T
    slope = [float(np.polyfit(T[:, d], P[:, d], 1)[0]) for d in range(3)]
    # lag-1 autocorrelation of the error within episodes
    ac = []
    for d in range(3):
        a = np.concatenate([(lab[k][:-1, d] - true[k][:-1, d]) for k in keys]); b = np.concatenate([(lab[k][1:, d] - true[k][1:, d]) for k in keys])
        ac.append(round(float(np.corrcoef(a, b)[0, 1]), 3) if a.std() > 0 else None)
    # displacement over a 30-step window (what a chunk commands): R2 of summed actions
    St, Sp = [], []
    for k in keys:
        t, p = true[k][:, :3], lab[k][:, :3].astype(np.float64)
        if len(t) < H: continue
        ct, cp = np.cumsum(np.vstack([np.zeros(3), t]), 0), np.cumsum(np.vstack([np.zeros(3), p]), 0)
        St.append(ct[H:] - ct[:-H]); Sp.append(cp[H:] - cp[:-H])
    St, Sp = np.concatenate(St), np.concatenate(Sp)
    # share of error that is a per-episode constant offset
    ep_mean = np.concatenate([np.repeat((lab[k][:, :3] - true[k][:, :3]).mean(0, keepdims=True), len(true[k]), 0) for k in keys])
    out[name] = {"step_r2": r2(P, T).round(3).tolist(), "bias": E.mean(0).round(4).tolist(),
                 "slope_pred_on_true": np.round(slope, 3).tolist(), "err_lag1_autocorr": ac,
                 "chunk30_displacement_r2": r2(Sp, St).round(3).tolist(),
                 "episode_offset_share_of_err_var": [round(float(v), 3) if np.isfinite(v) else None for v in ep_mean.var(0) / np.where(E.var(0) > 0, E.var(0), np.nan)]}
text = json.dumps(out, indent=1)
print(text)
if args.out:
    args.out.write_text(text + "\n")
