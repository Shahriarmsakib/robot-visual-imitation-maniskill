"""Smoke test: which pseudo-labeller gives the best-structured action labels on the 900 action-free demos?

Protocol (no leakage): models train on labelled demos 0-79; calibration (rescaling, noise levels) uses held-out
labelled demos 80-99; the 900 action-free demos' true actions and positions are used only for scoring
(except `pos_oracle`, which reads their recorded gripper positions on purpose, as the proprio reference).

Labellers (x, y, z; the gripper channel is the IDM's unless named otherwise):
  idm          image IDM on (o_t, o_t+1), trained on demos 0-79 (same recipe as the study)
  idm_rescale  idm with each axis rescaled so its slope on held-out demos 80-99 is 1 (bias correction)
  pos_oracle   recorded gripper positions, differenced and mapped to actions by a linear map fit on demos 0-99
  pos_cnn      gripper position estimated from each image by a CNN (trained on 0-79), then the same map
  fusion       Kalman/RTS smoother: IDM motion as the process model + CNN positions as measurements;
               noise levels from held-out demos 80-99; then the same position-to-action map
  fusion_gripw     fusion x, y, z + gripper from the finger opening: a CNN estimates the opening (sum of the two
                   finger joints) in each image (trained on 0-79), a logistic map from the opening over t..t+3 to
                   the command is fit on recorded openings of demos 0-99
  fusion_truegrip  fusion x, y, z + the TRUE gripper command (diagnostic only: reads the action-free demos' actions)
  posrec_truegrip  recorded positions -> the same map, TRUE gripper (diagnostic: is position-to-action lossless?)
  fusion_errshuf   fusion_truegrip with its errors moved to ANOTHER episode, TRUE gripper (diagnostic). Per axis
                   the labels are split as slope * true action + residual; each episode keeps slope * its own
                   true action and takes the residual of an episode of similar length. Keeps the shrink, the
                   residual's size and its pattern over time, but breaks the residual's link to the scene
Writes one npz per labeller (traj_<id> -> (T, 4)) plus a json with its achieved label quality on the action-free
demos (scored against their true actions, which no labeller reads), for scripts/build_pseudo_dataset.py and
scripts/label_error_structure.py. Used by scripts/run_phase3_noise.sh with ARGS="labeller:<name>".
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.action_free.data import DEFAULT_H5  # noqa: E402
from src.action_free.idm import label_quality, train_idm  # noqa: E402

TRAIN, CAL, FREE = range(0, 80), range(80, 100), range(100, 1000)
LAGS = (-1, 0, 1, 2)  # displacement windows d_{t+k} = p_{t+k+1} - p_{t+k} used to recover a_t


def load(f, i, frames=True):
    g = f[f"traj_{i}"]
    return (g["obs/sensor_data/base_camera/rgb"][:] if frames else None,
            g["actions"][:].astype(np.float32), g["obs/extra/tcp_pose"][:, :3].astype(np.float64))


def finger_opening(f, i):
    return f[f"traj_{i}/obs/agent/qpos"][:, 7:9].sum(1).astype(np.float64)


def disp_features(p):
    """p (T+1, 3) -> features (T, 3*len(LAGS)+1): displacement windows around each transition, zero-padded."""
    d = np.diff(p, axis=0)  # (T, 3), d[t] = p[t+1] - p[t]
    T = len(d)
    cols = []
    for k in LAGS:
        s = np.zeros_like(d)
        lo, hi = max(0, -k), min(T, T - k)
        s[lo:hi] = d[lo + k:hi + k]
        cols.append(s)
    return np.concatenate(cols + [np.ones((T, 1))], 1)


def fit_ridge(X, Y, lam=1e-6):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)


class PosCNN(nn.Module):
    """Same trunk as the IDM, one frame in, `out` values (normalised gripper xyz, or finger opening) out."""

    def __init__(self, width=32, out=3):
        super().__init__()

        def block(i, o, k):
            return nn.Sequential(nn.Conv2d(i, o, k, 2, k // 2), nn.GroupNorm(8, o), nn.ReLU())
        self.conv = nn.Sequential(block(3, width, 5), block(width, 2 * width, 3), block(2 * width, 4 * width, 3),
                                  block(4 * width, 4 * width, 3))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * width * 64, 256), nn.ReLU(), nn.Linear(256, out))

    def forward(self, x):
        return self.head(self.conv(x.permute(0, 3, 1, 2).float() / 255.0))


def train_pos(frames, pos, device, steps, seed=0):
    """Regress pos (N, d) from single frames; returns (predict(frames) -> (N, d), final loss)."""
    torch.manual_seed(seed)
    mu, sd = pos.mean(0), pos.std(0)
    X, Y = torch.as_tensor(frames), torch.as_tensor(((pos - mu) / sd).astype(np.float32))
    m = PosCNN(out=Y.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(X), (64,), generator=gen)
        loss = F.mse_loss(m(X[idx].to(device)), Y[idx].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()

    @torch.no_grad()
    def predict(fr):
        out = [m(torch.as_tensor(fr[i:i + 512], device=device)).cpu().numpy() for i in range(0, len(fr), 512)]
        return np.concatenate(out) * sd + mu
    return predict, float(loss.item())


GRIP_LEADS = (0, 1, 2, 3)


def grip_features(w):
    """Finger opening w (T+1,) -> (T, 8): opening at t..t+3 (clamped at the end), their 3 differences, bias."""
    T = len(w) - 1
    cols = [w[np.minimum(np.arange(T) + k, T)] for k in GRIP_LEADS]
    cols += [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
    return np.stack(cols + [np.ones(T)], 1)


def fit_grip_map(X, y):
    """Logistic regression (L-BFGS) from grip_features to the command (True = open); returns predict(X) -> bool."""
    Xt, yt = torch.as_tensor(X, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    mu, sd = Xt[:, :-1].mean(0), Xt[:, :-1].std(0) + 1e-8
    norm = lambda Z: torch.cat([(Z[:, :-1] - mu) / sd, Z[:, -1:]], 1)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=0.5, max_iter=3000)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(norm(Xt) @ w, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return lambda Z: (norm(torch.as_tensor(Z, dtype=torch.float64)) @ w > 0).detach().numpy()


def rts(z, v, R, Q):
    """1-D-per-axis Kalman filter + RTS smoother. z (T+1,3) position measurements, v (T,3) motion p_{t+1}-p_t,
    R, Q (3,) measurement / process variances. Returns smoothed positions (T+1, 3)."""
    n = len(z)
    xf, Pf, xp, Pp = np.zeros_like(z), np.zeros_like(z), np.zeros_like(z), np.zeros_like(z)
    xp[0], Pp[0] = z[0], R * 10
    for t in range(n):
        if t > 0:
            xp[t], Pp[t] = xf[t - 1] + v[t - 1], Pf[t - 1] + Q
        K = Pp[t] / (Pp[t] + R)
        xf[t], Pf[t] = xp[t] + K * (z[t] - xp[t]), (1 - K) * Pp[t]
    xs = xf.copy()
    for t in range(n - 2, -1, -1):
        C = Pf[t] / Pp[t + 1]
        xs[t] = xf[t] + C * (xs[t + 1] - xp[t + 1])
    return xs


def errshuf(labels, true):
    """Keep each episode's shrink (slope * its true action) and give it the x, y, z residual of another episode.

    Per axis, label = b * true + residual, with b the least-squares slope over all episodes. Episodes sorted by
    length are swapped in neighbouring pairs (rank 0 <-> 1, 2 <-> 3, ...; an odd last one takes its lower
    neighbour's), so no episode keeps its own residual. A donor's residual is cut to the receiver's length, or
    extended by its own time-reversed tail when shorter. The gripper channel is the true command."""
    keys = sorted(true, key=lambda k: (len(true[k]), k))
    T_all = np.concatenate([true[k][:, :3] for k in keys]); L_all = np.concatenate([labels[k][:, :3] for k in keys])
    b = np.array([np.polyfit(T_all[:, d], L_all[:, d], 1)[0] for d in range(3)])
    out = {}
    for r, k in enumerate(keys):
        j = keys[r ^ 1] if (r ^ 1) < len(keys) else keys[r - 1]
        e, T = labels[j][:, :3] - b * true[j][:, :3], len(true[k])
        while len(e) < T:
            e = np.concatenate([e, e[::-1][:T - len(e)]])
        out[k] = np.concatenate([np.clip(b * true[k][:, :3] + e[:T], -1, 1), true[k][:, 3:4]], 1).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max-free", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed for the IDM and the two CNNs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    f = h5py.File(REPO / DEFAULT_H5, "r")
    tr = [load(f, i) for i in TRAIN]
    cal = [load(f, i) for i in CAL]
    free_ids = list(FREE)[: args.max_free] if args.max_free else list(FREE)
    print(f"loaded labelled ({time.time() - t0:.0f}s), device {dev}", flush=True)

    # 1) position -> action map, from recorded positions of all labelled demos (0-99)
    X = np.concatenate([disp_features(p) for _, a, p in tr + cal])
    Y = np.concatenate([a[:, :3] for _, a, p in tr + cal])
    W = fit_ridge(X, Y)
    fit_r2 = 1 - ((X @ W - Y) ** 2).mean(0) / Y.var(0)
    # 2) motion model: action window -> displacement d_t (for the fusion process model), also from 0-99
    def act_features(a):
        T = len(a); cols = []
        for k in (-2, -1, 0):  # d_t responds to the current and two previous commands (controller lag)
            s = np.zeros((T, 3)); lo = max(0, -k); s[lo:] = a[lo + k:T + k, :3] if k < 0 else a[:, :3]; cols.append(s)
        return np.concatenate(cols + [np.ones((T, 1))], 1)
    Xa = np.concatenate([act_features(a) for _, a, p in tr + cal])
    Yd = np.concatenate([np.diff(p, axis=0) for _, a, p in tr + cal])
    Wm = fit_ridge(Xa, Yd)
    motion_r2 = 1 - ((Xa @ Wm - Yd) ** 2).mean(0) / Yd.var(0)
    print(f"position->action map R2 (true positions, labelled) {np.round(fit_r2, 3)}; "
          f"action->motion map R2 {np.round(motion_r2, 3)}", flush=True)

    # 3) models on demos 0-79
    idm, info = train_idm([(fr, a) for fr, a, _ in tr], dev, steps=args.steps, seed=args.seed)
    print(f"IDM trained ({time.time() - t0:.0f}s), loss {info['final_loss']:.4f}", flush=True)
    pos_pred, ploss = train_pos(np.concatenate([fr for fr, _, _ in tr]), np.concatenate([p for _, _, p in tr]),
                                dev, args.steps, seed=args.seed)
    print(f"position CNN trained ({time.time() - t0:.0f}s), loss {ploss:.4f}", flush=True)
    wid_pred, wloss = train_pos(np.concatenate([fr for fr, _, _ in tr]),
                                np.concatenate([finger_opening(f, i) for i in TRAIN])[:, None], dev, args.steps,
                                seed=args.seed)
    grip_map = fit_grip_map(np.concatenate([grip_features(finger_opening(f, i)) for i in list(TRAIN) + list(CAL)]),
                            np.concatenate([a[:, 3] > 0 for _, a, _ in tr + cal]))
    grip_fit = float(np.mean(grip_map(np.concatenate([grip_features(finger_opening(f, i)) for i in list(TRAIN) + list(CAL)]))
                             == np.concatenate([a[:, 3] > 0 for _, a, _ in tr + cal])))
    print(f"finger-opening CNN trained ({time.time() - t0:.0f}s), loss {wloss:.4f}; opening->command map accuracy "
          f"on recorded openings {grip_fit:.4f}", flush=True)

    # 4) calibration on held-out demos 80-99
    ci = [idm.predict(fr) for fr, _, _ in cal]
    ct = np.concatenate([a[:, :3] for _, a, _ in cal])
    cp_ = np.concatenate([x[:, :3] for x in ci])
    slope = np.array([np.polyfit(ct[:, d], cp_[:, d], 1)[0] for d in range(3)])  # label on true
    zc = [pos_pred(fr) for fr, _, _ in cal]
    R = np.concatenate([z - p for z, (_, _, p) in zip(zc, cal)]).var(0)
    vc = [act_features(x) @ Wm for x in ci]
    Q = np.concatenate([v - np.diff(p, axis=0) for v, (_, _, p) in zip(vc, cal)]).var(0)
    wc = np.concatenate([wid_pred(fr)[:, 0] - finger_opening(f, i) for (fr, _, _), i in zip(cal, CAL)])
    cal_info = {"finger_opening_err_sd_heldout": float(wc.std()), "grip_map_fit_acc_recorded": grip_fit,
                "idm_slope_heldout": slope.tolist(), "pos_cnn_err_var": R.tolist(), "idm_motion_err_var": Q.tolist(),
                "pos_to_action_fit_r2": fit_r2.tolist(), "action_to_motion_fit_r2": motion_r2.tolist(),
                "idm_train": info, "pos_cnn_final_loss": ploss}
    print("held-out IDM slope", np.round(slope, 3), "| CNN pos err sd (mm)", np.round(np.sqrt(R) * 1000, 1),
          "| IDM motion err sd (mm)", np.round(np.sqrt(Q) * 1000, 2), flush=True)

    # 5) label the action-free demos
    labs = {k: {} for k in ["idm", "idm_rescale", "pos_oracle", "pos_cnn", "fusion", "fusion_gripw", "fusion_truegrip",
                            "posrec_truegrip"]}
    true = {}
    for n, i in enumerate(free_ids):
        fr, a, p = load(f, i)
        true[f"traj_{i}"] = a
        li = idm.predict(fr)
        g = li[:, 3:4]
        labs["idm"][f"traj_{i}"] = li
        labs["idm_rescale"][f"traj_{i}"] = np.concatenate([np.clip(li[:, :3] / slope, -1, 1), g], 1)
        to_act = lambda pos: np.concatenate([np.clip(disp_features(pos) @ W, -1, 1), g], 1).astype(np.float32)
        labs["pos_oracle"][f"traj_{i}"] = to_act(p)
        z = pos_pred(fr)
        labs["pos_cnn"][f"traj_{i}"] = to_act(z)
        fu = to_act(rts(z, act_features(li) @ Wm, R, Q))
        labs["fusion"][f"traj_{i}"] = fu
        gw = np.where(grip_map(grip_features(wid_pred(fr)[:, 0])), 1.0, -1.0)[:, None]
        labs["fusion_gripw"][f"traj_{i}"] = np.concatenate([fu[:, :3], gw], 1).astype(np.float32)
        labs["fusion_truegrip"][f"traj_{i}"] = np.concatenate([fu[:, :3], a[:, 3:4]], 1).astype(np.float32)
        tg = a[:, 3:4]
        labs["posrec_truegrip"][f"traj_{i}"] = np.concatenate([to_act(p)[:, :3], tg], 1).astype(np.float32)
        if (n + 1) % 150 == 0:
            print(f"  labelled {n + 1}/{len(free_ids)} ({time.time() - t0:.0f}s)", flush=True)
    keys = list(true)
    labs["fusion_errshuf"] = errshuf(labs["fusion_truegrip"], true)
    for k, d in labs.items():
        np.savez(args.out / f"{k}.npz", **{kk: v.astype(np.float32) for kk, v in d.items()})
        q = label_quality(np.concatenate([d[kk] for kk in keys]), np.concatenate([true[kk] for kk in keys]))
        (args.out / f"{k}.json").write_text(json.dumps({"labeller": k, "episodes": len(keys), "device": dev,
                                                        "steps": args.steps, "achieved": q}, indent=1) + "\n")
        print(f"{k}: R2 x {q['x_r2']:.3f} y {q['y_r2']:.3f} z {q['z_r2']:.3f} | gripper acc {q['gripper_acc']:.3f}",
              flush=True)
    (args.out / "calibration.json").write_text(json.dumps(cal_info, indent=1))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
