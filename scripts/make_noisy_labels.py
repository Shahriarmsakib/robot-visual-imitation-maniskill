"""Controlled label corruption for the label-noise sweep: true actions of the action-free demos plus known errors.

Usage:
    python scripts/make_noisy_labels.py --out data/action_free/pseudo_labels/noise_xy025.npz --xy-r2 0.25
    python scripts/make_noisy_labels.py --out ... --z-r2 0.81 --grip-flip 0.06
    python scripts/make_noisy_labels.py --out ... --bias -0.02 0 -0.013
    python scripts/make_noisy_labels.py --out ... --xy-scale 0.4            # IDM-like shrinkage, no noise
    python scripts/make_noisy_labels.py --out ... --xy-r2 0.25 --ar1 0.9    # noise correlated over time

Output has the same format as the IDM pseudo-labels (one (T, 4) array per action-free episode), so
scripts/build_pseudo_dataset.py turns it into an arm-B training set. Corruptions (all optional, combinable):
  --x-r2/--y-r2/--z-r2/--xy-r2  i.i.d. Gaussian noise per step on that dimension, scaled so the labels reach the
                                given R^2 against the true actions (R^2 = 1 - sigma^2 / var(true)); 1 = no noise
  --bias bx by bz               constant offset added to every step's x/y/z action
  --grip-flip p                 each step's gripper label is flipped with probability p
  --x-scale/--y-scale/--z-scale/--xy-scale
                                multiply the true action by this factor before adding noise (IDM-like shrinkage
                                towards zero; 1 = none)
  --ar1 rho                     make the noise an AR(1) process within each episode with step-to-step correlation
                                rho (stationary, same variance as i.i.d.; 0 = independent)
xyz labels are clipped to [-1, 1] afterwards, like the IDM output. The achieved label quality is measured
with the same function as the IDM study (src.action_free.idm.label_quality) and printed / saved as json.
Without --ar1 and the scale options the noise is independent across steps, unlike the IDM's errors; the options
reproduce the two structural properties measured by scripts/label_error_structure.py. The noise is sized against
the variance of the true actions, so with a scale < 1 the achieved R^2 is lower than the --*-r2 target.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_free.data import ACTION_FREE_POOL, DEFAULT_H5  # noqa: E402


def r2_sigma(r2, var):
    if not 0 <= r2 <= 1:
        raise ValueError(f"target R^2 must be in [0, 1], got {r2}")
    return float(np.sqrt((1 - r2) * var))


def main():
    from src.action_free.idm import label_quality

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=DEFAULT_H5)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--x-r2", type=float, default=1.0)
    p.add_argument("--y-r2", type=float, default=1.0)
    p.add_argument("--z-r2", type=float, default=1.0)
    p.add_argument("--xy-r2", type=float, default=None, help="sets both --x-r2 and --y-r2")
    p.add_argument("--bias", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("BX", "BY", "BZ"))
    p.add_argument("--grip-flip", type=float, default=0.0)
    p.add_argument("--x-scale", type=float, default=1.0)
    p.add_argument("--y-scale", type=float, default=1.0)
    p.add_argument("--z-scale", type=float, default=1.0)
    p.add_argument("--xy-scale", type=float, default=None, help="sets both --x-scale and --y-scale")
    p.add_argument("--ar1", type=float, default=0.0, help="step-to-step correlation of the noise, in [0, 1)")
    p.add_argument("--seed", type=int, default=0, help="noise seed")
    p.add_argument("--max-action-free", type=int, default=None)
    args = p.parse_args()
    if args.xy_r2 is not None:
        args.x_r2 = args.y_r2 = args.xy_r2
    if args.xy_scale is not None:
        args.x_scale = args.y_scale = args.xy_scale
    if not 0 <= args.ar1 < 1:
        raise ValueError(f"--ar1 must be in [0, 1), got {args.ar1}")
    scale = np.asarray([args.x_scale, args.y_scale, args.z_scale], np.float32)

    ids = list(ACTION_FREE_POOL)[: args.max_action_free]
    with h5py.File(args.source, "r") as f:
        true = {f"traj_{i}": f[f"traj_{i}"]["actions"][:].astype(np.float32) for i in ids}
    all_true = np.concatenate(list(true.values()))
    var = all_true[:, :3].var(0)
    sigma = [r2_sigma(r, v) for r, v in zip((args.x_r2, args.y_r2, args.z_r2), var)]

    rng = np.random.default_rng(args.seed)
    labels = {}
    for k, a in true.items():
        lab = a.copy()
        z = rng.normal(0.0, 1.0, size=(len(a), 3))
        if args.ar1 > 0:  # stationary AR(1): unit variance, correlation ar1 between consecutive steps
            for t in range(1, len(z)):
                z[t] = args.ar1 * z[t - 1] + np.sqrt(1 - args.ar1 ** 2) * z[t]
        lab[:, :3] = lab[:, :3] * scale + z * np.asarray(sigma, np.float32) + np.asarray(args.bias, np.float32)
        lab[:, :3] = np.clip(lab[:, :3], -1, 1)
        flip = rng.random(len(a)) < args.grip_flip
        lab[flip, 3] = -lab[flip, 3]
        labels[k] = lab.astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **labels)
    q = label_quality(np.concatenate(list(labels.values())), all_true)
    info = {"corruption": {"x_r2": args.x_r2, "y_r2": args.y_r2, "z_r2": args.z_r2, "bias": args.bias,
                           "grip_flip": args.grip_flip, "scale_xyz": scale.tolist(), "ar1": args.ar1,
                           "noise_seed": args.seed, "sigma_xyz": sigma},
            "episodes": len(labels), "transitions": int(len(all_true)), "achieved": q}
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {args.out}: {len(labels)} episodes | achieved R2 x {q['x_r2']:.3f} y {q['y_r2']:.3f} "
          f"z {q['z_r2']:.3f} | gripper acc {q['gripper_acc']:.3f}")


if __name__ == "__main__":
    main()
