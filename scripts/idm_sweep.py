"""IDM pseudo-labelling sweep: train an inverse dynamics model on N labelled PickCube demos, label the 900
action-free demos, and measure label accuracy against their (hidden) true actions.

Usage:
    python scripts/idm_sweep.py [--ns 5 10 25 50 100] [--seeds 0 1 2] [--steps 3000]

Writes results/action_free/idm/{metrics.csv, summary.json, splits/*.json} and pseudo-labels to
data/action_free/pseudo_labels/idm_N{n}_seed{s}.npz (one (T, 4) array per action-free episode).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np

    from src.action_free.data import ACTION_FREE_POOL, DEFAULT_H5, LABELLED_POOL, load_episodes, save_split
    from src.action_free.idm import label_quality, train_idm
    from src.utils import select_device, sha256_file

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--h5", default=DEFAULT_H5)
    p.add_argument("--ns", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--offsets", type=int, nargs="+", default=[0, 1],
                   help="IDM frame window relative to t, chosen on the labelled-pool validation split")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("results/action_free/idm"))
    p.add_argument("--pseudo-dir", type=Path, default=Path("data/action_free/pseudo_labels"))
    p.add_argument("--max-action-free", type=int, default=None, help="limit the action-free pool (for tests)")
    args = p.parse_args()

    device = select_device(args.device)
    free_ids = list(ACTION_FREE_POOL)[: args.max_action_free]
    t0 = time.time()
    labelled_pool = load_episodes(args.h5, LABELLED_POOL)
    free = load_episodes(args.h5, free_ids)
    true_free = np.concatenate([free[i][1] for i in free_ids])
    print(f"device {device} | labelled pool {len(labelled_pool)} demos | action-free pool {len(free)} demos, "
          f"{len(true_free)} transitions | loaded in {time.time() - t0:.0f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    args.pseudo_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in args.ns:
        for seed in args.seeds:
            ids = save_split(args.out / "splits" / f"N{n}_seed{seed}.json", n, seed)
            t1 = time.time()
            model, info = train_idm([labelled_pool[i] for i in ids], device, steps=args.steps, seed=seed,
                                    offsets=tuple(args.offsets))
            train_s = time.time() - t1
            preds = {i: model.predict(free[i][0], tuple(args.offsets)) for i in free_ids}
            q = label_quality(np.concatenate([preds[i] for i in free_ids]), true_free)
            np.savez_compressed(args.pseudo_dir / f"idm_N{n}_seed{seed}.npz", **{f"traj_{i}": preds[i] for i in free_ids})
            row = {"n_labelled": n, "seed": seed, **info, "train_seconds": round(train_s, 1), **q}
            rows.append(row)
            print(f"N={n:3d} seed={seed}: xyz R2 x {q['x_r2']:.3f} y {q['y_r2']:.3f} z {q['z_r2']:.3f} | xyz MAE "
                  f"{q['xyz_mae']:.4f} (zero-action {q['xyz_zero_baseline_mae']:.4f}) | gripper acc "
                  f"{q['gripper_acc']:.3f} (majority {q['gripper_majority_baseline_acc']:.3f}) | {train_s:.0f}s")

    with open(args.out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"h5": args.h5, "h5_sha256": sha256_file(args.h5), "device": str(device), "steps": args.steps,
               "offsets": args.offsets,
               "labelled_pool": [LABELLED_POOL.start, LABELLED_POOL.stop - 1],
               "action_free_pool": [free_ids[0], free_ids[-1]], "action_free_transitions": int(len(true_free)),
               "by_n": {}}
    for n in args.ns:
        sub = [r for r in rows if r["n_labelled"] == n]
        summary["by_n"][str(n)] = {k: {"mean": float(np.mean([r[k] for r in sub])), "std": float(np.std([r[k] for r in sub])),
                                       "n_seeds": len(sub)}
                                   for k in ("x_r2", "y_r2", "z_r2", "xyz_mae", "gripper_acc")}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out / 'metrics.csv'}, {args.out / 'summary.json'}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
