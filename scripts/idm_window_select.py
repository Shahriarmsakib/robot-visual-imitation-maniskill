"""Choose the IDM frame window on a validation split inside the labelled pool (never on the action-free test pool).

Trains on 50 labelled-pool demos and validates on the other 50, for each candidate window.
Usage: python scripts/idm_window_select.py [--steps 3000]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CANDIDATES = [(0, 1), (0, 1, 2), (-1, 0, 1, 2)]


def main():
    import numpy as np

    from src.action_free.data import DEFAULT_H5, LABELLED_POOL, labelled_split, load_episodes
    from src.action_free.idm import label_quality, train_idm
    from src.utils import select_device

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--h5", default=DEFAULT_H5)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", type=Path, default=Path("results/action_free/idm/window_selection.json"))
    args = p.parse_args()
    device = select_device("auto")
    pool = load_episodes(args.h5, LABELLED_POOL)
    train_ids = labelled_split(50, seed=999)
    val_ids = sorted(set(LABELLED_POOL) - set(train_ids))
    val_true = np.concatenate([pool[i][1] for i in val_ids])
    results = []
    for offsets in CANDIDATES:
        t0 = time.time()
        model, _ = train_idm([pool[i] for i in train_ids], device, steps=args.steps, seed=0, offsets=offsets)
        q = label_quality(np.concatenate([model.predict(pool[i][0], offsets) for i in val_ids]), val_true)
        results.append({"offsets": list(offsets), **q})
        print(f"window {offsets}: R2 x {q['x_r2']:.3f} y {q['y_r2']:.3f} z {q['z_r2']:.3f} | xyz MAE {q['xyz_mae']:.4f} "
              f"(zero {q['xyz_zero_baseline_mae']:.4f}) | gripper {q['gripper_acc']:.3f} | {time.time() - t0:.0f}s")
    best = max(results, key=lambda r: np.mean([r["x_r2"], r["y_r2"], r["z_r2"]]) + r["gripper_acc"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rule": "max(mean xyz R2 + gripper acc) on validation", "train_ids": train_ids,
                                    "val_ids": val_ids, "results": results, "chosen": best["offsets"]}, indent=2) + "\n")
    print(f"chosen window: {best['offsets']} -> {args.out}")


if __name__ == "__main__":
    main()
