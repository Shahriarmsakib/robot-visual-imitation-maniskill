"""Pipeline check for the pseudo-label arm (arm B), no GPU needed.

Usage:
    python scripts/make_oracle_labels.py --split <splits/N100_seed0.json> --workdir <tmp dir> \
        [--compare data/action_free/pseudo_labels/idm_N100_seed0.npz]

1. Writes "oracle pseudo-labels": the TRUE actions of the action-free demos, in the IDM's npz format.
2. Builds arm B from them with scripts/build_pseudo_dataset.py (the same code path the real arm B uses).
3. Checks that the result, read in the training loader's episode order, is identical in every field to the
   source demo file. If it is, arm B with perfect labels trains on exactly the upper-bound data, so any gap
   between arm B and the upper bound comes from the pseudo-labels, not from the data pipeline.
With --compare, also prints per-dimension means of the true actions and of an IDM label file (label bias).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_free.data import ACTION_FREE_POOL, DEFAULT_H5  # noqa: E402


def _same(a, b):
    if isinstance(a, h5py.Dataset):
        return isinstance(b, h5py.Dataset) and np.array_equal(a[:], b[:])
    return set(a.keys()) == set(b.keys()) and all(_same(a[k], b[k]) for k in a.keys())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=DEFAULT_H5)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True, help="scratch folder for the oracle npz and dataset")
    p.add_argument("--compare", type=Path, default=None, help="IDM pseudo-label npz to compare means with")
    args = p.parse_args()

    from src.data.demo_dataset import episode_keys

    args.workdir.mkdir(parents=True, exist_ok=True)
    ids = list(ACTION_FREE_POOL)
    with h5py.File(args.source, "r") as f:
        labels = {f"traj_{i}": f[f"traj_{i}"]["actions"][:].astype(np.float32) for i in ids}
    oracle = args.workdir / "oracle.npz"
    np.savez_compressed(oracle, **labels)
    true = np.concatenate(list(labels.values()))
    print(f"oracle labels: true actions of {len(labels)} action-free demos ({len(true)} transitions)")
    print("  mean x/y/z, true actions:", np.round(true[:, :3].mean(0), 4).tolist())
    if args.compare:
        idm = np.load(args.compare)
        pred = np.concatenate([idm[k] for k in labels])
        print(f"  mean x/y/z, {args.compare.name}:", np.round(pred[:, :3].mean(0), 4).tolist())

    built = args.workdir / "B_oracle.h5"
    subprocess.run([sys.executable, "scripts/build_pseudo_dataset.py", "--source", args.source, "--split", str(args.split),
                    "--pseudo", str(oracle), "--out", str(built)], check=True)
    keys_built = episode_keys(built)
    with h5py.File(args.source, "r") as src, h5py.File(built, "r") as dst:
        keys_src = episode_keys(args.source, len(keys_built))
        diff = [k for k in keys_built if not _same(src[k], dst[k])]
    ok = keys_src == keys_built and not diff
    print(f"loader episode order identical: {keys_src == keys_built} ({len(keys_built)} episodes); "
          f"episodes differing in any field: {len(diff)} {diff[:5]}")
    print("PASS: arm B with true labels = upper-bound training data" if ok else "FAIL: arm-B pipeline changes the data")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
