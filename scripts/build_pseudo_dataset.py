"""Build ACT training sets for the action-free study, in the same h5/json format as ManiSkill's converted demos.

Usage:
    python scripts/build_pseudo_dataset.py --split results/.../splits/N25_seed0.json --out data/action_free/sets/A_N25_seed0.h5
    python scripts/build_pseudo_dataset.py --split ... --pseudo data/action_free/pseudo_labels/idm_N25_seed0.npz \
        --out data/action_free/sets/B_N25_seed0.h5

Without --pseudo: only the N labelled demos, with their true actions (arm A).
With --pseudo: the N labelled demos (true actions) plus every action-free demo in the npz, whose actions are
replaced by the IDM pseudo-labels (arm B). Observations are copied unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.action_free.data import DEFAULT_H5  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default=DEFAULT_H5)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--pseudo", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    split = json.loads(args.split.read_text())
    labelled = [f"traj_{i}" for i in split["labelled_ids"]]
    pseudo = dict(np.load(args.pseudo)) if args.pseudo else {}
    src_json = json.loads(Path(args.source[:-2] + "json").read_text())
    meta_by_key = {f"traj_{e['episode_id']}": e for e in src_json["episodes"]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    episodes_meta = []
    with h5py.File(args.source, "r") as src, h5py.File(args.out, "w") as dst:
        for key in labelled + sorted(pseudo, key=lambda k: int(k.split("_")[1])):
            src.copy(src[key], dst, name=key)
            g = dst[key]
            if key in pseudo:
                assert key not in labelled, f"{key} is both labelled and pseudo-labelled"
                assert pseudo[key].shape == g["actions"].shape, (key, pseudo[key].shape, g["actions"].shape)
                del g["actions"]
                g.create_dataset("actions", data=pseudo[key].astype(np.float32))
                g.attrs["label_source"] = "idm"
            else:
                g.attrs["label_source"] = "true"
            episodes_meta.append(meta_by_key[key])
        dst.attrs["labelled_ids"] = json.dumps(split["labelled_ids"])
        dst.attrs["pseudo_labels"] = str(args.pseudo) if args.pseudo else ""
    out_json = {k: v for k, v in src_json.items() if k != "episodes"}
    out_json["episodes"] = episodes_meta
    args.out.with_suffix(".json").write_text(json.dumps(out_json) + "\n")
    print(f"wrote {args.out}: {len(labelled)} labelled (true actions) + {len(pseudo)} pseudo-labelled episodes")


if __name__ == "__main__":
    main()
