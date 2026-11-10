"""Check that ACTDemoDataset's resize-on-access gives exactly the frames of the old up-front resize.

Usage:
    python scripts/check_lazy_resize.py --demo-path data/action_free/sets/<set>.h5 [--num-demos 50]

For every sample of the first --num-demos episodes, compares the 224x224 uint8 image returned by the dataset
with the old method (all frames of an episode resized together when loading), plus the state and action chunk
against the h5. Prints PASS / FAIL and the number of differing pixels; exits 1 on FAIL.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from h5py import File  # noqa: E402

from src.data.demo_dataset import ACTDemoDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--demo-path", required=True)
    p.add_argument("--num-demos", type=int, default=50)
    p.add_argument("--control-mode", default="pd_ee_delta_pos")
    p.add_argument("--num-queries", type=int, default=30)
    args = p.parse_args()

    ds = ACTDemoDataset(args.demo_path, args.num_queries, args.control_mode, num_traj=args.num_demos)
    resize = T.Compose([T.Resize((224, 224), antialias=True)])  # the old loader's transform
    old = {}
    with File(args.demo_path, "r") as f:
        for i, k in enumerate(ds.keys):
            cams = [torch.from_numpy(f[k]["obs/sensor_data"][c]["rgb"][:]) for c in f[k]["obs/sensor_data"]]
            old[i] = torch.stack([resize(c.permute(0, 3, 1, 2)) for c in cams], dim=1)
            assert torch.equal(ds.trajectories["actions"][i], torch.Tensor(f[k]["actions"][:]))

    bad_samples = bad_pixels = 0
    for idx in range(len(ds)):
        traj, ts = ds.slices[idx]
        new = ds[idx]["observations"]["rgb"]
        ref = old[traj][ts]
        if new.dtype != ref.dtype or new.shape != ref.shape or not torch.equal(new, ref):
            bad_samples += 1
            bad_pixels += int((new != ref).sum()) if new.shape == ref.shape else ref.numel()
    print(f"{len(ds)} samples from {ds.num_traj} episodes | frame shape {tuple(ref.shape)} {ref.dtype} | "
          f"stored frame {tuple(ds.trajectories['observations'][0]['rgb'].shape[1:])} | torch {torch.__version__}")
    print(f"differing samples {bad_samples}, differing pixel values {bad_pixels}")
    print("PASS" if bad_samples == 0 else "FAIL")
    sys.exit(0 if bad_samples == 0 else 1)


if __name__ == "__main__":
    main()
