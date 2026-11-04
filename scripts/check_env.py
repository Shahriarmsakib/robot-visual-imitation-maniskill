"""Audit the runtime and dataset, and write a reproducibility manifest (milestone M0).

Usage:
    python scripts/check_env.py [--render-test] [--out experiments/manifests/env_<name>.json]

Exits non-zero if a required package is missing, the dataset fails its checks,
or the optional render test fails.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {
    "repo": "https://github.com/haosulab/ManiSkill",
    "tag": "v3.0.1",
    "commit": "a4a4f9272ad64b1564035874b605ceb687b63ed8",
    "act_baseline_path": "examples/baselines/act",
}
DATASET = ROOT / "data/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
PACKAGES = ["mani_skill", "sapien", "torch", "torchvision", "gymnasium", "h5py", "numpy", "pin", "mplib"]


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info():
    import torch

    info = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info["cuda_device"] = props.name
        info["cuda_memory_gb"] = round(props.total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    info["recommended_device"] = "cuda" if info["cuda"] else ("mps" if info["mps"] else "cpu")
    info["recommended_sim_backend"] = "physx_cuda" if info["cuda"] else "physx_cpu"
    return info


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return {"commit": out.stdout.strip(), "dirty": bool(dirty.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def dataset_info(path):
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False, "ok": False}
    import h5py

    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    problems, samples = [], 0
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            actions = f[k]["actions"]
            rgb = f[k]["obs"]["sensor_data"]["base_camera"]["rgb"]
            samples += actions.shape[0]
            if actions.shape[1] != 4 or rgb.shape != (actions.shape[0] + 1, 128, 128, 3):
                problems.append(f"{k}: actions {actions.shape}, rgb {rgb.shape}")
            if not bool(f[k]["success"][-1]):
                problems.append(f"{k}: not successful at end")
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "sha256": sha.hexdigest(),
        "episodes": len(keys),
        "samples": samples,
        "problems": problems,
        "ok": not problems,
    }


def render_test():
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers environments

    env = gym.make(
        "PickCube-v1", obs_mode="rgb", control_mode="pd_ee_delta_pos",
        sim_backend="physx_cpu", max_episode_steps=100,
    )
    obs, _ = env.reset(seed=0)
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    start = time.time()
    for _ in range(100):
        env.step(env.action_space.sample())
    elapsed = time.time() - start
    env.close()
    ok = tuple(rgb.shape) == (1, 128, 128, 3) and float(rgb.float().std()) > 1.0
    return {"ok": ok, "rgb_shape": list(rgb.shape), "seconds_per_100_steps": round(elapsed, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--render-test", action="store_true", help="also step PickCube-v1 with RGB cameras")
    parser.add_argument("--out", type=Path, default=None, help="manifest path (default: experiments/manifests/env_<os>.json)")
    args = parser.parse_args()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "device": device_info(),
        "repo": git_commit(),
        "upstream": UPSTREAM,
        "dataset": dataset_info(DATASET),
    }
    if args.render_test:
        try:
            manifest["render_test"] = render_test()
        except Exception as exc:  # report the failure in the manifest instead of a bare traceback
            manifest["render_test"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out = args.out or ROOT / "experiments/manifests" / f"env_{platform.system().lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")

    missing = [p for p in ("mani_skill", "sapien", "torch") if manifest["packages"][p] is None]
    failed = []
    if missing:
        failed.append(f"missing packages: {missing}")
    if not manifest["dataset"]["ok"]:
        failed.append("dataset missing or invalid (run scripts/prepare_data.sh)")
    if args.render_test and not manifest["render_test"]["ok"]:
        failed.append("render test failed")
    if failed:
        print("FAILED: " + "; ".join(failed), file=sys.stderr)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
