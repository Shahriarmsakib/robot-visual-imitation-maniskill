"""Object-centric data-extraction smoke test: replay 5 PickCube demos with RGB-D and simulator ground truth.

Usage:
    python scripts/object_centric_smoke_test.py [--num-demos 5]
        [--source data/demos/PickCube-v1/motionplanning/trajectory.h5]
        [--out data/object_centric/pickcube_5demos_rgbd.h5] [--report-dir results/object_centric_smoke]

Prints array shapes and validation results, writes the HDF5 dataset, a figure and validation.json,
and ends with "SMOKE TEST PASSED" only if every required check passes (exit code 1 otherwise).
Runs on CPU (physx_cpu); no training. The source demonstration file is only read.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SOURCE = "data/demos/PickCube-v1/motionplanning/trajectory.h5"


def main():
    import gymnasium as gym
    import mani_skill
    import mani_skill.envs  # noqa: F401  registers environments

    from src.data.object_centric import load_source_episodes, replay_episode, save_hdf5, shape_table, validate
    from src.utils import sha256_file
    from src.visualization.object_centric_plots import frames_figure

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--out", type=Path, default=Path("data/object_centric/pickcube_5demos_rgbd.h5"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/object_centric_smoke"))
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"source demos not found: {args.source} (run scripts/prepare_data.sh first)")
    source_sha = sha256_file(args.source)
    meta, sources = load_source_episodes(args.source, args.num_demos)
    env_kwargs = dict(obs_mode="rgbd", control_mode=sources[0]["control_mode"], sim_backend="physx_cpu",
                      render_mode="rgb_array")
    env = gym.make("PickCube-v1", **env_kwargs)
    print(f"Replaying {len(sources)} demos from {args.source} ({meta['source_type']}, control {env_kwargs['control_mode']})")
    episodes = []
    for ep in sources:
        e = replay_episode(env, ep)
        episodes.append(e)
        print(f"  {e['key']} (seed {e['seed']}): T = {len(e['actions'])} actions, "
              f"success at end: replay {bool(e['success'][-1])}, recorded {bool(e['success_recorded'][-1])}")
    env.close()

    print("\nArray shapes per episode (T+1 observations, T actions):")
    for i, row in enumerate(shape_table(episodes)):
        print(f"  ep_{i:03d}: " + ", ".join(f"{k} {tuple(v)}" for k, v in row.items()))
    print(f"  dtypes: rgb {episodes[0]['rgb'].dtype}, depth {episodes[0]['depth'].dtype} (mm), "
          f"actions {episodes[0]['actions'].dtype}")

    checks = validate(episodes, expected=args.num_demos)
    print("\nValidation:")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    env_info = {"env_id": "PickCube-v1", **env_kwargs, "mani_skill_version": mani_skill.__version__,
                "camera": "base_camera", "source_type": meta["source_type"]}
    h5_path = save_hdf5(args.out, episodes, args.source, source_sha, env_info)
    fig_paths = frames_figure(episodes[0], args.report_dir)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "source": str(args.source), "source_sha256": source_sha, "dataset": str(h5_path),
        "dataset_size_mb": round(h5_path.stat().st_size / 1e6, 2), **env_info,
        "episodes": [{"name": f"ep_{i:03d}", "source_episode": e["key"], "seed": int(e["seed"]),
                      "T": len(e["actions"]), "success": bool(e["success"][-1]), "shapes": s}
                     for i, (e, s) in enumerate(zip(episodes, shape_table(episodes)))],
        "checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in checks],
    }
    (args.report_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {h5_path} ({report['dataset_size_mb']} MB), {fig_paths[0]}, {args.report_dir / 'validation.json'}")

    failed = [(n, d) for n, ok, d in checks if not ok]
    if failed:
        print("\nSMOKE TEST FAILED:")
        for n, d in failed:
            print(f"  {n}: {d}")
        sys.exit(1)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
