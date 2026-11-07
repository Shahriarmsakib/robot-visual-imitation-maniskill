"""Phase 2A: ACT-compatible object-centric dataset (first 5 PickCube demos, pd_ee_delta_pos, RGB-D).

Usage:
    python scripts/build_objectcentric_ee_delta.py [--num-demos 5]

1. Copies the official demos to a work folder and converts them with ManiSkill's replay tool
   (python -m mani_skill.trajectory.replay_trajectory -c pd_ee_delta_pos -o rgbd -b physx_cpu
   --use-first-env-state). The source file is never written; its hash is checked before and after.
2. Replays each converted demo from its recorded initial state with the converted 4-D actions and records
   RGB, depth, cube/TCP poses, goal, joint state, camera parameters and success from the simulator.
3. Validates the result (7 checks) and writes the HDF5 dataset, a figure, summary.json and stdout report.
Ends with "DATASET VALIDATION PASSED" only if every check passes (exit code 1 otherwise). No training.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = "data/demos/PickCube-v1/motionplanning/trajectory.h5"
TARGET_CONTROL = "pd_ee_delta_pos"


def convert(source, work_dir, num_demos):
    """Run ManiSkill's official converter on a copy of the source demos. Returns (converted h5 path, command, log)."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    shutil.copy2(source, work_dir / source.name)
    shutil.copy2(source.with_suffix(".json"), work_dir / source.with_suffix(".json").name)
    cmd = [sys.executable, "-m", "mani_skill.trajectory.replay_trajectory", "--traj-path", str(work_dir / source.name),
           "--use-first-env-state", "-c", TARGET_CONTROL, "-o", "rgbd", "--save-traj", "-b", "physx_cpu",
           "--count", str(num_demos)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = [l for l in (proc.stdout + proc.stderr).splitlines() if "Replayed" in l or "Error" in l]
    out = work_dir / f"{source.stem}.rgbd.{TARGET_CONTROL}.physx_cpu.h5"
    if proc.returncode != 0 or not out.exists():
        sys.exit(f"Conversion failed (exit {proc.returncode}): {log or proc.stderr[-2000:]}")
    return out, " ".join(["python"] + cmd[1:]), log


def main():
    import gymnasium as gym
    import mani_skill
    import mani_skill.envs  # noqa: F401  registers environments

    from src.data.object_centric import (load_source_episodes, replay_episode, save_hdf5, shape_table,
                                         validate_ee_delta)
    from src.utils import sha256_file
    from src.visualization.object_centric_plots import frames_figure

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-demos", type=int, default=5)
    parser.add_argument("--source", type=Path, default=Path(SOURCE))
    parser.add_argument("--work-dir", type=Path, default=Path("data/object_centric/conversion"))
    parser.add_argument("--out", type=Path, default=Path("data/object_centric/pickcube_5demos_objectcentric_ee_delta.h5"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/object_centric_ee_delta"))
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"source demos not found: {args.source} (run scripts/prepare_data.sh first)")
    source_sha = sha256_file(args.source)

    converted, command, log = convert(args.source, args.work_dir, args.num_demos)
    print("Conversion:", *log, sep="\n  ")
    conv_meta = json.loads(converted.with_suffix(".json").read_text())
    converted_ids = [e["episode_id"] for e in conv_meta["episodes"]]
    _, conv_eps = load_source_episodes(converted, len(converted_ids))
    _, joint_eps = load_source_episodes(args.source, args.num_demos)

    env = gym.make("PickCube-v1", obs_mode="rgbd", control_mode=TARGET_CONTROL, sim_backend="physx_cpu",
                   render_mode="rgb_array")
    action_space = env.action_space
    episodes = [replay_episode(env, ep) for ep in conv_eps]
    env.close()
    joint_env = gym.make("PickCube-v1", obs_mode="rgbd", control_mode="pd_joint_pos", sim_backend="physx_cpu",
                         render_mode="rgb_array")
    joint_replays = [replay_episode(joint_env, ep) for ep in joint_eps]
    joint_env.close()
    if sha256_file(args.source) != source_sha:
        sys.exit("source demonstration file changed during conversion; aborting")

    for e, j in zip(episodes, joint_replays):
        print(f"  {e['key']} (seed {e['seed']}): converted T = {len(e['actions'])} (original {len(j['actions'])}), "
              f"success at end: replay {bool(e['success'][-1])}, converted file {bool(e['success_recorded'][-1])}")

    print("\nArray shapes per episode (T+1 observations, T actions):")
    for i, row in enumerate(shape_table(episodes)):
        print(f"  ep_{i:03d}: " + ", ".join(f"{k} {tuple(v)}" for k, v in row.items()))

    checks = validate_ee_delta(episodes, joint_replays, action_space, args.num_demos, converted_ids)
    print("\nValidation:")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    env_info = {"env_id": "PickCube-v1", "obs_mode": "rgbd", "control_mode": TARGET_CONTROL, "sim_backend": "physx_cpu",
                "mani_skill_version": mani_skill.__version__, "camera": "base_camera",
                "conversion_command": command, "converted_file_sha256": sha256_file(converted),
                "action_layout": "[dx, dy, dz, gripper]: end-effector position delta (normalised to [-1, 1]) + gripper (+1 open, -1 close)"}
    episode_attrs = [{"original_T": len(j["actions"]), "conversion_success": True,
                      "replay_success": bool(e["success"][-1])} for e, j in zip(episodes, joint_replays)]
    h5_path = save_hdf5(args.out, episodes, args.source, source_sha, env_info,
                        description="ManiSkill PickCube-v1 expert demos converted to pd_ee_delta_pos (ACT action space), "
                                    "replayed with RGB-D and simulator ground truth",
                        episode_attrs=episode_attrs)
    fig = frames_figure(episodes[0], args.report_dir, name="ee_delta_frames", episode_label="ep_000 (pd_ee_delta_pos)")

    summary = {
        "dataset": str(h5_path), "dataset_size_mb": round(h5_path.stat().st_size / 1e6, 2),
        "regenerate": "python scripts/build_objectcentric_ee_delta.py --num-demos 5",
        "source": str(args.source), "source_sha256": source_sha, "source_unchanged": True, **env_info,
        "episodes": [{"name": f"ep_{i:03d}", "source_episode": e["key"], "seed": int(e["seed"]),
                      "T": len(e["actions"]), "original_T": len(j["actions"]),
                      "action_shape": list(e["actions"].shape), "conversion_success": e["key"] in
                      {f"traj_{k}" for k in converted_ids}, "replay_success": bool(e["success"][-1]), "shapes": s}
                     for i, (e, j, s) in enumerate(zip(episodes, joint_replays, shape_table(episodes)))],
        "checks": [{"check": n, "passed": ok, "detail": d} for n, ok, d in checks],
        "all_passed": all(ok for _, ok, _ in checks),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {h5_path} ({summary['dataset_size_mb']} MB), {fig[0]}, {args.report_dir / 'summary.json'}")

    failed = [(n, d) for n, ok, d in checks if not ok]
    if failed:
        print("\nDATASET VALIDATION FAILED:")
        for n, d in failed:
            print(f"  {n}: {d}")
        sys.exit(1)
    print("\nDATASET VALIDATION PASSED")


if __name__ == "__main__":
    main()
