"""Summaries and figures for the Phase A failure analysis (run after scripts/failure_analysis.py).

Usage:
    python scripts/summarize_failure_analysis.py [--dir results/phase_a_failure_analysis] [--primary step30000]

Reads metrics/episodes_<label>.csv and timeseries_<label>.npz for every analysed checkpoint and writes
metrics/episode_metrics.csv (all checkpoints), metrics/summary.json and figures/*.png|pdf.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.evaluation.diagnostics import CATEGORIES, CATEGORY_LABEL, Thresholds  # noqa: E402
from src.evaluation.failure_summary import STAGES, mcnemar_exact, summarize  # noqa: E402
from src.visualization import failure_plots  # noqa: E402

VIDEO_TITLES = {"successful_rollout": "Successful grasp and placement", "failed_grasp": "Failure to grasp",
                "dropped_cube": "Cube dropped during transport", "lost_success": "Reached target, success not held"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("results/phase_a_failure_analysis"))
    parser.add_argument("--primary", default="step30000", help="checkpoint label used for videos and time-series figure")
    args = parser.parse_args()
    m = args.dir / "metrics"

    labels = sorted(p.stem.removeprefix("episodes_") for p in m.glob("episodes_*.csv"))
    labels.sort(key=lambda l: l != args.primary)  # primary first: fixed colour slot 1
    if not labels:
        sys.exit(f"no episodes_*.csv in {m}; run scripts/failure_analysis.py first")
    rows = {l: list(csv.DictReader(open(m / f"episodes_{l}.csv"))) for l in labels}
    ts = {l: dict(np.load(m / f"timeseries_{l}.npz")) for l in labels}
    runs = {l: json.loads((m / f"run_{l}.json").read_text()) for l in labels}

    # Scenes must be identical across checkpoints for paired comparisons.
    base = rows[labels[0]]
    same_scenes = all(
        len(rows[l]) == len(base) and all(a["goal_z"] == b["goal_z"] and a["cube_goal_xy_dist0"] == b["cube_goal_xy_dist0"]
                                          for a, b in zip(base, rows[l])) for l in labels)

    summaries = {l: summarize(rows[l], ts[l]) for l in labels}
    summary = {
        "description": "Phase A failure analysis of the ACT RGB baseline on PickCube-v1 (simulation only).",
        "checkpoints": {l: {k: runs[l][k] for k in ("checkpoint_step", "checkpoint_sha256", "run", "sim_backend",
                                                       "scene_seed", "num_envs", "episode_id_rule", "checks")}
                        for l in labels},
        "same_scenes_across_checkpoints": same_scenes,
        "thresholds": runs[labels[0]]["thresholds"],
        "category_definitions": {k: CATEGORY_LABEL[k] for k, _ in CATEGORIES},
        "results": summaries,
    }
    if len(labels) >= 2 and same_scenes:
        a, b = labels[0], labels[1]
        summary["paired_final_success"] = {"first": a, "second": b, **mcnemar_exact(
            [r["final_success"] == "True" for r in rows[a]], [r["final_success"] == "True" for r in rows[b]])}
    (m / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with open(m / "episode_metrics.csv", "w", newline="") as fh:
        fields = list(rows[labels[0]][0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for l in labels:
            writer.writerows(rows[l])

    figs = args.dir / "figures"
    paths = []
    paths += failure_plots.success_breakdown(summaries, STAGES, figs)
    paths += failure_plots.failure_categories(summaries, [c for c in CATEGORIES if any(
        s["categories"][c[0]]["count"] for s in summaries.values()) or c[0] != "other"], figs)
    paths += failure_plots.success_by_goal_height(summaries, figs)
    paths += failure_plots.distance_over_time(ts[args.primary], rows[args.primary], None, figs)

    examples_path = m / f"examples_{args.primary}.json"
    if examples_path.exists():
        examples = json.loads(examples_path.read_text())
        frames = []
        for name, title in VIDEO_TITLES.items():
            npy = m / f"_frames_{args.primary}_{name}.npy"
            if examples.get(name) and npy.exists():
                frames.append((f"{title}: episode {examples[name]['episode']}", np.load(npy)))
        if frames:
            paths.append(failure_plots.rollout_examples(
                frames, figs / "rollout_examples.png",
                title=f"Representative rollouts, {args.primary} (first episode of each category; physx_cpu, scene seed "
                      f"{runs[args.primary]['scene_seed']})"))
    print("checkpoints:", labels, "| same scenes:", same_scenes)
    print("wrote", m / "summary.json", m / "episode_metrics.csv")
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
