"""Collect run logs into results/metrics/ (CSV + Markdown table).

Usage:
    python scripts/collect_results.py RUNS_DIR [RUNS_DIR ...] [--out results/metrics] [--include-smoke]

Point it at directories holding run folders (for example the downloaded Kaggle
outputs). Writes runs.csv (one row per run), evals.csv (every evaluation),
data_efficiency.csv and results_table.md.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results import data_efficiency_table, find_runs, run_table  # noqa: E402


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def pct(rate):
    return "" if rate is None else f"{100 * rate:.0f}%"


def markdown_table(rows):
    lines = [
        "| Run | Demos | Seed | Steps | Status | Final success_at_end | Final success_once | 95% CI (at_end) | Best at_end (step) | Backend | Commit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["final_n_trials"]
        at_end = f"{r['final_success_at_end']}/{n} ({pct(r['final_success_at_end_rate'])})" if n else "pending"
        once = f"{r['final_success_once']}/{n} ({pct(r['final_success_once_rate'])})" if n else "pending"
        ci = r["final_success_at_end_ci95"]
        ci = f"{pct(ci[0])}–{pct(ci[1])}" if ci else ""
        best = f"{pct(r['best_success_at_end_rate'])} ({r['best_success_at_end_step']})" if r["best_success_at_end_step"] is not None else ""
        lines.append(f"| {r['run_id']} | {r['num_demos']} | {r['seed']} | {r['steps_done']}/{r['total_steps']} | "
                     f"{r['status']} | {at_end} | {once} | {ci} | {best} | {r['sim_backend']} | {r['git_commit']} |")
    lines.append("")
    lines.append("Final = evaluation of the final model of a completed run (headline number). "
                 "Best = highest success_at_end over all evaluations; it is selected on the evaluation scenes "
                 "and therefore optimistic.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/metrics"))
    parser.add_argument("--include-smoke", action="store_true", help="also include runs named *smoke*")
    args = parser.parse_args()

    runs = find_runs(args.roots, include_smoke=args.include_smoke)
    if not runs:
        sys.exit(f"No runs found under {[str(r) for r in args.roots]}")
    table = run_table(runs)
    evals = [{"run_id": run.run_id, "variant": run.variant, "num_demos": run.num_demos, "seed": run.seed,
              "step": e["step"], "n_trials": e["n_trials"],
              "success_once": e["success_once"]["count"], "success_at_end": e["success_at_end"]["count"],
              "success_once_rate": e["success_once"]["rate"], "success_at_end_rate": e["success_at_end"]["rate"],
              "sim_backend": e.get("sim_backend"), "eval_seed": e.get("eval_seed")}
             for run in runs for e in run.evals]
    efficiency = data_efficiency_table(runs, "success_at_end") + data_efficiency_table(runs, "success_once")

    write_csv(args.out / "runs.csv", table)
    write_csv(args.out / "evals.csv", evals)
    write_csv(args.out / "data_efficiency.csv", efficiency)
    md = markdown_table(table)
    (args.out / "results_table.md").write_text(md)
    print(md)
    print(f"Wrote {len(table)} runs, {len(evals)} evaluations to {args.out}/")


if __name__ == "__main__":
    main()
