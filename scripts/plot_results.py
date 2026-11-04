"""Regenerate figures from logged runs.

Usage:
    python scripts/plot_results.py RUNS_DIR [RUNS_DIR ...] [--out results/figures] [--include-smoke]

Writes fig2_training_loss, fig2b_success_vs_step and fig3_data_efficiency as PNG and PDF.
Panels with no data are drawn as labelled "pending" placeholders.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results import data_efficiency_table, find_runs  # noqa: E402
from src.visualization.plots import plot_data_efficiency, plot_success_curves, plot_training_loss  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--include-smoke", action="store_true", help="also include runs named *smoke*")
    args = parser.parse_args()

    runs = find_runs(args.roots, include_smoke=args.include_smoke)
    print(f"{len(runs)} run(s): " + ", ".join(r.run_id for r in runs))
    table = data_efficiency_table(runs, "success_at_end") + data_efficiency_table(runs, "success_once")
    for paths in (plot_training_loss(runs, args.out), plot_success_curves(runs, args.out),
                  plot_data_efficiency(table, args.out)):
        print("wrote", ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
