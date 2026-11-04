"""Publication figures generated only from logged runs (PNG + PDF).

Missing data is drawn as a labelled "pending" panel, never as placeholder values.
Colours: the reference categorical palette in fixed slot order (blue, orange, aqua, yellow);
text stays in neutral inks; grid and axes are recessive.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.utils import wilson_interval  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
METRIC_COLOR = {"success_at_end": SLOTS[0], "success_once": SLOTS[1]}
METRIC_LABEL = {"success_at_end": "success at end", "success_once": "success once"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": INK_2, "lines.linewidth": 2.0,
        "lines.markersize": 6, "pdf.fonttype": 42,
    })


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def pending(ax, message):
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10,
            wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _kilo(x, _pos=None):
    return f"{x / 1000:g}k" if abs(x) >= 1000 else f"{x:g}"


def _run_label(run):
    return f"{run.num_demos} demos, seed {run.seed}"


def _status_note(run):
    return "" if run.done else f" (incomplete: {run.steps_done}/{run.total_steps})"


def plot_training_loss(runs, out_dir, name="fig2_training_loss"):
    """Training loss vs step; one line per run, colour per demo budget, seeds share the colour."""
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    runs = [r for r in runs if r.train]
    if not runs:
        pending(ax, "Pending: no training logs yet")
    else:
        budgets = sorted({r.num_demos for r in runs}, key=lambda d: d or 0)
        color = {d: SLOTS[i % len(SLOTS)] for i, d in enumerate(budgets)}
        labelled = set()
        for run in sorted(runs, key=lambda r: (r.num_demos or 0, r.seed)):
            steps = [r["step"] for r in run.train]
            loss = [r["loss"] for r in run.train]
            label = None if run.num_demos in labelled else f"{run.num_demos} demos"
            labelled.add(run.num_demos)
            ax.plot(steps, loss, color=color[run.num_demos], lw=1.6, alpha=0.9, label=label)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_kilo))
        ax.set_xlabel("training step (optimizer updates)")
        ax.set_ylabel("training loss (L1 + 10·KL, log scale)")
        if len(budgets) > 1:
            ax.legend(loc="upper right")
        seeds = sorted({r.seed for r in runs})
        seed_text = f"seed{'s' * (len(seeds) > 1)} {', '.join(map(str, seeds))}"
        ax.set_title(f"Training loss ({len(runs)} run{'s' * (len(runs) > 1)}; {seed_text})", loc="left")
        ax.text(0, -0.22, "Loss is not task success; see the success curves.", transform=ax.transAxes,
                color=MUTED, fontsize=8)
    return save(fig, out_dir, name)


def plot_success_curves(runs, out_dir, name="fig2b_success_vs_step"):
    """Success vs training step, one panel per run; shaded band = 95% Wilson interval."""
    style()
    runs = [r for r in runs if r.evals]
    if not runs:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        pending(ax, "Pending: no evaluations logged yet")
        return save(fig, out_dir, name)
    runs = sorted(runs, key=lambda r: (r.variant, r.num_demos or 0, r.seed))
    cols = min(3, len(runs))
    rows = int(np.ceil(len(runs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols + 0.8, 2.9 * rows + 0.6), squeeze=False,
                             sharey=True)
    for ax, run in zip(axes.flat, runs):
        last_rate = {m: 100 * run.evals[-1][m]["rate"] for m in METRIC_COLOR}
        # Nudge the two end labels apart when the lines finish close together.
        close = abs(last_rate["success_at_end"] - last_rate["success_once"]) < 7
        nudge = {"success_at_end": -5 if close else 0, "success_once": 5 if close else 0}
        if close and last_rate["success_at_end"] > last_rate["success_once"]:
            nudge = {k: -v for k, v in nudge.items()}
        for metric in ("success_at_end", "success_once"):
            steps = np.array([e["step"] for e in run.evals])
            counts = np.array([e[metric]["count"] for e in run.evals])
            ns = np.array([e[metric]["n"] for e in run.evals])
            rate = 100 * counts / ns
            lo, hi = zip(*(wilson_interval(c, n) for c, n in zip(counts, ns)))
            c = METRIC_COLOR[metric]
            ax.fill_between(steps, 100 * np.array(lo), 100 * np.array(hi), color=c, alpha=0.12, lw=0)
            ax.plot(steps, rate, color=c, marker="o", ms=5, label=METRIC_LABEL[metric])
            ax.annotate(f"{counts[-1]}/{ns[-1]}", (steps[-1], rate[-1]), xytext=(5, nudge[metric]),
                        textcoords="offset points", fontsize=8, color=INK_2, va="center")
        ax.set_title(_run_label(run) + _status_note(run), loc="left", fontsize=9.5)
        ax.set_xlabel("training step")
        # Always span the planned run length, so an unfinished run visibly stops short.
        total = run.total_steps or max(e["step"] for e in run.evals)
        ax.set_xlim(-0.03 * total, 1.12 * total)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_kilo))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
        ax.set_ylim(-2, 102)
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("success rate (%)")
    axes.flat[0].legend(loc="upper left", fontsize=8)
    n_trials = sorted({e["n_trials"] for r in runs for e in r.evals})
    fig.suptitle(f"Task success during training (EMA policy, {'/'.join(map(str, n_trials))} trials per point, "
                 "shaded: 95% CI)", x=0.01, ha="left", fontsize=10.5, color=INK, fontweight="bold")
    fig.tight_layout()
    return save(fig, out_dir, name)


def plot_data_efficiency(table, out_dir, name="fig3_data_efficiency", budgets=(10, 25, 50, 100)):
    """Final-model success vs demonstration budget. Error bars: seed std (>=2 seeds) or Wilson 95% CI (1 seed)."""
    style()
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x_of = {b: i for i, b in enumerate(budgets)}
    ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("number of expert demonstrations")
    ax.set_ylabel("final success rate (%)")
    plotted = False
    for k, metric in enumerate(("success_at_end", "success_once")):
        rows = [r for r in table if r["metric"] == metric and r["num_demos"] in x_of]
        if not rows:
            continue
        plotted = True
        offset = (k - 0.5) * 0.12
        xs = np.array([x_of[r["num_demos"]] for r in rows]) + offset
        mean = np.array([100 * r["mean_rate"] for r in rows])
        err_lo, err_hi = [], []
        for r, m in zip(rows, mean):
            if r["n_seeds"] > 1:
                err_lo.append(100 * r["std_rate"]); err_hi.append(100 * r["std_rate"])
            else:
                lo, hi = r["pooled_ci95"]
                err_lo.append(m - 100 * lo); err_hi.append(100 * hi - m)
        c = METRIC_COLOR[metric]
        ax.errorbar(xs, mean, yerr=[err_lo, err_hi], color=c, marker="o", ms=7, capsize=3, lw=2,
                    label=METRIC_LABEL[metric])
        for r, x in zip(rows, xs):
            if r["n_seeds"] > 1:
                ax.scatter([x] * r["n_seeds"], [100 * v for v in r["seed_rates"]], s=14, color=c, alpha=0.45,
                           zorder=1, lw=0)
            if metric == "success_at_end":
                ax.annotate(f"{r['n_seeds']} seed{'s' * (r['n_seeds'] > 1)}", (x, -2), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
    missing = [b for b in budgets if not any(r["num_demos"] == b for r in table)]
    if not plotted:
        ax.text(0.5, 0.5, "Pending: no completed runs yet", transform=ax.transAxes, ha="center", va="center",
                color=MUTED)
    else:
        ax.legend(loc="upper left")
        if missing:
            ax.text(0.99, 0.02, "pending: " + ", ".join(map(str, missing)) + " demos", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_title("Data efficiency: success of the final model vs demo budget", loc="left")
    ax.text(0, -0.24, "Error bars: std across seeds (≥2 seeds) or 95% Wilson CI (1 seed). Dots: individual seeds.",
            transform=ax.transAxes, color=MUTED, fontsize=8)
    return save(fig, out_dir, name)
