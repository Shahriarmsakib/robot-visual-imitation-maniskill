"""Figures for the Phase A failure analysis, drawn only from measured episode data."""
import numpy as np
from PIL import Image, ImageDraw

from src.visualization.plots import INK, INK_2, MUTED, SLOTS, plt, save, style
from src.visualization.rollouts import _font

MAGENTA = "#e87ba4"  # categorical slot 5
CKPT_COLOR = {}  # label -> colour, assigned in fixed slot order by first appearance


def _color(label):
    if label not in CKPT_COLOR:
        palette = [*SLOTS, MAGENTA]  # 5 distinct colours before any repeat
        CKPT_COLOR[label] = palette[len(CKPT_COLOR) % len(palette)]
    return CKPT_COLOR[label]


def _grouped_bars(ax, groups, series, value_fn, horizontal=False):
    """groups: list of (key, label); series: list of (name, summary). Bars with 95% CI whiskers and count labels."""
    k = len(series)
    width = 0.8 / k
    pos = np.arange(len(groups))
    for j, (name, summ) in enumerate(series):
        entries = [value_fn(summ, key) for key, _ in groups]
        empty = np.array([not e["n"] for e in entries])  # no episodes in this group: no bar, labelled n=0
        vals = np.array([0.0 if z else 100 * e["rate"] for e, z in zip(entries, empty)])
        lo = np.where(empty, 0, vals - np.array([0.0 if z else 100 * e["ci95"][0] for e, z in zip(entries, empty)]))
        hi = np.where(empty, 0, np.array([0.0 if z else 100 * e["ci95"][1] for e, z in zip(entries, empty)]) - vals)
        lo, hi = np.clip(lo, 0, None), np.clip(hi, 0, None)  # a 0% or 100% rate can round to a tiny negative error
        off = pos + (j - (k - 1) / 2) * width
        if horizontal:
            ax.barh(off, vals, height=width * 0.92, color=_color(name), label=name, xerr=[lo, hi],
                    error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
            for y, e, v, h in zip(off, entries, vals, hi):
                ax.text(v + h + 1.2, y, f"{e['count']}/{e['n']}", va="center", fontsize=7.5, color=INK_2)
        else:
            ax.bar(off, vals, width=width * 0.92, color=_color(name), label=name, yerr=[lo, hi],
                   error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
            for x, e, v, h, z in zip(off, entries, vals, hi, empty):
                ax.text(x, v + h + 1.5, "n=0" if z else f"{e['count']}", ha="center", fontsize=7.5, color=INK_2)
    labels = [lbl for _, lbl in groups]
    if horizontal:
        ax.set_yticks(pos, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.grid(axis="y", visible=False)
    else:
        ax.set_xticks(pos, labels)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)


def success_breakdown(summaries, stages, out_dir, name="success_breakdown"):
    style()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    _grouped_bars(ax, [(k, lbl.replace(" ", "\n", 1)) for k, lbl, _ in stages], list(summaries.items()),
                  lambda s, k: s["stages"][k])
    ax.set_ylabel("episodes (%)")
    ax.legend(loc="upper right", title="checkpoint", title_fontsize=8, fontsize=8)
    n = next(iter(summaries.values()))["n_episodes"]
    ax.set_title(f"Where episodes drop out of the pick-and-place pipeline (n = {n} per checkpoint)", loc="left")
    ax.text(0, -0.25, "Each bar: share of episodes where the event happened at least once (last bar: at step 100). "
            "Whiskers: 95% Wilson CI; numbers: episode counts.", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    return save(fig, out_dir, name)


def failure_categories(summaries, categories, out_dir, name="failure_categories"):
    style()
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    _grouped_bars(ax, categories, list(summaries.items()), lambda s, k: s["categories"][k], horizontal=True)
    ax.set_xlabel("episodes (%)")
    ax.legend(loc="lower right", title="checkpoint", title_fontsize=8, fontsize=8)
    n = next(iter(summaries.values()))["n_episodes"]
    ax.set_title(f"Episode outcome categories (mutually exclusive, n = {n} per checkpoint)", loc="left")
    return save(fig, out_dir, name)


def distance_over_time(ts, rows, examples, out_dir, name="distance_over_time", goal_thresh=0.025):
    """Median cube-to-goal distance over time per category (IQR band), plus the example episodes."""
    style()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    steps = np.arange(1, ts["obj_to_goal"].shape[1] + 1)
    # Categories are different entities from checkpoints, so they avoid the checkpoint colours (slots 1-2).
    shown = [("success", "Success at end", SLOTS[2]), ("reached_target_lost_success", "Reached target, lost", SLOTS[3]),
             ("grasped_not_delivered", "Held, never reached target", MAGENTA)]
    for key, label, color in shown:
        idx = [i for i, r in enumerate(rows) if r["category"] == key]
        if not idx:
            continue
        d = 100 * ts["obj_to_goal"][idx]
        ax.fill_between(steps, np.percentile(d, 25, axis=0), np.percentile(d, 75, axis=0), color=color, alpha=0.15, lw=0)
        ax.plot(steps, np.median(d, axis=0), color=color, lw=2, label=f"{label} (n={len(idx)})")
    ax.axhline(100 * goal_thresh, color=INK_2, lw=1, ls="--")
    ax.text(2, 100 * goal_thresh + 0.6, "success threshold 2.5 cm", fontsize=7.5, color=INK_2)
    ax.set_ylim(0, None)
    ax.set_xlabel("step")
    ax.set_ylabel("cube-to-goal distance (cm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Cube-to-goal distance over the episode (median, shaded IQR)", loc="left")
    return save(fig, out_dir, name)


def success_by_goal_height(summaries, out_dir, name="success_by_goal_height"):
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    first = next(iter(summaries.values()))
    groups = [(i, f"{b['goal_height_cm'][0]:.1f}–{b['goal_height_cm'][1]:.1f} cm\n(n={b['final_success']['n']})")
              for i, b in enumerate(first["by_goal_height"])]
    _grouped_bars(ax, groups, list(summaries.items()), lambda s, i: s["by_goal_height"][i]["final_success"])
    ax.set_ylim(0, 60)
    ax.set_xlabel("goal height above the cube's resting height")
    ax.set_ylabel("success at end (%)")
    ax.legend(loc="upper right", title="checkpoint", title_fontsize=8, fontsize=8)
    ax.set_title("Final success by goal height", loc="left")
    ax.text(0, -0.3, "Numbers above bars: successes. Same scenes for both checkpoints. Whiskers: 95% Wilson CI.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    return save(fig, out_dir, name)


def rollout_examples(example_frames, out_path, columns=6, title=None):
    """example_frames: list of (row_title, frames array (T, H, W, 3)). Evenly spaced frames per row."""
    rows = []
    for row_title, frames in example_frames:
        idx = np.linspace(0, len(frames) - 1, columns).round().astype(int)
        rows.append((row_title, [Image.fromarray(frames[i]) for i in idx]))
    w, h = rows[0][1][0].size
    gap, label_h, top = 4, 20, 28
    sheet = Image.new("RGB", (columns * w + (columns - 1) * gap, top + len(rows) * (h + label_h + gap)), "#fcfcfb")
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((4, top // 2), title, fill=INK, font=_font(15), anchor="lm")
    y = top
    for row_title, frames in rows:
        draw.text((4, y + label_h // 2), row_title, fill=INK, font=_font(13), anchor="lm")
        y += label_h
        for c, frame in enumerate(frames):
            sheet.paste(frame, (c * (w + gap), y))
        y += h + gap
    sheet.save(out_path)
    return out_path
