"""Figures for the action-free study (IDM pseudo-label quality vs number of labelled demos)."""
import numpy as np

from src.visualization.plots import INK_2, MUTED, SLOTS, plt, save, style


def idm_label_quality(rows, out_dir, name="idm_label_quality"):
    """rows: dicts from results/action_free/idm/metrics.csv. Mean +/- std over seeds at each N."""
    style()
    ns = sorted({int(r["n_labelled"]) for r in rows})
    x = np.arange(len(ns))

    def stat(key):
        vals = [[float(r[key]) for r in rows if int(r["n_labelled"]) == n] for n in ns]
        return np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals]), len(vals[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for color, axis in zip(SLOTS, "xyz"):
        m, s, k = stat(f"{axis}_r2")
        ax1.errorbar(x, m, yerr=s, color=color, marker="o", ms=6, capsize=3, lw=2, label=f"{axis} (end-effector delta)")
    ax1.axhline(0, color=INK_2, lw=1, ls="--", label="0 = no better than predicting the mean")
    ax1.set_xticks(x, [str(n) for n in ns])
    ax1.set_xlabel("labelled demos used to train the IDM")
    ax1.set_ylabel("pseudo-label R² on action-free demos")
    ax1.set_title("Movement labels", loc="left")
    ax1.legend(loc="lower right", fontsize=8)

    m, s, k = stat("gripper_acc")
    base, _, _ = stat("gripper_majority_baseline_acc")
    ax2.errorbar(x, 100 * m, yerr=100 * s, color=SLOTS[0], marker="o", ms=6, capsize=3, lw=2, label="IDM")
    ax2.plot(x, 100 * base, color=INK_2, lw=1, ls="--", label="always the majority value")
    ax2.set_ylim(40, 100)
    ax2.set_xticks(x, [str(n) for n in ns])
    ax2.set_xlabel("labelled demos used to train the IDM")
    ax2.set_ylabel("gripper open/close correct (%)")
    ax2.set_title("Gripper labels", loc="left")
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"IDM pseudo-label quality on 900 action-free PickCube demos (mean ± std over {k} "
                 f"seed{'s' * (k > 1)}; "
                 "true actions hidden from the IDM, used only for scoring)", x=0.01, ha="left", fontsize=10,
                 color=MUTED)
    fig.tight_layout()
    return save(fig, out_dir, name)
