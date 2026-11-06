"""Visual check that extracted RGB-D frames, cube and end-effector positions belong together."""
import numpy as np

from src.data.object_centric import project
from src.visualization.plots import INK_2, MUTED, SLOTS, plt, save, style


def frames_figure(ep, out_dir, name="smoke_frames", episode_label="ep_000"):
    """RGB (top) and depth (bottom) at the first, middle and last step, with projected ground truth."""
    style()
    T = len(ep["actions"])
    steps = [0, T // 2, T]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 7.4))
    K, E = ep["intrinsic_cv"][0], ep["extrinsic_cv"][0]
    # Clip the colour scale to the workspace: the far background (several metres) would flatten it.
    dmin, dmax = 0.2, 1.2
    for col, t in enumerate(steps):
        cube, tcp = ep["cube_pose"][t, :3], ep["tcp_pose"][t, :3]
        (cu, cv), _ = [a[0] for a in project(cube[None], K, E)]
        (tu, tv), _ = [a[0] for a in project(tcp[None], K, E)]
        ax = axes[0, col]
        ax.imshow(ep["rgb"][t])
        ax.scatter([cu], [cv], s=70, facecolors="none", edgecolors="white", linewidths=1.8, label="cube (ground truth)")
        ax.scatter([tu], [tv], s=70, marker="x", color=SLOTS[0], linewidths=2, label="end-effector (TCP)")
        ax.set_title(f"t = {t}" + (" (start)" if t == 0 else " (end)" if t == T else " (middle)"), loc="left", fontsize=10)
        ax.axis("off")
        ax.text(0, -0.04, f"cube xyz (m)\n ({cube[0]:+.3f}, {cube[1]:+.3f}, {cube[2]:+.3f})\n"
                          f"EE xyz (m)\n ({tcp[0]:+.3f}, {tcp[1]:+.3f}, {tcp[2]:+.3f})\n"
                          f"grasped: {'yes' if ep['is_grasped'][t] else 'no'}",
                transform=ax.transAxes, va="top", fontsize=7.5, family="monospace", color=INK_2)
        ax = axes[1, col]
        im = ax.imshow(np.clip(ep["depth"][t] / 1000.0, dmin, dmax), cmap="viridis", vmin=dmin, vmax=dmax)
        ax.scatter([cu], [cv], s=70, facecolors="none", edgecolors="white", linewidths=1.8)
        ax.scatter([tu], [tv], s=70, marker="x", color="white", linewidths=2)
        ax.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.85)
    cbar = fig.colorbar(im, ax=axes[1, :], fraction=0.025, pad=0.02)
    cbar.set_label("depth (m), clipped to 0.2–1.2", color=INK_2)
    fig.suptitle(f"PickCube-v1 demo {episode_label} (source {ep['key']}, seed {ep['seed']}): RGB (top), depth (bottom); "
                 f"markers = simulator ground truth projected with the camera parameters",
                 x=0.01, ha="left", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(hspace=0.62, wspace=0.08, top=0.92)
    return save(fig, out_dir, name)
