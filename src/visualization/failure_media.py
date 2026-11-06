"""Annotated frames and MP4 writing for failure-analysis rollouts."""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from src.visualization.rollouts import _font

BAR = 32  # keeps frame height a multiple of 16 for H.264 when size is 256
CATEGORY_BAR = {"success": "#1f7a3a"}
DEFAULT_BAR = "#b3261e"


def annotate_frame(frame, size, step, header, category, diag):
    """Resize, add a step counter, per-step simulator status and a category bar (text, not colour alone).

    diag holds the post-action state of the previous step (None for the initial frame).
    """
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size + BAR), "#fcfcfb")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, size, size, size + BAR], fill=CATEGORY_BAR.get(category, DEFAULT_BAR))
    draw.text((6, size + BAR // 2), header, fill="white", font=_font(13), anchor="lm")
    lines = [f"t={step}"]
    if diag is not None:
        lines.append(f"grasped: {'yes' if diag['is_grasped'] > 0.5 else 'no'}")
        lines.append(f"to goal: {100 * diag['obj_to_goal']:.1f} cm")
        lines.append(f"in target: {'yes' if diag['is_obj_placed'] > 0.5 else 'no'}")
        lines.append(f"arm static: {'yes' if diag['is_robot_static'] > 0.5 else 'no'}")
    font = _font(11)
    box_h = 14 * len(lines) + 6
    draw.rectangle([4, 4, 118, 4 + box_h], fill=(255, 255, 255))
    for i, line in enumerate(lines):
        draw.text((8, 7 + 14 * i), line, fill="#0b0b0b", font=font)
    return canvas


def write_mp4(frames, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16,
                            ffmpeg_log_level="error") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    return path
