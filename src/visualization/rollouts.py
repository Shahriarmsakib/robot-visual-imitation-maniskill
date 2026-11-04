"""Label rollout frames and write GIFs and a success/failure montage (Fig. 5)."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SUCCESS = "#1f7a3a"
FAILURE = "#b3261e"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def label_frame(frame, size, step, outcome_text, success):
    """Resize to size x size and add a step counter plus an outcome bar (text, not colour alone)."""
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((size, size), Image.BILINEAR)
    bar = max(18, size // 12)
    canvas = Image.new("RGB", (size, size + bar), SURFACE)
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, size, size, size + bar], fill=SUCCESS if success else FAILURE)
    draw.text((6, size + bar // 2), outcome_text, fill="white", font=_font(bar - 6), anchor="lm")
    draw.rectangle([4, 4, 4 + size // 4, 4 + bar - 4], fill=(255, 255, 255))
    draw.text((8, 4 + (bar - 4) // 2), f"t={step}", fill=INK, font=_font(bar - 8), anchor="lm")
    return canvas


def outcome_text(episode, success_at_end, success_once):
    if success_at_end:
        return f"ep {episode}: SUCCESS"
    if success_once:
        return f"ep {episode}: lifted, not held at end"
    return f"ep {episode}: FAILURE"


def write_gif(frames, path, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def montage(episodes, path, columns=6, title=None):
    """Grid of labelled frames: one row per episode, `columns` evenly spaced time points."""
    rows = []
    for ep in episodes:
        frames = ep["frames"]
        idx = np.linspace(0, len(frames) - 1, columns).round().astype(int)
        rows.append([frames[i] for i in idx])
    w, h = rows[0][0].size
    gap = 4
    top = 28 if title else 0
    sheet = Image.new("RGB", (columns * w + (columns - 1) * gap, top + len(rows) * h + (len(rows) - 1) * gap), SURFACE)
    if title:
        ImageDraw.Draw(sheet).text((4, top // 2), title, fill=INK, font=_font(16), anchor="lm")
    for r, row in enumerate(rows):
        for c, frame in enumerate(row):
            sheet.paste(frame, (c * (w + gap), top + r * (h + gap)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return Path(path)
