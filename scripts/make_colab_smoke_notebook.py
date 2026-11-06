"""Generate notebooks/object_centric_smoke_test_colab.ipynb (thin wrapper around the smoke-test script).

Edit the cells here, then: python scripts/make_colab_smoke_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Object-centric data extraction: smoke test (PickCube-v1, 5 demos, RGB-D)

Replays five official ManiSkill motion-planning demonstrations on `physx_cpu`, records RGB, depth, cube pose
(simulator ground truth), end-effector pose, joint positions/velocities, expert actions and success, validates
them, and saves an HDF5 file. All logic lives in `scripts/object_centric_smoke_test.py` and `src/data/object_centric.py`.

**Runtime:** *Runtime → Change runtime type → T4 GPU*. The simulation runs on CPU (`physx_cpu`), but the
camera renderer needs a GPU Vulkan driver; a CPU-only runtime cannot render images. No training happens here."""

SETUP = """import os, subprocess
REPO_URL = "https://github.com/Shahriarmsakib/robot-visual-imitation-maniskill.git"
CODE = "/content/robot-visual-imitation-maniskill"
if not os.path.exists(CODE):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CODE], check=True)
os.chdir(CODE)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)"""

INSTALL = """# Vulkan driver files + project packages (keeps Colab's own PyTorch)
!bash scripts/setup_linux_gpu.sh"""

DATA = """# Official PickCube-v1 motion-planning demos (read-only input; not modified)
!python -m mani_skill.utils.download_demo PickCube-v1 -o data/demos"""

RUN = """!python scripts/object_centric_smoke_test.py --num-demos 5"""

SHOW = """import json
from IPython.display import Image, display
display(Image("results/object_centric_smoke/smoke_frames.png"))
report = json.load(open("results/object_centric_smoke/validation.json"))
for c in report["checks"]:
    print(("PASS " if c["passed"] else "FAIL ") + c["check"] + ": " + c["detail"])"""

INSPECT = """# How to read the saved dataset
import h5py
with h5py.File("data/object_centric/pickcube_5demos_rgbd.h5", "r") as f:
    print(f.attrs["alignment"])
    ep = f["episodes/ep_000"]
    for name, item in ep.items():
        if isinstance(item, h5py.Dataset):
            print(f"  {name}: {item.shape} {item.dtype}")
        else:
            for k, v in item.items():
                print(f"  {name}/{k}: {v.shape} {v.dtype}")"""


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"gpuType": "T4"}},
    "cells": [md(INTRO), md("## 1. Code"), code(SETUP), md("## 2. Install"), code(INSTALL),
              md("## 3. Demonstrations"), code(DATA), md("## 4. Extract, validate, save"), code(RUN),
              md("## 5. Visualisation and validation summary"), code(SHOW), md("## 6. Dataset layout"), code(INSPECT)],
}
out = ROOT / "notebooks" / "object_centric_smoke_test_colab.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("Wrote", out.relative_to(ROOT))
