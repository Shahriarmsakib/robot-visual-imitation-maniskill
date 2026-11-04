"""Generate notebooks/kaggle_train.ipynb (thin launcher around the CLI scripts).

Edit the cells here, then: python scripts/make_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# ACT on ManiSkill PickCube-v1: Kaggle training launcher

Thin wrapper around the repository's scripts; all logic lives in `scripts/` and `src/`.

**Before running**
1. Settings: *Accelerator* → **GPU T4 x2** (or P100), *Internet* → **On** (needs a phone-verified Kaggle account).
2. Code: cloned from the public GitHub repo in `REPO_URL`. Alternatively set `REPO_URL = ""` and add the zip from `scripts/export_code_zip.sh` as a Kaggle Dataset input.
3. Run in the background with **Save Version → Save & Run All (Commit)**; it keeps running with the browser closed, up to Kaggle's 12-hour limit.

**Resuming after the 12-hour limit**: training checkpoints itself and stops cleanly after `MAX_HOURS`. Create a new version of this notebook, add the previous version's **output** as an input (*Add Input → Your Work → Notebooks*), keep the same `EXP_NAME`/`SEED`, and Save & Run All again. The run continues from `latest.pt` with identical optimiser, schedule, data order and RNG state.

Nothing here is a result until the eval rows in `metrics.jsonl` say so."""

CONFIG = """# ---- edit these ----
REPO_URL = "https://github.com/Shahriarmsakib/robot-visual-imitation-maniskill.git"  # empty = use an uploaded code dataset
REPO_REF = "main"
CONFIG = "configs/baseline_act_rgb_100d.json"
SEED = 1
EXP_NAME = f"act_pickcube_rgb_baseline_100d_seed{SEED}"
MAX_HOURS = 11.0         # stop and checkpoint before Kaggle's 12 h limit (leaves time for final eval + output upload)
EXTRA_ARGS = []          # e.g. ["--total-iters", "2000", "--eval-freq", "1000"] for a short GPU smoke run"""

GET_CODE = """import glob, os, shutil, subprocess

WORK = "/kaggle/working"
CODE = "/kaggle/temp/code"   # outside /kaggle/working so the code is not saved as output
if os.path.exists(CODE):
    shutil.rmtree(CODE)
if REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, CODE], check=True)
else:
    hits = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/scripts/train.py", recursive=True)]
    assert hits, "No code found: set REPO_URL or add the code zip dataset as an input"
    shutil.copytree(hits[0], CODE)
os.chdir(CODE)
print("code from", REPO_URL or hits[0])
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout or "(no git metadata: zip upload)")"""

SETUP = """def sh(cmd):
    \"\"\"Run a shell command, streaming output; stop the notebook if it fails.\"\"\"
    print("$", cmd)
    rc = subprocess.run(cmd, shell=True).returncode
    if rc != 0:
        raise RuntimeError(f"command failed (exit {rc}): {cmd}  <- scroll up for the error")

sh("bash scripts/setup_linux_gpu.sh")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")"""

DATA = """sh("bash scripts/prepare_data.sh 100")"""

CHECK = """sh(f"python scripts/check_env.py --render-test --sim-backend physx_cuda --out {WORK}/runs/env_kaggle.json")"""

RESTORE = """# Bring back the previous session's run (if its output was added as an input) so training resumes.
run_dir = f"{WORK}/runs/{EXP_NAME}"
prev = sorted(glob.glob(f"/kaggle/input/**/runs/{EXP_NAME}/checkpoints/latest.pt", recursive=True))
if os.path.exists(f"{run_dir}/checkpoints/latest.pt"):
    print("run dir already present:", run_dir)
elif prev:
    src_run = os.path.dirname(os.path.dirname(prev[-1]))
    shutil.copytree(src_run, run_dir)
    print("restored previous session from", src_run)
else:
    print("no previous checkpoint: starting a new run")"""

TRAIN = """import sys
cmd = [sys.executable, "scripts/train.py", "--config", CONFIG, "--seed", str(SEED), "--exp-name", EXP_NAME,
       "--runs-dir", f"{WORK}/runs", "--max-hours", str(MAX_HOURS), *EXTRA_ARGS]
print(" ".join(cmd))
proc = subprocess.run(cmd)
status = {0: "FINISHED", 3: "PAUSED (time budget): resume in a new version"}.get(proc.returncode, f"FAILED (exit {proc.returncode})")
print("training status:", status)
if proc.returncode not in (0, 3):
    raise RuntimeError("training failed; the traceback is above")"""

SUMMARY = """import json
rows = [json.loads(l) for l in open(f"{run_dir}/metrics.jsonl")]
for r in rows:
    if r["type"] == "eval":
        so, se = r["success_once"], r["success_at_end"]
        print(f"step {r['step']:>6}: success_once {so['count']}/{so['n']} ({so['rate']:.0%}), "
              f"success_at_end {se['count']}/{se['n']} ({se['rate']:.0%}), backend {r['sim_backend']}")
last = [r for r in rows if r["type"] == "train"][-1:]
print("last train row:", last)
sh(f"du -sh {run_dir}/checkpoints/*")"""


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "cells": [md(INTRO), code(CONFIG), md("## Code"), code(GET_CODE), md("## Environment"), code(SETUP),
              md("## Data (official demos, first 100 converted to RGB + pd_ee_delta_pos)"), code(DATA),
              md("## Check: packages, dataset hash, GPU rendering"), code(CHECK),
              md("## Resume support"), code(RESTORE), md("## Train"), code(TRAIN), md("## Summary"), code(SUMMARY)],
}
out = ROOT / "notebooks" / "kaggle_train.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("Wrote", out.relative_to(ROOT))
