"""Kaggle script kernel for scripts/run_phase3_noise.sh (GPU T4 x2, internet on): two label-noise conditions.

Push with the Kaggle CLI from a folder holding this file (as run.py) and a kernel-metadata.json:
    kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4
Edit SETTINGS for each session (see experiments/phase3_noise_sweep.md); outputs land in /kaggle/working.
"""
import os
import subprocess
import sys

REPO = "https://github.com/Shahriarmsakib/robot-visual-imitation-maniskill.git"
SETTINGS = {"N": "100", "SEED": "0", "ITERS": "12000", "EVAL_FREQ": "4000", "MAX_HOURS": "6.0",
            "TAG1": "xy_shrink04", "ARGS1": "--xy-scale 0.4",
            "TAG2": "xy025_ar09", "ARGS2": "--xy-r2 0.25 --ar1 0.9"}  # session 3: IDM-like error structure, ~4 h

code = "/tmp/code"
subprocess.run(["git", "clone", "--depth", "1", REPO, code], check=True)
print(subprocess.run(["git", "-C", code, "log", "-1", "--oneline"], capture_output=True, text=True).stdout, flush=True)
env = {**os.environ, "OUT": "/kaggle/working", "PYTHONPATH": code, **SETTINGS}
proc = subprocess.run(["bash", "scripts/run_phase3_noise.sh"], cwd=code, env=env)
sys.exit(proc.returncode)
