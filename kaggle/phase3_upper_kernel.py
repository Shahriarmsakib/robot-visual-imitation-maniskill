"""Kaggle script kernel for scripts/run_phase3_upper.sh (GPU T4, one GPU used, internet on).

Push with the Kaggle CLI from a folder holding this file (as run.py) and a kernel-metadata.json:
    kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4
SETTINGS below are passed to the run script as environment variables; outputs land in /kaggle/working.
"""
import os
import subprocess
import sys

REPO = "https://github.com/Shahriarmsakib/robot-visual-imitation-maniskill.git"
SETTINGS = {"SEED": "0", "ITERS": "12000", "EVAL_FREQ": "4000", "MAX_HOURS": "5.0"}  # ~4-4.5 h on one T4

code = "/tmp/code"
subprocess.run(["git", "clone", "--depth", "1", REPO, code], check=True)
print(subprocess.run(["git", "-C", code, "log", "-1", "--oneline"], capture_output=True, text=True).stdout, flush=True)
env = {**os.environ, "OUT": "/kaggle/working", **SETTINGS}
proc = subprocess.run(["bash", "scripts/run_phase3_upper.sh"], cwd=code, env=env)
sys.exit(proc.returncode)
