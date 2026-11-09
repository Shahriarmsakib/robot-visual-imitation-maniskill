"""Kaggle script kernel for scripts/run_gpu_gap_check.sh (GPU T4, internet on, no training).

Its kernel-metadata.json lists the upper-bound kernel as a source ("kernel_sources":
["shahriarmsakib/phase3-upper-n1000"]), so that run's final.pt is mounted under /kaggle/input.
Push with the Kaggle CLI from a folder holding this file (as run.py) and the metadata:
    kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4
"""
import glob
import os
import subprocess
import sys

REPO = "https://github.com/Shahriarmsakib/robot-visual-imitation-maniskill.git"

ckpts = sorted(glob.glob("/kaggle/input/**/phase3_upper_N1000_seed0/checkpoints/final.pt", recursive=True))
if not ckpts:
    sys.exit("final.pt of phase3_upper_N1000_seed0 not found under /kaggle/input (is the kernel source attached?)")
print("checkpoint:", ckpts[0], flush=True)

code = "/tmp/code"
subprocess.run(["git", "clone", "--depth", "1", REPO, code], check=True)
print(subprocess.run(["git", "-C", code, "log", "-1", "--oneline"], capture_output=True, text=True).stdout, flush=True)
env = {**os.environ, "OUT": "/kaggle/working", "CKPT": ckpts[0]}
proc = subprocess.run(["bash", "scripts/run_gpu_gap_check.sh"], cwd=code, env=env)
sys.exit(proc.returncode)
