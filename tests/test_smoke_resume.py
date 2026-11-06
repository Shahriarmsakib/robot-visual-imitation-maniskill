"""End-to-end: tiny training run on CPU, interrupted and resumed, must equal an uninterrupted run.

Slow (about a minute, needs the simulator to render): pytest --run-slow
"""
import json
import subprocess
import sys

import pytest
import torch

from conftest import ROOT


def _run(runs_dir, exp_name, *extra):
    cmd = [sys.executable, "scripts/train.py", "--config", "configs/smoke_mac.json", "--device", "cpu",
           "--runs-dir", str(runs_dir), "--exp-name", exp_name, "--log-freq", "1", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).returncode


def _losses(run_dir):
    rows = map(json.loads, open(run_dir / "metrics.jsonl"))
    return {r["step"]: r["loss"] for r in rows if r["type"] == "train"}


@pytest.mark.slow
def test_interrupted_run_resumes_exactly(tmp_path, demo_path):
    assert _run(tmp_path, "straight") == 0
    assert _run(tmp_path, "split", "--stop-at-step", "13") == 3
    assert _run(tmp_path, "split") == 0

    assert _losses(tmp_path / "straight") == _losses(tmp_path / "split")
    a = torch.load(tmp_path / "straight/checkpoints/final.pt", weights_only=False)
    b = torch.load(tmp_path / "split/checkpoints/final.pt", weights_only=False)
    assert a["step"] == b["step"] == 20
    assert all(torch.equal(a["ema_agent"][k], b["ema_agent"][k]) for k in a["ema_agent"])

    evals = [json.loads(l) for l in open(tmp_path / "straight/metrics.jsonl") if '"eval"' in l]
    assert [e["step"] for e in evals] == [0, 10, 20]
    assert all(e["n_trials"] == 2 for e in evals)

    # The trained checkpoint feeds the M2 tools: standalone evaluation and rollout export.
    final = tmp_path / "straight/checkpoints/final.pt"
    rc = subprocess.run([sys.executable, "scripts/evaluate.py", str(final), "--num-episodes", "2", "--num-envs", "2",
                         "--device", "cpu", "--out", str(tmp_path / "eval.json")], cwd=ROOT).returncode
    assert rc == 0 and json.loads((tmp_path / "eval.json").read_text())["n_trials"] == 2
    rc = subprocess.run([sys.executable, "scripts/export_rollouts.py", str(final), "--num-episodes", "1",
                         "--device", "cpu", "--out", str(tmp_path / "media")], cwd=ROOT).returncode
    assert rc == 0
    assert (tmp_path / "media/rollout_montage.png").exists() and list((tmp_path / "media").glob("ep00_*.gif"))

    # Failure analysis: runs twice on the same scenes and must reproduce itself exactly.
    for label in ("a", "b"):
        rc = subprocess.run([sys.executable, "scripts/failure_analysis.py", str(final), "--label", label,
                             "--num-episodes", "4", "--num-envs", "2", "--device", "cpu",
                             "--out", str(tmp_path / "fa")], cwd=ROOT).returncode
        assert rc == 0
    a = (tmp_path / "fa/metrics/episodes_a.csv").read_text().replace("\na,", "\nX,")
    b = (tmp_path / "fa/metrics/episodes_b.csv").read_text().replace("\nb,", "\nX,")
    assert a == b
    run = json.loads((tmp_path / "fa/metrics/run_a.json").read_text())
    assert run["checks"]["final_success_matches_env_metric"] and run["checks"]["n_episodes"] == 4
    rc = subprocess.run([sys.executable, "scripts/summarize_failure_analysis.py", "--dir", str(tmp_path / "fa"),
                         "--primary", "a"], cwd=ROOT).returncode
    assert rc == 0 and (tmp_path / "fa/figures/success_breakdown.png").exists()


@pytest.mark.slow
def test_resume_refuses_changed_hyperparameters(tmp_path, demo_path):
    assert _run(tmp_path, "run", "--stop-at-step", "5") == 3
    assert _run(tmp_path, "run", "--lr", "0.001") != 0
