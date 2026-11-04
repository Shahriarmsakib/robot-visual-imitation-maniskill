import json

import numpy as np
import pytest

from src.results import data_efficiency_table, find_runs, run_table
from src.utils import wilson_interval
from src.visualization.plots import plot_data_efficiency, plot_success_curves, plot_training_loss
from src.visualization.rollouts import label_frame, montage, write_gif


def _eval(step, once, end, n=100):
    entry = lambda c: {"count": c, "n": n, "rate": c / n, "ci95": list(wilson_interval(c, n))}
    return {"type": "eval", "step": step, "n_trials": n, "success_once": entry(once), "success_at_end": entry(end),
            "sim_backend": "physx_cuda"}


def _make_run(root, name, demos, seed, evals, done=True, total=30000):
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"exp_name": name, "num_demos": demos, "seed": seed, "total_iters": total}))
    (d / "sessions.jsonl").write_text(json.dumps({"device": "cuda", "sim_backend": "physx_cuda",
                                                   "git": {"commit": "abcdef123"}, "dataset": {"sha256": "d0f2"}}) + "\n")
    train = [{"type": "train", "step": s, "loss": 1.0 / s} for s in range(100, max(e["step"] for e in evals) + 1, 100)]
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train + evals))
    if done:
        (d / "DONE").write_text("done")
    return d


@pytest.fixture
def runs(tmp_path):
    _make_run(tmp_path, "a_100d_s1", 100, 1, [_eval(0, 0, 0), _eval(15000, 30, 20), _eval(30000, 60, 50)])
    _make_run(tmp_path, "a_100d_s2", 100, 2, [_eval(0, 0, 0), _eval(30000, 70, 60)])
    _make_run(tmp_path, "a_10d_s1", 10, 1, [_eval(0, 0, 0), _eval(30000, 12, 10)])
    _make_run(tmp_path, "a_25d_s1", 25, 1, [_eval(0, 0, 0), _eval(10000, 9, 8)], done=False)
    _make_run(tmp_path, "smoke_x", 100, 1, [_eval(0, 0, 0), _eval(30000, 99, 99)])
    return tmp_path


def test_find_runs_skips_smoke_unless_asked(runs):
    assert len(find_runs([runs])) == 4
    assert len(find_runs([runs], include_smoke=True)) == 5
    with pytest.raises(FileNotFoundError):
        find_runs([runs / "nope"])


def test_run_table_reports_final_only_for_completed_runs(runs):
    rows = {r["run_id"]: r for r in run_table(find_runs([runs]))}
    assert rows["a_100d_s1"]["final_success_at_end"] == 50 and rows["a_100d_s1"]["final_n_trials"] == 100
    assert rows["a_100d_s1"]["status"] == "complete"
    incomplete = rows["a_25d_s1"]
    assert incomplete["status"] == "incomplete (10000/30000)"
    assert incomplete["final_success_at_end"] is None  # never report a partial run as final
    assert incomplete["checkpoint"] is None


def test_data_efficiency_aggregates_completed_seeds(runs):
    table = {r["num_demos"]: r for r in data_efficiency_table(find_runs([runs]))}
    assert set(table) == {10, 100}  # 25 is incomplete: excluded
    assert table[100]["n_seeds"] == 2
    assert table[100]["mean_rate"] == pytest.approx(0.55)
    assert table[100]["pooled_count"] == 110 and table[100]["pooled_n"] == 200
    assert table[10]["std_rate"] is None


def test_figures_written_with_and_without_data(runs, tmp_path):
    found = find_runs([runs])
    table = data_efficiency_table(found) + data_efficiency_table(found, "success_once")
    out = tmp_path / "figs"
    paths = plot_training_loss(found, out) + plot_success_curves(found, out) + plot_data_efficiency(table, out)
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
    empty = plot_training_loss([], out, "empty_loss") + plot_data_efficiency([], out, "empty_eff")
    assert all(p.exists() for p in empty)


def test_rollout_media(tmp_path):
    frames = [label_frame(np.full((64, 64, 3), i * 20, np.uint8), 48, i, "ep 0: SUCCESS", True) for i in range(5)]
    assert frames[0].size == (48, 48 + 18)
    gif = write_gif(frames, tmp_path / "ep.gif", fps=10)
    sheet = montage([{"frames": frames}, {"frames": frames}], tmp_path / "m.png", columns=3, title="t")
    assert gif.stat().st_size > 0 and sheet.exists()
