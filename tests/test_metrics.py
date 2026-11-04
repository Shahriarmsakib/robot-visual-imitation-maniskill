import numpy as np
import pytest

from src.evaluation.evaluate import _episode_metrics, summarize
from src.utils import wilson_interval


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(2, 100)
    assert lo == pytest.approx(0.0055, abs=1e-3)
    assert hi == pytest.approx(0.0700, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_episode_metrics_gymnasium1_dict_skips_mask_keys():
    info = {
        "final_info": {"episode": {
            "success_once": np.array([True, False, True]), "_success_once": np.array([True, True, True]),
            "return": np.array([1.0, 0.0, 1.0]), "_return": np.array([True, True, True]),
        }},
        "_final_info": np.array([True, True, True]),
    }
    out = _episode_metrics(info)
    assert set(out) == {"success_once", "return"}
    assert out["success_once"].tolist() == [1.0, 0.0, 1.0]


def test_episode_metrics_gymnasium029_list():
    info = {"final_info": [{"episode": {"success_once": True}}, None, {"episode": {"success_once": False}}]}
    assert _episode_metrics(info)["success_once"].tolist() == [1.0, 0.0]


def test_summarize_counts_per_trial_not_per_batch():
    # 10 batches of 10 parallel envs = 100 trials (the brief's "Evaluated 10 episodes" pitfall).
    per_episode = {"success_at_end": np.concatenate([np.array([1.0] + [0.0] * 9)] * 2 + [np.zeros(10)] * 8)}
    s = summarize(per_episode)
    assert s["n_trials"] == 100
    assert s["success_at_end"]["count"] == 2
    assert s["success_at_end"]["rate"] == pytest.approx(0.02)
