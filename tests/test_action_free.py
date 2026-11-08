import numpy as np
import torch

from src.action_free.data import ACTION_FREE_POOL, LABELLED_POOL, labelled_split, transition_pairs, window_indices
from src.action_free.idm import IDM, label_quality, train_idm


def test_splits_deterministic_and_disjoint_from_action_free_pool():
    a, b = labelled_split(10, seed=1), labelled_split(10, seed=1)
    assert a == b and len(set(a)) == 10
    assert set(a) <= set(LABELLED_POOL) and not set(a) & set(ACTION_FREE_POOL)
    assert labelled_split(10, seed=2) != a
    assert sorted(labelled_split(100, seed=0)) == list(LABELLED_POOL)


def test_transition_pairs_align_action_t_with_o_t_to_o_t_plus_1():
    T = 5
    frames = np.stack([np.full((4, 4, 3), t, np.uint8) for t in range(T + 1)])  # frame t has value t
    actions = np.arange(T, dtype=np.float32)[:, None].repeat(4, 1)               # action t has value t
    o_t, o_next, a = transition_pairs(frames, actions)
    assert len(o_t) == len(o_next) == len(a) == T
    for t in range(T):
        assert o_t[t, 0, 0, 0] == t and o_next[t, 0, 0, 0] == t + 1 and a[t, 0] == t


def test_window_indices_centre_on_the_transition_and_clamp_at_edges():
    w = window_indices(4, (-1, 0, 1, 2))  # T = 4 actions, frames 0..4
    assert w.tolist() == [[0, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 4]]
    assert window_indices(3, (0, 1)).tolist() == [[0, 1], [1, 2], [2, 3]]  # (o_t, o_{t+1}) for a_t


def test_idm_learns_a_visible_shift_and_outputs_valid_actions():
    # Synthetic "video": a bright square moves by the x action; the IDM must recover dx from the frame pair.
    rng = np.random.default_rng(0)
    episodes = []
    for _ in range(4):
        T, pos = 30, 40
        frames, acts = [], []
        for t in range(T + 1):
            img = np.zeros((128, 128, 3), np.uint8)
            img[60:68, pos:pos + 8] = 255
            frames.append(img)
            if t < T:
                dx = int(rng.integers(-3, 4))
                pos = int(np.clip(pos + dx * 4, 0, 112))
                acts.append([dx / 3, 0.0, 0.0, 1.0 if dx >= 0 else -1.0])
        episodes.append((np.stack(frames), np.array(acts, np.float32)))
    torch.manual_seed(0)
    model, info = train_idm(episodes, torch.device("cpu"), steps=150, batch=32, seed=0)
    frames, a = episodes[0]
    pred = model.predict(frames)
    assert pred.shape == (len(a), 4) and set(np.unique(pred[:, 3])) <= {-1.0, 1.0}
    assert np.abs(pred[:, :3]).max() <= 1.0
    q = label_quality(pred, a)
    assert q["x_mae"] < q["xyz_zero_baseline_mae"] * 3  # far better than chance on the moving axis
    assert np.corrcoef(pred[:, 0], a[:, 0])[0, 1] > 0.5


def test_label_quality_perfect_prediction():
    true = np.array([[0.1, -0.2, 0.3, 1.0], [-0.1, 0.2, -0.3, -1.0], [0.05, 0.0, 0.1, 1.0]], np.float32)
    q = label_quality(true.copy(), true)
    assert q["xyz_mae"] == 0 and q["gripper_acc"] == 1 and q["x_r2"] == 1
