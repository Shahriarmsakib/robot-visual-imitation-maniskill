"""Demo loading and labelled/action-free splits for the action-free study.

Convention (same as the ACT pipeline): an episode has frames o_0..o_T and actions a_0..a_{T-1};
a_t is the action applied at o_t, so the transition o_t -> o_{t+1} is caused by a_t.
"""
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_H5 = "data/action_free/conversion/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
LABELLED_POOL = range(0, 100)      # labelled demos are drawn from these episode ids
ACTION_FREE_POOL = range(100, 1000)  # treated as action-free video; true actions used only for evaluation


def episode_keys(h5_path):
    with h5py.File(h5_path, "r") as f:
        return sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))


def load_episode(f, key):
    """(frames uint8 (T+1, H, W, 3), actions float32 (T, 4)) for one episode of an open h5 file."""
    g = f[key]
    frames = g["obs/sensor_data/base_camera/rgb"][:]
    actions = g["actions"][:].astype(np.float32)
    assert len(frames) == len(actions) + 1, (key, frames.shape, actions.shape)
    return frames, actions


def load_episodes(h5_path, ids):
    with h5py.File(h5_path, "r") as f:
        return {i: load_episode(f, f"traj_{i}") for i in ids}


def labelled_split(n_labelled, seed, pool=LABELLED_POOL):
    """The n labelled episode ids for a seed: a fixed random draw from the labelled pool (sorted)."""
    if n_labelled > len(pool):
        raise ValueError(f"n_labelled={n_labelled} exceeds labelled pool of {len(pool)}")
    rng = np.random.default_rng(10_000 + seed)
    return sorted(int(i) for i in rng.choice(list(pool), size=n_labelled, replace=False))


def save_split(path, n_labelled, seed):
    ids = labelled_split(n_labelled, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"n_labelled": n_labelled, "seed": seed, "labelled_ids": ids,
                                      "action_free_ids": [ACTION_FREE_POOL.start, ACTION_FREE_POOL.stop - 1]}) + "\n")
    return ids


def transition_pairs(frames, actions):
    """Inputs and targets for an IDM: (o_t, o_{t+1}) -> a_t for t = 0..T-1."""
    return frames[:-1], frames[1:], actions


def window_indices(T, offsets):
    """Frame indices (T, len(offsets)) of the window around each transition t = 0..T-1.

    offsets are relative to t (e.g. (0, 1) = (o_t, o_{t+1}); (-1, 0, 1, 2) adds one frame before and after);
    indices are clamped to the episode's frames 0..T.
    """
    t = np.arange(T)[:, None]
    return np.clip(t + np.asarray(offsets)[None], 0, T)
