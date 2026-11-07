"""Integrity tests for the Phase 2A ACT-compatible object-centric dataset.

The dataset-file tests run against data/object_centric/pickcube_5demos_objectcentric_ee_delta.h5 and skip if it
has not been built (python scripts/build_objectcentric_ee_delta.py). The slow test rebuilds a 2-demo copy.
"""
import subprocess
import sys

import h5py
import numpy as np
import pytest

from conftest import ROOT

DATASET = ROOT / "data/object_centric/pickcube_5demos_objectcentric_ee_delta.h5"
SOURCE = ROOT / "data/demos/PickCube-v1/motionplanning/trajectory.h5"
PER_STEP = ["obs/rgb", "obs/depth", "objects/cube_pose", "objects/goal_pos", "robot/tcp_pose", "robot/qpos",
            "robot/qvel", "robot/is_grasped", "obs_step"]


@pytest.fixture(scope="module")
def dataset():
    if not DATASET.exists():
        pytest.skip(f"dataset not built: {DATASET}")
    with h5py.File(DATASET, "r") as f:
        yield f


def _episodes(f):
    return [f["episodes"][k] for k in sorted(f["episodes"])]


def test_file_metadata(dataset):
    assert dataset.attrs["control_mode"] == "pd_ee_delta_pos"
    assert dataset.attrs["obs_mode"] == "rgbd" and dataset.attrs["sim_backend"] == "physx_cpu"
    assert "replay_trajectory" in dataset.attrs["conversion_command"]
    assert len(dataset["episodes"]) == 5


def test_actions_are_4d_and_valid(dataset):
    for g in _episodes(dataset):
        a = g["actions"][:]
        assert a.ndim == 2 and a.shape[1] == 4 and a.dtype == np.float32
        assert np.isfinite(a).all() and (np.abs(a) <= 1.0).all()
        assert set(np.unique(a[:, 3])) <= {-1.0, 1.0}


def test_timestep_alignment(dataset):
    for g in _episodes(dataset):
        T = int(g.attrs["T"])
        assert g["actions"].shape[0] == g["success"].shape[0] == g["action_step"].shape[0] == T
        for key in PER_STEP:
            assert g[key].shape[0] == T + 1, key
        assert np.array_equal(g["obs_step"][:], np.arange(T + 1))
        assert np.array_equal(g["action_step"][:], np.arange(T))


def test_values_complete_and_depth_valid(dataset):
    for g in _episodes(dataset):
        for key in ["objects/cube_pose", "objects/goal_pos", "robot/tcp_pose", "robot/qpos", "robot/qvel"]:
            assert np.isfinite(g[key][:]).all(), key
        depth = g["obs/depth"][:]
        assert depth.dtype == np.int16 and (depth > 0).all() and depth.max() <= 20000
        assert g["obs/rgb"].dtype == np.uint8 and g["obs/rgb"][:].std() > 1
        assert g["camera/intrinsic_cv"].shape == (3, 3) and g["camera/extrinsic_cv"].shape == (3, 4)


def test_all_demos_successful_and_poses_match_converted_states(dataset):
    for g in _episodes(dataset):
        assert bool(g["success"][-1]) and g.attrs["replay_success"] and g.attrs["conversion_success"]
        err = np.abs(g["objects/cube_pose"][:, :3] - g["objects/cube_pose_recorded"][:, :3]).max()
        assert err < 0.005


@pytest.mark.slow
def test_build_script_end_to_end(tmp_path):
    if not SOURCE.exists():
        pytest.skip("demos not downloaded")
    out = tmp_path / "ee.h5"
    proc = subprocess.run([sys.executable, "scripts/build_objectcentric_ee_delta.py", "--num-demos", "2", "--out", str(out),
                           "--work-dir", str(tmp_path / "work"), "--report-dir", str(tmp_path / "rep")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "DATASET VALIDATION PASSED" in proc.stdout
    with h5py.File(out, "r") as f:
        assert len(f["episodes"]) == 2 and f["episodes/ep_000/actions"].shape[1] == 4
