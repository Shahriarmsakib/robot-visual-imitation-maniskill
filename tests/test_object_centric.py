import subprocess
import sys

import h5py
import numpy as np
import pytest

from conftest import ROOT
from src.data.object_centric import project, save_hdf5

SOURCE = ROOT / "data/demos/PickCube-v1/motionplanning/trajectory.h5"


def _fake_episode(T=6):
    rng = np.random.default_rng(0)
    return {
        "rgb": rng.integers(0, 255, (T + 1, 8, 8, 3), dtype=np.uint8), "depth": np.full((T + 1, 8, 8), 500, np.int16),
        "qpos": np.zeros((T + 1, 9), np.float32), "qvel": np.zeros((T + 1, 9), np.float32),
        "tcp_pose": np.zeros((T + 1, 7), np.float32), "is_grasped": np.zeros(T + 1, bool),
        "cube_pose": np.zeros((T + 1, 7), np.float32), "cube_pose_recorded": np.zeros((T + 1, 7), np.float32),
        "goal_pos": np.zeros((T + 1, 3), np.float32), "intrinsic_cv": np.tile(np.eye(3), (T + 1, 1, 1)),
        "extrinsic_cv": np.tile(np.eye(3, 4), (T + 1, 1, 1)), "actions": np.arange(T * 8, dtype=np.float32).reshape(T, 8),
        "success": np.array([False] * (T - 1) + [True]), "key": "traj_0", "seed": 0, "control_mode": "pd_joint_pos",
    }


def test_project_pinhole():
    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
    E = np.hstack([np.eye(3), np.zeros((3, 1))])
    uv, z = project(np.array([[0.1, -0.2, 2.0]]), K, E)
    assert uv[0] == pytest.approx([69.0, 54.0]) and z[0] == pytest.approx(2.0)


def test_hdf5_layout_keeps_alignment(tmp_path):
    ep = _fake_episode()
    path = save_hdf5(tmp_path / "d.h5", [ep], "src.h5", "abc", {"env_id": "PickCube-v1"})
    with h5py.File(path, "r") as f:
        g = f["episodes/ep_000"]
        assert g["obs/rgb"].shape[0] == g["actions"].shape[0] + 1 == g["success"].shape[0] + 1
        assert np.array_equal(g["actions"][:], ep["actions"])
        assert g.attrs["T"] == 6 and g.attrs["success"]
        assert "actions[t] is applied at state t" in f.attrs["alignment"]


@pytest.mark.slow
def test_smoke_script_end_to_end(tmp_path):
    if not SOURCE.exists():
        pytest.skip("demos not downloaded")
    out = tmp_path / "d.h5"
    proc = subprocess.run([sys.executable, "scripts/object_centric_smoke_test.py", "--num-demos", "2", "--out", str(out),
                           "--report-dir", str(tmp_path / "rep")], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:]
    assert "SMOKE TEST PASSED" in proc.stdout
    with h5py.File(out, "r") as f:
        assert len(f["episodes"]) == 2 and f["episodes/ep_000/obs/depth"].dtype == np.int16
