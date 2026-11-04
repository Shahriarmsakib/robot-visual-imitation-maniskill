import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "data/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", help="run slow end-to-end tests (simulator + training)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow; use --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def demo_path():
    if not DEMO.exists():
        pytest.skip(f"dataset missing: {DEMO} (run scripts/prepare_data.sh)")
    return str(DEMO)
