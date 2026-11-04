"""Object-centric visual imitation learning in ManiSkill (project package)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The vendored upstream ACT code imports itself as the top-level package `act`.
_ACT_DIR = str(ROOT / "third_party" / "maniskill_act")
if _ACT_DIR not in sys.path:
    sys.path.insert(0, _ACT_DIR)
