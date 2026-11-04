"""Train (or resume) the ACT RGB policy. See `python scripts/train.py --help`.

Examples:
    python scripts/train.py --config configs/smoke_mac.json
    python scripts/train.py --config configs/baseline_act_rgb_100d.json --seed 1
"""
import os
import sys
from pathlib import Path

# Must be set before torch is imported: lets the few ops MPS lacks fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    # Imported here, not at module level: CPU eval workers (forkserver/spawn) re-import this
    # file, and on macOS a worker that inherits an initialised SAPIEN/Metal renderer fails to
    # render (IndexError: unordered_map::at in take_picture).
    from src.training.train_act import main

    sys.exit(main())
