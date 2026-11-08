"""Plot IDM pseudo-label quality vs number of labelled demos from results/action_free/idm/metrics.csv."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualization.action_free_plots import idm_label_quality  # noqa: E402

if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/action_free/idm/metrics.csv")
    rows = list(csv.DictReader(open(src)))
    for p in idm_label_quality(rows, src.parent):
        print("wrote", p)
