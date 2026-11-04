"""Read run directories written by the trainer and turn them into result tables.

A run directory holds config.json, sessions.jsonl, metrics.jsonl, checkpoints/ and,
when every step finished, DONE. Only logged numbers are reported; anything missing is
reported as missing, never filled in.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils import wilson_interval


@dataclass
class Run:
    path: Path
    config: dict
    sessions: list
    train: list
    evals: list
    done: bool
    checkpoints: list = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.config.get("exp_name") or self.path.name

    @property
    def variant(self) -> str:
        return self.config.get("variant", "act_rgb")

    @property
    def num_demos(self) -> Optional[int]:
        return self.config.get("num_demos")

    @property
    def seed(self) -> int:
        return self.config.get("seed")

    @property
    def total_steps(self) -> int:
        return self.config.get("total_iters")

    @property
    def steps_done(self) -> int:
        steps = [r["step"] for r in self.train] + [r["step"] for r in self.evals]
        return max(steps) if steps else 0

    @property
    def final_eval(self) -> Optional[dict]:
        """The evaluation of the final model, only if the run completed."""
        if not self.done:
            return None
        rows = [r for r in self.evals if r["step"] == self.total_steps]
        return rows[-1] if rows else None

    @property
    def is_smoke(self) -> bool:
        # The run's own name, or a runs dir named exactly "smoke" (runs/smoke/<run>); never the wider path.
        return "smoke" in self.run_id or self.path.parent.name == "smoke"


def _jsonl(path):
    return [json.loads(line) for line in open(path)] if path.exists() else []


def load_run(path) -> Run:
    path = Path(path)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{path} is not a run directory (no config.json)")
    evals = {}
    for row in _jsonl(path / "metrics.jsonl"):
        if row["type"] == "eval":
            evals[row["step"]] = row  # a step re-evaluated after a resume keeps the latest row
    ckpt_dir = path / "checkpoints"
    return Run(
        path=path,
        config=json.loads(config_path.read_text()),
        sessions=_jsonl(path / "sessions.jsonl"),
        train=[r for r in _jsonl(path / "metrics.jsonl") if r["type"] == "train"],
        evals=[evals[s] for s in sorted(evals)],
        done=(path / "DONE").exists(),
        checkpoints=sorted(p.name for p in ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else [],
    )


def find_runs(roots, include_smoke=False):
    """All run directories (containing config.json and metrics.jsonl) under the given paths."""
    runs = []
    for root in roots:
        if not Path(root).exists():
            raise FileNotFoundError(f"runs directory not found: {root}")
        for metrics in sorted(Path(root).rglob("metrics.jsonl")):
            if (metrics.parent / "config.json").exists():
                run = load_run(metrics.parent)
                if include_smoke or not run.is_smoke:
                    runs.append(run)
    return runs


def _rate(entry):
    return None if entry is None else entry["rate"]


def run_table(runs):
    """One row per run (Table 2 of the brief): what ran, how far, and its final measured success."""
    rows = []
    for run in runs:
        final = run.final_eval
        best = max(run.evals, key=lambda r: (r["success_at_end"]["rate"], -r["step"]), default=None)
        session = run.sessions[-1] if run.sessions else {}
        git = (session.get("git") or {}).get("commit") or ""
        rows.append({
            "run_id": run.run_id,
            "variant": run.variant,
            "num_demos": run.num_demos,
            "seed": run.seed,
            "steps_done": run.steps_done,
            "total_steps": run.total_steps,
            "status": "complete" if run.done else f"incomplete ({run.steps_done}/{run.total_steps})",
            "sessions": len(run.sessions),
            "final_n_trials": final["n_trials"] if final else None,
            "final_success_once": final["success_once"]["count"] if final else None,
            "final_success_once_rate": _rate(final["success_once"]) if final else None,
            "final_success_at_end": final["success_at_end"]["count"] if final else None,
            "final_success_at_end_rate": _rate(final["success_at_end"]) if final else None,
            "final_success_at_end_ci95": final["success_at_end"]["ci95"] if final else None,
            "best_success_at_end_rate": best["success_at_end"]["rate"] if best else None,
            "best_success_at_end_step": best["step"] if best else None,
            "checkpoint": "final.pt" if run.done else ("latest.pt" if "latest.pt" in run.checkpoints else None),
            "sim_backend": session.get("sim_backend"),
            "device": session.get("cuda_device") or session.get("device"),
            "git_commit": git[:7],
            "dataset_sha256": ((session.get("dataset") or {}).get("sha256") or "")[:12],
        })
    return rows


def data_efficiency_table(runs, metric="success_at_end"):
    """Final-model success per (variant, num_demos), aggregated over completed seeds.

    Uses the final evaluation, not the best one, to avoid selecting on the test scenes.
    With one seed the interval is the Wilson CI of that run; with more, seed spread is reported.
    """
    groups = {}
    for run in runs:
        if run.final_eval is not None:
            groups.setdefault((run.variant, run.num_demos), []).append(run)
    rows = []
    for (variant, demos), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        group = sorted(group, key=lambda r: r.seed)
        entries = [r.final_eval[metric] for r in group]
        rates = np.array([e["rate"] for e in entries])
        count, n = sum(e["count"] for e in entries), sum(e["n"] for e in entries)
        lo, hi = wilson_interval(count, n)
        rows.append({
            "variant": variant, "num_demos": demos, "metric": metric, "n_seeds": len(group),
            "seeds": [r.seed for r in group], "seed_rates": rates.tolist(),
            "mean_rate": float(rates.mean()), "std_rate": float(rates.std(ddof=1)) if len(rates) > 1 else None,
            "pooled_count": count, "pooled_n": n, "pooled_ci95": [lo, hi],
        })
    return rows
