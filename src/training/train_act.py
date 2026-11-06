"""ACT behaviour-cloning trainer with exact resume.

Derived from ManiSkill v3.0.1 examples/baselines/act/train_rgbd.py (Apache-2.0).
The model, loss, optimiser (AdamW, backbone lr split), step LR drop at 2/3 of
training, EMA (power 0.75) and evaluation protocol are unchanged. Changes:
- device selection (cuda > mps > cpu) instead of hard-coded `.cuda()`;
- full checkpoints (model, EMA, optimiser, scheduler, RNG, iteration) written
  atomically, and exact resume including the data order;
- evaluation at step 0 (untrained), every `eval_freq` updates, and after the
  final update (upstream skipped the final model), on seeded scenes;
- JSON-lines metrics with trial counts and confidence intervals, a config
  snapshot and provenance (git commit, dataset hash, package versions);
- optional wall-clock budget that checkpoints and exits cleanly;
- TensorBoard/W&B logging removed.
Steps count completed optimiser updates: step N means N updates have been applied.
"""
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, fields
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from src import ROOT
from src.data.demo_dataset import ACTDemoDataset
from src.evaluation.envs import make_eval_envs
from src.evaluation.evaluate import evaluate
from src.models.act_agent import ACTAgent
from src.utils import (append_jsonl, atomic_torch_save, get_rng_state, git_info, seed_everything,
                       select_device, select_sim_backend, set_rng_state, sha256_file)

CKPT_FORMAT = 1
DEFAULT_DEMO = "data/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"


@dataclass
class Args:
    exp_name: Optional[str] = None
    """run name; outputs go to <runs_dir>/<exp_name>"""
    seed: int = 1
    torch_deterministic: bool = True
    """sets torch.backends.cudnn.deterministic"""
    capture_video: bool = False
    """record rollout videos during evaluation"""

    env_id: str = "PickCube-v1"
    demo_path: str = DEFAULT_DEMO
    num_demos: Optional[int] = None
    """use the first N episodes of the demo file (default: all)"""
    total_iters: int = 30_000
    batch_size: int = 256

    # ACT
    lr: float = 1e-4
    kl_weight: float = 10
    temporal_agg: bool = True

    # Backbone
    position_embedding: str = "sine"
    backbone: str = "resnet18"
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False
    include_depth: bool = False

    # Transformer
    enc_layers: int = 2
    dec_layers: int = 4
    dim_feedforward: int = 512
    hidden_dim: int = 256
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    # Environment / evaluation
    max_episode_steps: Optional[int] = 100
    log_freq: int = 100
    eval_freq: int = 5000
    save_freq: Optional[int] = 5000
    """keep a permanent checkpoint every N steps (None: only latest/best/final)"""
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "auto"
    """auto: physx_cuda with an NVIDIA GPU, else physx_cpu"""
    num_dataload_workers: int = 0
    control_mode: str = "pd_ee_delta_pos"
    demo_type: Optional[str] = "motionplanning"

    # Added by this project
    device: str = "auto"
    """auto, cuda, mps or cpu"""
    runs_dir: str = "runs"
    ckpt_freq: int = 1000
    """overwrite <run>/checkpoints/latest.pt every N steps (resume point)"""
    resume: str = "auto"
    """auto: continue from latest.pt if it exists; none: refuse to reuse a run dir; or a checkpoint path"""
    max_hours: Optional[float] = None
    """stop cleanly (checkpoint + exit code 3) after this much wall-clock time"""
    stop_at_step: Optional[int] = None
    """stop like --max-hours once this step is reached (split a run across sessions; resume tests)"""
    eval_seed: int = 10_000
    """seed for evaluation scenes; with enhanced_determinism (since 2026-11-06) every evaluation uses the same scenes"""
    skip_initial_eval: bool = False


# Changing any of these mid-run would make a resumed run a different experiment.
RESUME_MUST_MATCH = [
    "seed", "env_id", "num_demos", "total_iters", "batch_size", "lr", "kl_weight", "temporal_agg",
    "position_embedding", "backbone", "lr_backbone", "masks", "dilation", "include_depth", "enc_layers",
    "dec_layers", "dim_feedforward", "hidden_dim", "dropout", "nheads", "num_queries", "pre_norm",
    "max_episode_steps", "control_mode", "eval_freq", "num_eval_episodes", "eval_seed",
]


class ResumableBatchSampler:
    """Yields random batches for `total` steps from a seeded generator, skipping the first `start` batches.

    Skipping replays only the index permutations (no data loading), so a resumed run sees
    exactly the batches an uninterrupted run would have seen.
    """

    def __init__(self, dataset_len, batch_size, total, seed, start=0):
        self.dataset_len, self.batch_size, self.total, self.seed, self.start = dataset_len, batch_size, total, seed, start

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed)
        sampler = BatchSampler(RandomSampler(range(self.dataset_len), generator=gen), self.batch_size, drop_last=True)
        step = 0
        while True:
            for batch in sampler:
                if step >= self.total:
                    return
                if step >= self.start:
                    yield batch
                step += 1

    def __len__(self):
        return self.total - self.start


def check_demo_matches(args):
    info = json.loads(Path(args.demo_path[:-2] + "json").read_text())
    env_kwargs = info["env_info"]["env_kwargs"]
    control_mode = env_kwargs.get("control_mode") or info["episodes"][0].get("control_mode")
    if control_mode != args.control_mode:
        raise ValueError(f"dataset control mode {control_mode} != --control-mode {args.control_mode}")


def package_versions():
    out = {}
    for name in ["mani_skill", "sapien", "torch", "torchvision", "gymnasium", "diffusers", "numpy"]:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_optimizer(agent, args):
    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr_backbone},
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    # LR drops by 10x after 2/3 of training; stepped every batch as upstream does.
    scheduler = optim.lr_scheduler.StepLR(optimizer, int((2 / 3) * args.total_iters))
    return optimizer, scheduler


def find_resume_checkpoint(args, run_dir):
    if args.resume == "none":
        if (run_dir / "checkpoints" / "latest.pt").exists():
            raise FileExistsError(f"{run_dir} already has checkpoints; use --resume auto or a new --exp-name")
        return None
    if args.resume == "auto":
        latest = run_dir / "checkpoints" / "latest.pt"
        return latest if latest.exists() else None
    path = Path(args.resume)
    if not path.exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    return path


def train(args: Args) -> int:
    """Run (or resume) training. Returns 0 when finished, 3 when stopped by --max-hours."""
    session_start = time.time()
    if args.exp_name is None:
        args.exp_name = f"{args.env_id}__act_rgb__{args.num_demos or 'all'}d__seed{args.seed}"
    run_dir = Path(args.runs_dir) / args.exp_name
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    if not Path(args.demo_path).exists():
        raise FileNotFoundError(f"demo file not found: {args.demo_path} (run scripts/prepare_data.sh)")
    check_demo_matches(args)
    device = select_device(args.device)
    sim_backend = select_sim_backend(args.sim_backend)
    resume_path = find_resume_checkpoint(args, run_dir)
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    if ckpt is not None:
        if not ckpt.get("resumable"):
            raise ValueError(f"{resume_path} holds policy weights only (no optimiser/RNG state) and cannot be "
                             "resumed exactly; resume from <run>/checkpoints/latest.pt")
        saved = ckpt["config"]
        diffs = {k: (saved.get(k), getattr(args, k)) for k in RESUME_MUST_MATCH if saved.get(k) != getattr(args, k)}
        if diffs:
            raise ValueError(f"resume config mismatch (saved, now): {diffs}")

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    dataset = ACTDemoDataset(args.demo_path, args.num_queries, args.control_mode, num_traj=args.num_demos,
                             include_depth=args.include_depth)
    start_step = ckpt["step"] if ckpt else 0
    loader = DataLoader(
        dataset,
        batch_sampler=ResumableBatchSampler(len(dataset), args.batch_size, args.total_iters, args.seed, start=start_step),
        num_workers=args.num_dataload_workers,
        pin_memory=device.type == "cuda",
        # Own generator: otherwise starting iteration draws from the global RNG, which shifts dropout/CVAE noise on resume.
        generator=torch.Generator().manual_seed(args.seed),
    )

    from diffusers.training_utils import EMAModel  # imported late: slow import

    agent = ACTAgent(dataset.state_dim, dataset.act_dim, args).to(device)
    ema_agent = ACTAgent(dataset.state_dim, dataset.act_dim, args).to(device)
    optimizer, scheduler = build_optimizer(agent, args)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    best = {"success_once": -1.0, "success_at_end": -1.0}

    if ckpt is not None:
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)
        best = ckpt["best"]
        print(f"Resumed from {resume_path} at step {start_step}")

    config = {k: v for k, v in asdict(args).items()}
    session = {
        "type": "session", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "start_step": start_step, "resumed_from": str(resume_path) if resume_path else None,
        "device": str(device), "sim_backend": sim_backend,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": package_versions(), "git": git_info(),
        "dataset": {"path": args.demo_path, "sha256": sha256_file(args.demo_path), "episodes": dataset.num_traj,
                    "samples": len(dataset), "episode_keys": dataset.keys},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    if ckpt is None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    append_jsonl(run_dir / "sessions.jsonl", session)
    print(f"Run {args.exp_name}: device={device} sim_backend={sim_backend} samples={len(dataset)} "
          f"steps {start_step}->{args.total_iters}")

    eval_envs = make_eval_envs(
        args.env_id, args.num_eval_envs, sim_backend, args.control_mode, args.max_episode_steps,
        include_depth=args.include_depth, video_dir=str(run_dir / "videos") if args.capture_video else None,
    )

    def save(tag, step, full=False):
        """full=True stores everything needed to resume (~5x larger); otherwise only the evaluated EMA policy."""
        ema.copy_to(ema_agent.parameters())
        state = {"format": CKPT_FORMAT, "step": step, "config": config, "norm_stats": dataset.norm_stats,
                 "ema_agent": ema_agent.state_dict(), "resumable": full}
        if full:
            state.update(agent=agent.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                         lr_scheduler=scheduler.state_dict(), rng=get_rng_state(), best=best)
        atomic_torch_save(state, ckpt_dir / f"{tag}.pt")

    def run_eval(step):
        # Evaluation must not perturb training randomness (in-process envs draw from the global RNGs).
        rng_before = get_rng_state()
        ema.copy_to(ema_agent.parameters())
        t0 = time.time()
        result = evaluate(ema_agent, eval_envs, args.num_eval_episodes, args.num_queries, args.temporal_agg,
                          args.max_episode_steps, device, sim_backend, norm_stats=dataset.norm_stats, seed=args.eval_seed)
        s = result["summary"]
        append_jsonl(metrics_path, {"type": "eval", "step": step, "seconds": round(time.time() - t0, 1),
                                    "eval_seed": args.eval_seed, "sim_backend": sim_backend, **s})
        (run_dir / "eval").mkdir(exist_ok=True)
        np.savez_compressed(run_dir / "eval" / f"step_{step:07d}.npz", **result["per_episode"])
        print(f"[eval] step {step}: success_once {s['success_once']['count']}/{s['n_trials']}, "
              f"success_at_end {s['success_at_end']['count']}/{s['n_trials']} ({time.time() - t0:.0f}s)")
        set_rng_state(rng_before)
        for k in best:
            if s[k]["rate"] > best[k]:
                best[k] = s[k]["rate"]
                save(f"best_eval_{k}", step)

    is_eval_step = lambda step: step % args.eval_freq == 0 or step == args.total_iters

    if start_step == 0 and not args.skip_initial_eval:
        run_eval(0)
    if ckpt is not None:
        # Restore last: building the eval envs above draws from the global RNGs.
        set_rng_state(ckpt["rng"])

    agent.train()
    step = start_step
    exit_code = 0
    t_log = time.time()
    for batch in loader:
        obs = {k: v.to(device, non_blocking=True) for k, v in batch["observations"].items()}
        actions = batch["actions"].to(device, non_blocking=True)
        loss_dict = agent.compute_loss(obs=obs, action_seq=actions)
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()
        scheduler.step()
        ema.step(agent.parameters())
        step += 1

        if step % args.log_freq == 0 or step == args.total_iters:
            append_jsonl(metrics_path, {
                "type": "train", "step": step, "loss": loss_dict["loss"].item(), "l1": loss_dict["l1"].item(),
                "kl": loss_dict["kl"].item(), "lr": optimizer.param_groups[0]["lr"],
                "lr_backbone": optimizer.param_groups[1]["lr"],
                "sec_per_step": round((time.time() - t_log) / args.log_freq, 4),
            })
            print(f"step {step}: loss {loss_dict['loss'].item():.4f}")
            t_log = time.time()

        if is_eval_step(step):
            run_eval(step)
        if args.save_freq and step % args.save_freq == 0:
            save(f"step_{step:07d}", step)
        if step % args.ckpt_freq == 0 or is_eval_step(step) or step == args.total_iters:
            save("latest", step, full=True)

        out_of_time = args.max_hours is not None and time.time() - session_start > args.max_hours * 3600
        stop_requested = args.stop_at_step is not None and step >= args.stop_at_step
        if step < args.total_iters and (out_of_time or stop_requested):
            save("latest", step, full=True)
            print(f"Stopped at step {step} ({'time budget' if out_of_time else 'stop_at_step'}); "
                  "checkpoint saved. Rerun the same command to resume.")
            exit_code = 3
            break

    if exit_code == 0:
        save("final", step)
        (run_dir / "DONE").write_text(f"completed {step} steps\n")
    eval_envs.close()
    return exit_code


def main(argv=None):
    import tyro

    argv = sys.argv[1:] if argv is None else argv
    # Optional --config file.json supplies defaults; explicit CLI flags override it.
    defaults = Args()
    if "--config" in argv:
        i = argv.index("--config")
        overrides = json.loads(Path(argv[i + 1]).read_text())
        overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        unknown = set(overrides) - {f.name for f in fields(Args)}
        if unknown:
            raise ValueError(f"unknown keys in {argv[i + 1]}: {sorted(unknown)}")
        defaults = Args(**overrides)
        argv = argv[:i] + argv[i + 2:]
    args = tyro.cli(Args, default=defaults, args=argv)
    return train(args)
