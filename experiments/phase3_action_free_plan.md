# Phase 3: learning from action-free demonstrations (plan, 2026-11-08)

**Question.** On ManiSkill PickCube-v1, when only N expert demos keep their actions and the rest are action-free video, how does policy success depend on N for (A) a policy trained on the N labelled demos only, (B) VPT-style IDM pseudo-labelling, and (C) a continuous latent action model with a jointly trained action decoder?

Background and method choice: `docs/literature_action_free_video.md` (deep-research review, 2026-11-08). In short: IDM pseudo-labelling follows Morin et al. (ICML 2026, MIT code); the latent action model is a clean-room re-implementation of the CLAM idea checked against LAOM (Apache-2.0). The official CLAM repository has no licence, so none of its code is used.

## Status

| Step | Status | Evidence |
|---|---|---|
| 1,000 demos converted (RGB, `pd_ee_delta_pos`) | Done | 1,000/1,000 successful |
| IDM pseudo-label quality, N = 5–100 × 3 seeds | Done | `results/action_free/idm/README.md` |
| Policy training on pseudo-labels (arm B) | Not started | |
| Latent action model (arm C) | Not started | |

## Policy experiment design

- Pools: labelled demos drawn from 0–99 (same seeded splits as the IDM sweep), action-free pool 100–999.
- Arms per N:
  - A: ACT on the N labelled demos only;
  - B: ACT on N labelled + 900 IDM-pseudo-labelled demos;
  - C: latent action model with a jointly trained decoder (later).
- Reference D: ACT on all 1,000 demos with true actions (the ceiling every arm is bounded by).
- Same ACT settings and step budget for every arm; evaluation with `enhanced_determinism` scenes (identical across runs), 100 trials, final and best checkpoints reported separately; failure analysis per final checkpoint.

## Compute (measured Phase A speed: ACT RGB 224×224, about 0.97 s/step on a Kaggle T4)

| Option | Runs | GPU-hours | Kaggle time (2 runs per session, ~30 h/week) |
|---|---|---|---|
| Minimal: N ∈ {10, 25, 100}, 1 seed, arms A + B, plus D | 7 | ~60 | ~2 weeks |
| Minimal + latent action arm C | 10 | ~85 | ~3 weeks |
| Full: N ∈ {5, 10, 25, 50, 100}, 3 seeds, A + B + C, plus D | 46 | ~400 | not feasible on free compute |

The 1,000-demo dataset at 224×224 occupies about 12 GB of RAM in ACT's in-memory loader. That fits Kaggle (about 29 GB) but not comfortably the Mac (24 GB). Feeding ACT the native 128×128 images would cut both memory and time, but changes the protocol relative to Phase A; if adopted, it must apply to every arm. Its speed-up has not been measured.

## Open decisions

1. Which compute option to run (recommended: Minimal, then add C).
2. Image resolution for the policy (224 as in Phase A, or native 128 for all arms).
3. Where pseudo-labels are produced: rerun the IDM on Kaggle-rendered frames (about 30 min of GPU, fully consistent), or upload the Mac pseudo-labels (renders differ in rare pixels).
