# Novelty check: better pseudo-action labels (2026-11-15)

> Based on web searches on 2026-11-15 (search-engine summaries plus abstracts; full texts not yet read). "Not
> found" means these searches did not find it, not that it does not exist. Read the closest papers before claiming
> anything is new.

## Why this check

The noise sweep (`results/action_free/noise_sweep/README.md`) found that pseudo-labels fail because of their error
*structure*, not their size. Random errors as large as the IDM's are nearly harmless (93/100 vs 95/100), while
shrinking the horizontal motion (×0.7: 25/100, ×0.4: 1/100) or making errors correlated over time (52/100) breaks
the policy, although shrunk labels can have high per-step R² (0.91). One training seed per condition. Three method
ideas followed:

1. **Chunk-level IDM:** predict the motion over a whole action chunk, not one step.
2. **Distributional IDM:** predict a distribution of actions (mixture, diffusion or autoregressive tokens) instead
   of the mean, which is shrunk under uncertainty.
3. **Position-based labels:** estimate where the gripper is in each frame (a heatmap), smooth the path over time
   (e.g. a Kalman smoother) and take actions as position differences. Ideas 2 and 3 can be combined: a probabilistic
   tracker.

## What already exists

| Idea | Closest prior work found | Verdict |
|---|---|---|
| 1. Multi-step / chunk IDM | RoboCurate (arXiv 2602.18742): a diffusion-transformer IDM that takes a current and a future frame and outputs the *action sequence* between them, used to pseudo-label video for policy training | **Not new as a method.** |
| 2. Distributional IDM | RoboCurate (diffusion / flow-matching IDM); UVA (arXiv 2503.00200, diffusion heads, can act as an IDM); Vidar (arXiv 2507.12898, masked IDM with video diffusion) | **Not new as a method.** No paper found that motivates it by shrinkage of the mean prediction. |
| 3. Pose-based labels | Song et al. (arXiv 2504.13803): labels human demos with end-effector poses from a known tool model, actions as next-step pose; GVF-TAPE (arXiv 2509.00361): end-effector poses from generated frames, converted to actions; EC-Flow (ICCV 2025): embodiment-centric flow plus the robot URDF to compute actions; PoseDiff (arXiv 2509.24591): unified robot pose estimation and video-to-action, compared against RoboPEPP and Seer's IDM | **Not new as an idea.** Heatmap robot keypoints are also established: DREAM (arXiv 1911.09231), CtRNet / CtRNet-X (arXiv 2409.10441). |
| 2 + 3 combined | No paper found with a probabilistic gripper tracker plus temporal smoothing (e.g. Kalman / RTS) to produce uncertainty-weighted action labels | **Possibly new, but incremental.** On its own it reads as engineering. |
| Theory of IDM labelling | Morin et al. (arXiv 2602.02762, ICML 2026): sample efficiency of IDM labelling; Schäfer et al. (arXiv 2601.21718, ICML 2026): bias-variance trade-off of predictive inverse dynamics vs BC | Related framing; neither, as far as the abstracts show, studies the *structure* of label errors. |
| Label-quality metric that predicts policy success | Latent-action papers use per-step error or linear probes (LAOM, arXiv 2502.00379); mixed evidence on probes predicting success (arXiv 2502.09680 for, arXiv 2608.18746 against); a world-model diagnostic score that ranks planning success (arXiv 2606.09028) | **Not found for pseudo-action labels.** |
| Controlled study: which label errors hurt a chunked policy (shrinkage, temporal correlation vs i.i.d. noise of equal size) | Not found. Related: action-chunking theory on compounding error (arXiv 2507.09061, ICLR 2026; arXiv 2608.02547) | **Not found: the most original part so far.** |

## What this means

- A paper whose headline is a *new labeller* (chunk IDM, diffusion IDM, pose-based labels) would be incremental:
  each piece exists.
- The open gap is the **analysis**: *what makes pseudo-action labels useful to a chunked visual policy*. The noise
  sweep already answers part of that with controlled evidence. No work found shows that error structure (scale
  bias, temporal correlation, chunk-level displacement error) rather than per-step accuracy determines success.
- Suggested framing: **"Pseudo-action labels fail through bias and persistence, not noise"**, with three
  contributions:
  1. the controlled error-structure study (done for one task; needs seeds and tasks);
  2. a cheap label-quality metric (chunk-displacement R² plus scale/slope) shown to predict policy success across
     many labellers, compared against per-step R²;
  3. evidence that labellers built to avoid these flaws do better: plain IDM, rescaled IDM, chunk IDM, a
     diffusion IDM and pose-differencing, all scored by the metric and by policy success. The probabilistic tracker
     can be one of these labellers rather than the whole contribution.

## Before committing to this

- Read RoboCurate, PoseDiff, Morin et al. and Schäfer et al. in full. Check whether any reports scale bias or
  chunk-level error of its labels, which would weaken contribution 2.
- Search citing papers of Morin et al. (2602.02762) and LAOM (2502.00379) for newer label-quality studies.
- Extend beyond PickCube (2–3 ManiSkill tasks) and to ≥ 3 seeds; one task and one seed will not convince reviewers.

## Sources

Morin et al. <https://arxiv.org/html/2602.02762> · Schäfer et al. <https://arxiv.org/html/2601.21718v1> ·
RoboCurate <https://arxiv.org/pdf/2602.18742> · UVA <https://arxiv.org/pdf/2503.00200v1> ·
Vidar <https://arxiv.org/pdf/2507.12898v4> · CoMo <https://arxiv.org/html/2505.17006v1> ·
Song et al. <https://arxiv.org/pdf/2504.13803> · GVF-TAPE <https://arxiv.org/pdf/2509.00361> ·
EC-Flow <https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_EC-Flow_Enabling_Versatile_Robotic_Manipulation_from_Action-Unlabeled_Videos_via_Embodiment-Centric_ICCV_2025_paper.pdf> ·
PoseDiff <https://arxiv.org/pdf/2509.24591> · DREAM <https://arxiv.org/pdf/1911.09231> ·
CtRNet-X <https://arxiv.org/pdf/2409.10441> · LAOM <https://arxiv.org/html/2502.00379v5> ·
object-centric LAM <https://arxiv.org/pdf/2502.09680> · latent world model metrics <https://arxiv.org/pdf/2608.18746> ·
diagnostic score <https://arxiv.org/html/2606.09028> · action chunking theory <https://arxiv.org/pdf/2507.09061>,
<https://arxiv.org/pdf/2608.02547>
