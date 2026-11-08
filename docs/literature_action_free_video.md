# Build the IDM baseline first, then a clean LAM

> Literature review compiled with AI-assisted deep research on 2026-11-08 (licences checked on GitHub that day). Some 2026 entries were found via search only and are marked as unverified in the text; check primary sources before citing.

The best plan for a small, honest PickCube-v1 study is to write the core methods yourself in a few hundred lines of PyTorch and cite the papers that define them, rather than adopt a large codebase. The first arm is **VPT-style inverse-dynamics (IDM) pseudo-labelling**, as formalised in Morin et al. (ICML 2026, MIT code). The second is a **CLAM-style continuous latent action model whose action decoder trains jointly on the few labelled demos**, written clean-room and checked against LAOM's Apache-2.0 code. The third is plain BC and ACT/DP trained on the labelled demos alone. The current CLAM repo cannot legally go into a public GitHub project: it has **no licence**, imports a trainer that does not exist, and its authors marked its action indexing "TODO: fix this". That makes your Mac workarounds a reason to drop the code, not to build on it. The literature also predicts the outcome, which makes this a sharp test rather than a fishing trip. Two studies found that, with no distractors, IDM pseudo-labelling is a strong and often hard-to-beat baseline: Morin et al. 2026 and Nikulin et al. ICML 2025. CLAM's own ablation finds continuous, jointly trained latents essential (about 74% vs 15–23% on MetaWorld). No published paper runs this BC vs IDM vs LAM label-budget sweep on ManiSkill, so a careful version is a real, if modest, contribution. The biggest design risk is not the method choice but a **floor effect**. Your ACT RGB baseline reaches 18/100 with 100 labelled demos, so unless the unlabelled pool is large (about 1,000 motion-planning demos, where ACT/DP reach 0.98–1.00 in the ManiSkill3 paper), every arm will sit near zero and the comparison will say nothing.

## Twenty-odd candidates shrink to three that fit a T4 and a public licence

The four method families differ sharply in what a free T4 or a Mac can run, and in what licence the code carries. The table below covers the candidates that matter. Licences were checked against the GitHub API and the LICENSE files on 2026-11-08. "Re-implement" means the idea is easy to write from scratch, or the code cannot be redistributed.

| Method (family) | Paper / venue | Core idea | Code / licence | Framework | Compute (reported) | Benchmarks | ManiSkill | T4/Mac fit and adaptation |
|---|---|---|---|---|---|---|---|---|
| **IDM labelling (VPT/BCO recipe)** | Morin et al., arXiv 2602.02762, ICML 2026 | IDM trained on labelled data pseudo-labels action-free demos; BC on the union | `sachaMorin/idm-ssil` **MIT**; sub-repos (`uva`, `lapo-plus`) unchecked | PyTorch | 4xH100 for their UVA-based runs; the method itself is a small regressor | Procgen, Push-T, LIBERO-10 | None | **Excellent**: a ResNet-18 IDM at 128x128 trains in hours; low difficulty |
| **CLAM** (LAM) | arXiv 2505.04999, 2025; venue unconfirmed | Continuous latent actions from future prediction plus an action decoder trained jointly on a few labels | `clamrobot/clam` **no licence**, missing module, TODO action indexing | PyTorch + TensorFlow import | Not reported | DMControl, MetaWorld, CALVIN, WidowX | None | Idea fits well; **code unusable for publication**; re-implement |
| **LAOM** (LAM) | Nikulin et al., arXiv 2502.00379, ICML 2025 | LAPO with continuous latents, no quantisation, and supervision from about 2.5% labels | `dunnolab/laom` **Apache-2.0** | PyTorch (inferred) | Small; 64x64 images | Distracting Control Suite | None | **Good**: legal reference implementation; moderate porting |
| **LAPO** (LAM) | Schmidt & Jiang, arXiv 2312.10812, ICLR 2024 | Latent IDM + forward model, latent BC, decode with few labels | `schmidtdominik/LAPO` **no licence** | PyTorch (inferred) | About 1 h per stage on one GPU; 40 GB RAM by default | Procgen | None | Feasible but **cannot be redistributed**; use as quantised ablation, re-implemented |
| LAPA / UniVLA / villa-X / Moto | ICLR 2025 / RSS 2025 / 2025 / ICCV 2025 | VQ latent actions used to pretrain VLAs | MIT / Apache-2.0 / MIT / Apache-2.0 | JAX VLA (LAPA); PyTorch | 8 H100 x 34 h (LAPA); about 960 A100-h (UniVLA); 955M-param LAM (villa-X) | LIBERO, CALVIN, SimplerEnv | Only indirect, via SimplerEnv on ManiSkill2 | **Infeasible** to train; at most frozen-LAM inference |
| tinyworlds / open-genie / Jafar | Genie reproductions (Genie, arXiv 2402.15391) | FSQ/VQ latent action tokeniser inside a world model | MIT / MIT / Apache-2.0 | PyTorch / PyTorch / JAX | Small | Games | None | Liftable LAM modules; no action decoder |
| **ATM** (tracks) | Wen et al., arXiv 2401.00025, RSS 2024 | Track transformer pretrained on action-free video guides a BC policy | **MIT** | PyTorch | Not reported | LIBERO | None | Moderate; CoTracker preprocessing; GT-track oracle possible |
| HinFlow (tracks) | arXiv 2512.19269, 2025–26 | 2D flow planner + flow-conditioned policy, hindsight online relabelling | **No licence** | PyTorch | One RTX 3090: about 30 min pretraining, about 11 h online | LIBERO + 3 ManiSkill tasks (not PickCube) | **Yes**, but `pd_ee_delta_pose` | Closest ManiSkill fit, but not publishable code |
| AMPLIFY (tracks) | arXiv 2506.14198, 2025 | FSQ motion tokens; forward model on video, IDM on few labels | **MIT** | PyTorch | Not reported | LIBERO | None | Moderate; conceptually aligned |
| **AVDC** (video prediction) | Ko et al., arXiv 2310.08576, ICLR 2024 | Video diffusion, then optical flow + depth, then rigid transform; needs no labels | **MIT** | PyTorch | 4xV100-32GB, about 24 h; 201M params at 128x128 | Meta-World, iTHOR | None | Heavy; needs a scaled-down UNet and a grasp heuristic |
| Seer / GR-1 / UVA | ICLR 2025 / 2023 / 2025 | Foresight + IDM; video pretraining; joint video-action diffusion | Apache-2.0 / Apache-2.0 / MIT | PyTorch | 8x4090 (Seer); 46M trainable (GR-1) | CALVIN, LIBERO, Push-T | None | Hard to port; UVA most approachable |
| UWM, VPP, Vidar, Dreamitate, This&That | 2024–2025 | SVD-scale or unified world models | None / MIT / None / CC BY-NC / None | PyTorch | 24–80 GB GPUs | CALVIN, LIBERO, DROID | None | **Infeasible** |

Sources for the table: Morin et al. ([arXiv](https://arxiv.org/abs/2602.02762), [code](https://github.com/sachaMorin/idm-ssil)); CLAM ([arXiv](https://arxiv.org/abs/2505.04999), [repo](https://github.com/clamrobot/clam)); LAOM ([arXiv](https://arxiv.org/abs/2502.00379), [repo](https://github.com/dunnolab/laom)); LAPO ([repo](https://github.com/schmidtdominik/LAPO)); LAPA ([repo](https://github.com/LatentActionPretraining/LAPA)); UniVLA ([repo](https://github.com/OpenDriveLab/UniVLA)); villa-X ([repo](https://github.com/microsoft/villa-x)); Moto ([licence](https://github.com/TencentARC/Moto/blob/main/LICENSE.txt)); tinyworlds ([repo](https://github.com/AlmondGod/tinyworlds)); ATM ([repo](https://github.com/Large-Trajectory-Model/ATM)); HinFlow ([paper](https://arxiv.org/html/2512.19269), [repo](https://github.com/yzc0731/HinFlow)); AMPLIFY ([arXiv](https://arxiv.org/abs/2506.14198), [repo](https://github.com/pairlab/AMPLIFY)); AVDC ([paper](https://arxiv.org/pdf/2310.08576), [repo](https://github.com/flow-diffusion/AVDC)); Seer ([paper](https://arxiv.org/pdf/2412.15109)); GR-1 ([paper](https://arxiv.org/pdf/2312.13139)); UVA ([repo](https://github.com/ShuangLI59/unified_video_action)); UWM ([repo](https://github.com/WEIRDLabUW/unified-world-model)); Dreamitate ([repo](https://github.com/cvlab-columbia/dreamitate)).

Two patterns stand out. First, **none of the latent-action or IDM methods ships ManiSkill 3 support**. LAPA, UniVLA and Moto touch ManiSkill only through SimplerEnv, which is built on ManiSkill2 ([SimplerEnv](https://github.com/simpler-env/SimplerEnv)). Every arm therefore needs a data adapter, and you already have one in your validated RGB-D pipeline. Second, the licences split cleanly. **MIT/Apache code exists for every family** (Morin et al., LAOM, ATM, AMPLIFY, AVDC). The repos most tempting for ManiSkill (CLAM, LAPO, HinFlow) carry no licence, which means default all-rights-reserved. You can read them and cite them. Copying them into a public repository is a different matter.

## IDM pseudo-labelling is the arm most likely to work, and the one reviewers will expect

Morin et al. 2026 is the closest published template for your study. It defines "IDM labeling" as training an IDM on a small labelled set D_L, pseudo-labelling an action-free set D_U, and behaviour-cloning on the union. It compares this against BC, a video-model-plus-IDM variant (VM-IDM) and LAPO/LAPO+, using **about 5% labelled demos** in which D_U is the same task's expert demos with actions removed ([arXiv](https://arxiv.org/abs/2602.02762)). That is your setup almost exactly. On Push-T and LIBERO-10, "IDM-based policies outperform BC in most settings and by an especially large margin on Push-T". The authors attribute the Push-T margin to the action being a simple function of visible state change ([paper PDF](https://arxiv.org/pdf/2602.02762)). Their theory explains why: the IDM's advantage comes from a lower-complexity hypothesis class and lower stochasticity than the expert policy.

PickCube with `pd_ee_delta_pos` looks like Push-T in this respect. A 3-D end-effector delta is nearly visible between two consecutive 128x128 frames from a fixed camera, so an IDM should learn it from very few demos. The gripper dimension is the exception. Its open/close command shows up only indirectly, as the fingers move over the next frames, so it is the likeliest source of pseudo-label error. This is an inference, not a published result. It argues for reporting **per-dimension IDM error on held-out labelled demos**, and for letting the IDM see a short frame stack (Morin et al. use 4 frames) rather than one pair.

The same theory also predicts the IDM's main risk. Morin et al. argue the IDM's gains shrink when the expert policy is already low-complexity and near-deterministic, and they list "D_U is curated in-task expert data" as a limitation ([paper PDF](https://arxiv.org/pdf/2602.02762)). ManiSkill's motion-planning demos are about as deterministic as experts get. So plain BC and ACT/DP might close the gap faster than in their benchmarks. That is a result worth reporting either way. It is also why you should include very small budgets (N = 5 or 10).

DynaMo is a useful orthogonal comparison if you want a fourth arm later. It pretrains the visual encoder with self-supervised inverse and forward dynamics on all frames, then trains a policy head on the labelled data, with no pseudo-labels at all. It needs 1–3 single-GPU hours of pretraining and has MIT code ([arXiv](https://arxiv.org/pdf/2409.12192), [repo](https://github.com/jeffacce/dynamo_ssl)).

## A clean-room continuous LAM beats patching the official CLAM repo

The evidence favours CLAM's design, not its code. CLAM's MetaWorld ablation reports about **16% for discrete latents without joint training, 15% for discrete with joint training, 23% for continuous without joint training, and 74% for continuous with joint training** ([arXiv v2](https://arxiv.org/html/2505.04999v2)). Image-based MetaWorld results follow the same order: 76% for ST-ViViT-CLAM against 20% for LAPA and 9% for LAPO, with 100 labelled trajectories. LAOM (ICML 2025) supports the same conclusion from a different direction. On distractor-laden control tasks, quantisation hurt latent quality even without distractors. Supervising the LAM with ground-truth actions on just **2.5% of the data gave about a 4.2x downstream improvement** ([arXiv](https://arxiv.org/abs/2502.00379), [HTML](https://arxiv.org/html/2502.00379)). The November 2026 ELA preprint goes further and trains LAM, latent policy and decoder end-to-end with "a handful" of labelled demos ([arXiv](https://arxiv.org/abs/2611.05571)). Its code status is unknown.

The official CLAM repository has four problems that rule it out for a public project. **The licence field is null and the root holds no LICENSE file.** `clam/trainers/__init__.py` imports a `diffusion_clam_trainer` module that does not exist. `main.py` imports TensorFlow and a top-level `resolvers` module. The README's data-generation section is a "TODO" ([repo](https://github.com/clamrobot/clam), [trainers/__init__.py](https://github.com/clamrobot/clam/blob/main/clam/trainers/__init__.py)). The project page links a second repo that now returns 404 ([project page](https://liralab.usc.edu/clam/)).

The action indexing confirms your a_{t+1} suspicion as far as the code can show it. The joint-training branch uses `batch.actions[:, context_len:]`, while the offline decoder uses `batch.actions[:, -2]`. Both are flagged "TODO: fix this" ([clam_trainer.py](https://github.com/clamrobot/clam/blob/main/clam/trainers/clam_trainer.py), [action_decoder_trainer.py](https://github.com/clamrobot/clam/blob/main/clam/trainers/action_decoder_trainer.py)). If a_i is the action taken at o_i, the joint branch labels the transition o_{c-1}→o_c with a_c, a one-step offset. Whether that offset is a real bug depends on dataset conventions the repo never released, so it stays unproven. The point for you: your current CLAM numbers rest on an indexing choice the authors themselves did not trust.

A CLAM-style LAM for PickCube needs four parts:

- a small CNN latent IDM, (o_t, o_{t+1}) → continuous z of dimension 8–16;
- a forward model, (o_t, z) → o_{t+1} (or its features);
- an MLP decoder z → a_t, trained jointly on the labelled transitions;
- a latent BC policy, decoded to actions at test time.

That is roughly 300 lines. Write it from the paper equations. Use **LAOM's Apache-2.0 code** as the legally reusable reference for training details, since LAOM also renders 64x64 images and ships a labelled-subset sampler ([repo](https://github.com/dunnolab/laom)). A quantised (FSQ/VQ) variant of the same model with the decoder trained post hoc gives you a LAPO-style ablation for free. It also tests the continuous-and-joint claim on a grasping task, which none of these papers covers.

Two cheap diagnostics make the LAM arm much more credible than CLAM's own paper. CLAM reports downstream success only and never measures latents against ground-truth actions ([arXiv v2](https://arxiv.org/html/2505.04999v2)). LAOM's **linear probe** fixes this: a linear map from z to the true `pd_ee_delta_pos`, trained without gradients flowing into the encoder ([arXiv HTML](https://arxiv.org/html/2502.00379)). Since your pipeline has ground-truth actions for every demo, you can report the probe's R² as a function of N for free.

## Track- and video-prediction arms are worth a footnote, not a headline

The point-track family is the most interesting extension, but its best ManiSkill fit can't be published. HinFlow is the only method found whose official code supports ManiSkill tasks. It runs on 128x128 RGB on one RTX 3090, with about 30 minutes of pretraining and about 11 hours of online relabelling. But it targets Place Sphere, Pull Cube Tool and Poke Cube rather than PickCube, needs `pd_ee_delta_pose` rather than your `pd_ee_delta_pos`, and **has no licence file** ([paper](https://arxiv.org/html/2512.19269), [repo](https://github.com/yzc0731/HinFlow)). ATM (MIT) and AMPLIFY (MIT) are the publishable alternatives, both built around CoTracker preprocessing on LIBERO ([ATM](https://github.com/Large-Trajectory-Model/ATM), [AMPLIFY](https://github.com/pairlab/AMPLIFY)).

ManiSkill's real advantage here is the oracle arm. You can project points sampled on the cube through simulator object poses and camera intrinsics to get exact 2D and 3D tracks. That separates tracker error from policy error, as long as you label the arm as privileged. One unverified risk: a cube rendered at 128x128 covers only a few to tens of pixels, so CoTracker accuracy may be poor.

Video prediction is the weakest bet at this compute scale. UniPi has no official code. In AVDC's Meta-World table, a UniPi-style video-plus-IDM baseline scored **6.1% against 16.2% for BC-Scratch and 43.1% for full AVDC** ([AVDC paper](https://arxiv.org/pdf/2310.08576)). AVDC's own inverse-dynamics ablation also scored 6.1%. AVDC itself is MIT-licensed and native at 128x128, but its reference cost is 4xV100-32GB for about 24 hours on a 201M-parameter model. Its flow-to-action step also needs depth, segmentation and a grasp heuristic to produce `pd_ee_delta_pos`. Everything newer (Seer, VPP, UWM, Vidar, Dreamitate) is SVD- or VLA-scale. If you want a video arm, the defensible minimum is Morin et al.'s VM-IDM: a small next-frame predictor feeding the same IDM. They show it equals IDM labelling in the infinite-data limit, and that it trails IDM labelling slightly in practice ([arXiv](https://arxiv.org/abs/2602.02762)).

## The experiment must be designed around a 0.98 oracle, not an 18% baseline

The ManiSkill3 paper's PickCube-v1 tables report *best success during training* from motion-planning demos ([ManiSkill3 paper](https://arxiv.org/html/2410.00425)):

| Method | 100 demos (state) | 100 demos (RGB) | 1000 demos (state) | 1000 demos (RGB) |
|---|---|---|---|---|
| BC | 0.00 | 0.00 | 0.03 | 0.03 |
| ACT | 0.70 | 0.28 | 1.00 | 0.98 |
| DP | 1.00 | 0.76 | 1.00 | 1.00 |

Your 18/100 *final* ACT RGB result at 30k steps is consistent with the paper's 0.28 *best*. The difference is plausibly explained by four things:

- best checkpoint vs final checkpoint;
- unknown training-step counts in the paper;
- evaluation noise: at p ≈ 0.2 with 100 episodes, the binomial standard error is about 4 points, so the 95% interval is about ±8;
- possible differences in episode length or camera settings.

Report both best and final, and say which the paper used. The official baselines page still marks results "WIP", so these appendix numbers are the only published anchor ([ManiSkill docs](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/baselines.html)).

The table has a direct design consequence. Every pseudo-labelling or LAM arm is bounded above by the same policy trained on *all* demos with true labels. With D_U = 100 demos and an ACT backbone, that ceiling is about 0.2–0.3 and every arm will crowd near zero. With **D_U = 1,000 demos, the oracle reaches 0.98 for ACT and 1.00 for DP**, leaving a wide band in which IDM and LAM can show gains.

Plain MLP BC sits at 0.00–0.03 even with 1,000 demos. Treat it as a floor, not a competitor. The honest action-only baseline is ACT or DP trained on the N labelled demos. Keep the downstream policy architecture fixed across arms, so only the label source changes.

The resulting design has four pieces:

- **Arms:** five, all using one policy backbone:
  - BC (MLP) on D_L;
  - ACT or DP on D_L;
  - IDM labelling on D_L ∪ D_U;
  - continuous LAM with a joint decoder;
  - an all-labels oracle.
  - A quantised LAM is an optional ablation.
- **Label sweep:** N ∈ {5, 10, 25, 50, 100}. Use at least 3 seeds and resample D_L each seed, as Morin et al. do ([paper PDF](https://arxiv.org/pdf/2602.02762)).
- **Diagnostics:**
  - IDM per-dimension error;
  - LAM linear-probe R²;
  - success over at least 100 episodes.
- **Fast track:** state observations for quick iteration, with the RGB results as the headline.

On compute, the IDM and LAM models are a few million parameters and should train in hours on a T4. That is an estimate, not a measurement. ManiSkill GPU-parallel simulation keeps evaluation cheap (RGB PPO solves PickCube in about 10 minutes on GPU per the [ManiSkill3 paper](https://arxiv.org/html/2410.00425)). Three hardware constraints apply:

- the T4 has no bf16, so use fp16;
- on Mac MPS, avoid anything that requires JAX, Flash-Attention or hard-coded `.cuda()` calls (which rules out LAPA's VLA, UniVLA training and villa-X);
- Kaggle's session limit was not verified in this research, so checkpoint often.

Do the time alignment once, in your own data loader. Pair a_t with the transition o_t→o_{t+1}. Add a unit test showing that an IDM trained on the full labelled set reaches near-zero error at that alignment and clearly worse error at a_{t+1}. That one test turns the CLAM indexing ambiguity into a documented fact about your pipeline.

## Conclusion

The ranked recommendation:

1. **IDM pseudo-labelling.** Write it from scratch and cite VPT and Morin et al.; the reference code is MIT.
2. **A continuous LAM with a jointly trained decoder.** Re-implement CLAM's idea clean-room, with LAOM's Apache-2.0 code as the legal reference and a quantised variant as an ablation.
3. **ACT/DP on the labelled demos only**, as the action-only reference.

ATM-style tracks with a simulator oracle are a strong follow-up paper, not part of version one. Video-generation methods should be cited and left out.

The main risks run in an unusual direction. The danger is less that the fancy method fails and more that the experiment cannot tell the arms apart. Three things could cause that:

- a floor effect if D_U is too small;
- a near-deterministic motion-planning expert that lets ACT/DP catch up quickly;
- label misalignment that quietly handicaps one arm.

Results consistent with the literature would look like this: IDM labelling roughly matching the LAM on clean renders, the LAM ahead only at the smallest N, and the quantised LAM last. That would still be a useful finding, because it would be the first such sweep on a grasping task in a GPU simulator. Confirm that "first" claim with a final literature check before using it, since the 2026 LAM literature moves weekly.

The project's credibility rests on a few things that cost little:

- MIT/Apache-only dependencies;
- best and final success reported side by side;
- resampled seeds;
- linear probes and per-dimension IDM errors.

Replicating CLAM's headline number would not add to that credibility.
