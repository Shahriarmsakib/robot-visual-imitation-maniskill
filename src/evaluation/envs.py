"""Evaluation environments.

Derived from `FlattenRGBDObservationWrapper` (examples/baselines/act/train_rgbd.py) and
`make_eval_envs` (examples/baselines/act/act/make_env.py) in ManiSkill v3.0.1 (Apache-2.0).
Changes: CPU vector envs use same-step autoreset so gymnasium >= 1.0 behaves like the
gymnasium 0.29 API the upstream evaluator was written for.
"""
import sys
from functools import partial
from typing import Dict, Optional

import gymnasium as gym
import torch
import torchvision.transforms as T

import mani_skill.envs  # noqa: F401  registers environments
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """Turns ManiSkill rgb/rgbd observations into {"state", "rgb"[, "depth"]} with 224x224 images."""

    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images_rgb, images_depth = [], []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                images_rgb.append(self.transforms(cam_data["rgb"].permute(0, 3, 1, 2)))
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                images_depth.append(self.transforms(depth.permute(0, 3, 1, 2)))

        ret = dict()
        if self.include_state:
            ret["state"] = common.flatten_state_dict(observation, use_torch=True)
        if self.include_rgb:
            ret["rgb"] = torch.stack(images_rgb, dim=1)  # (B, num_cams, 3, 224, 224), uint8
        if self.include_depth:
            ret["depth"] = torch.stack(images_depth, dim=1)
        return ret


def _cpu_vector_env(env_fns):
    kwargs = {}
    if hasattr(gym.vector, "AutoresetMode"):  # gymnasium >= 1.1
        kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    if len(env_fns) == 1:
        return gym.vector.SyncVectorEnv(env_fns, **kwargs)
    # macOS: workers forked from the forkserver share renderer state and only ~2 can render
    # (IndexError: unordered_map::at in take_picture); spawned workers start clean.
    context = "spawn" if sys.platform == "darwin" else "forkserver"
    return gym.vector.AsyncVectorEnv(env_fns, context=context, **kwargs)


def make_eval_envs(
    env_id: str,
    num_envs: int,
    sim_backend: str,
    control_mode: str,
    max_episode_steps: Optional[int] = None,
    include_depth: bool = False,
    video_dir: Optional[str] = None,
):
    """Vectorised evaluation envs. On CPU only the first env records video; on GPU all do."""
    env_kwargs = dict(
        control_mode=control_mode, reward_mode="sparse",
        obs_mode="rgbd" if include_depth else "rgb", render_mode="rgb_array",
    )
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=include_depth)]

    if sim_backend == "physx_cpu":
        def make(seed, record):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if record:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True,
                                        source_type="act", source_desc="act evaluation rollout")
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env
            return thunk

        return _cpu_vector_env([make(i, bool(video_dir) and i == 0) for i in range(num_envs)])

    env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=1, **env_kwargs)
    steps = gym_utils.find_max_episode_steps_value(env)
    for wrapper in wrappers:
        env = wrapper(env)
    if video_dir:
        env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True,
                            source_type="act", source_desc="act evaluation rollout", max_steps_per_video=steps)
    return ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
