"""ACT demonstration dataset.

Derived from `SmallDemoDataset_ACTPolicy` in ManiSkill v3.0.1
examples/baselines/act/train_rgbd.py (Apache-2.0). Changes: control mode is a
constructor argument instead of a global, and the episode keys used are recorded
so each run documents exactly which demonstrations it trained on.
"""
import numpy as np
import torch
import torchvision.transforms as T
from h5py import File
from torch.utils.data import Dataset

import src  # noqa: F401  puts the vendored `act` package on sys.path
from act.utils import load_content_from_h5_file
from mani_skill.utils import common


def episode_keys(path, num_traj=None):
    """Episode keys in numeric order (traj_0, traj_1, ...), truncated to num_traj."""
    with File(path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
    if num_traj is not None:
        if num_traj > len(keys):
            raise ValueError(f"num_demos={num_traj} but {path} has only {len(keys)} episodes")
        keys = keys[:num_traj]
    return keys


class ACTDemoDataset(Dataset):
    """Holds every (observation, action chunk) pair in memory, images resized to 224x224 uint8."""

    def __init__(self, data_path, num_queries, control_mode, num_traj=None, include_depth=False):
        self.keys = episode_keys(data_path, num_traj)
        with File(data_path, "r") as f:
            raw = [load_content_from_h5_file(f[k]) for k in self.keys]

        self.include_depth = include_depth
        self.control_mode = control_mode
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])

        observations = [self.process_obs(ep["obs"]) for ep in raw]
        actions = [torch.Tensor(ep["actions"]) for ep in raw]
        self.trajectories = {"observations": observations, "actions": actions}
        self.obs_keys = list(observations[0].keys())

        # At the end of a delta-control episode the arm holds still (zero delta) and the gripper keeps its last command.
        self.pad_action_arm = None
        if "delta_pos" in control_mode or control_mode == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros((actions[0].shape[1] - 1,))

        self.slices = [(i, ts) for i, a in enumerate(actions) for ts in range(a.shape[0])]
        self.num_traj = len(actions)
        self.num_queries = num_queries
        self.delta_control = "delta" in control_mode
        self.norm_stats = self.get_norm_stats() if not self.delta_control else None

    @property
    def state_dim(self):
        return self.trajectories["observations"][0]["state"].shape[1]

    @property
    def act_dim(self):
        return self.trajectories["actions"][0].shape[1]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]
        state = self.trajectories["observations"][traj_idx]["state"][ts]
        act_seq = self.trajectories["actions"][traj_idx][ts : ts + self.num_queries]
        action_len = act_seq.shape[0]

        # Pad past the end of the episode so every observation can be used for training.
        if action_len < self.num_queries:
            if self.pad_action_arm is not None:
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries - action_len, 1)], dim=0)
            elif not self.delta_control:
                act_seq = torch.cat([act_seq, act_seq[-1].repeat(self.num_queries - action_len, 1)], dim=0)

        if not self.delta_control:
            state = (state - self.norm_stats["state_mean"][0]) / self.norm_stats["state_std"][0]
            act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        obs = dict(state=state, rgb=self.trajectories["observations"][traj_idx]["rgb"][ts])
        if self.include_depth:
            obs["depth"] = self.trajectories["observations"][traj_idx]["depth"][ts]
        return {"observations": obs, "actions": act_seq}

    def process_obs(self, obs_dict):
        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        images_rgb, images_depth = [], []
        for cam_data in sensor_data.values():
            rgb = torch.from_numpy(cam_data["rgb"])  # (ep_len, H, W, 3)
            images_rgb.append(self.transforms(rgb.permute(0, 3, 1, 2)))  # (ep_len, 3, 224, 224)
            if self.include_depth:
                depth = torch.Tensor(cam_data["depth"].astype(np.float32) / 1024).to(torch.float16)
                images_depth.append(self.transforms(depth.permute(0, 3, 1, 2)))
        rgb = torch.stack(images_rgb, dim=1)  # (ep_len, num_cams, 3, 224, 224), uint8

        # Some extras (e.g. is_grasped) are 1-D; give them a feature axis before flattening.
        obs_dict["extra"] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict["extra"].items()}
        state = common.flatten_state_dict(obs_dict, use_torch=True)

        processed = dict(state=state, rgb=rgb)
        if self.include_depth:
            processed["depth"] = torch.stack(images_depth, dim=1)
        return processed

    def get_norm_stats(self):
        all_state, all_action = [], []
        for traj_idx, ts in self.slices:
            all_state.append(self.trajectories["observations"][traj_idx]["state"][ts])
            act_seq = self.trajectories["actions"][traj_idx][ts : ts + self.num_queries]
            if act_seq.shape[0] < self.num_queries:
                act_seq = torch.cat([act_seq, act_seq[-1].repeat(self.num_queries - act_seq.shape[0], 1)], dim=0)
            all_action.append(act_seq)
        all_state = torch.stack(all_state)
        all_action = torch.concatenate(all_action)
        return {
            "action_mean": all_action.mean(dim=0, keepdim=True),
            "action_std": torch.clip(all_action.std(dim=0, keepdim=True), 1e-2, np.inf),
            "state_mean": all_state.mean(dim=0, keepdim=True),
            "state_std": torch.clip(all_state.std(dim=0, keepdim=True), 1e-2, np.inf),
        }
