import pytest
import torch

from src.data.demo_dataset import ACTDemoDataset, episode_keys
from src.models.act_agent import ACTAgent
from src.training.train_act import Args


@pytest.fixture(scope="module")
def dataset(demo_path):
    return ACTDemoDataset(demo_path, num_queries=30, control_mode="pd_ee_delta_pos", num_traj=3)


def test_episode_keys_numeric_order(demo_path):
    keys = episode_keys(demo_path, 12)
    assert keys[:3] == ["traj_0", "traj_1", "traj_2"] and keys[-1] == "traj_11"
    with pytest.raises(ValueError):
        episode_keys(demo_path, 10_000)


def test_dataset_shapes(dataset):
    assert dataset.keys == ["traj_0", "traj_1", "traj_2"]
    assert dataset.norm_stats is None  # delta control: no normalisation, as upstream
    item = dataset[0]
    assert item["observations"]["rgb"].shape == (1, 3, 224, 224)
    assert item["observations"]["rgb"].dtype == torch.uint8
    assert item["observations"]["state"].shape == (dataset.state_dim,)
    assert item["actions"].shape == (30, 4)


def test_dataset_pads_end_with_zero_arm_and_last_gripper(dataset):
    last = dataset.trajectories["actions"][0].shape[0] - 1
    actions = dataset[last]["actions"]  # only the first row is real
    assert torch.all(actions[1:, :3] == 0)
    assert torch.all(actions[1:, 3] == actions[0, 3])


def test_model_loss_and_action_shapes(dataset):
    torch.manual_seed(0)
    agent = ACTAgent(dataset.state_dim, dataset.act_dim, Args())
    batch = torch.utils.data.default_collate([dataset[i] for i in range(2)])
    loss = agent.compute_loss(batch["observations"], batch["actions"])
    assert set(loss) == {"l1", "kl", "loss"} and torch.isfinite(loss["loss"])
    loss["loss"].backward()
    agent.eval()
    with torch.no_grad():
        assert agent.get_action(batch["observations"]).shape == (2, 30, 4)
