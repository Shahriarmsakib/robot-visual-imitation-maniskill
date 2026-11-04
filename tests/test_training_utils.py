import torch

from src.training.train_act import ResumableBatchSampler
from src.utils import atomic_torch_save, get_rng_state, set_rng_state


def test_resumable_sampler_matches_uninterrupted_order():
    full = list(ResumableBatchSampler(50, 8, total=20, seed=3))
    resumed = list(ResumableBatchSampler(50, 8, total=20, seed=3, start=13))
    assert len(full) == 20 and len(resumed) == 7
    assert resumed == full[13:]


def test_resumable_sampler_spans_epochs_without_repeats_within_epoch():
    batches = list(ResumableBatchSampler(16, 4, total=8, seed=0))
    first_epoch = sorted(i for b in batches[:4] for i in b)
    assert first_epoch == list(range(16))


def test_rng_state_roundtrip():
    state = get_rng_state()
    a = torch.rand(3)
    set_rng_state(state)
    assert torch.equal(a, torch.rand(3))


def test_atomic_save_leaves_no_temp_file(tmp_path):
    path = tmp_path / "ckpt" / "latest.pt"
    atomic_torch_save({"x": torch.ones(2)}, path)
    assert torch.load(path)["x"].sum() == 2
    assert [p.name for p in path.parent.iterdir()] == ["latest.pt"]
