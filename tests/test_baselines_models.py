"""Matched baselines: same network, mixer swapped. Must be RC-equivariant and parameter-matched."""

import pytest
import torch

from hifold.model.hifold import HiFold, HiFoldConfig

MATCHED = {"attention": dict(mixer="attention", attn_dim=224), "conv": dict(mixer="conv", conv_width=73, conv_layers=6)}


def _n(cfg):
    return sum(p.numel() for p in HiFold(cfg).parameters())


@pytest.mark.parametrize("name", list(MATCHED))
def test_parameter_matched_within_5_percent(name):
    ours = _n(HiFoldConfig())
    base = _n(HiFoldConfig(**MATCHED[name]))
    assert abs(base / ours - 1) < 0.05, (name, ours, base)


@pytest.mark.parametrize("mixer,extra", [("attention", dict(attn_heads=4)), ("conv", dict(conv_layers=3))])
@pytest.mark.parametrize("train", [False, True])
def test_matched_baselines_equivariant(mixer, extra, train):
    cfg = HiFoldConfig(seq_len=2**15, tower_channels=[16, 16, 24, 24, 32, 32], d_model=32, n_blocks=3, d_state=4,
                       d_pair=8, cell_dim=8, mixer=mixer, **extra)
    torch.manual_seed(0)
    m = HiFold(cfg).train(train)
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.add_(0.1 * torch.randn_like(p))
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (1, cfg.seq_len)), 4).float()
    cells = torch.tensor([[0, 2]])
    y = m(x, cells)
    y_rc = m(x.flip(1).flip(2), cells)
    torch.testing.assert_close(y, y_rc.flip(-1).flip(-2), atol=1e-4, rtol=1e-5)
