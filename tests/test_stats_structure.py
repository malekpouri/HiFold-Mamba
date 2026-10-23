import numpy as np
import pytest
import torch

from hifold import structure as S
from hifold.eval import stats


# --- block bootstrap ---------------------------------------------------------------------

def test_block_ids_by_midpoint():
    wins = [("chr1", 0, 1_048_576), ("chr1", 32_768, 1_081_344), ("chr1", 2_000_000, 3_048_576), ("chr2", 0, 1_048_576)]
    b = stats.block_ids(wins)
    assert b[0] == b[1] and len(set(b)) == 3


def test_per_block_ignores_nan():
    s = np.array([1.0, np.nan, 3.0, 5.0])
    out = stats.per_block(s, np.array([0, 0, 1, 1]))
    assert np.allclose(out, [1.0, 4.0])


def test_bootstrap_detects_true_gain_and_uses_per_block_max():
    rng = np.random.default_rng(0)
    n_win = 400
    blocks = np.repeat(np.arange(100), 4)
    base1 = rng.normal(0.5, 0.05, (3, n_win))
    base2 = base1.copy()
    base2[:, :200] += 0.03          # base2 is better in the first half of the blocks
    ours = np.maximum(base1, base2) + 0.02 + rng.normal(0, 0.01, (3, n_win))
    res = stats.block_bootstrap(ours, np.stack([base1, base2]), blocks, n_boot=2000)
    assert res.mean_delta == pytest.approx(0.02, abs=0.005)
    assert res.ci_low > 0 and res.p_one_sided < 0.025 and res.success


def test_bootstrap_null_is_not_significant():
    rng = np.random.default_rng(1)
    blocks = np.repeat(np.arange(80), 3)
    base = rng.normal(0.5, 0.05, (3, 240))
    ours = base + rng.normal(0, 0.02, base.shape)
    res = stats.block_bootstrap(ours, base[None], blocks, n_boot=2000)
    assert not res.success and res.ci_low < 0 < res.ci_high


def test_seed_average_before_blocks():
    blocks = np.array([0, 0])
    ours = np.array([[1.0, 1.0], [0.0, np.nan]])     # seed means per window: 0.5, 1.0
    base = np.zeros((1, 2, 2))
    res = stats.block_bootstrap(ours, base, blocks, margin=0, n_boot=100)
    assert res.mean_delta == pytest.approx(0.75)


def test_holm():
    assert stats.holm([0.001, 0.02, 0.03], alpha=0.025).tolist() == [True, False, False]
    assert stats.holm([0.005, 0.011, 0.02], alpha=0.025).tolist() == [True, True, True]


# --- insulation --------------------------------------------------------------------------

def _brute_insulation(c, w):
    n = c.shape[0]
    g = np.exp(2 * np.tanh(c / 2))
    out = np.full(n, np.nan)
    for i in range(w, n - w):
        vals = [g[a, b] for a in range(i - w, i) for b in range(i + 1, i + w + 1)
                if b - a >= 2 and not np.isnan(c[a, b])]
        out[i] = np.mean(vals) if vals else np.nan
    return out


def test_insulation_matches_brute_force_with_nans():
    rng = np.random.default_rng(2)
    n, w = 60, 7
    c = rng.normal(0, 0.7, (n, n))
    c = (c + c.T) / 2
    c[np.abs(np.subtract.outer(np.arange(n), np.arange(n))) < 2] = np.nan
    c[10, :] = c[:, 10] = np.nan                     # a masked bin
    assert np.allclose(S.insulation_np(c, w), _brute_insulation(c, w), equal_nan=True)


def test_insulation_dips_at_domain_boundary():
    n, w = 100, 10
    c = np.full((n, n), -1.0)
    c[:50, :50] = 1.0
    c[50:, 50:] = 1.0                                # two domains, boundary at 50
    ins = S.insulation_np(c, w)
    assert np.nanargmin(ins) in (49, 50)
    bs = S.boundary_strength(ins)
    assert np.nanargmax(bs) in (49, 50)


def test_insulation_is_differentiable_and_rot180_equivariant():
    rng = np.random.default_rng(3)
    c = torch.tensor(rng.normal(size=(2, 40, 40)), dtype=torch.float64)
    c = (c + c.transpose(-1, -2)) / 2
    c.requires_grad_(True)
    ins = S.insulation(c, w=5)
    ins[~torch.isnan(ins)].sum().backward()
    assert torch.isfinite(c.grad).all() and c.grad.abs().sum() > 0
    ins_rot = S.insulation(torch.flip(c.detach(), (-1, -2)), w=5)
    assert torch.allclose(torch.flip(ins.detach(), (-1,)), ins_rot, equal_nan=True)


def test_insulation_gradient_finite_with_nan_pixels():
    c = torch.randn(30, 30, dtype=torch.float64)
    c = (c + c.T) / 2
    c[3, :] = float("nan")
    c.requires_grad_(True)
    ins = S.insulation(c, w=4)
    ins[~torch.isnan(ins)].sum().backward()
    assert torch.isfinite(c.grad).all()


def test_g_transform_bounds():
    x = torch.linspace(-50, 50, 101)
    y = S.g_transform(x)
    assert (y >= np.exp(-2) - 1e-9).all() and (y <= np.exp(2) + 1e-9).all()
    assert S.g_transform(torch.tensor(0.0)).item() == pytest.approx(1.0)


def test_auroc_and_ap():
    s = np.array([0.9, 0.8, 0.3, 0.1, np.nan])
    y = np.array([1, 0, 1, 0, 1])
    assert S.auroc(s, y) == pytest.approx(0.75)
    assert S.average_precision(s, y) == pytest.approx((1 + 2 / 3) / 2)


def test_boundary_strength_torch_matches_numpy_and_loss_has_finite_grad():
    import torch
    from hifold.train import boundary_loss
    rng = np.random.default_rng(1)
    n, w = 120, 10
    c = rng.normal(0, 0.5, (n, n))
    c = (c + c.T) / 2
    c[:60, :60] += 1.0
    c[60:, 60:] += 1.0
    ins = S.insulation_np(c, w)
    ref = S.boundary_strength(ins, 5)
    got = S.boundary_strength_t(torch.as_tensor(ins), 5).numpy()
    assert np.allclose(got, ref, equal_nan=True)
    assert np.nanargmax(got) == 60                       # strongest boundary at the domain border
    t = torch.as_tensor(c, dtype=torch.float32)
    p = (t + 0.3 * torch.randn_like(t)).requires_grad_()
    lb = boundary_loss(p[None], t[None], w)
    lb.backward()
    assert lb.item() > 0 and torch.isfinite(p.grad).all()
    assert boundary_loss(t[None], t[None], w).item() == 0
