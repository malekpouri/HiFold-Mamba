"""Mandatory RC-equivariance test (architecture.md; protocol G6). Training must not start unless it passes.

    y    = model(x, cell)
    y_rc = model(reverse_complement(x), cell)
    assert allclose(y, rot180(y_rc))

Cases: random sequence, real hg38 sequence, cell embedding (several cells), eval mode and
deterministic train mode (dropout 0), with and without gradient checkpointing; fp32 exact
(atol 1e-4, rtol 1e-5) and bf16 loose (rtol 1e-2).
"""

import os

import pytest
import torch

from hifold.model.hifold import HiFold, HiFoldConfig
from hifold.model.rc import RCPS, rc, rc_conv, rc_linear, RCLayerNorm
from hifold.model.ssm import BiMamba, selective_scan, selective_scan_ref

HG38 = os.environ.get("HG38_FA", "/mnt/storage/datasets/Human genomes Dataset/new_idea/datasets/reference_hg38/hg38.fa")


def small_cfg(**kw):
    # 2^15 bp -> 6 pools -> 512 bins: the full head/mixer geometry at a CPU-friendly length
    base = dict(seq_len=2**15, tower_channels=[16, 16, 24, 24, 32, 32], d_model=32, n_blocks=3,
                d_state=4, d_pair=8, cell_dim=8, n_cells=5)
    base.update(kw)
    return HiFoldConfig(**base)


def onehot(idx):
    return torch.nn.functional.one_hot(idx, 4).float()


def reverse_complement(x):
    return x.flip(1).flip(2)      # ACGT order: channel flip = complement


def rot180(y):
    return y.flip(-1).flip(-2)


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    m = HiFold(cfg)
    # perturb FiLM and the distance bias away from their zero init, so the test exercises them
    with torch.no_grad():
        for p in m.parameters():
            if p.abs().sum() == 0:
                p.add_(0.1 * torch.randn_like(p))
    return m


def _check(model, x, cells, atol, rtol, dtype=torch.float32):
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        y = model(x, cells).float()
        y_rc = model(reverse_complement(x), cells).float()
    assert y.shape[-1] == model.cfg.out_bins
    torch.testing.assert_close(y, rot180(y_rc), atol=atol, rtol=rtol)
    return y


# --- component-level -----------------------------------------------------------------------

def test_selective_scan_chunked_equals_reference():
    torch.manual_seed(0)
    b, L, E, N = 2, 64, 6, 4
    x = torch.randn(b, L, E, dtype=torch.float64)
    dt = torch.nn.functional.softplus(torch.randn(b, L, E, dtype=torch.float64))
    A = -torch.rand(E, N, dtype=torch.float64) * 8
    B, C = torch.randn(b, L, N, dtype=torch.float64), torch.randn(b, L, N, dtype=torch.float64)
    D = torch.randn(E, dtype=torch.float64)
    ref = selective_scan_ref(x, dt, A, B, C, D)
    for chunk in (8, 16, 32):
        torch.testing.assert_close(selective_scan(x, dt, A, B, C, D, chunk=chunk), ref)


@pytest.mark.parametrize("layer", [lambda: rc_conv(4, 8, 5), lambda: rc_linear(6, 8), lambda: RCLayerNorm(8),
                                   lambda: RCPS(BiMamba(8, d_state=4, chunk=16))])
def test_components_equivariant(layer):
    torch.manual_seed(1)
    f = layer()
    c_in = {RCLayerNorm: 8}.get(type(f), None)
    if isinstance(f, RCPS) and isinstance(f.inner, BiMamba):
        c_in = 8
    elif isinstance(f, RCPS):
        c_in = f.inner.conv.in_channels if hasattr(f.inner, "conv") else f.inner.in_features
    with torch.no_grad():
        for p in f.parameters():
            p.add_(0.1 * torch.randn_like(p))
    x = torch.randn(2, 32, c_in)
    torch.testing.assert_close(f(rc(x)), rc(f(x)), atol=1e-5, rtol=1e-5)


# --- full network ---------------------------------------------------------------------------

@pytest.mark.parametrize("train,ckpt", [(False, False), (True, False), (True, True)])
def test_full_model_fp32_random_sequence(train, ckpt):
    cfg = small_cfg(checkpoint=ckpt, dropout=0.0)
    m = _model(cfg).train(train)
    torch.manual_seed(2)
    x = onehot(torch.randint(0, 4, (2, cfg.seq_len)))
    cells = torch.tensor([[0, 3], [4, 1]])
    _check(m, x, cells, atol=1e-4, rtol=1e-5)


def test_full_model_fp32_cells_differ():
    cfg = small_cfg()
    m = _model(cfg).eval()
    x = onehot(torch.randint(0, 4, (1, cfg.seq_len)))
    y = _check(m, x, torch.tensor([[0, 1]]), atol=1e-4, rtol=1e-5)
    assert (y[0, 0] - y[0, 1]).abs().max() > 1e-4      # FiLM actually conditions the output


def test_full_model_symmetric_output():
    cfg = small_cfg()
    m = _model(cfg).eval()
    y = m(onehot(torch.randint(0, 4, (1, cfg.seq_len))), torch.tensor([[2]]))
    torch.testing.assert_close(y, y.transpose(-1, -2))


@pytest.mark.skipif(not os.path.exists(HG38), reason="hg38 FASTA not available")
def test_full_model_fp32_real_sequence():
    from hifold.data.fasta import Fasta
    cfg = small_cfg()
    fa = Fasta(HG38, keep={"chr5"})
    s = fa.fetch("chr5", 61_097_984, 61_097_984 + cfg.seq_len)
    idx = torch.tensor([{"A": 0, "C": 1, "G": 2, "T": 3}.get(ch, 0) for ch in s])
    x = onehot(idx)[None]
    _check(_model(cfg).eval(), x, torch.tensor([[1]]), atol=1e-4, rtol=1e-5)


def test_full_model_bf16_loose():
    cfg = small_cfg()
    m = _model(cfg).eval()
    x = onehot(torch.randint(0, 4, (1, cfg.seq_len)))
    _check(m, x, torch.tensor([[0]]), atol=5e-2, rtol=1e-2, dtype=torch.bfloat16)


@pytest.mark.slow
def test_full_length_geometry_fp32():
    """Production geometry: 1,048,576 bp -> 512 bins -> 448 x 448 (narrow channels to fit CPU)."""
    cfg = HiFoldConfig(tower_channels=[8] * 10 + [16], d_model=16, n_blocks=3, d_state=4, d_pair=8, cell_dim=8)
    m = _model(cfg).eval()
    x = onehot(torch.randint(0, 4, (1, cfg.seq_len)))
    y = _check(m, x, torch.tensor([[0]]), atol=1e-4, rtol=1e-5)
    assert y.shape == (1, 1, 448, 448)


def test_negative_control_unpaired_head_is_detected():
    """The test must have power: a head without the [Z, JZ] pairing breaks equivariance."""
    cfg = small_cfg()
    m = _model(cfg).eval()
    head = m.head

    def unpaired(z, cells=None):
        u, v = head.P(z), head.Q(z)
        mm = torch.einsum("bid,bjd->bij", u, v) * head.scale
        mm = 0.5 * (mm + mm.transpose(1, 2))
        return mm[:, head.crop:-head.crop, head.crop:-head.crop]

    m.head.forward = unpaired
    x = onehot(torch.randint(0, 4, (1, cfg.seq_len)))
    with pytest.raises(AssertionError):
        _check(m, x, torch.tensor([[0]]), atol=1e-4, rtol=1e-5)
